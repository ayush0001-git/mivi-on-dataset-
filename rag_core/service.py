"""Serving concerns for the MIVI API — sessions, rate limiting, answer cache, metrics.

Everything in this module is IN-PROCESS state, and that is a deliberate,
documented limit rather than an oversight:

    one process = one unit of correctness.

Behind a load balancer with N instances you get N independent copies, which
means (a) conversation memory only works if the balancer is session-sticky,
and (b) the effective rate limit is N x the configured one. The swap-in for a
multi-instance deployment is Redis — every structure below is a narrow class
with a three-method surface (`snapshot`/`record_turn`, `allow`, `get`/`put`),
so the backing store can be replaced without a single route handler changing.
Nothing here holds a lock across an I/O call, so that swap stays mechanical.

Why not just use the pipeline's own module-level cache: it keys on question
text alone. See `answer_cache_key` — that is a correctness bug at scale, not a
tuning detail.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Callable

# TTLs and rate windows use time.monotonic(), never time.time(). A wall-clock
# step (NTP correction, VM resume, DST on a badly configured box) would either
# expire every session at once or freeze a rate-limit bucket forever. Only
# `uptime`/`created_at`, which are for humans, use wall clock.
_now = time.monotonic


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:  # a typo'd env var must not silently become 0 rps
        raise ValueError(f"{name} must be an integer, got {os.getenv(name)!r}") from None


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {os.getenv(name)!r}") from None


# ---------------------------------------------------------------------------
# Untrusted input hygiene
# ---------------------------------------------------------------------------
# Session ids come from the browser and are used as dict keys and log fields.
# Anything outside this alphabet is rejected and replaced with a fresh id
# rather than trusted: an unbounded client-chosen key is a memory-growth and
# log-injection surface, and the cost of being wrong is one lost conversation.
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
_PROFILE_KEY_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")

# The profile is interpolated into the LLM prompt, so it is prompt-injection
# surface as much as it is data. Values are length-capped here; the pipeline's
# own injection guards are the second layer.
PROFILE_MAX_KEYS = 24
PROFILE_MAX_VALUE_CHARS = 200
PROFILE_MAX_LIST_ITEMS = 12


def new_session_id() -> str:
    return uuid.uuid4().hex


def valid_session_id(sid: str | None) -> bool:
    return bool(sid and _SESSION_ID_RE.match(sid))


def sanitize_profile(profile: Any) -> dict[str, Any]:
    """Coerce a client-supplied profile into something safe to put in a prompt.

    Deliberately not a whitelist of field names: the pipeline owns the profile
    schema and adds fields over time, and a whitelist here would silently drop
    them. What is enforced is *shape* — key syntax, key count, value types,
    value length — which is what makes it safe regardless of schema drift.
    """
    if not isinstance(profile, dict):
        return {}
    out: dict[str, Any] = {}
    for key, value in profile.items():
        if len(out) >= PROFILE_MAX_KEYS:
            break
        if not isinstance(key, str) or not _PROFILE_KEY_RE.match(key):
            continue
        if value is None or isinstance(value, bool) or isinstance(value, (int, float)):
            out[key] = value
        elif isinstance(value, str):
            out[key] = value.strip()[:PROFILE_MAX_VALUE_CHARS]
        elif isinstance(value, (list, tuple)):
            out[key] = [
                str(v).strip()[:PROFILE_MAX_VALUE_CHARS]
                for v in list(value)[:PROFILE_MAX_LIST_ITEMS]
                if isinstance(v, (str, int, float))
            ]
        # anything else (nested dicts, bytes) is dropped, not stringified
    return out


def merge_profile(base: dict[str, Any] | None, incoming: dict[str, Any] | None) -> dict[str, Any]:
    """Later turns win, but only for fields they actually assert.

    A client that re-sends `{"budget": null}` must not erase a budget the
    student stated three turns ago — nulls are "unknown", not "cleared".
    """
    merged = dict(base or {})
    for key, value in sanitize_profile(incoming).items():
        if value is None or (isinstance(value, str) and not value):
            continue
        merged[key] = value
    return merged


def client_key(session_id: str | None, forwarded_for: str | None,
               peer_ip: str | None, trust_proxy: bool) -> str:
    """The identity a rate-limit bucket hangs off.

    X-Forwarded-For is client-controlled unless something in front of us
    rewrites it, so it is only honoured when the operator declares that a
    proxy is in front (MIVI_TRUST_PROXY). Take the FIRST hop — the balancer
    appends its own peer, so the leftmost entry is the original client.
    A directly-exposed server that trusts XFF has no rate limiting at all.
    """
    if trust_proxy and forwarded_for:
        first = forwarded_for.split(",")[0].strip()
        if first:
            return f"ip:{first[:64]}"
    if peer_ip:
        return f"ip:{peer_ip[:64]}"
    return f"sess:{session_id or 'anon'}"


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------
@dataclass
class SessionSnapshot:
    """An immutable copy handed to a request handler.

    Handlers get copies, never the live objects: two requests for one session
    (a double-tapped send button) would otherwise mutate the same list while
    another thread iterates it. Copies are cheap — history is capped at a
    couple of dozen short strings.
    """
    session_id: str
    history: list[dict[str, str]]
    profile: dict[str, Any]
    turns: int
    is_new: bool


@dataclass
class _SessionRecord:
    profile: dict[str, Any] = field(default_factory=dict)
    history: deque = field(default_factory=deque)
    turns: int = 0
    created_at: float = field(default_factory=_now)
    created_wall: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=_now)


class SessionStore:
    """Bounded, TTL'd, thread-safe conversation memory.

    Two independent bounds, because they fail differently:
      * `ttl_s`     — a student who closed the tab must not hold RAM forever.
      * `max_sessions` — a burst of real traffic (or a bot minting session ids)
        must degrade by dropping the coldest conversation, not by OOM-killing
        the process and taking every live conversation with it.

    Eviction is amortised O(1), not a sweep: the OrderedDict is kept in
    least-recently-used order, so everything expired clusters at the front and
    we stop at the first live entry. A per-request full scan of 20k sessions
    would be the same "scan everything" mistake the retrieval layer avoids.
    """

    def __init__(self, ttl_s: float, max_sessions: int, max_messages: int) -> None:
        if max_messages % 2:  # history is stored as (user, assistant) pairs
            max_messages += 1
        self._ttl = ttl_s
        self._max = max_sessions
        self._max_messages = max_messages
        self._lock = threading.Lock()
        self._sessions: OrderedDict[str, _SessionRecord] = OrderedDict()
        self._evicted = 0
        self._expired = 0

    def _reap_locked(self, budget: int = 64) -> None:
        cutoff = _now() - self._ttl
        while self._sessions and budget > 0:
            sid, rec = next(iter(self._sessions.items()))
            if rec.last_seen > cutoff:
                break  # LRU order => everything behind this one is younger
            self._sessions.popitem(last=False)
            self._expired += 1
            budget -= 1
        while len(self._sessions) > self._max:
            self._sessions.popitem(last=False)
            self._evicted += 1

    def snapshot(self, session_id: str | None) -> SessionSnapshot:
        """Fetch (or open) a session and return a private copy of its state."""
        sid = session_id if valid_session_id(session_id) else new_session_id()
        with self._lock:
            rec = self._sessions.get(sid)
            is_new = rec is None
            if rec is None:
                rec = _SessionRecord()
                self._sessions[sid] = rec
            rec.last_seen = _now()
            self._sessions.move_to_end(sid)
            # Reap AFTER inserting, never before: reaping first leaves room for
            # the new session and then puts the store one over the cap.
            # move_to_end already protects this session from its own eviction.
            self._reap_locked()
            return SessionSnapshot(
                session_id=sid,
                history=[dict(m) for m in rec.history],
                profile=dict(rec.profile),
                turns=rec.turns,
                is_new=is_new,
            )

    def record_turn(self, session_id: str, user_message: str, assistant_message: str,
                    profile: dict[str, Any] | None) -> None:
        with self._lock:
            rec = self._sessions.get(session_id)
            if rec is None:  # evicted mid-request; re-open rather than lose the turn
                rec = _SessionRecord()
                self._sessions[session_id] = rec
            rec.history.append({"role": "user", "content": user_message})
            rec.history.append({"role": "assistant", "content": assistant_message})
            while len(rec.history) > self._max_messages:
                rec.history.popleft()
            if profile is not None:
                rec.profile = sanitize_profile(profile)
            rec.turns += 1
            rec.last_seen = _now()
            self._sessions.move_to_end(session_id)
            self._reap_locked()

    def drop(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"live": len(self._sessions), "capacity": self._max,
                    "expired_total": self._expired, "evicted_total": self._evicted}


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
class RateLimiter:
    """Token bucket per client key.

    Why a bucket and not a fixed window: real students are bursty and abuse is
    sustained. A fixed window of 30/min either rejects the three follow-ups
    someone fires off in five seconds, or is loosened until it permits a
    genuine 30 rps flood — and it has the classic boundary flaw where 30
    requests at 11:59:59 plus 30 at 12:00:00 sail through as 60 in one second.
    A bucket separates the two dials: `burst` is how much slack a human gets,
    `rate_per_min` is the sustained cost an abuser has to pay. It also gives an
    honest Retry-After — the exact time until one token exists — instead of
    "wait until the window rolls".

    Refill is computed lazily from the last touch, so there is no sweeper
    thread and an idle bucket costs nothing.
    """

    def __init__(self, rate_per_min: float, burst: int, max_keys: int = 50_000) -> None:
        if rate_per_min <= 0 or burst <= 0:
            raise ValueError("rate_per_min and burst must both be positive")
        self._rate = rate_per_min / 60.0
        self._burst = float(burst)
        self._max_keys = max_keys
        self._lock = threading.Lock()
        # LRU-capped: an attacker rotating source IPs would otherwise grow this
        # map without bound. Evicting the least-recently-used bucket can only
        # ever *forgive* a client that has already gone quiet, never punish one.
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()
        self.rejected = 0

    def allow(self, key: str) -> tuple[bool, float]:
        """-> (allowed, retry_after_seconds). retry_after is 0 when allowed."""
        now = _now()
        with self._lock:
            tokens, last = self._buckets.get(key, (self._burst, now))
            tokens = min(self._burst, tokens + (now - last) * self._rate)
            if tokens >= 1.0:
                self._buckets[key] = (tokens - 1.0, now)
                self._buckets.move_to_end(key)
                while len(self._buckets) > self._max_keys:
                    self._buckets.popitem(last=False)
                return True, 0.0
            self._buckets[key] = (tokens, now)
            self._buckets.move_to_end(key)
            self.rejected += 1
            return False, (1.0 - tokens) / self._rate

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {"tracked_keys": len(self._buckets), "rejected_total": self.rejected,
                    "rate_per_min": round(self._rate * 60, 2), "burst": int(self._burst)}


# ---------------------------------------------------------------------------
# Answer cache
# ---------------------------------------------------------------------------
# Fields that can change WHICH colleges are retrieved or what the numbers in
# the answer are. The cache key must contain all of them.
#
# THE BUG THIS EXISTS TO PREVENT: keying an answer cache on question text alone
# is not a staleness trade-off, it is a wrong-answer generator. Two students
# both type "best BTech colleges near me" — one with budget 80,000 in Dehradun,
# one with budget 8,00,000 in Pune. Text-only keying serves the second student
# the first student's colleges and the first student's fee figures, presented
# as fact and with real citations. It looks perfect and it is entirely wrong.
# The pipeline's own module-level cache has exactly this shape, which is why
# the server disables it (MME_CACHE=0) and owns caching here instead.
CACHE_KEY_PROFILE_FIELDS = frozenset({
    "budget", "max_tuition_inr", "min_tuition_inr", "marks_pct", "score_pct",
    "field_interest", "course_terms", "program_level", "location", "city",
    "state", "college_type", "hostel_required", "entrance_exam", "include_abroad",
})

# Only the grounded answer contract is cached. A cached entry must never carry
# `profile` or any other per-student key — that is how one student's stated
# marks end up in another student's response.
ANSWER_CONTRACT_KEYS = ("answer", "citations", "answered", "reason_if_unanswered")


def normalise_question(question: str) -> str:
    """Lowercase + collapse whitespace, and nothing else.

    Do NOT extend this to strip non-Latin characters: that folds every Hindi
    or Tamil question onto the same key (a bug this repo has already been bitten
    by once). Cheap normalisation only.
    """
    return re.sub(r"\s+", " ", question.strip().lower())


def _canonical(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, (list, tuple)):
        return sorted(_canonical(v) for v in value)
    return value


def answer_cache_key(question: str, profile: dict[str, Any] | None) -> str:
    """sha256 over the normalised question + the filter-relevant profile.

    Non-filter fields (the student's name, their school) are excluded on
    purpose: they do not change which colleges match, so including them would
    shatter the hit rate for no correctness gain. The flip side is that a
    personalised greeting must not be cached — hence callers only consult the
    cache on the first turn of a session, before any personalisation exists.
    """
    relevant = {k: _canonical(v) for k, v in (profile or {}).items()
                if k in CACHE_KEY_PROFILE_FIELDS and v not in (None, "", [])}
    payload = json.dumps([normalise_question(question), relevant],
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def project_answer(result: dict[str, Any]) -> dict[str, Any]:
    """Strip a pipeline result down to the shareable answer contract."""
    return {k: result[k] for k in ANSWER_CONTRACT_KEYS if k in result}


class AnswerCache:
    """LRU + TTL cache of grounded answers.

    Copies go in and copies come out. Handlers decorate the result they get
    (session_id, resolved colleges), and handing back the stored dict would let
    request N's decoration leak into request N+1's response — a mutation bug
    that only ever shows up under concurrency.
    """

    def __init__(self, ttl_s: float, max_entries: int) -> None:
        self._ttl = ttl_s
        self._max = max_entries
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            stored_at, value = entry
            if _now() - stored_at > self._ttl:
                del self._entries[key]
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return json.loads(json.dumps(value))

    def put(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._entries[key] = (_now(), json.loads(json.dumps(value)))
            self._entries.move_to_end(key)
            while len(self._entries) > self._max:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self.hits + self.misses
            return {"entries": len(self._entries), "capacity": self._max,
                    "hits": self.hits, "misses": self.misses,
                    "hit_rate": round(self.hits / total, 3) if total else 0.0}


class TTLSnapshot:
    """Single-slot memoiser for an expensive-ish, shared, read-only payload.

    Used for /api/health: a load balancer hits it once per instance per second
    and it must not turn into a per-probe SQL fanout. The producer runs under
    the lock, so a burst of probes arriving on a cold slot produces one refresh,
    not one per probe (thundering-herd control matters more here than the
    microseconds of lock hold).
    """

    def __init__(self, ttl_s: float) -> None:
        self._ttl = ttl_s
        self._lock = threading.Lock()
        self._value: Any = None
        self._at: float = -1e9

    def get(self, produce: Callable[[], Any]) -> Any:
        with self._lock:
            if self._value is None or _now() - self._at > self._ttl:
                self._value = produce()
                self._at = _now()
            return self._value

    def invalidate(self) -> None:
        with self._lock:
            self._at = -1e9


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
class Metrics:
    """Per-request counters + a bounded latency window.

    In memory only. The pipeline already appends a rich per-answer record to
    metrics/metrics.jsonl; duplicating that from the request path would add a
    disk write (and lock contention) to every health probe. This is the cheap
    operational view — what a dashboard scrapes — not the analytics record.

    Percentiles come from a fixed-size ring of recent samples, so memory is
    constant no matter how long the process lives, and the numbers describe
    *recent* behaviour rather than being dragged around by a cold-start outlier
    from six hours ago.
    """

    def __init__(self, window: int = 2048) -> None:
        self._lock = threading.Lock()
        self._latencies: dict[str, deque] = {}
        self._window = window
        self._counts: dict[tuple[str, int], int] = {}
        self.cache_hits = 0
        self.inflight = 0
        self.peak_inflight = 0
        self.queue_timeouts = 0
        self.started_wall = time.time()

    def record(self, route: str, status: int, latency_ms: float, cache_hit: bool) -> None:
        with self._lock:
            self._counts[(route, status)] = self._counts.get((route, status), 0) + 1
            samples = self._latencies.get(route)
            if samples is None:
                samples = self._latencies[route] = deque(maxlen=self._window)
            samples.append(latency_ms)
            if cache_hit:
                self.cache_hits += 1

    def enter(self) -> None:
        with self._lock:
            self.inflight += 1
            self.peak_inflight = max(self.peak_inflight, self.inflight)

    def leave(self) -> None:
        with self._lock:
            self.inflight -= 1

    def queue_timeout(self) -> None:
        with self._lock:
            self.queue_timeouts += 1

    @staticmethod
    def _pct(sorted_samples: list[float], q: float) -> float:
        if not sorted_samples:
            return 0.0
        idx = min(len(sorted_samples) - 1, int(q * len(sorted_samples)))
        return round(sorted_samples[idx], 1)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counts = dict(self._counts)
            latencies = {r: sorted(s) for r, s in self._latencies.items()}
            inflight, peak = self.inflight, self.peak_inflight
            cache_hits, timeouts = self.cache_hits, self.queue_timeouts
        routes: dict[str, Any] = {}
        for (route, status), n in counts.items():
            entry = routes.setdefault(route, {"requests": 0, "by_status": {}})
            entry["requests"] += n
            entry["by_status"][str(status)] = entry["by_status"].get(str(status), 0) + n
        for route, samples in latencies.items():
            entry = routes.setdefault(route, {"requests": 0, "by_status": {}})
            entry["latency_ms"] = {
                "p50": self._pct(samples, 0.50),
                "p95": self._pct(samples, 0.95),
                "max": round(samples[-1], 1) if samples else 0.0,
                "samples": len(samples),
            }
        return {
            "uptime_s": round(time.time() - self.started_wall, 1),
            "inflight": inflight,
            "peak_inflight": peak,
            "cache_hits": cache_hits,
            "queue_timeouts": timeouts,
            "routes": routes,
        }


# ---------------------------------------------------------------------------
# Configured singletons — one set per process, built from the environment.
# ---------------------------------------------------------------------------
SESSION_TTL_S = _env_float("MIVI_SESSION_TTL_S", 1800)       # 30 min of silence
MAX_SESSIONS = _env_int("MIVI_MAX_SESSIONS", 20_000)
MAX_HISTORY_MESSAGES = _env_int("MIVI_HISTORY_MESSAGES", 16)  # 8 turns; the
# pipeline prompt only reads the last few, but keeping a couple of spare turns
# means a client that reconnects mid-conversation still gets coherent context.

# Two limiters, checked together, because they defend different things.
# Per-IP is the box's armour, and it has to stay loose: an entire school lab or
# a CGNAT'd mobile carrier shares one address. Per-session is fairness between
# students, and can be tight because one human cannot legitimately outpace it.
# A client can rotate session ids to dodge the tight bucket, but it still has
# to get past the IP bucket to do damage.
RATE_IP_PER_MIN = _env_float("MIVI_RATE_IP_PER_MIN", 120)
RATE_IP_BURST = _env_int("MIVI_RATE_IP_BURST", 40)
RATE_SESSION_PER_MIN = _env_float("MIVI_RATE_SESSION_PER_MIN", 20)
RATE_SESSION_BURST = _env_int("MIVI_RATE_SESSION_BURST", 6)

CACHE_TTL_S = _env_float("MIVI_CACHE_TTL_S", 900)            # 15 min
CACHE_MAX_ENTRIES = _env_int("MIVI_CACHE_MAX", 2000)
HEALTH_TTL_S = _env_float("MIVI_HEALTH_TTL_S", 5)

sessions = SessionStore(SESSION_TTL_S, MAX_SESSIONS, MAX_HISTORY_MESSAGES)
ip_limiter = RateLimiter(RATE_IP_PER_MIN, RATE_IP_BURST)
session_limiter = RateLimiter(RATE_SESSION_PER_MIN, RATE_SESSION_BURST)
answer_cache = AnswerCache(CACHE_TTL_S, CACHE_MAX_ENTRIES)
health_snapshot = TTLSnapshot(HEALTH_TTL_S)
metrics = Metrics()


def config_snapshot() -> dict[str, Any]:
    """The knobs that are actually in force — the first thing you want when an
    instance behaves differently from the one next to it."""
    return {
        "session_ttl_s": SESSION_TTL_S,
        "max_sessions": MAX_SESSIONS,
        "history_messages": MAX_HISTORY_MESSAGES,
        "rate_ip": ip_limiter.stats(),
        "rate_session": session_limiter.stats(),
        "cache_ttl_s": CACHE_TTL_S,
        "cache_max_entries": CACHE_MAX_ENTRIES,
    }
