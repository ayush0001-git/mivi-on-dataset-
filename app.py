"""MIVI HTTP API — the counsellor ("mimi") as a service embeddable in makemyeducation.com.

    uvicorn app:app --host 0.0.0.0 --port 8000

Concurrency model, which is the whole design:

    event loop (async)  ->  bounded threadpool  ->  sync RAG pipeline

The pipeline is synchronous and spends its time in two places that both block a
thread: sentence-transformers encoding (CPU, releases the GIL inside torch) and
LLM HTTP calls (network wait). Calling it directly from an async handler would
park the *event loop* for seconds, at which point every other connected student
— including the load balancer's health probe — stops being served. So every
call into the pipeline goes through `run_in_threadpool`, and the number of
those threads is capped by a semaphore.

The semaphore is the important half. An uncapped threadpool under a traffic
spike does not fail gracefully: 300 threads all encoding at once thrash cache
and context-switch, every single answer gets slower, and the provider starts
429ing us anyway. Capped, the (N+1)th request *queues* — and if it cannot get a
slot within MIVI_QUEUE_TIMEOUT_S it is told to retry instead of being left to
rot in a queue longer than the client's own timeout. Fast answers stay fast;
overload is visible as a clean 503 rather than as universal slowness.

Serving state (sessions, rate limits, answer cache, metrics) lives in
rag_core/service.py and is in-process — see that module for the single-instance
caveat and the Redis swap-in.
"""
from __future__ import annotations

import os

# Must precede any rag_core import: rag_core.config reads this at import time.
# The pipeline's own module-level cache keys on question text ALONE, which
# serves one student's budget-filtered colleges to another (see
# service.answer_cache_key). The server keys on question + filter-relevant
# profile instead, so the unsafe one stays off unless an operator insists.
os.environ.setdefault("MME_CACHE", "0")

import asyncio
import json
import logging
import math
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from rag_core import service
from rag_core.store import db

log = logging.getLogger("mivi.api")
logging.basicConfig(
    level=os.getenv("MIVI_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# A college id IS the public URL slug (the Mongo ObjectId the courses export
# joins on), so a citation converts to a real, clickable makemyeducation.com
# page with string concatenation and no extra lookup. That property is the
# entire website-integration seam.
PUBLIC_COLLEGE_URL = os.getenv("MIVI_PUBLIC_COLLEGE_URL",
                               "https://makemyeducation.com/college/").rstrip("/") + "/"

# Never "*". This is going on a real site with a real audience; a wildcard on a
# production origin means any page anywhere can spend our LLM budget. Origins
# are exact-matched (no regex) so a lookalike domain cannot slip in.
DEFAULT_ORIGINS = [
    "https://makemyeducation.com",
    "https://www.makemyeducation.com",
]
CORS_ORIGINS = [o.strip() for o in os.getenv(
    "MIVI_CORS_ORIGINS", ",".join(DEFAULT_ORIGINS)).split(",") if o.strip()]
if os.getenv("MIVI_DEV") == "1":  # local UI development only
    CORS_ORIGINS += ["http://localhost:8000", "http://127.0.0.1:8000",
                     "http://localhost:3000", "http://localhost:5173"]

# Default sized for a box serving a network-bound pipeline: most of a request's
# wall time is spent waiting on the LLM, so threads > cores is correct here.
# Raise it only alongside evidence that the provider's rate limit and the box's
# memory can take it — the embedding model is loaded once and shared, but each
# in-flight encode wants its own working set.
MAX_CONCURRENCY = int(os.getenv("MIVI_MAX_CONCURRENCY", "16"))
QUEUE_TIMEOUT_S = float(os.getenv("MIVI_QUEUE_TIMEOUT_S", "20"))
MAX_MESSAGE_CHARS = int(os.getenv("MIVI_MAX_MESSAGE_CHARS", "2000"))
# Whether to believe X-Forwarded-For when identifying a client for rate
# limiting. Defaults to OFF: X-Forwarded-For is caller-supplied, so trusting it
# by default means anyone can mint a fresh rate-limit bucket per request by
# sending a different header value — the limiter is then decorative. Turn it on
# ONLY when this process sits behind a proxy you control that overwrites the
# header (nginx/ALB/Cloudflare); in that deployment the peer address is the
# proxy and every client would otherwise share one bucket.
TRUST_PROXY = os.getenv("MIVI_TRUST_PROXY", "0") == "1"
MAX_CITED_COLLEGES = 12

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")  # ObjectIds are 24 hex; stay lenient


class _State:
    """Process-wide serving state that the lifespan owns."""
    started_wall: float = time.time()
    sem: asyncio.Semaphore | None = None
    warm_ok: bool = False
    warm_error: str | None = None
    warm_started: bool = False
    warm_probe_hits: int | None = None


STATE = _State()


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
def _warm_worker() -> None:
    """Load the embedding model + retrieval index off the request path.

    Best-effort by contract: a warm failure must not stop the process from
    serving. Readiness is defined by the corpus being present, not by this —
    an un-warmed instance answers, it just pays the model load on its first
    question. Blocking startup on a multi-second model load would instead make
    every deploy look like an outage to the load balancer.

    Note what is NOT here any more: the old demo server replayed a canned
    question list through the live LLM at boot. At one instance that was a
    warm cache; at ten instances behind a balancer it is a synchronised burst
    of paid calls that trips the provider's rate limit on every deploy.
    """
    t0 = time.perf_counter()
    try:
        from rag_core.retrieve import encode_query, search, warm  # late: pulls in torch

        warm()
        # warm() is best-effort by contract — it logs a missing index and
        # returns normally — so "warm() came back" is not evidence that this
        # instance can retrieve. Probe the real path once (embed + search) and
        # let health report what was actually proven, not what was attempted.
        hits = search("engineering colleges", top_k=1,
                      query_vec=encode_query("warm up probe"))
        STATE.warm_probe_hits = len(hits)
        STATE.warm_ok = True
        log.info("warm-up complete in %.1fs (probe returned %d hit(s))",
                 time.perf_counter() - t0, len(hits))
    except Exception as exc:  # noqa: BLE001 - warming is advisory
        STATE.warm_error = f"{type(exc).__name__}: {exc}"
        log.warning("warm-up failed (serving continues, first answer will be slow): %s",
                    STATE.warm_error)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    STATE.started_wall = time.time()
    STATE.sem = asyncio.Semaphore(MAX_CONCURRENCY)

    # anyio's default thread limiter (40) sits underneath run_in_threadpool. If
    # it were smaller than our semaphore, requests would queue invisibly inside
    # anyio where no timeout or metric can see them; the semaphore has to be the
    # only place backpressure happens.
    try:
        import anyio.to_thread

        limiter = anyio.to_thread.current_default_thread_limiter()
        limiter.total_tokens = max(limiter.total_tokens, MAX_CONCURRENCY + 8)
    except Exception as exc:  # noqa: BLE001 - non-fatal tuning
        log.warning("could not raise anyio thread limiter: %s", exc)

    STATE.warm_started = True
    threading.Thread(target=_warm_worker, name="mivi-warm", daemon=True).start()
    log.info("MIVI API up: concurrency=%d origins=%s", MAX_CONCURRENCY, CORS_ORIGINS)
    yield


app = FastAPI(
    title="MIVI — MakeMyEducation AI counsellor",
    version=os.getenv("MIVI_VERSION", "2.0"),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Middleware. Registration order matters: the LAST middleware added is the
# outermost, so CORS is added last and therefore decorates error responses too.
# Without that, a browser sees "CORS error" instead of the real 429/503 and the
# site cannot tell the student anything useful.
# ---------------------------------------------------------------------------
_METRIC_PATH_RE = re.compile(r"^/api/college/[^/]+$")

# Closed label set for metrics. Anything else buckets to "/api/other" — see
# _metric_route on why an open set is a memory sink.
_KNOWN_API_ROUTES = frozenset({
    "/api/chat", "/api/answer", "/api/health", "/api/metrics", "/api/colleges",
})


def _metric_route(path: str) -> str:
    """Collapse ids out of the path — per-college metric series would be 35k
    time series for zero insight.

    Unknown /api/ paths collapse to a single "other" bucket rather than being
    used verbatim. A metric label taken from an unauthenticated URL is an
    unbounded-cardinality sink: anyone hitting /api/<random> in a loop grows
    the counters and latency lists forever, and /api/metrics then has to
    serialise all of it on the event loop. The label set has to be closed over
    routes we actually serve, not over whatever the internet sends.
    """
    if _METRIC_PATH_RE.match(path):
        return "/api/college/{id}"
    if path.startswith("/api/session/"):
        return "/api/session/{id}"
    if not path.startswith("/api/"):
        return "static"
    return path if path in _KNOWN_API_ROUTES else "/api/other"


@app.middleware("http")
async def _observe(request: Request, call_next):
    t0 = time.perf_counter()
    request.state.cache_hit = False
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        service.metrics.record(
            _metric_route(request.url.path), status,
            (time.perf_counter() - t0) * 1000.0,
            bool(getattr(request.state, "cache_hit", False)),
        )


app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-Session-Id"],
    # No cookies are used — the session id travels in the JSON body — so
    # credentials stay off. It is also the only way an explicit origin list
    # keeps its meaning.
    allow_credentials=False,
    max_age=600,
)


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    """A student must never see a stack trace, and an integrator must always
    get JSON. The trace goes to the log with an id they can quote."""
    ref = uuid.uuid4().hex[:12]
    log.exception("unhandled error ref=%s path=%s", ref, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Something went wrong on our side. Please try again.",
                 "error_ref": ref},
    )


# ---------------------------------------------------------------------------
# Corpus readiness / health
# ---------------------------------------------------------------------------
def _safe_meta(value: Any) -> str:
    """build_meta is written by the ETL for operators, and /api/health is
    reachable from the public internet. Source paths in it are a machine's
    absolute filesystem layout — useful in a log, not something to publish — so
    anything path-shaped is reduced to its last component."""
    text = str(value)
    if "/" in text or "\\" in text:
        text = re.split(r"[/\\]", text)[-1]
    return text[:120]


def _corpus_probe() -> dict[str, Any]:
    """Cheap-ish corpus facts. Always reached through the TTL snapshot AND a
    worker thread: COUNT(*) over 211k courses measured ~10 ms here, which is
    nothing once per 5 s and unacceptable on the event loop once per probe."""
    # The store is Postgres. This used to report a `db_file` name and probe
    # `DB_FILE.exists()` — SQLite's questions, answered by a compatibility shim,
    # so /api/health described a file that has not been the store since the
    # migration. Reported as a backend now, still without the DSN: this endpoint
    # is public and must not publish credentials or host layout.
    info: dict[str, Any] = {
        "store": "postgres", "present": False,
        "colleges": 0, "courses": 0, "build": {}, "error": None,
    }
    try:
        from rag_core.store.backend import get_backend

        be = get_backend()
        info["colleges"] = be.one("SELECT COUNT(*) FROM colleges") or 0
        info["courses"] = be.one("SELECT COUNT(*) FROM courses") or 0
        info["present"] = True
        # Whatever the ETL recorded — build time, source fingerprints, index
        # dimensions. Read generically so the API does not have to be edited
        # every time the builder learns a new key.
        rows = be.q("SELECT key, value FROM build_meta LIMIT 64")
        info["build"] = {r["key"]: _safe_meta(r["value"]) for r in rows}
    except Exception as exc:  # noqa: BLE001 - a half-built DB is a normal state
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def _health_payload() -> dict[str, Any]:
    corpus = _corpus_probe()
    ready = bool(corpus["present"] and corpus["colleges"] > 0 and not corpus["error"])
    if not ready:
        status = "not_ready"
    elif STATE.warm_ok:
        status = "ok"
    elif STATE.warm_error:
        # Corpus present, retrieval probe failed (missing dense index, model
        # not downloaded). Still 200: retrieval degrades to lexical-only rather
        # than failing, and pulling the instance would trade degraded answers
        # for no answers. The status word is how the operator finds out.
        status = "degraded"
    else:
        status = "warming"  # serving, but the first answers pay the model load
    return {
        "status": status,
        "ready": ready,
        "uptime_s": round(time.time() - STATE.started_wall, 1),
        "version": app.version,
        "corpus": corpus,
        "index": {"warm": STATE.warm_ok,
                  "warming": STATE.warm_started and not STATE.warm_ok and not STATE.warm_error,
                  "probe_hits": STATE.warm_probe_hits,
                  "error": STATE.warm_error},
        "concurrency": {"limit": MAX_CONCURRENCY,
                        "inflight": service.metrics.inflight,
                        "queue_timeout_s": QUEUE_TIMEOUT_S},
        "sessions": service.sessions.stats(),
        "answer_cache": service.answer_cache.stats(),
    }


async def _in_thread(fn, *args):
    """Hop for CHEAP blocking work — one indexed SQLite lookup, the TTL'd
    health probe.

    Deliberately not admission-controlled: the semaphore rations the expensive
    path (embedding + LLM), and making a 1 ms college-page query queue behind a
    chat spike would take the website's static pages down with the chatbot.

    It still has to be a thread. db.get_conn() is thread-local by design, so
    the connection must be *acquired on the thread that uses it* — passing one
    across threads would put every worker on a single shared connection.
    """
    return await run_in_threadpool(fn, *args)


async def _corpus_state() -> dict[str, Any]:
    return await _in_thread(service.health_snapshot.get, _health_payload)


async def _require_corpus() -> None:
    payload = await _corpus_state()
    if not payload["ready"]:
        # 503 + Retry-After, not a 500: "the corpus has not been built on this
        # instance yet" is a deployment state, and the caller should come back.
        raise HTTPException(
            status_code=503,
            detail="The college corpus is not loaded on this instance yet. "
                   "Please try again shortly.",
            headers={"Retry-After": "30"},
        )


# ---------------------------------------------------------------------------
# Rate limiting + admission control
# ---------------------------------------------------------------------------
def _too_many(retry_after: float) -> HTTPException:
    return HTTPException(
        status_code=429,
        detail="You are sending questions faster than mimi can think. "
               "Give it a few seconds and ask again.",
        # Honest and exact: seconds until this bucket holds one token again.
        headers={"Retry-After": str(max(1, math.ceil(retry_after)))},
    )


def _enforce_ip_limit(request: Request) -> None:
    """The outer guard, and the first thing every expensive route runs.

    Deliberately checked BEFORE any server-side state is allocated: an
    unauthenticated request must not be able to make us mint a session (or any
    other per-caller record) before the cheapest rejection has had its say.

    Health and metrics are exempt on purpose — rate-limiting the load
    balancer's probe is how an instance removes itself from rotation under the
    exact load it is coping with fine.
    """
    key = service.client_key(
        None,
        request.headers.get("x-forwarded-for"),
        request.client.host if request.client else None,
        TRUST_PROXY,
    )
    allowed, retry_after = service.ip_limiter.allow(key)
    if not allowed:
        raise _too_many(retry_after)


def _enforce_session_limit(session_id: str) -> None:
    """The inner, per-student guard. Tighter than the IP bucket because one
    human cannot legitimately outrun it, and looser to dodge (rotating session
    ids still leaves you facing the IP bucket)."""
    allowed, retry_after = service.session_limiter.allow(f"sess:{session_id}")
    if not allowed:
        raise _too_many(retry_after)


async def _run_blocking(fn, *args, **kwargs):
    """Admission-controlled hop onto the threadpool.

    Acquiring the semaphore with a timeout is what turns a spike into a queue
    with a bound instead of an unbounded backlog. asyncio.Semaphore handles the
    cancelled acquire correctly (the token is handed to the next waiter), so a
    timed-out request cannot leak a slot.
    """
    sem = STATE.sem
    if sem is None:  # only possible if a handler runs before lifespan startup
        return await run_in_threadpool(fn, *args, **kwargs)
    try:
        await asyncio.wait_for(sem.acquire(), timeout=QUEUE_TIMEOUT_S)
    except (asyncio.TimeoutError, TimeoutError):
        service.metrics.queue_timeout()
        raise HTTPException(
            status_code=503,
            detail="mimi is at capacity right now. Please try again in a moment.",
            headers={"Retry-After": "5"},
        ) from None
    service.metrics.enter()
    try:
        return await run_in_threadpool(fn, *args, **kwargs)
    finally:
        service.metrics.leave()
        sem.release()


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    session_id: str | None = Field(default=None, max_length=64)
    # Whatever the site already knows about the student (budget, marks, city).
    # Sanitised in service.sanitize_profile before it can reach a prompt.
    profile: dict[str, Any] | None = None


class AnswerRequest(BaseModel):
    """The pre-existing contract. Kept byte-compatible on purpose: the demo UI
    and any integration built against it keep working through this rewrite."""
    question: str = Field(max_length=MAX_MESSAGE_CHARS)
    mode: str = "db"                    # "db" | "web"
    prefs: dict[str, Any] | None = None
    history: list[dict[str, Any]] | None = None
    known_profile: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# College hydration — citations -> real pages
# ---------------------------------------------------------------------------
_AGG_KEYS = ("n_courses", "program_levels", "course_titles", "departments",
             "entrance_exams", "min_tuition_inr", "max_tuition_inr",
             "min_hostel_inr", "has_hostel", "has_scholarship", "card")


def _json_list(value: Any) -> list:
    if not value:
        return []
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except (TypeError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _college_url(college_id: str) -> str:
    return f"{PUBLIC_COLLEGE_URL}{college_id}"


def _lookup(ids: list[str]) -> dict[str, dict]:
    """Hydrate ids -> rows. Call only from a worker thread (see _in_thread)."""
    return db.fetch_colleges(db.get_conn(), ids)


def _resolve_citations(citations: list[str] | None) -> list[dict[str, Any]]:
    """Turn cited ids into link-ready records for the website. Runs on a worker
    thread (one indexed lookup, own connection).

    Ids that are not in the store are dropped rather than echoed back: a link
    to a college page that does not exist is worse than one fewer link, and the
    answer text itself was already verified against the retrieved context.
    """
    ids, seen = [], set()
    for cid in citations or []:
        if isinstance(cid, str) and _ID_RE.match(cid) and cid not in seen:
            seen.add(cid)
            ids.append(cid)
        if len(ids) >= MAX_CITED_COLLEGES:
            break
    if not ids:
        return []
    try:
        rows = _lookup(ids)
    except Exception as exc:  # noqa: BLE001 - links are a bonus, never the answer
        log.warning("citation hydration failed: %s", exc)
        return []
    out = []
    for cid in ids:  # preserve the order the answer cited them in
        row = rows.get(cid)
        if row is None:
            continue
        out.append({
            "college_id": cid,
            "name": row.get("name"),
            "city": row.get("city"),
            "state": row.get("state"),
            "url": _college_url(cid),
        })
    return out


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
def _answer_sync(question: str, history: list[dict] | None,
                 profile: dict[str, Any] | None) -> dict[str, Any]:
    """The one place the blocking pipeline is called. Runs on a worker thread."""
    from rag_core.pipeline import answer_question  # late import: heavy module

    return answer_question(question, history=history, known_profile=profile)


@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request):
    """Conversational endpoint — this is what makemyeducation.com embeds.

    Response: {answer, citations[], answered, reason_if_unanswered, session_id,
               profile, colleges[], cached}
    `colleges[]` mirrors `citations[]` with everything needed to render a link,
    so the site never has to know how ids map to URLs.
    """
    _enforce_ip_limit(request)
    await _require_corpus()
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="message must not be empty")

    snap = service.sessions.snapshot(req.session_id)
    _enforce_session_limit(snap.session_id)
    profile = service.merge_profile(snap.profile, req.profile)

    # Cache only the opening turn. Later turns depend on conversation history
    # (and may be personalised with the student's name), and a cache key that
    # does not include the transcript would serve one conversation's answer
    # into another's context.
    cache_key = service.answer_cache_key(message, profile) if not snap.history else None
    result = service.answer_cache.get(cache_key) if cache_key else None
    cached = result is not None
    request.state.cache_hit = cached

    if not cached:
        try:
            result = await _run_blocking(_answer_sync, message, snap.history, profile)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001 - provider/model failure
            log.exception("pipeline failed for session=%s", snap.session_id)
            raise HTTPException(
                status_code=503,
                detail="mimi could not reach its models just now. Please try again.",
                headers={"Retry-After": "5"},
            ) from exc
        # Only grounded, answered results are cached. Caching a failure would
        # pin a transient provider outage in place for the whole TTL.
        if cache_key and result.get("answered"):
            service.answer_cache.put(cache_key, service.project_answer(result))

    answer_text = str(result.get("answer", ""))
    profile_out = service.sanitize_profile(result.get("profile") or profile)
    service.sessions.record_turn(snap.session_id, message, answer_text, profile_out)

    return {
        "answer": answer_text,
        "citations": result.get("citations") or [],
        "answered": bool(result.get("answered")),
        "reason_if_unanswered": result.get("reason_if_unanswered"),
        "session_id": snap.session_id,
        "profile": profile_out,
        "colleges": await _in_thread(_resolve_citations, result.get("citations")),
        "cached": cached,
    }


@app.post("/api/answer")
async def api_answer(req: AnswerRequest, request: Request):
    """Backwards-compatible single-shot endpoint (the original contract).

    Same admission control and rate limiting as /api/chat — compatibility does
    not extend to letting an old client bypass the protections. The response
    body is passed through from the pipeline untouched.
    """
    _enforce_ip_limit(request)
    question = req.question.strip()
    if not question:
        return {"answer": "Please type a question.", "citations": [],
                "answered": False, "reason_if_unanswered": "Empty question."}

    if req.mode == "web":
        try:
            from rag_core.webmode import web_answer
        except Exception as exc:  # noqa: BLE001 - optional side path
            raise HTTPException(status_code=503,
                                detail="Web mode is not available on this deployment.") from exc
        return await _run_blocking(web_answer, question, req.prefs)

    await _require_corpus()
    # History is client-supplied here (this endpoint is stateless by design);
    # trim it so an oversized transcript cannot be used to inflate prompt cost.
    history = [m for m in (req.history or []) if isinstance(m, dict)][-12:]
    try:
        return await _run_blocking(_answer_sync, question, history or None,
                                   service.sanitize_profile(req.known_profile)
                                   if req.known_profile is not None else None)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        log.exception("pipeline failed on /api/answer")
        raise HTTPException(status_code=503,
                            detail="mimi could not reach its models just now.") from exc


@app.get("/api/health")
async def health():
    """Load-balancer probe. TTL-cached, no LLM, no retrieval.

    Readiness is expressed in the HTTP status so a balancer needs no body
    parsing: 200 = this instance can answer, 503 = pull it from rotation. The
    JSON body is for the human who then has to work out why.
    """
    payload = await _corpus_state()
    if not payload["ready"]:
        return JSONResponse(status_code=503, content=payload,
                            headers={"Retry-After": "30", "Cache-Control": "no-store"})
    return JSONResponse(content=payload, headers={"Cache-Control": "no-store"})


@app.get("/api/metrics")
async def metrics():
    """Operational counters for this process (latency percentiles, cache hit
    rate, rate-limit rejections, queue pressure). Per-answer analytics stay in
    metrics/metrics.jsonl — this is the live view."""
    snap = service.metrics.snapshot()
    snap["config"] = service.config_snapshot()
    snap["cache"] = service.answer_cache.stats()
    snap["sessions"] = service.sessions.stats()
    return snap


@app.get("/api/college/{college_id}")
async def college(college_id: str, request: Request):
    """The stored retrieval card plus the structured fields behind it.

    The card is served verbatim from the store rather than re-rendered: it is
    the exact text the retriever indexed and the generator was shown, so what
    the site displays and what the model reasoned over cannot drift apart.
    """
    _enforce_ip_limit(request)
    if not _ID_RE.match(college_id):
        raise HTTPException(status_code=404, detail="Unknown college id.")
    await _require_corpus()

    rows = await _in_thread(_lookup, [college_id])
    row = rows.get(college_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Unknown college id.")

    card = row.get("card")
    if not card:  # corpus built before cards were rendered — never invent one
        agg = {k: row.get(k) for k in _AGG_KEYS if k in row}
        card = db.render_card(row, agg, {})

    is_abroad = bool(row.get("is_abroad"))
    return {
        "college_id": college_id,
        "url": _college_url(college_id),
        "card": card,
        "name": row.get("name"),
        "short_name": row.get("short_name"),
        "city": row.get("city"),
        "district": row.get("district"),
        "state": row.get("state"),
        "country": row.get("country"),
        "type": row.get("type"),
        "establish_year": row.get("establish_year"),
        "naac_grade": row.get("naac_grade"),
        "nirf_ranking": row.get("nirf_ranking"),
        "affiliated_university": row.get("affiliated_university"),
        "website_url": row.get("website_url"),
        "rating": row.get("rating"),
        "is_abroad": is_abroad,
        "n_courses": row.get("n_courses") or 0,
        "program_levels": _json_list(row.get("program_levels")),
        "course_titles": _json_list(row.get("course_titles"))[:50],
        "entrance_exams": _json_list(row.get("entrance_exams")),
        "min_tuition_inr": row.get("min_tuition_inr"),
        "max_tuition_inr": row.get("max_tuition_inr"),
        # Explicit, because the fee fields alone are ambiguous: abroad rows
        # carry fees in the institution's own currency and the INR columns are
        # NULL by design. A client that renders "Rs {min_tuition_inr}" for an
        # overseas college would be publishing a fabricated number.
        "fees_are_inr": not is_abroad,
        "min_hostel_inr": row.get("min_hostel_inr"),
        "has_hostel": bool(row.get("has_hostel")),
        "has_scholarship": bool(row.get("has_scholarship")),
    }


@app.get("/api/colleges")
async def colleges(request: Request, ids: str = ""):
    """Bounded id -> display-name lookup for rendering citations.

    Deliberately NOT the old "dump every college" map: that was defensible at
    15 rows and is a multi-megabyte response at 35k. Callers pass the ids they
    actually need (comma-separated, capped), which is the same shape at 15
    colleges or at 35,000.
    """
    wanted = [i.strip() for i in ids.split(",") if i.strip() and _ID_RE.match(i.strip())]
    if not wanted:
        return {}
    # Same gates as every other corpus-reading route. Without them this one
    # answered a missing corpus with a 500 + error_ref (a page of citations
    # rendering as an internal error) instead of the clean 503 its siblings
    # return, and it was the one unmetered path into the store.
    _enforce_ip_limit(request)
    await _require_corpus()
    rows = await _in_thread(_lookup, wanted[:MAX_CITED_COLLEGES * 4])
    return {cid: {"name": r.get("name"), "city": r.get("city"), "state": r.get("state"),
                  "url": _college_url(cid)} for cid, r in rows.items()}


@app.delete("/api/session/{session_id}")
async def end_session(session_id: str):
    """"Clear chat" — frees the server-side memory immediately instead of
    waiting out the TTL. Not an auth boundary: session ids are unguessable
    random ids, and the worst a guesser achieves is deleting a transcript."""
    if not service.valid_session_id(session_id):
        raise HTTPException(status_code=404, detail="Unknown session.")
    return {"dropped": service.sessions.drop(session_id)}


@app.get("/")
async def home():
    """The bundled demo UI, when it is deployed alongside. In production the
    chat widget lives on makemyeducation.com and talks to /api/chat."""
    index = ROOT / "static" / "index.html"
    if not index.exists():
        return JSONResponse({"service": "MIVI API", "docs": "/docs",
                             "health": "/api/health"})
    return FileResponse(index)


if __name__ == "__main__":
    import uvicorn

    # One worker per process by design: sessions, the answer cache and the rate
    # limiters are in-process, so N workers means N independent copies. Scale
    # up with MIVI_MAX_CONCURRENCY first; scale out only behind a session-sticky
    # balancer, or after moving service.py's state to Redis.
    uvicorn.run("app:app", host=os.getenv("MIVI_HOST", "0.0.0.0"),
                port=int(os.getenv("MIVI_PORT", "8000")), workers=1,
                log_level=os.getenv("MIVI_LOG_LEVEL", "info").lower())
