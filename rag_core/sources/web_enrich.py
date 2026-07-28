"""Harvest missing college facts from each college's OWN website.

WHY THIS EXISTS
The MakeMyEducation catalogue is strong on identity (name, city, type, courses,
tuition) and empty on almost everything a student asks next: scholarships,
hostel charges, placements, admission dates. Measured across the corpus those
fields are ~100% blank. The colleges themselves publish that information, so
this module goes and reads it.

WHY IT DOES NOT USE FIRECRAWL AS THE ENGINE
Firecrawl is configured on this account with 990 credits left on a 1,000/month
plan. At ~3 pages per college that is ~330 of 38,700 colleges — under 1%. So
the primary engine is plain HTTP: free, unmetered, and entirely adequate for
college websites, which are overwhelmingly static HTML. Firecrawl stays wired
in as a FALLBACK for the minority of sites that are JavaScript-rendered or
actively block a simple client, where a credit actually buys something.

WHAT IT WILL NOT DO
- It will not ignore robots.txt. These are third-party sites, not ours.
- It will not hammer a host: requests are paced per-domain, and concurrency is
  across domains rather than within one.
- It will not write into the curated tables. Everything lands in
  `college_web_facts` with its source URL and fetch date, because an LLM
  reading a fee off a college's PDF is a weaker claim than the catalogue and
  MIVI has to be able to say which is which. Promotion is a separate,
  reviewed decision.

Run:
    python -m rag_core.sources.web_enrich --limit 25          # smoke test
    python -m rag_core.sources.web_enrich --concurrency 12    # full sweep
    python -m rag_core.sources.web_enrich --status            # progress report
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import os
import sys
import threading
import time
import urllib.robotparser
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import httpx

# Imported for its side effect as much as its values, and it must come BEFORE
# the backend import: rag_core.config is the only module that calls
# load_dotenv(), and store.backend reads MIVI_PG_DSN at ITS import time. Get the
# order wrong and a sweep silently targets the built-in localhost default
# instead of the configured server — hours of politeness-limited crawling
# landing in the wrong database.
from .. import config  # noqa: F401
from ..store.backend import get_backend
from .verify_facts import verify as verify_fact

UA = ("Mozilla/5.0 (compatible; MakeMyEducationBot/1.0; "
      "+https://makemyeducation.com/about) MIVI college-data enrichment")

HEADERS = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-IN,en;q=0.9",
}

# Pages worth spending a request on, in priority order. The keys double as the
# `kind` recorded in college_web_facts.
PAGE_TARGETS = (
    # Ordered by what the catalogue is missing MOST and what students ask for
    # first. The cap below means a college with every page only gets the top of
    # this list, so the order is the priority.
    ("scholarship", ("scholarship", "scholarships", "financial-aid", "financial aid",
                     "fee-concession", "freeship", "fee-waiver")),
    ("fees", ("fee", "fees", "fee-structure", "tuition", "fee structure",
              "fee-details", "fees-and-payment")),
    ("cutoff", ("cutoff", "cut-off", "cut off", "merit-list", "merit list",
                "last-rank", "closing-rank", "counselling", "selection")),
    ("admission", ("admission", "admissions", "apply", "eligibility", "how-to-apply",
                   "important-dates")),
    ("placement", ("placement", "placements", "training-placement", "career",
                   "recruiters", "campus-recruitment")),
    ("hostel", ("hostel", "hostels", "accommodation", "campus-life", "facilities",
                "mess")),
    ("events", ("event", "events", "news", "notice", "announcement", "circular",
                "fest", "activities")),
)

# THE budget dial. Every page is one rate-limited extraction call, so under a
# free-tier quota (14/min) this decides coverage vs depth directly:
#   4 pages -> ~10,000 colleges get scholarship/fees/cutoff/admission
#   7 pages -> ~5,700 colleges get all of it, and the rest get nothing
# Breadth wins at this stage: a student is far more likely to ask about a
# college we have partially covered than to need the events page of one we
# covered exhaustively. Raise it once the account is on a paid tier.
MAX_PAGES_PER_COLLEGE = int(os.getenv("MIVI_PAGES_PER_COLLEGE", "4"))
PAGE_CHAR_BUDGET = 6_000        # per page handed to the extractor
FETCH_TIMEOUT = 20.0
MAX_HTML_BYTES = 3_000_000      # skip pathological pages rather than parse them


# ---------------------------------------------------------------------------
# HTML -> text/links, on stdlib only
# ---------------------------------------------------------------------------

class _Extract(HTMLParser):
    """Pull visible text + anchors. A real parser (trafilatura/bs4) would be
    nicer, but this avoids adding a dependency for what is fundamentally
    "strip tags and collect hrefs", and it cannot fail on malformed markup the
    way a stricter parser can."""

    _SKIP = {"script", "style", "noscript", "svg", "head"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.chunks: list[str] = []
        self.links: list[tuple[str, str]] = []   # (href, anchor text)
        self._skip_depth = 0
        self._href: str | None = None
        self._anchor: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag == "a":
            self._href = dict(attrs).get("href")
            self._anchor = []
        elif tag in ("br", "p", "div", "li", "tr", "h1", "h2", "h3"):
            self.chunks.append("\n")

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "a":
            if self._href:
                self.links.append((self._href, " ".join(self._anchor).strip()[:120]))
            self._href, self._anchor = None, []

    def handle_data(self, data):
        if self._skip_depth:
            return
        self.chunks.append(data)
        if self._href is not None:
            self._anchor.append(data)

    def text(self) -> str:
        raw = "".join(self.chunks)
        raw = re.sub(r"[ \t\r\f\v]+", " ", raw)
        return re.sub(r"\n\s*\n\s*", "\n", raw).strip()


def parse_html(html: str) -> tuple[str, list[tuple[str, str]]]:
    p = _Extract()
    try:
        p.feed(html)
    except Exception:  # noqa: BLE001 - malformed markup must never kill a sweep
        pass
    return p.text(), p.links


# ---------------------------------------------------------------------------
# Politeness: robots.txt + per-domain pacing
# ---------------------------------------------------------------------------

class Politeness:
    """robots.txt compliance and one-request-at-a-time-per-host pacing.

    Concurrency is deliberately ACROSS domains, never within one: 12 workers
    hitting 12 different colleges is neighbourly, 12 workers hitting one small
    college's shared hosting is a denial of service.
    """

    def __init__(self, delay: float = 1.5):
        self.delay = delay
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self._next_ok: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._guard = asyncio.Lock()

    async def _lock_for(self, host: str) -> asyncio.Lock:
        async with self._guard:
            return self._locks.setdefault(host, asyncio.Lock())

    async def allowed(self, client: httpx.AsyncClient, url: str) -> bool:
        host = urlparse(url).netloc
        if host not in self._robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                r = await client.get(f"{urlparse(url).scheme}://{host}/robots.txt",
                                     timeout=10.0)
                if r.status_code == 200 and len(r.text) < 500_000:
                    rp.parse(r.text.splitlines())
                else:
                    rp = None          # no usable robots.txt -> default allow
            except Exception:          # noqa: BLE001 - unreachable robots != disallow
                rp = None
            self._robots[host] = rp
        rp = self._robots[host]
        if rp is None:
            return True
        try:
            return rp.can_fetch(UA, url)
        except Exception:  # noqa: BLE001
            return True

    async def wait(self, url: str) -> None:
        host = urlparse(url).netloc
        lock = await self._lock_for(host)
        async with lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            due = self._next_ok.get(host, 0.0)
            if due > now:
                await asyncio.sleep(due - now)
            self._next_ok[host] = max(now, due) + self.delay


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def normalise_site(url: str | None) -> str | None:
    """Coerce whatever the catalogue stored into a fetchable origin."""
    if not url:
        return None
    u = str(url).strip()
    if not u or u.lower() in {"na", "n/a", "-", "nil", "none"}:
        return None
    if not re.match(r"^https?://", u, re.I):
        u = "https://" + u.lstrip("/")
    p = urlparse(u)
    if not p.netloc or "." not in p.netloc:
        return None
    return u


class FirecrawlBudget:
    """Spends the Firecrawl allowance only where plain HTTP actually failed.

    The account holds ~1,000 credits/month, which is under 1% of the corpus —
    so credits are worthless as a bulk engine and valuable as a rescue. Plain
    HTTP fails on roughly a quarter of college sites (JS-rendered menus, WAFs
    that reject a non-browser client); those pages are exactly what a credit
    buys, and everything else must never touch it.

    A reserve is held back so a long sweep cannot drain the month's allowance
    in one run and leave nothing for the next.
    """

    ENDPOINT = "https://api.firecrawl.dev/v2/scrape"
    USAGE = "https://api.firecrawl.dev/v2/team/credit-usage"

    def __init__(self, reserve: int = 100, max_spend: int | None = None):
        import os

        # Several keys, pooled. One free-tier account (1,000 credits/month) is
        # not enough for the blocked-site backlog — the first full sweep spent
        # ~880 of them in one run — so keys are read as a comma-separated list
        # and drained in order. FIRECRAWL_API_KEY stays supported so nothing
        # that set only that keeps working.
        raw = (os.getenv("FIRECRAWL_API_KEYS")
               or os.getenv("FIRECRAWL_API_KEY") or "")
        self.keys = [k.strip() for k in raw.split(",") if k.strip()]
        self.reserve = reserve
        self.max_spend = max_spend
        self.spent = 0
        self._i = 0                          # index of the key in use
        self._left: list[int] = []           # spendable credits per key
        self.remaining: int | None = None
        self.enabled = bool(self.keys)

    async def prime(self, client: httpx.AsyncClient) -> None:
        if not self.enabled:
            return
        total = 0
        self._left = []
        for key in self.keys:
            try:
                r = await client.get(self.USAGE,
                                     headers={"Authorization": f"Bearer {key}"},
                                     timeout=20.0)
                have = int((r.json().get("data") or {}).get("remainingCredits", 0))
            except Exception:  # noqa: BLE001 - unknown balance: treat as empty
                have = 0
            # The reserve is held back PER KEY, not once across the pool: each
            # account has its own monthly cycle, so leaving a cushion on only
            # one of them would drain the others to zero.
            spendable = max(0, have - self.reserve)
            self._left.append(spendable)
            total += spendable
        if self.max_spend is not None:
            total = min(total, self.max_spend)
        self._budget = total
        self.remaining = total
        self.enabled = total > 0
        print(f"[firecrawl] {len(self.keys)} key(s), "
              f"{[l + self.reserve for l in self._left]} credits "
              f"(reserve {self.reserve} each) -> "
              f"{'spending up to ' + str(total) if self.enabled else 'DISABLED'}",
              file=sys.stderr)

    def _key(self) -> str | None:
        """The current key, advancing past any that are spent out."""
        while self._i < len(self.keys):
            if self._left[self._i] > 0:
                return self.keys[self._i]
            self._i += 1
            if self._i < len(self.keys):
                print(f"[firecrawl] key {self._i} exhausted, moving to the next",
                      file=sys.stderr)
        return None

    def available(self) -> bool:
        return (self.enabled and self.spent < getattr(self, "_budget", 0)
                and self._key() is not None)

    async def scrape(self, client: httpx.AsyncClient, url: str) -> str | None:
        key = self._key() if self.available() else None
        if key is None:
            return None
        self.spent += 1
        self._left[self._i] -= 1
        try:
            r = await client.post(
                self.ENDPOINT,
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"},
                json={"url": url, "formats": ["html"], "onlyMainContent": False},
                timeout=90.0,
            )
            if r.status_code in (402, 429):
                # Out of credit or throttled on THIS key: burn it down so the
                # pool moves on instead of retrying a dead account.
                print(f"[firecrawl] key {self._i + 1} returned "
                      f"{r.status_code}; retiring it", file=sys.stderr)
                self._left[self._i] = 0
                return None
            if r.status_code != 200:
                return None
            data = (r.json() or {}).get("data") or {}
            # `html` keeps the anchors the link picker needs; markdown drops the
            # href structure this crawler navigates by.
            return data.get("html") or data.get("rawHtml") or None
        except Exception:  # noqa: BLE001
            return None


async def fetch(client: httpx.AsyncClient, pol: Politeness, url: str,
                fc: "FirecrawlBudget | None" = None) -> str | None:
    """One page. Returns HTML, or None for anything not worth retrying.

    Plain HTTP first; Firecrawl only as a rescue when that returns nothing and
    robots.txt permits the fetch. robots is checked BEFORE the fallback too —
    paying a credit does not buy permission.
    """
    if not await pol.allowed(client, url):
        return None
    await pol.wait(url)
    html = None
    try:
        r = await client.get(url)
        if (r.status_code == 200
                and "html" in r.headers.get("content-type", "").lower()
                and len(r.content) <= MAX_HTML_BYTES):
            html = r.text
    except Exception:  # noqa: BLE001 - DNS/TLS/timeout are all "try the rescue"
        html = None
    if html:
        return html
    if fc is not None:
        return await fc.scrape(client, url)
    return None


def pick_pages(base_url: str, links: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Choose which subpages to spend requests on: (kind, absolute_url).

    Matching looks at BOTH the href and the anchor text, because Indian college
    sites very often link "Scholarship" to something like /page.php?id=42 where
    the URL alone gives nothing away.
    """
    origin = urlparse(base_url).netloc
    chosen: dict[str, str] = {}
    for href, anchor in links:
        if not href or href.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue
        absolute = urljoin(base_url, href)
        p = urlparse(absolute)
        if p.scheme not in ("http", "https") or p.netloc != origin:
            continue  # stay on the college's own domain
        hay = f"{p.path.lower()} {anchor.lower()}"
        for kind, keys in PAGE_TARGETS:
            if kind in chosen:
                continue
            if any(k in hay for k in keys):
                chosen[kind] = absolute
                break
        if len(chosen) >= MAX_PAGES_PER_COLLEGE:
            break
    return list(chosen.items())


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

EXTRACT_SYSTEM = """You extract verifiable facts from a college's own web page for a
student advice service. You will be given the college name and the visible text
of ONE page from its official website.

Return ONLY JSON, exactly this shape:
{"summary": "<=45 words, plain factual prose, or empty string if nothing useful>",
 "facts": {"<field>": "<value>", ...},
 "confidence": <0.0-1.0>}

Hard rules:
1. Extract ONLY what the page literally states. Never infer, never complete a
   partial figure, never use outside knowledge about this or any college.
2. If the page says nothing concrete about the topic (a nav page, a landing
   page, a 'coming soon'), return {"summary":"","facts":{},"confidence":0.0}.
   An empty result is a CORRECT and expected answer — it is far better than a
   vague one.
3. Amounts: copy the number and its currency/period exactly as written
   ("Rs 45,000 per year", "$12,000/semester"). Never convert. Never annualise.
4. Do not include marketing language, rankings the page claims about itself,
   or anything phrased as a promise ("best placements in the region").
5. confidence reflects how explicitly the page states these facts: 0.9 for a
   clear fee table, 0.3 for a passing mention, 0.0 for nothing."""

EXTRACT_USER = """COLLEGE: {name}{where}
TOPIC: {kind}
SOURCE URL: {url}

PAGE TEXT:
{text}"""


class ExtractPacer:
    """Paces extraction calls to the provider's actual per-minute quota.

    Both configured providers are free tier, and the tighter one (Gemini) allows
    15 requests/minute. Ten workers calling freely turn that into a continuous
    429 storm: every call still costs a round trip, the retry burns the next
    slot, and throughput collapses BELOW the quota while looking like progress.

    Pacing to the known limit is strictly better than discovering it by
    failing — the sweep then runs at the fastest rate the quota allows, and the
    logs show real extractions instead of pages of quota errors. Raise
    MIVI_EXTRACT_RPM when the account moves to a paid tier; that single number
    is what turns this from a multi-day job into a multi-hour one.
    """

    def __init__(self, rpm: int | None = None):
        self.rpm = rpm or int(os.getenv("MIVI_EXTRACT_RPM", "14"))
        self._interval = 60.0 / max(1, self.rpm)
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            due = max(now, self._next)
            self._next = due + self._interval
        delay = due - now
        if delay > 0:
            time.sleep(delay)


PACER = ExtractPacer()

# Deterministic extractors, loaded once. Each is a plain-Python parser for one
# topic that costs no API quota, so wherever one succeeds the LLM call is
# skipped entirely — the difference between a multi-day sweep and a multi-hour
# one. A missing module is normal (they are added topic by topic), so the
# import failure is silent and the LLM fallback simply keeps handling that kind.
_DETERMINISTIC: dict = {}


def _load_extractors() -> dict:
    if _DETERMINISTIC:
        return _DETERMINISTIC
    import importlib

    for kind, _ in PAGE_TARGETS:
        try:
            mod = importlib.import_module(
                f"rag_core.sources.extractors.{kind}")
            if hasattr(mod, "extract"):
                _DETERMINISTIC[kind] = mod.extract
        except Exception:  # noqa: BLE001 - absent or broken module: use the LLM
            continue
    if _DETERMINISTIC:
        print(f"[web] deterministic extractors: {', '.join(sorted(_DETERMINISTIC))}",
              file=sys.stderr)
    return _DETERMINISTIC


# Below this, a deterministic result is not trusted on its own and the page is
# escalated to the LLM. Set deliberately at the same 0.5 the card renderer uses
# to decide whether a fact is worth showing at all.
DETERMINISTIC_MIN_CONFIDENCE = float(os.getenv("MIVI_DET_MIN_CONF", "0.6"))


def extract_topic(name: str, where: str, kind: str, url: str, text: str,
                  use_llm: bool = True, min_conf: float | None = None) -> dict | None:
    """Deterministic parser first, LLM only when it declines.

    Ordering matters for cost, not just speed: the LLM path is capped at ~14
    calls/minute across the whole account, so every page a parser handles is a
    page that does not queue behind that limit. Measured: the parsers run in
    0.3-1.8 ms against roughly 4,000 ms per LLM call, so a parser hit is not an
    optimisation — it is the difference between a 28-hour sweep and a 30-minute
    one.

    `use_llm=False` is the fast pass: parsers only, no quota, fetch-bound. It
    covers what it can across the whole corpus and leaves the rest for a slow
    LLM pass later. Getting scholarship data for 8,000 colleges today beats
    getting all seven fields for 400 colleges today.
    """
    fn = _load_extractors().get(kind)
    if fn is not None:
        try:
            got = fn(text, url, name)
        except Exception as exc:  # noqa: BLE001 - a parser bug must not stop a sweep
            print(f"[web] {kind} parser raised on {url}: {exc}", file=sys.stderr)
            got = None
        # OCR text is noisier than HTML, so a parser reading a scan reports
        # lower confidence for the same quality of finding. Holding it to the
        # HTML gate discarded every OCR result — 16 successful recognitions
        # yielded zero facts. A lower gate is safe HERE because the caller caps
        # the result at the OCR ceiling and anything under 0.5 is quarantined,
        # never shown: the choice is between a reviewable low-confidence fact
        # and nothing at all.
        gate = DETERMINISTIC_MIN_CONFIDENCE if min_conf is None else min_conf
        if got and float(got.get("confidence") or 0) >= gate:
            got["_via"] = "parser"
            return got
    if not use_llm:
        return None
    got = extract_facts(name, where, kind, url, text)
    if got:
        got["_via"] = "llm"
    return got


def extract_facts(name: str, where: str, kind: str, url: str, text: str) -> dict | None:
    """One LLM call -> structured facts for one page. None on failure.

    Runs on a worker thread (see enrich_one), so the blocking pace-wait here
    never stalls the event loop — other domains keep fetching while this waits
    for its quota slot.
    """
    from ..config import FAST_MODEL
    from ..llm import chat
    from ..pipeline import _parse_json

    if len(text.strip()) < 200:
        return None
    PACER.wait()
    try:
        raw = chat(
            [{"role": "system", "content": EXTRACT_SYSTEM},
             {"role": "user", "content": EXTRACT_USER.format(
                 name=name, where=where, kind=kind, url=url,
                 text=text[:PAGE_CHAR_BUDGET])}],
            model=FAST_MODEL, json_mode=True, temperature=0.0, max_tokens=500,
        )
        out = _parse_json(raw)
    except Exception as exc:  # noqa: BLE001 - one bad page must not stop a sweep
        print(f"[web] extract failed {url}: {exc}", file=sys.stderr)
        return None
    if not isinstance(out, dict):
        return None
    summary = str(out.get("summary") or "").strip()
    facts = out.get("facts") if isinstance(out.get("facts"), dict) else {}
    try:
        conf = float(out.get("confidence") or 0.0)
    except (TypeError, ValueError):
        conf = 0.0
    if not summary and not facts:
        return None
    return {"summary": summary[:600], "facts": facts, "confidence": max(0.0, min(1.0, conf))}


# ---------------------------------------------------------------------------
# Persistence
#
# Every write below goes to Postgres. Reads go through get_backend() and use
# the codebase's `?` placeholders; writes go straight to psycopg and use its
# own `%s`, because they never pass through the backend's rewriter. The
# placeholder tells you which path a statement takes.
#
# This module is why the store had to move. It writes continuously, from every
# worker, while students read; SQLite serialises writers, so the old code
# funnelled every UPDATE through one connection behind one asyncio lock and
# still contended with every reader on the file.
# ---------------------------------------------------------------------------

def _now() -> datetime:
    # fetched_at and last_attempt are TIMESTAMPTZ. Passing a real datetime lets
    # "harvested more than N days ago" be a SQL predicate, instead of the
    # parse-every-row loop that SQLite's text timestamps forced (see
    # requeue_stale).
    return datetime.now(timezone.utc)


def _pool():
    """The connection pool every write in this module goes through.

    Deliberately the pool and not backend.execute()/executemany(), which are
    otherwise the right door for a writer: the harvest checkpoint is TWO
    statements — the extracted facts, and the web_crawl_state rows declaring
    those colleges done — and they have to land in ONE transaction (see
    HarvestWriter). Those helpers give a transaction per call, so a kill landing
    between them would either strand facts under a college still listed pending
    or, worse, mark a college harvested with its facts missing, and nothing
    would ever come back for it.

    sweep() calls this before the first request goes out. get_backend() already
    fails hard on an unreachable database, and tripping that HERE is the point:
    a sweep that started anyway would spend hours being politely rate-limited by
    third-party websites and then drop every extracted fact on the floor.
    """
    pool = getattr(get_backend(), "_pool", None)
    if pool is None:   # only reachable if the backend stops exposing a pool
        raise RuntimeError("web_enrich needs the Postgres connection pool and "
                           "the active backend does not expose one")
    return pool


FACT_UPSERT = """
INSERT INTO college_web_facts
    (college_id, kind, summary, payload, source_url, fetched_at, extractor,
     confidence)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (college_id, kind) DO UPDATE SET
    summary    = EXCLUDED.summary,
    payload    = EXCLUDED.payload,
    source_url = EXCLUDED.source_url,
    fetched_at = EXCLUDED.fetched_at,
    extractor  = EXCLUDED.extractor,
    confidence = EXCLUDED.confidence"""

# An upsert, not the bare UPDATE this used to be. An UPDATE that matches no row
# reports success and quietly loses the result, so any college whose queue row
# went missing — requeued by hand, trimmed, or seeded by a second process that
# lost the race — would be re-crawled on every sweep forever, spending the
# politeness budget on work already done. INSERT ... ON CONFLICT cannot no-op.
#
# attempts is read from the TABLE, never from EXCLUDED: EXCLUDED.attempts is the
# literal 1 this statement proposed, so using it would reset the counter and
# hide the college that has now failed nine times.
#
# website_url is set on insert but NOT in the DO UPDATE list — seed_queue owns
# that column, and a sweep must not overwrite a corrected URL with the one it
# happened to be handed.
STATE_UPSERT = """
INSERT INTO web_crawl_state
    (college_id, website_url, status, last_attempt, attempts, pages_fetched)
VALUES (%s, %s, %s, %s, 1, %s)
ON CONFLICT (college_id) DO UPDATE SET
    status        = EXCLUDED.status,
    last_attempt  = EXCLUDED.last_attempt,
    attempts      = web_crawl_state.attempts + 1,
    pages_fetched = EXCLUDED.pages_fetched"""

# DO NOTHING, not DO UPDATE. seed_queue's SELECT already excludes every college
# that has a row, so a conflict here can only mean another process seeded the
# same college in between — and ITS row carries the live status. Writing
# 'pending' over that would resurrect a finished college for a second crawl.
STATE_SEED = """
INSERT INTO web_crawl_state (college_id, website_url, status, attempts)
VALUES (%s, %s, %s, 0)
ON CONFLICT (college_id) DO NOTHING"""


def seed_queue() -> int:
    """Populate web_crawl_state from colleges that have a usable website URL.

    Ordered by n_courses so the sweep spends its first (and possibly only)
    hours on the colleges students are most likely to actually reach.
    """
    rows = get_backend().q(
        """SELECT c.college_id, c.website_url
           FROM colleges c LEFT JOIN college_agg a USING(college_id)
           WHERE c.website_url IS NOT NULL AND TRIM(c.website_url) <> ''
             AND c.college_id NOT IN (SELECT college_id FROM web_crawl_state)
           ORDER BY COALESCE(a.n_courses, 0) DESC"""
    )
    if not rows:
        return 0
    seed = []
    for r in rows:
        site = normalise_site(r["website_url"])
        seed.append((r["college_id"], site, "pending" if site else "no_site"))
    with _pool().connection() as c:
        c.cursor().executemany(STATE_SEED, seed)
    return len(seed)


class HarvestWriter:
    """Buffers one batch of harvest rows and lands it in a single transaction.

    Why buffer at all, when Postgres takes concurrent writes happily: crawling
    a college takes SECONDS — politeness policy is 1.5 s between requests to a
    domain — so a transaction held open across one would pin an XID for minutes
    and keep autovacuum off the two tables this sweep churns hardest. Buffering
    checks a pooled connection out only for the flush, which is milliseconds.

    The batch is also the crash unit, and both statements belong to it
    together: the extracted facts and the web_crawl_state rows saying those
    colleges are done commit as one. A killed sweep therefore loses at most
    `every` colleges and re-crawls exactly those. It can never mark a college
    harvested with its facts missing, nor strand facts under a college still
    listed pending — which is what keeps `--status` an honest resume point.
    """

    def __init__(self, every: int = 25):
        self.every = every
        self._facts: list[tuple] = []
        self._state: list[tuple] = []
        self._done = 0

    def fact(self, cid: str, kind: str, url: str, got: dict) -> None:
        # Serialised here rather than at flush time: a value that will not
        # encode is one extractor misbehaving on one page, and it has to fail
        # that page instead of aborting the whole batch's transaction.
        self._facts.append(
            (cid, kind, got["summary"],
             json.dumps(got["facts"], ensure_ascii=False),
             url, _now(), got.get("_via", "llm"), got["confidence"]))

    def mark(self, cid: str, status: str, pages: int, site: str | None) -> None:
        # `site` is carried only for the insert branch of STATE_UPSERT, i.e. the
        # college whose queue row is gone; on the normal path the row exists and
        # this value is discarded.
        self._state.append((cid, site, status, _now(), pages))
        self._done += 1

    @property
    def due(self) -> bool:
        return self._done >= self.every

    async def flush(self) -> None:
        """Swap the buffers, then commit off the event loop so the workers
        currently mid-fetch keep progressing while this writes."""
        facts, self._facts = self._facts, []
        state, self._state = self._state, []
        self._done = 0
        if facts or state:
            await asyncio.to_thread(self._commit, facts, state)

    @staticmethod
    def _commit(facts: list[tuple], state: list[tuple]) -> None:
        with _pool().connection() as c:   # commits on exit, rolls back on error
            cur = c.cursor()
            if facts:
                cur.executemany(FACT_UPSERT, facts)
            if state:
                cur.executemany(STATE_UPSERT, state)


async def enrich_one(client, pol, writer: "HarvestWriter", college: dict, fc=None,
                     use_llm: bool = True) -> dict:
    """Homepage -> pick subpages -> fetch -> extract -> store."""
    cid, name, site = college["college_id"], college["name"], college["website_url"]
    where = f", {college['city']}" if college.get("city") else ""
    stats = {"pages": 0, "facts": 0, "status": "empty"}

    home = await fetch(client, pol, site, fc)
    if home is None:
        stats["status"] = "blocked"
        return stats
    stats["pages"] += 1
    _, links = parse_html(home)
    targets = pick_pages(site, links)
    if not targets:
        stats["status"] = "empty"
        return stats

    for kind, url in targets:
        # NO Firecrawl for sub-pages. Measured: passing the budget down here
        # spent 11 credits on 25 colleges — at that rate the whole month's
        # allowance covers ~2,250 of 38,700 — and dropped throughput from
        # 1.8 colleges/s to 0.1 because each call blocks for tens of seconds.
        #
        # A credit is only worth spending when the SITE blocks plain HTTP
        # outright (handled at the homepage fetch above, where one credit
        # unlocks a whole college). If the homepage came back fine, a missing
        # /fees page is simply a page that does not exist — rendering it in a
        # headless browser will not conjure one.
        html = await fetch(client, pol, url, None)
        if html is None:
            continue
        stats["pages"] += 1
        text, _ = parse_html(html)
        # The LLM call is sync and CPU/network bound elsewhere; keep the event
        # loop free so other domains keep progressing.
        got = await asyncio.to_thread(extract_topic, name, where, kind, url,
                                      text, use_llm)
        if not got:
            continue

        # GATE. Nothing reaches the served corpus on the extractor's own word.
        # The decisive check is that every number the fact asserts appears
        # VERBATIM in `text` — the page it was just read from. An extractor
        # cannot self-detect the failure this catches: measured on the first
        # 400 harvested facts, 2% asserted figures that were not on their source
        # page at all, and one of them carried a self-reported confidence of
        # 0.95. A score is an opinion; anchoring is evidence.
        #
        # UNCERTAIN is kept, not discarded: those are usually real values with a
        # dropped unit ("Rs 45,000" without "per year"), which is worth
        # reviewing rather than throwing away. They are stored with their
        # verdict and the card renderer never shows them.
        checked = await asyncio.to_thread(
            verify_fact, got, text, name, college.get("city"))
        got["confidence"] = checked.confidence
        got["_verdict"] = checked.verdict
        got["_reasons"] = "; ".join(checked.reasons)[:400]
        if checked.anchors:
            got["_anchor"] = " | ".join(list(checked.anchors.values())[:2])[:600]

        if checked.verdict == "REJECTED":
            stats["rejected"] = stats.get("rejected", 0) + 1
            print(f"[web] REJECTED {kind} for {name[:34]}: "
                  f"{got['_reasons'][:90]}", file=sys.stderr)
            continue

        writer.fact(cid, kind, url, got)
        stats["facts"] += 1
        if checked.verdict == "UNCERTAIN":
            stats["uncertain"] = stats.get("uncertain", 0) + 1
    # ---- PDFs, and OCR for the scanned ones -------------------------------
    # Measured: only 14% of college sites carry a rupee figure in their HTML,
    # while 28% link a PDF about fees/scholarships/prospectus — and 70% of THOSE
    # are scans with no extractable text. So the fee data is disproportionately
    # in images, which is why OCR is here rather than being called optional.
    #
    # Runs only when the HTML pass came up short: a college whose fee page
    # parsed cleanly does not need its prospectus rasterised, and OCR is by far
    # the most expensive step in the sweep.
    if stats["facts"] < 2:
        try:
            from .pdf_harvest import fetch_pdf, find_pdf_links, pdf_to_text

            # OCR is the most expensive step in the sweep — two full-page
            # renders plus recognition per PDF — so it is spent only on the
            # topics the catalogue is actually missing and students actually
            # ask about. A scanned placement brochure or hostel photo-sheet is
            # not worth a minute of CPU; a fee circular is.
            OCR_KINDS = {"fees", "scholarship", "admission", "cutoff", "courses"}
            for kind, pdf_url in find_pdf_links(site, links, limit=3):
                if kind not in OCR_KINDS:
                    continue
                data, note = await fetch_pdf(client, pol, pdf_url)
                if data is None:
                    continue
                stats["pages"] += 1
                text, note = pdf_to_text(data)
                ceiling = 1.0
                if not text:
                    # No text layer: it is a scan. OCR reads it twice at
                    # different DPI and keeps only the figures both passes
                    # agree on, so one misread digit cannot become a fee.
                    from .ocr import confidence_ceiling, ocr_pdf

                    text, rep = await asyncio.to_thread(ocr_pdf, data)
                    ceiling = confidence_ceiling(rep)
                    if text:
                        stats["ocr"] = stats.get("ocr", 0) + 1
                        print(f"[web] OCR {kind} for {name[:28]}: "
                              f"{len(text)} chars, {rep.get('corroborated', 0)} "
                              f"figures corroborated, ceiling {ceiling}",
                              file=sys.stderr)
                if not text:
                    continue

                got = await asyncio.to_thread(extract_topic, name, where, kind,
                                              pdf_url, text, use_llm,
                                              0.35 if ceiling < 1.0 else None)
                if not got:
                    continue
                # An OCR page can never publish at full confidence, however sure
                # the extractor is: the ceiling reflects how well the recogniser
                # could read the page at all.
                got["confidence"] = min(float(got.get("confidence") or 0), ceiling)

                checked = await asyncio.to_thread(
                    verify_fact, got, text, name, college.get("city"))
                got["confidence"] = min(checked.confidence, ceiling)
                got["_verdict"] = checked.verdict
                got["_reasons"] = "; ".join(checked.reasons)[:400]
                if checked.verdict == "REJECTED":
                    stats["rejected"] = stats.get("rejected", 0) + 1
                    continue
                writer.fact(cid, kind, pdf_url, got)
                stats["facts"] += 1
                stats["pdf_facts"] = stats.get("pdf_facts", 0) + 1
        except Exception as exc:  # noqa: BLE001 - a PDF must not kill a college
            print(f"[web] pdf/ocr step failed for {name[:30]}: "
                  f"{str(exc)[:90]}", file=sys.stderr)

    stats["status"] = "ok" if stats["facts"] else "empty"
    return stats


async def sweep(limit: int | None = None, concurrency: int = 10,
                delay: float = 1.5, firecrawl_reserve: int = 100,
                firecrawl_max: int | None = None,
                use_llm: bool = True) -> dict:
    _pool()   # fail now, not after an hour of polite crawling
    seeded = seed_queue()
    if seeded:
        print(f"[web] queued {seeded} colleges with a website URL", file=sys.stderr)

    # Ordered by n_courses, not by insertion order. SQLite's rowid gave the
    # seed order for free; Postgres has no such column, and ordering on ctid
    # would be ordering on physical position — which moves every time a row is
    # updated, i.e. every time this sweep touches it. Restating the seed's own
    # priority is both stable and what the ordering always meant.
    todo = get_backend().q(
        "SELECT s.college_id, s.website_url, c.name, c.city "
        "FROM web_crawl_state s "
        "JOIN colleges c USING(college_id) "
        "LEFT JOIN college_agg a USING(college_id) "
        "WHERE s.status = 'pending' AND s.website_url IS NOT NULL "
        "ORDER BY COALESCE(a.n_courses, 0) DESC LIMIT ?", [limit or 1_000_000])
    if not todo:
        print("[web] nothing pending", file=sys.stderr)
        return {"done": 0}

    print(f"[web] sweeping {len(todo)} colleges "
          f"(concurrency {concurrency}, {delay}s per domain)", file=sys.stderr)

    pol = Politeness(delay=delay)
    sem = asyncio.Semaphore(concurrency)
    counters = {"ok": 0, "empty": 0, "blocked": 0, "facts": 0}
    done = 0
    t0 = time.perf_counter()
    lock = asyncio.Lock()
    # 25 colleges per transaction: the crash window, and the same cadence the
    # SQLite version committed on. Small enough that a kill costs a minute of
    # crawling, large enough that the commit is not the bottleneck.
    writer = HarvestWriter(every=25)

    fc = FirecrawlBudget(reserve=firecrawl_reserve, max_spend=firecrawl_max)

    limits = httpx.Limits(max_connections=concurrency * 2,
                          max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(headers=HEADERS, timeout=FETCH_TIMEOUT,
                                 limits=limits, follow_redirects=True,
                                 verify=False) as client:
        await fc.prime(client)

        async def one(row: dict):
            nonlocal done
            async with sem:
                try:
                    st = await enrich_one(client, pol, writer, row, fc, use_llm)
                except Exception as exc:  # noqa: BLE001
                    st = {"status": "error", "pages": 0, "facts": 0}
                    print(f"[web] {row['college_id']}: {exc}", file=sys.stderr)
                # The lock no longer exists to serialise a single SQLite
                # writer — Postgres does not need that. It is here so the
                # buffer swap in flush() cannot interleave with a mark(), and
                # so the progress line reports a consistent snapshot.
                async with lock:
                    writer.mark(row["college_id"], st["status"], st["pages"],
                                row["website_url"])
                    counters[st["status"]] = counters.get(st["status"], 0) + 1
                    counters["facts"] += st["facts"]
                    done += 1
                    if writer.due:
                        await writer.flush()
                        rate = done / max(time.perf_counter() - t0, 1e-6)
                        print(f"[web] {done}/{len(todo)} ok={counters['ok']} "
                              f"empty={counters['empty']} blocked={counters['blocked']} "
                              f"rej={counters.get('rejected',0)} "
                              f"facts={counters['facts']} | {rate:.1f}/s",
                              file=sys.stderr)

        batch = concurrency * 20
        for i in range(0, len(todo), batch):
            await asyncio.gather(*(one(r) for r in todo[i:i + batch]))
    await writer.flush()   # the tail: fewer than `every` colleges, still owed
    mins = (time.perf_counter() - t0) / 60
    print(f"[web] done in {mins:.1f}m: {counters} | firecrawl credits spent: "
          f"{fc.spent}", file=sys.stderr)
    counters["firecrawl_spent"] = fc.spent
    return counters


def requeue_stale(days: int) -> int:
    """Mark colleges whose harvest is older than `days` for re-crawling.

    College websites change every admission cycle — fees go up, scholarship
    schemes are replaced, deadlines move. A harvested fact has a shelf life,
    and one presented as current when it is two years old is worse than not
    having it, because the student acts on it. Re-running with this flag on a
    schedule is what keeps the harvest honest.

    Colleges that came back `blocked` or `no_site` are deliberately NOT
    requeued here: retrying them on every cycle spends the whole budget on the
    sites least likely to ever answer.

    One statement, where SQLite needed a fetch-parse-chunk loop: fetched_at is
    a real TIMESTAMPTZ now, so the age test is a predicate the server can
    evaluate. The INNER JOIN is what the old `if not last: continue` meant —
    a college with no harvested facts has nothing that can have gone stale.
    """
    with _pool().connection() as c:
        return c.execute(
            "UPDATE web_crawl_state SET status = 'pending' WHERE college_id IN ("
            "  SELECT s.college_id FROM web_crawl_state s"
            "  JOIN college_web_facts w USING(college_id)"
            "  WHERE s.status = 'ok'"
            "  GROUP BY s.college_id"
            "  HAVING MAX(w.fetched_at) < now() - make_interval(days => %s))",
            (int(days),)).rowcount


def status_report() -> None:
    be = get_backend()
    print("-- web_crawl_state --")
    for r in be.q("SELECT status, COUNT(*) n FROM web_crawl_state "
                  "GROUP BY status ORDER BY n DESC"):
        print(f"  {r['status']:10} {r['n']:>7,}")
    print("-- harvested facts --")
    # AVG over a REAL column comes back double precision, and Postgres has no
    # round(double precision, int) — the cast is not decoration.
    for r in be.q("SELECT kind, COUNT(*) n, ROUND(AVG(confidence)::numeric, 2) c "
                  "FROM college_web_facts GROUP BY kind ORDER BY n DESC"):
        print(f"  {r['kind']:12} {r['n']:>7,}  avg confidence {r['c']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--delay", type=float, default=1.5,
                    help="seconds between requests to the SAME domain")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--no-llm", action="store_true",
                    help="parsers only — no LLM calls, no quota, fetch-bound. "
                         "The fast first pass across the whole corpus.")
    ap.add_argument("--firecrawl-reserve", type=int, default=100,
                    help="credits to hold back for later runs")
    ap.add_argument("--firecrawl-max", type=int, default=None,
                    help="hard cap on credits this run may spend")
    ap.add_argument("--refresh-days", type=int, default=None,
                    help="requeue colleges harvested more than N days ago, then sweep "
                         "(schedule this monthly to keep fees/scholarships current)")
    a = ap.parse_args()
    if a.status:
        status_report()
        return
    if a.refresh_days:
        n = requeue_stale(a.refresh_days)
        print(f"[web] requeued {n} colleges older than {a.refresh_days} days",
              file=sys.stderr)
    asyncio.run(sweep(limit=a.limit, concurrency=a.concurrency, delay=a.delay,
                      firecrawl_reserve=a.firecrawl_reserve,
                      firecrawl_max=a.firecrawl_max,
                      use_llm=not a.no_llm))


if __name__ == "__main__":
    main()
