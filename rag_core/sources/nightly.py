r"""The nightly freshness run: one bounded pass over everything that goes stale.

WHY THIS EXISTS
The first full pass over the corpus is a project — days of crawling, harvesting
and embedding. Keeping it true is a habit. Fees change every admission cycle,
scholarship schemes are replaced, colleges appear in the catalogue, and a fact
this pipeline harvested in July and still presents as current in February is
worse than no fact at all, because a student acts on it. So every stage of the
pipeline needs re-running forever, and something has to sequence them.

WHAT ONE RUN DOES, in dependency order
  1. catalogue   list stage (~78 requests, ~12 s: catches colleges that are NEW
                 in the catalogue) + a BOUNDED slice of the detail stage.
  2. discover    turns detail payloads already on disk into website URLs in the
                 harvest queue, so a site found tonight is harvested tonight.
  3. harvest     re-crawls colleges whose facts have gone stale, then works the
                 backlog of colleges never harvested.
  4. publish     rebuild the corpus (atomically, by the ETL) and refresh the
                 dense index (incrementally).
  5. report      what moved, what failed, coverage before and after — to a log
                 file and to the nightly_run history table.

WHY A TIME BUDGET AND NOT A WORK LIST
The detail crawl cannot finish. It is paced by the client's own production API,
and the last full pass measured 2.8 attempts/s with 10,885 of 35,298 requests
(31%) coming back 429 — the server was pushing back hard. At the sustainable
~0.5 req/s a night buys ~3,000 colleges out of a 22,148-college backlog. A job
sized by work would therefore either run for eight hours or need a human to
pick a number every night.

So the argument is `--hours`, every stage is cut off when its share of that is
spent, and progress carries to tomorrow in tables and append-only files. A
nightly job still running at 9am when students arrive is a production incident;
one that stopped early having done 80% of a night's work is a Tuesday.

The publish stage gets its time RESERVED off the top rather than sharing the
budget, because it is the stage that makes the other three visible: crawling all
night and then failing to rebuild the cards leaves students reading yesterday's
corpus, which is indistinguishable from not having run at all.

WHAT THIS MODULE WILL NOT DO
- It will not write a catalogue column. It writes its own bookkeeping
  (`nightly_*`), the harvest queue, and the append-only raw JSONL the ETL reads.
  Scraped and re-crawled facts reach the served corpus only through the ETL,
  with provenance, exactly as they do today.
- It will not raise anyone's crawl rate. It pins the API throttle so AIMD can
  only back OFF from the nightly rate (see `_pin_api_rate`): additive increase
  exists to find a ceiling during a one-off backfill, and a maintenance job has
  no business hunting for one.
- It will not publish anything itself. Atomic publication belongs to the ETL,
  which does it transactionally; this module only checks afterwards that the
  publication did not destroy days of harvested facts (`_verify_publish`).

SCHEDULING — Windows (this box; there is no cron here). Run once in an elevated
PowerShell. -WorkingDirectory is the load-bearing part: `python -m` resolves
`rag_core` from the current directory, and a scheduled task's default directory
is C:\Windows\System32, where the import fails instantly.

  $py = "E:\mivi on dataset\.venv\Scripts\python.exe"
  $a = New-ScheduledTaskAction -Execute $py `
         -Argument "-m rag_core.sources.nightly --hours 6" `
         -WorkingDirectory "E:\mivi on dataset"
  $t = New-ScheduledTaskTrigger -Daily -At 1:00AM
  $s = New-ScheduledTaskSettingsSet -StartWhenAvailable `
         -ExecutionTimeLimit (New-TimeSpan -Hours 8)
  Register-ScheduledTask -TaskName "MIVI nightly" -Action $a -Trigger $t `
         -Settings $s -RunLevel Highest -Force

  -StartWhenAvailable  runs the job after a missed window (the box was asleep at
      01:00) instead of skipping the night silently.
  -ExecutionTimeLimit 8h  is belt and braces. --hours 6 is the real bound; this
      is what kills a run wedged in a syscall no deadline of ours can interrupt.
      The Task Scheduler default is 72 hours, which is not a bound at all.
  -RunLevel Highest      so it runs with nobody logged in.

  The cmd.exe one-liner equivalent — note the `cd /d`, because schtasks has no
  working-directory switch at all:

  schtasks /Create /TN "MIVI nightly" /SC DAILY /ST 01:00 /RL HIGHEST /F /TR "cmd /c cd /d \"E:\mivi on dataset\" && .venv\Scripts\python.exe -m rag_core.sources.nightly --hours 6"

  Inspect / fire it now / remove it:
      schtasks /Query  /TN "MIVI nightly" /V /FO LIST
      schtasks /Run    /TN "MIVI nightly"
      schtasks /Delete /TN "MIVI nightly" /F

The equivalent Linux cron line (cron has no working directory either, hence the
explicit cd, and no `flock` — the Postgres lock in this module already makes two
overlapping runs impossible, including across machines):

  0 1 * * *  cd /srv/mivi && .venv/bin/python -m rag_core.sources.nightly --hours 6 >> logs/nightly-cron.log 2>&1

Run manually:
    python -m rag_core.sources.nightly --dry-run          # plan + slice sizes
    python -m rag_core.sources.nightly --hours 6
    python -m rag_core.sources.nightly --hours 2 --skip harvest
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# config is imported for its side effect as much as its values: it is the only
# module that calls load_dotenv(), and MIVI_PG_DSN lives in .env. Without it the
# whole run would talk to a default DSN that may not be the served corpus.
from .. import config
from ..store.backend import DEFAULT_PG_DSN, get_backend
from . import crawl_colleges, mme_api, web_enrich

STAGES = ("catalogue", "discover", "harvest", "publish")

LOG_DIR = config.ROOT / "logs"

# ---------------------------------------------------------------------------
# Budget constants. Every number here is measured on this corpus, and every one
# of them is the reason the stage above it is sized the way it is.
# ---------------------------------------------------------------------------

DEFAULT_HOURS = 6.0

# Reserved for stage 4 off the top of the budget. The last full ETL took 30.6 s
# (build_meta.build_seconds) and the dense index is incremental — a night that
# changed 3,000 colleges re-embeds 3,000 cards at a measured 5.6 cards/s, so
# ~9 minutes. 75 minutes is ~5x that, which covers a cold rebuild after an
# encoder change and still leaves the crawl stages the bulk of the night.
DEFAULT_PUBLISH_RESERVE_MIN = 75

# The catalogue stage's share of what is left after the reserve. It is capped
# rather than given the whole night because its rate is fixed by someone else's
# server: more time buys exactly 0.5 colleges/s, while the harvest sweep runs at
# a measured 1.9 colleges/s and can clear its entire queue in the remainder.
CATALOGUE_SHARE = 0.35

# Discovery is a local file scan, not a crawl. It gets a hard cap instead of a
# share; if it needs more than ten minutes something is wrong with it.
DISCOVER_MAX_SECONDS = 600

# MEASURED, detail crawl: the completed full pass did 35,298 attempts in 208
# minutes (2.8/s) and paid for it with 10,885 429s. 0.5 req/s is the rate the
# API was never observed to refuse, so it is what a maintenance job uses.
DETAIL_REQ_PER_SEC = 0.5

# MEASURED, harvest sweep: 1.9 colleges/s with the deterministic extractors
# (fastsweep: 3,825 colleges, ok 784 / empty 2,375 / blocked 649), against
# 0.1 colleges/s once the LLM is in the loop and the 14-calls/min quota binds.
# That 19x is why `use_llm` defaults to False here — a nightly run's job is to
# keep 10,209 colleges current, not to deepen 400 of them.
HARVEST_RATE_PER_SEC = 1.9
HARVEST_LLM_RATE_PER_SEC = 0.1

# Firecrawl: ~990 credits/month with 150 held in reserve leaves ~840, which is
# 28 a night. 25 leaves headroom for a manual rescue during the day.
FIRECRAWL_RESERVE = 150
FIRECRAWL_MAX_PER_NIGHT = 25

# How stale a harvested college has to be before it is re-crawled. 10,209
# colleges have a website; at 60 days that is ~170 re-crawls a night (90
# seconds of work) and every college's facts are at most two months old. Lower
# it and we re-read third-party sites that did not change; raise it and a fee
# revision can sit unnoticed through an admission cycle.
DEFAULT_REFRESH_DAYS = 60

# A college whose detail fetch failed (429) stays at the head of the backlog and
# is retried tomorrow. A college that SUCCEEDED is not re-fetched for this long.
DEFAULT_DETAIL_REFRESH_DAYS = 30

# Ids per asyncio.gather batch in the detail stage. Small enough that the
# deadline is checked ~every 50 s at 0.5 req/s, large enough that the pacing
# schedule stays full.
DETAIL_BATCH = 25

# Harvest chunk target. web_enrich.sweep() has no deadline of its own — it works
# whatever list it is given to the end — so the only way to bound it is to hand
# it a slice at a time and stop between slices. The per-slice cost (one seed
# query, one client, one credit-balance GET) is a second or two, so slices are
# cheap and 15 minutes of work is a good size.
HARVEST_CHUNK_SECONDS = 900

# The first slice of the night is sized for one minute, not fifteen. Once sweep()
# has a list it runs to the end of it, so the first slice is the one whose
# duration cannot be predicted from anything measured tonight — buy that
# measurement cheaply. Thereafter the estimate is MONOTONE (see stage_harvest):
# it only ever tightens, so a slice can only get smaller than one that already
# fitted, and the stage's worst-case overrun is one probe.
HARVEST_PROBE_SECONDS = 60

# Below this much time left, do not start another slice — finish the night on
# the publish stage instead of on a slice that cannot land.
HARVEST_MIN_SLICE_SECONDS = 120

# If the harvest clears its queue with more than this much of the night left,
# the leftover goes back into the detail crawl rather than being handed back.
# 10,885 colleges are waiting on 429 retries; an idle night is a wasted one.
SLACK_TO_REUSE_SECONDS = 1200

# Advisory-lock key. Arbitrary but fixed: "MIVI" as ASCII.
LOCK_KEY = 0x4D495649
LOCK_NAME = "nightly"

# A holder that has not heartbeated for this long is treated as dead and its
# session is terminated. Must be comfortably longer than the gap between
# heartbeats (60 s inside every loop, including while waiting on a subprocess).
LOCK_STALE_MINUTES = 45


class NightlyError(RuntimeError):
    """Cannot start: the lock is held, or the store is unreachable."""


# ---------------------------------------------------------------------------
# Bookkeeping schema.
#
# CRITICAL: none of these tables carries a FOREIGN KEY to colleges, unlike every
# other table in the corpus. The ETL republishes by TRUNCATE ... CASCADE, which
# also truncates anything referencing what it truncates — so an FK here would
# mean the nightly job's own crawl bookkeeping is silently wiped by stage 4,
# every night, and the backlog would restart from zero every morning. The
# discovery insert therefore filters unknown college_ids in SQL (see
# DISCOVER_INSERT) instead of letting a constraint do it.
# ---------------------------------------------------------------------------

_DDL = (
    """CREATE TABLE IF NOT EXISTS nightly_run (
           run_id          TEXT PRIMARY KEY,
           host            TEXT NOT NULL,
           pid             INTEGER,
           started_at      TIMESTAMPTZ NOT NULL,
           finished_at     TIMESTAMPTZ,
           hours_budget    REAL NOT NULL,
           status          TEXT NOT NULL,
           stages          JSONB,
           coverage_before JSONB,
           coverage_after  JSONB,
           note            TEXT)""",
    "CREATE INDEX IF NOT EXISTS idx_nightly_run_started "
    "ON nightly_run(started_at DESC)",
    """CREATE TABLE IF NOT EXISTS nightly_lock (
           lock_name    TEXT PRIMARY KEY,
           run_id       TEXT NOT NULL,
           host         TEXT,
           pid          INTEGER,
           backend_pid  INTEGER,
           acquired_at  TIMESTAMPTZ NOT NULL,
           heartbeat_at TIMESTAMPTZ NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS nightly_state (
           key        TEXT PRIMARY KEY,
           value      TEXT,
           updated_at TIMESTAMPTZ NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS nightly_detail_state (
           college_id    TEXT PRIMARY KEY,
           last_fetch_at TIMESTAMPTZ NOT NULL)""",
)


def _ensure_schema() -> None:
    be = get_backend()
    for stmt in _DDL:
        be.execute(stmt)


def _state_get(key: str, default: str = "") -> str:
    v = get_backend().one("SELECT value FROM nightly_state WHERE key = ?", [key])
    return default if v is None else str(v)


def _state_set(key: str, value) -> None:
    get_backend().execute(
        "INSERT INTO nightly_state(key, value, updated_at) VALUES(?, ?, now()) "
        "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, "
        "updated_at = EXCLUDED.updated_at", [key, str(value)])


# ---------------------------------------------------------------------------
# Time budget
# ---------------------------------------------------------------------------

class Budget:
    """The night, in seconds, with stage 4's share fenced off.

    Everything downstream asks this object two questions — "how much is left"
    and "may I still start something" — so there is exactly one place that
    decides when the night is over.

    monotonic(), not wall clock: a nightly job runs straight through the hour a
    DST change or an NTP step would move, and a budget that can go backwards is
    a budget that can strand a half-finished stage.
    """

    def __init__(self, hours: float, reserve_seconds: float):
        self.total = max(0.0, hours * 3600.0)
        self.reserve = max(0.0, reserve_seconds)
        self.note = ""
        # A short manual run (--hours 0.5) would otherwise have its entire budget
        # swallowed by the publish reserve and crawl nothing at all, silently.
        # Halving is not a fix — it is a compromise that keeps such a run useful
        # and says so out loud, because the alternative is an operator watching a
        # "successful" run that did nothing.
        if self.reserve > self.total / 2:
            self.note = (f"publish reserve {_hms(self.reserve)} is more than half "
                         f"of a {_hms(self.total)} budget; cut to "
                         f"{_hms(self.total / 2)} so the crawl stages get a window. "
                         f"Raise --hours or lower --publish-reserve-minutes.")
            self.reserve = self.total / 2
        self._t0 = time.monotonic()
        self.started_wall = datetime.now()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._t0

    @property
    def left(self) -> float:
        """Seconds until the whole run must be over."""
        return max(0.0, self.total - self.elapsed)

    @property
    def crawl_left(self) -> float:
        """Seconds the crawl stages may still spend, i.e. excluding the reserve.

        Stages 1-3 read this, so an early-finishing stage donates its slack to
        the next one automatically and no static plan has to be recomputed.
        """
        return max(0.0, self.left - self.reserve)

    @property
    def crawl_spent(self) -> bool:
        return self.crawl_left <= 0

    def stop_at(self, seconds: float) -> float:
        """A monotonic instant `seconds` from now, never past the crawl window."""
        return time.monotonic() + max(0.0, min(seconds, self.crawl_left))

    @property
    def finish_wall(self) -> datetime:
        return self.started_wall + timedelta(seconds=self.total)


def _past(stop_at: float) -> bool:
    return time.monotonic() >= stop_at


def _hms(sec: float) -> str:
    sec = int(max(sec, 0))
    if sec >= 3600:
        return f"{sec // 3600}h{(sec % 3600) // 60:02d}m"
    if sec >= 60:
        return f"{sec // 60}m{sec % 60:02d}s"
    return f"{sec}s"


# ---------------------------------------------------------------------------
# Locking
# ---------------------------------------------------------------------------

class RunLock:
    """One nightly run at a time, with a stale lock that expires by itself.

    TWO MECHANISMS, because neither is sufficient alone:

    * A Postgres session-level advisory lock is the real gate. It is held on a
      connection this class opens directly rather than borrowing from the pool
      (a pooled connection can be reset or handed to someone else, and the lock
      would go with it). Its virtue is that a crashed run releases it for free —
      the session dies with the process, no timeout to tune.

    * The `nightly_lock` row is what makes staleness recoverable in the one case
      the advisory lock gets wrong: the box loses power mid-run. Postgres keeps
      the orphaned backend — and its lock — until TCP keepalives notice, which
      by default can be hours. Tomorrow's run would refuse to start for a
      process that no longer exists. So the row carries a heartbeat and the
      backend's own pid, and a holder that has stopped heartbeating for
      LOCK_STALE_MINUTES is terminated rather than waited for.

    Heartbeats are written from inside every loop, not between stages: a stage
    can legitimately run for hours, and "alive" has to mean alive now.
    """

    def __init__(self, run_id: str, ttl_minutes: int = LOCK_STALE_MINUTES):
        self.run_id = run_id
        self.ttl = ttl_minutes
        self._conn = None
        self._last_beat = 0.0

    def acquire(self) -> None:
        import psycopg

        self._conn = psycopg.connect(DEFAULT_PG_DSN, autocommit=True)
        if not self._try_lock():
            holder = self._holder()
            if holder and self._steal(holder):
                print(f"[lock] took over a stale lock from {holder}", file=sys.stderr)
            else:
                self._conn.close()
                self._conn = None
                raise NightlyError(
                    f"another nightly run holds the lock ({holder or 'unknown holder'}). "
                    f"Two runs overlapping would double-crawl third-party sites and "
                    f"race the ETL. Wait for it, or kill it and re-run.")
        backend_pid = self._conn.execute("SELECT pg_backend_pid()").fetchone()[0]
        self._conn.execute(
            "INSERT INTO nightly_lock(lock_name, run_id, host, pid, backend_pid,"
            " acquired_at, heartbeat_at) VALUES(%s,%s,%s,%s,%s,now(),now()) "
            "ON CONFLICT (lock_name) DO UPDATE SET run_id=EXCLUDED.run_id, "
            "host=EXCLUDED.host, pid=EXCLUDED.pid, backend_pid=EXCLUDED.backend_pid, "
            "acquired_at=EXCLUDED.acquired_at, heartbeat_at=EXCLUDED.heartbeat_at",
            (LOCK_NAME, self.run_id, socket.gethostname(), os.getpid(), backend_pid))
        self._last_beat = time.monotonic()

    def _try_lock(self) -> bool:
        return bool(self._conn.execute(
            "SELECT pg_try_advisory_lock(%s)", (LOCK_KEY,)).fetchone()[0])

    def _holder(self) -> str | None:
        row = self._conn.execute(
            "SELECT run_id, host, pid, backend_pid, heartbeat_at, "
            "       heartbeat_at < now() - make_interval(mins => %s) AS stale "
            "FROM nightly_lock WHERE lock_name = %s", (self.ttl, LOCK_NAME)).fetchone()
        if not row:
            return None
        self._stale = bool(row[5])
        self._holder_backend = row[3]
        return (f"run {row[0]} on {row[1]} pid {row[2]}, last heartbeat "
                f"{row[4]:%Y-%m-%d %H:%M:%S}{' - STALE' if row[5] else ''}")

    def _steal(self, holder: str) -> bool:
        """Terminate a holder that stopped heartbeating, then retake the lock."""
        if not getattr(self, "_stale", False) or not self._holder_backend:
            return False
        print(f"[lock] holder has not heartbeated in {self.ttl}m ({holder}); "
              f"terminating backend {self._holder_backend}", file=sys.stderr)
        try:
            self._conn.execute("SELECT pg_terminate_backend(%s)",
                               (self._holder_backend,))
        except Exception as exc:  # noqa: BLE001 - it may already be gone
            print(f"[lock] terminate failed ({exc}); trying the lock anyway",
                  file=sys.stderr)
        time.sleep(1.0)   # the backend's locks are released as it exits
        return self._try_lock()

    def heartbeat(self, every: float = 60.0) -> None:
        """Cheap enough to call in a tight loop; writes at most once a minute."""
        if self._conn is None or time.monotonic() - self._last_beat < every:
            return
        try:
            self._conn.execute(
                "UPDATE nightly_lock SET heartbeat_at = now() WHERE lock_name = %s",
                (LOCK_NAME,))
            self._last_beat = time.monotonic()
        except Exception as exc:  # noqa: BLE001 - a missed beat must not end the run
            print(f"[lock] heartbeat failed: {exc}", file=sys.stderr)

    def release(self) -> None:
        if self._conn is None:
            return
        try:
            self._conn.execute("DELETE FROM nightly_lock WHERE lock_name = %s "
                               "AND run_id = %s", (LOCK_NAME, self.run_id))
            self._conn.execute("SELECT pg_advisory_unlock(%s)", (LOCK_KEY,))
        except Exception:  # noqa: BLE001 - closing the session releases it anyway
            pass
        finally:
            try:
                self._conn.close()
            finally:
                self._conn = None


# ---------------------------------------------------------------------------
# Coverage: the before/after picture the report is built from
# ---------------------------------------------------------------------------

_COVERAGE_SQL = """
SELECT (SELECT COUNT(*) FROM colleges)                                       AS colleges,
       (SELECT COUNT(*) FROM colleges WHERE source_level = 'detail')         AS detail_crawled,
       (SELECT COUNT(*) FROM colleges
         WHERE COALESCE(TRIM(website_url), '') <> '')                        AS with_website,
       (SELECT COUNT(*) FROM courses)                                        AS courses,
       (SELECT COUNT(*) FROM college_agg
         WHERE COALESCE(TRIM(card), '') <> '')                               AS cards,
       (SELECT COUNT(*) FROM web_crawl_state)                                AS queue_rows,
       (SELECT COUNT(*) FROM web_crawl_state WHERE status = 'pending')       AS queue_pending,
       (SELECT COUNT(*) FROM web_crawl_state WHERE status = 'ok')            AS harvest_ok,
       (SELECT COUNT(DISTINCT college_id) FROM college_web_facts)            AS colleges_harvested,
       (SELECT COUNT(*) FROM college_web_facts)                              AS web_facts
"""


def coverage() -> dict:
    row = get_backend().q(_COVERAGE_SQL)[0]
    return {k: int(v or 0) for k, v in dict(row).items()}


def _deltas(before: dict, after: dict) -> dict:
    return {k: after[k] - before.get(k, 0) for k in after
            if after[k] - before.get(k, 0) != 0}


# ---------------------------------------------------------------------------
# Stage 1 — catalogue refresh
# ---------------------------------------------------------------------------

def _pin_api_rate(rate: float) -> None:
    """Pin the shared AIMD throttle so it can only back off, never speed up.

    mme_api.THROTTLE is process-wide by design (it is the only way pacing means
    anything across concurrent workers), and it starts at 6 req/s with additive
    increase. That is right for a one-off backfill hunting for the server's
    ceiling. It is wrong here: the ceiling was already found the hard way —
    31% of the last full pass came back 429 — and a maintenance job that
    re-discovers it every night is just a nightly denial-of-service attempt on
    the client's own students.
    """
    mme_api.THROTTLE.rate = rate
    mme_api.THROTTLE.max_rate = rate          # no additive increase
    mme_api.THROTTLE.min_rate = min(0.2, rate)


def _missing_ids() -> set[str]:
    """College ids the API has already answered 404 for. Not worth a request."""
    path = crawl_colleges.MISSING_FILE
    if not path.exists():
        return set()
    return {ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip()}


def detail_targets(slice_size: int, exclude: set[str],
                   refresh_days: int = DEFAULT_DETAIL_REFRESH_DAYS) -> dict:
    """The slice of colleges to (re-)fetch tonight: backlog first, then oldest.

    SPLIT DELIBERATELY. A run that only ever backfills finishes the first pass
    and never notices that a fee changed; one that only refreshes never finishes
    the first pass. So 80% of the slice goes to colleges that have never been
    detail-crawled (22,148 of them, and 10,885 of those failed on 429 during the
    last full pass) and 20% to the oldest successful fetches.

    "Oldest" is COALESCE(last_fetch_at, epoch): a college the initial backfill
    crawled has no row here, and treating an unknown fetch date as maximally
    stale is the honest reading — we genuinely do not know how old it is.
    Backlog still outranks it, because a college with NO data beats a college
    with old data.
    """
    be = get_backend()
    skip = exclude | _missing_ids()
    n_backlog = max(1, int(slice_size * 0.8)) if slice_size else 0
    n_refresh = max(0, slice_size - n_backlog)

    # Over-fetch, then filter in Python: `skip` is up to tens of thousands of
    # ids and shipping it into the query as a NOT IN list would be a bigger
    # statement than the result. Colleges are ordered by n_courses so the
    # requests we can afford go to the colleges students actually reach.
    over = max(200, n_backlog * 3)
    backlog = [r["college_id"] for r in be.q(
        "SELECT c.college_id FROM colleges c "
        "LEFT JOIN nightly_detail_state d ON d.college_id = c.college_id "
        "LEFT JOIN college_agg a ON a.college_id = c.college_id "
        "WHERE c.source_level <> 'detail' "
        "  AND (d.last_fetch_at IS NULL "
        "       OR d.last_fetch_at < now() - make_interval(days => ?)) "
        "ORDER BY COALESCE(a.n_courses, 0) DESC LIMIT ?", [refresh_days, over])]
    backlog = [c for c in backlog if c not in skip][:n_backlog]

    stale = []
    if n_refresh:
        over = max(100, n_refresh * 3)
        stale = [r["college_id"] for r in be.q(
            "SELECT c.college_id FROM colleges c "
            "LEFT JOIN nightly_detail_state d ON d.college_id = c.college_id "
            "WHERE c.source_level = 'detail' "
            "  AND COALESCE(d.last_fetch_at, TIMESTAMPTZ 'epoch') "
            "      < now() - make_interval(days => ?) "
            "ORDER BY COALESCE(d.last_fetch_at, TIMESTAMPTZ 'epoch') ASC LIMIT ?",
            [refresh_days, over])]
        stale = [c for c in stale if c not in skip][:n_refresh]

    return {"backlog": backlog, "refresh": stale,
            "backlog_total": be.one(
                "SELECT COUNT(*) FROM colleges WHERE source_level <> 'detail'") or 0}


async def refresh_detail(ids: list[str], stop_at: float, lock: RunLock | None,
                         concurrency: int = 4) -> dict:
    """Fetch `ids` from the MME detail endpoint until they run out or time does.

    Appends the raw payload verbatim to the same append-only JSONL the initial
    crawl writes and the ETL reads. Verbatim and append-only are both load
    bearing: the ETL applies detail rows in file order with
    COALESCE(new, old), so a re-fetched college's newer line simply wins, and a
    killed run loses at most the lines it had not flushed.
    """
    out = {"attempted": 0, "ok": 0, "missing": 0, "failed": 0,
           "unattempted": len(ids)}
    if not ids:
        return out

    # crawl_colleges._Writer, not a new file handle: one process, one writer,
    # append-only, flushing every 200 rows. Reimplementing it here would give
    # two writers to one file, which is how JSONL lines get interleaved.
    writer = crawl_colleges._Writer(crawl_colleges.DETAIL_FILE)
    miss_fh = open(crawl_colleges.MISSING_FILE, "a", encoding="utf-8")
    sem = asyncio.Semaphore(concurrency)
    fetched: list[str] = []
    lock_io = asyncio.Lock()

    try:
        async with mme_api.make_client(concurrency=concurrency) as client:

            async def one(cid: str):
                async with sem:
                    try:
                        payload = await mme_api.fetch_college_detail(client, cid)
                    except mme_api.MMEApiError as exc:
                        async with lock_io:
                            if "404" in str(exc):
                                # A definitive answer, recorded so no future
                                # night spends a request on it again.
                                out["missing"] += 1
                                miss_fh.write(cid + "\n")
                            else:
                                # Almost always a 429. NOT recorded as fetched,
                                # so this college stays at the head of tomorrow's
                                # backlog instead of dropping out of the queue.
                                out["failed"] += 1
                        return
                    except Exception as exc:  # noqa: BLE001
                        async with lock_io:
                            out["failed"] += 1
                            print(f"[nightly] detail {cid}: {exc}", file=sys.stderr)
                        return
                    async with lock_io:
                        writer.write(payload)
                        out["ok"] += 1
                        fetched.append(cid)

            for i in range(0, len(ids), DETAIL_BATCH):
                if _past(stop_at):
                    break
                batch = ids[i:i + DETAIL_BATCH]
                await asyncio.gather(*(one(c) for c in batch))
                out["attempted"] += len(batch)
                out["unattempted"] = len(ids) - out["attempted"]
                if lock:
                    lock.heartbeat()
                if fetched:
                    # Checkpoint inside the loop, in the same shape the harvest
                    # writer uses: a killed run must resume, not restart. Flush
                    # the JSONL FIRST — the state row is what keeps a college out
                    # of tomorrow's slice, so it must never become durable before
                    # the payload it claims to describe. (_Writer buffers to 200
                    # rows, and a batch is 25.)
                    writer._fh.flush()
                    _mark_detail_fetched(fetched)
                    fetched = []
                print(f"[nightly] detail {out['attempted']}/{len(ids)} "
                      f"ok={out['ok']} 404={out['missing']} failed={out['failed']} "
                      f"| {_hms(max(0.0, stop_at - time.monotonic()))} left in stage",
                      file=sys.stderr)
    finally:
        writer.close()          # flushes; same ordering rule as above
        miss_fh.close()
        if fetched:
            _mark_detail_fetched(fetched)
    return out


def _mark_detail_fetched(ids: list[str]) -> None:
    get_backend().executemany(
        "INSERT INTO nightly_detail_state(college_id, last_fetch_at) "
        "VALUES(?, now()) ON CONFLICT (college_id) DO UPDATE SET "
        "last_fetch_at = EXCLUDED.last_fetch_at", [[c] for c in ids])


async def stage_catalogue(budget: Budget, args, lock: RunLock | None,
                          attempted: set[str], window: float | None = None,
                          list_stage: bool = True) -> dict:
    """List refresh (cheap, always) + a bounded slice of the detail crawl."""
    stop_at = budget.stop_at(window if window is not None
                             else budget.crawl_left * CATALOGUE_SHARE)
    out: dict = {}

    # ~78 requests, ~12 s, and the only thing that notices a college that is
    # NEW in the catalogue. Note the limitation, which is not this module's to
    # fix: crawl_list dedupes against ids already on disk, so it captures new
    # colleges but not edits to existing list rows. Edits arrive via the detail
    # stage below.
    #
    # list_stage=False is the slack pass late in the night: the catalogue does
    # not change between 01:00 and 05:00, so re-paging it would be 78 requests to
    # the client's production API for a guaranteed zero.
    if list_stage:
        t = time.perf_counter()
        out["list_new_colleges"] = await crawl_colleges.crawl_list(concurrency=4)
        out["list_seconds"] = round(time.perf_counter() - t, 1)
        if lock:
            lock.heartbeat()

    budget_left = max(0.0, stop_at - time.monotonic())
    slice_size = int(budget_left * args.mme_rate)
    if args.detail_slice is not None:
        slice_size = min(slice_size, args.detail_slice)
    targets = detail_targets(slice_size, attempted, args.detail_refresh_days)
    ids = targets["backlog"] + targets["refresh"]
    attempted.update(ids)
    out.update({"detail_slice": len(ids),
                "detail_backlog_picked": len(targets["backlog"]),
                "detail_refresh_picked": len(targets["refresh"]),
                "detail_backlog_total": int(targets["backlog_total"])})

    if ids:
        out.update({f"detail_{k}": v for k, v in
                    (await refresh_detail(ids, stop_at, lock,
                                          args.detail_concurrency)).items()})
    return out


# ---------------------------------------------------------------------------
# Stage 2 — website discovery
# ---------------------------------------------------------------------------

# Filtered through a join on colleges rather than trusting the FK, because the
# raw JSONL legitimately contains colleges the current catalogue does not (the
# crawl targets ids from the courses CSV) and one such id would abort the whole
# batch's transaction. DO UPDATE never overwrites a URL that is already there
# and only revives a row that was parked as 'no_site': resetting a finished
# college to 'pending' would send the sweep back over a site it just read.
#
# `?` placeholders, not web_enrich's `%s`: this goes through the backend's
# rewriter, which also doubles any literal `%`. Same table, different path in —
# and in this codebase the placeholder is how you tell which.
DISCOVER_INSERT = """
INSERT INTO web_crawl_state (college_id, website_url, status, attempts)
SELECT c.college_id, ?, 'pending', 0 FROM colleges c WHERE c.college_id = ?
ON CONFLICT (college_id) DO UPDATE SET
    website_url = COALESCE(NULLIF(TRIM(web_crawl_state.website_url), ''),
                           EXCLUDED.website_url),
    status = CASE WHEN COALESCE(TRIM(web_crawl_state.website_url), '') = ''
                  THEN 'pending' ELSE web_crawl_state.status END"""


def _colleges_without_site() -> set[str]:
    return {r["college_id"] for r in get_backend().q(
        "SELECT c.college_id FROM colleges c "
        "LEFT JOIN web_crawl_state s ON s.college_id = c.college_id "
        "WHERE COALESCE(NULLIF(TRIM(c.website_url), ''), "
        "               NULLIF(TRIM(s.website_url), '')) IS NULL")}


def stage_discover(budget: Budget, args, lock: RunLock | None,
                   window: float | None = None) -> dict:
    """Turn detail payloads already on disk into harvestable website URLs.

    WHY THIS IS NOT REDUNDANT WITH THE ETL. Stage 4 will load exactly the same
    `contactInfo.websiteUrl` values into colleges.website_url, and tomorrow's
    seed_queue() would pick them up. But stage 3 runs TONIGHT, before the ETL,
    so without this a website found at 01:30 waits a full day to be harvested —
    and the whole point of the nightly loop is that one night is one complete
    cycle. It also means a broken ETL costs freshness, not discovery.

    WHY IT SCANS BYTES AND NOT ROWS. colleges_detail.jsonl is 580 MB and
    append-only. Re-parsing it nightly to find the handful of new lines would
    cost ~a minute of JSON decoding for nothing, so the byte offset of the last
    complete line is checkpointed: the first run pays for the whole file, every
    run after it pays only for what last night appended. A torn final line (the
    crawler mid-write) does not advance the offset, so it is re-read whole next
    time rather than being skipped.

    This is the zero-network form of discovery, and it is the only one this
    module does. Guessing a college's domain, or asking a search engine, is a
    separate technique with its own budget and its own failure modes; it belongs
    in its own module, and the queue rows it produces land here identically.
    """
    stop_at = budget.stop_at(min(window if window is not None else DISCOVER_MAX_SECONDS,
                                 DISCOVER_MAX_SECONDS))
    path = crawl_colleges.DETAIL_FILE
    out = {"scanned_lines": 0, "scanned_mb": 0.0, "found": 0, "queued": 0,
           "from_offset": 0, "to_offset": 0, "reset": False}
    if not path.exists():
        out["note"] = f"{path.name} absent - nothing to scan"
        return out

    size = path.stat().st_size
    offset = int(_state_get("detail_scan_offset", "0") or 0)
    if offset > size:
        # The file shrank: it was rotated or compacted. The offset means nothing
        # now, so rescan from the start rather than silently skipping the file.
        out["reset"] = True
        offset = 0
    out["from_offset"] = offset

    want = _colleges_without_site()
    found: dict[str, str] = {}
    limit = args.discover_slice
    read_to = offset

    # Binary mode: the offset must be a byte position, and text mode's decoder
    # makes tell() meaningless mid-file.
    with open(path, "rb") as fh:
        fh.seek(offset)
        for raw in fh:
            if not raw.endswith(b"\n"):
                break                      # torn tail: leave the offset behind it
            read_to += len(raw)
            out["scanned_lines"] += 1
            if out["scanned_lines"] % 2000 == 0:
                if lock:
                    lock.heartbeat()
                if _past(stop_at):
                    break
            try:
                obj = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                continue
            college = obj.get("college") if isinstance(obj.get("college"), dict) else obj
            cid = str(college.get("_id") or "").strip()
            if not cid or cid not in want:
                continue
            contact = college.get("contactInfo")
            url = web_enrich.normalise_site(
                (contact or {}).get("websiteUrl") if isinstance(contact, dict) else None)
            if not url:
                continue
            found[cid] = url               # a later line wins: it is the newer fetch
            if len(found) >= limit:
                break

    out["to_offset"] = read_to
    out["scanned_mb"] = round((read_to - offset) / 1e6, 1)
    out["found"] = len(found)
    if found:
        out["queued"] = get_backend().executemany(
            DISCOVER_INSERT, [[url, cid] for cid, url in found.items()])
    # Only after the rows are committed: a crash between the scan and the insert
    # must re-scan, never skip.
    _state_set("detail_scan_offset", read_to)
    out["colleges_without_site"] = len(want)
    return out


# ---------------------------------------------------------------------------
# Stage 3 — harvest
# ---------------------------------------------------------------------------

async def stage_harvest(budget: Budget, args, lock: RunLock | None,
                        window: float | None = None) -> dict:
    """Re-queue stale colleges, then sweep the queue in deadline-sized slices.

    The chunking is the whole trick. web_enrich.sweep() takes a `limit` and
    works that many colleges to completion — it has no concept of a deadline —
    so calling it once with a slice sized from an estimated rate would overrun
    the night whenever the estimate was optimistic. Instead the stage runs a
    one-minute probe slice, then sizes each further slice for ~15 minutes at the
    SLOWEST rate seen so far. Monotone on purpose: a rate estimate that can rise
    can ask for a slice four times bigger than the window, and the stage would
    then blow through the deadline it exists to respect. Tightening only costs a
    second of per-slice overhead.

    What is still not guaranteed: the probe itself. If tonight's hosts are 10x
    slower than the measured 1.9 colleges/s, that first minute becomes ten. That
    is the cost of not being able to interrupt sweep(), it is bounded by one
    probe, and the publish reserve absorbs it — the run as a whole still ends
    inside --hours because stage 4 sizes its children off the clock that is left
    and kills them when it runs out.
    """
    stop_at = budget.stop_at(window if window is not None else budget.crawl_left)
    out: dict = {"chunks": 0, "processed": 0, "ok": 0, "empty": 0, "blocked": 0,
                 "error": 0, "facts": 0, "firecrawl_spent": 0}

    out["requeued_stale"] = web_enrich.requeue_stale(args.refresh_days)
    out["seeded"] = web_enrich.seed_queue()
    if lock:
        lock.heartbeat()

    rate = HARVEST_LLM_RATE_PER_SEC if args.llm else HARVEST_RATE_PER_SEC
    fc_left = args.firecrawl_max
    window = HARVEST_PROBE_SECONDS         # slow start

    while not _past(stop_at):
        remaining = stop_at - time.monotonic()
        if remaining < HARVEST_MIN_SLICE_SECONDS:
            break                          # not enough left for a slice to land
        if args.harvest_slice is not None and out["processed"] >= args.harvest_slice:
            break
        # The 25-college floor first (a slice smaller than that costs more in
        # setup than it does work), then the operator's cap — in that order,
        # because a cap that the floor can overshoot is not a cap.
        chunk = max(25, int(rate * min(window, remaining)))
        if args.harvest_slice is not None:
            chunk = min(chunk, args.harvest_slice - out["processed"])

        t = time.perf_counter()
        counters = await web_enrich.sweep(
            limit=chunk, concurrency=args.harvest_concurrency,
            delay=args.harvest_delay, firecrawl_reserve=FIRECRAWL_RESERVE,
            # Divided across the night's slices, not granted per slice: the
            # month's allowance is ~840 credits and one night must not eat it.
            firecrawl_max=max(0, fc_left), use_llm=args.llm)
        dt = max(time.perf_counter() - t, 1e-6)
        processed = sum(int(counters.get(k, 0))
                        for k in ("ok", "empty", "blocked", "error"))
        out["chunks"] += 1
        for k in ("ok", "empty", "blocked", "error", "facts"):
            out[k] += int(counters.get(k, 0))
        out["processed"] += processed
        spent = int(counters.get("firecrawl_spent", 0))
        out["firecrawl_spent"] += spent
        fc_left -= spent
        if lock:
            lock.heartbeat()

        if not processed:
            out["note"] = "queue empty"    # nothing pending: the night is ahead
            break
        # Monotone down. A faster slice does not license a bigger one.
        rate = min(rate, max(processed / dt, 0.01))
        window = HARVEST_CHUNK_SECONDS
        print(f"[nightly] harvest chunk {out['chunks']}: {processed} colleges in "
              f"{_hms(dt)} ({rate:.2f}/s) | facts {out['facts']} | "
              f"{_hms(max(0.0, stop_at - time.monotonic()))} left in stage",
              file=sys.stderr)

    out["measured_rate_per_sec"] = round(rate, 2)
    return out


# ---------------------------------------------------------------------------
# Stage 4 — publish (ETL + dense index)
# ---------------------------------------------------------------------------

def _import_error(module: str) -> str | None:
    """Does `module` import in the interpreter the child will run in?

    Probed in a subprocess rather than imported here: this module must not pull
    an ETL's dependencies (or a half-edited ETL) into a process that then sleeps
    until tomorrow, and the answer that matters is the child's, not ours. One
    second, once per run, in exchange for "build_db cannot import db.connect"
    instead of "exit code 1".
    """
    r = subprocess.run([sys.executable, "-c", f"import {module}"],
                       cwd=str(config.ROOT), capture_output=True, text=True,
                       timeout=120)
    if r.returncode == 0:
        return None
    tail = [ln for ln in (r.stderr or "").strip().splitlines() if ln.strip()]
    return tail[-1][:300] if tail else f"exit {r.returncode}"


def resolve_etl(explicit: str | None) -> tuple[str, list[list[str]], str | None]:
    """Which command rebuilds the corpus. Resolved at runtime, on purpose.

    The Postgres-native ETL (rag_core.store.build_pg) is being written in
    parallel with this module, and the SQLite-era chain it replaces
    (build_db -> migrate_postgres) still calls `db.connect`, which the store
    migration removed. So there is no entry point this module can hard-code
    today that is both correct now and correct next week. It probes instead,
    prefers the Postgres-native one, and reports precisely why a candidate is
    unusable — a missing ETL has to degrade the run, not crash it, because
    stages 1-3 banked a night of crawling either way.
    """
    if explicit:
        return "explicit", [shlex.split(explicit)], None
    import importlib.util as iu

    py = sys.executable
    if iu.find_spec("rag_core.store.build_pg") is not None:
        problem = _import_error("rag_core.store.build_pg")
        return ("rag_core.store.build_pg",
                [] if problem else [[py, "-m", "rag_core.store.build_pg"]], problem)
    if iu.find_spec("rag_core.store.build_db") is not None:
        problem = (_import_error("rag_core.store.build_db")
                   or _import_error("rag_core.store.migrate_postgres"))
        return ("rag_core.store.build_db + migrate_postgres (legacy SQLite chain)",
                [] if problem else [[py, "-m", "rag_core.store.build_db"],
                                    [py, "-m", "rag_core.store.migrate_postgres"]],
                problem)
    return "none", [], "no ETL module found"


def _run_child(argv: list[str], timeout: float, log_fh, lock: RunLock | None) -> dict:
    """Run one build step as a subprocess, killed if it outlives the budget.

    Subprocess and not an in-process import, for three reasons: the index build
    pulls in sentence-transformers and a 59 MB matrix that must not stay
    resident in a process that then sleeps until tomorrow; `python -m` is the
    documented interface of both tools, so this module does not depend on their
    internals; and a step that hangs can be killed, which an imported function
    cannot.

    Killing is safe on both: the ETL publishes in one transaction (Postgres
    rolls it back) and the index writes through os.replace with checkpoints.
    """
    label = " ".join(argv[-2:])
    print(f"[nightly] $ {' '.join(argv)}  (limit {_hms(timeout)})", file=sys.stderr)
    log_fh.write(f"\n--- {datetime.now():%H:%M:%S} $ {' '.join(argv)}\n")
    log_fh.flush()
    t = time.perf_counter()
    # The child writes straight into the run's log file. An unattended job's
    # output belongs in the log, and streaming it through this process would buy
    # a console nobody is watching at 01:00.
    proc = subprocess.Popen(argv, cwd=str(config.ROOT), stdout=log_fh,
                            stderr=subprocess.STDOUT)
    # 30 s floor: a child given three seconds is a child killed during Python's
    # own interpreter startup, which tells nobody anything. This is the only
    # place the run may exceed --hours, and by at most half a minute.
    deadline = time.monotonic() + max(30.0, timeout)
    while True:
        try:
            code = proc.wait(timeout=min(60.0, max(1.0, deadline - time.monotonic())))
            break
        except subprocess.TimeoutExpired:
            if lock:
                lock.heartbeat()            # a long child is not a dead run
            if time.monotonic() >= deadline:
                proc.kill()
                proc.wait()
                dt = round(time.perf_counter() - t, 1)
                print(f"[nightly] {label} KILLED after {_hms(dt)} - out of budget",
                      file=sys.stderr)
                return {"cmd": label, "status": "killed", "seconds": dt}
    dt = round(time.perf_counter() - t, 1)
    status = "ok" if code == 0 else "failed"
    print(f"[nightly] {label} {status} in {_hms(dt)} (exit {code})", file=sys.stderr)
    return {"cmd": label, "status": status, "exit": code, "seconds": dt}


def _verify_publish(before: dict) -> dict:
    """Is the corpus servable, and did the rebuild keep the harvest?

    The second question is the one worth asking every night. The ETL
    republishes by TRUNCATE ... CASCADE and re-loads; catalogue tables rebuild
    from source in seconds, but `college_web_facts` and `web_crawl_state` are
    the accumulated output of a politeness-limited sweep over tens of thousands
    of third-party sites. Re-earning them costs days. build_db carries them over
    explicitly — and a future ETL that forgets to would destroy them silently,
    with a green build and no error anywhere. So it is checked, here, against a
    count taken immediately before the rebuild started.

    The tolerance is not slack: a legitimate rebuild DOES drop harvested rows
    whose college has left the catalogue, so a small decrease is correct and
    only a collapse is a bug.
    """
    now = coverage()
    ok = (now["colleges"] > 0 and now["courses"] > 0 and now["cards"] > 0)
    floor = before["web_facts"] - max(5, int(before["web_facts"] * 0.05))
    kept = now["web_facts"] >= floor
    out = {"corpus_servable": ok, "harvest_preserved": kept,
           "cards": now["cards"], "colleges": now["colleges"],
           "web_facts_before": before["web_facts"], "web_facts_after": now["web_facts"]}
    if not ok:
        out["error"] = ("the corpus is not servable after the rebuild "
                        f"({now['colleges']} colleges, {now['cards']} cards)")
    if not kept:
        out["error"] = (f"the rebuild LOST harvested facts: {before['web_facts']} "
                        f"-> {now['web_facts']}. The ETL is not carrying "
                        f"college_web_facts/web_crawl_state across the swap.")
    return out


def stage_publish(budget: Budget, args, lock: RunLock | None, log_fh) -> dict:
    """Rebuild the corpus, then refresh the dense index over the new cards."""
    label, commands, problem = resolve_etl(args.etl_cmd)
    # Snapshotted here, not at the start of the run: the harvest stage has been
    # adding facts for hours, and the question this answers is specifically
    # "did the REBUILD destroy anything".
    before = coverage()
    out: dict = {"etl": label, "steps": []}
    if not commands:
        out["status"] = "unavailable"
        out["error"] = (f"ETL {label} cannot run ({problem}). The crawl stages "
                        f"still banked their work in the raw JSONL and the "
                        f"harvest tables, but nothing reaches students until a "
                        f"rebuild succeeds.")
        return out

    # The index needs the tail of the window; the ETL gets the rest. The last
    # full ETL took 30.6 s, so this is generous by design — a slow rebuild is
    # not a reason to kill it, running past sunrise is. The ETL wins when the
    # night has overrun and there is little left: published data that is not yet
    # densely indexed still answers questions, an unpublished corpus answers
    # nothing, and the index is incremental so tomorrow catches it up.
    index_reserve = min(max(600.0, budget.left * 0.35), budget.left * 0.5)
    for argv in commands:
        etl_budget = max(60.0, budget.left - index_reserve)
        step = _run_child(argv, etl_budget, log_fh, lock)
        out["steps"].append(step)
        if step["status"] != "ok":
            break

    out["verify"] = _verify_publish(before)
    if not out["verify"]["corpus_servable"]:
        # Embedding against a broken corpus would replace a good index with a
        # bad one, and the index is what retrieval actually reads.
        out["status"] = "failed"
        out["error"] = out["verify"].get("error")
        return out

    if not args.skip_index:
        out["index"] = _run_child([sys.executable, "-m", "rag_core.index_build"],
                                  max(120.0, budget.left), log_fh, lock)
        if out["index"]["status"] != "ok":
            # Worth spelling out, because the failure is invisible from outside:
            # the corpus IS published, so lexical search finds tonight's changes
            # and the API stays up. Only the dense half is stale, and a card the
            # vectors do not know about is one a student's paraphrase cannot
            # reach — which looks like a retrieval quality problem, not an
            # outage, and would otherwise go unexplained for days.
            out["index_note"] = ("corpus published but the dense index was NOT "
                                 "refreshed; colleges whose cards changed tonight "
                                 "are reachable lexically, not semantically")
    out["status"] = "ok" if all(
        s["status"] == "ok" for s in out["steps"]) and (
        args.skip_index or out.get("index", {}).get("status") == "ok") else "degraded"
    if not out["verify"]["harvest_preserved"]:
        out["status"] = "failed"
        out["error"] = out["verify"]["error"]
    return out


# ---------------------------------------------------------------------------
# Plan (--dry-run) — the same arithmetic the real run uses
# ---------------------------------------------------------------------------

def build_plan(args) -> dict:
    reserve = min(args.publish_reserve_minutes * 60, args.hours * 3600)
    total = args.hours * 3600
    crawl = max(0.0, total - reserve)
    cat = crawl * CATALOGUE_SHARE
    disc = max(0.0, min(DISCOVER_MAX_SECONDS, crawl - cat))
    harv = max(0.0, crawl - cat - disc)

    cov = coverage()
    detail_slice = int(cat * args.mme_rate)
    if args.detail_slice is not None:
        detail_slice = min(detail_slice, args.detail_slice)
    rate = HARVEST_LLM_RATE_PER_SEC if args.llm else HARVEST_RATE_PER_SEC
    harvest_slice = int(harv * rate)
    if args.harvest_slice is not None:
        harvest_slice = min(harvest_slice, args.harvest_slice)

    targets = detail_targets(detail_slice, set(), args.detail_refresh_days)
    backlog_total = int(targets["backlog_total"])
    nights = (backlog_total / detail_slice) if detail_slice else 0.0

    etl_label, etl_cmds, etl_problem = resolve_etl(args.etl_cmd)
    return {
        "coverage": cov, "windows": {"total": total, "reserve": reserve,
                                     "catalogue": cat, "discover": disc,
                                     "harvest": harv},
        "detail_slice": detail_slice,
        "detail_backlog_picked": len(targets["backlog"]),
        "detail_refresh_picked": len(targets["refresh"]),
        "detail_backlog_total": backlog_total,
        "nights_to_clear_backlog": nights,
        "harvest_slice": harvest_slice, "harvest_rate": rate,
        "queue_pending": cov["queue_pending"],
        # seed_queue() runs at the top of stage 3, so the pending count the plan
        # shows is not the work available — this is the rest of it.
        "queue_to_seed": max(0, cov["with_website"] - cov["queue_rows"]),
        "colleges_without_site": cov["colleges"] - cov["with_website"],
        "etl": etl_label, "etl_commands": [" ".join(c) for c in etl_cmds],
        "etl_problem": etl_problem,
    }


def print_plan(args, plan: dict) -> None:
    w, cov = plan["windows"], plan["coverage"]
    enabled = [s for s in STAGES if _enabled(args, s)]
    p = lambda s="": print(s, file=sys.stdout)  # noqa: E731

    # ASCII only below: this prints to a cp1252 Windows console, where an em dash
    # comes out as a replacement character in the middle of an operator's plan.
    p("=" * 78)
    p("MIVI nightly - DRY RUN (nothing is crawled, written or published)")
    p("=" * 78)
    p(f"  budget          {args.hours:g} h  ({_hms(w['total'])})")
    p(f"  publish reserve {_hms(w['reserve'])} held back off the top")
    p(f"  crawl window    {_hms(w['total'] - w['reserve'])}")
    p(f"  stages          {' -> '.join(enabled)}"
      + (f"   (skipped: {', '.join(s for s in STAGES if s not in enabled)})"
         if len(enabled) < len(STAGES) else ""))
    p(f"  log file        {_log_path()}")
    p(f"  store           {DEFAULT_PG_DSN.rsplit('@', 1)[-1]}")
    p()
    p("COVERAGE NOW")
    p(f"  colleges                  {cov['colleges']:>9,}")
    p(f"  detail-crawled            {cov['detail_crawled']:>9,}  "
      f"({100 * cov['detail_crawled'] / max(cov['colleges'], 1):.0f}%)")
    p(f"  with a website URL        {cov['with_website']:>9,}  "
      f"({100 * cov['with_website'] / max(cov['colleges'], 1):.0f}%)")
    p(f"  courses                   {cov['courses']:>9,}")
    p(f"  cards rendered            {cov['cards']:>9,}")
    p(f"  harvest queue pending     {cov['queue_pending']:>9,}")
    p(f"  colleges harvested ok     {cov['harvest_ok']:>9,}")
    p(f"  web facts stored          {cov['web_facts']:>9,}  "
      f"across {cov['colleges_harvested']:,} colleges")
    p()
    p("PLAN")
    p(f"  1 catalogue   window {_hms(w['catalogue']):>7}  "
      f"list stage (~78 req, ~12 s) + detail slice")
    p(f"                  detail slice {plan['detail_slice']:,} colleges "
      f"at {args.mme_rate:g} req/s")
    p(f"                    backlog  {plan['detail_backlog_picked']:>6,} of "
      f"{plan['detail_backlog_total']:,} never detail-crawled")
    p(f"                    refresh  {plan['detail_refresh_picked']:>6,} oldest "
      f"successful fetches (> {args.detail_refresh_days}d)")
    p(f"                  -> ~{plan['nights_to_clear_backlog']:.1f} nights to clear "
      f"the backlog at this slice size")
    p(f"  2 discover    window {_hms(w['discover']):>7}  "
      f"scan new bytes of {crawl_colleges.DETAIL_FILE.name} for website URLs")
    p(f"                  cap {args.discover_slice:,} URLs; "
      f"{plan['colleges_without_site']:,} colleges have none")
    p(f"                  scan resumes at byte "
      f"{int(_state_get('detail_scan_offset', '0') or 0):,} of "
      f"{_file_size(crawl_colleges.DETAIL_FILE):,}")
    p(f"  3 harvest     window {_hms(w['harvest']):>7}  "
      f"requeue > {args.refresh_days}d, then sweep in "
      f"{_hms(HARVEST_CHUNK_SECONDS)} slices")
    p(f"                  ~{plan['harvest_slice']:,} colleges at "
      f"{plan['harvest_rate']:g}/s "
      f"({'LLM extraction' if args.llm else 'deterministic extractors only'})")
    p(f"                  queue has {plan['queue_pending']:,} pending "
      f"+ {plan['queue_to_seed']:,} colleges with a URL not yet queued; "
      f"firecrawl cap {args.firecrawl_max}")
    p(f"  4 publish     window {_hms(w['reserve']):>7}  ETL: {plan['etl']}")
    for c in plan["etl_commands"]:
        p(f"                  $ {c}")
    if plan["etl_problem"]:
        p(f"                  !! UNAVAILABLE: {plan['etl_problem']}")
        p(f"                  !! stages 1-3 would still bank their work; nothing")
        p(f"                     reaches students until a rebuild succeeds.")
    if not args.skip_index:
        p(f"                  $ {sys.executable} -m rag_core.index_build")
    p(f"                  then verify: corpus servable + harvest preserved")
    p()
    if plan["harvest_slice"] > plan["queue_pending"]:
        p(f"  NOTE the harvest can clear its whole queue "
          f"({plan['queue_pending']:,}) inside its window. Any slack over "
          f"{_hms(SLACK_TO_REUSE_SECONDS)} goes back into the detail crawl.")
    p(f"  NOTE this run assumes it owns the pipelines. A backfill sweep still "
      f"running\n       would double-crawl the same hosts - check first.")
    p()
    cmdline = f"-m rag_core.sources.nightly --hours {args.hours:g}"
    p("SCHEDULING  (elevated PowerShell, once. -WorkingDirectory is what makes")
    p("             `python -m` resolve the package; a task's default directory")
    p("             is C:\\Windows\\System32.)")
    p(f'    $a = New-ScheduledTaskAction -Execute "{sys.executable}" `')
    p(f'           -Argument "{cmdline}" -WorkingDirectory "{config.ROOT}"')
    p('    $t = New-ScheduledTaskTrigger -Daily -At 1:00AM')
    p('    $s = New-ScheduledTaskSettingsSet -StartWhenAvailable `')
    p('           -ExecutionTimeLimit (New-TimeSpan -Hours 8)')
    p('    Register-ScheduledTask -TaskName "MIVI nightly" -Action $a '
      '-Trigger $t `')
    p('           -Settings $s -RunLevel Highest -Force')
    p("  cmd.exe equivalent (schtasks has no working-directory switch):")
    p(f'    schtasks /Create /TN "MIVI nightly" /SC DAILY /ST 01:00 /RL HIGHEST '
      f'/F /TR "cmd /c cd /d \\"{config.ROOT}\\" && '
      f'.venv\\Scripts\\python.exe {cmdline}"')
    p("  Linux cron (no flock needed - the Postgres lock already excludes a "
      "second run):")
    # Printing the Windows root inside a cron line would be nonsense, so a
    # deploy path stands in when this is not being read on a POSIX box.
    root = config.ROOT if config.ROOT.as_posix().startswith("/") else Path("/srv/mivi")
    p(f"    0 1 * * *  cd {root.as_posix()} && .venv/bin/python {cmdline} "
      f">> logs/nightly-cron.log 2>&1")
    p("=" * 78)


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


# ---------------------------------------------------------------------------
# Logging: one file per run, tee'd to stderr
# ---------------------------------------------------------------------------

def _log_path(when: datetime | None = None) -> Path:
    when = when or datetime.now()
    return LOG_DIR / f"nightly-{when:%Y-%m-%d}.log"


class _Tee:
    """Writes to the log file AND the console.

    Every stage this module drives already logs well to stderr — the crawler's
    throttle messages, the sweep's per-batch counters, the index's ETA. Routing
    that through `logging` would mean editing four modules this workflow does
    not own, so stderr is duplicated instead: `print(file=sys.stderr)` looks the
    name up at call time, so replacing it captures all of them.
    """

    def __init__(self, *streams):
        self._streams = [s for s in streams if s is not None]

    def write(self, data: str) -> int:
        for s in self._streams:
            try:
                s.write(data)
            except Exception:  # noqa: BLE001 - a closed console must not end the run
                pass
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            try:
                s.flush()
            except Exception:  # noqa: BLE001
                pass

    def isatty(self) -> bool:
        # False even when the console is one: it makes the progress lines in
        # index_build and the crawler stop rewriting with \r, which is
        # unreadable in a log file.
        return False


# ---------------------------------------------------------------------------
# Run history
# ---------------------------------------------------------------------------

def _open_run(run_id: str, hours: float) -> None:
    get_backend().execute(
        "INSERT INTO nightly_run(run_id, host, pid, started_at, hours_budget, "
        "status) VALUES(?, ?, ?, now(), ?, 'running') "
        "ON CONFLICT (run_id) DO UPDATE SET started_at = now(), status = 'running'",
        [run_id, socket.gethostname(), os.getpid(), hours])


def _close_run(run_id: str, status: str, stages: dict, before: dict,
               after: dict, note: str) -> None:
    get_backend().execute(
        "UPDATE nightly_run SET finished_at = now(), status = ?, "
        "stages = CAST(? AS JSONB), coverage_before = CAST(? AS JSONB), "
        "coverage_after = CAST(? AS JSONB), note = ? WHERE run_id = ?",
        [status, json.dumps(stages, default=str), json.dumps(before),
         json.dumps(after), note[:2000], run_id])


def print_report(run_id: str, budget: Budget, status: str, stages: dict,
                 before: dict, after: dict) -> None:
    d = _deltas(before, after)
    print("=" * 78, file=sys.stderr)
    print(f"MIVI nightly {run_id} - {status.upper()} in {_hms(budget.elapsed)} "
          f"of {_hms(budget.total)}", file=sys.stderr)
    print("=" * 78, file=sys.stderr)
    for name in STAGES:
        st = stages.get(name)
        if not st:
            continue
        head = f"  {name:10} {st.get('status', '?'):10} {_hms(st.get('seconds', 0)):>7}"
        print(head, file=sys.stderr)
        for k, v in st.items():
            if k in ("status", "seconds"):
                continue
            print(f"      {k}: {v}", file=sys.stderr)
    print("  coverage", file=sys.stderr)
    for k in after:
        arrow = f"  ({d[k]:+,})" if k in d else ""
        print(f"      {k:24} {after[k]:>9,}{arrow}", file=sys.stderr)
    if not d:
        print("      (nothing moved)", file=sys.stderr)
    print("=" * 78, file=sys.stderr)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _enabled(args, stage: str) -> bool:
    if args.only:
        return stage in args.only
    return stage not in (args.skip or ())


def _skip(stages: dict, name: str, args) -> None:
    """Record a stage that did not run, and why.

    An absent stage and a stage that found nothing to do look identical in a
    report, and they call for opposite reactions from whoever reads it at 8am.
    """
    reason = ("asked not to (--only/--skip)" if not _enabled(args, name)
              else "out of crawl budget before it started")
    stages[name] = {"status": "skipped", "reason": reason, "seconds": 0}
    print(f"[nightly] {name} skipped - {reason}", file=sys.stderr)


async def _timed(name: str, coro_or_call, stages: dict) -> dict:
    """Run one stage fail-soft and record it.

    Per-stage isolation is the point: discovery breaking on a torn JSONL line
    must not cost the rebuild, and a rebuild failing must not lose the report of
    the four hours of crawling that preceded it. Only Exception is caught —
    KeyboardInterrupt and SystemExit still stop the run, because a human asking
    it to stop is not a stage failure.
    """
    t = time.perf_counter()
    try:
        out = await coro_or_call if asyncio.iscoroutine(coro_or_call) else coro_or_call()
        out = dict(out or {})
        out.setdefault("status", "ok")
    except Exception as exc:  # noqa: BLE001
        import traceback
        out = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        print(f"[nightly] stage {name} FAILED: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
    out["seconds"] = round(time.perf_counter() - t, 1)
    stages[name] = out
    return out


async def run(args) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_fh = open(_log_path(), "a", encoding="utf-8", buffering=1)
    real_stderr = sys.stderr
    sys.stderr = _Tee(log_fh, real_stderr)

    # Seconds are in the id because it is the primary key of the history table:
    # two manual runs in the same minute would otherwise overwrite each other's
    # record, and the second one's report would replace the first one's evidence.
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{socket.gethostname()[:24]}"
    budget = Budget(args.hours, args.publish_reserve_minutes * 60)
    stages: dict = {}
    status = "ok"
    note = ""
    lock = RunLock(run_id, args.lock_ttl_minutes)

    print(f"\n{'#' * 78}\n[nightly] {run_id} start "
          f"{budget.started_wall:%Y-%m-%d %H:%M:%S} local | budget "
          f"{args.hours:g}h (until {budget.finish_wall:%H:%M:%S}) | "
          f"reserve {_hms(budget.reserve)} for publish", file=sys.stderr)
    if budget.note:
        print(f"[nightly] WARNING: {budget.note}", file=sys.stderr)

    try:
        _ensure_schema()
        lock.acquire()
        _open_run(run_id, args.hours)
        before = coverage()
        _pin_api_rate(args.mme_rate)
        attempted: set[str] = set()

        if _enabled(args, "catalogue"):
            await _timed("catalogue",
                         stage_catalogue(budget, args, lock, attempted), stages)
        else:
            _skip(stages, "catalogue", args)
        if not _enabled(args, "discover") or budget.crawl_spent:
            _skip(stages, "discover", args)
        else:
            await _timed("discover",
                         lambda: stage_discover(budget, args, lock), stages)
        if not _enabled(args, "harvest") or budget.crawl_spent:
            _skip(stages, "harvest", args)
        else:
            await _timed("harvest", stage_harvest(budget, args, lock), stages)

        # The harvest sweep runs at 1.9 colleges/s and its queue is 10,209
        # colleges deep, so on most nights it finishes early — and if it was
        # skipped entirely, the whole window is free. The detail backlog is
        # 22,148 deep and can never finish. Handing the slack back would mean the
        # box idles while colleges wait on a 429 retry.
        slack = budget.crawl_left
        if slack > SLACK_TO_REUSE_SECONDS and _enabled(args, "catalogue"):
            print(f"[nightly] {_hms(slack)} of crawl budget unspent - "
                  f"a second detail slice", file=sys.stderr)
            await _timed("catalogue-extra",
                         stage_catalogue(budget, args, lock, attempted,
                                         window=slack, list_stage=False), stages)

        if _enabled(args, "publish"):
            await _timed("publish",
                         lambda: stage_publish(budget, args, lock, log_fh), stages)
        else:
            _skip(stages, "publish", args)

        after = coverage()
        failed = [n for n, s in stages.items()
                  if s.get("status") not in ("ok", "skipped", None)]
        status = "ok" if not failed else "degraded"
        note = ("; ".join(f"{n}: {stages[n].get('error', stages[n].get('status'))}"
                          for n in failed))[:2000]
        print_report(run_id, budget, status, stages, before, after)
        _close_run(run_id, status, stages, before, after, note)
        return 0 if status == "ok" else 1

    except NightlyError as exc:
        print(f"[nightly] cannot start: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        # Everything is checkpointed — the raw JSONL is flushed, the harvest
        # commits per 25 colleges, the detail state per batch — so an interrupted
        # run resumes rather than repeating. Say so in the history table.
        print("[nightly] interrupted - checkpointed, tomorrow resumes here",
              file=sys.stderr)
        try:
            _close_run(run_id, "interrupted", stages, {}, coverage(), "SIGINT")
        except Exception:  # noqa: BLE001
            pass
        return 130
    finally:
        lock.release()
        sys.stderr = real_stderr
        log_fh.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m rag_core.sources.nightly",
        description="Nightly freshness run: catalogue, websites, harvest, "
                    "rebuild, report - inside a time budget.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--hours", type=float, default=DEFAULT_HOURS,
                   help="total budget. Stages stop cleanly when it is spent")
    p.add_argument("--publish-reserve-minutes", type=int,
                   default=DEFAULT_PUBLISH_RESERVE_MIN,
                   help="held back off the top for the ETL + index rebuild")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and slice sizes; touch nothing")
    p.add_argument("--only", nargs="*", choices=STAGES, default=None,
                   help="run only these stages")
    p.add_argument("--skip", nargs="*", choices=STAGES, default=(),
                   help="skip these stages")

    g = p.add_argument_group("catalogue")
    g.add_argument("--mme-rate", type=float, default=DETAIL_REQ_PER_SEC,
                   help="detail requests/second against the MME API. The AIMD "
                        "throttle may go below this, never above")
    g.add_argument("--detail-slice", type=int, default=None,
                   help="hard cap on colleges fetched (default: whatever the "
                        "window affords at --mme-rate)")
    g.add_argument("--detail-concurrency", type=int, default=4)
    g.add_argument("--detail-refresh-days", type=int,
                   default=DEFAULT_DETAIL_REFRESH_DAYS,
                   help="how old a successful detail fetch must be to be redone")

    g = p.add_argument_group("discover")
    g.add_argument("--discover-slice", type=int, default=5000,
                   help="cap on website URLs queued in one run")

    g = p.add_argument_group("harvest")
    g.add_argument("--refresh-days", type=int, default=DEFAULT_REFRESH_DAYS,
                   help="requeue colleges harvested longer ago than this")
    g.add_argument("--harvest-slice", type=int, default=None,
                   help="hard cap on colleges swept")
    g.add_argument("--harvest-concurrency", type=int, default=12,
                   help="workers ACROSS domains (never within one)")
    g.add_argument("--harvest-delay", type=float, default=1.5,
                   help="seconds between requests to the same domain")
    g.add_argument("--llm", action="store_true",
                   help="allow LLM extraction. Measured 0.1 colleges/s against "
                        "1.9 with the deterministic parsers, so this trades 19x "
                        "coverage for depth")
    g.add_argument("--firecrawl-max", type=int, default=FIRECRAWL_MAX_PER_NIGHT,
                   help="credits this run may spend (~840/month spendable)")

    g = p.add_argument_group("publish")
    g.add_argument("--etl-cmd", default=None,
                   help="override the corpus rebuild command")
    g.add_argument("--skip-index", action="store_true",
                   help="rebuild the corpus but not the dense index")

    p.add_argument("--lock-ttl-minutes", type=int, default=LOCK_STALE_MINUTES,
                   help="a holder silent this long is treated as dead")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        try:
            _ensure_schema()
            print_plan(args, build_plan(args))
        except Exception as exc:  # noqa: BLE001
            print(f"[nightly] cannot plan: {exc}", file=sys.stderr)
            return 2
        return 0
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
