"""Re-check facts that were harvested BEFORE the verifier existed.

The first ~2,000 facts were written by extractors that graded their own work.
That is not good enough: an extractor's confidence is its opinion of itself,
and the failure this catches — a number that never appeared on the page — is
exactly the failure a self-grader cannot see.

This re-fetches each fact's recorded source_url, re-runs the original page text
through rag_core/sources/verify_facts, and files the result:

    CONFIRMED  -> keeps its place in the corpus, now with a source snippet
    UNCERTAIN  -> moved to quarantine: retained, never shown to a student
    REJECTED   -> removed from the served corpus, kept in quarantine with the
                  reason, because deleting the evidence would make the same
                  mistake unauditable

Nothing is hard-deleted. A rejected fact is more useful in quarantine than
gone: it shows which extractor produced it, and that is how the extractor gets
fixed rather than re-run.

Run:
    python -m rag_core.sources.reverify --limit 100    # sample first
    python -m rag_core.sources.reverify               # everything unverified
    python -m rag_core.sources.reverify --report      # what happened
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone

import httpx

from .. import config  # noqa: F401  - loads .env before the backend reads it
from ..store.backend import get_backend
from .verify_facts import cross_check, verify
from .web_enrich import HEADERS, FETCH_TIMEOUT, Politeness, parse_html

DDL = """
CREATE TABLE IF NOT EXISTS fact_verification (
    college_id   TEXT NOT NULL,
    kind         TEXT NOT NULL,
    verdict      TEXT NOT NULL,          -- CONFIRMED | UNCERTAIN | REJECTED
    adj_confidence REAL,
    reasons      TEXT,
    anchor       TEXT,                   -- the page text the numbers came from
    source_url   TEXT,
    checked_at   TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (college_id, kind)
);
CREATE INDEX IF NOT EXISTS idx_factverif_verdict ON fact_verification(verdict);

-- Quarantine: facts that exist but must never reach a student. Kept rather
-- than deleted so a bad extractor can be identified from its output.
CREATE TABLE IF NOT EXISTS college_web_facts_quarantine (
    college_id   TEXT NOT NULL,
    kind         TEXT NOT NULL,
    summary      TEXT,
    payload      TEXT,
    source_url   TEXT,
    fetched_at   TIMESTAMPTZ,
    extractor    TEXT,
    confidence   REAL,
    verdict      TEXT NOT NULL,
    reasons      TEXT,
    quarantined_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (college_id, kind)
);
"""


def ensure_tables() -> None:
    be = get_backend()
    for stmt in [s for s in DDL.split(";") if s.strip()]:
        be.execute(stmt)


def _now():
    return datetime.now(timezone.utc)


def pending(limit: int | None = None) -> list[dict]:
    """Facts with no verification record yet, newest first — the most recently
    harvested are the ones most likely still being produced by a live sweep."""
    be = get_backend()
    sql = """
        SELECT w.college_id, w.kind, w.summary, w.payload, w.source_url,
               w.fetched_at, w.extractor, w.confidence,
               c.name AS college_name, c.city
        FROM college_web_facts w
        JOIN colleges c ON c.college_id = w.college_id
        LEFT JOIN fact_verification v
               ON v.college_id = w.college_id AND v.kind = w.kind
        WHERE v.college_id IS NULL AND w.source_url IS NOT NULL
        ORDER BY w.fetched_at DESC
    """
    if limit:
        sql += " LIMIT ?"
        return be.q(sql, [limit])
    return be.q(sql)


async def _fetch(client, pol, url: str) -> str | None:
    if not await pol.allowed(client, url):
        return None
    await pol.wait(url)
    try:
        r = await client.get(url)
    except Exception:  # noqa: BLE001
        return None
    if r.status_code != 200 or "html" not in r.headers.get("content-type", "").lower():
        return None
    text, _ = parse_html(r.text)
    return text


def _record(be, row: dict, res, page_ok: bool) -> None:
    reasons = "; ".join(res.reasons)[:900]
    anchor = ""
    if res.anchors:
        anchor = " | ".join(list(res.anchors.values())[:3])[:900]

    be.execute(
        "INSERT INTO fact_verification"
        "(college_id, kind, verdict, adj_confidence, reasons, anchor, source_url, checked_at) "
        "VALUES(?,?,?,?,?,?,?,?) "
        "ON CONFLICT (college_id, kind) DO UPDATE SET "
        "verdict=EXCLUDED.verdict, adj_confidence=EXCLUDED.adj_confidence, "
        "reasons=EXCLUDED.reasons, anchor=EXCLUDED.anchor, checked_at=EXCLUDED.checked_at",
        [row["college_id"], row["kind"], res.verdict, res.confidence,
         reasons, anchor, row["source_url"], _now()])

    if res.verdict == "CONFIRMED":
        return

    # Quarantine first, THEN remove from the served table — in that order, so a
    # crash between the two loses nothing rather than losing the fact entirely.
    # json.dumps, NOT str(): Postgres hands back JSONB as a dict, and str() on a
    # dict produces Python repr with single quotes — which is not JSON and made
    # restoring from quarantine fail. Quarantine only has value if what it holds
    # can be put back.
    payload = row.get("payload")
    if not isinstance(payload, str):
        payload = json.dumps(payload or {}, ensure_ascii=False)

    be.execute(
        "INSERT INTO college_web_facts_quarantine"
        "(college_id, kind, summary, payload, source_url, fetched_at, extractor,"
        " confidence, verdict, reasons, quarantined_at) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT (college_id, kind) DO UPDATE SET "
        "verdict=EXCLUDED.verdict, reasons=EXCLUDED.reasons, "
        "quarantined_at=EXCLUDED.quarantined_at",
        [row["college_id"], row["kind"], row.get("summary"),
         payload, row.get("source_url"), row.get("fetched_at"),
         row.get("extractor"), row.get("confidence"), res.verdict, reasons, _now()])

    if res.verdict == "REJECTED":
        be.execute("DELETE FROM college_web_facts WHERE college_id=? AND kind=?",
                   [row["college_id"], row["kind"]])


async def run(limit: int | None = None, concurrency: int = 8,
              delay: float = 1.5) -> dict:
    ensure_tables()
    be = get_backend()
    rows = pending(limit)
    if not rows:
        print("[reverify] nothing pending", file=sys.stderr)
        return {}

    print(f"[reverify] {len(rows)} facts to re-check "
          f"(concurrency {concurrency}, {delay}s/domain)", file=sys.stderr)

    pol = Politeness(delay=delay)
    sem = asyncio.Semaphore(concurrency)
    counts = {"CONFIRMED": 0, "UNCERTAIN": 0, "REJECTED": 0, "unfetchable": 0}
    lock = asyncio.Lock()
    t0 = time.perf_counter()
    # Pages are shared between kinds on the same site; caching means a college
    # with four facts on one page costs one fetch, not four.
    page_cache: dict[str, str | None] = {}

    async with httpx.AsyncClient(headers=HEADERS, timeout=FETCH_TIMEOUT,
                                 follow_redirects=True, verify=False) as client:
        async def one(row: dict, i: int):
            async with sem:
                url = row["source_url"]
                if url in page_cache:
                    text = page_cache[url]
                else:
                    text = await _fetch(client, pol, url)
                    page_cache[url] = text

                fact = {"summary": row.get("summary"),
                        "facts": {}, "confidence": row.get("confidence") or 0.0}
                try:
                    p = row.get("payload")
                    fact["facts"] = json.loads(p) if isinstance(p, str) else (p or {})
                except Exception:  # noqa: BLE001
                    fact["facts"] = {}

                if text is None:
                    # The page is gone or blocks us NOW. That is not evidence the
                    # fact was wrong when harvested, so it is quarantined as
                    # UNVERIFIABLE rather than rejected — deleting a possibly
                    # correct fact because a server had a bad day is worse.
                    res = type("R", (), {"verdict": "UNCERTAIN", "confidence": 0.0,
                                         "reasons": ["source page no longer fetchable"],
                                         "anchors": {}})()
                    async with lock:
                        counts["unfetchable"] += 1
                else:
                    res = verify(fact, text, row.get("college_name") or "",
                                 row.get("city"))

                async with lock:
                    counts[res.verdict] = counts.get(res.verdict, 0) + 1
                    _record(be, row, res, text is not None)
                    n = sum(counts[k] for k in ("CONFIRMED", "UNCERTAIN", "REJECTED"))
                    if n % 50 == 0:
                        rate = n / max(time.perf_counter() - t0, 1e-6)
                        print(f"[reverify] {n}/{len(rows)} "
                              f"ok={counts['CONFIRMED']} "
                              f"uncertain={counts['UNCERTAIN']} "
                              f"rejected={counts['REJECTED']} | {rate:.1f}/s",
                              file=sys.stderr)

        batch = concurrency * 20
        for i in range(0, len(rows), batch):
            chunk = rows[i:i + batch]
            await asyncio.gather(*(one(r, i + j) for j, r in enumerate(chunk)))

    mins = (time.perf_counter() - t0) / 60
    print(f"[reverify] done in {mins:.1f}m: {counts}", file=sys.stderr)
    return counts


def report() -> None:
    be = get_backend()
    ensure_tables()
    print("-- verification verdicts --")
    for r in be.q("SELECT verdict, COUNT(*) n, ROUND(AVG(adj_confidence)::numeric,2) c "
                  "FROM fact_verification GROUP BY verdict ORDER BY n DESC"):
        print(f"   {r['verdict']:11} {r['n']:>6,}  avg adj conf {r['c']}")
    total = be.one("SELECT COUNT(*) FROM fact_verification") or 0
    conf = be.one("SELECT COUNT(*) FROM fact_verification WHERE verdict='CONFIRMED'") or 0
    if total:
        print(f"\n   confirmed rate: {100 * conf / total:.1f}% of {total:,} checked")
    print("\n-- top rejection reasons --")
    for r in be.q("SELECT reasons, COUNT(*) n FROM fact_verification "
                  "WHERE verdict='REJECTED' GROUP BY reasons ORDER BY n DESC LIMIT 6"):
        print(f"   {r['n']:>5}  {str(r['reasons'])[:96]}")
    print("\n-- quarantine by extractor (which one to fix) --")
    for r in be.q("SELECT extractor, verdict, COUNT(*) n "
                  "FROM college_web_facts_quarantine "
                  "GROUP BY extractor, verdict ORDER BY n DESC LIMIT 10"):
        print(f"   {str(r['extractor']):12} {r['verdict']:10} {r['n']:>5}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--delay", type=float, default=1.5)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
        return
    asyncio.run(run(a.limit, a.concurrency, a.delay))


if __name__ == "__main__":
    main()
