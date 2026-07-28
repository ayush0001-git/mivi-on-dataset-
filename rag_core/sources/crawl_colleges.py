"""Resumable crawler that resolves every college ObjectId into real facts.

Two stages, run independently:

  stage "list"    ~78 requests  -> data/raw/colleges_list.jsonl
      Pages the bulk endpoint at 500/page and captures the whole catalogue.
      This is what makes the CSV joinable: it maps `college` ObjectId ->
      collegeName, city, state, type, establishYear, address, details.

  stage "detail"  ~35k requests -> data/raw/colleges_detail.jsonl
      Per-college enrichment (NAAC, NIRF, district, contact, placements,
      accommodation, scholarships, facilities, FAQ, admissions). Restricted by
      default to the ObjectIds the courses CSV actually references, so we never
      spend requests on colleges no student query can reach.

Both stages are append-only JSONL and resumable: the ids already on disk are
skipped, so an interrupted run (or a rate-limit wall) is restarted with the
same command and loses nothing. Raw JSON is kept verbatim on purpose — the
ETL into SQLite can then be re-run and re-shaped without re-crawling.

Run:
    python -m rag_core.sources.crawl_colleges list
    python -m rag_core.sources.crawl_colleges detail
    python -m rag_core.sources.crawl_colleges detail --limit 500   # smoke test
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from .mme_api import (
    MMEApiError,
    PAGE_LIMIT,
    fetch_college_detail,
    fetch_college_page,
    make_client,
)

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
LIST_FILE = RAW_DIR / "colleges_list.jsonl"
DETAIL_FILE = RAW_DIR / "colleges_detail.jsonl"
MISSING_FILE = RAW_DIR / "colleges_missing.txt"

# Default concurrency. Deliberately modest: this is the client's own
# production API serving real students, and a crawl that degrades the live
# site is a failure no matter how fast it finishes.
DEFAULT_CONCURRENCY = 8


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn final line from a killed run — skip it


def done_ids(path: Path, key: str = "_id") -> set[str]:
    """ObjectIds already captured, so a resumed run skips them."""
    out = set()
    for obj in _iter_jsonl(path):
        cid = obj.get(key) or (obj.get("college") or {}).get("_id")
        if cid:
            out.add(str(cid))
    return out


class _Writer:
    """Serialised append-only JSONL writer.

    A single writer task owns the file handle: concurrent workers hand rows to
    a queue instead of interleaving partial lines into the same file.
    """

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")
        self._n = 0

    def write(self, obj: dict) -> None:
        self._fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._n += 1
        if self._n % 200 == 0:
            self._fh.flush()  # bound the loss window if the process is killed

    def close(self) -> None:
        self._fh.flush()
        self._fh.close()


async def crawl_list(concurrency: int = 4) -> int:
    """Stage 1: page the bulk list endpoint until the catalogue is exhausted."""
    seen = done_ids(LIST_FILE)
    writer = _Writer(LIST_FILE)
    t0 = time.perf_counter()
    added = 0
    try:
        async with make_client(concurrency=concurrency) as client:
            first, total = await fetch_college_page(client, 1)
            pages = max(1, -(-total // PAGE_LIMIT))  # ceil
            print(
                f"[list] {total} colleges across {pages} pages "
                f"({len(seen)} already on disk)",
                file=sys.stderr,
            )
            for row in first:
                if row.get("_id") and row["_id"] not in seen:
                    writer.write(row)
                    seen.add(row["_id"])
                    added += 1

            sem = asyncio.Semaphore(concurrency)

            async def one(page: int):
                async with sem:
                    try:
                        rows, _ = await fetch_college_page(client, page)
                        return rows
                    except MMEApiError as exc:
                        print(f"[list] page {page} failed: {exc}", file=sys.stderr)
                        return []

            for chunk_start in range(2, pages + 1, concurrency * 2):
                chunk = range(chunk_start, min(chunk_start + concurrency * 2, pages + 1))
                for rows in await asyncio.gather(*(one(p) for p in chunk)):
                    for row in rows:
                        if row.get("_id") and row["_id"] not in seen:
                            writer.write(row)
                            seen.add(row["_id"])
                            added += 1
                print(
                    f"[list] page {chunk.stop - 1}/{pages} — {len(seen)} colleges",
                    file=sys.stderr,
                )
    finally:
        writer.close()
    print(
        f"[list] done: +{added} new, {len(seen)} total in "
        f"{time.perf_counter() - t0:.0f}s -> {LIST_FILE}",
        file=sys.stderr,
    )
    return added


def csv_college_ids() -> list[str]:
    """The ObjectIds the courses CSV actually references (crawl scope)."""
    from .courses_csv import iter_course_rows  # local import: avoids a cycle

    ids: dict[str, None] = {}
    for row in iter_course_rows():
        cid = row.get("college_id")
        if cid:
            ids.setdefault(cid, None)
    return list(ids)


async def crawl_detail(
    concurrency: int = DEFAULT_CONCURRENCY, limit: int | None = None
) -> int:
    """Stage 2: per-college enrichment for CSV-referenced colleges."""
    try:
        targets = csv_college_ids()
    except Exception as exc:  # noqa: BLE001 - CSV not reachable: fall back to list
        print(f"[detail] CSV scope unavailable ({exc}); using crawled list",
              file=sys.stderr)
        targets = [r["_id"] for r in _iter_jsonl(LIST_FILE) if r.get("_id")]

    done = done_ids(DETAIL_FILE)
    missing = set()
    if MISSING_FILE.exists():
        missing = {ln.strip() for ln in MISSING_FILE.read_text().splitlines() if ln.strip()}

    todo = [c for c in targets if c not in done and c not in missing]
    if limit:
        todo = todo[:limit]
    print(
        f"[detail] {len(targets)} referenced | {len(done)} done | "
        f"{len(missing)} known-missing | {len(todo)} to fetch",
        file=sys.stderr,
    )
    if not todo:
        return 0

    writer = _Writer(DETAIL_FILE)
    miss_fh = open(MISSING_FILE, "a", encoding="utf-8")
    sem = asyncio.Semaphore(concurrency)
    t0 = time.perf_counter()
    counters = {"ok": 0, "miss": 0, "err": 0}
    lock = asyncio.Lock()

    try:
        async with make_client(concurrency=concurrency) as client:

            async def one(cid: str):
                async with sem:
                    try:
                        payload = await fetch_college_detail(client, cid)
                    except MMEApiError as exc:
                        async with lock:
                            if "404" in str(exc):
                                counters["miss"] += 1
                                miss_fh.write(cid + "\n")
                            else:
                                counters["err"] += 1
                                print(f"[detail] {cid}: {exc}", file=sys.stderr)
                        return
                    async with lock:
                        writer.write(payload)
                        counters["ok"] += 1
                        n = counters["ok"] + counters["miss"] + counters["err"]
                        if n % 250 == 0:
                            rate = n / max(time.perf_counter() - t0, 1e-6)
                            eta = (len(todo) - n) / max(rate, 1e-6) / 60
                            miss_fh.flush()
                            print(
                                f"[detail] {n}/{len(todo)} "
                                f"ok={counters['ok']} miss={counters['miss']} "
                                f"err={counters['err']} | {rate:.1f}/s | ETA {eta:.0f}m",
                                file=sys.stderr,
                            )

            # Bounded batches keep the pending-task list (and memory) flat even
            # though `todo` is tens of thousands of ids long.
            batch = concurrency * 25
            for i in range(0, len(todo), batch):
                await asyncio.gather(*(one(c) for c in todo[i : i + batch]))
    finally:
        writer.close()
        miss_fh.close()

    print(
        f"[detail] done: ok={counters['ok']} miss={counters['miss']} "
        f"err={counters['err']} in {(time.perf_counter() - t0) / 60:.1f}m",
        file=sys.stderr,
    )
    return counters["ok"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["list", "detail"])
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    ap.add_argument("--limit", type=int, default=None,
                    help="detail stage: cap how many colleges to fetch (smoke test)")
    args = ap.parse_args()

    if args.stage == "list":
        asyncio.run(crawl_list(concurrency=min(args.concurrency, 6)))
    else:
        asyncio.run(crawl_detail(concurrency=args.concurrency, limit=args.limit))


if __name__ == "__main__":
    main()
