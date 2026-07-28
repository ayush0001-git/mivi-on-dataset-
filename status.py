"""One command that tells you — or a fresh Claude session — where this project is.

    python status.py

RESUME.md records the PLAN; this reports the live STATE, because the two drift
the moment anything runs. It answers the three questions a resumed session
actually needs: is the store reachable, is the index consistent with the cards,
and what is still unfinished.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _p(label: str, value) -> None:
    print(f"  {label:<34} {value}")


def main() -> None:
    print("=" * 66)
    print("MIVI — live state")
    print("=" * 66)

    # ---- store ------------------------------------------------------------
    try:
        from rag_core.store.backend import get_backend
        be = get_backend()
        one = be.one
    except Exception as exc:  # noqa: BLE001
        print(f"\n  !! Postgres unreachable: {str(exc)[:120]}")
        print("     Start the service, then re-run. Everything else depends on it.")
        return

    print("\nCORPUS")
    colleges = one("SELECT COUNT(*) FROM colleges") or 0
    _p("colleges", f"{colleges:,}")
    _p("courses", f"{one('SELECT COUNT(*) FROM courses') or 0:,}")
    cards = one("SELECT COUNT(*) FROM college_agg WHERE card IS NOT NULL") or 0
    _p("cards rendered", f"{cards:,}")
    detail = one("SELECT COUNT(*) FROM colleges WHERE source_level = 'detail'") or 0
    _p("detail-enriched", f"{detail:,}")

    # The detail crawler's progress is NOT visible in the line above, and reading
    # it that way has now misled two sessions into calling a healthy job dead.
    # The crawler appends to JSONL; that only reaches Postgres when the ETL runs,
    # so a crawl can run for hours with this count frozen. The file is the truth.
    raw = ROOT / "data" / "raw" / "colleges_detail.jsonl"
    if raw.exists():
        st = raw.stat()
        age = time.time() - st.st_mtime
        moving = "WRITING NOW" if age < 120 else f"last write {age / 60:.0f}m ago"
        with raw.open("rb") as fh:                       # 45k+ lines; count bytes
            rows = sum(chunk.count(b"\n") for chunk in iter(lambda: fh.read(1 << 20), b""))
        _p("  detail JSONL on disk", f"{rows:,} rows | {moving}")

    print("\nENRICHMENT")
    for label, sql in (
        ("web facts", "SELECT COUNT(*) FROM college_web_facts"),
        ("  ...across colleges", "SELECT COUNT(DISTINCT college_id) FROM college_web_facts"),
        ("  ...CONFIRMED by verifier",
         "SELECT COUNT(*) FROM fact_verification WHERE verdict='CONFIRMED'"),
        ("  ...quarantined",
         "SELECT COUNT(*) FROM college_web_facts_quarantine"),
        ("registry facts (NIRF/FRA)", "SELECT COUNT(*) FROM registry_fact"),
        ("  ...across colleges", "SELECT COUNT(DISTINCT college_id) FROM registry_fact"),
        ("harvest queue still pending",
         "SELECT COUNT(*) FROM web_crawl_state WHERE status='pending'"),
    ):
        try:
            _p(label, f"{one(sql) or 0:,}")
        except Exception:  # noqa: BLE001 - table may not exist yet
            _p(label, "n/a")

    # ---- index consistency: the thing most likely to be stale -------------
    print("\nDENSE INDEX")
    idx = ROOT / "data" / "index.npz"
    if not idx.exists():
        _p("status", "MISSING — run: python -m rag_core.index_build")
    else:
        try:
            from rag_core.index_build import _card_hash, load_index
            loaded = load_index()
            n = len(loaded["ids"])
            _p("vectors", f"{n:,}")
            _p("complete flag", loaded["meta"].get("complete", "?"))
            # card_hashes is a top-level array in the npz aligned with `ids`,
            # not a dict inside meta — read it there or this check silently
            # reports "cannot check" and the staleness answer, which is the whole
            # reason this script exists, never gets computed.
            hashes: dict = {}
            try:
                import numpy as np
                with np.load(idx, allow_pickle=True) as z:
                    if "card_hashes" in z.files:
                        raw = z["card_hashes"]
                        if getattr(raw, "shape", None) == ():          # 0-d pickle
                            raw = raw.item()
                        if isinstance(raw, dict):
                            hashes = raw
                        else:
                            hashes = dict(zip(loaded["ids"], list(raw)))
            except Exception:  # noqa: BLE001 - absence is reported, not fatal
                hashes = {}
            # Sample rather than hash 38,700 cards: a drift this large shows up
            # in any sample, and the point is a yes/no answer in one second.
            rows = be.q("SELECT college_id, card FROM college_agg "
                        "WHERE card IS NOT NULL ORDER BY college_id LIMIT 400")
            stale = sum(1 for r in rows
                        if hashes.get(r["college_id"]) not in (None, _card_hash(r["card"])))
            if not hashes:
                _p("card match", "cannot check (no hashes stored)")
            elif stale:
                _p("card match",
                   f"STALE — {stale}/{len(rows)} sampled cards changed since build")
                print("     -> run: python -m rag_core.index_build   (incremental)")
            else:
                _p("card match", f"in sync (sampled {len(rows)})")
            if n and colleges and n < colleges * 0.98:
                print(f"     -> only {n:,} of {colleges:,} colleges are indexed; "
                      f"the rest are unreachable by semantic search")
        except Exception as exc:  # noqa: BLE001
            _p("status", f"unreadable: {str(exc)[:80]}")

    # ---- anything still running -------------------------------------------
    print("\nBACKGROUND JOBS")
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -match "
             "'index_build|web_enrich|crawl_colleges|reverify|registries|fee_authority' } | "
             "ForEach-Object { $_.CommandLine }"],
            capture_output=True, text=True, timeout=30)
        lines = [l for l in (out.stdout or "").splitlines() if l.strip()]
        if not lines:
            _p("running", "none")
        for line in lines:
            for name in ("index_build", "web_enrich", "crawl_colleges",
                         "reverify", "registries", "fee_authority"):
                if name in line:
                    _p("running", name)
                    break
    except Exception:  # noqa: BLE001 - not fatal, just unknown
        _p("running", "could not check")

    print("\nNEXT")
    resume = ROOT / "RESUME.md"
    if resume.exists():
        print(f"  Read {resume.name} for the plan, open decisions, and the")
        print("  whole-programme-vs-annual fee finding.")
    print("=" * 66)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
