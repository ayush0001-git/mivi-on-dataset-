"""State Fee Regulating Authority data — the authoritative source for fees.

WHY THIS IS THE BEST FEE SOURCE AVAILABLE, MEASURED AGAINST THE ALTERNATIVES
The catalogue has a fee on 38% of colleges. Filling the rest was attempted three
ways and measured each time:

    college websites (HTML)   14% of sites carry any rupee figure
    college websites (PDF)    70% of fee PDFs are SCANS with no text;
                              1 in 10 relevant PDFs yielded a figure
    commercial aggregators    declined: competitors' terms forbid it, their data
                              is second-hand, and three aggregators "agreeing"
                              usually means one source copied three times

This is better than all of them. In most Indian states a statutory Fee
Regulating Authority must APPROVE the fee a private professional college may
charge, and it publishes the approved figure. So this is not a scrape of
somebody's estimate — it is the legally binding number, with the date of the
order attached.

Maharashtra publishes it through the API behind mahafra.org:

    GET api.mahafraportal.org/mahadbt/inst_district_stream_api.php
        ?ac_year=2024-25&district=Pune&sub_type=ENGG
    -> [{inst_id, name, district, stream, status, dom,
         tution_fees_fra, dev_fees_fra, app_fees_fra}, ...]

Measured for Pune alone: ENGG 70, MBA 101, MCA 36, ARCH 17 institutions — 224
colleges in one district of one state, each with an approved fee and a name to
match on.

MATCHING DISCIPLINE IS THE SAME AS NIRF
The authority keys on its own institute codes (EN6122), so the join is by name
plus district and it REFUSES rather than guesses. Attaching one college's
approved fee to another would be worse than having no fee: the number is real,
precise, official-looking, and about somebody else.

Run:
    python -m rag_core.sources.fee_authority --state mh --year 2024-25 --dry-run
    python -m rag_core.sources.fee_authority --state mh --year 2024-25
    python -m rag_core.sources.fee_authority --report
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from .. import config  # noqa: F401 - loads .env before the backend reads it
from ..store.backend import get_backend

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "raw" / "registries" / "fra"

MH_API = "https://api.mahafraportal.org/mahadbt/inst_district_stream_api.php"

# Confirmed live against the API. Codes that return nothing for a district are
# simply absent there, not an error — many districts have no architecture or
# hotel-management college at all.
MH_STREAMS = ("ENGG", "MBA", "MCA", "ARCH", "PHARM", "HMCT", "MMS", "POLY",
              "BED", "LAW", "NURSING", "DIPLOMA", "BSCNURSING", "AYUR", "HOMEO")

MH_DISTRICTS = (
    "Ahmednagar", "Akola", "Amravati", "Aurangabad", "Beed", "Bhandara",
    "Buldhana", "Chandrapur", "Dhule", "Gadchiroli", "Gondia", "Hingoli",
    "Jalgaon", "Jalna", "Kolhapur", "Latur", "Mumbai", "Mumbai Suburban",
    "Nagpur", "Nanded", "Nandurbar", "Nashik", "Osmanabad", "Palghar",
    "Parbhani", "Pune", "Raigad", "Ratnagiri", "Sangli", "Satara",
    "Sindhudurg", "Solapur", "Thane", "Wardha", "Washim", "Yavatmal",
)

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (compatible; MakeMyEducationBot/1.0; "
                   "+https://makemyeducation.com/about)"),
    "Accept": "application/json, text/plain, */*",
    # The API only answers requests that look like they came from its own SPA.
    "Origin": "https://mahafra.org",
    "Referer": "https://mahafra.org/",
}

# Strict, for the same reason as NIRF: a mis-attached approved fee is worse than
# a missing one.
NAME_SIM_MIN = 0.78
_STOP = {"the", "of", "and", "for", "in", "at", "a", "an", "s"}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", str(s or "").lower())).strip()


def _sig(s: str) -> set[str]:
    return {w for w in _norm(s).split() if len(w) > 2 and w not in _STOP}


def _int(v) -> int | None:
    s = re.sub(r"[^\d]", "", str(v or ""))
    if not s:
        return None
    n = int(s)
    # Approved fees are annual and in rupees. Anything outside this is a data
    # error in the source, not a cheap or expensive college.
    return n if 1_000 <= n <= 50_000_000 else None


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_mh(year: str = "2024-25", districts=MH_DISTRICTS, streams=MH_STREAMS,
             refresh: bool = False, delay: float = 1.2) -> list[dict]:
    """Every (district, stream) combination, cached per combination on disk.

    Cached hard: an approved-fee order changes once a year, so re-fetching 540
    combinations on every run would be pointless load on a government server
    that is clearly running on a modest host (it leaks PHP notices when a
    parameter is missing).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out: list[dict] = []
    with httpx.Client(headers=HEADERS, timeout=30.0, follow_redirects=True,
                      verify=False) as client:
        for district in districts:
            got_district = 0
            for stream in streams:
                cache = CACHE_DIR / f"mh_{year}_{district}_{stream}.json".replace(" ", "_")
                if cache.exists() and not refresh:
                    try:
                        rows = json.loads(cache.read_text(encoding="utf-8"))
                    except (ValueError, OSError):
                        rows = []
                else:
                    try:
                        r = client.get(MH_API, params={
                            "ac_year": year, "district": district,
                            "sub_type": stream})
                        rows = r.json() if r.status_code == 200 else []
                    except Exception:  # noqa: BLE001 - absent combos are normal
                        rows = []
                    if not isinstance(rows, list):
                        rows = []
                    cache.write_text(json.dumps(rows, ensure_ascii=False),
                                     encoding="utf-8")
                    time.sleep(delay)
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    row["_district"] = district
                    row["_stream"] = stream
                    row["_year"] = year
                    out.append(row)
                    got_district += 1
            if got_district:
                print(f"[fra-mh] {district:18} {got_district:>4} approved records",
                      file=sys.stderr)
    return out


# ---------------------------------------------------------------------------
# Match
# ---------------------------------------------------------------------------

def _catalogue_by_district() -> tuple[dict, list]:
    rows = get_backend().q(
        "SELECT college_id, name, city, district, state FROM colleges "
        "WHERE NOT is_abroad AND state ILIKE 'Maharashtra'")
    idx: dict[str, list[dict]] = {}
    for c in rows:
        for key in (c.get("district"), c.get("city")):
            if key:
                idx.setdefault(_norm(key), []).append(c)
    return idx, rows


def match_one(entry: dict, by_district: dict) -> tuple[dict | None, float, str]:
    name = entry.get("name") or ""
    if not name.strip():
        return None, 0.0, "authority record has no institution name"
    sig = _sig(name)
    if not sig:
        return None, 0.0, "no distinctive words in name"

    cands = by_district.get(_norm(entry.get("_district")), [])
    if not cands:
        return None, 0.0, f"no catalogue college in district {entry.get('_district')}"

    best, score, why = None, 0.0, ""
    for c in cands:
        cat_sig = _sig(c["name"])
        if not cat_sig:
            continue
        # Every distinctive word of the AUTHORITY name must be present in the
        # catalogue name (extra words on the catalogue side are fine — it often
        # carries a longer legal name). This is what stops a trust's two
        # colleges in the same district being confused for each other.
        missing = sig - cat_sig
        if len(missing) > max(1, len(sig) // 4):
            continue
        ratio = difflib.SequenceMatcher(None, _norm(name), _norm(c["name"])).ratio()
        overlap = len(sig & cat_sig) / max(1, len(sig | cat_sig))
        s = max(ratio, 0.5 * ratio + 0.5 * overlap)
        if s > score:
            best, score, why = c, s, f"name~{ratio:.2f} overlap~{overlap:.2f}"
    if best is None or score < NAME_SIM_MIN:
        return None, score, f"below floor {NAME_SIM_MIN} ({why})"
    return best, score, why


# ---------------------------------------------------------------------------
# Persist
# ---------------------------------------------------------------------------

def _now():
    return datetime.now(timezone.utc)


def ingest_mh(year: str = "2024-25", dry_run: bool = False,
              refresh: bool = False) -> dict:
    from .registries import ensure_tables

    ensure_tables()
    rows = fetch_mh(year, refresh=refresh)
    if not rows:
        print("[fra-mh] nothing fetched", file=sys.stderr)
        return {}

    by_district, all_mh = _catalogue_by_district()
    print(f"\n[fra-mh] {len(rows)} approved-fee records vs "
          f"{len(all_mh)} Maharashtra colleges in the catalogue", file=sys.stderr)

    be = get_backend()
    stats = {"rows": len(rows), "matched": 0, "unmatched": 0, "no_fee": 0}
    samples: list[str] = []

    for e in rows:
        tuition = _int(e.get("tution_fees_fra"))
        total = _int(e.get("app_fees_fra"))
        if not tuition and not total:
            stats["no_fee"] += 1
            continue

        college, score, why = match_one(e, by_district)
        payload = {
            "approved_tuition_inr": tuition,
            "approved_total_inr": total,
            "development_fees_inr": _int(e.get("dev_fees_fra")),
            "stream": e.get("_stream"),
            "status": e.get("status"),
            "order_date": e.get("dom"),
            "authority_inst_id": e.get("inst_id"),
            "authority_name": e.get("name"),
            "district": e.get("_district"),
        }
        if college is None:
            stats["unmatched"] += 1
            if not dry_run:
                be.execute(
                    "INSERT INTO registry_unmatched"
                    "(registry, ref_id, category, year, name, city, state,"
                    " best_score, note, payload) VALUES(?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT (registry, ref_id, category, year) DO UPDATE SET "
                    "best_score=EXCLUDED.best_score, note=EXCLUDED.note",
                    ["fra_mh", str(e.get("inst_id")), e.get("_stream"), year,
                     e.get("name"), e.get("_district"), "Maharashtra",
                     score, why, json.dumps(payload, ensure_ascii=False)])
            continue

        stats["matched"] += 1
        if len(samples) < 8:
            fee = tuition or total
            samples.append(f"  Rs {fee:>9,}  {str(e.get('name'))[:38]:40} -> "
                           f"{college['name'][:34]:36} ({score:.2f})")
        if not dry_run:
            be.execute(
                "INSERT INTO registry_fact"
                "(college_id, registry, ref_id, category, year, payload,"
                " source_url, detail_url, match_score, match_note, fetched_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT (college_id, registry, category, year) DO UPDATE SET "
                "payload=EXCLUDED.payload, match_score=EXCLUDED.match_score, "
                "match_note=EXCLUDED.match_note, fetched_at=EXCLUDED.fetched_at",
                [college["college_id"], "fra_mh", str(e.get("inst_id")),
                 e.get("_stream"), year, json.dumps(payload, ensure_ascii=False),
                 "https://mahafra.org/", MH_API, score, why, _now()])

    print(f"\n[fra-mh] matched {stats['matched']}/{stats['rows']} "
          f"({100 * stats['matched'] / max(stats['rows'], 1):.1f}%), "
          f"{stats['unmatched']} refused, {stats['no_fee']} had no usable fee"
          f"{'  (DRY RUN — nothing written)' if dry_run else ''}", file=sys.stderr)
    print("\n".join(samples), file=sys.stderr)
    return stats


def report() -> None:
    be = get_backend()
    n = be.one("SELECT COUNT(*) FROM registry_fact WHERE registry='fra_mh'") or 0
    c = be.one("SELECT COUNT(DISTINCT college_id) FROM registry_fact "
               "WHERE registry='fra_mh'") or 0
    print(f"-- Maharashtra FRA --\n   {n:,} approved-fee records "
          f"across {c:,} colleges")
    if not n:
        return
    for r in be.q("SELECT category, COUNT(*) n, "
                  "ROUND(AVG((payload->>'approved_tuition_inr')::numeric)) avg_fee "
                  "FROM registry_fact WHERE registry='fra_mh' "
                  "AND payload->>'approved_tuition_inr' IS NOT NULL "
                  "GROUP BY category ORDER BY n DESC LIMIT 10"):
        print(f"   {str(r['category']):12} {r['n']:>5}  avg approved tuition "
              f"Rs {int(r['avg_fee'] or 0):,}")
    gained = be.one("""SELECT COUNT(DISTINCT f.college_id) FROM registry_fact f
                       LEFT JOIN college_agg a USING(college_id)
                       WHERE f.registry='fra_mh'
                         AND (a.min_tuition_inr IS NULL)""") or 0
    print(f"\n   colleges gaining a fee they had NONE for: {gained:,}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", default="mh", choices=["mh"])
    ap.add_argument("--year", default="2024-25")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
        return
    ingest_mh(a.year, dry_run=a.dry_run, refresh=a.refresh)


if __name__ == "__main__":
    main()
