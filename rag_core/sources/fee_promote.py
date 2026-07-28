"""Put a dated, sourced fee into the database itself — beside the catalogue.

WHAT THE CLIENT ASKED FOR
"fees ko db me bhi update kar lo" — the harvested fees should be IN the
database, not only rendered on a card, so the website can read them.

WHY THIS DOES NOT OVERWRITE courses.tuition_min_inr
That column is the client's own catalogue. A harvested fee can be wrong, and if
it is written over the original there is nothing to fall back to and no way to
tell which colleges were affected. A wrong fee is also the single most damaging
error this system can make: a student who could afford a college is told they
cannot. So the verified figure lands in its OWN columns, carrying where it came
from and what year it is for. The product can prefer it; nothing is destroyed.

WHY MOST HARVESTED FEES CANNOT BE PROMOTED AT ALL
The `fees` fact payload is not a fee — it is a bag of whatever labelled amounts
the page happened to list. Two real examples from the harvest:

    {"1. Tuition Fee for NRI students": "US$ 7000.00",
     "7. For Hosteller Student: Hostel Management ... Charges": "Rs. 4000.00"}
    {"fees / fees after deduction": "Rs.500/-"}

The first is a foreign-currency NRI fee next to a hostel charge; the second is a
refund clause. Promoting either into a student-facing tuition field would put a
confident, precise, wrong number in front of somebody. So a line item is only
promoted when its LABEL says tuition, its CURRENCY is rupees, its MAGNITUDE is
plausible for an Indian programme, and the fact carries a stated year.

TIERS, in the order they are preferred
    fra_approved   The state Fee Regulating Authority's approved figure. Legally
                   binding, annual, and dated by an order number. The best fee
                   data that exists for India.
    college_site   A tuition line item from the college's own website, gated as
                   above, with the year the page states.

Run:
    python -m rag_core.sources.fee_promote --dry-run
    python -m rag_core.sources.fee_promote
    python -m rag_core.sources.fee_promote --report
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone

from .. import config  # noqa: F401 - loads .env before the backend reads it
from ..store.backend import get_backend

DDL = (
    "ALTER TABLE colleges ADD COLUMN IF NOT EXISTS verified_fee_inr BIGINT",
    "ALTER TABLE colleges ADD COLUMN IF NOT EXISTS verified_fee_period TEXT",
    "ALTER TABLE colleges ADD COLUMN IF NOT EXISTS verified_fee_year TEXT",
    "ALTER TABLE colleges ADD COLUMN IF NOT EXISTS verified_fee_basis TEXT",
    "ALTER TABLE colleges ADD COLUMN IF NOT EXISTS verified_fee_label TEXT",
    "ALTER TABLE colleges ADD COLUMN IF NOT EXISTS verified_fee_source TEXT",
    "ALTER TABLE colleges ADD COLUMN IF NOT EXISTS verified_fee_at TIMESTAMPTZ",
)

# A label that names tuition, and not one of the many other things a fee table
# lists. Ordered as: must look like tuition, must not look like something else.
_TUITION_LABEL = re.compile(
    r"\b(tuition|academic\s*fee|course\s*fee|programme?\s*fee|annual\s*fee"
    r"|semester\s*fee|college\s*fee)\b", re.I)
_NOT_TUITION = re.compile(
    r"\b(nri|foreign|international|saarc|hostel|mess|transport|bus|caution"
    r"|refund|deposit|security|library|exam(?:ination)?|late|fine|penalty"
    r"|registration|application|admission\s*form|alumni|insurance|uniform"
    r"|development|one[\s-]?time|misc)\b", re.I)

# Rupee amounts only. A "US$ 7000" tuition is real but not comparable to the
# catalogue's INR figures, and silently mixing currencies is how a college looks
# 85x more expensive than it is.
_RUPEES = re.compile(r"(?:₹|rs\.?|inr)\s*([\d,]+(?:\.\d{1,2})?)", re.I)

# Plausible for one Indian programme. Below the floor is almost always a form
# fee or a per-month figure; above the ceiling is a data-entry error or a total
# for a whole institution.
MIN_INR = 3_000
MAX_INR = 5_000_000


def _amount(value: str) -> int | None:
    m = _RUPEES.search(str(value or ""))
    if not m:
        return None
    try:
        n = int(round(float(m.group(1).replace(",", ""))))
    except ValueError:
        return None
    return n if MIN_INR <= n <= MAX_INR else None


def _period(label: str, raw: str) -> str:
    """Annual, semester, or unstated — never guessed as annual by default.

    Defaulting an unmarked fee to "per year" is exactly the bug that made every
    catalogue fee wrong by 2-4x (see sources/courses_csv.py). Not repeating it.
    """
    hay = f"{label} {raw}".lower()
    if re.search(r"per\s*sem|/\s*sem|semester", hay):
        return "semester"
    if re.search(r"per\s*(annum|year|yr)|/\s*(yr|year)|annual|p\.a\.", hay):
        return "year"
    return "unspecified"


def _pick_tuition(payload) -> tuple[int, str, str] | None:
    """(amount_inr, label, period) for the one line item worth promoting."""
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (TypeError, ValueError):
            return None
    if not isinstance(payload, dict):
        return None

    best: tuple[int, str, str] | None = None
    for label, raw in payload.items():
        if not _TUITION_LABEL.search(str(label)) or _NOT_TUITION.search(str(label)):
            continue
        amt = _amount(raw)
        if amt is None:
            continue
        period = _period(str(label), str(raw))
        # Lowest qualifying tuition wins. Fee tables list the general category
        # first and management/NRI quotas after, and quoting the highest would
        # overstate what most students pay.
        if best is None or amt < best[0]:
            best = (amt, " ".join(str(label).split())[:120], period)
    return best


def collect() -> tuple[list[tuple], dict]:
    be = get_backend()
    now = datetime.now(timezone.utc)
    stats = {"fra": 0, "site": 0, "site_rejected": 0, "site_no_year": 0}
    chosen: dict[str, tuple] = {}

    # ---- tier 1: Fee Regulating Authority ---------------------------------
    for r in be.q("SELECT college_id, category, year, payload, source_url "
                  "FROM registry_fact WHERE registry LIKE 'fra%'"):
        p = r["payload"]
        if isinstance(p, str):
            try:
                p = json.loads(p)
            except (TypeError, ValueError):
                continue
        amt = (p or {}).get("approved_tuition_inr") or (p or {}).get("approved_total_inr")
        if not amt:
            continue
        label = ("Fee Regulating Authority approved tuition"
                 if (p or {}).get("approved_tuition_inr")
                 else "Fee Regulating Authority approved total")
        if (p or {}).get("stream"):
            label += f" [{p['stream']}]"
        chosen[r["college_id"]] = (
            int(amt), "year", r["year"], "fra_approved", label,
            r["source_url"], now, r["college_id"])
        stats["fra"] += 1

    # ---- tier 2: the college's own website --------------------------------
    for r in be.q("SELECT college_id, payload, source_url, stated_year, confidence "
                  "FROM college_web_facts WHERE kind = 'fees'"):
        if r["college_id"] in chosen:
            continue                      # tier 1 outranks a scraped page
        if not r["stated_year"]:
            # Undated: it cannot be aged, so it cannot be offered as current.
            stats["site_no_year"] += 1
            continue
        if float(r["confidence"] or 0) < 0.6:
            stats["site_rejected"] += 1
            continue
        hit = _pick_tuition(r["payload"])
        if not hit:
            stats["site_rejected"] += 1
            continue
        amt, label, period = hit
        chosen[r["college_id"]] = (
            amt, period, r["stated_year"], "college_site", label,
            r["source_url"], now, r["college_id"])
        stats["site"] += 1

    return list(chosen.values()), stats


def run(*, dry_run: bool = False) -> dict:
    be = get_backend()
    if not dry_run:
        for stmt in DDL:
            be.execute(stmt)

    rows, stats = collect()
    print(f"[fee] promotable: {stats['fra']:,} from a Fee Regulating Authority, "
          f"{stats['site']:,} from college websites", file=sys.stderr)
    print(f"[fee] website facts NOT promoted: {stats['site_no_year']:,} undated, "
          f"{stats['site_rejected']:,} with no rupee tuition line item",
          file=sys.stderr)

    for r in rows[:10]:
        print(f"    {r[3]:<13} {r[0]:>9,} /{r[1]:<12} {r[2]:<8} {r[4][:52]}",
              file=sys.stderr)

    if dry_run:
        print("[fee] dry run — nothing written", file=sys.stderr)
        return stats

    be.executemany(
        "UPDATE colleges SET verified_fee_inr = ?, verified_fee_period = ?, "
        "verified_fee_year = ?, verified_fee_basis = ?, verified_fee_label = ?, "
        "verified_fee_source = ?, verified_fee_at = ? WHERE college_id = ?", rows)
    n = be.one("SELECT COUNT(*) FROM colleges WHERE verified_fee_inr IS NOT NULL") or 0
    print(f"[fee] {n:,} colleges now carry a dated, sourced fee in the database",
          file=sys.stderr)
    stats["stored"] = n
    return stats


def report() -> None:
    be = get_backend()
    rows = be.q("SELECT verified_fee_basis b, verified_fee_year y, COUNT(*) n, "
                "MIN(verified_fee_inr) lo, MAX(verified_fee_inr) hi "
                "FROM colleges WHERE verified_fee_inr IS NOT NULL "
                "GROUP BY 1, 2 ORDER BY 1, 2 DESC")
    if not rows:
        print("no verified fees yet — run without --report first")
        return
    print(f"{'basis':<14} {'year':<9} {'colleges':>8}  {'range':>24}")
    for r in rows:
        print(f"{r['b']:<14} {str(r['y']):<9} {r['n']:>8,}  "
              f"Rs {r['lo']:,} - Rs {r['hi']:,}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
        return
    run(dry_run=a.dry_run)


if __name__ == "__main__":
    main()
