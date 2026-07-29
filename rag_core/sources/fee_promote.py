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

WHICH OF A COLLEGE'S SEVERAL APPROVED FEES GETS PROMOTED
An authority approves a fee per COURSE, so one college has as many approved
fees as it has regulated programmes — BBA 41k, B.Tech 70k, MBA 62k — and this
module has exactly one field to put a fee in. 295 of the 682 colleges with FRA
facts carry more than one row, and the spread inside a single college reaches
Rs 24,000 - Rs 86,000.

Until this rule existed the winner was whichever row Postgres happened to
return last, which is not a choice at all: it is heap order. Measured on the
live table, a no-op re-write of 779 rows — exactly what a re-run of any adapter
does, since ON CONFLICT DO UPDATE writes a new tuple at the end of the heap —
moved 7 colleges onto a different fee, and moved them the wrong way:

    69ecbcb3...  Rs  44,350 BBA 2026-27  ->  Rs 1,22,000 B.Tech 2021-22

That is 2.75x the fee, sourced from a five-year-old session, appearing without
anything in the data having changed. So the row is chosen by a rule that is a
TOTAL order over the candidates — no ties left for the database to break:

    1. the most recent academic session      (newest approved fee wins)
    2. then the LOWEST qualifying tuition    (same rule, and same reason, as
       _pick_tuition below: fee tables run general category first and
       management/NRI quotas after, so the highest overstates what most
       students pay)
    3. then category, then registry          (never reached on today's data —
       registry_fact is keyed on (college_id, registry, category, year), so
       steps 1-3 are already unique — but they make the order provably total
       rather than accidentally total)

Step 1 reads the session as the fee's EFFECTIVE-FROM date, which is how the
authorities write it: "for academic session 2016-17 onward", "for the session
2027-28 onwards has been fixed". The newest such row is the one in force.

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

# registry_fact.year is TEXT, and the adapters do not agree on a shape:
# '2024-25' (Maharashtra, Haryana) next to '2024-2025' (Madhya Pradesh). Only
# the opening year is comparable across both, and it is enough to order
# sessions — no academic session starts twice in one calendar year.
_SESSION_START = re.compile(r"(\d{4})")


def _session_start(year) -> int:
    """The calendar year an academic session opens in, for ordering.

    Unparseable sorts OLDEST rather than raising: a malformed session must
    never outrank a real one, and a college whose only row is malformed still
    has a defensible fee to promote.
    """
    m = _SESSION_START.search(str(year or ""))
    return int(m.group(1)) if m else -1


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
        # The VALUE must be disqualified too, not just the key. _amount() takes
        # the first rupee figure anywhere in the string, so a key reading
        # "Tuition Fee" whose value reads "Rs 500 (hostel) Rs 90,000" would have
        # promoted the hostel charge as tuition. Extractors flatten a whole table
        # row into the value often enough that the key alone is not the label for
        # the number that gets picked.
        if _NOT_TUITION.search(str(raw)):
            continue
        amt = _amount(raw)
        if amt is None:
            continue
        # A value holding several rupee figures is a flattened row, and there is
        # no way to know which one the key names. Refuse rather than take the
        # first: this is a fee a student would plan around.
        if len(_RUPEES.findall(str(raw))) > 1:
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
    stats = {"fra": 0, "fra_rows": 0, "fra_contested": 0,
             "site": 0, "site_rejected": 0, "site_no_year": 0}
    chosen: dict[str, tuple] = {}

    # ---- tier 1: Fee Regulating Authority ---------------------------------
    # ILIKE, not LIKE: the registry names are lowercase today, but a
    # case-sensitive match is a silent miss waiting to happen, and a MISS here
    # does not look like a bug — the college simply carries no fee.
    #
    # The ranking is done in Python rather than by the ORDER BY, because the
    # amount lives inside the JSON payload under a key that varies by row
    # (approved_tuition_inr, else approved_total_inr) and is only known after
    # the payload is parsed. The ORDER BY still earns its place: it makes the
    # INPUT sequence stable, so the "keep the first" tie-break below is
    # deterministic even if the key ever stops being total.
    ranked: dict[str, tuple] = {}
    seen: dict[str, int] = {}
    for r in be.q("SELECT college_id, registry, category, year, payload, source_url "
                  "FROM registry_fact WHERE registry ILIKE 'fra%' "
                  "ORDER BY college_id, year DESC, category, registry"):
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
        # The stream names the course this fee was approved FOR, and once a
        # college has several it is the only thing distinguishing the promoted
        # figure from the ones that lost. category carries the same value and
        # covers an adapter that omits the payload key.
        stream = (p or {}).get("stream") or r["category"]
        if stream:
            label += f" [{stream}]"

        cid = r["college_id"]
        amt = int(amt)
        # Newest session first, then cheapest, then a stable name. See the
        # module docstring for why each step is there and in that order.
        key = (-_session_start(r["year"]), amt,
               str(r["category"] or ""), str(r["registry"] or ""))
        stats["fra_rows"] += 1
        seen[cid] = seen.get(cid, 0) + 1
        if cid not in ranked or key < ranked[cid][0]:
            ranked[cid] = (key, (
                amt, "year", r["year"], "fra_approved", label,
                r["source_url"], now, cid))

    # "year" is asserted, not assumed. Maharashtra and Haryana approve an
    # annual fee outright; Madhya Pradesh approves 517 of its 749 per SEMESTER
    # but stores approved_tuition_inr already annualised (it equals
    # approved_tuition_annual_inr on all 749 rows, checked against the live
    # table). Stamping "year" on a semester figure would be the 2-4x bug
    # _period() below exists to prevent.
    chosen.update({cid: row for cid, (_, row) in ranked.items()})
    stats["fra"] = len(ranked)
    stats["fra_contested"] = sum(1 for n in seen.values() if n > 1)

    # ---- tier 2: the college's own website --------------------------------
    # No ranking needed here, unlike tier 1: college_web_facts holds exactly one
    # kind='fees' row per college (672 rows across 672 colleges on the live
    # table), so there is nothing to choose between. _pick_tuition already
    # resolves the several line items INSIDE that one payload, by the same
    # lowest-qualifying rule. ORDER BY only to keep the output reproducible.
    for r in be.q("SELECT college_id, payload, source_url, stated_year, confidence "
                  "FROM college_web_facts WHERE kind = 'fees' ORDER BY college_id"):
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

    # Sorted by college_id, the last element of each row tuple. The CONTENT is
    # already order-independent — the tier-1 key is a total order — so this is
    # about the OUTPUT: it makes the sample printed below, and the sequence
    # handed to executemany, byte-identical between runs, which is what makes
    # "run --dry-run twice and diff" a real check rather than a hopeful one.
    return sorted(chosen.values(), key=lambda row: row[-1]), stats


def run(*, dry_run: bool = False) -> dict:
    be = get_backend()
    if not dry_run:
        for stmt in DDL:
            be.execute(stmt)

    rows, stats = collect()
    # COLLEGES, not facts. This used to count facts while writing colleges, so
    # it reported 1,188 promotable and then wrote 682 — the same overwrite the
    # ranking above now resolves, showing up in the log as a number that never
    # matched what landed in the table.
    print(f"[fee] promotable: {stats['fra']:,} colleges from a Fee Regulating "
          f"Authority (from {stats['fra_rows']:,} approved fees; "
          f"{stats['fra_contested']:,} colleges had more than one and were "
          f"resolved by newest session, then lowest tuition), "
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
