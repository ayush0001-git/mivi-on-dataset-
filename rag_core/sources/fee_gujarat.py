"""Gujarat FRC (Technical) — legally permitted ANNUAL fee per institute+programme.

WHY THIS SOURCE
The Fee Regulatory Committee (Technical), constituted under s.9(1) of the
Gujarat Professional Technical Educational Colleges or Institutions (Regulation
of Admission and Fixation of Fees) Act 2007, fixes the maximum fee a private
technical college may charge. It publishes the figure per institute, per
programme, per ACADEMIC YEAR, in three-year "fee blocks".

    https://frctech.gujarat.gov.in/FeeStructure
    -> GetAllYearDataF -> {"blocks": {"Fee Block Year 2026-2029": [...],
                                      "Fee Block Year 2023-2026": [...]}}

Each row: {Institute, InstituteCode, District, Program, "2026-2027": "91500",
"2027-2028": ..., "2028-2029": ...}. That is a name, a place, a programme and an
annual rupee figure — everything the join needs.

THE FIGURES ARE VERIFIED ANNUAL, NOT WHOLE-PROGRAMME
Checked against the committee's own scanned order
(gujarat_frctech_order_2026-29_upto10pct_15Jul2026.pdf, Appendix-IV), whose
column header reads "Maximum Ceiling limit of the fee structure for the year in
Rs. per annum per student". Three institutions in that appendix reproduce
exactly in this JSON's 2025-2026 column:

    Manjushree Inst. of Pharmacy, Gandhinagar   68000  == JSON 12117 2025-2026
    Prime College of Diploma, Navsari           31500  == JSON 31914 2025-2026
    Merchant Engineering College, Basana        66150  == JSON 11165 2025-2026

So the JSON is the order, transcribed — not a secondary estimate.

INSTITUTIONS WHOSE FEE IS **NOT** DETERMINED ARE EXCLUDED BY CODE, NOT BY NAME
gujarat_frctech_list_fee_not_determined_2026-29_04May2026.pdf (Phase-2) names
two institutions that submitted closure documents. Both are already absent from
the fee JSON, but they are excluded here by INSTITUTE CODE as a standing guard.
By code and not by name deliberately: one of them is "L. J. Institute of
Management Studies (MCA), Ahmedabad" (94519), and a name-based exclusion would
also have dropped four unrelated, legitimately-priced L. J. institutes
(55108, 54517, 51154, 52237). Sentinel cells ("started in 2025-26", "Not
Responded", "Closed From 2024-25", "CoE University from 2025-26") are likewise
never read as fees.

MATCHING
Reuses registries.match_one unchanged — family gate, token containment, dotted
acronym, leading brand, city/district aliasing — and then REFUSES FURTHER. Two
extra gates, both measured on this data during the dry run:

  1. FULL token containment. match_one tolerates one short missing token for
     honorifics ("Dr", "Shri"). In Gujarat the trust brand is itself often
     short, so that tolerance is fatal here:
         "Sal College Of Engineering"  -> "L D College of Engineering"   0.92
         "Sal School Of Architecture"  -> "L.J. School of Architecture"  0.92
         "S V Institute Of Management (MCA)" -> "Institute of Hotel Management"
     SAL is a private trust; L D College of Engineering is the state's
     best-known government college. Requiring every FRC token to be present
     kills all three.
  2. HEAD-TOKEN gate on the FRC side. registries' leading-brand gate guards the
     catalogue side only; this is its mirror.

Neither gate lowers a floor. Both only refuse.

CONTENTION
Two DIFFERENT institutes resolving to one catalogue college are both refused —
never the higher score. Institute identity is the normalised core name, so the
same institute written "Ahmedabad Institute Of Technology (MBA)" and "Ahmedabad
Institute Of Technology" is one institute with two programmes, not a conflict.
Where several genuinely distinct institutes claim one college, a survivor must
also satisfy catalogue_tokens subset FRC_tokens — which is true of a constituent
school ("Dr. Subhash University" inside "Dr. Subhash University, School of
Pharmacy") and false of a sibling institute ("...Institute of Computer Studies"
against a catalogue "...Institute of Business Management and Computer Studies").

Run:
    python -m rag_core.sources.fee_gujarat --dry-run
    python -m rag_core.sources.fee_gujarat
    python -m rag_core.sources.fee_gujarat --report
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from .. import config  # noqa: F401 - loads .env before the backend reads it
from ..store.backend import get_backend
from . import registries as R

ROOT = Path(__file__).resolve().parents[2]
FEES_DIR = ROOT / "data" / "raw" / "registries" / "fees"
FEE_JSON = FEES_DIR / "gujarat_frctech_feestructure.json"
CIRCULARS_JSON = FEES_DIR / "gujarat_frctech_fee_circulars.json"

SOURCE_URL = "https://frctech.gujarat.gov.in/FeeStructure"
AUTHORITY = "Fee Regulatory Committee (Technical), Gujarat"
REGISTRY = "frc_gj"
STATE = "Gujarat"

# From gujarat_frctech_list_fee_not_determined_2026-29_04May2026.pdf, Appendix-I
# (Phase-2). Excluded by CODE — see module docstring.
FEE_NOT_DETERMINED_CODES = {
    "94519",  # L. J. Institute of Management Studies (MCA), Ahmedabad
    "31681",  # Shri Chanasma ... Institute of Diploma Engineering, Shelavi, Patan
}

# Cells that are status text, not money.
_SENTINEL = re.compile(
    r"start|not\s*respond|clos|coe\s|univ|pending|nil|n\.?a\.?$|-{2,}", re.I)

_GEO_PREFIX = re.compile(
    r"^(dist|dis|distt|ta|tal|taluka|at|po|near|opp|village|vill|dt)[.:\s]", re.I)

_GENERIC = {"college", "institute", "institution", "school", "faculty",
            "engineering", "technology", "technical", "science", "sciences",
            "management", "studies", "pharmacy", "architecture", "planning",
            "research", "education", "degree", "diploma", "computer",
            "applications", "university", "campus", "centre", "center",
            "department", "polytechnic", "vocational", "hotel", "business",
            "administration", "commerce", "arts", "training"}

_ABBREV = {"eng": "engineering", "engg": "engineering", "tech": "technology",
           "edu": "education", "inst": "institute", "insitute": "institute",
           "instt": "institute", "univ": "university", "mgmt": "management",
           "dip": "diploma", "sci": "science", "pharm": "pharmacy",
           "appl": "applications", "res": "research", "clg": "college"}

# Programme tags the committee appends to disambiguate its OWN rows. They are
# not part of the institution's name and they break token containment.
_PROGRAM_TAG = re.compile(
    r"\((?:mca|mba|be|b\.e|btech|b\.tech|me|m\.e|mtech|m\.tech|bca|bba|"
    r"b\.pharm[a-z.]*|m\.pharm[a-z.]*|diploma|degree|dip|pharmacy|"
    r"b\.arch|m\.arch|arch)\)", re.I)


def _expand(s: str) -> str:
    return " ".join(_ABBREV.get(w, w) for w in R._norm(s).split())


def _tok(s: str) -> set[str]:
    return {w for w in _expand(s).split() if len(w) > 2 and w not in R._STOP}


def _programme(v) -> str:
    """Programme label, whitespace-collapsed.

    The feed contains 'Diploma \\nEngg.' with an embedded newline alongside
    'Diploma Engg.'. Left alone those are two different categories, so one
    college would carry the same diploma fee twice under keys that only differ
    by a line break — and registry_fact keys on category.
    """
    return re.sub(r"\s+", " ", str(v or "")).strip()


def _fee(v) -> int | None:
    """An annual rupee figure, or None when the cell is status text."""
    s = str(v or "").strip()
    if not s or _SENTINEL.search(s):
        return None
    s = s.replace(",", "")
    if not re.fullmatch(r"\d+(?:\.\d+)?", s):
        return None
    n = int(round(float(s)))
    # Approved annual technical-programme fees in Gujarat sit well inside this.
    # Anything outside is a transcription error in the source, not a bargain.
    return n if 1_000 <= n <= 5_000_000 else None


def current_academic_year(today: datetime | None = None) -> str:
    """The academic year in force, labelled the way this feed labels them.

    Indian academic years start around June, so 29 Jul 2026 is 2026-2027.
    """
    d = today or datetime.now(timezone.utc)
    start = d.year if d.month >= 6 else d.year - 1
    return f"{start}-{start + 1}"


def applicable_year(row: dict, years: list[str], today: datetime | None = None) -> str | None:
    """Which of the block's years to publish as THIS row's fee.

    Not the first one. Taking the earliest priced column stamped 256 rows of the
    2023-2026 block with a 2023-2024 fee, i.e. MIVI would have quoted a
    three-year-old figure as today's cost. The rule instead is: the year in
    force if this block covers it, else the most recent year that has already
    started, else — for a block entirely in the future — its first year.
    """
    cur = current_academic_year(today)
    priced = [y for y in years if _fee(row.get(y)) is not None]
    if not priced:
        return None
    if cur in priced:
        return cur
    past = [y for y in priced if y <= cur]
    return max(past) if past else min(priced)


def _is_closed(row: dict, years: list[str]) -> bool:
    return any(re.search(r"clos", str(row.get(y) or ""), re.I) for y in years)


# ---------------------------------------------------------------------------
# Load (from disk — the data is already downloaded)
# ---------------------------------------------------------------------------

def load_blocks() -> dict[str, list[dict]]:
    doc = json.loads(FEE_JSON.read_text(encoding="utf-8"))
    return doc.get("blocks") or {}


def block_years(rows: list[dict]) -> list[str]:
    return sorted({k for r in rows for k in r if re.fullmatch(r"\d{4}-\d{4}", k)})


# Circulars that are NOT a general fee order for a block: single-institution
# determinations, corrigenda, and the not-determined lists. Including them was
# measured to be actively misleading — the newest circular mentioning the
# 2023-26 block is "Fee structure determined for the constituent institutions of
# Silver Oak University", which would have stamped one university's amendment
# date onto all 269 matched rows of that block.
_SPECIFIC = re.compile(r"not determined|corrigendum|reduction in fees", re.I)
_GENERAL = re.compile(r"\bfor the institutions\b|\bfor institutions\b", re.I)

_BLOCK_PATTERNS = (
    ("Fee Block Year 2026-2029", ("2026-27 to 2028-29", "2026-29")),
    ("Fee Block Year 2023-2026", ("2023-24 to 2025-26", "2023-26")),
)


def order_dates() -> dict[str, dict]:
    """The general fee orders that govern each block, from the circulars feed.

    The fee table carries no per-institute order date: a block's figures are
    fixed across a series of dated lists (2026-29 ran to four), and the feed
    never says which list a given institute appeared in. So this records the
    BLOCK-level order — the earliest general order, which is the one that
    established the block — together with every subsequent general order date,
    and the payload labels it as block-scoped. Picking one date and presenting
    it as this institute's order date would be fabricating provenance.
    """
    try:
        circs = json.loads(CIRCULARS_JSON.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    found: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    for c in circs:
        title = str(c.get("ecitizenTitle") or "")
        low = title.lower()
        if "fee structure" not in low and "fee block" not in low:
            continue
        if _SPECIFIC.search(title) or not _GENERAL.search(title):
            continue
        date = str(c.get("ecitizenStartDate") or "")[:10]
        if not date:
            continue
        for block, pats in _BLOCK_PATTERNS:
            if any(p in low for p in pats):
                found[block].append((date, title))
    out: dict[str, dict] = {}
    for block, lst in found.items():
        lst.sort()
        out[block] = {
            "order_date": lst[0][0],
            "order_circular": lst[0][1][:220],
            "block_order_dates": sorted({d for d, _t in lst}),
        }
    return out


# ---------------------------------------------------------------------------
# Match
# ---------------------------------------------------------------------------

def clean_name(raw: str, place_keys: set[str], aliases: dict) -> tuple[str, list[str]]:
    """Institute core name + place hints recovered from its trailing segments.

    FRC names carry the location inline ("A Y Dadabhai Technical Institute,
    Kosamba, Dist. Surat"). Left in place those tokens are never present in the
    catalogue name and token containment refuses every such row. Segments are
    only dropped when the catalogue's OWN place vocabulary recognises them, or
    when they are explicitly geographic ("Dist. ..."), so this reads places out
    of the data rather than guessing at them.
    """
    s = _PROGRAM_TAG.sub(" ", str(raw or ""))
    s = re.sub(r"\((?:erstwhile|formerly)[^)]*\)?", " ", s, flags=re.I)
    parts = [p.strip() for p in s.split(",") if p.strip()]
    hints: list[str] = []
    while len(parts) > 1:
        last = parts[-1]
        stripped = _GEO_PREFIX.sub("", last).strip()
        key = R._place_key(stripped, aliases)
        if key in place_keys or _GEO_PREFIX.match(last):
            if key:
                hints.append(key)
            parts.pop()
            continue
        break
    core = ", ".join(parts).replace("&", " and ")
    core = re.sub(r"\s+", " ", core).strip(" ,.-")
    return core, hints


def _brandless(core: str) -> bool:
    """No identity-bearing word — 'Engineering College, Tuwa' after place strip."""
    return not (_tok(core) - _GENERIC)


def _head(core: str) -> str:
    for w in _expand(core).split():
        if len(w) > 2 and w not in R._STOP:
            return w
    return ""


def attempt(core: str, city: str, by_city, by_token, aliases):
    """registries.match_one, then two extra refusals. Never a lowered floor."""
    c, score, why = R.match_one({"name": core, "city": city},
                                by_city, by_token, aliases)
    if c is None:
        return None, score, why
    fsig, csig = _tok(core), _tok(c["name"])
    missing = fsig - csig
    if missing:
        return None, score, f"FRC tokens absent from catalogue {sorted(missing)}"
    h = _head(core)
    if h and h not in csig and h not in re.sub(r"[^a-z]", "", R._norm(c["name"])):
        return None, score, f"head token '{h}' absent from catalogue name"
    return c, score, why


def resolve(rows: list[dict], by_city, by_token, aliases, place_keys):
    """Match every row, then apply contention refusal. Returns (kept, refused)."""
    claims: dict[str, list] = collections.defaultdict(list)
    refused: list[tuple[dict, float, str]] = []

    for r in rows:
        core, hints = clean_name(r.get("Institute"), place_keys, aliases)
        if not core or _brandless(core):
            refused.append((r, 0.0, f"no identity-bearing word in '{core}'"))
            continue
        district = R._place_key(r.get("District"), aliases)
        cands = [core]
        first = core.split(",")[0].strip()
        if first != core and not _brandless(first):
            cands.append(first)
        got = None
        best_score, best_why = 0.0, "no candidate cleared the gates"
        for cand in cands:
            for city in hints + [district]:
                c, s, w = attempt(cand, city, by_city, by_token, aliases)
                if s > best_score:
                    best_score, best_why = s, w
                if c is not None:
                    got = (c, s, w, cand)
                    break
            if got:
                break
        if got is None:
            refused.append((r, best_score, best_why))
        else:
            claims[got[0]["college_id"]].append((r, got))

    kept: list[tuple[dict, tuple]] = []
    for _cid, lst in claims.items():
        institutes = {_expand(g[3]) for _r, g in lst}
        if len(institutes) == 1:
            kept.extend(lst)
            continue
        # Several distinct institutes claim this college.
        survivors = []
        for r, g in lst:
            # A constituent school restates the catalogue entity and adds to it.
            # A sibling institute does not.
            if _tok(g[0]["name"]) <= _tok(g[3]):
                survivors.append((r, g))
            else:
                refused.append((r, g[1],
                                f"contends for {g[0]['name'][:60]!r} with "
                                f"{len(institutes) - 1} other institute(s)"))
        by_prog: dict[str, list] = collections.defaultdict(list)
        for r, g in survivors:
            by_prog[_programme(r.get("Program"))].append((r, g))
        for prog, plst in by_prog.items():
            if len({_expand(g[3]) for _r, g in plst}) > 1:
                # THE HARD CASE: same college, same programme, two institutes.
                # Refuse both. The higher score was measured wrong on real pairs.
                for r, g in plst:
                    refused.append((r, g[1],
                                    f"two institutes contend for {prog} at "
                                    f"{g[0]['name'][:60]!r} — both refused"))
            else:
                kept.extend(plst)
    return kept, refused


# ---------------------------------------------------------------------------
# Persist
# ---------------------------------------------------------------------------

def _now():
    return datetime.now(timezone.utc)


def purge(blocks: list[str] | None = None) -> int:
    """Drop this registry's existing rows before a rebuild.

    ON CONFLICT alone is not enough to make the run idempotent, because the
    primary key includes `year` and a row can legitimately move between years:
    correcting the applicable-year rule re-keyed 256 rows from 2023-2024 to
    2025-2026 and left the superseded originals behind, so 161 colleges briefly
    carried two different annual fees for the same programme. The input is a
    static file on disk, so a rebuild is authoritative — clear, then insert.
    """
    be = get_backend()
    removed = 0
    for table in ("registry_fact", "registry_unmatched"):
        if blocks:
            for block in blocks:
                removed += be.execute(
                    f"DELETE FROM {table} WHERE registry=? "
                    "AND payload->>'fee_block' = ?", [REGISTRY, block]) or 0
        else:
            removed += be.execute(f"DELETE FROM {table} WHERE registry=?",
                                  [REGISTRY]) or 0
    return removed


def ingest(dry_run: bool = False, blocks: list[str] | None = None) -> dict:
    R.ensure_tables()
    if not dry_run:
        purge(blocks)
    all_blocks = load_blocks()
    if not all_blocks:
        print("[frc-gj] no blocks in the fee JSON", file=sys.stderr)
        return {}
    wanted = blocks or list(all_blocks)
    orders = order_dates()

    colleges = [c for c in R._load_catalogue()
                if str(c.get("state") or "").strip().lower() == STATE.lower()]
    aliases = R._load_place_aliases()
    by_city = R._index_by_city(colleges, aliases)
    by_token = R._index_by_token(colleges)
    place_keys = set(by_city)
    print(f"[frc-gj] {len(colleges):,} Gujarat colleges in the catalogue, "
          f"{len(place_keys)} place keys", file=sys.stderr)

    be = get_backend()
    total = {"rows": 0, "matched": 0, "refused": 0, "skipped": 0}
    samples: list[str] = []

    for block in wanted:
        rows = all_blocks.get(block) or []
        years = block_years(rows)
        if not years:
            continue
        order = orders.get(block, {})

        usable, skipped = [], 0
        for r in rows:
            if str(r.get("InstituteCode")) in FEE_NOT_DETERMINED_CODES:
                skipped += 1
                continue
            if _is_closed(r, years):
                skipped += 1
                continue
            if not any(_fee(r.get(y)) is not None for y in years):
                skipped += 1
                continue
            usable.append(r)

        kept, refused = resolve(usable, by_city, by_token, aliases, place_keys)
        total["rows"] += len(rows)
        total["skipped"] += skipped
        total["matched"] += len(kept)
        total["refused"] += len(refused)

        print(f"\n[frc-gj] {block}: {len(rows)} rows -> {len(usable)} with a "
              f"usable fee ({skipped} skipped: not-determined / closed / "
              f"status-text)", file=sys.stderr)
        print(f"[frc-gj]   matched {len(kept)} "
              f"({100 * len(kept) / max(len(usable), 1):.1f}% of usable) across "
              f"{len({g[0]['college_id'] for _r, g in kept})} colleges, "
              f"{len(refused)} refused", file=sys.stderr)

        for r, (college, score, why, cand) in kept:
            fees = {y: _fee(r.get(y)) for y in years}
            current = applicable_year(r, years) or years[0]
            in_force = current == current_academic_year()
            payload = {
                "annual_fee_inr": fees[current],
                "academic_year": current,
                "fees_by_year_inr": fees,
                "fee_block": block,
                # The 2023-2026 block lapsed on 31 Mar 2026. Its figures are the
                # last legally permitted ones for those institutes, not this
                # year's — a consumer that quotes them as current is quoting an
                # expired ceiling, so say so in the row itself.
                "is_current": in_force,
                "superseded": not in_force,
                "programme": _programme(r.get("Program")),
                "authority": AUTHORITY,
                "authority_institute_code": str(r.get("InstituteCode")),
                "authority_name": r.get("Institute"),
                "district": r.get("District"),
                "fee_basis": "per annum per student, maximum permitted",
                # Block-level, not per-institute: the feed carries no per-row
                # order date and inventing one would fabricate provenance.
                "order_date": order.get("order_date"),
                "order_date_scope": (
                    "fee block — the order that established this block, not "
                    "attributable to this institute individually"
                    if order.get("order_date") else None),
                "order_circular": order.get("order_circular"),
                "block_order_dates": order.get("block_order_dates"),
            }
            if len(samples) < 10:
                samples.append(
                    f"  Rs {fees[current]:>8,}/yr {_programme(r.get('Program'))[:12]:14}"
                    f"{str(r.get('Institute'))[:40]:42} -> "
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
                    [college["college_id"], REGISTRY, str(r.get("InstituteCode")),
                     _programme(r.get("Program")), current,
                     json.dumps(payload, ensure_ascii=False),
                     SOURCE_URL, SOURCE_URL, score, why, _now()])

        if not dry_run:
            for r, score, why in refused:
                fees = {y: _fee(r.get(y)) for y in years}
                current = applicable_year(r, years) or years[0]
                be.execute(
                    "INSERT INTO registry_unmatched"
                    "(registry, ref_id, category, year, name, city, state,"
                    " best_score, note, payload) VALUES(?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT (registry, ref_id, category, year) DO UPDATE SET "
                    "best_score=EXCLUDED.best_score, note=EXCLUDED.note",
                    [REGISTRY, str(r.get("InstituteCode")),
                     _programme(r.get("Program")), current, r.get("Institute"),
                     r.get("District"), STATE, score, why[:400],
                     json.dumps({"fees_by_year_inr": fees, "fee_block": block},
                                ensure_ascii=False)])

    print(f"\n[frc-gj] TOTAL {total['matched']} facts written, "
          f"{total['refused']} refused, {total['skipped']} skipped"
          f"{'   (DRY RUN — nothing written)' if dry_run else ''}",
          file=sys.stderr)
    print("\n".join(samples), file=sys.stderr)
    return total


def report() -> None:
    be = get_backend()
    n = be.one(f"SELECT COUNT(*) FROM registry_fact WHERE registry='{REGISTRY}'") or 0
    c = be.one("SELECT COUNT(DISTINCT college_id) FROM registry_fact "
               f"WHERE registry='{REGISTRY}'") or 0
    print(f"-- Gujarat FRC (Technical) --\n   {n:,} approved annual-fee records "
          f"across {c:,} colleges")
    if not n:
        return
    for r in be.q("SELECT category, COUNT(*) n, "
                  "ROUND(AVG((payload->>'annual_fee_inr')::numeric)) avg_fee "
                  f"FROM registry_fact WHERE registry='{REGISTRY}' "
                  "AND payload->>'annual_fee_inr' IS NOT NULL "
                  "GROUP BY category ORDER BY n DESC LIMIT 12"):
        print(f"   {str(r['category']):16} {r['n']:>4}  avg permitted "
              f"Rs {int(r['avg_fee'] or 0):,}/yr")
    gained = be.one("SELECT COUNT(DISTINCT f.college_id) FROM registry_fact f "
                    "LEFT JOIN college_agg a USING(college_id) "
                    f"WHERE f.registry='{REGISTRY}' AND a.min_tuition_inr IS NULL") or 0
    print(f"\n   colleges gaining a fee they had NONE for: {gained:,}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Gujarat FRC(T) approved annual fees")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--block", action="append",
                    help="restrict to a block, e.g. 'Fee Block Year 2026-2029'")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
        return
    ingest(dry_run=a.dry_run, blocks=a.block)


if __name__ == "__main__":
    main()
