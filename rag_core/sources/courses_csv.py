"""Stream-normalise the makeMyEducation.courses MongoDB export.

The export is the live `courses` collection: 211,511 rows / 169 MB / 37 columns,
flattened Mongo-style (`tutionFees[0].startPrice`, `eligibility[0].exams[0]`).
It is streamed, never loaded whole — a dict-per-row parse of this file peaks
well over 1 GB otherwise, and the ETL has to run on the same box as the server.

Three things about this data drive every downstream decision:

1. NO COLLEGE IDENTITY. The only college reference is `college`, an ObjectId.
   Name / city / state / type all come from the API (see mme_api.py). A course
   row on its own cannot answer a single student question.

2. FEES ARE NOT ALL RUPEES. `isAbroad=true` rows (3,657 of them) carry fees in
   the institution's own currency — the CSV drops the `currencySymbol` the API
   returns. Comparing them against an Indian student's rupee budget would be
   catastrophically wrong (a "3,000,000" MEng is not 30 lakh INR), so the
   rupee-typed field is populated ONLY for domestic rows and abroad amounts are
   kept in a separate, explicitly untyped field.

3. MOST OPTIONAL COLUMNS ARE EMPTY. programLevel / department / courseMode /
   durationUnit / keyHighlights are filled on ~10 rows out of 211k. They are
   parsed when present but nothing may depend on them; `programLevel` is
   instead DERIVED from the course title, which is 100% filled.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# The export lives outside the repo (169 MB); override with MME_COURSES_CSV.
DEFAULT_CSV = Path(
    r"D:\make my education\colleg data cleaning\colleg data cleaning"
    r"\makeMyEducation.courses (2).csv"
)

csv.field_size_limit(10**9)  # free-text `description` blows past the default


def csv_path() -> Path:
    import os

    return Path(os.getenv("MME_COURSES_CSV") or DEFAULT_CSV)


def _clean(v) -> str:
    return "" if v is None else str(v).strip()


def _int(v) -> int | None:
    s = _clean(v).replace(",", "")
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _bool(v) -> bool:
    return _clean(v).lower() == "true"


# Titles whose level cannot be inferred from their prefix, checked first and
# by exact title. Two traps live here:
#   - MBBS starts with "M" but is India's flagship UNDERGRADUATE medical degree.
#     Left to the prefix rules it matched `^m\.?[a-z]` and 814 course rows were
#     labelled Postgraduate, hiding them from every undergraduate search.
#   - GNM/ANM/DPharm are nursing and pharmacy diplomas with no "diploma" in the
#     title; 3,400 rows fell through to "Other".
_LEVEL_BY_TITLE = {
    "mbbs": "Undergraduate", "bds": "Undergraduate", "bams": "Undergraduate",
    "bhms": "Undergraduate", "bums": "Undergraduate", "bpt": "Undergraduate",
    "bsn": "Undergraduate", "bvsc": "Undergraduate", "bnys": "Undergraduate",
    "gnm": "Diploma/Certificate", "anm": "Diploma/Certificate",
    "dpharm": "Diploma/Certificate", "dmlt": "Diploma/Certificate",
    "md": "Postgraduate", "ms": "Postgraduate", "mds": "Postgraduate",
    "dm": "Doctorate", "mch": "Doctorate",
}

# Prefix rules, matched against the TITLE ALONE. They are anchored with ^, and
# an earlier version tested them against "title + full_name" — which made every
# $-anchored rule unreachable, because the haystack never ended where the title
# did. Order matters: doctoral before PG, PG before UG.
_LEVEL_RULES = (
    ("Doctorate", (r"^ph\.?d", r"^d\.?sc", r"^dlitt", r"^dm\b", r"^mch\b")),
    ("Postgraduate", (r"^m\.?[a-z]", r"^pg\b", r"^pgd", r"^llm")),
    ("Undergraduate", (r"^b\.?[a-z]", r"^llb")),
    ("Diploma/Certificate", (r"^diploma", r"^certif", r"^adv", r"^iti\b",
                             r"^polytechnic", r"^d\.?[a-z]")),
)

# Last resort, matched against title + full_name. Unanchored on purpose, so
# "Diploma in Civil Engineering" is caught however it is titled — but checked
# only after the title-based rules, since a full_name mentioning "Master"
# should never outrank an unambiguous bachelor title.
_LEVEL_WORDS = (
    ("Doctorate", (r"doctor of philosophy", r"\bph\.?d\b")),
    ("Diploma/Certificate", (r"\bdiploma\b", r"\bcertificate\b", r"\bcertification\b")),
    ("Postgraduate", (r"\bmaster\b", r"\bpost[- ]?graduate\b")),
    ("Undergraduate", (r"\bbachelor\b", r"\bunder[- ]?graduate\b")),
)


def program_level(title: str, full_name: str = "") -> str:
    t = (title or "").strip().lower()
    t_key = re.sub(r"[^a-z]", "", t)
    if t_key in _LEVEL_BY_TITLE:
        return _LEVEL_BY_TITLE[t_key]
    for level, pats in _LEVEL_RULES:
        if any(re.search(p, t) for p in pats):
            return level
    hay = f"{t} {(full_name or '').lower()}".strip()
    for level, pats in _LEVEL_WORDS:
        if any(re.search(p, hay) for p in pats):
            return level
    return "Other"


# Ceiling above which a RUPEE tuition is a data-entry error rather than a price.
# Measured: 20 rows carry 8,900,000,000 (Rs 890 crore/year) and 249 domestic
# rows exceed Rs 1 crore/year. Rs 5 crore is ~15x India's most expensive real
# programme, so nothing legitimate is anywhere near it — but the bound is
# deliberately generous rather than tight, because the cost of deleting a real
# fee is higher than the cost of keeping an implausible one out of a ranking.
# Applies to INR-typed fields ONLY: `*_raw` holds foreign currencies where a
# large number can be perfectly correct (3,000,000 JPY is a real tuition).
MAX_PLAUSIBLE_INR = 50_000_000


def _fee_range(start, end) -> tuple[int | None, int | None]:
    """Normalise a (startPrice, endPrice) pair into a sane min/max.

    Zero is the collection's "not recorded" sentinel, not a free course, so it
    is mapped to None — a 0 that survives into the index would make every
    "cheapest college" answer wrong."""
    lo, hi = _int(start), _int(end)
    lo = lo if lo and lo > 0 else None
    hi = hi if hi and hi > 0 else None
    if lo and hi and lo > hi:
        lo, hi = hi, lo
    if lo and not hi:
        hi = lo
    if hi and not lo:
        lo = hi
    return lo, hi


def _inr(value: int | None, is_abroad: bool) -> int | None:
    """Rupee-typed value, or None if it is foreign or physically impossible."""
    if is_abroad or value is None:
        return None
    return value if 0 < value <= MAX_PLAUSIBLE_INR else None


def normalise_row(row: dict) -> dict:
    """One raw CSV row -> one flat, typed course record."""
    title = _clean(row.get("title"))
    full = _clean(row.get("courseFullName"))
    is_abroad = _bool(row.get("isAbroad"))

    tui_lo, tui_hi = _fee_range(
        row.get("tutionFees[0].startPrice"), row.get("tutionFees[0].endPrice")
    )
    hos_lo, hos_hi = _fee_range(
        row.get("hostelFees[0].startPrice"), row.get("hostelFees[0].endPrice")
    )
    one_time = _int(row.get("oneTimeFees")) or None

    # Entrance exams: two columns carry the same content in this export
    # (`entrance` is a comma string, `exams[0]` repeats it). Merge + dedupe.
    exams: list[str] = []
    for key in ("eligibility[0].entrance", "eligibility[0].exams[0]"):
        for part in re.split(r"[,;/]", _clean(row.get(key))):
            part = part.strip()
            if part and part.lower() not in {e.lower() for e in exams}:
                exams.append(part)

    duration_value = _int(row.get("durationValue"))
    if not duration_value or duration_value <= 0:
        duration_value = None

    highlights = [
        _clean(row.get(k)) for k in ("keyHighlights[0]", "keyHighlights[1]")
    ]

    return {
        "course_id": _clean(row.get("_id")),
        "college_id": _clean(row.get("college")),
        "title": title,
        "full_name": full,
        "program_level": _clean(row.get("programLevel")) or program_level(title, full),
        "department": _clean(row.get("department")),
        "course_mode": _clean(row.get("courseMode")),
        "duration_value": duration_value,
        "duration_unit": _clean(row.get("durationUnit")) or "year",
        # Rupee-typed ONLY for domestic rows — see module docstring (2) — and
        # only when the figure is physically possible (MAX_PLAUSIBLE_INR).
        # An out-of-range value is treated exactly like a 0: "not recorded",
        # so it can never be quoted to a student or win a fee ranking.
        "tuition_min_inr": _inr(tui_lo, is_abroad),
        "tuition_max_inr": _inr(tui_hi, is_abroad),
        # Raw amount kept ALWAYS and unclamped: for abroad rows the currency is
        # not rupees, and for domestic rows this preserves the original figure
        # so a data-quality report can still find it.
        "tuition_min_raw": tui_lo,
        "tuition_max_raw": tui_hi,
        "hostel_min_inr": _inr(hos_lo, is_abroad),
        "hostel_max_inr": _inr(hos_hi, is_abroad),
        "one_time_fees": one_time,
        # NOT defaulted to "Year". The source's tuitionFeePeriod is empty on
        # every row, and defaulting it here was a real, damaging assumption:
        # every fee in the corpus was then presented to students as annual.
        #
        # Cross-checked against Maharashtra's Fee Regulating Authority (the
        # legally approved annual fee, 564 same-stream pairs), the catalogue
        # figure divided by the approved fee tracks the COURSE LENGTH:
        #     MBA  (2 years)  median ratio 2.00
        #     MCA  (2 years)  median ratio 2.39
        #     BTech (4 years) median ratio 3.55
        # so these are whole-programme fees, not annual ones. Presenting
        # Rs 2,39,000 as "per year" for a college whose approved annual fee is
        # Rs 67,272 tells a student they cannot afford a college they can.
        #
        # duration_value is populated on 10 of 211,071 rows, so the annual
        # figure CANNOT be derived — which means the honest thing is to stop
        # claiming a period we do not know, not to guess a better one.
        "fee_period": _clean(row.get("tuitionFeePeriod")) or "unspecified",
        "entrance_exams": exams,
        "min_marks": _clean(row.get("minimumMarks")),
        "description": _clean(row.get("description")),
        "fees_description": _clean(row.get("feesDescription")),
        "key_highlights": [h for h in highlights if h],
        "is_abroad": is_abroad,
        "is_top_course": _bool(row.get("isTopCourse")),
        "scholarship_available": _bool(row.get("scholarshipAvailable")),
        "created_at": _clean(row.get("createdAt")),
    }


def iter_course_rows(path: Path | None = None):
    """Stream normalised course records. Rows missing an id or a college
    reference are unusable (they can never be joined or cited) and are dropped
    with a count reported at the end."""
    path = path or csv_path()
    if not path.exists():
        raise FileNotFoundError(
            f"courses CSV not found: {path}\n"
            f"Set MME_COURSES_CSV to its location."
        )
    dropped = 0
    kept = 0
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for raw in csv.DictReader(f):
            rec = normalise_row(raw)
            if not rec["course_id"] or not rec["college_id"]:
                dropped += 1
                continue
            kept += 1
            yield rec
    if dropped:
        print(
            f"[courses_csv] {kept} rows kept, {dropped} dropped "
            f"(missing _id or college)",
            file=sys.stderr,
        )


if __name__ == "__main__":
    import collections
    import json

    levels = collections.Counter()
    colleges = set()
    n = 0
    fees = 0
    for rec in iter_course_rows():
        n += 1
        levels[rec["program_level"]] += 1
        colleges.add(rec["college_id"])
        if rec["tuition_max_inr"]:
            fees += 1
    print(
        json.dumps(
            {
                "rows": n,
                "colleges": len(colleges),
                "with_inr_tuition": fees,
                "levels": levels.most_common(),
            },
            indent=1,
        )
    )
