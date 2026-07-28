"""Load the dataset and render each college as one retrievable 'card'.

Chunking choice: one college = one chunk. Each record is ~200 words, citations
are per-college, and questions never need sub-record granularity — so
record-level chunks are both the simplest and the most faithful unit.
"""
import csv

from .config import DATA_CSV

INT_COLS = ("annual_fees_inr", "last_year_cutoff_pct", "total_seats", "established_year")

_ROWS = None  # the dataset is immutable per process; parse the CSV once


def load_colleges():
    global _ROWS
    if _ROWS is None:
        with open(DATA_CSV, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            for c in INT_COLS:
                r[c] = int(float(r[c]))
            r["avg_placement_lpa"] = float(r["avg_placement_lpa"])
        _ROWS = rows
    return _ROWS


_STREAM_RULES = (
    ("Engineering/Technology", ("b.tech", "m.tech", "b.arch")),
    ("Polytechnic (diplomas only)", ("diploma cse", "diploma me", "diploma civil")),
    ("Management/Commerce", ("mba", "pgdm", "bba", "b.com", "m.com", "ca-foundation")),
    ("Medical", ("mbbs", "bds", "nursing")),
    ("Pharmacy", ("b.pharm", "d.pharm", "m.pharm")),
    ("Law", ("llb", "llm")),
    ("Design/Fine Arts", ("b.des", "b.f.a", "m.des")),
    ("Hospitality", ("bhm", "hospitality", "culinary")),
    ("Media/Journalism", ("bjmc", "mass comm", "film")),
    ("Computer Applications", ("bca", "mca")),
    ("Arts & Science", ("b.a;", "b.sc;", "b.sc ", "m.a",)),
)


def streams(r):
    """Derived stream tags — lets 'management colleges?'-style questions
    retrieve and answer reliably without depending on course-name recall."""
    c = r["courses_offered"].lower() + ";"
    return [name for name, keys in _STREAM_RULES if any(k in c for k in keys)]


# Support/eligibility attributes that live ONLY in the free-text About field —
# the exact weak spot for queries like "a girl from a hill district on a tight
# budget". Surfacing them as explicit tags gives retrieval a keyword handle it
# otherwise lacks (no structured column captures gender/region/income). Rules
# are conservative: a tag is added only on an unambiguous phrase in About.
_ATTR_RULES = (
    ("fee concession for female/women students",
     ("female student", "women in all branches", "girl child", "single-girl")),
    ("scholarship for students from hill districts",
     ("hill district",)),
    ("support for low-income families (need/income-based)",
     ("family income", "income threshold", "need-based", "low-income", "below two",
      "below four", "below five")),
    ("waiver for SC/ST / reserved categories",
     ("scheduled caste", "scheduled tribe", "reserved categor")),
    ("merit scholarship for top scorers",
     ("merit scholarship", "merit waiver", "top ten", "top rankers", "scoring above")),
    ("admission without an entrance exam",
     ("no entrance exam", "no written entrance", "no entrance test",
      "entry is on class 12", "admission is on class 12", "on class 12 marks")),
    ("mandatory internship / industry training",
     ("industry internship", "industrial exposure", "industrial training",
      "mandatory eight-week")),
    ("evening/part-time batches for working students",
     ("evening batch", "evening skills", "working students")),
)


def attributes(r):
    """Derived support/eligibility tags mined from the About free text."""
    about = r["about"].lower()
    return [name for name, keys in _ATTR_RULES if any(k in about for k in keys)]


def college_card(r):
    # `0` placement is spelled out here so the generator can never read it as "worst".
    placement = (
        "not reported / not applicable (see About)"
        if r["avg_placement_lpa"] == 0
        else f"{r['avg_placement_lpa']} LPA average"
    )
    attr = attributes(r)
    return (
        f"{r['name']} (id {r['college_id']}) — {r['type']} institution in {r['city']}, {r['state']}. "
        f"Streams: {', '.join(streams(r)) or 'General'}. "
        + (f"Support & eligibility: {', '.join(attr)}. " if attr else "")
        + f"Courses offered: {r['courses_offered']}. "
        f"Tuition: Rs {r['annual_fees_inr']:,} per academic YEAR "
        f"(excludes hostel, mess and any extra charges described in About). "
        f"Last-year admission cutoff: {r['last_year_cutoff_pct']}% aggregate — a hard minimum. "
        f"Total seats: {r['total_seats']}. Hostel available: {r['hostel_available']}. "
        f"NAAC grade: {r['naac_grade']}. Average placement: {placement}. "
        f"Established: {r['established_year']}. "
        f"About: {r['about']}"
    )
