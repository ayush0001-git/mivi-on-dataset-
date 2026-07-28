"""Write an enriched copy of the courses export with the college RESOLVED.

The raw export is unreadable by a human: `college` is a bare Mongo ObjectId, so
you cannot tell what "69b9016f216756163868f1a2" is without querying something.
This writes a sibling CSV with four columns inserted immediately to the right of
`college` —

    college | collegeName | collegeCity | collegeState | collegeType

— and back-fills the columns the export leaves empty.

The ORIGINAL FILE IS NEVER TOUCHED. It is the production export; this writes
`makeMyEducation.courses.enriched.csv` next to it and leaves the source
byte-identical, so a bad run costs nothing and the ETL can still read either.

Columns back-filled (all were ~0% populated in the export, measured):
  programLevel   derived from the course title, which is 100% filled
  department     derived from title + courseFullName
  durationUnit   defaulted to "year" only where durationValue is present
  currencySymbol NEW — resolved from isAbroad + the detail crawl, so a reader
                 can never mistake an overseas fee for rupees

Run:
    python -m rag_core.sources.enrich_csv                 # full file
    python -m rag_core.sources.enrich_csv --limit 5000    # smoke test
    python -m rag_core.sources.enrich_csv --out D:/x.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

from ..store.db import DB_FILE, connect
from .courses_csv import csv_path, program_level

csv.field_size_limit(10**9)

# Inserted directly to the right of `college`, in this order.
NEW_AFTER_COLLEGE = ("collegeName", "collegeCity", "collegeState", "collegeType")
EXTRA_TAIL = ("currencySymbol",)

# Department derived from the course title/full name. The export's own
# `department` column is populated on ~10 rows out of 211,511, so this is the
# only way the field becomes usable. Order matters — the first hit wins, so the
# narrow professional streams are tested before the broad "Science"/"Arts" ones
# that would otherwise swallow them (a BSc Nursing is Nursing, not Science).
_DEPT_RULES = (
    ("Nursing", (r"\bnursing\b", r"^gnm$", r"^anm$", r"\bmidwifer")),
    ("Pharmacy", (r"\bpharm", r"^dpharm$")),
    ("Medicine", (r"^mbbs$", r"\bmedicin", r"\bsurgery\b", r"^md$", r"^ms$",
                  r"\bayurved", r"\bhomoeopath", r"\bhomeopath", r"\bunani\b",
                  r"\bphysiotherap", r"\bdental\b", r"^bds$", r"^mds$")),
    ("Engineering", (r"\bengineer", r"^b\.?tech$", r"^m\.?tech$", r"^be$", r"^me$",
                     r"\btechnolog", r"\bpolytechnic\b")),
    ("Computer Applications", (r"^bca$", r"^mca$", r"computer applicat")),
    ("Management", (r"\bmanagement\b", r"^mba$", r"^bba$", r"\bpgdm\b",
                    r"business administrat")),
    ("Commerce", (r"^b\.?com$", r"^m\.?com$", r"\bcommerce\b", r"\baccountanc")),
    ("Law", (r"^ll\.?[bm]$", r"\blaw\b", r"\blegal\b")),
    ("Education", (r"^b\.?ed$", r"^m\.?ed$", r"\beducation\b", r"\bteacher train")),
    ("Architecture & Design", (r"\barchitect", r"\bdesign\b", r"^b\.?des$",
                               r"\bfine art", r"\bfashion\b", r"\binterior\b")),
    ("Hotel Management", (r"\bhotel\b", r"\bhospitality\b", r"\bculinar",
                          r"\btourism\b", r"^bhm$")),
    ("Agriculture", (r"\bagricultur", r"\bhorticultur", r"\bveterinar",
                     r"\bfisher", r"\bforestr", r"\bdairy\b")),
    ("Media & Journalism", (r"\bjournalis", r"\bmass comm", r"\bmedia\b",
                            r"\bfilm\b", r"\banimation\b")),
    ("Science", (r"^b\.?sc", r"^m\.?sc", r"\bscience\b", r"\bphysics\b",
                 r"\bchemistr", r"\bbiolog", r"\bmathemat", r"\bbiotech")),
    ("Arts & Humanities", (r"^b\.?a$", r"^m\.?a$", r"\barts\b", r"\bhistor",
                           r"\bsociolog", r"\bpsycholog", r"\bpolitical\b",
                           r"\benglish\b", r"\bhindi\b", r"\bsanskrit\b")),
    ("Vocational", (r"\bcertificat", r"\bvocational\b", r"^iti$", r"\bskill\b")),
)


# Abbreviations are a controlled vocabulary; courseFullName is free text typed
# by whoever entered the course. Where the two disagree the title wins, so a
# row like title="MBA" / courseFullName="Master of Computer Applications"
# (a real row: IIM Kashipur) lands in Management rather than being filed under
# the department its typo implies. Measured: 237 such contradictions in 211k
# rows — rare, but every one of them is a course a student could fail to find.
_DEPT_BY_TITLE = {
    "mba": "Management", "emba": "Management", "bba": "Management",
    "pgdm": "Management", "mms": "Management",
    "mca": "Computer Applications", "bca": "Computer Applications",
    "bcom": "Commerce", "mcom": "Commerce",
    "btech": "Engineering", "mtech": "Engineering", "be": "Engineering",
    "llb": "Law", "llm": "Law", "bed": "Education", "med": "Education",
    "bpharm": "Pharmacy", "mpharm": "Pharmacy", "dpharm": "Pharmacy",
    "gnm": "Nursing", "anm": "Nursing", "bsn": "Nursing",
    "mbbs": "Medicine", "bds": "Medicine", "mds": "Medicine",
    "bams": "Medicine", "bhms": "Medicine", "bpt": "Medicine",
}


def department(title: str, full_name: str) -> str:
    key = re.sub(r"[^a-z]", "", (title or "").lower())
    if key in _DEPT_BY_TITLE:
        return _DEPT_BY_TITLE[key]
    hay = f"{title} {full_name}".strip().lower()
    for dept, pats in _DEPT_RULES:
        if any(re.search(p, hay) for p in pats):
            return dept
    return ""


def load_college_lookup() -> dict[str, tuple]:
    """college_id -> (name, city, state, type). Built from the corpus, which is
    itself built from the live API — so the names written here are exactly the
    names the website shows."""
    conn = connect(DB_FILE, read_only=True)
    try:
        rows = conn.execute(
            "SELECT college_id, name, city, state, type FROM colleges"
        ).fetchall()
    finally:
        conn.close()
    return {r["college_id"]: (r["name"] or "", r["city"] or "",
                              r["state"] or "", r["type"] or "") for r in rows}


def load_currency() -> dict[str, str]:
    """course_id -> currency symbol, where the detail crawl resolved one."""
    conn = connect(DB_FILE, read_only=True)
    try:
        rows = conn.execute(
            "SELECT course_id, currency FROM courses "
            "WHERE currency IS NOT NULL AND TRIM(currency) <> ''"
        ).fetchall()
    finally:
        conn.close()
    return {r["course_id"]: r["currency"] for r in rows}


def enrich(src: Path, dst: Path, limit: int | None = None) -> dict:
    lookup = load_college_lookup()
    currency = load_currency()
    print(f"[csv] {len(lookup):,} colleges / {len(currency):,} resolved currencies",
          file=sys.stderr)

    stats = {"rows": 0, "named": 0, "unmatched": 0, "level": 0, "dept": 0, "cur": 0}
    t0 = time.perf_counter()
    unmatched_examples: list[str] = []

    with open(src, newline="", encoding="utf-8", errors="replace") as fin:
        reader = csv.DictReader(fin)
        fields = list(reader.fieldnames or [])
        if "college" not in fields:
            raise SystemExit("source CSV has no `college` column")

        out_fields = list(fields)
        at = out_fields.index("college") + 1
        for i, name in enumerate(NEW_AFTER_COLLEGE):
            if name not in out_fields:
                out_fields.insert(at + i, name)
        for name in EXTRA_TAIL:
            if name not in out_fields:
                out_fields.append(name)

        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".tmp")
        # newline="" + utf-8-sig: the people who open this file open it in
        # Excel, which needs the BOM to render Devanagari college names.
        with open(tmp, "w", newline="", encoding="utf-8-sig") as fout:
            writer = csv.DictWriter(fout, fieldnames=out_fields,
                                    extrasaction="ignore")
            writer.writeheader()
            for row in reader:
                stats["rows"] += 1
                cid = (row.get("college") or "").strip()
                info = lookup.get(cid)
                if info:
                    row["collegeName"], row["collegeCity"] = info[0], info[1]
                    row["collegeState"], row["collegeType"] = info[2], info[3]
                    stats["named"] += 1
                else:
                    # Deleted since the export. Say so in the cell rather than
                    # leaving a blank that reads as "we forgot to look it up".
                    row["collegeName"] = "(not found on makemyeducation.com)"
                    row["collegeCity"] = row["collegeState"] = row["collegeType"] = ""
                    stats["unmatched"] += 1
                    if len(unmatched_examples) < 5 and cid:
                        unmatched_examples.append(cid)

                title = (row.get("title") or "").strip()
                full = (row.get("courseFullName") or "").strip()
                if not (row.get("programLevel") or "").strip():
                    row["programLevel"] = program_level(title, full)
                    stats["level"] += 1
                if not (row.get("department") or "").strip():
                    d = department(title, full)
                    if d:
                        row["department"] = d
                        stats["dept"] += 1
                if not (row.get("durationUnit") or "").strip():
                    dv = (row.get("durationValue") or "").strip()
                    # Only where a duration actually exists — inventing "year"
                    # next to an empty value would fabricate a course length.
                    if dv and dv not in ("0", "0.0"):
                        row["durationUnit"] = "year"

                sym = currency.get((row.get("_id") or "").strip())
                if not sym:
                    sym = "$" if str(row.get("isAbroad", "")).lower() == "true" else "₹"
                else:
                    stats["cur"] += 1
                row["currencySymbol"] = sym

                writer.writerow(row)
                if stats["rows"] % 25_000 == 0:
                    print(f"[csv] {stats['rows']:,} rows…", file=sys.stderr)
                if limit and stats["rows"] >= limit:
                    break
        tmp.replace(dst)

    stats["seconds"] = round(time.perf_counter() - t0, 1)
    stats["unmatched_examples"] = unmatched_examples
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    src = a.src or csv_path()
    dst = a.out or src.with_name(src.stem.split(" (")[0] + ".enriched.csv")
    if dst.resolve() == src.resolve():
        raise SystemExit("refusing to overwrite the source export")

    print(f"[csv] {src}\n[csv] -> {dst}", file=sys.stderr)
    s = enrich(src, dst, a.limit)
    print(
        f"[csv] done in {s['seconds']}s: {s['rows']:,} rows | "
        f"{s['named']:,} named ({100 * s['named'] / max(s['rows'], 1):.2f}%) | "
        f"{s['unmatched']:,} unmatched | filled programLevel {s['level']:,}, "
        f"department {s['dept']:,}, currency-from-crawl {s['cur']:,}",
        file=sys.stderr,
    )
    if s["unmatched_examples"]:
        print(f"[csv] unmatched examples: {s['unmatched_examples']}", file=sys.stderr)


if __name__ == "__main__":
    main()
