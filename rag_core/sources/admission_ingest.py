"""Recover the admission criteria already sitting in the detail crawl.

WHY THIS EXISTS
The detail endpoint returns an `admission` list per college — 265,567 entries
across 40,395 colleges, all of it already downloaded and none of it in the
database. It answers the question students ask most often and MIVI currently
cannot answer at all: "is meri percentage se admission ho jayega?"

WHY IT IS NOT A STRAIGHT COPY
Most of the payload is unusable, and copying it would have put visible garbage
in front of students. Measured over 15,000 detail rows:

  eligibility           83,294 values, but 59,763 of them (72%) are website
                        navigation text — "Guide for Countries Education Loan
                        for USA Education Loan for UK ..." — captured by
                        whatever scraper originally populated the source DB.
                        A student would have been told their eligibility was
                        "Education Loan for Canada".
  admissionStatus       "Open" on 265,566 of 265,567 entries.
  admissionMode         "Merit Based" on 85,814 of 85,815 entries.

Those last two are schema defaults, not observations. A field that holds the
same value for every course in the country carries no information, and
rendering it as "admissions are open" would be inventing a fact — the exact
failure this project guards against everywhere else. They are dropped.

Inside the 28% that is real there is still bad data:
  "Class 12: 450%"   x150   marks typed into a percent field
  "60", "8428"       x262   bare numbers with no subject
so every value passes a gate before it is stored.

WHAT SURVIVES
  min_marks      a qualifying percentage, with the stage it applies to
  qualification  prose criteria ("passed 10+2 from a recognised board")
  documents      the certificate list a candidate must produce
  boards         accepted boards/qualifications, from selectionCriteria

Two-tier provenance is preserved: these are the CLIENT's own records, so they
sit in their own table rather than being merged into `courses`, and they render
under their own heading. Nothing here overwrites the catalogue.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from ..store.backend import get_backend
from ..store.db import split_exams_and_boards, token_is_junk

ROOT = Path(__file__).resolve().parents[2]
DETAIL_JSONL = ROOT / "data" / "raw" / "colleges_detail.jsonl"

# Navigation and marketing text that the source scraper swept into the
# eligibility field. Matched case-insensitively against the whole value.
_BOILERPLATE = re.compile(
    r"education\s+loan|guide\s+for\s+countries|study\s+abroad"
    r"|top\s+universit|scholarship\s+guide|visa\s+guide"
    r"|admission\s+guide\s+for", re.I)

# "Class 12: 45.0%", "Graduation: 50%", "Class 10: 33%"
# PAGE-SCRAPE DETECTOR.
#
# The source DB's eligibility field does not only contain the study-abroad nav
# block; some rows hold a whole listing or FAQ page that a scraper flattened.
# Those read as plausible prose, so the boilerplate, bare-number and length
# gates all pass, and the text was stored as a per-course criterion. Measured
# live before this gate existed: 396 rows across 48 colleges, e.g.
#
#   NDIM, under "Bachelor of Computer Application":
#     "...accepts IPU CET, CUET; Graduate with 50% aggregate in any subject
#      View course details Admission Info ... BCA Admissions 2026 1 courses"
#     -> a PGDM rule under the BCA's name; a Class-12 student is told to be a
#        graduate already.
#   DPMI, on 8 courses:
#     "...Certificate Class 10 Q: Are admissions open at DPMI? A: Yes,
#      admissions at DPMI are currently open"
#     -> reinstates verbatim the "admissions are open" claim this module
#        deliberately refuses to make from admissionStatus.
#
# Interface furniture, not prose a college wrote about eligibility.
_UI_MARKER = re.compile(
    r"view\s+(?:course\s+)?details|admission\s+info\b|know\s+more"
    r"|apply\s+now|read\s+more|click\s+here|compare\s+colleges"
    r"|seat\s+intake\b|\bcourses?\s+eligibility\b|download\s+brochure"
    r"|check\s+(?:fees|eligibility)|view\s+all|show\s+more"
    r"|\d+\s+courses?\b|shortlist", re.I)
# A flattened FAQ. "Q: ... A: ..." is never how a criterion is written.
_FAQ = re.compile(r"\bq\s*[:.]\s*\w|\ba\s*[:.]\s*yes\b|frequently\s+asked", re.I)


def _is_page_scrape(v: str) -> str | None:
    """Why this value is a scraped page rather than a criterion, or None."""
    if _FAQ.search(v):
        return "flattened FAQ text, not a criterion"
    hits = {m.group(0).lower() for m in _UI_MARKER.finditer(v)}
    # One marker can appear in honest prose ("2 courses" in a sentence). Two
    # distinct ones is interface text.
    if len(hits) >= 2:
        return f"scraped page furniture ({len(hits)} UI markers)"
    return None


_MARKS = re.compile(
    r"^(class\s*\d{1,2}|graduation|bachelor'?s?|diploma|post\s*graduation)"
    r"\s*[:\-]\s*(\d{1,3}(?:\.\d+)?)\s*%?$", re.I)

# A value that is nothing but digits/punctuation says nothing about who may
# apply. "60" could be a percentage, a seat count, or a row id.
_BARE_NUMBER = re.compile(r"^[\d\s.,%/-]+$")

_DOC_HINT = re.compile(r"mark\s*sheet|certificate|transcript|migration"
                       r"|character\s+certificate|photograph", re.I)

MIN_PROSE_CHARS = 25        # shorter than this is a fragment, not a criterion
MAX_PROSE_CHARS = 1_200     # longer is a pasted page, not a criterion
MAX_PCT = 100.0             # 450% is a marks total in a percent field


def classify(value: str) -> tuple[str | None, dict | None, str]:
    """(kind, payload, reason). kind is None when the value must be dropped.

    Returning the reason for every rejection is deliberate: the counts it
    produces are the only way to tell "the field was empty" from "the gate is
    too strict", and the latter has already happened once on this project.
    """
    v = " ".join(str(value or "").split())
    if not v:
        return None, None, "empty"
    if _BOILERPLATE.search(v):
        return None, None, "website navigation text"
    scrape = _is_page_scrape(v)
    if scrape:
        return None, None, scrape
    # Prose carrying the source DB's placeholder is not a criterion — it is two
    # scraped fields run together, and the rule after the placeholder belongs to
    # whichever programme the scraper was on. Live example: "Course Name Common
    # Eligibility BPharm Class 12 ... DPh" was stored under Bachelor of
    # Physiotherapy, so a BPT applicant was shown a pharmacy requirement.
    if _JUNK_CONTAINS.search(v):
        return None, None, "placeholder-prefixed text from a field concatenation"

    m = _MARKS.match(v)
    if m:
        pct = float(m.group(2))
        if not 0 < pct <= MAX_PCT:
            return None, None, f"impossible percentage ({pct:g}%)"
        stage = " ".join(w.capitalize() for w in m.group(1).split())
        return "min_marks", {"stage": stage, "min_percent": pct}, ""

    if _BARE_NUMBER.match(v):
        return None, None, "bare number, no subject"
    if len(v) < MIN_PROSE_CHARS:
        return None, None, f"too short ({len(v)} chars)"
    if len(v) > MAX_PROSE_CHARS:
        v = v[:MAX_PROSE_CHARS].rsplit(" ", 1)[0] + "…"

    kind = "documents" if _DOC_HINT.search(v) else "qualification"
    return kind, {"text": v}, ""


# selectionCriteria mixes three unrelated things, so it cannot be stored under
# one label. Measured over 15,000 detail rows, its tokens are:
#   school boards      CBSE 12th (29,823), ISC (27,601), UP 12th, WBCHSE, AHSEC,
#                      Maharashtra HSC, RBSE, ICSE, Karnataka 2nd PUC
#   entrance exams     JEE Main (4,244), NEET PG (2,514), MHT CET, CUET, GATE
#   foreign tests      IELTS (2,148), TOEFL, PTE — abroad listings only
# Calling IELTS an "accepted board" would be a wrong fact, and "which exams does
# this college accept" is the more useful question anyway, so they are split.
# The exam/board classifier lives in store/db.py (split_exams_and_boards) and is
# shared with the card renderer, so a token cannot be classified one way when it
# is stored and another way when it is shown. It used to be duplicated here, and
# two copies of a regex this fiddly is two copies that drift.
#
# Why the split exists at all. Measured over 15,000 detail rows, selectionCriteria
# mixes three unrelated things:
#   school boards      CBSE 12th (29,823), ISC (27,601), UP 12th, WBCHSE, AHSEC,
#                      Maharashtra HSC, RBSE, ICSE, Karnataka 2nd PUC
#   entrance exams     JEE Main (4,244), NEET PG (2,514), MHT CET, CUET, GATE
#   foreign tests      IELTS (2,148), TOEFL, PTE — abroad listings only
# Calling IELTS an "accepted board" would be a wrong fact, and "which exams does
# this college accept" is the more useful question anyway.
#
# The placeholder check below stays local: it applies to PROSE, not to a token
# list, and it is the signal that a value is two scraped fields run together.
_JUNK_CONTAINS = re.compile(r"courses?(\s+name)?\s*common|common\s+course", re.I)


def _boards(value: str) -> tuple[str | None, dict | None, str]:
    """Split selectionCriteria into accepted boards vs accepted exams."""
    v = " ".join(str(value or "").split())
    if not v:
        return None, None, "empty"
    if _BOILERPLATE.search(v):
        return None, None, "website navigation text"

    if _is_page_scrape(v):
        return None, None, "scraped page furniture"

    boards, exams, unknown = [], [], 0
    for part in re.split(r"[,;/]| and ", v):
        p = part.strip()
        if not (2 <= len(p) <= 60) or token_is_junk(p):
            continue
        # One classifier, shared with the card renderer (store/db.py), so a token
        # cannot be an exam here and a board there.
        got_exams, got_boards = split_exams_and_boards([p])
        if got_exams:
            exams.extend(got_exams)
        elif got_boards:
            boards.extend(got_boards)
        else:
            # Neither. Previously this fell into `boards` by default, which
            # labelled state entrance exams and stray phrases as school boards.
            # Dropped instead: an unrecognised token is not evidence of anything,
            # and a wrong label on a real-looking value is worse than omission.
            unknown += 1

    if not boards and not exams:
        return None, None, (f"no recognisable board or exam names"
                            f"{f' ({unknown} unclassifiable)' if unknown else ''}")
    payload: dict = {}
    if boards:
        payload["boards"] = boards[:12]
    if exams:
        payload["exams"] = exams[:12]
    # One kind so a course keeps a single row, but the two lists stay distinct
    # inside it — a renderer must never present an exam as a school board.
    return "accepted_from", payload, ""


def _payload(line: str) -> dict | None:
    try:
        p = json.loads(line)
    except Exception:  # noqa: BLE001 - a torn line must not stop the pass
        return None
    p = p.get("payload") or p.get("data") or p
    if isinstance(p, dict) and isinstance(p.get("data"), dict):
        p = p["data"]
    return p if isinstance(p, dict) else None


DDL = """
CREATE TABLE IF NOT EXISTS college_admission (
    college_id    TEXT NOT NULL,
    course_title  TEXT NOT NULL DEFAULT '',
    kind          TEXT NOT NULL,
    payload       JSONB NOT NULL,
    source        TEXT NOT NULL DEFAULT 'mme_detail',
    PRIMARY KEY (college_id, course_title, kind)
);
CREATE INDEX IF NOT EXISTS college_admission_cid
    ON college_admission (college_id);
"""


def run(path: Path = DETAIL_JSONL, *, dry_run: bool = False,
        limit: int | None = None) -> dict:
    b = get_backend()
    if not dry_run:
        for stmt in filter(str.strip, DDL.split(";")):
            b.execute(stmt)

    # DB-scoped by instruction: this recovers facts for colleges we already
    # have. An admission row whose college is not in the catalogue is skipped,
    # never inserted — the crawl must not become a back door for new colleges.
    known = {r["college_id"] for r in b.q("SELECT college_id FROM colleges")}
    print(f"[admission] {len(known):,} known colleges; "
          f"reading {path.name}", file=sys.stderr)

    dropped: Counter[str] = Counter()
    kept: Counter[str] = Counter()
    rows: dict[tuple[str, str, str], dict] = {}
    seen_colleges: set[str] = set()
    unknown_college = 0
    n = 0

    with path.open(encoding="utf-8") as fh:
        for line in fh:
            n += 1
            if limit and n > limit:
                break
            p = _payload(line)
            if not p:
                continue
            for a in (p.get("admission") or []):
                if not isinstance(a, dict):
                    continue
                cid = str(a.get("college") or "").strip()
                if not cid:
                    continue
                if cid not in known:
                    unknown_college += 1
                    continue
                title = " ".join(str(a.get("selectedCourseTitle")
                                     or a.get("courseName") or "").split())
                seen_colleges.add(cid)

                for field, fn in (("eligibility", classify),
                                  ("minimumQualification", classify),
                                  ("eligibilityDetails", classify),
                                  ("requiredDocuments", classify),
                                  ("selectionCriteria", _boards)):
                    raw = a.get(field)
                    if isinstance(raw, list):
                        raw = ", ".join(str(x) for x in raw if x)
                    if not raw:
                        continue
                    kind, payload, why = fn(raw)
                    if not kind:
                        dropped[why] += 1
                        continue
                    key = (cid, title, kind)
                    # First writer wins: entries are ordered newest-first by
                    # the API, and a later duplicate is the older record.
                    if key not in rows:
                        rows[key] = payload
                        kept[kind] += 1

    print(f"[admission] scanned {n:,} detail rows", file=sys.stderr)
    print(f"[admission] kept {sum(kept.values()):,} facts across "
          f"{len(seen_colleges):,} colleges: "
          f"{', '.join(f'{k}={v:,}' for k, v in kept.most_common())}",
          file=sys.stderr)
    # Never a silent cap: every rejection is named and counted.
    for why, c in dropped.most_common():
        print(f"[admission]   dropped {c:>7,}  {why}", file=sys.stderr)
    if unknown_college:
        print(f"[admission]   skipped {unknown_college:,} rows whose college "
              f"is not in the catalogue", file=sys.stderr)

    if dry_run:
        print("[admission] dry run — nothing written", file=sys.stderr)
        for (cid, title, kind), payload in list(rows.items())[:6]:
            print(f"    {cid[:8]} {kind:14} {title[:26]:26} "
                  f"{json.dumps(payload)[:70]}", file=sys.stderr)
        return {"kept": sum(kept.values()), "colleges": len(seen_colleges),
                "dropped": sum(dropped.values()), "written": 0}

    # Clean replace, not a merge. Everything here is re-derived from the JSONL
    # on every run, so an upsert alone would strand rows that a tightened gate
    # now rejects -- the first run stored "boards: Course Name Common" for
    # thousands of courses, and an upsert would have left every one of them in
    # place while reporting success.
    removed = b.execute("DELETE FROM college_admission WHERE source = 'mme_detail'")
    if removed:
        print(f"[admission] cleared {removed:,} rows from the previous run",
              file=sys.stderr)

    # `?` placeholders, not `%s`: the backend rewrites `?` and doubles every
    # literal `%` while doing it, so a hand-written `%s` arrives as `%%s` and
    # the statement ends up with no placeholders at all.
    b.executemany(
        "INSERT INTO college_admission (college_id, course_title, kind, payload)"
        " VALUES (?, ?, ?, ?)"
        " ON CONFLICT (college_id, course_title, kind) DO UPDATE"
        " SET payload = EXCLUDED.payload",
        [(cid, title, kind, json.dumps(payload))
         for (cid, title, kind), payload in rows.items()])
    total = b.one("SELECT COUNT(*) FROM college_admission") or 0
    colleges = b.one("SELECT COUNT(DISTINCT college_id) FROM college_admission") or 0
    print(f"[admission] table now holds {total:,} facts across "
          f"{colleges:,} colleges", file=sys.stderr)
    return {"kept": sum(kept.values()), "colleges": colleges,
            "dropped": sum(dropped.values()), "written": total}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be kept and dropped, write nothing")
    ap.add_argument("--limit", type=int, help="stop after N detail rows")
    ap.add_argument("--path", type=Path, default=DETAIL_JSONL)
    a = ap.parse_args()
    run(a.path, dry_run=a.dry_run, limit=a.limit)


if __name__ == "__main__":
    main()
