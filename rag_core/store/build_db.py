"""ETL: raw crawl + course export -> data/mivi.db.

Run:
    python -m rag_core.store.build_db              # full rebuild (~minutes)
    python -m rag_core.store.build_db --limit 20000  # fast end-to-end smoke

Three inputs, three very different shapes:

  data/raw/colleges_list.jsonl    38,700 rows — the whole catalogue, cheap.
  data/raw/colleges_detail.jsonl  partial + GROWING — the enrichment crawl
                                  writes it while this runs, so it is treated
                                  as strictly optional overlay.
  the courses CSV                 211,511 rows / 169 MB — streamed, never held.

Design constraints that shaped every pass below:

* NOTHING IS LOADED WHOLE except the college id set (38.7k short strings,
  ~4 MB) which is needed to reject un-joinable course rows. Courses are
  streamed and inserted in batches; the aggregate pass is a single ordered
  scan grouped on the fly. Peak RSS stays flat regardless of corpus size.

* THE BUILD IS ATOMIC. Everything lands in data/mivi.build.db and is
  os.replace()d onto data/mivi.db at the very end. A crashed or killed build
  leaves the served DB untouched — a half-populated index that answers
  "cheapest BTech" from 40% of the corpus is worse than no index at all.

* FEES NEVER CROSS CURRENCIES. Course rows arrive with the rupee-typed fields
  already NULLed for isAbroad rows (see sources/courses_csv.py); the aggregate
  pass additionally refuses to fold any is_abroad row into min/max_tuition_inr.
  Two independent guards, because one wrong number here is a wrong answer to a
  student about money.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

from ..sources import courses_csv
from . import db

ROOT = db.ROOT
RAW_DIR = ROOT / "data" / "raw"
LIST_FILE = RAW_DIR / "colleges_list.jsonl"
DETAIL_FILE = RAW_DIR / "colleges_detail.jsonl"

# A --limit build is a code test, not a corpus. It is deliberately published
# somewhere else: silently swapping a 20k-course DB over the served 211k one
# would leave every "how many colleges offer X" answer quietly wrong, which is
# the same failure the atomic swap exists to prevent.
SMOKE_FILE = ROOT / "data" / "mivi.smoke.db"

# 5k rows/statement: large enough that per-statement overhead disappears, small
# enough that the bound parameter list and the WAL frame batch stay modest.
COURSE_BATCH = 5_000
AGG_BATCH = 2_000

# Cap on entrance exams kept per college. The tail is noise ("Merit", "Direct
# Admission", one-off typos) and the card only ever prints the first 10.
MAX_EXAMS = 15

# Sub-collections from the detail endpoint that are worth persisting. `course`
# is deliberately excluded: the CSV export is the authoritative course record
# and duplicating it here would create two sources of truth for fees.
FACT_KINDS = ("placement", "accommodation", "scholarship", "facility", "faq", "admission")

# Only these four are rendered into the card by db.render_card; the rest are
# stored for the answer layer to quote on demand.
CARD_FACT_KINDS = ("placement", "scholarship", "accommodation", "facility")


# ---------------------------------------------------------------------------
# scalar normalisation
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
# Accept any plausible founding year; the junk this has to survive is a
# trailing zero-width space, not a wrong century.
_YEAR_RE = re.compile(r"(1[0-9]{3}|20[0-9]{2})")


def _text(v) -> str:
    """Any API string -> plain prose.

    `aboutCollege` / `details` come back as HTML fragments (`<p>…</p>`, escaped
    entities). Those columns are read straight into the retrieval card, which
    is embedded and handed to the generator, so tags are pure token waste and
    `&amp;` shows up literally in a student-facing answer. Stripped here rather
    than in render_card so the stored column and the card can never disagree.
    """
    if v is None:
        return ""
    s = str(v)
    if "<" in s:
        s = _TAG_RE.sub(" ", s)
    if "&" in s:
        s = html.unescape(s)
    # U+200B: 678 of the 38,700 establishYear values (and some names) carry a
    # trailing zero-width space from the CMS editor.
    return _WS_RE.sub(" ", s.replace("​", "")).strip()


def _opt(v) -> str | None:
    """Empty string -> NULL. A column that is absent must read as absent, not
    as a recorded empty value, because render_card omits absent facts entirely
    and would otherwise print 'NAAC grade: .'"""
    s = _text(v)
    return s or None


def _year(v) -> int | None:
    """establishYear is a string and 678 rows have junk appended (zero-width
    space). Pull the year out rather than dropping an otherwise good college."""
    m = _YEAR_RE.search(str(v or ""))
    return int(m.group(1)) if m else None


def _flag(v) -> int:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return int(bool(v))
    return int(str(v or "").strip().lower() in {"true", "1", "yes"})


def _jarr(seq) -> str | None:
    """JSON array column, or NULL when empty — see _opt."""
    return json.dumps(seq, ensure_ascii=False) if seq else None


# ---------------------------------------------------------------------------
# raw JSONL reading
# ---------------------------------------------------------------------------


def iter_jsonl(path: Path, *, growing: bool = False):
    """Stream a JSONL file, yielding (obj, decode_failures_so_far).

    `growing=True` is the detail crawl: another process is appending to this
    file right now, so the final line is routinely half-written. That torn line
    is expected and skipped. A decode failure anywhere *else* would mean real
    corruption, so the caller reports the total and it is loud if it is > 1.
    """
    if not path.exists():
        return
    bad = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                if not growing and bad > 1:
                    raise
                continue
            yield obj, bad


# ---------------------------------------------------------------------------
# pass 1 — colleges from the bulk list
# ---------------------------------------------------------------------------

_LIST_COLS = (
    "college_id", "name", "about", "details", "address", "city", "state",
    "country", "type", "establish_year", "is_abroad", "is_verified",
    "is_featured", "logo_url", "banner_url",
)
_LIST_SQL = (
    f"INSERT OR REPLACE INTO colleges({','.join(_LIST_COLS)}) "
    f"VALUES({','.join('?' * len(_LIST_COLS))})"
)


def load_colleges(conn: sqlite3.Connection) -> tuple[set[str], dict]:
    """The catalogue. Returns the id set every later pass joins against."""
    if not LIST_FILE.exists():
        raise FileNotFoundError(
            f"{LIST_FILE} missing — run: python -m rag_core.sources.crawl_colleges list"
        )
    ids: set[str] = set()
    stats = {"rows": 0, "nameless": 0, "dupes": 0, "no_year": 0}
    batch: list[tuple] = []

    conn.execute("BEGIN")
    for obj, _ in iter_jsonl(LIST_FILE):
        stats["rows"] += 1
        cid = str(obj.get("_id") or "").strip()
        name = _text(obj.get("collegeName"))
        if not cid or not name:
            # name is NOT NULL and a nameless college is uncitable — there is
            # no page to send a student to. Dropped, counted, reported.
            stats["nameless"] += 1
            continue
        if cid in ids:
            stats["dupes"] += 1
        year = _year(obj.get("establishYear"))
        if obj.get("establishYear") and year is None:
            stats["no_year"] += 1
        ids.add(cid)
        batch.append((
            cid, name, _opt(obj.get("aboutCollege")), _opt(obj.get("details")),
            _opt(obj.get("address")), _opt(obj.get("city")), _opt(obj.get("state")),
            _opt(obj.get("country")), _opt(obj.get("type")), year,
            _flag(obj.get("isAbroad")), _flag(obj.get("isVerified")),
            _flag(obj.get("isFeatured")), _opt(obj.get("logoUrl")),
            _opt(obj.get("bannerImage")),
        ))
        if len(batch) >= COURSE_BATCH:
            conn.executemany(_LIST_SQL, batch)
            batch.clear()
    if batch:
        conn.executemany(_LIST_SQL, batch)
    conn.execute("COMMIT")
    stats["loaded"] = len(ids)
    return ids, stats


# ---------------------------------------------------------------------------
# pass 2 — detail overlay (optional, partial, concurrently growing)
# ---------------------------------------------------------------------------

_DETAIL_SET = (
    "short_name", "district", "pincode", "naac_grade", "nirf_ranking",
    "other_ranking", "affiliated_university", "autonomous_status", "rating",
    "why_choose", "website_url", "email", "phone", "source_updated_at",
)
_DETAIL_SQL = (
    "UPDATE colleges SET "
    + ", ".join(f"{c}=COALESCE(?,{c})" for c in _DETAIL_SET)
    + ", source_level='detail' WHERE college_id=?"
)

# A detail row for a college the list pass never saw is still fully usable —
# the detail payload is a superset of the list fields — so it is inserted
# rather than discarded. The crawl targets ids taken from the courses CSV, not
# from the list, so this genuinely happens.
_DETAIL_INSERT_COLS = _LIST_COLS + _DETAIL_SET + ("source_level",)
_DETAIL_INSERT_SQL = (
    f"INSERT OR REPLACE INTO colleges({','.join(_DETAIL_INSERT_COLS)}) "
    f"VALUES({','.join('?' * len(_DETAIL_INSERT_COLS))})"
)


def _detail_values(c: dict) -> list:
    contact = c.get("contactInfo") if isinstance(c.get("contactInfo"), dict) else {}
    return [
        _opt(c.get("shortName")),
        _opt(c.get("district")),
        _opt(c.get("pincode")),
        _opt(c.get("naacGrade")),
        _opt(c.get("nirfRanking")),
        _opt(c.get("otherRanking")),
        _opt(c.get("affiliatedUniversity")),
        _flag(c.get("autonomousStatus")) if c.get("autonomousStatus") is not None else None,
        _opt(c.get("rating")),
        _opt(c.get("whyChooseThisCollege")),
        _opt(contact.get("websiteUrl")),
        _opt(contact.get("email")) or _opt(contact.get("officialEmail"))
        or _opt(contact.get("admissionEmail")),
        _opt(contact.get("primaryContact")) or _opt(contact.get("admissionHelpline"))
        or _opt(contact.get("alternativeContact")),
        _opt(c.get("updatedAt")) or _opt(c.get("createdAt")),
    ]


_FACT_SKIP_KEYS = {
    "_id", "id", "college", "collegeId", "college_id", "__v", "createdAt",
    "updatedAt", "status", "isActive", "is_abroad", "isAbroad", "slug", "order",
    "courseFullNames", "cutoffDetails", "applicationSteps", "requiredDocuments",
}
_FACT_SKIP_SUBSTR = ("url", "pdf", "image", "logo", "banner", "video", "link", "brochure")
# camelCase audit columns — approvedBy/createdBy carry an internal admin
# ObjectId and approvedAt/publishedAt an ISO timestamp. Both were rendering
# into student-facing cards. Suffix match is case-sensitive on purpose so
# "format" is not mistaken for a timestamp field.
_FACT_SKIP_SUFFIX = ("At", "By", "Id", "_at", "_by", "_id")


def _fact_key_ok(key: str) -> bool:
    return (
        key not in _FACT_SKIP_KEYS
        and not key.endswith(_FACT_SKIP_SUFFIX)
        and not any(s in key.lower() for s in _FACT_SKIP_SUBSTR)
    )


def _compact(obj):
    """Recursively drop null / "" / [] / {} before persisting a fact payload.

    Measured: the raw detail sub-collections average 10.3 KB per college, and
    ~80% of that is the CMS's unfilled fields ("admissionMode": "", six empty
    eligibilityCutoffs, …). At 35k colleges that is ~360 MB of nothing in a
    store whose whole justification is being a single portable file. An empty
    string is indistinguishable from an absent key downstream, so nothing is
    lost — only bytes.
    """
    if isinstance(obj, dict):
        out = {k: _compact(v) for k, v in obj.items()}
        return {k: v for k, v in out.items() if v not in (None, "", [], {})}
    if isinstance(obj, list):
        out = [_compact(v) for v in obj]
        return [v for v in out if v not in (None, "", [], {})]
    return obj


def _pretty_key(k: str) -> str:
    s = _WS_RE.sub(" ", _CAMEL_RE.sub(" ", k).replace("_", " ").replace("-", " ")).strip()
    return s[:1].upper() + s[1:] if s else s


def _fact_text(items: list, *, max_items: int = 3, max_chars: int = 480) -> str:
    """Flatten one sub-collection into a sentence render_card can paste in.

    Deliberately schema-agnostic: the six sub-collections have six different
    shapes, coverage across 35k colleges is uneven, and the crawl that
    populates them is still running — a hand-written renderer per kind would
    be guessing at payloads that have not been observed yet. So: take the
    non-empty scalar/list fields, skip ids, timestamps and asset URLs, and
    label them from their own key names.
    """
    segs: list[str] = []
    for item in items[:max_items]:
        if isinstance(item, str):
            t = _text(item)
            if t:
                segs.append(t)
            continue
        if not isinstance(item, dict):
            continue
        for key, val in item.items():
            if not _fact_key_ok(key):
                continue
            if isinstance(val, bool) or val is None:
                continue
            if isinstance(val, (int, float)):
                out = str(val)
            elif isinstance(val, str):
                out = _text(val)
            elif isinstance(val, list):
                parts = [
                    _text(x) for x in val
                    if isinstance(x, (str, int, float)) and _text(x)
                ]
                out = ", ".join(parts[:8])
            else:
                # Nested dicts (eligibilityCutoffs and friends) are ~always all
                # empty strings in this corpus; not worth the noise.
                continue
            if not out or out.startswith("http"):
                continue
            segs.append(f"{_pretty_key(key)}: {out}")
            if sum(len(s) + 2 for s in segs) > max_chars:
                break
    text = "; ".join(segs)
    return text[:max_chars].rstrip(" ;")


def overlay_details(conn: sqlite3.Connection, ids: set[str]) -> tuple[dict, dict]:
    """Fold the enrichment crawl in. Returns (stats, abroad_currency_map).

    Never fails the build if the file is absent, short, or torn: it is a
    strictly additive overlay on a corpus that is already complete enough to
    serve. Re-running the build after the crawl finishes picks up the rest.
    """
    stats = {"rows": 0, "updated": 0, "inserted": 0, "torn": 0, "facts": 0}
    abroad_currency: dict[str, str] = {}
    if not DETAIL_FILE.exists():
        print("[build] no colleges_detail.jsonl — building from the list only",
              file=sys.stderr)
        return stats, abroad_currency

    conn.execute("BEGIN")
    torn = 0
    for obj, torn in iter_jsonl(DETAIL_FILE, growing=True):
        c = obj.get("college")
        if not isinstance(c, dict):
            continue
        cid = str(c.get("_id") or "").strip()
        name = _text(c.get("collegeName"))
        if not cid or not name:
            continue
        stats["rows"] += 1

        if cid in ids:
            conn.execute(_DETAIL_SQL, _detail_values(c) + [cid])
            stats["updated"] += 1
        else:
            conn.execute(_DETAIL_INSERT_SQL, [
                cid, name, _opt(c.get("aboutCollege")), _opt(c.get("details")),
                _opt(c.get("address")), _opt(c.get("city")), _opt(c.get("state")),
                _opt(c.get("country")), _opt(c.get("type")),
                _year(c.get("establishYear")), _flag(c.get("isAbroad")),
                _flag(c.get("isVerified")), _flag(c.get("isFeatured")),
                _opt(c.get("logoUrl")), _opt(c.get("bannerImage")),
                *_detail_values(c), "detail",
            ])
            ids.add(cid)
            stats["inserted"] += 1

        # Currency for the abroad rows the CSV stripped it from. Only the
        # non-rupee symbol is worth carrying: domestic rows are set to INR
        # wholesale at insert time.
        for course in obj.get("course") or []:
            sym = _text(course.get("currencySymbol")) if isinstance(course, dict) else ""
            if sym and sym not in {"₹", "Rs", "Rs.", "INR"}:
                abroad_currency[cid] = sym
                break

        for kind in FACT_KINDS:
            items = obj.get(kind)
            if not isinstance(items, list) or not items:
                continue  # empty array == no data, not an empty fact
            items = _compact(items)
            if not items:
                continue  # every field in it was blank
            conn.execute(
                "INSERT OR REPLACE INTO college_facts(college_id,kind,payload) "
                "VALUES(?,?,?)",
                (cid, kind, json.dumps(items, ensure_ascii=False,
                                       separators=(",", ":"))),
            )
            stats["facts"] += 1
            text = _fact_text(items)
            if text and kind in CARD_FACT_KINDS:
                conn.execute(
                    f"INSERT INTO _card_extra(college_id,{kind}) VALUES(?,?) "
                    f"ON CONFLICT(college_id) DO UPDATE SET {kind}=excluded.{kind}",
                    (cid, text),
                )
    conn.execute("COMMIT")
    stats["torn"] = torn
    return stats, abroad_currency


# ---------------------------------------------------------------------------
# pass 3 — courses (the 211k-row stream)
# ---------------------------------------------------------------------------

_COURSE_COLS = (
    "course_id", "college_id", "title", "full_name", "program_level",
    "department", "duration_value", "duration_unit", "course_mode",
    "tuition_min_inr", "tuition_max_inr", "tuition_min_raw", "tuition_max_raw",
    "currency", "hostel_min_inr", "hostel_max_inr", "one_time_fees",
    "fee_period", "entrance_exams", "min_marks", "description",
    "fees_description", "key_highlights", "is_abroad", "is_top_course",
    "scholarship_available",
)
# OR IGNORE only ever suppresses a duplicate course_id (foreign keys are
# pre-filtered below and verified by foreign_key_check at the end). The
# suppressed count is measured from total_changes and reported, not swallowed.
_COURSE_SQL = (
    f"INSERT OR IGNORE INTO courses({','.join(_COURSE_COLS)}) "
    f"VALUES({','.join('?' * len(_COURSE_COLS))})"
)


def load_courses(
    conn: sqlite3.Connection,
    ids: set[str],
    abroad_currency: dict[str, str],
    limit: int | None,
) -> dict:
    """Stream the CSV into `courses`, dropping rows that cannot be joined.

    An unjoinable course row is unusable *and* fatal: colleges.college_id is a
    declared foreign key, so a single orphan would abort the load. They are
    filtered against the in-memory id set (one hash lookup per row, no query)
    and counted — the count is the real join-coverage number, so it is printed
    rather than hidden.
    """
    stats = {"read": 0, "inserted": 0, "orphans": 0, "dupes": 0}
    orphan_examples: list[str] = []
    batch: list[tuple] = []

    def flush() -> None:
        if not batch:
            return
        before = conn.total_changes
        conn.executemany(_COURSE_SQL, batch)
        wrote = conn.total_changes - before
        stats["inserted"] += wrote
        stats["dupes"] += len(batch) - wrote
        batch.clear()

    conn.execute("BEGIN")
    for rec in courses_csv.iter_course_rows():
        stats["read"] += 1
        cid = rec["college_id"]
        if cid not in ids:
            stats["orphans"] += 1
            if len(orphan_examples) < 5:
                orphan_examples.append(cid)
            continue
        abroad = bool(rec["is_abroad"])
        # Domestic rows are rupees by construction (courses_csv types the *_inr
        # fields only for them). Abroad rows get the symbol the detail crawl
        # actually observed, or NULL — an unlabelled foreign amount must stay
        # unlabelled rather than be guessed into a currency.
        currency = abroad_currency.get(cid) if abroad else "INR"
        batch.append((
            rec["course_id"], cid, rec["title"] or None, rec["full_name"] or None,
            rec["program_level"] or None, rec["department"] or None,
            rec["duration_value"], rec["duration_unit"] or None,
            rec["course_mode"] or None,
            rec["tuition_min_inr"], rec["tuition_max_inr"],
            rec["tuition_min_raw"], rec["tuition_max_raw"], currency,
            rec["hostel_min_inr"], rec["hostel_max_inr"], rec["one_time_fees"],
            rec["fee_period"] or None, _jarr(rec["entrance_exams"]),
            rec["min_marks"] or None, rec["description"] or None,
            rec["fees_description"] or None, _jarr(rec["key_highlights"]),
            int(abroad), int(rec["is_top_course"]), int(rec["scholarship_available"]),
        ))
        if len(batch) >= COURSE_BATCH:
            flush()
        if limit and stats["read"] >= limit:
            break
    flush()
    conn.execute("COMMIT")
    stats["orphan_examples"] = orphan_examples
    return stats


# ---------------------------------------------------------------------------
# pass 4 — per-college aggregates + retrieval card
# ---------------------------------------------------------------------------

_AGG_SELECT = """
SELECT college_id, title, full_name, program_level, department, entrance_exams,
       tuition_min_inr, tuition_max_inr, hostel_min_inr, hostel_max_inr,
       scholarship_available, is_abroad
  FROM courses
 ORDER BY college_id
"""

_AGG_UPDATE = """
UPDATE college_agg SET
    n_courses=?, program_levels=?, course_titles=?, departments=?,
    entrance_exams=?, min_tuition_inr=?, max_tuition_inr=?, min_hostel_inr=?,
    has_hostel=?, has_scholarship=?
 WHERE college_id=?
"""


# The CSV's `entrance` column is scraped free text, not a controlled
# vocabulary. Measured over this corpus: 337,336 exam mentions / 2,638 distinct
# strings, of which ~4,500 mentions are not exams at all and would be printed
# to a student under "Entrance exams accepted":
#     2,197x  site navigation ("Guide for Countries Education Loan for USA …")
#      ~900x  marks cutoffs   ("Class 12: 50.0%") — that is min_marks, not an exam
#      ~700x  course names    ("B.Tech. in Computer Science and Engineering")
# Real values are short codes: CBSE 12th, ISC, JEE Main, NEET PG, GATE, SITEEE.
# Filtered here, in the rollup that feeds the card and the entrance_exam filter,
# and NOT in `courses.entrance_exams` — the per-course column stays verbatim so
# the raw layer remains a faithful copy of the export and this judgement call
# can be revisited without re-crawling.
_DEGREE = (r"bachelor|master|diploma|b\.?tech|m\.?tech|b\.?sc|m\.?sc|bba|mba"
           r"|bca|mca|b\.?com|m\.?com|b\.?pharm|ll\.?b|ll\.?m|ph\.?d")
_EXAM_REJECT = re.compile(
    r"%"                                    # a cutoff percentage, not an exam
    rf"|\b(?:{_DEGREE})\b[^,]*\b(?:in|of)\b"  # a course name
    r"|education loan|guide for|click here",  # scraped page furniture
    re.IGNORECASE,
)


def _is_exam(s: str) -> bool:
    """Reject text that cannot be an entrance exam name. Conservative: an exam
    code is short, so anything past 8 words / 48 chars is prose that leaked in."""
    return 2 <= len(s) <= 48 and len(s.split()) <= 8 and not _EXAM_REJECT.search(s)


def _min(cur, val):
    return val if cur is None or (val is not None and val < cur) else cur


def _max(cur, val):
    return val if cur is None or (val is not None and val > cur) else cur


def _rollup(cid: str, rows: list[sqlite3.Row]) -> tuple:
    titles: Counter[str] = Counter()
    levels: Counter[str] = Counter()
    depts: Counter[str] = Counter()
    exams: dict[str, str] = {}          # lowercased -> first-seen casing
    fulls: dict[str, None] = {}
    lo = hi = hostel = None
    has_hostel = has_schol = 0

    for r in rows:
        if r["title"]:
            titles[r["title"]] += 1
        if r["full_name"]:
            fulls.setdefault(r["full_name"], None)
        if r["program_level"]:
            levels[r["program_level"]] += 1
        if r["department"]:
            depts[r["department"]] += 1
        for e in json.loads(r["entrance_exams"]) if r["entrance_exams"] else ():
            e = e.strip()
            k = e.lower()
            if k and k not in exams and len(exams) < MAX_EXAMS and _is_exam(e):
                exams[k] = e
        if r["scholarship_available"]:
            has_schol = 1
        # Second guard on the currency invariant: even if an abroad row ever
        # reached the DB with a rupee-typed fee, it must not enter an INR
        # aggregate that a budget filter will compare against.
        if r["is_abroad"]:
            continue
        lo = _min(lo, r["tuition_min_inr"])
        hi = _max(hi, r["tuition_max_inr"])
        if r["hostel_min_inr"] is not None or r["hostel_max_inr"] is not None:
            has_hostel = 1
            hostel = _min(hostel, r["hostel_min_inr"])

    # Most-common-first: the card shows only the first 28 distinct titles, so
    # the ones the college actually runs at scale must be the ones that survive
    # truncation.
    title_list = [t for t, _ in titles.most_common()]
    blob = " ".join(title_list + list(fulls))
    return (
        (len(rows), _jarr([l for l, _ in levels.most_common()]), _jarr(title_list),
         _jarr([d for d, _ in depts.most_common()]), _jarr(list(exams.values())),
         lo, hi, hostel, has_hostel, has_schol, cid),
        (cid, blob),
    )


def build_agg(conn: sqlite3.Connection) -> dict:
    """One ordered scan of `courses`, grouped on the fly.

    Ordering by college_id lets the index drive the scan and means only a
    single college's rows are ever resident — the alternative (GROUP BY with
    JSON aggregation, or a Python dict keyed by college) would hold 35k
    accumulators plus every distinct title in memory at once.
    """
    conn.execute("BEGIN")
    # Every college gets a row, including the ones with zero courses: the card
    # lives in college_agg and retrieval must still be able to return a
    # course-less college by name.
    conn.execute("INSERT OR REPLACE INTO college_agg(college_id,n_courses) "
                 "SELECT college_id,0 FROM colleges")
    conn.execute("COMMIT")

    cur = conn.execute(_AGG_SELECT)
    updates: list[tuple] = []
    blobs: list[tuple] = []
    group: list[sqlite3.Row] = []
    current: str | None = None
    n_groups = 0

    conn.execute("BEGIN")

    def close_group() -> None:
        nonlocal n_groups
        if current is None:
            return
        upd, blob = _rollup(current, group)
        updates.append(upd)
        blobs.append(blob)
        n_groups += 1
        if len(updates) >= AGG_BATCH:
            conn.executemany(_AGG_UPDATE, updates)
            conn.executemany(
                "INSERT INTO _card_extra(college_id,blob) VALUES(?,?) "
                "ON CONFLICT(college_id) DO UPDATE SET blob=excluded.blob", blobs)
            updates.clear()
            blobs.clear()

    for row in cur:
        if row["college_id"] != current:
            close_group()
            current = row["college_id"]
            group = []
        group.append(row)
    close_group()
    if updates:
        conn.executemany(_AGG_UPDATE, updates)
        conn.executemany(
            "INSERT INTO _card_extra(college_id,blob) VALUES(?,?) "
            "ON CONFLICT(college_id) DO UPDATE SET blob=excluded.blob", blobs)

    # Hostel and scholarship are also asserted by the detail crawl, independent
    # of whether any course row happened to carry a fee for them.
    conn.execute("UPDATE college_agg SET has_hostel=1 WHERE college_id IN "
                 "(SELECT college_id FROM college_facts WHERE kind='accommodation')")
    conn.execute("UPDATE college_agg SET has_scholarship=1 WHERE college_id IN "
                 "(SELECT college_id FROM college_facts WHERE kind='scholarship')")
    conn.execute("COMMIT")
    return {"colleges_with_courses": n_groups}


_CARD_SELECT = """
SELECT c.college_id, c.name, c.city, c.district, c.state, c.country, c.type,
       c.establish_year, c.naac_grade, c.nirf_ranking, c.affiliated_university,
       c.is_abroad, c.about, c.details,
       a.n_courses, a.program_levels, a.course_titles, a.entrance_exams,
       a.min_tuition_inr, a.max_tuition_inr, a.min_hostel_inr, a.has_hostel,
       x.blob, x.placement, x.scholarship, x.accommodation, x.facility
  FROM colleges c
  JOIN college_agg a USING(college_id)
  LEFT JOIN _card_extra x USING(college_id)
 ORDER BY c.college_id
"""


def render_cards(conn: sqlite3.Connection) -> dict:
    """Store db.render_card() output per college and populate the FTS index.

    The card is rendered here, once, rather than at query time: it is the exact
    text the dense index embeds and the generator is shown, and a card computed
    in two places is a card that eventually differs in two places.
    """
    # Facts harvested from the colleges' own websites, loaded once up front.
    # This survives a rebuild because web_enrich writes into its own table and
    # nothing here deletes it — the sweep is slow and rate-limited, so throwing
    # it away on every ETL run would mean it never accumulates.
    web: dict[str, dict] = {}
    try:
        for r in conn.execute(
            "SELECT college_id, kind, summary, confidence, fetched_at "
            "FROM college_web_facts WHERE summary IS NOT NULL AND TRIM(summary) <> ''"
        ):
            web.setdefault(r["college_id"], {})[r["kind"]] = {
                "summary": r["summary"], "confidence": r["confidence"],
                "fetched_at": r["fetched_at"]}
    except sqlite3.OperationalError:
        pass  # table predates this build; nothing harvested yet

    conn.execute("BEGIN")
    conn.execute("DELETE FROM colleges_fts")
    cards: list[tuple] = []
    fts: list[tuple] = []
    n = 0
    for row in conn.execute(_CARD_SELECT):
        college = dict(row)
        agg = {k: college[k] for k in (
            "n_courses", "program_levels", "course_titles", "entrance_exams",
            "min_tuition_inr", "max_tuition_inr", "min_hostel_inr", "has_hostel")}
        facts = {k: college.get(k) for k in CARD_FACT_KINDS if college.get(k)}
        card = db.render_card(college, agg, facts, web.get(college["college_id"]))
        cards.append((card, college["college_id"]))
        location = ", ".join(dict.fromkeys(
            p for p in (college["city"], college["district"], college["state"],
                        college["country"]) if p))
        fts.append((college["college_id"], college["name"], location,
                    college["blob"] or "", college["details"] or college["about"] or ""))
        n += 1
        if len(cards) >= AGG_BATCH:
            conn.executemany("UPDATE college_agg SET card=? WHERE college_id=?", cards)
            conn.executemany(
                "INSERT INTO colleges_fts(college_id,name,location,course_blob,about) "
                "VALUES(?,?,?,?,?)", fts)
            cards.clear()
            fts.clear()
    if cards:
        conn.executemany("UPDATE college_agg SET card=? WHERE college_id=?", cards)
        conn.executemany(
            "INSERT INTO colleges_fts(college_id,name,location,course_blob,about) "
            "VALUES(?,?,?,?,?)", fts)
    conn.execute("COMMIT")
    conn.execute("INSERT INTO colleges_fts(colleges_fts) VALUES('optimize')")
    return {"cards": n}


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def quality_report(conn: sqlite3.Connection) -> None:
    """Measure and print what the corpus does NOT know. Changes nothing.

    Every number below is a property of the live source, not of this ETL, and
    none of them may be "fixed" here: a plausibility floor on tuition would
    silently delete real Rs 2,000/year polytechnic fees along with the bad
    rows, and inventing a value is worse than reporting an absent one. The
    point is that whoever reads a superlative answer knows the shape of the
    gaps behind it.
    """
    one = lambda sql: conn.execute(sql).fetchone()[0]  # noqa: E731
    priced = one("SELECT COUNT(*) FROM courses WHERE tuition_min_inr IS NOT NULL")
    total = one("SELECT COUNT(*) FROM courses WHERE is_abroad=0")
    cheap = one("SELECT COUNT(*) FROM courses "
                "WHERE tuition_min_inr IS NOT NULL AND tuition_min_inr < 5000")
    with_courses = one("SELECT COUNT(*) FROM college_agg WHERE n_courses > 0")
    with_fee = one("SELECT COUNT(*) FROM college_agg WHERE min_tuition_inr IS NOT NULL")
    schol = one("SELECT COUNT(*) FROM courses WHERE scholarship_available=1")

    print(
        f"[build] data quality:\n"
        f"          tuition recorded on {priced}/{total} domestic courses "
        f"({100 * priced / max(total, 1):.0f}%) — the rest are 0 in the source, "
        f"which means 'not recorded', not free\n"
        f"          {with_fee}/{with_courses} colleges have any INR fee range\n"
        f"          {cheap} priced courses are under Rs 5,000/yr — a mix of real "
        f"government-polytechnic fees and source data-entry errors "
        f"(observed: a BTech listed at Rs 2,040). NOT clamped: a plausibility "
        f"floor would delete the real ones too\n"
        f"          scholarship_available is true on {schol} of 211k course rows, "
        f"so has_scholarship comes from the detail crawl alone",
        file=sys.stderr,
    )


def _source_fingerprint(path: Path) -> str:
    """Cheap stand-in for a content hash: size + mtime.

    Deliberately NOT a SHA-256 — that means re-reading 169 MB on every build
    for a value whose only job is to let the server notice that the vector
    index and the DB were built from different snapshots. Size+mtime changes
    whenever the export is replaced, which is the case that matters.
    """
    st = path.stat()
    return f"{st.st_size}-{int(st.st_mtime)}"


def _fresh_build_db(build_file: Path) -> sqlite3.Connection:
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(build_file) + suffix)
        if p.exists():
            p.unlink()
    conn = db.connect(build_file)
    # Explicit transaction control: this is a bulk loader, and the implicit
    # per-statement transactions of the default isolation level would turn one
    # 5k-row executemany into 5k WAL commits.
    conn.isolation_level = None
    db.init_schema(conn)
    # Rebuildable derived store on a throwaway file — a torn build is discarded
    # wholesale, so paying for fsync per commit buys nothing.
    conn.execute("PRAGMA synchronous=OFF")
    # 211k FK checks against colleges are pure overhead when every course row is
    # already filtered against the in-memory id set; foreign_key_check below is
    # the real gate and runs once.
    conn.execute("PRAGMA foreign_keys=OFF")
    # Overrides db.connect's temp_store=MEMORY: the serving connections' temp
    # usage is trivial, this builder's staging table is 35k rows of prose.
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute("PRAGMA cache_size=-262144")  # 256 MB — the ETL owns the box
    conn.execute("""CREATE TEMP TABLE _card_extra(
        college_id TEXT PRIMARY KEY, blob TEXT, placement TEXT,
        scholarship TEXT, accommodation TEXT, facility TEXT)""")
    return conn


def carry_over_harvest(conn: sqlite3.Connection, target: Path) -> int:
    """Copy web-harvested facts + crawl state from the live DB into the build.

    The ETL republishes a FRESH database, so anything not re-derived from source
    is destroyed by the swap. Catalogue tables are fine — they rebuild from the
    CSV and the crawl JSONL in seconds. `college_web_facts` and
    `web_crawl_state` are NOT: they are the accumulated output of a slow,
    politeness-limited sweep over tens of thousands of third-party websites,
    and re-earning them would take days. They have to be carried across.

    Rows whose college no longer exists are dropped by the INSERT's foreign key
    rather than being carried as orphans.
    """
    if not target.exists():
        return 0
    moved = 0
    conn.execute("ATTACH DATABASE ? AS prev", (str(target),))
    try:
        for table in ("college_web_facts", "web_crawl_state"):
            try:
                conn.execute(
                    f"INSERT OR REPLACE INTO {table} "
                    f"SELECT p.* FROM prev.{table} p "
                    f"WHERE p.college_id IN (SELECT college_id FROM colleges)"
                )
                moved += conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            except sqlite3.OperationalError:
                pass  # table absent in the previous build — nothing to carry
        conn.commit()
    finally:
        conn.execute("DETACH DATABASE prev")
    return moved


def _swap_into_place(conn: sqlite3.Connection, build_file: Path, target: Path) -> None:
    """Publish the build atomically."""
    # Checkpoint + close first: the build's own -wal/-shm must be folded into
    # the file being published, or the swap ships a half-written page image.
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()
    for suffix in ("-wal", "-shm"):
        p = Path(str(build_file) + suffix)
        if p.exists():
            p.unlink()

    try:
        os.replace(build_file, target)
    except PermissionError as exc:  # Windows: a reader holds the target open
        raise RuntimeError(
            f"could not replace {target} — a process still has it open. "
            f"The new build is intact at {build_file}; stop the reader and "
            f"move it into place."
        ) from exc

    # Only now: the OLD database's sidecars describe pages that no longer
    # exist. Replaying a stale -wal over the new file would corrupt it. Done
    # AFTER the replace, never before — a swap that fails must leave the
    # currently-served database completely untouched.
    for suffix in ("-wal", "-shm"):
        p = Path(str(target) + suffix)
        try:
            if p.exists():
                p.unlink()
        except OSError as exc:
            print(f"[build] WARNING: stale {p.name} could not be removed ({exc}). "
                  f"Delete it before serving — SQLite will otherwise replay it "
                  f"over the new corpus.", file=sys.stderr)


def build(limit: int | None = None, out: Path | None = None) -> dict:
    t0 = time.perf_counter()
    csv_file = courses_csv.csv_path()
    if not csv_file.exists():
        raise FileNotFoundError(
            f"courses CSV not found: {csv_file} — set MME_COURSES_CSV")

    target = out or (SMOKE_FILE if limit else db.DB_FILE)
    # Staging path is derived from the target so a smoke build can never stomp
    # on a real build's half-written file, and vice versa.
    build_file = target.with_suffix(".build.db")

    conn = _fresh_build_db(build_file)
    try:
        t = time.perf_counter()
        ids, cstats = load_colleges(conn)
        print(f"[build] colleges: {cstats['loaded']} loaded "
              f"({cstats['rows']} rows, {cstats['nameless']} unnamed dropped, "
              f"{cstats['dupes']} duplicate ids, {cstats['no_year']} unparseable "
              f"establishYear) in {time.perf_counter()-t:.1f}s", file=sys.stderr)

        t = time.perf_counter()
        dstats, currency = overlay_details(conn, ids)
        print(f"[build] detail overlay: {dstats['rows']} rows "
              f"({dstats['updated']} enriched, {dstats['inserted']} new colleges, "
              f"{dstats['facts']} fact blocks, {dstats['torn']} torn lines, "
              f"{len(currency)} foreign currencies) in {time.perf_counter()-t:.1f}s",
              file=sys.stderr)

        t = time.perf_counter()
        ustats = load_courses(conn, ids, currency, limit)
        print(f"[build] courses: {ustats['inserted']} inserted of {ustats['read']} read "
              f"— {ustats['orphans']} dropped (college_id not in catalogue), "
              f"{ustats['dupes']} duplicate course_ids "
              f"in {time.perf_counter()-t:.1f}s", file=sys.stderr)
        if ustats["orphan_examples"]:
            print(f"[build]   orphan examples: {ustats['orphan_examples']}",
                  file=sys.stderr)

        # Before aggregates/cards: the carried-over web facts are rendered into
        # the cards below, so they have to be present first.
        carried = carry_over_harvest(conn, target)
        if carried:
            print(f"[build] carried over {carried} web-harvest rows from the "
                  f"previous database", file=sys.stderr)

        t = time.perf_counter()
        astats = build_agg(conn)
        print(f"[build] aggregates: {astats['colleges_with_courses']} colleges with "
              f"courses in {time.perf_counter()-t:.1f}s", file=sys.stderr)

        t = time.perf_counter()
        rstats = render_cards(conn)
        print(f"[build] cards + FTS: {rstats['cards']} in "
              f"{time.perf_counter()-t:.1f}s", file=sys.stderr)

        # The one place FK integrity is actually asserted. Loud by design: a
        # violation means the orphan filter above is broken, and a corpus that
        # cites a college it does not have is worse than a failed build.
        bad = conn.execute("PRAGMA foreign_key_check").fetchall()
        if bad:
            raise RuntimeError(f"foreign key violations after load: {len(bad)} "
                               f"(first: {tuple(bad[0])})")

        quality_report(conn)

        n_colleges = conn.execute("SELECT COUNT(*) FROM colleges").fetchone()[0]
        n_courses = conn.execute("SELECT COUNT(*) FROM courses").fetchone()[0]
        meta = {
            "corpus_rows": ustats["read"],          # rows seen in the export
            "n_colleges": n_colleges,
            "n_courses": n_courses,
            "n_facts": conn.execute("SELECT COUNT(*) FROM college_facts").fetchone()[0],
            "detail_rows_used": dstats["rows"],
            "courses_dropped_unjoinable": ustats["orphans"],
            "source_csv": str(csv_file),
            "source_csv_sha": _source_fingerprint(csv_file),
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "build_seconds": f"{time.perf_counter()-t0:.1f}",
            "limit": str(limit or ""),
        }
        conn.execute("BEGIN")
        for k, v in meta.items():
            db.set_meta(conn, k, str(v))
        conn.execute("COMMIT")
        conn.execute("ANALYZE")
    except BaseException:
        conn.close()
        raise
    _swap_into_place(conn, build_file, target)

    size_mb = target.stat().st_size / 1e6
    print(f"[build] OK -> {target} ({size_mb:.0f} MB) | "
          f"{meta['n_colleges']} colleges, {meta['n_courses']} courses, "
          f"{meta['n_facts']} fact blocks | {meta['build_seconds']}s",
          file=sys.stderr)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--limit", type=int, default=None,
        help="cap course rows read from the CSV (smoke build). Colleges are "
             "always loaded in full — they are cheap; the 169 MB CSV is not. "
             f"A limited build publishes to {SMOKE_FILE.name}, never over the "
             f"served corpus.")
    ap.add_argument("--out", type=Path, default=None,
                    help="override the published path (default: data/mivi.db)")
    args = ap.parse_args()
    build(limit=args.limit, out=args.out)


if __name__ == "__main__":
    main()
