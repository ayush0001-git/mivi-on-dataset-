"""ETL: raw crawl + course export -> the PostgreSQL corpus, published atomically.

Run:
    python -m rag_core.store.build_pg                # full rebuild
    python -m rag_core.store.build_pg --limit 20000  # fast end-to-end smoke

Replaces build_db.py. Same three inputs, same normalisation, same quality
report — the difference is that there is no second engine and no copy step: the
rows are shaped in Python and streamed straight into Postgres with COPY.

  data/raw/colleges_list.jsonl    38,700 rows — the whole catalogue, cheap.
  data/raw/colleges_detail.jsonl  partial + GROWING — the enrichment crawl
                                  writes it while this runs, so it is treated
                                  as a strictly optional overlay.
  the courses CSV                 211,511 rows / 169 MB — streamed, never held.

WHY COPY AND NOT executemany. 211k INSERT round-trips is minutes of latency for
work the server does in seconds; COPY is one stream. Measured on this corpus the
whole course load lands in ~10 s. Nothing is accumulated: the CSV is streamed,
each row is written as it is normalised, and peak RSS stays flat.

HOW THE BUILD IS ATOMIC — and why it is NOT one transaction.
Everything is built in a staging SCHEMA (default `mivi_build`) and moved onto the
serving schema in a single transaction at the very end. The serving corpus is
untouched for the whole build, so the phases in between are free to commit: a
crash leaves nothing but an orphaned staging schema, which the next run drops.
That is exactly the shape build_db.py had — many transactions into a throwaway
file, one atomic os.replace at the end.

The publish moves tables (`ALTER TABLE ... SET SCHEMA`) rather than renaming the
schema itself, which the obvious `ALTER SCHEMA public RENAME` would do more
neatly. It cannot be used here: pg_trgm and unaccent are installed IN public
(verified: pg_extension.extnamespace = 'public'), so renaming public aside and
dropping it would take the trigram operator classes with it — and CASCADE would
take every GIN index that depends on them, including the ones this build just
created. Moving tables leaves public, and its extensions, in place.

Table moves preserve the relation OID, so the indexes and the ANALYZE statistics
built during the load survive the swap: the corpus is fully planned from the
first query after publish, not after some later autovacuum.

WHAT MUST SURVIVE A REBUILD. college_web_facts and web_crawl_state are DAYS of
politeness-limited crawling of third-party sites; student_profile and
conversation_turn are the students' own memory. None of them are re-derivable
from source, so they are copied into the staging schema before the cards are
rendered (the cards quote the web facts) and merged again inside the publish
transaction, because the harvest sweep keeps writing to the serving schema while
this build runs and anything it wrote in the meantime would otherwise be lost.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from ..sources import courses_csv
from . import db

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
LIST_FILE = RAW_DIR / "colleges_list.jsonl"
DETAIL_FILE = RAW_DIR / "colleges_detail.jsonl"

SCHEMA_FILE = Path(__file__).resolve().parent / "postgres.sql"

DEFAULT_DSN = os.getenv(
    "MIVI_PG_DSN", "postgresql://postgres:postgres@localhost:5432/postgres"
)

SERVING_SCHEMA = "public"
BUILD_SCHEMA = "mivi_build"

# A --limit build is a code test, not a corpus, so it is published somewhere
# else. Swapping a 20k-course build over the served 211k one would leave every
# "how many colleges offer X" answer quietly wrong — the same failure the atomic
# publish exists to prevent.
SMOKE_SCHEMA = "mivi_smoke"

# Rows per FETCH on the two streaming re-reads (rollup, card render). Large
# enough that the round trip disappears, small enough that only ~10k rows are
# ever resident.
FETCH_SIZE = 10_000

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

# Tables that exist to build the corpus and must never be published. UNLOGGED:
# they are dropped before the swap, so paying WAL for them buys nothing. The
# published tables stay logged — SET LOGGED rewrites and WAL-logs the whole
# table, which costs back everything an UNLOGGED load saved.
HELPER_TABLES = ("_detail_raw", "_facts_raw", "_card_facts", "_agg_roll", "_cards")

# Tables carried across a rebuild instead of being re-derived.
#   key       — what the merge conflicts on.
#   fk        — scoped to colleges: a row whose college no longer exists is
#               dropped rather than carried as an orphan (Postgres enforces the
#               FK, so carrying one would abort the transaction, not skip a row).
#   live      — the serving schema is the newer writer, so the second (publish
#               time) merge lets its version overwrite the staged one. FALSE
#               means the opposite: the staged row is authoritative by then and
#               must not be clobbered.
CARRY_TABLES = (
    # Days of rate-limited crawling of 38k third-party websites. Losing this is
    # the single worst outcome of a rebuild. The sweep writes throughout the
    # build, so its rows are always the fresher ones.
    ("college_web_facts", ("college_id", "kind"), True, True),
    ("web_crawl_state", ("college_id",), True, True),
    # Just as un-re-derivable, and previously absent from this list, so an ETL
    # run would have destroyed all of it with no error:
    #   registry_fact       NIRF ranks and state Fee Regulating Authority
    #                       approved fees, matched to colleges by hand-tuned
    #                       entity resolution that took three corrections to
    #                       get right. Re-earning it means re-scraping and
    #                       re-matching every registry.
    #   registry_unmatched  the registry rows that did NOT match. Kept because
    #                       it is the only record of what the matcher missed.
    #   college_admission   148,631 admission criteria recovered from the detail
    #                       crawl, gated per value.
    #   fact_verification   the verifier's verdicts. Losing them silently
    #                       un-verifies every fact the sweep ever confirmed.
    #   ...quarantine       the rejects. Losing them loses the evidence for why
    #                       a fact was refused, and the ability to restore it.
    ("registry_fact", ("college_id", "registry", "category", "year"), True, True),
    ("registry_unmatched", ("registry", "ref_id", "category", "year"), False, True),
    ("college_admission", ("college_id", "course_title", "kind"), True, True),
    ("fact_verification", ("college_id", "kind"), True, True),
    ("college_web_facts_quarantine", ("college_id", "kind"), True, True),
    # The students' own memory — marks, budget, stream, and the transcript that
    # lets a long chat be summarised rather than truncated. Written live.
    ("student_profile", ("session_id",), False, True),
    ("conversation_turn", ("session_id", "turn_no", "role"), False, True),
    # NOT live. build_meta is the build's record of itself, written after the
    # first carry — letting the publish-time merge overwrite it republished the
    # PREVIOUS build's built_at, source_csv_sha and row counts over this one's,
    # which is exactly the drift that build_meta exists to detect. Keys only the
    # old corpus had are still carried; the new build's own keys win.
    ("build_meta", ("key",), False, False),
    # Seeded from aliases.py after the carry, so a hand-edited row survives; a
    # publish-time overwrite would undo an edit made during the build.
    ("place_alias", ("alias",), False, False),
)

# place_alias is in schema.sql (the SQLite schema) but not yet in postgres.sql,
# and postgres.sql belongs to someone else this week. Defined here so the
# published schema is complete either way: both statements are IF NOT EXISTS, so
# they become no-ops the moment postgres.sql grows its own copy.
# Why the table exists at all: measured, "Vizag" matches 0 colleges and
# "Visakhapatnam" 348, so a student typing the name everyone actually says gets
# a well-formed, grounded, completely useless "no colleges found".
GAP_DDL = (
    """CREATE TABLE IF NOT EXISTS place_alias (
           alias     TEXT PRIMARY KEY,
           canonical TEXT NOT NULL,
           kind      TEXT NOT NULL,
           source    TEXT
       )""",
    "CREATE INDEX IF NOT EXISTS idx_alias_canonical ON place_alias(canonical)",
    # The dated fee that sources/fee_promote.py writes onto colleges. Declared
    # here so the staging table has the columns the card SELECT reads.
    #
    # Deliberately NOT carried over from the serving schema: unlike the harvest,
    # this value is fully DERIVED — from registry_fact plus college_web_facts,
    # both of which ARE carried. Recomputing it after the swap cannot drift from
    # its inputs, whereas copying it could preserve a figure whose source row
    # was meanwhile corrected. See promote_verified_fees() below.
    "ALTER TABLE colleges ADD COLUMN IF NOT EXISTS verified_fee_inr BIGINT",
    "ALTER TABLE colleges ADD COLUMN IF NOT EXISTS verified_fee_period TEXT",
    "ALTER TABLE colleges ADD COLUMN IF NOT EXISTS verified_fee_year TEXT",
    "ALTER TABLE colleges ADD COLUMN IF NOT EXISTS verified_fee_basis TEXT",
    "ALTER TABLE colleges ADD COLUMN IF NOT EXISTS verified_fee_label TEXT",
    "ALTER TABLE colleges ADD COLUMN IF NOT EXISTS verified_fee_source TEXT",
    "ALTER TABLE colleges ADD COLUMN IF NOT EXISTS verified_fee_at TIMESTAMPTZ",
    # The academic year a harvested page states its figures are for, written by
    # sources/web_enrich.py. college_web_facts IS declared in postgres.sql, so
    # the clone-if-missing step below never touches it and this column would be
    # absent from staging — which crashed a build: the existence probe found the
    # column in `public` while the SELECT ran against the staging schema.
    "ALTER TABLE college_web_facts ADD COLUMN IF NOT EXISTS stated_year TEXT",
)


# ---------------------------------------------------------------------------
# scalar normalisation — unchanged from build_db.py except that the flags are
# real booleans now, because the Postgres columns are BOOLEAN rather than
# SQLite's INTEGER.
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
# Accept any plausible founding year; the junk this has to survive is a
# trailing zero-width space, not a wrong century.
_YEAR_RE = re.compile(r"(1[0-9]{3}|20[0-9]{2})")
# An ISO-ish date, optionally followed by a time. See _ts().
_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}([T ].*)?$")


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


def _flag(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    return str(v or "").strip().lower() in {"true", "1", "yes"}


def _jarr(seq) -> str | None:
    """A JSONB column, serialised. Passed as a *string* on purpose: psycopg
    adapts a Python list to a Postgres ARRAY, which a jsonb column rejects."""
    return json.dumps(seq, ensure_ascii=False) if seq else None


def _ts(v) -> str | None:
    """A TIMESTAMPTZ value, or NULL if it does not look like one.

    In SQLite this column was TEXT and took whatever the CMS held. It is a real
    timestamp now, and COPY aborts the entire stream on one unparseable value —
    so a junk `updatedAt` in a file that is still being crawled would throw away
    a build that has already read 600 MB. An absent timestamp costs nothing
    downstream; a failed build costs the whole run.
    """
    s = _text(v)
    return s if _TS_RE.match(s) else None


def _as_list(v) -> list:
    """A JSON array column, however it arrives.

    Postgres hands JSONB back already parsed, so `json.loads` on it raises. The
    string branch is kept because the same helper reads values this process
    just serialised.
    """
    if not v:
        return []
    if isinstance(v, list):
        return v
    try:
        out = json.loads(v)
    except (TypeError, ValueError):
        return []
    return out if isinstance(out, list) else []


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
# DDL handling
# ---------------------------------------------------------------------------


def _statements(script: str) -> list[str]:
    """Split a DDL script into statements.

    Splitting on ';' would be fine today and broken the first time someone puts
    a semicolon inside one of postgres.sql's 60 lines of prose commentary — a
    file several people edit. Quotes, line comments, block comments and dollar
    quotes are tracked instead.
    """
    out: list[str] = []
    buf: list[str] = []
    i, n = 0, len(script)
    quote = None  # "'", '"', a $tag$ string, or None
    while i < n:
        ch = script[i]
        if quote is None:
            if script.startswith("--", i):
                j = script.find("\n", i)
                i = n if j < 0 else j + 1
                continue
            if script.startswith("/*", i):
                j = script.find("*/", i + 2)
                i = n if j < 0 else j + 2
                continue
            if ch in "'\"":
                quote = ch
            elif ch == "$":
                m = re.match(r"\$[A-Za-z_0-9]*\$", script[i:])
                if m:
                    quote = m.group(0)
                    buf.append(quote)
                    i += len(quote)
                    continue
            elif ch == ";":
                stmt = "".join(buf).strip()
                if stmt:
                    out.append(stmt)
                buf = []
                i += 1
                continue
        elif len(quote) == 1:
            if ch == quote:
                # '' / "" is an escaped quote, not a terminator.
                if i + 1 < n and script[i + 1] == quote:
                    buf.append(ch)
                    i += 1
                else:
                    quote = None
        elif script.startswith(quote, i):
            buf.append(quote)
            i += len(quote)
            quote = None
            continue
        buf.append(ch)
        i += 1
    stmt = "".join(buf).strip()
    if stmt:
        out.append(stmt)
    return out


_INDEX_RE = re.compile(r"^CREATE\s+(UNIQUE\s+)?INDEX\b", re.IGNORECASE)
_EXT_RE = re.compile(r"^CREATE\s+EXTENSION\b", re.IGNORECASE)


def load_schema_ddl() -> tuple[list[str], list[str], list[str]]:
    """postgres.sql, split into (extensions, tables, indexes).

    The index statements are pulled out by pattern rather than by position in
    the file: they MUST run after the bulk load — maintaining a GIN index during
    a 211k-row COPY costs far more than building it once — and that ordering is
    too important to depend on a section comment staying where it is.
    """
    ext, tables, indexes = [], [], []
    for stmt in _statements(SCHEMA_FILE.read_text(encoding="utf-8")):
        if _EXT_RE.match(stmt):
            ext.append(stmt)
        elif _INDEX_RE.match(stmt):
            indexes.append(stmt)
        else:
            tables.append(stmt)
    return ext, tables, indexes


def _ident(name: str) -> sql.Identifier:
    return sql.Identifier(name)


def create_staging(conn, stage: str) -> None:
    """A fresh, empty staging schema with the serving schema's DDL applied."""
    ext, tables, indexes = load_schema_ddl()
    del indexes  # created after the load; see build()
    with conn.transaction(), conn.cursor() as cur:
        # Extensions are global by name, so these are no-ops here — but they are
        # run against public explicitly so that a machine where pg_trgm is NOT
        # yet installed gets it somewhere permanent, instead of inside a staging
        # schema that this build later drops out from under its own indexes.
        cur.execute("SET LOCAL search_path = public")
        for stmt in ext:
            cur.execute(stmt)
        # A leftover schema means a previous run died. Its contents are worthless
        # (the source files have moved on) and keeping it would silently double
        # the disk this corpus needs.
        cur.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(_ident(stage)))
        cur.execute(sql.SQL("CREATE SCHEMA {}").format(_ident(stage)))
        cur.execute(sql.SQL("SET LOCAL search_path = {}, public").format(_ident(stage)))
        for stmt in tables:
            cur.execute(stmt)
        for stmt in GAP_DDL:
            cur.execute(stmt)
        # Every carried table must EXIST in the staging schema or the carry-over
        # INSERT has nowhere to land. Cloning the live definition beats writing
        # DDL here: these tables are owned by the modules that harvest into them
        # (sources/registries.py, sources/admission_ingest.py,
        # sources/verify_facts.py), and a second hand-maintained copy of their
        # shape is a copy that drifts. INCLUDING ALL keeps the primary keys the
        # carry-over's ON CONFLICT depends on.
        for table, *_ in CARRY_TABLES:
            cur.execute("SELECT to_regclass(%s) IS NOT NULL AS live",
                        (f"public.{table}",))
            if not cur.fetchone()["live"]:
                continue        # never harvested on this machine; nothing to hold
            cur.execute("SELECT to_regclass(%s) IS NOT NULL AS staged",
                        (f"{stage}.{table}",))
            if cur.fetchone()["staged"]:
                continue        # postgres.sql already defines it
            cur.execute(sql.SQL("CREATE TABLE {}.{} (LIKE public.{} INCLUDING ALL)")
                        .format(_ident(stage), _ident(table), _ident(table)))
            print(f"[build] staged {table} by cloning the live definition",
                  file=sys.stderr)
        # The two big staging tables take their column TYPES from the real ones
        # (`AS SELECT … WHERE false`), so they cannot drift when postgres.sql
        # changes and COPY into them can never coerce a value differently from
        # the target. `AS SELECT` and not `LIKE` because LIKE also copies NOT
        # NULL — and a staging table exists precisely to hold the columns this
        # pass knows about, leaving the rest (source_level) to be filled by the
        # statement that promotes the row.
        cur.execute(f"CREATE UNLOGGED TABLE _detail_raw AS SELECT "
                    f"{', '.join(_LIST_COLS + _DETAIL_SET)} FROM colleges WHERE false")
        cur.execute("ALTER TABLE _detail_raw ADD COLUMN seq BIGINT")
        cur.execute("CREATE UNLOGGED TABLE _facts_raw ("
                    "seq BIGINT, college_id TEXT, kind TEXT, payload JSONB, "
                    "card_text TEXT)")
        cur.execute("CREATE UNLOGGED TABLE _card_facts ("
                    "college_id TEXT PRIMARY KEY, placement TEXT, scholarship TEXT, "
                    "accommodation TEXT, facility TEXT)")
        cur.execute(f"CREATE UNLOGGED TABLE _agg_roll AS SELECT "
                    f"{', '.join(c for c in _AGG_ROLL_COLS if c != 'blob')} "
                    f"FROM college_agg WHERE false")
        cur.execute("ALTER TABLE _agg_roll ADD COLUMN blob TEXT")
        cur.execute("ALTER TABLE _agg_roll ADD PRIMARY KEY (college_id)")
        cur.execute("CREATE UNLOGGED TABLE _cards ("
                    "college_id TEXT PRIMARY KEY, card TEXT)")


# ---------------------------------------------------------------------------
# pass 1 — colleges from the bulk list
# ---------------------------------------------------------------------------

_LIST_COLS = (
    "college_id", "name", "about", "details", "address", "city", "state",
    "country", "type", "establish_year", "is_abroad", "is_verified",
    "is_featured", "logo_url", "banner_url",
)

_DETAIL_SET = (
    "short_name", "district", "pincode", "naac_grade", "nirf_ranking",
    "other_ranking", "affiliated_university", "autonomous_status", "rating",
    "why_choose", "website_url", "email", "phone", "source_updated_at",
)

_DETAIL_RAW_COLS = ("seq",) + _LIST_COLS + _DETAIL_SET


def load_colleges(conn) -> tuple[set[str], dict]:
    """The catalogue. Returns the id set every later pass joins against."""
    if not LIST_FILE.exists():
        raise FileNotFoundError(
            f"{LIST_FILE} missing — run: python -m rag_core.sources.crawl_colleges list"
        )
    ids: set[str] = set()
    stats = {"rows": 0, "nameless": 0, "dupes": 0, "no_year": 0}
    copy_sql = f"COPY colleges ({', '.join(_LIST_COLS)}) FROM STDIN"

    with conn.transaction(), conn.cursor() as cur, cur.copy(copy_sql) as cp:
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
                # SQLite took INSERT OR REPLACE here, i.e. last-one-wins. COPY
                # has no conflict clause and one duplicate aborts the stream, so
                # the id set — which this pass has to build anyway — is the
                # filter, and the FIRST row wins. The list export repeats whole
                # records rather than revising them, so the two differ only in
                # which identical copy is stored.
                stats["dupes"] += 1
                continue
            year = _year(obj.get("establishYear"))
            if obj.get("establishYear") and year is None:
                stats["no_year"] += 1
            ids.add(cid)
            cp.write_row((
                cid, name, _opt(obj.get("aboutCollege")), _opt(obj.get("details")),
                _opt(obj.get("address")), _opt(obj.get("city")), _opt(obj.get("state")),
                _opt(obj.get("country")), _opt(obj.get("type")), year,
                _flag(obj.get("isAbroad")), _flag(obj.get("isVerified")),
                _flag(obj.get("isFeatured")), _opt(obj.get("logoUrl")),
                _opt(obj.get("bannerImage")),
            ))
    stats["loaded"] = len(ids)
    return ids, stats


# ---------------------------------------------------------------------------
# pass 2 — detail overlay (optional, partial, concurrently growing)
# ---------------------------------------------------------------------------

# One statement instead of 38k UPDATEs. DISTINCT ON keeps the LAST row for a
# college, which is what the per-row SQLite loop effectively did when the crawl
# had re-visited a college. The difference, worth knowing: the loop COALESCEd
# every row in turn, so a value present only in an EARLIER visit survived. Here
# it does not. Re-crawls in this corpus re-emit the whole payload, so in practice
# the newest row is a superset.
_DETAIL_UPDATE = f"""
UPDATE colleges c SET
    {', '.join(f'{col} = COALESCE(d.{col}, c.{col})' for col in _DETAIL_SET)},
    source_level = 'detail'
  FROM (SELECT DISTINCT ON (college_id) * FROM _detail_raw
         ORDER BY college_id, seq DESC) d
 WHERE d.college_id = c.college_id
"""

# A detail row for a college the list pass never saw is still fully usable — the
# detail payload is a superset of the list fields — so it is inserted rather than
# discarded. The crawl targets ids taken from the courses CSV, not from the list,
# so this genuinely happens.
_DETAIL_INSERT = f"""
INSERT INTO colleges ({', '.join(_LIST_COLS + _DETAIL_SET)}, source_level)
SELECT {', '.join('d.' + c for c in _LIST_COLS + _DETAIL_SET)}, 'detail'
  FROM (SELECT DISTINCT ON (college_id) * FROM _detail_raw
         ORDER BY college_id, seq DESC) d
 WHERE NOT EXISTS (SELECT 1 FROM colleges c WHERE c.college_id = d.college_id)
"""

# The EXISTS is not redundant with the two statements above. A fact block whose
# college has no name was skipped there, and college_facts.college_id is a real
# foreign key — one orphan would abort a transaction carrying the whole detail
# overlay, not skip a row the way SQLite's OR IGNORE did. One index probe per
# fact block is cheap insurance against a file that is being appended to while
# it is read.
_FACTS_INSERT = """
INSERT INTO college_facts (college_id, kind, payload)
SELECT DISTINCT ON (f.college_id, f.kind) f.college_id, f.kind, f.payload
  FROM _facts_raw f
 WHERE f.payload IS NOT NULL
   AND EXISTS (SELECT 1 FROM colleges c WHERE c.college_id = f.college_id)
 ORDER BY f.college_id, f.kind, f.seq DESC
"""

_CARD_FACTS_INSERT = f"""
INSERT INTO _card_facts (college_id, {', '.join(CARD_FACT_KINDS)})
SELECT college_id,
       {', '.join(f"max(card_text) FILTER (WHERE kind = '{k}')" for k in CARD_FACT_KINDS)}
  FROM (SELECT DISTINCT ON (college_id, kind) college_id, kind, card_text
          FROM _facts_raw
         WHERE card_text IS NOT NULL AND kind IN
               ({', '.join(f"'{k}'" for k in CARD_FACT_KINDS)})
         ORDER BY college_id, kind, seq DESC) t
 GROUP BY college_id
"""


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
        _ts(c.get("updatedAt")) or _ts(c.get("createdAt")),
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
    eligibilityCutoffs, …). At 35k colleges that is ~360 MB of nothing. An empty
    string is indistinguishable from an absent key downstream, so nothing is
    lost — only bytes, and here also the JSONB parse and toast-compression cost
    of writing them.
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


def overlay_details(conn, facts_conn, ids: set[str]) -> tuple[dict, dict]:
    """Fold the enrichment crawl in. Returns (stats, abroad_currency_map).

    Never fails the build if the file is absent, short, or torn: it is a
    strictly additive overlay on a corpus that is already complete enough to
    serve. Re-running the build after the crawl finishes picks up the rest.

    Two connections, because this file is 583 MB of JSON and parsing it is the
    single most expensive thing the build does — more than the 211k-row course
    load. A COPY owns the protocol of the connection it runs on, so two streams
    cannot interleave on one socket; with a second connection both the college
    rows and their fact blocks are written from ONE pass over the file instead of
    reading and re-parsing it twice.
    """
    stats = {"rows": 0, "updated": 0, "inserted": 0, "torn": 0, "facts": 0}
    abroad_currency: dict[str, str] = {}
    if not DETAIL_FILE.exists():
        print("[build] no colleges_detail.jsonl — building from the list only",
              file=sys.stderr)
        return stats, abroad_currency

    detail_sql = f"COPY _detail_raw ({', '.join(_DETAIL_RAW_COLS)}) FROM STDIN"
    facts_sql = "COPY _facts_raw (seq, college_id, kind, payload, card_text) FROM STDIN"
    torn = 0
    t0 = time.perf_counter()

    with facts_conn.transaction(), facts_conn.cursor() as fcur, \
            fcur.copy(facts_sql) as fcp:
        with conn.transaction(), conn.cursor() as cur, cur.copy(detail_sql) as cp:
            seq = 0
            for obj, torn in iter_jsonl(DETAIL_FILE, growing=True):
                c = obj.get("college")
                if not isinstance(c, dict):
                    continue
                cid = str(c.get("_id") or "").strip()
                name = _text(c.get("collegeName"))
                if not cid or not name:
                    continue
                stats["rows"] += 1
                seq += 1
                cp.write_row((
                    seq, cid, name, _opt(c.get("aboutCollege")),
                    _opt(c.get("details")), _opt(c.get("address")),
                    _opt(c.get("city")), _opt(c.get("state")),
                    _opt(c.get("country")), _opt(c.get("type")),
                    _year(c.get("establishYear")), _flag(c.get("isAbroad")),
                    _flag(c.get("isVerified")), _flag(c.get("isFeatured")),
                    _opt(c.get("logoUrl")), _opt(c.get("bannerImage")),
                    *_detail_values(c),
                ))
                if stats["rows"] % 5000 == 0:
                    print(f"[build]   detail: {stats['rows']} rows staged "
                          f"({time.perf_counter() - t0:.0f}s)", file=sys.stderr)

                # Currency for the abroad rows the CSV stripped it from. Only the
                # non-rupee symbol is worth carrying: domestic rows are set to INR
                # wholesale at insert time.
                for course in obj.get("course") or []:
                    symbol = (_text(course.get("currencySymbol"))
                              if isinstance(course, dict) else "")
                    if symbol and symbol not in {"₹", "Rs", "Rs.", "INR"}:
                        abroad_currency[cid] = symbol
                        break

                for kind in FACT_KINDS:
                    items = obj.get(kind)
                    if not isinstance(items, list) or not items:
                        continue  # empty array == no data, not an empty fact
                    items = _compact(items)
                    if not items:
                        continue  # every field in it was blank
                    stats["facts"] += 1
                    text = _fact_text(items)
                    fcp.write_row((
                        seq, cid, kind,
                        json.dumps(items, ensure_ascii=False, separators=(",", ":")),
                        text if (text and kind in CARD_FACT_KINDS) else None,
                    ))

    with conn.transaction(), conn.cursor() as cur:
        cur.execute(_DETAIL_UPDATE)
        stats["updated"] = cur.rowcount
        cur.execute(_DETAIL_INSERT)
        stats["inserted"] = cur.rowcount
        cur.execute(_FACTS_INSERT)
        stats["facts_stored"] = cur.rowcount
        cur.execute(_CARD_FACTS_INSERT)
        # The courses pass filters against this set, so ids inserted here have to
        # join it or their course rows would be dropped as unjoinable.
        cur.execute("SELECT college_id FROM colleges")
        ids.clear()
        ids.update(r["college_id"] for r in cur.fetchall())

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


def load_courses(conn, ids: set[str], abroad_currency: dict[str, str],
                 limit: int | None) -> dict:
    """Stream the CSV into `courses`, dropping rows that cannot be joined.

    An unjoinable course row is unusable *and* fatal: colleges.college_id is a
    declared foreign key that Postgres enforces during COPY, so a single orphan
    would abort the load. They are filtered against the in-memory id set (one
    hash lookup per row, no query) and counted — the count is the real join
    coverage number, so it is printed rather than hidden.
    """
    stats = {"read": 0, "inserted": 0, "orphans": 0, "dupes": 0}
    orphan_examples: list[str] = []
    # SQLite used INSERT OR IGNORE for duplicate course_ids, i.e. first-one-wins.
    # A duplicate would abort a COPY instead, so the same rule is applied here
    # with a set. Measured on this export: 211,511 rows read, 440 dropped as
    # unjoinable, 211,071 stored — zero duplicate ids. The set therefore costs
    # ~25 MB to protect against a duplicate appearing in a future export and
    # killing a build 200,000 rows in, which is a trade worth making.
    seen: set[str] = set()
    copy_sql = f"COPY courses ({', '.join(_COURSE_COLS)}) FROM STDIN"
    t0 = time.perf_counter()

    with conn.transaction(), conn.cursor() as cur, cur.copy(copy_sql) as cp:
        for rec in courses_csv.iter_course_rows():
            stats["read"] += 1
            if stats["read"] % 50_000 == 0:
                print(f"[build]   courses: {stats['read']} rows read "
                      f"({time.perf_counter() - t0:.0f}s)", file=sys.stderr)
            cid = rec["college_id"]
            if cid not in ids:
                stats["orphans"] += 1
                if len(orphan_examples) < 5:
                    orphan_examples.append(cid)
                continue
            course_id = rec["course_id"]
            if course_id in seen:
                stats["dupes"] += 1
                continue
            seen.add(course_id)
            abroad = bool(rec["is_abroad"])
            # Domestic rows are rupees by construction (courses_csv types the
            # *_inr fields only for them). Abroad rows get the symbol the detail
            # crawl actually observed, or NULL — an unlabelled foreign amount
            # must stay unlabelled rather than be guessed into a currency.
            currency = abroad_currency.get(cid) if abroad else "INR"
            cp.write_row((
                course_id, cid, rec["title"] or None, rec["full_name"] or None,
                rec["program_level"] or None, rec["department"] or None,
                rec["duration_value"], rec["duration_unit"] or None,
                rec["course_mode"] or None,
                rec["tuition_min_inr"], rec["tuition_max_inr"],
                rec["tuition_min_raw"], rec["tuition_max_raw"], currency,
                rec["hostel_min_inr"], rec["hostel_max_inr"], rec["one_time_fees"],
                rec["fee_period"] or None, _jarr(rec["entrance_exams"]),
                rec["min_marks"] or None, rec["description"] or None,
                rec["fees_description"] or None, _jarr(rec["key_highlights"]),
                abroad, bool(rec["is_top_course"]), bool(rec["scholarship_available"]),
            ))
            stats["inserted"] += 1
            if limit and stats["read"] >= limit:
                break

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

_AGG_ROLL_COLS = (
    "college_id", "n_courses", "program_levels", "course_titles", "departments",
    "entrance_exams", "min_tuition_inr", "max_tuition_inr", "min_hostel_inr",
    "has_hostel", "has_scholarship", "blob",
)

_AGG_UPDATE = """
UPDATE college_agg a SET
    n_courses = r.n_courses, program_levels = r.program_levels,
    course_titles = r.course_titles, departments = r.departments,
    entrance_exams = r.entrance_exams, min_tuition_inr = r.min_tuition_inr,
    max_tuition_inr = r.max_tuition_inr, min_hostel_inr = r.min_hostel_inr,
    has_hostel = r.has_hostel, has_scholarship = r.has_scholarship
  FROM _agg_roll r
 WHERE r.college_id = a.college_id
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


def _rollup(cid: str, rows: list[dict]) -> tuple:
    titles: Counter[str] = Counter()
    levels: Counter[str] = Counter()
    depts: Counter[str] = Counter()
    exams: dict[str, str] = {}          # lowercased -> first-seen casing
    fulls: dict[str, None] = {}
    lo = hi = hostel = None
    has_hostel = has_schol = False

    for r in rows:
        if r["title"]:
            titles[r["title"]] += 1
        if r["full_name"]:
            fulls.setdefault(r["full_name"], None)
        if r["program_level"]:
            levels[r["program_level"]] += 1
        if r["department"]:
            depts[r["department"]] += 1
        for e in _as_list(r["entrance_exams"]):
            e = str(e).strip()
            k = e.lower()
            if k and k not in exams and len(exams) < MAX_EXAMS and _is_exam(e):
                exams[k] = e
        if r["scholarship_available"]:
            has_schol = True
        # Second guard on the currency invariant: even if an abroad row ever
        # reached the DB with a rupee-typed fee, it must not enter an INR
        # aggregate that a budget filter will compare against.
        if r["is_abroad"]:
            continue
        lo = _min(lo, r["tuition_min_inr"])
        hi = _max(hi, r["tuition_max_inr"])
        if r["hostel_min_inr"] is not None or r["hostel_max_inr"] is not None:
            has_hostel = True
            hostel = _min(hostel, r["hostel_min_inr"])

    # Most-common-first: the card shows only the first 28 distinct titles, so
    # the ones the college actually runs at scale must be the ones that survive
    # truncation.
    title_list = [t for t, _ in titles.most_common()]
    blob = " ".join(title_list + list(fulls))
    return (
        cid, len(rows), _jarr([l for l, _ in levels.most_common()]),
        _jarr(title_list), _jarr([d for d, _ in depts.most_common()]),
        _jarr(list(exams.values())), lo, hi, hostel, has_hostel, has_schol, blob,
    )


def build_agg(conn, reader) -> dict:
    """One ordered scan of `courses`, grouped on the fly.

    Only a single college's rows are ever resident. The alternative — GROUP BY
    with JSON aggregation in SQL — cannot express the rollup: the exam filter
    and the most-common-first title ordering are Python, and pushing them into
    SQL would mean re-implementing _is_exam as a regex operator.

    The scan runs on a SECOND connection because a COPY takes over the protocol
    of the connection it runs on: FETCHing from a server-side cursor and writing
    rollup rows to a COPY stream cannot interleave on one socket. The reader sees
    committed staging data, which is why every pass above commits.
    """
    with conn.transaction(), conn.cursor() as cur:
        # Every college gets a row, including the ones with zero courses: the
        # card lives in college_agg and retrieval must still be able to return a
        # course-less college by name.
        cur.execute("INSERT INTO college_agg (college_id, n_courses) "
                    "SELECT college_id, 0 FROM colleges")

    n_groups = 0
    copy_sql = f"COPY _agg_roll ({', '.join(_AGG_ROLL_COLS)}) FROM STDIN"
    with conn.transaction(), conn.cursor() as wcur, wcur.copy(copy_sql) as cp:
        with reader.transaction():
            with reader.cursor(name="agg_scan", row_factory=dict_row) as rcur:
                rcur.itersize = FETCH_SIZE
                rcur.execute(_AGG_SELECT)
                group: list[dict] = []
                current: str | None = None
                for row in rcur:
                    if row["college_id"] != current:
                        if current is not None:
                            cp.write_row(_rollup(current, group))
                            n_groups += 1
                        current = row["college_id"]
                        group = []
                    group.append(row)
                if current is not None:
                    cp.write_row(_rollup(current, group))
                    n_groups += 1

    with conn.transaction(), conn.cursor() as cur:
        cur.execute(_AGG_UPDATE)
        # Hostel and scholarship are also asserted by the detail crawl,
        # independent of whether any course row happened to carry a fee for them.
        cur.execute("UPDATE college_agg SET has_hostel = TRUE WHERE college_id IN "
                    "(SELECT college_id FROM college_facts WHERE kind = 'accommodation')")
        cur.execute("UPDATE college_agg SET has_scholarship = TRUE WHERE college_id IN "
                    "(SELECT college_id FROM college_facts WHERE kind = 'scholarship')")
    return {"colleges_with_courses": n_groups}


_CARD_SELECT = """
SELECT c.college_id, c.name, c.city, c.district, c.state, c.country, c.type,
       c.establish_year, c.naac_grade, c.nirf_ranking, c.affiliated_university,
       c.is_abroad, c.about, c.details,
       c.verified_fee_inr, c.verified_fee_period, c.verified_fee_year,
       c.verified_fee_basis, c.verified_fee_label,
       a.n_courses, a.program_levels, a.course_titles, a.entrance_exams,
       a.min_tuition_inr, a.max_tuition_inr, a.min_hostel_inr, a.has_hostel,
       r.blob, x.placement, x.scholarship, x.accommodation, x.facility
  FROM colleges c
  JOIN college_agg a USING (college_id)
  LEFT JOIN _agg_roll r USING (college_id)
  LEFT JOIN _card_facts x USING (college_id)
 ORDER BY c.college_id
"""

# Verbatim from migrate_postgres.py: the weights (A name, B location, C course
# titles, D overview) are what retrieve.py's ts_rank ordering was tuned against,
# so this is a definition being moved, not rewritten.
_SEARCH_DOC = """
UPDATE college_agg a SET search_doc =
    setweight(to_tsvector('simple', unaccent(coalesce(c.name, ''))), 'A')
 || setweight(to_tsvector('simple', unaccent(
        coalesce(c.city,'') || ' ' || coalesce(c.district,'') || ' ' ||
        coalesce(c.state,''))), 'B')
 || setweight(to_tsvector('simple', unaccent(
        coalesce(a.course_titles::text, ''))), 'C')
 || setweight(to_tsvector('simple', unaccent(
        left(coalesce(c.details, c.about, ''), 4000))), 'D')
FROM colleges c WHERE c.college_id = a.college_id
"""


def render_cards(conn, reader) -> dict:
    """Store db.render_card() output per college, then build the tsvector.

    The card is rendered here, once, rather than at query time: it is the exact
    text the dense index embeds and the generator is shown, and a card computed
    in two places is a card that eventually differs in two places.
    """
    # Facts harvested from the colleges' own websites. Already carried into the
    # staging schema by carry_over_harvest() — this build must render them, or
    # the swap would publish cards that silently lost days of crawling.
    web: dict[str, dict] = {}
    with conn.cursor() as cur:
        # stated_year may not exist on a store built before it was added, and a
        # missing column must cost the harvest's dates, not the whole build.
        #
        # table_schema = current_schema() is load-bearing, not tidiness: without
        # it this probe saw the column in `public` and answered yes while the
        # SELECT below ran against the staging schema, which crashed the build
        # after 80 seconds of staging work. A cross-schema existence check is
        # always wrong in a two-schema build.
        cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.columns "
                    "WHERE table_schema = current_schema() "
                    "AND table_name='college_web_facts' "
                    "AND column_name='stated_year') AS ok")
        has_year = bool(cur.fetchone()["ok"])
        cur.execute(
            "SELECT college_id, kind, summary, confidence, fetched_at"
            + (", stated_year" if has_year else ", NULL AS stated_year") +
            " FROM college_web_facts WHERE summary IS NOT NULL AND TRIM(summary) <> ''"
        )
        for r in cur.fetchall():
            web.setdefault(r["college_id"], {})[r["kind"]] = {
                "summary": r["summary"], "confidence": r["confidence"],
                "fetched_at": r["fetched_at"],
                "stated_year": r["stated_year"]}

    # Promote the Fee Regulating Authority's approved fee onto colleges BEFORE
    # the cards are rendered, so this build's cards carry it rather than waiting
    # for the next one. carry_over_harvest() has already put registry_fact into
    # this schema, so the inputs are here.
    #
    # sources/fee_promote.py owns the full promotion, including the college-site
    # tier, which needs label classification that does not belong in SQL. Only
    # the FRA tier is repeated here, and only because it is a two-key JSONB read
    # of the same rows — small enough that keeping cards correct in one pass
    # beats avoiding the duplication. Both read registry_fact; neither invents.
    #
    # DISTINCT ON keeps one row per college: a college with an approved fee for
    # both ENGG and MBA has two, and this ORDER BY decides which.
    #
    # It MUST stay the same rule as fee_promote._session_start + its rank key,
    # because both write colleges.verified_fee_* and whichever runs last wins.
    # This clause used to be `ORDER BY college_id, amt ASC` — lowest fee of any
    # year — so a build running after a promote silently reverted 196 colleges
    # onto a rule that ignores the session, some of them onto fees fixed in
    # 2021-22. Kept identical here, in the same order and for the same reasons:
    #
    #   session DESC  the newest approved fee, reading the session as the
    #                 effective-from date the authority states
    #   amt ASC       then the lowest, since quoting a college's M.Pharmacy fee
    #                 overstates what most of its students pay
    #   category, registry   so nothing is left for heap order to decide
    #
    # substring() rather than LEFT(): year is TEXT and the adapters disagree on
    # its shape ('2024-25' vs '2024-2025'), so only the opening year compares
    # across them. NULLS LAST puts an unparseable session oldest, matching
    # fee_promote's -1, so it can never outrank a real one.
    #
    # COLLATE "C" on the name tie-break, because "same rule" has to mean the
    # same ORDER too. The database's default collation is locale-aware and
    # sorts 'Master of Computer Application' before 'MBA / PGDM' (case and
    # punctuation are ignored at the primary level); Python's `<` compares
    # codepoints and puts 'MBA / PGDM' first. Two colleges tie on both session
    # and fee, and picked a different course here than in fee_promote until
    # this collation was pinned. "C" is byte order, which is what Python does.
    try:
        with conn.transaction(), conn.cursor() as cur:
            cur.execute("""
                UPDATE colleges c SET
                    verified_fee_inr    = s.amt,
                    verified_fee_period = 'year',
                    verified_fee_year   = s.year,
                    verified_fee_basis  = 'fra_approved',
                    verified_fee_label  = s.label,
                    verified_fee_source = s.source_url,
                    verified_fee_at     = s.fetched_at
                FROM (
                    SELECT DISTINCT ON (college_id) college_id,
                        COALESCE((payload->>'approved_tuition_inr')::BIGINT,
                                 (payload->>'approved_total_inr')::BIGINT) AS amt,
                        year,
                        'Fee Regulating Authority approved tuition'
                          || COALESCE(' [' || COALESCE(payload->>'stream',
                                                       category) || ']', '') AS label,
                        source_url, fetched_at
                    FROM registry_fact
                    WHERE registry ILIKE 'fra%'
                      AND COALESCE(payload->>'approved_tuition_inr',
                                   payload->>'approved_total_inr') IS NOT NULL
                    ORDER BY college_id,
                             substring(year from '[0-9]{4}')::INT DESC NULLS LAST,
                             amt ASC,
                             category COLLATE "C", registry COLLATE "C"
                ) s
                WHERE c.college_id = s.college_id""")
            promoted = cur.rowcount
        if promoted:
            print(f"[cards] promoted an approved fee onto {promoted:,} colleges",
                  file=sys.stderr)
    except Exception as exc:  # noqa: BLE001 - no registry yet on a fresh store
        conn.rollback()
        print(f"[cards] fee promotion skipped ({str(exc)[:70]})", file=sys.stderr)

    # Registry facts (NIRF ranks, state Fee Regulating Authority approved fees)
    # and MakeMyEducation's own admission records. Both were being LOST here:
    # render_card takes them as arguments, and this build never passed either,
    # so a full rebuild silently stripped the registry lines out of every card
    # it touched -- 336 colleges' worth of matched government data, gone with no
    # error, because a card that is merely missing a paragraph still looks fine.
    registry: dict[str, list] = {}
    admission: dict[str, list] = {}
    for table, sql, sink in (
        ("registry_fact",
         "SELECT college_id, registry, category, year, payload, source_url, "
         "detail_url FROM registry_fact", registry),
        ("college_admission",
         "SELECT college_id, course_title, kind, payload FROM college_admission",
         admission),
    ):
        try:
            with conn.cursor() as cur:
                cur.execute(sql)
                for r in cur.fetchall():
                    sink.setdefault(r["college_id"], []).append(dict(r))
        except Exception as exc:  # noqa: BLE001 - table absent on a fresh store
            conn.rollback()
            print(f"[cards] {table} unavailable ({str(exc)[:60]}); "
                  f"cards will omit it", file=sys.stderr)

    n = 0
    with conn.transaction(), conn.cursor() as wcur, \
            wcur.copy("COPY _cards (college_id, card) FROM STDIN") as cp:
        with reader.transaction():
            with reader.cursor(name="card_scan", row_factory=dict_row) as rcur:
                rcur.itersize = FETCH_SIZE
                rcur.execute(_CARD_SELECT)
                for row in rcur:
                    college = dict(row)
                    agg = {k: college[k] for k in (
                        "n_courses", "program_levels", "course_titles",
                        "entrance_exams", "min_tuition_inr", "max_tuition_inr",
                        "min_hostel_inr", "has_hostel")}
                    facts = {k: college.get(k) for k in CARD_FACT_KINDS
                             if college.get(k)}
                    cid = college["college_id"]
                    cp.write_row((
                        cid,
                        db.render_card(college, agg, facts, web.get(cid),
                                       registry.get(cid), admission.get(cid)),
                    ))
                    n += 1

    with conn.transaction(), conn.cursor() as cur:
        cur.execute("UPDATE college_agg a SET card = s.card FROM _cards s "
                    "WHERE s.college_id = a.college_id")
        cur.execute(_SEARCH_DOC)
    return {"cards": n, "web_facts": sum(len(v) for v in web.values()),
            "registry_colleges": len(registry),
            "admission_colleges": len(admission)}


# ---------------------------------------------------------------------------
# carry-over
# ---------------------------------------------------------------------------


def _columns(cur, schema: str, table: str) -> list[str]:
    cur.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position",
        (schema, table),
    )
    return [r["column_name"] for r in cur.fetchall()]


def carry_over_harvest(conn, stage: str, source: str, *, merge: bool = False) -> dict:
    """Copy the not-re-derivable tables from the serving schema into the build.

    The ETL republishes a FRESH corpus, so anything not re-derived from source is
    destroyed by the swap. Catalogue tables are fine — they rebuild from the CSV
    and the crawl JSONL in minutes. These are not: college_web_facts and
    web_crawl_state are the accumulated output of a slow, politeness-limited
    sweep over tens of thousands of third-party websites, and re-earning them
    would take days. student_profile and conversation_turn belong to the
    students.

    Called twice. The first pass runs before the cards are rendered, because the
    cards quote the web facts. The second (`merge=True`) runs inside the publish
    transaction: the harvest sweep writes to the serving schema throughout this
    build, and everything it wrote after the first pass would otherwise be
    dropped on the floor by the swap.

    A fact the sweep wrote between the two passes is therefore stored but not yet
    quoted in that college's card until the next build. That is the right side of
    the trade: a card is regenerated every build, a harvested fact is not.

    The column list is intersected between the two schemas rather than hardcoded,
    so a column added on one side does not turn a carry-over into a crash — the
    worst outcome here is losing the harvest, and a column mismatch must degrade
    to "carried what both sides know about", not to "carried nothing".
    """
    moved: dict[str, int] = {}
    with conn.cursor() as cur:
        for table, key, fk, live in CARRY_TABLES:
            cur.execute("SELECT to_regclass(%s) IS NOT NULL AS ok",
                        (f"{source}.{table}",))
            if not cur.fetchone()["ok"]:
                continue  # table absent in the previous corpus — nothing to carry
            cols = [c for c in _columns(cur, stage, table)
                    if c in set(_columns(cur, source, table))]
            if not cols:
                continue
            col_sql = sql.SQL(", ").join(_ident(c) for c in cols)
            where = sql.SQL("")
            if fk:
                # An orphan would violate the foreign key and abort the whole
                # transaction, so it is filtered rather than left to fail.
                where = sql.SQL(
                    " WHERE EXISTS (SELECT 1 FROM {}.colleges c "
                    "WHERE c.college_id = s.college_id)").format(_ident(stage))
            updates = [c for c in cols if c not in key]
            conflict = (
                sql.SQL(" ON CONFLICT ({}) DO UPDATE SET {}").format(
                    sql.SQL(", ").join(_ident(k) for k in key),
                    sql.SQL(", ").join(
                        sql.SQL("{0} = EXCLUDED.{0}").format(_ident(c))
                        for c in updates),
                )
                if merge and live and updates else
                sql.SQL(" ON CONFLICT DO NOTHING")
            )
            cur.execute(
                sql.SQL("INSERT INTO {}.{} ({}) SELECT {} FROM {}.{} s{}{}").format(
                    _ident(stage), _ident(table), col_sql, col_sql,
                    _ident(source), _ident(table), where, conflict))
            moved[table] = cur.rowcount
    return {k: v for k, v in moved.items() if v}


def seed_place_aliases(conn) -> int:
    """Persist the curated city/state aliases so they are queryable from SQL.

    Kept in code as well as in the table (rag_core/store/aliases.py) so a fresh
    corpus is never missing them; carried rows that a human added by hand are
    left alone.
    """
    try:
        from .aliases import CITY_ALIASES, STATE_ALIASES
    except Exception as exc:  # noqa: BLE001 - aliases are an optimisation
        print(f"[build] place_alias not seeded: {exc}", file=sys.stderr)
        return 0
    rows = [(a, c, "city", "curated") for a, c in CITY_ALIASES.items()]
    rows += [(a, c, "state", "curated") for a, c in STATE_ALIASES.items()]
    with conn.transaction(), conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO place_alias (alias, canonical, kind, source) "
            "VALUES (%s, %s, %s, %s) ON CONFLICT (alias) DO NOTHING", rows)
    return len(rows)


# ---------------------------------------------------------------------------
# integrity + quality
# ---------------------------------------------------------------------------

_ORPHAN_CHECKS = (
    ("courses", "SELECT COUNT(*) AS n FROM courses t "
                "LEFT JOIN colleges c USING (college_id) WHERE c.college_id IS NULL"),
    ("college_agg", "SELECT COUNT(*) AS n FROM college_agg t "
                    "LEFT JOIN colleges c USING (college_id) WHERE c.college_id IS NULL"),
    ("college_facts", "SELECT COUNT(*) AS n FROM college_facts t "
                      "LEFT JOIN colleges c USING (college_id) WHERE c.college_id IS NULL"),
    ("college_web_facts", "SELECT COUNT(*) AS n FROM college_web_facts t "
                          "LEFT JOIN colleges c USING (college_id) WHERE c.college_id IS NULL"),
)


def check_integrity(conn) -> None:
    """Assert what the foreign keys should already guarantee.

    Postgres enforces the references during COPY, so this can only fail if the
    staging schema was created without them — a schema drift that would publish
    a corpus citing colleges it does not have. Loud by design: that is worse
    than a failed build.
    """
    with conn.cursor() as cur:
        for table, query in _ORPHAN_CHECKS:
            cur.execute(query)
            n = cur.fetchone()["n"]
            if n:
                raise RuntimeError(
                    f"{n} orphan row(s) in {table}: the college_id foreign key is "
                    f"missing from the staging schema")


def quality_report(conn) -> None:
    """Measure and print what the corpus does NOT know. Changes nothing.

    Every number below is a property of the live source, not of this ETL, and
    none of them may be "fixed" here: a plausibility floor on tuition would
    silently delete real Rs 2,000/year polytechnic fees along with the bad
    rows, and inventing a value is worse than reporting an absent one. The
    point is that whoever reads a superlative answer knows the shape of the
    gaps behind it.
    """
    with conn.cursor() as cur:
        def one(query: str) -> int:
            cur.execute(query)
            return cur.fetchone()["n"]

        priced = one("SELECT COUNT(*) AS n FROM courses WHERE tuition_min_inr IS NOT NULL")
        total = one("SELECT COUNT(*) AS n FROM courses WHERE NOT is_abroad")
        cheap = one("SELECT COUNT(*) AS n FROM courses "
                    "WHERE tuition_min_inr IS NOT NULL AND tuition_min_inr < 5000")
        with_courses = one("SELECT COUNT(*) AS n FROM college_agg WHERE n_courses > 0")
        with_fee = one("SELECT COUNT(*) AS n FROM college_agg "
                       "WHERE min_tuition_inr IS NOT NULL")
        schol = one("SELECT COUNT(*) AS n FROM courses WHERE scholarship_available")

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


# ---------------------------------------------------------------------------
# publish
# ---------------------------------------------------------------------------


def _staged_tables(conn, stage: str) -> list[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT tablename FROM pg_tables WHERE schemaname = %s "
                    "ORDER BY tablename", (stage,))
        return [r["tablename"] for r in cur.fetchall()
                if r["tablename"] not in HELPER_TABLES]


def _foreign_fks(conn, target: str, tables: list[str]) -> list[tuple[str, str, str]]:
    """Foreign keys pointing INTO the corpus from tables this build does not own.

    The corpus is not the only thing in the serving schema — other work adds
    tables there (a fact-resolution layer and a nightly-refresh ledger appeared
    in `public` while this build was running). Replacing `colleges` with DROP
    CASCADE would silently take any foreign key those tables declare onto it:
    no error, no data loss today, and an unenforced constraint from then on.

    Captured before the drop and re-created after the move, so the swap cannot
    quietly weaken a table it was never supposed to touch.
    """
    with conn.cursor() as cur:
        cur.execute(
            # quote_ident and not conrelid::regclass::text: regclass renders a
            # bare name when the schema happens to be on the search_path, and
            # this build's search_path points at the staging schema — an
            # unqualified name would re-add the constraint to the wrong table,
            # or to nothing.
            """SELECT quote_ident(n.nspname) || '.' || quote_ident(k.relname) AS child,
                      conname AS name, pg_get_constraintdef(pg_constraint.oid) AS def
                 FROM pg_constraint
                 JOIN pg_class k ON k.oid = conrelid
                 JOIN pg_namespace n ON n.oid = k.relnamespace
                WHERE contype = 'f'
                  AND confrelid IN (SELECT c.oid FROM pg_class c
                                      JOIN pg_namespace n ON n.oid = c.relnamespace
                                     WHERE n.nspname = %s AND c.relname = ANY(%s))
                  AND conrelid NOT IN (SELECT c.oid FROM pg_class c
                                         JOIN pg_namespace n ON n.oid = c.relnamespace
                                        WHERE n.nspname = %s AND c.relname = ANY(%s))""",
            (target, list(tables), target, list(tables)),
        )
        return [(r["child"], r["name"], r["def"]) for r in cur.fetchall()]


def publish(conn, stage: str, target: str) -> list[str]:
    """Move the staged corpus onto the serving schema in one transaction.

    Readers see the old corpus or the new one, never a mixture: the DROPs take
    an AccessExclusiveLock, so a query that arrives mid-swap waits for the commit
    and then reads the new tables. In-flight queries finish against the old ones.

    lock_timeout is set because the alternative is worse than failing: the
    harvest sweep writes continuously, and a publish that waits forever for its
    lock is a build that never ends while holding the entire new corpus in an
    uncommitted transaction. On timeout nothing changes, the staging schema is
    still there, and re-running the publish is cheap.
    """
    tables = _staged_tables(conn, stage)
    borrowed = _foreign_fks(conn, target, tables)
    with conn.transaction(), conn.cursor() as cur:
        cur.execute("SET LOCAL lock_timeout = '30s'")
        # The build itself runs with synchronous_commit off (a torn build is
        # discarded wholesale, so fsync per commit buys nothing). The publish is
        # the one transaction whose loss would matter.
        cur.execute("SET LOCAL synchronous_commit = on")

        late = carry_over_harvest(conn, stage, target, merge=True)
        if late:
            print(f"[build] merged rows written during the build: "
                  + ", ".join(f"{k}={v}" for k, v in late.items()), file=sys.stderr)

        for name in HELPER_TABLES:
            cur.execute(sql.SQL("DROP TABLE IF EXISTS {}.{}").format(
                _ident(stage), _ident(name)))
        # One DROP for all of them: dropping colleges first would cascade the
        # foreign keys off the others, and naming them together sidesteps the
        # dependency order entirely. Only the tables being replaced are named —
        # anything else in the serving schema is not this build's business.
        cur.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
            sql.SQL(", ").join(
                sql.SQL("{}.{}").format(_ident(target), _ident(t)) for t in tables)))
        for name in tables:
            cur.execute(sql.SQL("ALTER TABLE {}.{} SET SCHEMA {}").format(
                _ident(stage), _ident(name), _ident(target)))

        restored = 0
        for child, name, defn in borrowed:
            # A savepoint each: re-adding validates the child's existing rows, and
            # a row referencing a college this rebuild dropped would abort the
            # entire publish. Losing one borrowed constraint is worth reporting;
            # it is not worth refusing to ship the corpus over.
            try:
                with conn.transaction():
                    cur.execute(sql.SQL("ALTER TABLE {} ADD CONSTRAINT {} {}").format(
                        sql.SQL(child), _ident(name), sql.SQL(defn)))
                restored += 1
            except psycopg.Error as exc:
                print(f"[build] WARNING: {child}.{name} references the corpus and "
                      f"could not be restored after the swap ({str(exc).strip()[:120]}). "
                      f"That table is now unconstrained. Re-add it by hand:\n"
                      f"          ALTER TABLE {child} ADD CONSTRAINT {name} {defn};",
                      file=sys.stderr)
        if borrowed:
            print(f"[build] re-created {restored} of {len(borrowed)} foreign key(s) "
                  f"declared on the corpus by tables outside it", file=sys.stderr)

        # RESTRICT, not CASCADE: the schema must be empty by now, and if it is
        # not, something was staged that this function did not know to publish.
        # Failing here rolls the whole swap back, which is the safe direction.
        cur.execute(sql.SQL("DROP SCHEMA {} RESTRICT").format(_ident(stage)))
    return tables


def publish_smoke(conn, stage: str, smoke: str) -> None:
    """A --limit build is published to its own schema, never over the corpus."""
    with conn.transaction(), conn.cursor() as cur:
        for name in HELPER_TABLES:
            cur.execute(sql.SQL("DROP TABLE IF EXISTS {}.{}").format(
                _ident(stage), _ident(name)))
        cur.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(_ident(smoke)))
        cur.execute(sql.SQL("ALTER SCHEMA {} RENAME TO {}").format(
            _ident(stage), _ident(smoke)))


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def _connect(dsn: str, stage: str, *, writer: bool):
    conn = psycopg.connect(dsn, autocommit=True, row_factory=dict_row)
    with conn.cursor() as cur:
        cur.execute(sql.SQL("SET search_path = {}, public").format(_ident(stage)))
        # The ETL owns the box while it runs, and these are session-local.
        #   work_mem      — the rollup scan sorts 211k course rows by college_id
        #                   with no index to help it (indexes are built after the
        #                   load, on purpose), and the tsvector update is another
        #                   full pass. 4 MB spills both to disk.
        #   maintenance_work_mem — three GIN indexes are built at the end.
        cur.execute("SET work_mem = '256MB'")
        cur.execute("SET maintenance_work_mem = '512MB'")
        if writer:
            # The SQLite build ran PRAGMA synchronous=OFF for the same reason:
            # everything written here lands in a staging schema that is thrown
            # away wholesale if the build dies, so paying fsync per commit buys
            # nothing. The publish transaction turns it back on.
            cur.execute("SET synchronous_commit = off")
    if not writer:
        conn.autocommit = False
    return conn


def build(limit: int | None = None, dsn: str | None = None,
          target: str = SERVING_SCHEMA, stage: str = BUILD_SCHEMA,
          publish_it: bool = True) -> dict:
    t0 = time.perf_counter()
    dsn = dsn or DEFAULT_DSN
    csv_file = courses_csv.csv_path()
    if not csv_file.exists():
        raise FileNotFoundError(
            f"courses CSV not found: {csv_file} — set MME_COURSES_CSV")

    conn = _connect(dsn, stage, writer=True)
    reader = facts_conn = None
    try:
        create_staging(conn, stage)
        # Both extra connections are opened AFTER the staging schema exists —
        # search_path is resolved per statement, but a connection made before the
        # DROP/CREATE would be pointing at a schema that no longer exists.
        reader = _connect(dsn, stage, writer=False)
        facts_conn = _connect(dsn, stage, writer=True)

        t = time.perf_counter()
        ids, cstats = load_colleges(conn)
        print(f"[build] colleges: {cstats['loaded']} loaded "
              f"({cstats['rows']} rows, {cstats['nameless']} unnamed dropped, "
              f"{cstats['dupes']} duplicate ids, {cstats['no_year']} unparseable "
              f"establishYear) in {time.perf_counter()-t:.1f}s", file=sys.stderr)

        t = time.perf_counter()
        dstats, currency = overlay_details(conn, facts_conn, ids)
        facts_conn.close()
        facts_conn = None
        print(f"[build] detail overlay: {dstats['rows']} rows "
              f"({dstats['updated']} enriched, {dstats['inserted']} new colleges, "
              f"{dstats.get('facts_stored', 0)} fact blocks of {dstats['facts']} seen, "
              f"{dstats['torn']} torn lines, {len(currency)} foreign currencies) "
              f"in {time.perf_counter()-t:.1f}s", file=sys.stderr)

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
        carried = carry_over_harvest(conn, stage, target)
        if carried:
            print(f"[build] carried over from {target}: "
                  + ", ".join(f"{k}={v}" for k, v in carried.items()), file=sys.stderr)
        seed_place_aliases(conn)

        t = time.perf_counter()
        astats = build_agg(conn, reader)
        print(f"[build] aggregates: {astats['colleges_with_courses']} colleges with "
              f"courses in {time.perf_counter()-t:.1f}s", file=sys.stderr)

        t = time.perf_counter()
        rstats = render_cards(conn, reader)
        print(f"[build] cards + search_doc: {rstats['cards']} cards "
              f"({rstats['web_facts']} web facts rendered) in "
              f"{time.perf_counter()-t:.1f}s", file=sys.stderr)

        check_integrity(conn)

        t = time.perf_counter()
        _, _, indexes = load_schema_ddl()
        with conn.cursor() as cur:
            for stmt in indexes:
                cur.execute(stmt)
        print(f"[build] {len(indexes)} indexes built in {time.perf_counter()-t:.1f}s",
              file=sys.stderr)

        quality_report(conn)

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS n FROM colleges")
            n_colleges = cur.fetchone()["n"]
            cur.execute("SELECT COUNT(*) AS n FROM courses")
            n_courses = cur.fetchone()["n"]
            cur.execute("SELECT COUNT(*) AS n FROM college_facts")
            n_facts = cur.fetchone()["n"]
        meta = {
            "corpus_rows": ustats["read"],          # rows seen in the export
            "n_colleges": n_colleges,
            "n_courses": n_courses,
            "n_facts": n_facts,
            "detail_rows_used": dstats["rows"],
            "courses_dropped_unjoinable": ustats["orphans"],
            "source_csv": str(csv_file),
            "source_csv_sha": _source_fingerprint(csv_file),
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "build_seconds": f"{time.perf_counter()-t0:.1f}",
            "limit": str(limit or ""),
            "built_by": "rag_core.store.build_pg",
        }
        with conn.transaction(), conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO build_meta (key, value) VALUES (%s, %s) "
                "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                [(k, str(v)) for k, v in meta.items()])

        # ANALYZE before the swap, not after: a table keeps its OID when it moves
        # schema, so these statistics follow it and the corpus is fully planned
        # from the very first query after publish. Named table by table because a
        # bare ANALYZE walks every table the role can see — measured at 39 s here,
        # most of it re-analyzing the corpus this build is about to replace, and
        # it takes locks on tables the harvest sweep is writing to.
        t = time.perf_counter()
        staged = _staged_tables(conn, stage)
        with conn.cursor() as cur:
            for name in staged:
                cur.execute(sql.SQL("ANALYZE {}.{}").format(
                    _ident(stage), _ident(name)))
        print(f"[build] ANALYZE {len(staged)} tables in {time.perf_counter()-t:.1f}s",
              file=sys.stderr)

        if reader is not None:
            reader.close()
            reader = None

        if not publish_it:
            print(f"[build] --no-publish: corpus left in schema {stage}",
                  file=sys.stderr)
            meta["published_to"] = stage
        elif limit:
            publish_smoke(conn, stage, SMOKE_SCHEMA)
            meta["published_to"] = SMOKE_SCHEMA
            print(f"[build] smoke build published to schema {SMOKE_SCHEMA} — the "
                  f"served corpus in {target} is untouched", file=sys.stderr)
        else:
            tables = publish(conn, stage, target)
            meta["published_to"] = target
            print(f"[build] published {len(tables)} tables onto {target}",
                  file=sys.stderr)
    finally:
        for extra in (reader, facts_conn):
            if extra is not None:
                extra.close()
        conn.close()

    print(f"[build] OK -> {meta['published_to']} | "
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
             f"A limited build publishes to schema {SMOKE_SCHEMA}, never over "
             f"the served corpus.")
    ap.add_argument("--dsn", default=DEFAULT_DSN,
                    help="Postgres DSN (default: $MIVI_PG_DSN)")
    ap.add_argument("--target-schema", default=SERVING_SCHEMA,
                    help=f"schema the corpus is served from (default: {SERVING_SCHEMA})")
    ap.add_argument("--build-schema", default=BUILD_SCHEMA,
                    help=f"staging schema (default: {BUILD_SCHEMA})")
    ap.add_argument("--no-publish", action="store_true",
                    help="build into the staging schema and stop — for "
                         "inspecting a corpus before it goes live")
    args = ap.parse_args()
    build(limit=args.limit, dsn=args.dsn, target=args.target_schema,
          stage=args.build_schema, publish_it=not args.no_publish)


if __name__ == "__main__":
    main()
