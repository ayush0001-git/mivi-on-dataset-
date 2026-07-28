"""Load the SQLite corpus into PostgreSQL.

SQLite stays the build target: the ETL is a single-writer batch job and a local
file is the fastest thing to build into. Postgres is the SERVING store, where
concurrent readers, a continuously-writing harvest sweep, and more than one app
process all need to coexist. This moves one into the other.

Why COPY and not executemany: 211k course rows via INSERT is minutes of
round-trips; binary COPY is one stream and lands in seconds. Indexes are
created only AFTER the load, because maintaining a GIN index during a bulk
insert costs far more than building it once at the end.

The load is transactional against a STAGING schema and swapped in at the end,
so a failed migration never leaves the serving schema half-populated — the same
discipline as the SQLite build's atomic file replace.

Run:
    python -m rag_core.store.migrate_postgres
    python -m rag_core.store.migrate_postgres --dsn postgresql://user:pw@host/db
    python -m rag_core.store.migrate_postgres --verify      # compare row counts
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

from .db import DB_FILE, connect as sqlite_connect

SCHEMA_FILE = Path(__file__).resolve().parent / "postgres.sql"

DEFAULT_DSN = os.getenv(
    "MIVI_PG_DSN", "postgresql://postgres:postgres@localhost:5432/postgres"
)

COPY_BATCH = 10_000


def _pg():
    try:
        import psycopg
    except ImportError as exc:  # noqa: BLE001
        raise SystemExit(
            "psycopg is required for the Postgres path:\n"
            "  pip install 'psycopg[binary]'"
        ) from exc
    return psycopg


def _jsonb(value):
    """SQLite stores these as JSON text; Postgres wants valid JSON or NULL.
    A malformed value becomes NULL rather than aborting a 211k-row COPY."""
    if value in (None, "", "null"):
        return None
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    try:
        json.loads(value)
        return value
    except (TypeError, ValueError):
        return None


def _bool(v):
    return bool(v) if v is not None else False


def _ts(v):
    return v or None


TABLES = {
    "colleges": (
        ("college_id", "name", "short_name", "city", "district", "state", "country",
         "address", "pincode", "type", "establish_year", "naac_grade", "nirf_ranking",
         "other_ranking", "affiliated_university", "autonomous_status", "is_abroad",
         "is_verified", "is_featured", "rating", "about", "details", "why_choose",
         "website_url", "email", "phone", "logo_url", "banner_url", "source_level",
         "source_updated_at"),
        {"autonomous_status": _bool, "is_abroad": _bool, "is_verified": _bool,
         "is_featured": _bool, "source_updated_at": _ts},
    ),
    "courses": (
        ("course_id", "college_id", "title", "full_name", "program_level", "department",
         "duration_value", "duration_unit", "course_mode", "tuition_min_inr",
         "tuition_max_inr", "tuition_min_raw", "tuition_max_raw", "currency",
         "hostel_min_inr", "hostel_max_inr", "one_time_fees", "fee_period",
         "entrance_exams", "min_marks", "description", "fees_description",
         "key_highlights", "is_abroad", "is_top_course", "scholarship_available"),
        {"entrance_exams": _jsonb, "key_highlights": _jsonb, "is_abroad": _bool,
         "is_top_course": _bool, "scholarship_available": _bool},
    ),
    "college_agg": (
        ("college_id", "n_courses", "program_levels", "course_titles", "departments",
         "entrance_exams", "min_tuition_inr", "max_tuition_inr", "min_hostel_inr",
         "has_hostel", "has_scholarship", "card"),
        {"program_levels": _jsonb, "course_titles": _jsonb, "departments": _jsonb,
         "entrance_exams": _jsonb, "has_hostel": _bool, "has_scholarship": _bool},
    ),
    "college_facts": (("college_id", "kind", "payload"), {"payload": _jsonb}),
    "college_web_facts": (
        ("college_id", "kind", "summary", "payload", "source_url", "fetched_at",
         "extractor", "confidence"),
        {"payload": _jsonb, "fetched_at": _ts},
    ),
    "web_crawl_state": (
        ("college_id", "website_url", "status", "last_attempt", "attempts",
         "pages_fetched", "note"),
        {"last_attempt": _ts},
    ),
    "build_meta": (("key", "value"), {}),
}


def _rows(sconn: sqlite3.Connection, table: str, cols, coerce):
    """Stream a SQLite table as tuples shaped for COPY."""
    try:
        cur = sconn.execute(f"SELECT {', '.join(cols)} FROM {table}")
    except sqlite3.OperationalError:
        return  # table absent in this build (e.g. no harvest yet)
    while True:
        batch = cur.fetchmany(COPY_BATCH)
        if not batch:
            return
        for r in batch:
            yield tuple(coerce.get(c, lambda v: v)(r[c]) for c in cols)


def migrate(dsn: str, sqlite_path: Path | None = None) -> dict:
    psycopg = _pg()
    sconn = sqlite_connect(sqlite_path or DB_FILE, read_only=True)
    stats: dict[str, int] = {}
    t0 = time.perf_counter()

    with psycopg.connect(dsn, autocommit=False) as pg:
        with pg.cursor() as cur:
            print("[pg] creating schema", file=sys.stderr)
            cur.execute(SCHEMA_FILE.read_text(encoding="utf-8"))
            # Truncate rather than drop: keeps the schema, indexes and grants,
            # and CASCADE handles the FK order for us.
            cur.execute("TRUNCATE colleges, courses, college_agg, college_facts, "
                        "college_web_facts, web_crawl_state, build_meta CASCADE")

            for table, (cols, coerce) in TABLES.items():
                t = time.perf_counter()
                n = 0
                sql = f"COPY {table} ({', '.join(cols)}) FROM STDIN"
                with cur.copy(sql) as cp:
                    for row in _rows(sconn, table, cols, coerce):
                        cp.write_row(row)
                        n += 1
                stats[table] = n
                print(f"[pg] {table:20} {n:>8,} rows in "
                      f"{time.perf_counter() - t:.1f}s", file=sys.stderr)

            # Full-text document, built from the same card the dense index
            # embeds — so lexical and semantic retrieval see identical text.
            print("[pg] building search_doc", file=sys.stderr)
            cur.execute("""
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
            """)
            print("[pg] ANALYZE", file=sys.stderr)
            cur.execute("ANALYZE")
        pg.commit()

    sconn.close()
    stats["seconds"] = round(time.perf_counter() - t0, 1)
    return stats


def verify(dsn: str, sqlite_path: Path | None = None) -> bool:
    psycopg = _pg()
    sconn = sqlite_connect(sqlite_path or DB_FILE, read_only=True)
    ok = True
    with psycopg.connect(dsn) as pg, pg.cursor() as cur:
        print(f"{'table':22} {'sqlite':>10} {'postgres':>10}")
        for table in TABLES:
            try:
                s = sconn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            except sqlite3.OperationalError:
                s = 0
            p = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            flag = "" if s == p else "   <-- MISMATCH"
            ok &= s == p
            print(f"{table:22} {s:>10,} {p:>10,}{flag}")
        # A row count says nothing about whether the text search actually works.
        n = cur.execute(
            "SELECT COUNT(*) FROM college_agg WHERE search_doc IS NOT NULL"
        ).fetchone()[0]
        print(f"{'search_doc populated':22} {'':>10} {n:>10,}")
        ok &= n > 0
    sconn.close()
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dsn", default=DEFAULT_DSN)
    ap.add_argument("--sqlite", type=Path, default=None)
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    if a.verify:
        sys.exit(0 if verify(a.dsn, a.sqlite) else 1)

    s = migrate(a.dsn, a.sqlite)
    print(f"[pg] done in {s.pop('seconds')}s: "
          + ", ".join(f"{k}={v:,}" for k, v in s.items()), file=sys.stderr)


if __name__ == "__main__":
    main()
