"""The corpus store: PostgreSQL, and nothing else.

WHY ONE ENGINE NOW
The build used to target a local SQLite file and serving used to target
Postgres, which meant two dialects, two schemas, and a copy step between them
that could only ever drift. Postgres holds the whole corpus today — 38,700
colleges, 211,071 courses — so the second engine buys nothing.

Be honest about what the move is NOT: measured, Postgres is only ~9 ms faster
per filter query (16.6 ms vs 25.8 ms p50) against an LLM call of 1-3 s. Nobody
will feel that. What it buys is CONCURRENT WRITES (the website-harvest sweep
writes while students read), more than one app process, and strict typing —
Postgres rejected 20 course rows carrying a tuition of 8,900,000,000 that
SQLite had been storing without complaint.

WHAT THIS MODULE OWNS
Every connection to that database, and the one seam retrieval sits on:
`?` placeholders (rewritten to `%s` here) and full-text search (`tsvector @@
plainto_tsquery` + `ts_rank`). Retrieval SQL is written once, in one dialect,
and never branches on the engine.

The boolean idioms in that SQL — `NOT c.is_abroad`, `a.has_hostel` rather than
`= 0` / `= 1` — started as portability insurance. They stay because they are
also the only form Postgres accepts against a real BOOLEAN column.
"""
from __future__ import annotations

import os
import re
import sys
import threading

DEFAULT_PG_DSN = os.getenv(
    "MIVI_PG_DSN", "postgresql://postgres:postgres@localhost:5432/postgres"
)

# The API admits at most this many requests at once (app.py: MAX_CONCURRENCY).
# The pool is sized against it below rather than against a guess.
_SERVING_CONCURRENCY = int(os.getenv("MIVI_MAX_CONCURRENCY", "16") or 16)


def _dsn_label(dsn: str) -> str:
    """host:port/db with any credentials removed — safe to print or log."""
    return dsn.rsplit("@", 1)[-1] if "@" in dsn else dsn


def _q_to_pct(sql: str) -> str:
    """`?` placeholders -> `%s`. Safe here because none of the retrieval SQL
    contains a literal `?` inside a string constant; a generic rewriter would
    have to tokenise, and inventing that risk for no benefit would be worse
    than the assertion below."""
    if "'?'" in sql or '"?"' in sql:
        raise ValueError("SQL contains a literal '?'; the rewriter cannot handle it")
    # `%` in LIKE patterns lives in PARAMS, not in the SQL text, so it is safe.
    return sql.replace("%", "%%").replace("?", "%s")


class Row(dict):
    """A dict row. Everything in rag_core reads columns by name; the old
    positional-index path (sqlite3.Row's `row[0]`) was carried over from the
    SQLite era and removed: every legacy caller has been migrated, and
    silently returning `list(values())[i]` for an integer key was a memory-
    order trap on duplicate column names.
    """
    __slots__ = ()


def _row_factory(cursor):
    desc = cursor.description
    cols = [d.name for d in desc] if desc else []
    return lambda values: Row(zip(cols, values))


_PHRASE_RE = re.compile(r'"([^"]+)"')
_WORD_RE = re.compile(r"[\w']+")
# FTS5's connectives arrive as bare tokens between the quoted terms. They are
# syntax, not search terms, and must not survive into the tsquery.
_FTS_OPS = frozenset(("or", "and", "not", "near"))


def _tsquery_arms(match: str) -> list[str]:
    """The OR-arms of an FTS5 MATCH expression, as plain text.

    retrieve.py._fts_query emits every term as its own quoted phrase joined by
    OR — `"punjab" OR "btech"` — and says why: FTS5's implicit AND matches
    nothing at all on a conversational question. That intent is the thing that
    has to survive translation, and flattening the expression into one string
    destroys it, because plainto_tsquery ANDs every word it is handed. Measured
    on the live corpus: the flattened form built `'punjab' & 'btech' & 'or' &
    'engineering'` (note `or` promoted to a required lexeme) and returned 0 rows
    for every multi-term question, while `btech` alone matches 141 Punjab
    colleges. So the arms are kept separate here and ORed in SQL.

    Only the fallback path strips connectives: retrieve.py puts them outside the
    quotes, so a phrase like "arts and science" keeps its `and` as a real word.
    """
    arms = [a.strip() for a in _PHRASE_RE.findall(match)]
    if not arms:
        # Unquoted input — a caller passing a bare phrase rather than an FTS5
        # expression. One word per arm, connectives dropped.
        arms = [w for w in _WORD_RE.findall(match) if w.lower() not in _FTS_OPS]
    out: list[str] = []
    seen: set[str] = set()
    for arm in arms:
        # Re-tokenise: punctuation inside an arm would otherwise reach
        # plainto_tsquery, which is harmless but makes the arm unpredictable.
        words = " ".join(_WORD_RE.findall(arm))
        key = words.lower()
        if words and key not in seen:
            seen.add(key)
            out.append(words)
    # retrieve.py already caps its own token list; this bounds the SQL text for
    # any other caller, since each arm adds a function call to the statement.
    return out[:24]


class PostgresBackend:
    name = "postgres"

    def __init__(self, dsn: str | None = None,
                 min_size: int = 2, max_size: int | None = None):
        from psycopg_pool import ConnectionPool

        self.dsn = dsn or DEFAULT_PG_DSN

        # SIZING. A pool, not a connection per request: Postgres forks a backend
        # process per connection, so opening one per query would cost more than
        # every query it serves.
        #   max_size — the API never has more than MIVI_MAX_CONCURRENCY (16)
        #     requests in flight, and a request borrows a connection only for
        #     the duration of one ~20 ms query, never across the 1-3 s LLM call.
        #     16 is therefore already an over-estimate of simultaneous
        #     borrowers; the +4 covers the writers that run OUTSIDE that
        #     semaphore — memory upserts, the 5-second health probe, a harvest
        #     sweep sharing the process.
        #   ceiling of 32 — Postgres's default max_connections is 100 and every
        #     connection is a forked process. A pool allowed to outgrow the
        #     server turns a traffic spike into "FATAL: too many clients", which
        #     is a worse outage than queueing for a connection.
        #   min_size 2 — two warm connections so the first student of the day
        #     does not pay TCP + auth, without holding 16 idle forks overnight.
        if max_size is None:
            max_size = min(max(_SERVING_CONCURRENCY + 4, 8), 32)
        self._pool = ConnectionPool(
            self.dsn, min_size=min_size, max_size=max_size,
            kwargs={"row_factory": _row_factory}, open=False, timeout=10.0,
        )
        # wait=True so an unreachable database fails HERE, at startup, with the
        # real reason — not later, inside whichever student's query happened to
        # be first.
        try:
            self._pool.open(wait=True, timeout=10.0)
        except BaseException:
            # __init__ is the thing raising, so no caller ever receives a
            # reference to close this pool — anything left running leaks, and a
            # supervisor that retries startup stacks up another set of
            # reconnect-looping worker threads on every attempt.
            #
            # psycopg already covers the ordinary case: pool.wait() calls
            # close() itself before raising PoolTimeout. This covers the exits it
            # does not, the realistic one being Ctrl-C during that 10 s wait
            # (hence BaseException, not Exception). close() is idempotent, so on
            # the PoolTimeout path this is a free no-op.
            #
            # Short timeout because it only runs when the DSN is already known
            # bad; note psycopg cannot always join a worker blocked in connect(),
            # so "couldn't stop thread" may still be logged on the way out.
            self._pool.close(timeout=1.0)
            raise

    # -- read ---------------------------------------------------------------

    def q(self, sql: str, params=()) -> list[dict]:
        with self._pool.connection() as c:
            return c.execute(_q_to_pct(sql), tuple(params)).fetchall()

    def one(self, sql: str, params=()):
        rows = self.q(sql, params)
        if not rows:
            return None
        return next(iter(rows[0].values()))

    def lexical(self, match: str, where: str, params: list, limit: int) -> list[str]:
        """Full-text ranking over college_agg.search_doc.

        `match` arrives in FTS5 syntax because that is what retrieve.py still
        emits; `_tsquery_arms` recovers its OR-arms (see there for why the
        OR is the load-bearing part).

        ts_rank is DESCENDING for better matches, the opposite of SQLite's
        bm25(), which is exactly the kind of difference that belongs behind
        this interface rather than in retrieval.
        """
        arms = _tsquery_arms(match)
        if not arms:
            return []
        # One plainto_tsquery per arm, ORed with `||` (tsquery OR). The arms stay
        # PARAMETERS: to_tsquery would let a stray apostrophe in a college name
        # raise a syntax error mid-question, and this form cannot raise at all.
        expr = "(" + " || ".join(["plainto_tsquery('simple', ?)"] * len(arms)) + ")"
        # The expression is written out twice rather than shared via a CTE.
        # Measured both on the live corpus: identical plans, because with a
        # literal regconfig plainto_tsquery is IMMUTABLE and Postgres folds it to
        # a constant tsquery at plan time either way. Duplication costs nothing
        # and keeps this a single flat statement.
        sql = (
            "SELECT c.college_id FROM college_agg a "
            "JOIN colleges c ON c.college_id = a.college_id "
            f"WHERE a.search_doc @@ {expr} AND {where} "
            f"ORDER BY ts_rank(a.search_doc, {expr}) DESC "
            "LIMIT ?"
        )
        return [r["college_id"]
                for r in self.q(sql, [*arms, *params, *arms, limit])]

    # -- write --------------------------------------------------------------
    # Retrieval never calls these. They exist so the writers — memory upserts,
    # the harvest sweep, the ETL's bookkeeping — stop opening a private psycopg
    # connection per statement, which cost a server-side fork to run one INSERT.

    def execute(self, sql: str, params=()) -> int:
        """One statement, committed on success and rolled back on error.
        Returns rows affected."""
        with self._pool.connection() as c:
            return c.execute(_q_to_pct(sql), tuple(params)).rowcount

    def executemany(self, sql: str, seq_params) -> int:
        """A batch in ONE transaction — all of it lands or none of it does.

        That is what the harvest checkpoint wants: a sweep killed mid-batch
        must not leave half a college's facts written with the crawl state
        already marked done, because nothing would ever come back for the rest.
        """
        rows = [tuple(p) for p in seq_params]
        if not rows:
            return 0
        with self._pool.connection() as c, c.cursor() as cur:
            cur.executemany(_q_to_pct(sql), rows)
            return cur.rowcount

    # -- lifecycle ----------------------------------------------------------

    def health(self) -> dict:
        try:
            n = self.one("SELECT COUNT(*) FROM colleges")
            return {"backend": "postgres", "present": True, "colleges": n,
                    "pool": self._pool.get_stats().get("pool_size")}
        except Exception as exc:  # noqa: BLE001
            return {"backend": "postgres", "present": False, "error": str(exc)[:200]}

    def close(self):
        self._pool.close()


_backend = None
_lock = threading.Lock()


def get_backend() -> PostgresBackend:
    """The process-wide backend.

    An unreachable database is a hard, loud failure. This used to fall back to
    the local SQLite file, which was defensible while that file was rebuilt
    nightly — a student reading slightly stale answers beats a site-wide 503.
    It is indefensible now: nothing writes that file any more, so falling back
    would serve a corpus frozen on the day of this migration, quoting dead fees
    for closed colleges with exactly the confidence of a fresh answer. A 503 is
    recoverable. A plausible wrong answer to a 17-year-old choosing a college
    is not.
    """
    global _backend
    if _backend is not None:
        return _backend
    with _lock:
        if _backend is not None:
            return _backend
        want = os.getenv("MIVI_DB_BACKEND", "postgres").strip().lower()
        if want and want not in ("postgres", "postgresql", "pg"):
            # Almost certainly a config file left over from the SQLite era.
            # Ignoring it silently would serve Postgres while the operator
            # believed they had pinned something else.
            raise RuntimeError(
                f"MIVI_DB_BACKEND={want!r} is not a valid setting any more — "
                f"SQLite has been removed and Postgres is the only store. "
                f"Unset the variable or set it to 'postgres'."
            )
        pool_owner = None
        try:
            pool_owner = PostgresBackend()
            pool_owner.one("SELECT 1")
        except Exception as exc:  # noqa: BLE001
            if pool_owner is not None:
                try:
                    pool_owner.close()
                except Exception:  # noqa: BLE001
                    pass
            raise RuntimeError(
                f"MIVI cannot reach PostgreSQL at {_dsn_label(DEFAULT_PG_DSN)}. "
                f"The corpus lives there and there is no local copy to fall back "
                f"on.\n  reason: {exc}\n"
                f"  check : is the server running, and does MIVI_PG_DSN in .env "
                f"point at it?\n"
                f"  check : has the corpus been loaded into it "
                f"(rag_core/store/build_db.py)?"
            ) from exc
        print(f"[store] backend=postgres ({_dsn_label(pool_owner.dsn)})",
              file=sys.stderr)
        _backend = pool_owner
        # Bump SCHEMA_VERSION in lockstep with migration scripts. A wrong
        # version is a soft warning: the corpus may still work, but the
        # server has no way to know which queries are safe. The schema
        # table itself is absent before the first ETL run, which is
        # reported as "not initialised yet" rather than a mismatch.
        try:
            v = pool_owner.one(
                "SELECT value FROM build_meta WHERE key = 'schema_version'")
            expected = "2.0"
            if v is None:
                print("[store] build_meta has no schema_version — "
                      "ETL has not yet recorded one (first build?)", file=sys.stderr)
            elif str(v) != expected:
                print(f"[store] WARNING: schema_version is {v!r}, expected "
                      f"{expected!r}. Run: python -m rag_core.store.migrate_postgres",
                      file=sys.stderr)
            else:
                print(f"[store] schema_version OK ({v})", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 - advisory only
            print(f"[store] schema_version check skipped: {str(exc)[:80]}",
                  file=sys.stderr)
        return _backend


def reset_backend():
    """Drop the cached backend (tests, and re-pointing at another DSN in one
    process). Closes the pool so its connections are not left forked."""
    global _backend
    b, _backend = _backend, None
    if b is not None:
        try:
            b.close()
        except Exception:  # noqa: BLE001
            pass
