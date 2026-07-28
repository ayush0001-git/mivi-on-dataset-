"""Is the database actually complete yet?

WHY THIS MODULE EXISTS
That is the client's central question, and until now the only way to answer it
was to ask whoever happened to have run the last sweep. Numbers that live in
someone's head get quoted from memory, drift, and are always optimistic. This
module reads them out of the live store instead, so "how complete are we" is a
command anyone can run and a number nobody has to be told.

WHAT IT IS HONEST ABOUT

1. THE FUNNEL, not a single percentage. Every college that reaches a harvested
   fact has survived several independent filters: the MME detail API had to
   reach it, the detail record had to carry a website URL, that URL had to get
   queued and be usable, and a page on it had to actually contain a fact. Each
   filter drops a large share of what enters it, and the drop at each step is
   where the remaining work is.

2. BACKLOG IS NOT LOSS. This is the distinction the first version of this file
   got wrong, and it is the one that matters most for planning. 22,148 colleges
   are not detail-crawled because the API has not reached them yet — run the job
   longer and they arrive. 6,343 are missing a website URL after a successful
   detail crawl — those are gone on this path no matter how long anything runs.
   Both are "not covered"; only one is recoverable by waiting, so the report
   labels every drop as BACKLOG or LOSS and never sums them into one number.

3. SAMPLE SIZE. The projected ceiling is a product of two measured conditional
   rates, and a rate measured on twelve colleges is not a measurement. Every
   proportion here is reported with a Wilson 95% interval and the n it rests on,
   and the ceiling refuses to state a point estimate until n is large enough to
   support one. The first version of this module printed a confident "36.0%
   ceiling" off 12 crawled colleges; the true interval on that evidence spans
   19% to 50%. Rule 6 of this project — correctness beats coverage — applies to
   the dashboard's own numbers first.

READ-ONLY, ALWAYS. This module opens no sockets and writes no rows. It is safe
to run while a harvest sweep writes into the same tables: each statement gets
one MVCC snapshot, so the worst case is a number a few seconds old.

CLI:  python -m rag_core.coverage [--json]
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from .store.backend import get_backend

# The pre-migration store. backend.py declares Postgres the only store and
# refuses to fall back, which is right for serving — but the harvest sweep
# currently running still writes HERE, so at the time of writing Postgres holds
# 7 enriched colleges while this file holds 1,139. A dashboard that read only
# the store of record would report "0.0% enriched" and be wrong in the most
# misleading direction available: it would say the enrichment work had failed
# when it is in fact the furthest-along part of the project.
#
# So this file is PROBED, never reported from. A handful of portable COUNT
# queries, opened mode=ro so it cannot be written and with a short busy timeout
# so a locked file degrades to "no probe" instead of hanging the report. The
# probe exists to raise the discrepancy, not to become a second reporting path —
# running the whole report against both engines would mean two SQL dialects
# again, which is exactly what the Postgres migration removed.
LEGACY_SQLITE = Path(__file__).resolve().parent.parent / "data" / "mivi.db"

# ---------------------------------------------------------------------------
# What "covered" means, field by field.
#
# TRIM(x) <> '' as well as NOT NULL throughout: the MME API returns empty
# strings, not nulls, for fields it has no value for. Counting those as present
# is the easiest way to make this dashboard lie, and it would flatter exactly
# the fields the client cares most about.
# ---------------------------------------------------------------------------

def _txt(col: str) -> str:
    return f"{col} IS NOT NULL AND TRIM({col}) <> ''"


# The fields a student's question actually lands on. Ordered as the client asks
# about them, not by how well they score.
CATALOGUE_FIELDS = (
    ("name",              _txt("c.name")),
    ("city",              _txt("c.city")),
    ("state",             _txt("c.state")),
    ("type",              _txt("c.type")),
    ("establish year",    "c.establish_year IS NOT NULL AND c.establish_year > 0"),
    ("courses (>=1)",     "COALESCE(a.n_courses, 0) > 0"),
    ("fees (any course)", "a.min_tuition_inr IS NOT NULL AND a.min_tuition_inr > 0"),
)

# The known hole. Reported separately and always, even at 0%, because a
# dashboard that only lists what it has is how a gap stops being visible.
CATALOGUE_GAP_FIELDS = (
    ("NAAC grade",       _txt("c.naac_grade")),
    ("NIRF ranking",     _txt("c.nirf_ranking")),
    ("other ranking",    _txt("c.other_ranking")),
    ("rating",           _txt("c.rating")),
    ("affiliation",      _txt("c.affiliated_university")),
    ("hostel fee",       "a.min_hostel_inr IS NOT NULL AND a.min_hostel_inr > 0"),
    ("scholarship flag", "a.has_scholarship"),
)

COURSE_FIELDS = (
    ("title",          _txt("title")),
    ("program level",  _txt("program_level")),
    ("department",     _txt("department")),
    ("duration value", "duration_value IS NOT NULL AND duration_value > 0"),
    ("duration unit",  _txt("duration_unit")),
    # > 0 rather than IS NOT NULL, matching the college-level "fees" check.
    # A zero tuition for an Indian college is a placeholder far more often than
    # it is a free seat, so counting it as covered would flatter the number that
    # matters most to a student. There are currently no zeros in either column,
    # so this only decides what happens the day one appears — and it should
    # under-count, not over-count (rule 6).
    ("tuition",        "tuition_min_inr IS NOT NULL AND tuition_min_inr > 0"),
    ("fee period",     _txt("fee_period")),
    ("entrance exams", "entrance_exams IS NOT NULL "
                       "AND entrance_exams::text NOT IN ('null', '[]', '{}')"),
    ("min marks",      _txt("min_marks")),
    ("hostel fee",     "hostel_min_inr IS NOT NULL AND hostel_min_inr > 0"),
)

# ---------------------------------------------------------------------------
# Degeneracy checks.
#
# A column that is 100% populated with ONE value is worse than an empty column:
# it scores as complete and carries no information. `courses.fee_period` is the
# live example — every one of 211,071 rows says 'Year', which means a
# per-semester fee in the source is now indistinguishable from a per-year one.
# That is a direct hit on rule 3 (never convert, never annualise), so the
# dashboard has to surface it rather than report "fee period 100%".
# ---------------------------------------------------------------------------
DEGENERACY_CHECKS = (
    ("courses", "fee_period", "fee period",
     "every fee carries the same period label, so a per-semester fee cannot be "
     "distinguished from a per-year one — rule 3 cannot be verified downstream"),
    ("courses", "duration_unit", "duration unit",
     "the unit is present on nearly every row while duration_value is not, so "
     "the field is a default, not evidence"),
    ("colleges", "type", "type",
     "a single institution type across the corpus would mean the field carries "
     "no filtering power"),
)

# Deliberate duplication of web_enrich.PAGE_TARGETS' keys. Importing that module
# to read them would pull the whole crawler — httpx, the extractor registry, a
# provider budget probe — into a read-only reporting command, and it is a file
# under active edit by another workflow: a syntax error there would take this
# dashboard down with it. The union with the kinds actually present in the table
# means a new kind still appears here without this list being updated.
DECLARED_WEB_KINDS = ("scholarship", "fees", "cutoff", "admission",
                      "placement", "hostel", "events")

# Age bands. A college website re-publishes fees and scholarship schemes once an
# admission cycle, so ~180 days is where a harvested number stops being safe to
# quote and ~365 days is where it is probably wrong.
STALE_WARN_DAYS = 180
STALE_BAD_DAYS = 365

# Windows tried, narrowest first, when measuring throughput. Narrowest-with-
# enough-samples wins so a currently-running sweep is measured at ITS rate
# rather than averaged against the hours the machine sat idle: an all-time
# average is always too optimistic while a job runs and too pessimistic after
# it stops.
RATE_WINDOWS_H = (1, 6, 24, 168)
RATE_MIN_SAMPLES = 25

# Below this many crawled colleges the ceiling is an interval, not a number.
# 300 is roughly where the Wilson half-width on a ~20-60% rate falls under
# 6 percentage points, which is the point at which the estimate can actually
# distinguish "15% ceiling" from "35% ceiling" — the decision it exists to
# inform.
CEILING_MIN_SAMPLES = 300
CEILING_TARGET_MARGIN = 0.05

Z95 = 1.959963985


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def _wilson(k: int, n: int, z: float = Z95) -> dict | None:
    """Wilson score interval for a proportion.

    Wilson rather than the textbook normal interval because the normal one is
    actively wrong at the sample sizes and extreme rates this report deals in —
    it produces bounds outside [0,1] and collapses to zero width at p=0, which
    would let "0 of 40 colleges yielded a fact" print as a confident 0% ceiling.
    """
    if n <= 0:
        return None
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"p": p, "n": n, "k": k,
            "lo": max(0.0, centre - half), "hi": min(1.0, centre + half),
            "half_width": half}


def _samples_for_margin(p: float, margin: float, z: float = Z95) -> int:
    """How many samples the current rate would need for a given +/- margin.

    Printed next to an under-powered estimate so "we cannot state the ceiling
    yet" comes with the number that would fix it, instead of being a dead end.
    """
    p = min(max(p, 0.01), 0.99)
    return int(math.ceil(z * z * p * (1 - p) / (margin * margin)))


# ---------------------------------------------------------------------------
# store access
# ---------------------------------------------------------------------------

def _rows(be, sql: str, params=()) -> list[dict]:
    return be.q(sql, list(params))


def _row(be, sql: str, params=()) -> dict:
    got = _rows(be, sql, params)
    return dict(got[0]) if got else {}


def _safe(fn, default):
    """A missing or half-built table must degrade one section, not the report.

    Two other workflows own tables this reads and one of them is mid-job at any
    given time. A dashboard that prints nothing because college_web_facts does
    not exist yet is useless exactly when it is most needed.
    """
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        return {**default, "error": str(exc).strip().splitlines()[0][:160]}


def _pct(part: int, whole: int) -> float:
    return round(100.0 * part / whole, 1) if whole else 0.0


def _field_coverage(be, from_sql: str, fields) -> dict:
    """One pass, one COUNT(*) FILTER per field.

    Aliased f0..fn rather than by name: several labels contain spaces and one
    ("name") collides with a real column, and quoting an identifier built from a
    display label is how a reporting query grows an injection hole.
    """
    sel = ", ".join(f"COUNT(*) FILTER (WHERE {expr}) AS f{i}"
                    for i, (_, expr) in enumerate(fields))
    r = _row(be, f"SELECT COUNT(*) AS n, {sel} FROM {from_sql}")
    n = int(r.get("n") or 0)
    return {
        "total": n,
        "fields": [{"field": label,
                    "present": int(r.get(f"f{i}") or 0),
                    "pct": _pct(int(r.get(f"f{i}") or 0), n)}
                   for i, (label, _) in enumerate(fields)],
    }


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------

_COLLEGE_FROM = "colleges c LEFT JOIN college_agg a USING (college_id)"


def catalogue(be) -> dict:
    core = _field_coverage(be, _COLLEGE_FROM, CATALOGUE_FIELDS)
    gaps = _field_coverage(be, _COLLEGE_FROM, CATALOGUE_GAP_FIELDS)
    courses = _field_coverage(be, "courses", COURSE_FIELDS)
    shape = _row(be, """
        SELECT COUNT(*) FILTER (WHERE is_abroad)   AS abroad,
               COUNT(*) FILTER (WHERE is_verified) AS verified
        FROM colleges""")
    return {
        "colleges": core["total"],
        "courses": courses["total"],
        "abroad": int(shape.get("abroad") or 0),
        "verified": int(shape.get("verified") or 0),
        "college_fields": core["fields"],
        "college_gap_fields": gaps["fields"],
        "course_fields": courses["fields"],
    }


def data_quality(be) -> dict:
    """Fields that score as covered without carrying information."""
    flags = []
    for table, col, display, why in DEGENERACY_CHECKS:
        r = _row(be, f"SELECT COUNT(*) AS n, COUNT({col}) AS filled, "
                     f"COUNT(DISTINCT {col}) AS d FROM {table}")
        n, filled, d = (int(r.get("n") or 0), int(r.get("filled") or 0),
                        int(r.get("d") or 0))
        if not filled:
            continue
        top = _row(be, f"SELECT {col} AS v, COUNT(*) AS c FROM {table} "
                       f"WHERE {col} IS NOT NULL GROUP BY 1 ORDER BY 2 DESC LIMIT 1")
        share = _pct(int(top.get("c") or 0), filled)
        # One distinct value, or one value covering ~everything: either way the
        # column cannot discriminate and its coverage score is meaningless.
        if d <= 1 or share >= 99.0:
            flags.append({"column": f"{table}.{col}", "display": display,
                          "distinct_values": d, "filled": filled, "of": n,
                          "top_value": top.get("v"), "top_share_pct": share,
                          "why": why})

    pair = _row(be, """
        SELECT COUNT(*) FILTER (WHERE duration_unit IS NOT NULL
                                  AND TRIM(duration_unit) <> '') AS unit,
               COUNT(*) FILTER (WHERE duration_value IS NOT NULL
                                  AND duration_value > 0)        AS val
        FROM courses""")
    unit, val = int(pair.get("unit") or 0), int(pair.get("val") or 0)
    orphan_units = unit - val if unit > val else 0
    return {"degenerate": flags, "orphan_duration_units": orphan_units,
            "duration_unit_filled": unit, "duration_value_filled": val}


def legacy_probe(path: Path | str | None = None) -> dict | None:
    """Harvest counts from the pre-migration SQLite store, or None.

    Deliberately only counts. Nothing here renders, and nothing here can write:
    mode=ro is enforced by SQLite itself rather than by this code being careful.
    Every failure path returns None because a probe of a legacy file must never
    be the reason the dashboard fails.
    """
    p = Path(path or LEGACY_SQLITE)
    if not p.exists():
        return None
    try:
        con = sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True, timeout=3.0)
    except Exception:  # noqa: BLE001
        return None
    try:
        cur = con.cursor()
        have = {r[0] for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"college_web_facts", "web_crawl_state"} <= have:
            return None
        facts = cur.execute("SELECT COUNT(*) FROM college_web_facts").fetchone()[0]
        cols = cur.execute(
            "SELECT COUNT(DISTINCT college_id) FROM college_web_facts").fetchone()[0]
        status = {r[0]: r[1] for r in cur.execute(
            "SELECT status, COUNT(*) FROM web_crawl_state GROUP BY 1")}
        extractors = {r[0]: r[1] for r in cur.execute(
            "SELECT extractor, COUNT(*) FROM college_web_facts GROUP BY 1")}
        newest = cur.execute(
            "SELECT MAX(fetched_at) FROM college_web_facts").fetchone()[0]
        by_kind = {r[0]: r[1] for r in cur.execute(
            "SELECT kind, COUNT(*) FROM college_web_facts GROUP BY 1")}
    except Exception:  # noqa: BLE001
        return None
    finally:
        con.close()

    attempted = sum(v for k, v in status.items() if k not in ("pending", "no_site"))
    return {"path": str(p), "facts": facts, "colleges_with_any_web_fact": cols,
            "crawl_status": status, "queued": sum(status.values()),
            "attempted": attempted, "ok": status.get("ok", 0),
            "pending": status.get("pending", 0), "extractors": extractors,
            "by_kind": by_kind, "newest_fact": newest}


def enrichment(be) -> dict:
    reach = _row(be, f"""
        SELECT COUNT(*) AS n,
               COUNT(*) FILTER (WHERE c.source_level = 'detail') AS detail,
               COUNT(*) FILTER (WHERE {_txt('c.website_url')})   AS site,
               COUNT(*) FILTER (WHERE c.source_level = 'detail'
                                  AND {_txt('c.website_url')})   AS detail_site,
               COUNT(*) FILTER (WHERE {_txt('c.email')})         AS email,
               COUNT(*) FILTER (WHERE {_txt('c.phone')})         AS phone
        FROM colleges c""")
    n = int(reach.get("n") or 0)

    status = _safe(lambda: {"rows": [
        {"status": r["status"], "n": int(r["n"])}
        for r in _rows(be, "SELECT status, COUNT(*) AS n FROM web_crawl_state "
                           "GROUP BY status ORDER BY n DESC")]}, {"rows": []})

    queue = _safe(lambda: _row(be, """
        SELECT COUNT(*) AS queued,
               COUNT(*) FILTER (WHERE status <> 'no_site')      AS usable,
               COUNT(*) FILTER (WHERE status =  'no_site')      AS no_site,
               COUNT(*) FILTER (WHERE last_attempt IS NOT NULL) AS attempted,
               COUNT(*) FILTER (WHERE status =  'pending')      AS pending,
               COUNT(*) FILTER (WHERE status =  'ok')           AS ok,
               COALESCE(SUM(pages_fetched), 0)                  AS pages
        FROM web_crawl_state"""), {})

    # Per kind: the primary key is (college_id, kind), so the row count IS the
    # number of colleges holding that kind. Confidence comes along because a
    # kind that is 80% covered at 0.35 confidence is not 80% covered.
    per_kind = _safe(lambda: {"rows": [
        {"kind": r["kind"], "colleges": int(r["n"]),
         "avg_confidence": round(float(r["conf"]), 2) if r["conf"] is not None else None,
         "pct_of_corpus": _pct(int(r["n"]), n)}
        for r in _rows(be, "SELECT kind, COUNT(*) AS n, AVG(confidence)::numeric AS conf "
                           "FROM college_web_facts GROUP BY kind")]}, {"rows": []})
    have = {r["kind"]: r for r in per_kind.get("rows", [])}
    kinds = list(DECLARED_WEB_KINDS) + sorted(set(have) - set(DECLARED_WEB_KINDS))
    web_kinds = [have.get(k, {"kind": k, "colleges": 0, "avg_confidence": None,
                              "pct_of_corpus": 0.0}) for k in kinds]

    any_fact = _safe(lambda: _row(be, """
        SELECT COUNT(DISTINCT college_id) AS colleges, COUNT(*) AS facts
        FROM college_web_facts"""), {})

    cat_kinds = _safe(lambda: {"rows": [
        {"kind": r["kind"], "colleges": int(r["n"])}
        for r in _rows(be, "SELECT kind, COUNT(*) AS n FROM college_facts "
                           "GROUP BY kind ORDER BY n DESC")]}, {"rows": []})

    return {
        "colleges": n,
        "detail_crawled": int(reach.get("detail") or 0),
        "website_known": int(reach.get("site") or 0),
        "detail_and_website": int(reach.get("detail_site") or 0),
        "email_known": int(reach.get("email") or 0),
        "phone_known": int(reach.get("phone") or 0),
        "crawl_status": status.get("rows", []),
        "crawl_status_error": status.get("error"),
        "queue": queue,
        "web_kinds": web_kinds,
        "colleges_with_any_web_fact": int(any_fact.get("colleges") or 0),
        "web_facts_total": int(any_fact.get("facts") or 0),
        "catalogue_fact_kinds": cat_kinds.get("rows", []),
    }


# Substrings that mark an `extractor` label as a model rather than a parser.
# The live store holds 'fast-model', written by an earlier revision of the sweep
# that recorded the model slot instead of the 'parser'/'llm' tag the current one
# writes. Mapping it by hand would be a lie the day it changes again, so the
# rollup is heuristic AND the raw labels are always printed underneath.
MODEL_HINTS = ("model", "llm", "gpt", "gemini", "llama", "groq", "flash",
               "mini", "sonnet", "haiku", "opus", "qwen", "mistral")
PARSER_LABELS = ("parser", "deterministic", "regex", "extractor")
REGISTRY_HINTS = ("registry", "aishe", "ugc", "nirf", "aicte", "nba", "gov")


def _channel(extractor: str | None) -> str:
    """Which pipeline produced a fact.

    Registry prefixes are matched in advance so a government-registry loader
    lands in its own bucket the day it starts writing, instead of silently
    inflating 'other'.
    """
    e = (extractor or "").strip().lower()
    if not e:
        return "unlabelled"
    if any(h in e for h in PARSER_LABELS):
        return "deterministic parser"
    if any(h in e for h in REGISTRY_HINTS):
        return "government registry"
    if any(h in e for h in MODEL_HINTS):
        return "LLM extraction"
    return "unclassified"


# Always printed, in this order, even at zero. The government-registry row
# being 0 is a finding — it is a channel the plan depends on and nothing writes
# it yet — and a table that omitted empty rows would hide that.
CHANNELS = ("deterministic parser", "LLM extraction", "government registry")


def source_mix(be) -> dict:
    got = _safe(lambda: {"rows": _rows(be, """
        SELECT extractor, COUNT(*) AS n, AVG(confidence)::numeric AS conf
        FROM college_web_facts GROUP BY extractor ORDER BY 2 DESC""")},
        {"rows": []})
    if got.get("error"):
        return {"channels": [], "labels": [], "total": 0, "error": got["error"]}

    agg: dict[str, dict] = {}
    labels = []
    total = 0
    for r in got["rows"]:
        ch = _channel(r["extractor"])
        k = int(r["n"])
        total += k
        labels.append({"extractor": r["extractor"], "facts": k, "channel": ch})
        slot = agg.setdefault(ch, {"facts": 0, "_wsum": 0.0})
        slot["facts"] += k
        if r["conf"] is not None:
            slot["_wsum"] += float(r["conf"]) * k

    order = list(CHANNELS) + sorted(set(agg) - set(CHANNELS))
    out = []
    for ch in order:
        slot = agg.get(ch, {"facts": 0, "_wsum": 0.0})
        f = slot["facts"]
        out.append({"channel": ch, "facts": f, "pct": _pct(f, total),
                    "avg_confidence": round(slot["_wsum"] / f, 2) if f else None})
    for lab in labels:
        lab["pct"] = _pct(lab["facts"], total)
    return {"channels": out, "labels": labels, "total": total}


def funnel(be, enr: dict) -> dict:
    """The report's actual thesis.

    Every drop is split into BACKLOG (work not yet done — the same job, run
    longer, recovers it) and LOSS (filtered out — no amount of runtime recovers
    it; it needs a fix or a different source). Reporting one blended "lost here"
    column, as the first version did, made 22,148 un-crawled colleges look like
    the same kind of problem as 6,343 colleges that have no website at all. They
    are opposite problems with opposite responses.
    """
    q = enr.get("queue") or {}
    queued = int(q.get("queued") or 0)
    no_site = int(q.get("no_site") or 0)
    attempted = int(q.get("attempted") or 0)
    pending = int(q.get("pending") or 0)
    total = enr["colleges"] or 0
    detail = enr["detail_crawled"]
    site = enr["website_known"]
    facts = enr["colleges_with_any_web_fact"]

    # (label, count, backlog out of the previous stage, loss out of it, note)
    spec = [
        ("colleges in catalogue", total, 0, 0, ""),
        ("detail-crawled (MME API)", detail, total - detail, 0,
         "API pacing, ~0.5 req/s — recoverable by running longer"),
        ("website URL known", site, 0, detail - site,
         "detail record carries no URL — dead end on this path"),
        ("queued for crawl", queued, max(site - queued - no_site, 0), no_site,
         "backlog = seed_queue has not covered these yet; "
         "loss = URL present but unusable"),
        ("crawl attempted", attempted, pending, queued - attempted - pending,
         "pending is the sweep's remaining work"),
        (">=1 fact extracted", facts, 0, attempted - facts,
         "fetched but nothing extractable — empty/blocked/error"),
    ]
    top = total or 1
    out = []
    prev = None
    for label, n, backlog, loss, note in spec:
        out.append({
            "stage": label, "count": n, "pct_of_corpus": _pct(n, top),
            # step_pct is the conditional survival rate at this filter, which is
            # the number that says which filter to fix. pct_of_corpus cannot: a
            # stage looks terrible purely because the stage above it was.
            "step_pct": None if prev is None else _pct(n, prev),
            "backlog": None if prev is None else max(backlog, 0),
            "loss": None if prev is None else max(loss, 0),
            "note": note,
        })
        prev = n
    return {"stages": out,
            "total_backlog": sum(s["backlog"] or 0 for s in out),
            "total_loss": sum(s["loss"] or 0 for s in out)}


def staleness(be) -> dict:
    web = _safe(lambda: _row(be, f"""
        SELECT COUNT(*) AS n,
               MIN(fetched_at) AS oldest,
               percentile_disc(0.5) WITHIN GROUP (ORDER BY fetched_at) AS median,
               MAX(fetched_at) AS newest,
               COUNT(*) FILTER (WHERE fetched_at
                     < now() - make_interval(days => {STALE_WARN_DAYS})) AS warn,
               COUNT(*) FILTER (WHERE fetched_at
                     < now() - make_interval(days => {STALE_BAD_DAYS}))  AS bad
        FROM college_web_facts"""), {})
    cat = _safe(lambda: _row(be, """
        SELECT COUNT(*) AS n,
               MIN(source_updated_at) AS oldest,
               percentile_disc(0.5) WITHIN GROUP (ORDER BY source_updated_at) AS median,
               MAX(source_updated_at) AS newest
        FROM colleges WHERE source_updated_at IS NOT NULL"""), {})

    now = datetime.now(timezone.utc)

    def block(r: dict) -> dict:
        def age(ts):
            return round((now - ts).total_seconds() / 86400.0, 1) if ts else None
        return {"rows": int(r.get("n") or 0),
                "oldest": r.get("oldest"), "oldest_age_days": age(r.get("oldest")),
                "median": r.get("median"), "median_age_days": age(r.get("median")),
                "newest": r.get("newest"), "newest_age_days": age(r.get("newest")),
                "error": r.get("error")}

    w = block(web)
    w.update({"past_warn": int(web.get("warn") or 0),
              "past_bad": int(web.get("bad") or 0),
              "warn_days": STALE_WARN_DAYS, "bad_days": STALE_BAD_DAYS})
    return {"harvested_facts": w, "catalogue": block(cat)}


def _measure_rate(be, table: str, col: str) -> dict:
    """Throughput of a stage, measured from the timestamps it leaves behind.

    Rate is samples/span WITHIN the window, not samples/window: a sweep that ran
    for ten minutes of the last hour did six times the per-hour rate that
    samples/window would report, and projecting off the understated figure
    promises a finish date nobody hits.

    Returns a state, never None, because "idle since Tuesday" and "ran 20
    minutes ago but only touched 12 rows" are different facts and the first
    version of this collapsed both into a bare "no activity in 168h", which was
    simply untrue of the second.
    """
    last = _row(be, f"SELECT MAX({col}) AS t, COUNT({col}) AS n FROM {table}")
    ever = last.get("t")
    for w in RATE_WINDOWS_H:
        r = _row(be, f"SELECT COUNT(*) AS n, MIN({col}) AS t0, MAX({col}) AS t1 "
                     f"FROM {table} WHERE {col} > now() - make_interval(hours => ?)",
                 [w])
        n = int(r.get("n") or 0)
        if not n or not r.get("t0"):
            continue
        span_h = (r["t1"] - r["t0"]).total_seconds() / 3600.0
        if n < RATE_MIN_SAMPLES:
            # Recent, but too few events to divide by a span without inventing
            # precision. Say so and carry the evidence.
            if w == RATE_WINDOWS_H[-1]:
                return {"state": "sparse", "samples": n, "window_h": w,
                        "span_h": round(span_h, 3), "last_activity": ever,
                        "need_samples": RATE_MIN_SAMPLES}
            continue
        if span_h < 1.0 / 60:
            # Under a minute of span is one commit batch landing, not a rate.
            continue
        return {"state": "measured", "per_hour": n / span_h, "samples": n,
                "window_h": w, "span_h": round(span_h, 3), "last_activity": r["t1"]}
    return {"state": "idle", "last_activity": ever,
            "total_events": int(last.get("n") or 0),
            "window_h": RATE_WINDOWS_H[-1]}


def _eta(remaining: int, rate: dict) -> dict:
    if remaining <= 0:
        return {"remaining": 0, "state": "complete"}
    per_hour = rate.get("per_hour") if rate.get("state") == "measured" else None
    if not per_hour or per_hour <= 0:
        return {"remaining": remaining, "state": "no_eta"}
    hours = remaining / per_hour
    now = datetime.now(timezone.utc)
    return {"remaining": remaining, "state": "projected",
            "eta_hours": round(hours, 1), "eta_days": round(hours / 24.0, 1),
            "finish_utc": datetime.fromtimestamp(now.timestamp() + hours * 3600,
                                                 timezone.utc)}


def _ceiling(total: int, site_ci: dict, ok_k: int, ok_n: int,
             today: int, source: str) -> dict | None:
    """Ceiling = P(website URL | detail-crawled) x P(>=1 fact | crawled).

    Both factors are proportions with intervals, so the ceiling gets an interval
    too — the product of the lower bounds to the product of the upper bounds.
    That is deliberately the widest honest reading: the two rates are estimated
    on different samples and a tighter combination would need an assumption
    about their independence that nothing here has tested.
    """
    ok_ci = _wilson(ok_k, ok_n)
    if not site_ci or not ok_ci:
        return None
    point = site_ci["p"] * ok_ci["p"]
    lo = site_ci["lo"] * ok_ci["lo"]
    hi = site_ci["hi"] * ok_ci["hi"]
    return {
        "source": source,
        "evidence": "sufficient" if ok_n >= CEILING_MIN_SAMPLES else "insufficient",
        "p_website_given_detail": round(site_ci["p"], 4),
        "website_n": site_ci["n"],
        "website_ci": [round(site_ci["lo"], 4), round(site_ci["hi"], 4)],
        "p_facts_given_crawled": round(ok_ci["p"], 4),
        "facts_n": ok_n,
        "facts_ci": [round(ok_ci["lo"], 4), round(ok_ci["hi"], 4)],
        "projected_pct": round(100.0 * point, 1),
        "projected_colleges": int(total * point),
        "range_pct": [round(100.0 * lo, 1), round(100.0 * hi, 1)],
        "range_colleges": [int(total * lo), int(total * hi)],
        "today_colleges": today,
        "today_pct": _pct(today, total),
        "min_samples": CEILING_MIN_SAMPLES,
        "need_samples": _samples_for_margin(ok_ci["p"], CEILING_TARGET_MARGIN),
        "target_margin_pp": round(CEILING_TARGET_MARGIN * 100, 1),
    }


def projection(be, enr: dict, legacy: dict | None = None) -> dict:
    """When does each stage finish, and — the question that matters — AT WHAT?

    Time to drain a queue is a scheduling fact. The CEILING is the product of two
    conditional rates that no amount of extra runtime changes, and reporting only
    an ETA lets "finishes in 3 days" be heard as "complete in 3 days". That
    misunderstanding is the reason this section exists.
    """
    q = enr.get("queue") or {}
    total = enr["colleges"] or 0

    detail_rate = _safe(lambda: _measure_rate(be, "colleges", "source_updated_at"),
                        {"state": "idle"})
    harvest_rate = _safe(lambda: _measure_rate(be, "web_crawl_state", "last_attempt"),
                         {"state": "idle"})

    stages = [
        {"stage": "detail-crawl the catalogue (MME API)", "rate": detail_rate,
         **_eta(total - enr["detail_crawled"], detail_rate)},
        {"stage": "seed the crawl queue (seed_queue)", "rate": {"state": "n/a"},
         **_eta(max(enr["website_known"] - int(q.get("queued") or 0), 0),
                {"state": "n/a"})},
        {"stage": "harvest the queued websites", "rate": harvest_rate,
         **_eta(int(q.get("pending") or 0), harvest_rate)},
    ]

    # --- the ceiling ------------------------------------------------------
    detail = enr["detail_crawled"] or 0
    attempted = int(q.get("attempted") or 0)
    site_ci = _wilson(enr["detail_and_website"], detail)
    ceiling = _ceiling(total, site_ci, int(q.get("ok") or 0), attempted,
                       enr["colleges_with_any_web_fact"], "postgres")

    # The best-evidenced ceiling, which is not necessarily the one from the store
    # of record. P(website | detail) still comes from Postgres because that is
    # where the catalogue lives, but P(fact | crawled) is taken from whichever
    # store has actually crawled more colleges. Preferring the store of record
    # for its own sake would mean answering the client's central question from a
    # 12-college sample while a 4,849-college one sat on disk.
    legacy_ceiling = None
    if legacy and legacy.get("attempted", 0) > attempted:
        legacy_ceiling = _ceiling(total, site_ci, legacy["ok"], legacy["attempted"],
                                  legacy["colleges_with_any_web_fact"],
                                  "legacy sqlite (P(fact|crawled) only)")

    # --- where the recoverable loss is ------------------------------------
    by_status = {r["status"]: r["n"] for r in enr.get("crawl_status", [])}
    losses = {k: v for k, v in by_status.items()
              if k in ("empty", "blocked", "error", "no_site")}
    biggest = None
    if losses and attempted:
        k = max(losses, key=lambda x: losses[x])
        pending = int(q.get("pending") or 0)
        ci = _wilson(losses[k], attempted)
        biggest = {
            "status": k, "colleges": losses[k], "sample_n": attempted,
            "pct_of_attempted": _pct(losses[k], attempted),
            "ci_pct": [round(100 * ci["lo"], 1), round(100 * ci["hi"], 1)] if ci else None,
            # Scaled over the whole queue, because a fix is worth what it
            # recovers across the full run, not across the sample crawled so far.
            "projected_at_queue_end": int(losses[k] + losses[k] / attempted * pending),
            "confident": attempted >= CEILING_MIN_SAMPLES,
        }

    # The same question asked of the legacy store, where the sample is large
    # enough for the answer to mean something.
    legacy_loss = None
    if legacy and legacy.get("attempted"):
        ls = {k: v for k, v in legacy["crawl_status"].items()
              if k in ("empty", "blocked", "error", "no_site")}
        if ls:
            k = max(ls, key=lambda x: ls[x])
            ci = _wilson(ls[k], legacy["attempted"])
            legacy_loss = {
                "status": k, "colleges": ls[k], "sample_n": legacy["attempted"],
                "pct_of_attempted": _pct(ls[k], legacy["attempted"]),
                "ci_pct": [round(100 * ci["lo"], 1), round(100 * ci["hi"], 1)],
                "projected_at_queue_end": int(
                    ls[k] + ls[k] / legacy["attempted"] * legacy.get("pending", 0)),
                "confident": legacy["attempted"] >= CEILING_MIN_SAMPLES,
            }

    return {"stages": stages, "ceiling": ceiling, "legacy_ceiling": legacy_ceiling,
            "biggest_loss": biggest, "legacy_biggest_loss": legacy_loss}


def report(legacy_path: Path | str | None = None,
           probe_legacy: bool = True) -> dict:
    be = get_backend()
    cat = _safe(lambda: catalogue(be), {"colleges": 0, "courses": 0,
                                        "college_fields": [], "college_gap_fields": [],
                                        "course_fields": []})
    enr = _safe(lambda: enrichment(be),
                {"colleges": cat.get("colleges", 0), "detail_crawled": 0,
                 "website_known": 0, "detail_and_website": 0,
                 "colleges_with_any_web_fact": 0, "web_kinds": [],
                 "crawl_status": [], "queue": {}, "catalogue_fact_kinds": []})
    legacy = legacy_probe(legacy_path) if probe_legacy else None

    # A discrepancy only counts as one if the legacy store is AHEAD. Behind or
    # equal is the expected end state of the migration and not worth alarming
    # anyone about.
    split = None
    if legacy:
        pg_cols = enr.get("colleges_with_any_web_fact") or 0
        if legacy["colleges_with_any_web_fact"] > pg_cols:
            split = {
                "legacy_path": legacy["path"],
                "legacy_colleges": legacy["colleges_with_any_web_fact"],
                "postgres_colleges": pg_cols,
                "legacy_attempted": legacy["attempted"],
                "postgres_attempted": int((enr.get("queue") or {}).get("attempted") or 0),
                "legacy_facts": legacy["facts"],
                "postgres_facts": enr.get("web_facts_total") or 0,
            }

    return {
        "generated_at": datetime.now(timezone.utc),
        "backend": be.name,
        "catalogue": cat,
        "data_quality": _safe(lambda: data_quality(be), {"degenerate": []}),
        "enrichment": enr,
        "legacy_store": legacy,
        "store_split": split,
        "source_mix": _safe(lambda: source_mix(be),
                            {"channels": [], "labels": [], "total": 0}),
        "funnel": _safe(lambda: funnel(be, enr), {"stages": []}),
        "staleness": _safe(lambda: staleness(be), {}),
        "projection": _safe(lambda: projection(be, enr, legacy),
                            {"stages": [], "ceiling": None, "biggest_loss": None}),
    }


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

BAR_W = 18


def _bar(pct: float) -> str:
    """A coverage table is read for its shape before its numbers; the bar is
    what makes a 3% row and a 97% row distinguishable at a glance."""
    fill = int(round((pct or 0) / 100.0 * BAR_W))
    return "#" * fill + "." * (BAR_W - fill)


def _n(x) -> str:
    if isinstance(x, bool):
        return str(x)
    return f"{x:,}" if isinstance(x, int) else ("-" if x is None else str(x))


def _ts(x) -> str:
    if not isinstance(x, datetime):
        return "-"
    return x.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _fields_table(out: list, rows: list, total: int, flagged: set = frozenset()) -> None:
    for f in rows:
        mark = " (!)" if f["field"] in flagged else ""
        out.append(f"  {f['field']:<16} {_n(f['present']):>9} / {_n(total):<9} "
                   f"{f['pct']:>5.1f}%  {_bar(f['pct'])}{mark}")


def render(rep: dict) -> str:
    o: list[str] = []
    cat, enr = rep["catalogue"], rep["enrichment"]
    dq = rep.get("data_quality") or {}
    total = cat.get("colleges") or 0
    flagged = {d["display"] for d in dq.get("degenerate", []) if d.get("display")}

    o.append("=" * 78)
    o.append("MIVI COVERAGE DASHBOARD — is the database actually complete yet?")
    o.append(f"generated {_ts(rep['generated_at'])}   store={rep['backend']}")
    o.append("=" * 78)

    # --- the store split, if there is one ---------------------------------
    sp = rep.get("store_split")
    if sp:
        o.append("")
        o.append("!! THE HARVEST IS NOT IN THE STORE THIS REPORT READS")
        o.append(f"   postgres (store of record): "
                 f"{_n(sp['postgres_colleges'])} colleges enriched, "
                 f"{_n(sp['postgres_attempted'])} crawled")
        o.append(f"   legacy sqlite (live sweep): "
                 f"{_n(sp['legacy_colleges'])} colleges enriched, "
                 f"{_n(sp['legacy_attempted'])} crawled")
        o.append(f"   {sp['legacy_path']}")
        o.append("   Sections 1-5 below describe POSTGRES and therefore understate")
        o.append("   enrichment badly. The sweep writes SQLite; migrating it is what")
        o.append("   makes this dashboard correct. Section 6 uses the larger sample")
        o.append("   where it changes the answer, and says so.")

    # --- headline ---------------------------------------------------------
    fact_cols = enr.get("colleges_with_any_web_fact") or 0
    fn = rep.get("funnel") or {}
    pr0 = rep.get("projection") or {}
    o.append("")
    o.append("HEADLINE")
    o.append(f"  catalogue    {_n(total)} colleges, {_n(cat.get('courses') or 0)} courses")
    o.append(f"  enriched     {_n(fact_cols)} colleges hold >=1 harvested fact "
             f"({_pct(fact_cols, total)}% of the corpus)")
    if sp:
        o.append(f"               {_n(sp['legacy_colleges'])} "
                 f"({_pct(sp['legacy_colleges'], total)}%) counting the legacy store")
    # Prefer whichever ceiling the evidence actually supports.
    lc, c = pr0.get("legacy_ceiling"), pr0.get("ceiling")
    best = lc if (lc and lc["evidence"] == "sufficient") else c
    if best and best["evidence"] == "sufficient":
        o.append(f"  ceiling      ~{best['projected_pct']}% "
                 f"({_n(best['projected_colleges'])} colleges) once every queue "
                 f"drains, 95% CI {best['range_pct'][0]}-{best['range_pct'][1]}%")
        o.append(f"               n={_n(best['facts_n'])} crawled, "
                 f"via {best['source']}")
    elif best:
        o.append(f"  ceiling      UNKNOWN — {best['range_pct'][0]}% to "
                 f"{best['range_pct'][1]}% (95% CI on {_n(best['facts_n'])} crawled "
                 f"colleges; needs {_n(best['min_samples'])})")
    if fn.get("stages"):
        o.append(f"  gap split    {_n(fn.get('total_backlog') or 0)} colleges are "
                 f"BACKLOG (a longer run reaches them), "
                 f"{_n(fn.get('total_loss') or 0)} are LOSS (a longer run does not)")

    # --- catalogue --------------------------------------------------------
    o.append("")
    o.append("-" * 78)
    o.append("1. CATALOGUE COVERAGE (from the MME API)")
    o.append("-" * 78)
    _fields_table(o, cat.get("college_fields", []), total, flagged)
    o.append(f"  ({_n(cat.get('abroad') or 0)} abroad, "
             f"{_n(cat.get('verified') or 0)} flagged verified — both counted above)")
    o.append("")
    o.append("  the known hole — what the harvest exists to fill:")
    _fields_table(o, cat.get("college_gap_fields", []), total, flagged)
    o.append("")
    o.append(f"  per course ({_n(cat.get('courses') or 0)} rows):")
    _fields_table(o, cat.get("course_fields", []), cat.get("courses") or 0, flagged)

    if dq.get("degenerate") or dq.get("orphan_duration_units"):
        o.append("")
        o.append("  (!) DATA QUALITY — these score as covered but carry no information:")
        for d in dq.get("degenerate", []):
            o.append(f"      {d['column']}: {d['distinct_values']} distinct value(s), "
                     f"'{d['top_value']}' on {d['top_share_pct']}% of "
                     f"{_n(d['filled'])} filled rows")
            o.append(f"        -> {d['why']}")
        if dq.get("orphan_duration_units"):
            o.append(f"      courses.duration_unit is set on "
                     f"{_n(dq['duration_unit_filled'])} rows but duration_value on "
                     f"only {_n(dq['duration_value_filled'])}")
            o.append(f"        -> {_n(dq['orphan_duration_units'])} rows carry a unit "
                     f"with no number to attach it to")

    # --- enrichment -------------------------------------------------------
    o.append("")
    o.append("-" * 78)
    o.append("2. ENRICHMENT COVERAGE (from college websites)")
    o.append("-" * 78)
    for label, key in (("detail-crawled", "detail_crawled"),
                       ("website URL known", "website_known"),
                       ("email known", "email_known"),
                       ("phone known", "phone_known")):
        v = enr.get(key) or 0
        o.append(f"  {label:<16} {_n(v):>9} / {_n(total):<9} "
                 f"{_pct(v, total):>5.1f}%  {_bar(_pct(v, total))}")

    q = enr.get("queue") or {}
    if q:
        o.append("")
        o.append(f"  crawl queue: {_n(int(q.get('queued') or 0))} seeded, "
                 f"{_n(int(q.get('attempted') or 0))} attempted, "
                 f"{_n(int(q.get('pending') or 0))} pending, "
                 f"{_n(int(q.get('pages') or 0))} pages fetched")
    if enr.get("crawl_status"):
        o.append("  crawl outcome:  " + "  ".join(
            f"{r['status']}={_n(r['n'])}" for r in enr["crawl_status"]))
    if enr.get("crawl_status_error"):
        o.append(f"  (crawl state unavailable: {enr['crawl_status_error']})")

    o.append("")
    o.append(f"  harvested facts per kind ({_n(enr.get('web_facts_total') or 0)} "
             f"fact rows over {_n(fact_cols)} colleges):")
    for k in enr.get("web_kinds", []):
        conf = "  conf -" if k["avg_confidence"] is None else f"  conf {k['avg_confidence']:.2f}"
        o.append(f"  {k['kind']:<16} {_n(k['colleges']):>9} / {_n(total):<9} "
                 f"{k['pct_of_corpus']:>5.1f}%  {_bar(k['pct_of_corpus'])}{conf}")
    if enr.get("catalogue_fact_kinds"):
        o.append("")
        o.append("  catalogue-supplied fact blocks (MME detail API, not scraped):")
        for k in enr["catalogue_fact_kinds"]:
            o.append(f"  {k['kind']:<16} {_n(k['colleges']):>9} / {_n(total):<9} "
                     f"{_pct(k['colleges'], total):>5.1f}%")

    # --- source mix -------------------------------------------------------
    mix = rep.get("source_mix") or {}
    o.append("")
    o.append("-" * 78)
    o.append("3. SOURCE MIX — where the harvested facts came from")
    o.append("-" * 78)
    if mix.get("error"):
        o.append(f"  unavailable: {mix['error']}")
    elif not mix.get("total"):
        o.append("  no harvested facts yet")
    else:
        for ch in mix.get("channels", []):
            conf = "conf -" if ch["avg_confidence"] is None else f"conf {ch['avg_confidence']:.2f}"
            o.append(f"  {ch['channel']:<21}{_n(ch['facts']):>8}  "
                     f"{ch['pct']:>5.1f}%  {_bar(ch['pct'])}  {conf}")
        o.append(f"  {'total':<21}{_n(mix.get('total') or 0):>8}")
        o.append("  raw extractor labels behind that rollup:")
        for lab in mix.get("labels", []):
            o.append(f"      {str(lab['extractor']):<20} {_n(lab['facts']):>7}  "
                     f"-> {lab['channel']}")
        o.append("  a parser fact costs no API quota; an LLM fact costs one call")
        o.append("  against a ~14/min account-wide cap.")

    lg = rep.get("legacy_store")
    if sp and lg and lg.get("extractors"):
        o.append("")
        o.append(f"  the legacy store's mix ({_n(lg['facts'])} facts) — this is the")
        o.append("  real picture of which pipeline is carrying the work:")
        ltot = sum(lg["extractors"].values()) or 1
        for lab, k in sorted(lg["extractors"].items(), key=lambda x: -x[1]):
            o.append(f"      {str(lab):<20} {_n(k):>7}  {_pct(k, ltot):>5.1f}%  "
                     f"-> {_channel(lab)}")

    # --- funnel -----------------------------------------------------------
    o.append("")
    o.append("-" * 78)
    o.append("4. THE FUNNEL — where the colleges are actually lost")
    o.append("-" * 78)
    o.append(f"  {'stage':<26}{'count':>8}{'corpus':>9}{'step':>7}"
             f"{'BACKLOG':>10}{'LOSS':>9}")
    for i, s in enumerate(fn.get("stages", [])):
        label = ("  " + "  " * i + s["stage"])[:26]
        step = "-" if s["step_pct"] is None else f"{s['step_pct']:.1f}%"
        o.append(f"  {label:<26}{_n(s['count']):>8}{s['pct_of_corpus']:>8.1f}%"
                 f"{step:>7}{_n(s['backlog']):>10}{_n(s['loss']):>9}")
    o.append("")
    o.append("  BACKLOG = not done yet; the same job, run longer, reaches these.")
    o.append("  LOSS    = filtered out; runtime never recovers these.")
    for s in fn.get("stages", []):
        if s.get("note") and ((s.get("backlog") or 0) or (s.get("loss") or 0)):
            o.append(f"    {s['stage']}: {s['note']}")

    # --- staleness --------------------------------------------------------
    st = rep.get("staleness") or {}
    o.append("")
    o.append("-" * 78)
    o.append("5. STALENESS — how old is what we hold")
    o.append("-" * 78)
    for label, key in (("harvested facts", "harvested_facts"), ("catalogue", "catalogue")):
        b = st.get(key) or {}
        if b.get("error"):
            o.append(f"  {label}: unavailable ({b['error']})")
            continue
        if not b.get("rows"):
            o.append(f"  {label}: nothing recorded yet")
            continue
        o.append(f"  {label} ({_n(b['rows'])} rows)")
        o.append(f"    oldest  {_ts(b['oldest'])}  ({b['oldest_age_days']} days old)")
        o.append(f"    median  {_ts(b['median'])}  ({b['median_age_days']} days old)")
        o.append(f"    newest  {_ts(b['newest'])}  ({b['newest_age_days']} days old)")
        if "past_warn" in b:
            o.append(f"    past {b['warn_days']}d: {_n(b['past_warn'])}   "
                     f"past {b['bad_days']}d: {_n(b['past_bad'])}")
            if b["past_warn"]:
                o.append(f"    -> re-crawl: python -m rag_core.sources.web_enrich "
                         f"--requeue-stale {b['warn_days']}")

    # --- projection -------------------------------------------------------
    pr = rep.get("projection") or {}
    o.append("")
    o.append("-" * 78)
    o.append("6. PROJECTION — when does this finish, and finish AT WHAT")
    o.append("-" * 78)
    for s in pr.get("stages", []):
        o.append(f"  {s['stage']}")
        if s["state"] == "complete":
            o.append("    complete — nothing remaining")
            continue
        r = s.get("rate") or {}
        o.append(f"    left    {_n(s['remaining'])}")
        rs = r.get("state")
        if rs == "measured":
            o.append(f"    rate    {r['per_hour']:.1f}/hour ({r['samples']} events "
                     f"over {r['span_h']}h inside a {r['window_h']}h window)")
        elif rs == "sparse":
            o.append(f"    rate    unmeasurable — only {r['samples']} events in "
                     f"{r['window_h']}h, need {r['need_samples']} "
                     f"(last {_ts(r.get('last_activity'))})")
        elif rs == "idle":
            o.append(f"    rate    idle — no activity in {r.get('window_h')}h "
                     f"(last {_ts(r.get('last_activity'))})")
        else:
            o.append("    rate    n/a — this stage is one command, not a rate")
        if s["state"] == "projected":
            o.append(f"    ETA     {s['eta_hours']}h = {s['eta_days']} days "
                     f"-> {_ts(s['finish_utc'])}")
        else:
            o.append("    ETA     none — refusing to project without a measured rate")

    def ceiling_block(c: dict) -> None:
        o.append(f"    -- from {c['source']} --")
        o.append(f"    P(website URL | detail-crawled) = {c['p_website_given_detail']:.3f}"
                 f"  [{c['website_ci'][0]:.3f}-{c['website_ci'][1]:.3f}]  "
                 f"n={_n(c['website_n'])}")
        o.append(f"    P(>=1 fact    | crawled)        = {c['p_facts_given_crawled']:.3f}"
                 f"  [{c['facts_ci'][0]:.3f}-{c['facts_ci'][1]:.3f}]  "
                 f"n={_n(c['facts_n'])}")
        if c["evidence"] == "sufficient":
            o.append(f"    => ceiling ~{c['projected_pct']}% of the corpus "
                     f"({_n(c['projected_colleges'])} colleges), "
                     f"95% CI {c['range_pct'][0]}-{c['range_pct'][1]}% "
                     f"({_n(c['range_colleges'][0])}-{_n(c['range_colleges'][1])})")
        else:
            o.append(f"    => NOT STATEABLE. Point estimate would be "
                     f"{c['projected_pct']}%, but the 95% interval spans "
                     f"{c['range_pct'][0]}-{c['range_pct'][1]}% "
                     f"({_n(c['range_colleges'][0])}-{_n(c['range_colleges'][1])} "
                     f"colleges).")
            o.append(f"       That rests on {_n(c['facts_n'])} crawled colleges — too "
                     f"few to state a ceiling. Reporting the midpoint would be")
            o.append(f"       inventing precision. Crawl ~{_n(c['need_samples'])} "
                     f"(>= {_n(c['min_samples'])} minimum) to state it to "
                     f"+/-{c['target_margin_pp']}pp.")
        o.append(f"    today: {_n(c['today_colleges'])} colleges ({c['today_pct']}%).")

    c, lc = pr.get("ceiling"), pr.get("legacy_ceiling")
    if c or lc:
        o.append("")
        o.append("  THE CEILING on the current path")
        if c:
            ceiling_block(c)
        if lc:
            o.append("")
            ceiling_block(lc)
            o.append("    The legacy sample is the one to plan against: same corpus,")
            o.append("    same crawler, orders of magnitude more evidence.")
        o.append("")
        o.append("    ASSUMPTION: both rates hold as the crawl moves down the")
        o.append("    n_courses ordering. They probably DEGRADE — the queue is")
        o.append("    priority-ordered, so the best-covered colleges went first,")
        o.append("    which makes any figure here an upper bound rather than an")
        o.append("    expectation. Raising the ceiling needs a different SOURCE")
        o.append("    (a registry keyed on college identity, not a website URL);")
        o.append("    more hours of the same sweep cannot move it.")

    # Prefer the legacy sample here too: "which failure mode should we fix" is a
    # question 12 crawls cannot answer and 4,849 can.
    b = pr.get("legacy_biggest_loss") or pr.get("biggest_loss")
    if b:
        o.append("")
        o.append("  BIGGEST RECOVERABLE LOSS")
        ci = f", 95% CI {b['ci_pct'][0]}-{b['ci_pct'][1]}%" if b.get("ci_pct") else ""
        o.append(f"    status '{b['status']}': {_n(b['colleges'])} colleges = "
                 f"{b['pct_of_attempted']}% of {_n(b['sample_n'])} attempted{ci}")
        if b["confident"]:
            o.append(f"    ~{_n(b['projected_at_queue_end'])} colleges projected at "
                     f"queue end")
        else:
            o.append(f"    projection withheld: {_n(b['sample_n'])} attempted is too "
                     f"small a sample to scale over the queue")
        if b["status"] == "empty":
            o.append("    'empty' = homepage fetched, no useful sub-page found. The")
            o.append("    request was already paid for, so this is the cheapest ground")
            o.append("    to recover: better link discovery, not more crawling.")

    o.append("")
    o.append("=" * 78)
    return "\n".join(o)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m rag_core.coverage",
        description="Coverage, funnel and completion projection for the MIVI corpus.")
    ap.add_argument("--json", action="store_true",
                    help="machine-readable output (for a dashboard or a CI gate)")
    ap.add_argument("--legacy", metavar="PATH", default=None,
                    help=f"legacy SQLite store to probe (default {LEGACY_SQLITE})")
    ap.add_argument("--no-legacy", action="store_true",
                    help="skip the legacy probe; report Postgres alone")
    a = ap.parse_args(argv)

    rep = report(legacy_path=a.legacy, probe_legacy=not a.no_legacy)
    if a.json:
        # default=str so TIMESTAMPTZ values serialise as ISO-8601 rather than
        # exploding a monitoring job on a datetime.
        print(json.dumps(rep, indent=2, default=str))
    else:
        print(render(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
