"""Score the MIVI eval set against the live pipeline, or validate it offline.

    python evals/run_evals.py --dry-run          # no LLM calls, no cost
    python evals/run_evals.py                    # full run
    python evals/run_evals.py --kind absence --limit 3
    python evals/run_evals.py --json report.json

Exit code is 0 only when every selected case passes, so this is usable as a CI
gate directly.

WHY TWO MODES
-------------
An eval suite over a 35k-college corpus has two independent ways to be wrong,
and they need different tools:

  * the SYSTEM regresses — caught only by actually asking the pipeline;
  * the EVAL SET goes stale — the corpus is rebuilt, a count shifts from 113 to
    118, and a green suite starts asserting a lie. That failure is worse,
    because it is silent and it discredits every other case.

`--dry-run` targets the second: it re-executes each case's `truth_sql` against
the corpus and fails if the recorded `truth_value` no longer holds, and it
confirms every `grounded` college_id still resolves to the same name. It spends
zero LLM calls, so it can run on every commit while the full suite runs nightly.

DATA SOURCE for dry-run: `data/mivi.db` when it exists. Until the ETL has built
it, dry-run falls back to `data/raw/colleges_list.jsonl` for identity checks and
reports the SQL assertions as SKIPPED rather than inventing a verdict — an
unrunnable check must never be reported as a passing one.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

EVAL_SET = Path(__file__).parent / "eval_set.json"
DB_FILE = ROOT / "data" / "mivi.db"
COLLEGES_JSONL = ROOT / "data" / "raw" / "colleges_list.jsonl"

KINDS = {
    "lookup", "filtered_search", "aggregate", "superlative", "absence",
    "abroad_currency", "multilingual", "safety", "ambiguous",
}

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
LATIN = re.compile(r"[A-Za-z]")


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
def _norm(text: str) -> str:
    """Casefold and strip the digit grouping that makes Indian money strings
    unmatchable. "Rs 1,50,000" and "150000" are the same claim to a student, so
    they must be the same claim to the scorer — otherwise every fee assertion
    becomes a test of the model's comma placement."""
    return re.sub(r"(?<=\d),(?=\d)", "", text).casefold()


def _contains(haystack: str, needle: str) -> bool:
    return _norm(needle) in _norm(haystack)


def _check_include(answer: str, spec) -> list[str]:
    """`spec` entries are strings (required) or lists (at least one required)."""
    fails = []
    for item in spec:
        if isinstance(item, list):
            if not any(_contains(answer, alt) for alt in item):
                fails.append(f"answer contains none of {item}")
        elif not _contains(answer, item):
            fails.append(f"answer is missing {item!r}")
    return fails


# ---------------------------------------------------------------------------
# Corpus access — SQLite if the ETL has run, JSONL otherwise
# ---------------------------------------------------------------------------
class Corpus:
    """Read-only view used by both modes. `sql_ok` is False when only the raw
    JSONL is available, which downgrades SQL assertions to SKIP."""

    def __init__(self) -> None:
        self.conn: sqlite3.Connection | None = None
        self.backend = None
        self.names: dict[str, str] | None = None
        self.source = "none"

        # POSTGRES FIRST. This runner still preferred data/mivi.db, which the
        # migration retired — a 451 MB file left on disk from the day before.
        # The suite whose job is catching stale expectations was therefore
        # validating truth_sql against a stale corpus: no admission criteria, no
        # NAAC, no registry facts, and fees from before the whole-programme
        # correction. An eval that reads the wrong database is worse than no
        # eval, because it reports PASS.
        try:
            from rag_core.store.backend import get_backend

            be = get_backend()
            be.one("SELECT 1 FROM colleges LIMIT 1")
            self.backend = be
            self.source = f"postgres:{be.label if hasattr(be, 'label') else 'live'}"
            return
        except Exception as exc:  # noqa: BLE001 - fall back, but say so
            print(f"[corpus] Postgres unavailable ({str(exc)[:90]})",
                  file=sys.stderr)

        if DB_FILE.exists():
            try:
                self.conn = sqlite3.connect(
                    f"file:{DB_FILE.as_posix()}?mode=ro", uri=True)
                self.conn.execute("SELECT 1 FROM colleges LIMIT 1")
                self.source = f"sqlite:{DB_FILE} (LEGACY — retired by the "
                self.source += "Postgres migration; results may be stale)"
                return
            except sqlite3.Error as e:
                # A half-built DB is worse than none: fall back loudly rather
                # than validate against a table the ETL is still filling.
                print(f"[corpus] {DB_FILE} unusable ({e}); using JSONL",
                      file=sys.stderr)
                self.conn = None
        if COLLEGES_JSONL.exists():
            self.names = {}
            with open(COLLEGES_JSONL, encoding="utf-8") as f:
                for line in f:
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if d.get("_id"):
                        self.names[d["_id"]] = d.get("collegeName") or ""
            self.source = f"jsonl:{COLLEGES_JSONL} ({len(self.names)} colleges)"

    @property
    def sql_ok(self) -> bool:
        return self.backend is not None or self.conn is not None

    @property
    def available(self) -> bool:
        return self.sql_ok or self.names is not None

    def name_of(self, college_id: str) -> str | None:
        if self.backend is not None:
            return self.backend.one(
                "SELECT name FROM colleges WHERE college_id=?", [college_id])
        if self.conn is not None:
            row = self.conn.execute(
                "SELECT name FROM colleges WHERE college_id=?", (college_id,)
            ).fetchone()
            return row[0] if row else None
        if self.names is not None:
            return self.names.get(college_id)
        return None

    def exists(self, college_id: str) -> bool:
        return self.name_of(college_id) is not None

    def scalar(self, sql: str):
        if self.backend is not None:
            return self.backend.one(sql)
        assert self.conn is not None
        return self.conn.execute(sql).fetchone()[0]


# ---------------------------------------------------------------------------
# Dry run — validate the eval set itself
# ---------------------------------------------------------------------------
REQUIRED_KEYS = ("id", "question", "kind", "must_include", "must_not_include",
                 "expect_answered", "expect_citations", "notes")


def validate_case(case: dict, corpus: Corpus, seen_ids: set) -> tuple[list[str], int]:
    """Returns (problems, skipped_sql_checks)."""
    problems, skipped = [], 0
    cid = case.get("id", "<no id>")

    for key in REQUIRED_KEYS:
        if key not in case:
            problems.append(f"missing required key {key!r}")
    if cid in seen_ids:
        problems.append(f"duplicate case id {cid!r}")
    seen_ids.add(cid)

    if case.get("kind") not in KINDS:
        problems.append(f"unknown kind {case.get('kind')!r} (expected one of {sorted(KINDS)})")
    if case.get("expect_answered") not in (True, False, None):
        problems.append("expect_answered must be true, false or null")
    if case.get("expect_citations") not in (True, False, None):
        problems.append("expect_citations must be true, false or null")
    if case.get("expect_script") not in (None, "devanagari", "latin"):
        problems.append(f"bad expect_script {case.get('expect_script')!r}")
    if not str(case.get("notes", "")).strip():
        problems.append("notes is empty — every case must say what it catches")

    for key in ("must_include", "must_not_include"):
        val = case.get(key)
        if not isinstance(val, list):
            problems.append(f"{key} must be a list")
            continue
        for item in val:
            if isinstance(item, list):
                if key == "must_not_include":
                    problems.append("must_not_include may not contain alternatives")
                elif not item or not all(isinstance(x, str) for x in item):
                    problems.append(f"{key} alternative group must be non-empty strings")
            elif not isinstance(item, str):
                problems.append(f"{key} entries must be strings or lists of strings")

    # A case can contradict itself: expect_citations false + expect_answered
    # true is fine (counselling turns), but demanding citations while forbidding
    # them is not.
    if case.get("expect_citations") and case.get("kind") == "safety":
        problems.append("safety cases must not expect citations")

    # The anti-fabrication guard: every college this case was authored against
    # must still be in the corpus, under the same name.
    for g in case.get("grounded", []):
        gid, gname = g.get("college_id"), g.get("name")
        if not corpus.available:
            skipped += 1
            continue
        actual = corpus.name_of(gid)
        if actual is None:
            problems.append(f"grounded college_id {gid} is NOT in the corpus")
        elif gname and actual != gname:
            problems.append(f"grounded {gid}: corpus says {actual!r}, case says {gname!r}")

    # The anti-staleness guard.
    if "truth_sql" in case:
        if "truth_value" not in case:
            problems.append("truth_sql given without truth_value")
        elif not corpus.sql_ok:
            skipped += 1
        else:
            try:
                got = corpus.scalar(case["truth_sql"])
            # Not sqlite3.Error: the corpus is Postgres now, and catching only
            # the SQLite exception let a psycopg error escape and kill the whole
            # dry run instead of failing the one case that asked for it.
            except Exception as e:  # noqa: BLE001
                problems.append(f"truth_sql failed: {str(e)[:150]}")
            else:
                want = case["truth_value"]
                if str(got) != str(want):
                    problems.append(
                        f"STALE EXPECTATION: truth_sql now returns {got!r}, "
                        f"case asserts {want!r} — fix the case, not the model")
                # The asserted number should actually be asserted somewhere.
                elif isinstance(want, int) and not any(
                    _contains(str(want), i) if isinstance(i, str)
                    else any(_contains(str(want), a) for a in i)
                    for i in case.get("must_include", [])
                ) and case["kind"] in ("aggregate", "superlative"):
                    problems.append(
                        f"truth_value {want} is not referenced in must_include — "
                        f"the case measures the corpus but never checks the answer")
    return problems, skipped


def dry_run(cases: list[dict]) -> int:
    corpus = Corpus()
    print(f"corpus: {corpus.source}")
    if not corpus.sql_ok:
        print("NOTE: data/mivi.db not available — SQL truth assertions are "
              "SKIPPED, not passed. Re-run once the ETL has built the store.")
    print()

    seen: set = set()
    failed = total_skipped = 0
    for case in cases:
        problems, skipped = validate_case(case, corpus, seen)
        total_skipped += skipped
        if problems:
            failed += 1
            print(f"[INVALID] {case.get('id')}")
            for p in problems:
                print(f"        - {p}")

    by_kind = OrderedDict()
    for case in cases:
        by_kind[case.get("kind")] = by_kind.get(case.get("kind"), 0) + 1
    print("cases per kind:")
    for kind, n in sorted(by_kind.items()):
        print(f"  {kind:<16} {n}")
    print(f"\n{len(cases)} cases, {len(cases) - failed} valid, {failed} invalid, "
          f"{total_skipped} corpus checks skipped")
    return 1 if failed else 0


# ---------------------------------------------------------------------------
# Live run
# ---------------------------------------------------------------------------
def score(case: dict, result: dict, corpus: Corpus) -> list[str]:
    fails = []
    answer = result.get("answer", "") or ""
    citations = [c for c in (result.get("citations") or []) if isinstance(c, str)]

    expected = case.get("expect_answered")
    if expected is not None and bool(result.get("answered")) != expected:
        fails.append(f"answered={result.get('answered')} (expected {expected})")

    fails += _check_include(answer, case.get("must_include", []))
    for item in case.get("must_not_include", []):
        if _contains(answer, item):
            fails.append(f"answer contains forbidden {item!r}")

    want_cites = case.get("expect_citations")
    if want_cites is True and not citations:
        fails.append("expected at least one citation, got none")
    elif want_cites is False and citations:
        fails.append(f"expected no citations, got {citations}")

    # Citation validity is checked on EVERY case regardless of expectation: a
    # citation that does not resolve to a real college is an ungrounded answer
    # no matter what the case was written to probe. This is the one check that
    # cannot be waived.
    if corpus.available:
        unknown = [c for c in citations if not corpus.exists(c)]
        if unknown:
            fails.append(f"cited ids not present in the colleges table: {unknown}")

    script = case.get("expect_script")
    if script == "devanagari" and not DEVANAGARI.search(answer):
        fails.append("reply is not in Devanagari — the student wrote Hindi")
    elif script == "latin":
        if DEVANAGARI.search(answer):
            fails.append("reply is in Devanagari — the student wrote Roman script")
        elif not LATIN.search(answer):
            fails.append("reply contains no Latin script")
    return fails


def live_run(cases: list[dict], sleep_s: float, report_path: Path | None) -> int:
    from rag_core.pipeline import answer_question

    corpus = Corpus()
    print(f"corpus: {corpus.source}\n")
    if not corpus.available:
        print("WARNING: no corpus available — citation validity cannot be "
              "checked. Results are weaker than they look.\n", file=sys.stderr)

    stats: dict[str, list[int]] = {}
    records = []
    
    eval_mrr_sum = 0.0
    eval_recall10_sum = 0.0
    eval_retrieval_cases = 0
    
    for i, case in enumerate(cases):
        if i and sleep_s:
            time.sleep(sleep_s)  # stay inside provider rate limits
        t0 = time.perf_counter()
        try:
            result = answer_question(case["question"])
        except Exception as e:
            # An exception is a failure, not a crash of the harness — the other
            # cases still carry information.
            result = {"answer": "", "citations": [], "answered": False,
                      "reason_if_unanswered": f"harness caught: {e!r}"}
            print(f"[ERROR] {case['id']}: {e!r}", file=sys.stderr)
        latency = time.perf_counter() - t0

        fails = score(case, result, corpus)
        kind = case["kind"]
        stats.setdefault(kind, [0, 0])
        stats[kind][1] += 1
        stats[kind][0] += not fails

        print(f"[{'PASS' if not fails else 'FAIL'}] {kind:<16} {case['id']} "
              f"({latency:.1f}s)")
        for f in fails:
            print(f"        - {f}")
        if fails:
            print(f"        answer: {result.get('answer', '')[:240]!r}")
            print(f"        citations: {result.get('citations')}")
        records.append({"id": case["id"], "kind": kind, "passed": not fails,
                        "failures": fails, "latency_s": round(latency, 2),
                        "answered": result.get("answered"),
                        "citations": result.get("citations"),
                        "answer": result.get("answer")})
        
        grounded = [g["college_id"] for g in case.get("grounded") or []]
        if grounded:
            retrieved_ids = result.get("retrieved_ids", [])
            target = grounded[0]
            if target in retrieved_ids:
                rank = retrieved_ids.index(target) + 1
                eval_mrr_sum += 1.0 / rank
                if rank <= 10:
                    eval_recall10_sum += 1.0
            eval_retrieval_cases += 1

    print(f"\n{'kind':<18}{'pass':>6}{'total':>7}{'rate':>8}")
    print("-" * 39)
    total_pass = total_all = 0
    for kind in sorted(stats):
        p, t = stats[kind]
        total_pass += p
        total_all += t
        print(f"{kind:<18}{p:>6}{t:>7}{p / t:>7.0%}")
    print("-" * 39)
    print(f"{'TOTAL':<18}{total_pass:>6}{total_all:>7}"
          f"{(total_pass / total_all if total_all else 0):>7.0%}")

    # Abstention is reported separately because it is the metric a counsellor
    # is judged on: refusing exactly when the corpus cannot support an answer.
    refuse_want = sum(1 for c in cases if c.get("expect_answered") is False)
    refuse_got = sum(1 for c, r in zip(cases, records)
                     if c.get("expect_answered") is False
                     and r["answered"] is False)
    answer_want = sum(1 for c in cases if c.get("expect_answered") is True)
    answer_got = sum(1 for c, r in zip(cases, records)
                     if c.get("expect_answered") is True and r["answered"] is True)
    print("\nabstention")
    if refuse_want:
        print(f"  refused when it should:      {refuse_got}/{refuse_want}")
    if answer_want:
        print(f"  answered when it should:     {answer_got}/{answer_want} "
              f"({answer_want - answer_got} false refusals)")

    if eval_retrieval_cases > 0:
        print("\nretrieval")
        print(f"  Recall@10:                   {eval_recall10_sum / eval_retrieval_cases:>7.1%}")
        print(f"  MRR:                         {eval_mrr_sum / eval_retrieval_cases:>7.3f}")

    if report_path:
        report_path.write_text(json.dumps(records, ensure_ascii=False, indent=1),
                               encoding="utf-8")
        print(f"\nwrote {report_path}")
    return 0 if total_pass == total_all else 1


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="validate the eval set against the corpus; no LLM calls")
    ap.add_argument("--kind", action="append", default=None,
                    help=f"only run this kind (repeatable). one of: {', '.join(sorted(KINDS))}")
    ap.add_argument("--limit", type=int, default=None, help="run at most N cases")
    ap.add_argument("--id", action="append", default=None, help="run only these case ids")
    ap.add_argument("--sleep", type=float, default=3.0,
                    help="seconds between live calls (provider rate limits)")
    ap.add_argument("--json", type=Path, default=None, help="write a JSON report here")
    args = ap.parse_args()

    raw = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    cases = raw["cases"] if isinstance(raw, dict) else raw

    if args.kind:
        unknown = set(args.kind) - KINDS
        if unknown:
            print(f"unknown kind(s): {sorted(unknown)}", file=sys.stderr)
            return 2
        cases = [c for c in cases if c.get("kind") in set(args.kind)]
    if args.id:
        cases = [c for c in cases if c.get("id") in set(args.id)]
    if args.limit is not None:
        cases = cases[:args.limit]
    if not cases:
        print("no cases selected", file=sys.stderr)
        return 2

    return dry_run(cases) if args.dry_run else live_run(cases, args.sleep, args.json)


if __name__ == "__main__":
    sys.exit(main())
