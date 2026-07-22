"""Run the eval set against the live pipeline and print pass rates.

    python evals/run_evals.py

Imports the pipeline in-process (one embedding-model load for all cases).
Each case checks the JSON output against expectations: answered flag, ids that
must / must not be cited, required phrases, script of the reply, and a special
guard for the placement=0 trap.

Cases carry an optional "tier": "core" (default) or "adversarial". Core cases
are the ones the system is expected to pass; adversarial cases are deliberately
hard (subjective judgement, multi-hop, free-text-only constraints) and are
reported separately — a failing test I wrote myself says more than a
suspiciously perfect one. Two extra numbers are reported on top of the pass
rate: an ABSTENTION score (does it refuse exactly when it should?) which maps
directly to the brief's "graceful I-don't-know" requirement.
"""
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from rag_core.pipeline import answer_question  # noqa: E402

DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def check(expect, result):
    fails = []
    citations = result.get("citations", [])
    answer_lower = result.get("answer", "").lower()

    if "answered" in expect and result.get("answered") != expect["answered"]:
        fails.append(f"answered={result.get('answered')} (expected {expect['answered']})")
    for c in expect.get("must_cite", []):
        if c not in citations:
            fails.append(f"missing citation {c}")
    if expect.get("must_cite_any") and not any(c in citations for c in expect["must_cite_any"]):
        fails.append(f"none of must_cite_any {expect['must_cite_any']} were cited")
    for c in expect.get("must_not_cite", []):
        if c in citations:
            fails.append(f"must-not-cite {c} was cited")
    if expect.get("answer_any") and not any(t.lower() in answer_lower for t in expect["answer_any"]):
        fails.append(f"answer contains none of {expect['answer_any']}")
    for t in expect.get("answer_none", []):
        if t.lower() in answer_lower:
            fails.append(f"answer contains forbidden phrase {t!r}")
    if expect.get("answer_devanagari") and not DEVANAGARI.search(result.get("answer", "")):
        fails.append("answer is not in Devanagari/Hindi")
    if expect.get("special") == "c006_guard":
        if "C006" in citations and not any(
            p in answer_lower
            for p in ["not reported", "not applicable", "no campus placement",
                      "report placement", "not report", "no placement data",
                      "placement data is not"]
        ):
            fails.append("cites C006 without explaining that 0 = not reported")
    return fails


def main():
    cases = json.loads((Path(__file__).parent / "eval_set.json").read_text(encoding="utf-8"))
    stats = {"core": [0, 0], "adversarial": [0, 0]}  # tier -> [passed, total]
    # abstention: cases with an explicit answered expectation
    refuse_expected = refuse_correct = answer_expected = answer_correct = 0

    for i, case in enumerate(cases):
        if i:
            time.sleep(3)  # stay inside free-tier requests-per-minute limits
        result = answer_question(case["question"])
        fails = check(case["expect"], result)
        tier = case.get("tier", "core")
        stats[tier][1] += 1
        stats[tier][0] += not fails

        exp = case["expect"]
        if exp.get("answered") is False:
            refuse_expected += 1
            refuse_correct += result.get("answered") is False
        elif exp.get("answered") is True:
            answer_expected += 1
            answer_correct += result.get("answered") is True

        status = "PASS" if not fails else "FAIL"
        print(f"[{status}] ({tier[:3]}) {case['id']}")
        if fails:
            for f in fails:
                print(f"        - {f}")
            print(f"        answer: {result.get('answer', '')[:220]}")
            print(f"        citations: {result.get('citations')}")

    cp, ct = stats["core"]
    ap, at = stats["adversarial"]
    print(f"\nCore pass rate:        {cp}/{ct} ({cp / ct:.0%})" if ct else "\nNo core cases")
    if at:
        print(f"Adversarial pass rate: {ap}/{at} ({ap / at:.0%})  "
              f"(deliberately hard — failures here are expected and informative)")
    print("\nAbstention (the 'graceful I-don't-know' the brief asks for):")
    if refuse_expected:
        print(f"  Correctly refused when it should: {refuse_correct}/{refuse_expected} "
              f"({refuse_correct / refuse_expected:.0%})")
    if answer_expected:
        false_refusals = answer_expected - answer_correct
        print(f"  Did NOT wrongly refuse answerable questions: "
              f"{answer_correct}/{answer_expected} "
              f"({false_refusals} false refusals)")


if __name__ == "__main__":
    main()
