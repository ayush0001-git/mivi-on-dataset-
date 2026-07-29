"""Does MIVI give the SAME answer when the SAME question is typed in three scripts?

WHY THIS EXISTS
Indian students type the same question three ways, often within one session:

    English     "Which is the cheapest GNM college in Haryana?"
    Hinglish    "Haryana mein sabse sasta GNM college kaunsa hai?"
    Devanagari  "हरियाणा में सबसे सस्ता GNM कॉलेज कौन सा है?"

The first is what benchmarks test. The second is what students actually write,
and no public retrieval benchmark covers romanised Hindi — MTEB, MIRACL and
XTREME-UP all evaluate native script. So an architecture review can tell you
which embedding model wins on Devanagari and still tell you nothing about the
mode your users are in.

This measures it on MIVI's own corpus instead of trusting a leaderboard.

WHAT IT MEASURES, AND WHY THIS RATHER THAN COSINE
Not embedding similarity — the ANSWER. A card retrieved at rank 3 instead of
rank 1 does not matter here, because counts and superlatives are computed in
SQL over the whole matching set and the generator is verified against them.
What matters is whether a student asking in Hinglish is told a different college
or a different fee than a student asking in English. That is the failure a
student would actually experience, and it is invisible to a cosine metric.

Reported per group:
  filters    did the router extract the same constraints from each script?
  overlap    Jaccard over the retrieved college ids
  top1       did all three scripts put the same college first?

A drop concentrated in `filters` is a ROUTER problem (cheap to fix: the
heuristic floor in pipeline.py already covers Latin, not Devanagari).
A drop in `overlap` with filters intact is a RANKER problem, and only then is
the embedding model implicated at all.

Run:  python -m evals.script_parity
      python -m evals.script_parity --verbose
"""
from __future__ import annotations

import argparse
import sys

from rag_core import pipeline, retrieve

# Each group is the SAME question in English, Hinglish and Devanagari.
# Chosen to cover the query shapes that actually reach MIVI: superlative,
# filtered search, count, eligibility, and an exact institution name.
GROUPS: list[dict] = [
    {
        "name": "cheapest-gnm-haryana",
        "shape": "superlative + course + state",
        "english": "Which is the cheapest GNM college in Haryana?",
        "hinglish": "Haryana mein sabse sasta GNM college kaunsa hai?",
        "devanagari": "हरियाणा में सबसे सस्ता GNM कॉलेज कौन सा है?",
    },
    {
        "name": "oldest-kerala",
        "shape": "superlative + state",
        "english": "What is the oldest college in Kerala?",
        "hinglish": "Kerala ka sabse purana college kaunsa hai?",
        "devanagari": "केरल का सबसे पुराना कॉलेज कौन सा है?",
    },
    {
        "name": "btech-punjab-budget",
        "shape": "course + state + budget",
        "english": "BTech colleges in Punjab under 1.5 lakh",
        "hinglish": "Punjab mein 1.5 lakh ke andar BTech college batao",
        "devanagari": "पंजाब में डेढ़ लाख के अंदर बीटेक कॉलेज बताओ",
    },
    {
        "name": "count-mbbs-up",
        "shape": "count + course + state",
        "english": "How many colleges offer MBBS in Uttar Pradesh?",
        "hinglish": "Uttar Pradesh mein kitne college MBBS karate hain?",
        "devanagari": "उत्तर प्रदेश में कितने कॉलेज एमबीबीएस कराते हैं?",
    },
    {
        "name": "hostel-dehradun",
        "shape": "city + facility",
        "english": "Colleges in Dehradun with hostel facility",
        "hinglish": "Dehradun mein hostel wale college kaun se hain?",
        "devanagari": "देहरादून में हॉस्टल वाले कॉलेज कौन से हैं?",
    },
    {
        "name": "govt-engineering-uttarakhand",
        "shape": "type + course + state",
        "english": "Government engineering colleges in Uttarakhand",
        "hinglish": "Uttarakhand ke sarkari engineering college",
        "devanagari": "उत्तराखंड के सरकारी इंजीनियरिंग कॉलेज",
    },
]

SCRIPTS = ("english", "hinglish", "devanagari")
TOP_K = 10

# Filter keys that change WHICH colleges are admissible. A difference in any of
# these is a real divergence; anything else is presentation.
_HARD_KEYS = ("state", "city", "college_type", "course_terms", "program_level",
              "max_tuition_inr", "min_tuition_inr", "hostel_required",
              "entrance_exam")


def _route(question: str) -> dict:
    """Filters the pipeline would use, via its own heuristic floor.

    Deliberately the FLOOR and not the LLM router: the floor is deterministic,
    so this measures script handling rather than one sample of a rate-limited
    model's mood.
    """
    r = pipeline._heuristic_route(question)
    f = pipeline._clean_filters(r.get("filters"))
    return {k: f[k] for k in _HARD_KEYS if k in f}


def _ids(question: str, filters: dict) -> list[str]:
    try:
        hits = retrieve.search(question, filters=filters or None, top_k=TOP_K)
    except Exception as exc:  # noqa: BLE001
        print(f"    retrieval failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return []
    return [h.get("college_id") for h in hits if h.get("college_id")]


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def run(verbose: bool = False) -> dict:
    rows = []
    for g in GROUPS:
        per: dict[str, dict] = {}
        for s in SCRIPTS:
            q = g[s]
            f = _route(q)
            per[s] = {"q": q, "filters": f, "ids": _ids(q, f)}

        base = per["english"]
        filters_match = all(per[s]["filters"] == base["filters"] for s in SCRIPTS)
        overlaps = {s: _jaccard(base["ids"], per[s]["ids"])
                    for s in ("hinglish", "devanagari")}
        tops = {s: (per[s]["ids"][0] if per[s]["ids"] else None) for s in SCRIPTS}
        top1_match = len({t for t in tops.values() if t}) == 1 and all(tops.values())

        rows.append({"group": g["name"], "shape": g["shape"],
                     "filters_match": filters_match, "overlaps": overlaps,
                     "top1_match": top1_match, "per": per})

        flag = "  " if (filters_match and top1_match) else "!!"
        print(f"{flag} {g['name']:30} filters={'same' if filters_match else 'DIFFER'}"
              f"  hinglish={overlaps['hinglish']:.2f}"
              f"  devanagari={overlaps['devanagari']:.2f}"
              f"  top1={'same' if top1_match else 'DIFFER'}")
        if verbose or not (filters_match and top1_match):
            for s in SCRIPTS:
                print(f"      {s:11} {per[s]['filters']}")

    n = len(rows)
    fm = sum(r["filters_match"] for r in rows)
    t1 = sum(r["top1_match"] for r in rows)
    mh = sum(r["overlaps"]["hinglish"] for r in rows) / max(n, 1)
    md = sum(r["overlaps"]["devanagari"] for r in rows) / max(n, 1)

    print()
    print(f"groups                       {n}")
    print(f"same filters across scripts  {fm}/{n}")
    print(f"same top-1 college           {t1}/{n}")
    print(f"mean overlap  hinglish       {mh:.2f}")
    print(f"mean overlap  devanagari     {md:.2f}")
    print()
    # The diagnosis is the point of the script, so state it rather than leaving
    # the reader to infer it from four numbers.
    if fm < n:
        print("DIAGNOSIS: the ROUTER is the weak link — the same question yields "
              "different hard filters depending on script, so the candidate SET "
              "differs before ranking runs. Fix the extractor, not the encoder.")
    elif min(mh, md) < 0.5:
        print("DIAGNOSIS: filters agree but the RANKER diverges. This is the only "
              "case where the embedding model is implicated.")
    else:
        print("DIAGNOSIS: script parity holds. No embedding change is justified "
              "by this evidence.")
    return {"groups": n, "filters_match": fm, "top1_match": t1,
            "mean_overlap_hinglish": round(mh, 3),
            "mean_overlap_devanagari": round(md, 3), "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verbose", action="store_true",
                    help="print extracted filters for every group")
    run(verbose=ap.parse_args().verbose)


if __name__ == "__main__":
    main()
