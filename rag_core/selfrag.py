"""Self-RAG: let the system notice its own retrieval failed, and fix it.

THE FAILURE THIS EXISTS FOR
At 15 colleges, retrieval could not really miss — everything fit in the prompt.
At 38,700 it misses constantly and SILENTLY: the student asks for "affordable
BCA in Vizag", the extractor writes city="Vizag" while the corpus says
"Visakhapatnam", the prefilter returns zero rows, and MIVI truthfully but
uselessly says nothing matched. The answer is well-formed, grounded, and wrong
in the only way that matters — the colleges were there.

Deterministic verification (pipeline._verify) cannot catch this. It checks that
what was SAID is supported by what was RETRIEVED; it has nothing to say about
what was never retrieved at all. This module closes that gap.

WHY IT IS CONDITIONAL, NOT ALWAYS-ON
Every grading call is a real LLM round trip (~1-2 s) on top of the router and
generator MIVI already pays for. Running it on every query would make the
common case — a question that retrieved fine — noticeably slower to buy nothing.
So the loop is gated on cheap, deterministic signals that retrieval actually
struggled (zero rows, a suspiciously small result set, a refusal). Those are
computed in code from data we already have, at zero cost.

THE LOOP (bounded at 2 extra retrievals, hard)
    retrieve -> cheap signal says "this looks wrong"?
             -> grade (LLM, small model): what is missing, what should change?
             -> apply a REPAIR to the filters
             -> retrieve again
Unbounded refinement is a latency hole with sharply diminishing returns; two
attempts recover the common cases (a spelling variant, an over-tight budget,
a filter the student never actually asked for) and stop.
"""
from __future__ import annotations

import json
import re
import sys

from . import config
from .llm import chat

# Hard ceiling on extra retrieval rounds. Not a tuning knob: each round is a
# full retrieve + an LLM grade, and a student waiting 12 s for a better answer
# has already left.
MAX_ROUNDS = 2

# Below this many hits for a question that named real constraints, retrieval
# probably over-filtered rather than genuinely found little.
THIN_RESULTS = 3


# ---------------------------------------------------------------------------
# Cheap gate — pure code, no LLM
# ---------------------------------------------------------------------------

def needs_repair(hits: list, filters: dict, total: int, question_kind: str) -> str | None:
    """Should we spend an LLM call trying to repair retrieval? Returns the
    reason, or None to proceed normally.

    Deliberately conservative. A false positive here costs a student ~2 s on a
    question that was already answerable; a false negative costs them a wrong
    "nothing found". Both matter, so this only fires on signals that are
    genuinely abnormal rather than merely unlucky.
    """
    stated = {k: v for k, v in (filters or {}).items()
              if v not in (None, "", [], False) and k != "include_abroad"}

    if not hits:
        # Zero rows with constraints stated is the classic over-filter. Zero
        # rows with NO constraints means the corpus really has nothing (or the
        # question was not about colleges) — repairing filters cannot help.
        return "zero_results" if stated else None

    # A budget/location question that returns a handful out of 38,700 is
    # suspicious; the same count for a single named college is expected.
    if (total <= THIN_RESULTS and len(stated) >= 2
            and question_kind in ("enumerate", "recommend")):
        return "thin_results"

    return None


def refusal_signal(result: dict, hits: list) -> bool:
    """The generator refused while holding non-empty context. Either the cards
    genuinely lack the field (correct — absence questions must refuse) or
    retrieval brought the wrong colleges. Worth one grading call to tell those
    apart, because they look identical from here."""
    return bool(hits) and result.get("answered") is False


# ---------------------------------------------------------------------------
# Grading — one small-model call
# ---------------------------------------------------------------------------

GRADE_SYSTEM = """You are the retrieval critic for a college-search assistant covering
~38,700 Indian colleges. A database query was run from a student's question and
returned too little. Decide WHY and propose ONE repair.

You are given the question, the filters that were applied, and how many
colleges matched.

Return ONLY JSON:
{
  "diagnosis": "over_filtered" | "wrong_spelling" | "genuinely_absent" | "not_a_search" | "fallback_web" | "fallback_deep_research",
  "drop_filters": [<filter names that the student never actually stated>],
  "fix_city": <corrected city/state spelling as the CORPUS would store it, or null>,
  "relax_budget_to": <int rupees per year, or null>,
  "extra_terms": [<alternative course words to try, e.g. "BCA" -> "Bachelor of Computer Applications">],
  "reason": "<one short sentence>"
}

Rules:
- "over_filtered": the filters are stricter than the student's words. Name the
  ones to drop. A filter the student never said is the most common cause.
- "wrong_spelling": an Indian place or course name written differently from the
  corpus. Correct it in fix_city / extra_terms. Examples: Vizag ->
  Visakhapatnam, Bangalore -> Bengaluru, Trivandrum -> Thiruvananthapuram,
  Pondicherry -> Puducherry, Gurgaon -> Gurugram, Baroda -> Vadodara.
- "genuinely_absent": the constraints are faithful and the answer really is
  "few or none". Propose NO repair. This is a correct and common verdict —
  choose it rather than inventing a relaxation that would answer a DIFFERENT
  question than the student asked.
- "not_a_search": the message was not a college search at all.
- "fallback_web": use this if the question requires real-time facts, current events, or news that a static database cannot answer.
- "fallback_deep_research": use this if the question is highly complex, comparative ("which is better X or Y"), or requires synthesizing data from multiple external sources.
- NEVER relax a budget by more than ~25%, and only when the student's wording
  was approximate ("around", "about", "under ~2 lakh"). A hard budget is a hard
  constraint: showing a student colleges they cannot afford is worse than
  showing them nothing.
- Propose the SMALLEST repair that could plausibly work."""


def grade(question: str, filters: dict, total: int, reason: str,
          calls: list | None = None) -> dict | None:
    """Ask the small model what to change. None if the call fails — retrieval
    repair is an enhancement and must never break the answer path."""
    payload = {
        "question": question[:600],
        "filters_applied": {k: v for k, v in (filters or {}).items() if v not in (None, "", [])},
        "colleges_matched": total,
        "signal": reason,
    }
    try:
        raw = chat(
            [{"role": "system", "content": GRADE_SYSTEM},
             {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            model=config.FAST_MODEL, json_mode=True, temperature=0.0,
            max_tokens=300, usage_log=calls,
        )
        text = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL).strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
        out = json.loads(text)
        return out if isinstance(out, dict) else None
    except Exception as exc:  # noqa: BLE001
        print(f"[selfrag] grade failed: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------------

# Filters retrieval may drop when the critic says the student never stated them.
# college_type, city and state are NOT droppable here: silently widening a
# location or turning a government-college request into a private one answers a
# question the student did not ask. Those are corrected (fix_city) or left be.
_DROPPABLE = {"program_level", "hostel_required", "entrance_exam",
              "min_tuition_inr", "course_terms"}


def apply_repair(filters: dict, verdict: dict) -> tuple[dict, str] | None:
    """Build the repaired filter set. Returns (filters, human_note) or None
    when no safe repair applies."""
    if not verdict:
        return None
    diagnosis = str(verdict.get("diagnosis") or "")
    if diagnosis in ("genuinely_absent", "not_a_search"):
        return None
    if diagnosis in ("fallback_web", "fallback_deep_research"):
        return filters, diagnosis

    new = dict(filters or {})
    notes: list[str] = []

    for key in verdict.get("drop_filters") or []:
        if key in _DROPPABLE and key in new:
            new.pop(key)
            notes.append(f"dropped {key}")

    city = verdict.get("fix_city")
    if isinstance(city, str) and city.strip():
        old = new.get("city") or new.get("state") or ""
        if city.strip().lower() != str(old).lower():
            new["city"] = city.strip()
            notes.append(f"read '{old}' as '{city.strip()}'")

    budget = verdict.get("relax_budget_to")
    cur = filters.get("max_tuition_inr") if filters else None
    if isinstance(budget, (int, float)) and cur:
        # Enforce the 25% ceiling in CODE. The prompt asks for it, but a budget
        # is the one constraint where a disobedient model does real harm, so it
        # is not left to the model's discretion.
        capped = int(min(float(budget), cur * 1.25))
        if capped > cur:
            new["max_tuition_inr"] = capped
            notes.append(f"widened budget to Rs {capped:,}")

    terms = [t for t in (verdict.get("extra_terms") or []) if isinstance(t, str) and t.strip()]
    if terms:
        existing = list(new.get("course_terms") or [])
        merged = existing + [t for t in terms if t.lower() not in {e.lower() for e in existing}]
        if merged != existing:
            new["course_terms"] = merged[:6]
            notes.append(f"also tried {', '.join(terms[:3])}")

    if not notes or new == filters:
        return None
    return new, "; ".join(notes)


def context_note(note: str, verdict: dict) -> str:
    """A line for the generator so the ANSWER stays honest about what happened.

    Without this the student is silently shown results for a query they did not
    type. Telling them "I read Vizag as Visakhapatnam" is both more useful and
    more trustworthy than quietly substituting it.
    """
    reason = str((verdict or {}).get("reason") or "").strip()
    return (f"RETRIEVAL NOTE (state this naturally in one short clause, do not "
            f"belabour it): the first search found too little, so it was retried "
            f"after {note}."
            + (f" {reason}" if reason else "") + "\n")
