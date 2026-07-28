"""Loop-RAG: decide whether a FINISHED answer ships, and what to change if not.

WHERE THIS SITS
    selfrag.py   repairs RETRIEVAL before generation (wrong city spelling, a
                 filter the student never stated). It never sees an answer.
    pipeline._verify  checks the ANSWER deterministically — citations exist,
                 every number traces to a cited card, named colleges are cited —
                 and retries the generator on the SAME context until it passes
                 or fails closed.
    this module  is the OUTER loop: the answer is complete and internally
                 consistent, and the question is whether it is good enough to
                 put in front of a student. Neither layer above can ask that,
                 because both are satisfied by a well-formed answer to the
                 wrong question.

WHY THE DIAGNOSIS MATTERS MORE THAN THE RETRY
The four ways a completed answer can be not-good-enough need four different
fixes, and applying the wrong one makes things worse, not slower:

  ungrounded      -> REGENERATE on the same context. Retrieval was fine; the
                     prose overreached. Re-retrieving would throw away good
                     cards and re-roll the same dice.
  wrong context   -> RE-RETRIEVE with different filters/query. Regenerating
                     here just produces a better-written answer about the wrong
                     colleges, which is the more dangerous failure.
  ambiguous       -> ASK ONE QUESTION. "Which college is best?" over 38,700
                     colleges has no right answer; a confident list is a guess
                     dressed as a recommendation. One question beats two more
                     passes of guessing.
  data absent     -> SHIP THE REFUSAL. The catalogue holds no cutoffs, no seat
                     counts and no placement salaries (~100% empty for those
                     fields). A refusal naming an adjacent fact is the CORRECT
                     output, and looping on it burns 4 s to arrive at the same
                     honest answer — so this path must terminate, hard. It is
                     the single most important thing this module gets right.

THE BUDGET IS CODE, NOT A PROMPT
At most MAX_EXTRA_PASSES extra passes, and a wall-clock ceiling checked BEFORE
a pass is issued (a pass costs ~2-4 s: measured LLM latency is 1-3 s and
retrieval 0.29 s — encode 40 ms + count 37 ms + search 214 ms). A student
waiting 12 s has left, so the ceiling is on total answer time and a pass is
only issued when the estimated cost still fits inside it. LoopBudget also
refuses a repeat of a (action, diagnosis) pair it already paid for, and allows
at most one clarification ever — a second identical pass has no new information
to work with and can only cost latency.

The decision itself spends NO LLM call. Every signal is computed from data the
pipeline already has in hand, so the common case — a good answer — pays
microseconds to be recognised as good.

ADOPTION (pipeline.py needs no restructuring)
    budget = looprag.LoopBudget(started_at=t_start)
    while (plan := looprag.should_retry(result, hits, filters, question,
                                        budget=budget,
                                        question_kind=question_kind,
                                        total=total)):
        if plan.action == "retrieve":
            hits = _search(plan.query or search_query, plan.filters, top_k)
            ... rebuild user_msg ...
        user_msg += plan.instruction
        result = <regenerate + the existing _verify loop>
    # plan is None -> ship `result` exactly as the pipeline built it.

Self-test: python -m rag_core.looprag
"""
from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# The budget. Numbers, not vibes.
# ---------------------------------------------------------------------------

# Hard ceiling on extra passes. Two recovers the cases that are recoverable at
# all (wrong college retrieved, an over-tight filter, one bad generation); a
# third pass is a latency hole with no measured win.
MAX_EXTRA_PASSES = int(os.getenv("LOOP_MAX_PASSES", "2") or 2)

# Ceiling on TOTAL answer wall-clock, measured from the start of the request, so
# that one ~4 s pass still lands inside the ~12 s where students stop waiting.
#
# This deliberately DISABLES the loop on slow turns, and that is not a
# compromise: measured on the live pipeline just now, a rate-limited turn that
# failed over between providers took 19.9 s all by itself. There is no version
# of "one more pass" that helps that student. It is safe to give up because the
# pipeline already fails closed — when grounding cannot be verified it ships an
# honest "I couldn't verify this", not a guess. This loop only ever buys a
# BETTER answer, never a safer one, so running out of clock costs quality and
# never costs correctness.
WALL_CEILING_S = float(os.getenv("LOOP_WALL_CEILING_S", "9.0") or 9.0)

# What each pass is expected to cost. Used to refuse a pass we cannot finish
# rather than to start one we will abandon halfway.
PASS_COST_S = {
    "regenerate": 2.5,   # one generator call (1-3 s) + deterministic verify
    "retrieve": 4.0,     # retrieval (~0.3 s) + a generator call, both again
    "clarify": 2.0,      # one short generator call; the text is already written
}

# Above this many matching colleges, a question that stated no course, no place
# and no budget cannot have a defensible "best" answer — the shown list is an
# arbitrary slice. Deliberately generous: 200 is ~0.5% of the corpus, so this
# only fires on genuinely wide-open questions.
AMBIGUOUS_TOTAL = 200

# Diagnosis labels (also the metrics keys — keep them stable).
SHIP = "ship"                       # good enough, or correctly refused
UNGROUNDED = "ungrounded"           # prose overreached its context
WRONG_CONTEXT = "wrong_context"     # we answered about the wrong colleges
AMBIGUOUS = "ambiguous"             # any answer would be a guess
ABSENT = "absent"                   # the field is not in the catalogue at all
DEAD_END = "dead_end_refusal"       # refusal is right but offers nothing next
PROVIDER_ERROR = "provider_error"   # the LLM never answered


# ---------------------------------------------------------------------------
# Cheap text helpers (local on purpose: importing pipeline here would be a
# circular import, and these are three lines each)
# ---------------------------------------------------------------------------

def _norm(s) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s or "").lower())


def _norm_ws(s) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]+", " ", str(s or "").lower())).strip()


# Fields the corpus structurally does not carry. This list is not a guess: it
# mirrors GENERATOR_SYSTEM rule 5 ("NO admission cutoffs, NO seat counts and NO
# placement salary figures") plus the things no college catalogue holds at all
# (exam dates, papers, syllabi). A refusal about one of these is FINAL — no
# amount of re-retrieval can produce a number that does not exist.
_ABSENT_FIELD_RE = re.compile(
    r"cut\s?-?\s?offs?|closing rank|opening rank|merit list|"
    r"\bseats?\b|seat matrix|intake capacity|"
    r"\bl\.?p\.?a\.?\b|\bctc\b|\bpackages?\b|"
    r"placement (?:salary|package|percentage|record|stats)|"
    r"(?:average|highest|median|lowest) (?:salary|package|ctc)|"
    r"question paper|previous year paper|syllabus|"
    r"(?:counselling|counseling|exam|admission|application) date|last date|"
    r"kitni seats?|kitne seats?|kitna package|cutoff kitna|"
    r"placement kaisa|package kitna",
    re.I)

# ...and the fields it DOES carry. A question mixing both ("cutoff and fees
# for X") is still worth a retrieval fix for the answerable half, so the
# absent-field short-circuit only fires when nothing answerable was asked.
_ANSWERABLE_FIELD_RE = re.compile(
    r"\bfees?\b|tuition|kharch|\bcosts?\b|\bcourses?\b|programme?s?\b|degree|"
    r"hostel|scholarship|naac|nirf|affiliat|established|address|"
    r"entrance|eligib|admission process|\bnames?\b|list",
    re.I)

# SINGULAR forms only, and that is the whole trick. "Rajagiri College",
# "Vellore Institute of Technology" name one institution; "colleges in Kochi",
# "BCA colleges" name a CATEGORY. The self-test caught this as a real bug: with
# plurals included, every ordinary "colleges in <city>" question was diagnosed
# as "the question names a college we did not retrieve" and bought itself a
# pointless 4 s re-retrieval pass.
_INSTITUTION_WORD_RE = re.compile(
    r"college|university|institute|institution|academy|"
    r"vidyalaya|vidyapeeth|mahavidyalaya|polytechnic|iit|nit|iiit|aiims|iim|"
    r"nift|nlu|bits", re.I)

# A name's words sit next to the institution word; a PLACE sits behind a
# preposition. Scanning stops at these so "college in Kochi" never contributes
# "kochi" as if it were part of a college's name.
_PLACE_PREP = frozenset("in at near around from within inside outside ke me mein".split())

# Words that appear in thousands of Indian college names and therefore identify
# nothing. What is left after removing them is the discriminating part of a
# name ("vellore", "manipal", "thapar") — the only part worth matching on.
_GENERIC_NAME_TOKENS = frozenset("""
college colleges university universities institute institutes institution
academy school schools of the and for in at national indian government govt
public private deemed autonomous technology technological science sciences arts
commerce management engineering studies research centre center education
educational group campus society trust memorial shri sri smt late new india
bharat state central higher secondary degree first grade womens women mens
iit nit iiit vit srm bits iim nlu aiims nift
""".split())
# The acronyms on that last line are deliberately NOT discriminating: the corpus
# stores expanded names ("Indian Institute of Technology Delhi"), so matching a
# retrieved name against "iit" would fail for every correctly retrieved IIT and
# fire a re-retrieval on a perfectly good context. The city that follows the
# acronym does the discriminating instead.

# Question scaffolding that is not part of any name. The verbs on the third line
# were added after a live run read "help me CHOOSE a college" as a named college
# called "choose" — which then suppressed the ambiguity check on the most
# ambiguous question in the eval set.
_ASK_STOPWORDS = frozenset("""
does do is are was were what which where who how much many tell me about give
show list fee fees tuition cost costs course courses offer offers offered
admission cutoff placement details detail info information good best cheap
cheapest near nearby any there this that their its it a an the
choose decide pick help want need looking find apply join study get know
recommend suggest compare start take tell think should would could like prefer
kya kaise kitna kitni kaun konsa kaunsa batao chahiye mujhe mera meri hai hain
ka ki ke ko se me in on for and or with without please can i we they you my your
""".split())

# A course the question names concretely — enough specificity that the answer
# is not a shot in the dark, so no clarification is warranted.
_DEGREE_IN_Q_RE = re.compile(
    r"\b(?:mbbs|bds|bams|bhms|bpt|bsn|llb|llm|mba|pgdm|mca|bca|bba|"
    r"b\.?com|m\.?com|b\.?sc|m\.?sc|b\.?tech|m\.?tech|b\.?pharm|d\.?pharm|"
    r"b\.?ed|m\.?ed|bhm|bfa|gnm|anm|phd|iti|diploma|"
    r"nursing|engineering|medical|medicine|law|pharmacy|management|design|"
    r"architecture|agriculture|paramedical|hotel management|journalism)\b",
    re.I)

# Wide-open asks. Every one of these is answerable ONLY once a field, a place
# or a budget is known.
_VAGUE_ASK_RE = re.compile(
    r"best college|good college|nice college|top college|which college|"
    r"what college|any college|koi college|suggest|recommend|"
    r"kaun\s?sa college|kon\s?sa college|kaunsa|konsa|"
    r"a[cn]?ch?ha college|sabse a[cn]?ch?ha|college batao|"
    r"where should i (?:study|apply|go)|what should i do|"
    r"help me (?:choose|decide|pick)|kahan (?:padhu|padhoon|admission)",
    re.I)

# Money written in an answer. Only used to notice that an UNCITED answer is
# stating figures — the tight numeric check lives in pipeline._numeric_violations.
_MONEY_IN_ANSWER_RE = re.compile(r"(?:rs\.?|₹|inr)\s*[\d][\d,]{3,}", re.I)

# pipeline's own fail-closed texts, matched on reason_if_unanswered so we can
# tell "the model refused" from "our verifier rejected the model".
_VERIFY_FAILED_RE = re.compile(r"grounding verification failed", re.I)
_PROVIDER_ERROR_RE = re.compile(r"transient provider error", re.I)


def _stated(filters: dict) -> dict:
    """The constraints the student actually imposed. include_abroad is a view
    setting the pipeline manages, never something a student asked for."""
    return {k: v for k, v in (filters or {}).items()
            if v not in (None, "", [], False) and k != "include_abroad"}


def _named_institution(question: str) -> tuple[set[str], str]:
    """(discriminating tokens, retrieval query) for a college the question names.

    The words around the institution word are where a name lives ("does VELLORE
    INSTITUTE OF TECHNOLOGY offer..."), which is far more reliable than
    capitalisation — students type lowercase. The scan grows outward from that
    word and stops at question scaffolding ("does", "fees") or a preposition,
    because everything past those belongs to the question, not to the name.
    """
    words = _norm_ws(question).split()
    idx = next((i for i, w in enumerate(words) if _INSTITUTION_WORD_RE.fullmatch(w)),
               None)
    if idx is None:
        return set(), ""
    start = idx
    for j in range(idx - 1, max(-1, idx - 6), -1):
        if words[j] in _ASK_STOPWORDS or words[j] in _PLACE_PREP:
            break
        start = j
    end = idx + 1
    for j in range(idx + 1, min(len(words), idx + 5)):
        if words[j] in _ASK_STOPWORDS or words[j] in _PLACE_PREP:
            break
        end = j + 1
    window = words[start:end]
    query = " ".join(window)
    tokens = {w for w in window if len(w) >= 4 and w not in _GENERIC_NAME_TOKENS}
    return tokens, query


def _name_mismatch(question: str, hits: list, filters: dict) -> tuple[set[str], str] | None:
    """The question names a college whose discriminating words appear in NO
    retrieved name. That is retrieval missing the subject of the question — the
    one failure a better-written answer cannot fix."""
    tokens, query = _named_institution(question)
    # A place is not a name. If the word also arrived as a place filter, the
    # router already read it as geography and it discriminates nothing.
    places = {t for key in ("city", "state") for t in _norm_ws(filters.get(key)).split()}
    tokens -= places
    if not tokens:
        return None
    haystack = " ".join(_norm_ws(h.get("name")) for h in hits)
    if any(re.search(rf"\b{re.escape(t)}", haystack) for t in tokens):
        return None
    return tokens, query


def _place_mismatch(hits: list, filters: dict) -> tuple[str, str] | None:
    """A place filter was applied and not one retrieved college is in it.

    Happens when the pipeline relaxed a place it could not match (a spelling
    the corpus does not use) — the student asked about one city and is looking
    at another, which is worse than being told nothing matched.
    """
    if not hits:
        return None
    for key in ("city", "state"):
        want = _norm(filters.get(key))
        if not want:
            continue
        got = [_norm(h.get(key)) for h in hits]
        if not any(g and (g == want or want in g or g in want) for g in got):
            return key, str(filters[key])
    return None


def _resolve_place(value: str, kind: str) -> tuple[str, str | None]:
    """Corpus spelling for a place the student typed, via the curated alias
    table (microseconds, deterministic). Never fatal: an unresolvable place is
    returned unchanged so the caller can decide to drop the filter instead."""
    try:
        from .store import aliases

        return aliases.resolve(value, kind)
    except Exception as exc:  # noqa: BLE001 - alias resolution is optional
        print(f"[looprag] alias lookup unavailable: {exc}", file=sys.stderr)
        return value, None


# Filters this loop may drop. Money and geography are absent on purpose: both
# selfrag and this module refuse to widen them, because a student shown
# colleges they cannot afford, or in a city they cannot move to, has been
# answered a different question than the one they asked.
_DROPPABLE = ("program_level", "hostel_required", "entrance_exam", "min_tuition_inr")


def _cited_here(result: dict, hits: list) -> list[str]:
    ids = {h.get("college_id") for h in hits}
    return [c for c in (result.get("citations") or []) if isinstance(c, str) and c in ids]


def _ghost_citations(result: dict, hits: list) -> list[str]:
    ids = {h.get("college_id") for h in hits}
    return [c for c in (result.get("citations") or [])
            if isinstance(c, str) and c not in ids]


def _names_a_hit(result: dict, hits: list) -> bool:
    """The prose names a retrieved college. 12+ normalised characters, the same
    bar pipeline._name_violations uses — shorter names ("City College") collide
    with ordinary prose and would make this fire on nothing."""
    answer = _norm(result.get("answer"))
    return any(len(_norm(h.get("name"))) >= 12 and _norm(h.get("name")) in answer
               for h in hits)


# ---------------------------------------------------------------------------
# Diagnosis — pure, no LLM, no side effects beyond an alias lookup
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Diagnosis:
    label: str
    action: str | None          # None => ship the answer as it is
    reason: str                 # one sentence, goes into logs and metrics
    ask_about: tuple[str, ...] = ()
    filters: dict | None = None
    query: str | None = None
    absent_field: str | None = None


def diagnose(result: dict, hits: list, filters: dict, question: str, *,
             question_kind: str | None = None, total: int | None = None,
             history: list | None = None) -> Diagnosis:
    """Why this answer is (or is not) shippable. Never raises."""
    hits = list(hits or [])
    filters = dict(filters or {})
    question = str(question or "")
    answer = str(result.get("answer") or "")
    answered = bool(result.get("answered"))
    reason_txt = str(result.get("reason_if_unanswered") or "")
    stated = _stated(filters)
    cited = _cited_here(result, hits)
    # "?" in the answer is the cheapest possible check for "the counsellor
    # already asked instead of guessing" — and it is the common case, because
    # GENERATOR_SYSTEM rule 12a tells it to. Measured on the live pipeline: "help
    # me choose a college" came back as one warm line plus a follow-up question,
    # i.e. exactly what a clarify pass would have produced. Spending 2 s to
    # rewrite a good question into a slightly better one is the kind of loop that
    # makes a system slower without making it righter.
    already_asking = "?" in answer

    # --- terminal cases first: things no pass can improve -------------------
    if _PROVIDER_ERROR_RE.search(reason_txt):
        # The provider never answered. llm.chat already retried up to 4x and
        # failed over; another pass waits on the same dead endpoint.
        return Diagnosis(PROVIDER_ERROR, None,
                         "the provider never returned an answer; retrying waits on "
                         "the same outage")

    if question_kind == "profile_share":
        # A counselling turn states no facts and cites nothing by design
        # (pipeline rule 12a). There is nothing here to ground or re-retrieve.
        return Diagnosis(SHIP, None, "counselling turn: no factual claim to verify")

    # --- ungrounded: retrieval was fine, the prose was not ------------------
    if _VERIFY_FAILED_RE.search(reason_txt):
        return Diagnosis(UNGROUNDED, "regenerate",
                         "the pipeline's own verifier rejected every attempt on this "
                         "context; try a deliberately smaller answer")

    ghosts = _ghost_citations(result, hits)
    if ghosts:
        return Diagnosis(UNGROUNDED, "regenerate",
                         f"cites {len(ghosts)} id(s) that are not in the retrieved "
                         f"context: {ghosts[:3]}")

    if answered and not cited and hits and (_names_a_hit(result, hits)
                                            or _MONEY_IN_ANSWER_RE.search(answer)):
        return Diagnosis(UNGROUNDED, "regenerate",
                         "states college facts (a name or a rupee figure) without "
                         "citing the college they came from")

    # --- a refusal: which KIND of refusal decides everything ----------------
    if not answered:
        absent = _ABSENT_FIELD_RE.search(question) or _ABSENT_FIELD_RE.search(reason_txt)
        if absent and not _ANSWERABLE_FIELD_RE.search(question):
            field = absent.group(0)
            if cited or not hits:
                # THE PATH THAT MUST NOT LOOP. The catalogue has no cutoffs,
                # seats or salary figures; a refusal that names an adjacent
                # fact is the correct, finished answer.
                return Diagnosis(ABSENT, None,
                                 f"'{field}' is not a field the catalogue carries; the "
                                 f"refusal is correct and complete",
                                 absent_field=field)
            return Diagnosis(DEAD_END, "regenerate",
                             f"correctly refuses '{field}' but cites nothing, so the "
                             f"student is left with no next step",
                             absent_field=field)

        mismatch = _name_mismatch(question, hits, filters)
        if mismatch:
            tokens, query = mismatch
            # Lookup shape: the student named a college, so the money/geography
            # filters that removed it are exactly what has to come off. Abroad
            # is force-included — a foreign institution must not be missed just
            # because the default view is domestic.
            retry_filters = {"include_abroad": True}
            return Diagnosis(WRONG_CONTEXT, "retrieve",
                             f"the question names a college ({'/'.join(sorted(tokens))}) "
                             f"that appears in none of the {len(hits)} retrieved names",
                             filters=retry_filters, query=query)

        place = _place_mismatch(hits, filters)
        if place:
            key, want = place
            fixed, note = _resolve_place(want, key)
            retry_filters = dict(filters)
            if note:
                retry_filters[key] = fixed
                why = f"{note}; no retrieved college was in '{want}'"
            else:
                retry_filters.pop(key, None)
                why = (f"no retrieved college is in '{want}' and it resolves to no "
                       f"corpus {key}; searching without the {key} filter")
            return Diagnosis(WRONG_CONTEXT, "retrieve", why, filters=retry_filters)

        if not hits and stated:
            droppable = [k for k in _DROPPABLE if k in stated]
            if droppable:
                retry_filters = {k: v for k, v in filters.items() if k != droppable[0]}
                return Diagnosis(WRONG_CONTEXT, "retrieve",
                                 f"nothing matched and '{droppable[0]}' is a constraint "
                                 f"the student did not necessarily impose",
                                 filters=retry_filters)
            # Only money and geography are left, and this loop does not widen
            # either. Refusing to loop IS the correct decision here.
            return Diagnosis(SHIP, None,
                             "nothing matched, and the only constraints left are the "
                             "budget/location the student actually set")

        if not hits and not already_asking and _is_vague(question, filters):
            return Diagnosis(AMBIGUOUS, "clarify",
                             "nothing matched and the question named no field, place or "
                             "budget to search on",
                             ask_about=_ask_axes(question, filters, history)[:1])

        if hits and not cited:
            return Diagnosis(DEAD_END, "regenerate",
                             "refuses while holding college cards but cites none of "
                             "them, so the answer offers no way forward")

        return Diagnosis(ABSENT, None,
                         "honest refusal with a cited adjacent fact: nothing to add")

    # --- answered, grounded, but possibly a guess ---------------------------
    if (len(cited) >= 2 and not already_asking
            and _is_vague(question, filters) and _many(total, hits)):
        return Diagnosis(AMBIGUOUS, "clarify",
                         f"presents {len(cited)} colleges as a recommendation while "
                         f"{total if total is not None else 'thousands of'} match and "
                         f"the student named no field, place or budget",
                         ask_about=_ask_axes(question, filters, history)[:1])

    return Diagnosis(SHIP, None, "grounded and specific enough to ship")


def _is_vague(question: str, filters: dict) -> bool:
    """No course, no place, no budget, no named college — and phrased as an
    open-ended ask. All four have to hold: a student who said "BCA in Kochi
    under 1 lakh" gets an answer, not a question."""
    stated = _stated(filters)
    if stated or _DEGREE_IN_Q_RE.search(question or ""):
        return False
    if _named_institution(question)[0]:
        return False
    return bool(_VAGUE_ASK_RE.search(question or ""))


def _many(total: int | None, hits: list) -> bool:
    """Are there far too many matches for an unconstrained pick to mean
    anything? With a real SQL total this is exact. Without one, a full page of
    hits from an unfiltered search over 38,700 colleges already implies
    thousands — but we stay conservative and require the page to be full."""
    if total is not None:
        return total >= AMBIGUOUS_TOTAL
    return len(hits) >= 8


_AXIS_ORDER = ("course", "location", "budget")


def _ask_axes(question: str, filters: dict, history: list | None) -> tuple[str, ...]:
    """Missing dimensions, most decision-changing first. Field before place
    before budget: the field decides which colleges are even candidates."""
    have_course = bool(_stated(filters).get("course_terms")
                       or _DEGREE_IN_Q_RE.search(question or ""))
    have_place = bool(_stated(filters).get("city") or _stated(filters).get("state"))
    have_budget = bool(_stated(filters).get("max_tuition_inr"))
    got = {"course": have_course, "location": have_place, "budget": have_budget}
    return tuple(a for a in _AXIS_ORDER if not got[a])


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------

class LoopBudget:
    """What this question is still allowed to spend. Constructed once per
    question and passed to every should_retry call.

    Enforcement lives here rather than in the caller's discipline: should_retry
    debits the budget when it ISSUES a plan, so a caller that forgets to report
    back cannot loop forever — it can only over-charge itself, which fails in
    the safe direction.
    """

    def __init__(self, started_at: float | None = None,
                 wall_ceiling_s: float = WALL_CEILING_S,
                 max_passes: int = MAX_EXTRA_PASSES,
                 clock=time.perf_counter):
        self._clock = clock
        self.started_at = self._clock() if started_at is None else float(started_at)
        self.wall_ceiling_s = float(wall_ceiling_s)
        self.max_passes = int(max_passes)
        self.passes: list[tuple[str, str]] = []   # [(action, diagnosis label)]
        self.blocked: str | None = None           # why the last plan was refused

    def elapsed_s(self) -> float:
        return self._clock() - self.started_at

    def remaining_s(self) -> float:
        return self.wall_ceiling_s - self.elapsed_s()

    def allows(self, action: str, label: str) -> tuple[bool, str | None]:
        """Can we afford this exact pass? (ok, refusal reason)."""
        if len(self.passes) >= self.max_passes:
            return False, "pass_limit"
        if (action, label) in self.passes:
            # The same fix for the same diagnosis has already been tried. It has
            # no new information to work with; a second run costs latency only.
            return False, "repeat"
        if action == "clarify" and any(a == "clarify" for a, _ in self.passes):
            return False, "already_clarified"
        need = PASS_COST_S.get(action, 3.0)
        if self.remaining_s() < need:
            return False, "wall_clock"
        return True, None

    def spend(self, action: str, label: str) -> int:
        self.passes.append((action, label))
        self.blocked = None
        return len(self.passes)

    def exhausted(self) -> bool:
        """True when no further pass of any kind could be issued."""
        return (len(self.passes) >= self.max_passes
                or self.remaining_s() < min(PASS_COST_S.values()))

    def as_log(self) -> dict:
        return {"loop_passes": [f"{a}:{w}" for a, w in self.passes],
                "loop_blocked": self.blocked,
                "loop_elapsed_s": round(self.elapsed_s(), 3)}


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RetryPlan:
    """What to change on the next pass, and why. `instruction` is plain text to
    append to the generator's user message; `filters`/`query` are only set for
    action="retrieve"."""
    action: str                       # "regenerate" | "retrieve" | "clarify"
    why: str                          # diagnosis label
    reason: str                       # one human sentence
    instruction: str
    filters: dict | None = None
    query: str | None = None
    clarify_question: str | None = None
    ask_about: tuple[str, ...] = ()
    pass_no: int = 1
    est_seconds: float = 0.0
    final: bool = True                # no further pass will be issued after this

    def as_log(self) -> dict:
        return {"action": self.action, "why": self.why, "pass_no": self.pass_no,
                "final": self.final, "reason": self.reason}


_CLARIFY_QUESTIONS = {
    "course": "Which field are you leaning towards — engineering, medical, "
              "commerce, arts, or something else?",
    "location": "Which city or state should I look in?",
    "budget": "Roughly what yearly tuition budget are we working with?",
}


def _clarify_text(axes: tuple[str, ...]) -> tuple[str, str]:
    """(one question, the axis it asks about). Exactly one axis, always: a
    counsellor who asks three things at once gets none of them answered."""
    axis = axes[0] if axes else "course"
    return _CLARIFY_QUESTIONS.get(axis, _CLARIFY_QUESTIONS["course"]), axis


def _instruction(dx: Diagnosis) -> str:
    """The text the next pass runs on. Every branch says what to change AND
    what to keep — an instruction that only says "fix it" reproduces the same
    answer, which is what pipeline's own retry already established."""
    if dx.label == UNGROUNDED:
        return (
            "SALVAGE PASS (this is the last attempt on this context). The previous "
            "answer could not be verified as grounded, so answer SMALLER and safer: "
            "name at most 3 colleges, one short line each, full name and city. Quote "
            "a number ONLY if that exact figure is printed in that college's own card "
            "— if it is not there, describe the field without a number instead of "
            "estimating. Cite every college you name and nothing else. Do not add a "
            "college to make the answer feel fuller.\n")
    if dx.label == DEAD_END:
        field = f" '{dx.absent_field}'" if dx.absent_field else ""
        return (
            f"THE REFUSAL IS CORRECT — KEEP IT. What is missing is the way forward: "
            f"we genuinely do not have{field}, so say that plainly in one line, then "
            f"add ONE short line with something the listings DO carry for the "
            f"college(s) in CONTEXT (courses, tuition range, entrance exams accepted, "
            f"hostel, NAAC) and cite it. Never invent the missing figure, never "
            f"soften the refusal into a maybe, and point them to the college's own "
            f"admission page for the part we cannot give.\n")
    if dx.label == WRONG_CONTEXT:
        return (
            "CONTEXT REPLACED: the previous attempt answered from colleges that were "
            "not what the question was about, and retrieval has been re-run. Answer "
            "from the NEW CONTEXT only and ignore everything you said before. If the "
            "search had to be reinterpreted (a place read differently, a constraint "
            "set aside), state that in one short clause so the student knows what was "
            "searched — never substitute it silently.\n")
    if dx.label == AMBIGUOUS:
        question, axis = _clarify_text(dx.ask_about)
        return (
            f"ASK, DO NOT GUESS. This question cannot be answered well yet — the "
            f"student's {axis} is unknown, and any list you show would be an "
            f"arbitrary slice of thousands of colleges. Reply in TWO short lines: one "
            f"warm line saying there is plenty to work with, then exactly ONE "
            f"question — this one, in the student's own language: \"{question}\" "
            f"Ask about nothing else, name no colleges, quote no fees, cite nothing. "
            f"Set answered:true and citations:[].\n")
    return ""


def should_retry(result, hits, filters, question, *, budget: LoopBudget | None = None,
                 question_kind: str | None = None, total: int | None = None,
                 history: list | None = None) -> RetryPlan | None:
    """Is this completed answer good enough to ship? None means yes — ship it.

    A RetryPlan means no, and says what to change: regenerate on the same
    context, re-retrieve with different filters/query, or ask ONE clarifying
    question. The plan is only ever returned when the budget can actually pay
    for it, so a caller that simply loops `while (plan := should_retry(...))`
    is already bounded.

    `budget`: pass ONE LoopBudget per question to get the full MAX_EXTRA_PASSES.
    Omitting it is safe rather than convenient — an unbudgeted call gets a
    single-shot budget and the returned plan carries final=True, so a caller
    that never built a budget can never loop more than once.
    """
    if not isinstance(result, dict):
        return None
    try:
        dx = diagnose(result, hits, filters, question,
                      question_kind=question_kind, total=total, history=history)
    except Exception as exc:  # noqa: BLE001 - the loop must never break an answer
        print(f"[looprag] diagnosis failed, shipping as-is: {exc}", file=sys.stderr)
        return None
    if dx.action is None:
        return None

    own_budget = budget is None
    if own_budget:
        budget = LoopBudget(max_passes=1)
    ok, blocked = budget.allows(dx.action, dx.label)
    if not ok:
        budget.blocked = blocked
        print(f"[looprag] {dx.label} -> no pass ({blocked}, "
              f"{budget.remaining_s():.1f}s left): {dx.reason}", file=sys.stderr)
        return None

    pass_no = budget.spend(dx.action, dx.label)
    clarify_q = _clarify_text(dx.ask_about)[0] if dx.label == AMBIGUOUS else None
    plan = RetryPlan(
        action=dx.action,
        why=dx.label,
        reason=dx.reason,
        instruction=_instruction(dx),
        filters=dx.filters,
        query=dx.query or None,
        clarify_question=clarify_q,
        ask_about=dx.ask_about,
        pass_no=pass_no,
        est_seconds=PASS_COST_S.get(dx.action, 3.0),
        final=own_budget or budget.exhausted(),
    )
    print(f"[looprag] pass {pass_no}/{budget.max_passes} {dx.action} <- {dx.label}: "
          f"{dx.reason}", file=sys.stderr)
    return plan


# ---------------------------------------------------------------------------
# Self-test: python -m rag_core.looprag
#
# The decision logic is pure, so it is tested against real answer/hit shapes
# with no database, no LLM and no network. The case that matters most is
# test_correct_refusal_ships: a cutoff refusal is the RIGHT answer, and a loop
# there costs a student 4 s to be told the same true thing.
# ---------------------------------------------------------------------------

def _selftest() -> int:
    import unittest

    def card(name, city, courses="BCA; BBA; BCom", fee="Rs 45,000-90,000",
             extra=""):
        return (f"{name}, {city}\nType: Private\nEstablished: 2004\n"
                f"Courses offered (3): {courses}.\nTuition/year: {fee}\n{extra}")

    def hit(cid, name, city, state="Kerala", **kw):
        return {"college_id": cid, "name": name, "city": city, "state": state,
                "card": kw.pop("card", None) or card(name, city), "score": 1.0,
                "min_tuition_inr": kw.pop("min_tuition_inr", 45000),
                "max_tuition_inr": kw.pop("max_tuition_inr", 90000),
                "n_courses": 3, **kw}

    def answer(text, cites=(), answered=True, reason=None):
        return {"answer": text, "citations": list(cites), "answered": answered,
                "reason_if_unanswered": reason}

    A = "652f1a9c4d3b2e7a91c0f8d1"
    B = "652f1a9c4d3b2e7a91c0f8d2"
    C = "652f1a9c4d3b2e7a91c0f8d3"

    KOCHI = [hit(A, "Rajagiri College of Social Sciences", "Kochi"),
             hit(B, "Sacred Heart College Thevara", "Kochi"),
             hit(C, "Bharata Mata College Thrikkakara", "Kochi")]

    class T(unittest.TestCase):

        # -- shipping ----------------------------------------------------
        def test_grounded_specific_answer_ships(self):
            q = "Which colleges in Kochi offer BCA under 1 lakh?"
            r = answer("3 colleges in Kochi fit that budget:\n1. Rajagiri College "
                       "of Social Sciences, Kochi — Rs 45,000-90,000/year.",
                       [A, B, C])
            f = {"city": "Kochi", "course_terms": ["BCA"], "max_tuition_inr": 100000}
            b = LoopBudget()
            self.assertIsNone(should_retry(r, KOCHI, f, q, budget=b,
                                           question_kind="enumerate", total=3))
            self.assertEqual(b.passes, [])

        def test_correct_refusal_ships(self):
            """A cutoff refusal that names an adjacent fact is FINISHED."""
            q = "What was last year's cutoff for Rajagiri College of Social Sciences?"
            r = answer("We don't have cutoff numbers for Rajagiri College of Social "
                       "Sciences, Kochi — that one's best checked on their own "
                       "admissions page. What I can tell you: tuition runs "
                       "Rs 45,000-90,000/year and it offers BCA, BBA and BCom.",
                       [A], answered=False,
                       reason="Admission cutoffs are not part of the catalogue.")
            b = LoopBudget()
            dx = diagnose(r, KOCHI, {"include_abroad": True}, q)
            self.assertEqual(dx.label, ABSENT)
            self.assertIsNone(dx.action)
            self.assertIsNone(should_retry(r, KOCHI, {"include_abroad": True}, q,
                                           budget=b, question_kind="lookup"))
            self.assertEqual(b.passes, [], "a correct refusal must not spend a pass")

        def test_placement_package_refusal_ships(self):
            q = "average package of Sacred Heart College Thevara?"
            r = answer("We don't have placement salary figures for Sacred Heart "
                       "College Thevara, Kochi. It does offer BCA, BBA and BCom at "
                       "Rs 45,000-90,000/year.", [B], answered=False,
                       reason="Placement salary figures are not in the catalogue.")
            self.assertIsNone(should_retry(r, KOCHI, {}, q, budget=LoopBudget()))

        def test_program_absent_refusal_ships(self):
            q = "Does Rajagiri College of Social Sciences offer an M.Sc in Data Science?"
            r = answer("We don't have an M.Sc in Data Science at Rajagiri College of "
                       "Social Sciences, Kochi — what it does offer is BCA, BBA and "
                       "BCom.", [A], answered=False,
                       reason="M.Sc Data Science is not listed for that college.")
            self.assertIsNone(should_retry(r, KOCHI, {}, q, budget=LoopBudget()))

        def test_counselling_turn_ships(self):
            q = "bhaiya mere 62% aaye hain"
            r = answer("62% is workable — plenty of good options. Which field are you "
                       "drawn to?", [], answered=True)
            self.assertIsNone(should_retry(r, KOCHI, {}, q, budget=LoopBudget(),
                                           question_kind="profile_share"))

        def test_provider_error_does_not_loop(self):
            r = answer("Sorry — I hit a temporary issue.", [], answered=False,
                       reason="Transient provider error: no valid output after retries.")
            b = LoopBudget()
            self.assertIsNone(should_retry(r, [], {}, "fees of Rajagiri College?",
                                           budget=b))
            self.assertEqual(diagnose(r, [], {}, "x").label, PROVIDER_ERROR)

        # -- ungrounded --------------------------------------------------
        def test_failclosed_answer_regenerates_smaller(self):
            q = "Which colleges in Kochi offer BCA under 1 lakh?"
            r = answer("I found related colleges but could not verify a fully "
                       "grounded answer, so I would rather not guess.", [],
                       answered=False,
                       reason="Grounding verification failed after retry.")
            p = should_retry(r, KOCHI, {"city": "Kochi"}, q, budget=LoopBudget())
            self.assertIsNotNone(p)
            self.assertEqual((p.action, p.why), ("regenerate", UNGROUNDED))
            self.assertIn("at most 3 colleges", p.instruction)
            self.assertIsNone(p.filters, "a regenerate must not change retrieval")

        def test_ghost_citation_regenerates(self):
            q = "fees at Sacred Heart College Thevara?"
            r = answer("Sacred Heart College Thevara, Kochi charges Rs 45,000-90,000"
                       "/year.", [B, "ffffffffffffffffffffffff"])
            p = should_retry(r, KOCHI, {}, q, budget=LoopBudget())
            self.assertEqual((p.action, p.why), ("regenerate", UNGROUNDED))

        def test_uncited_money_claim_regenerates(self):
            q = "cheapest BCA college in Kochi?"
            r = answer("The cheapest option runs about Rs 38,000/year.", [])
            p = should_retry(r, KOCHI, {"city": "Kochi"}, q, budget=LoopBudget(),
                             question_kind="recommend", total=41)
            self.assertEqual((p.action, p.why), ("regenerate", UNGROUNDED))

        # -- wrong context -----------------------------------------------
        def test_wrong_college_retrieved_reretrieves(self):
            q = "Does Vellore Institute of Technology offer an M.Tech?"
            r = answer("We don't have an M.Tech listed there.", [], answered=False,
                       reason="M.Tech not listed.")
            p = should_retry(r, KOCHI, {"city": "Kochi"}, q, budget=LoopBudget(),
                             question_kind="lookup")
            self.assertEqual((p.action, p.why), ("retrieve", WRONG_CONTEXT))
            self.assertIn("vellore", p.query)
            self.assertEqual(p.filters, {"include_abroad": True},
                             "a named college must not stay behind the old filters")

        def test_mixed_absent_and_answerable_still_fixes_retrieval(self):
            """'cutoff and fees for X' — the fees half is answerable, so a wrong
            college is still worth one re-retrieval."""
            q = "cutoff and fees for Vellore Institute of Technology"
            r = answer("We don't have that.", [], answered=False, reason="no cutoff")
            p = should_retry(r, KOCHI, {}, q, budget=LoopBudget())
            self.assertEqual(p.why, WRONG_CONTEXT)

        def test_place_mismatch_reretrieves_with_corpus_spelling(self):
            q = "affordable BCA colleges in Vizag"
            hits = [hit(A, "Gayatri Vidya Parishad Degree College", "Hyderabad",
                        state="Telangana")]
            r = answer("Nothing matched in Vizag.", [], answered=False,
                       reason="no match")
            p = should_retry(r, hits, {"city": "Vizag", "course_terms": ["BCA"]}, q,
                             budget=LoopBudget())
            self.assertEqual((p.action, p.why), ("retrieve", WRONG_CONTEXT))
            self.assertEqual(p.filters["city"], "Visakhapatnam")
            self.assertEqual(p.filters["course_terms"], ["BCA"],
                             "the course the student cares about must survive")

        def test_unresolvable_place_drops_the_filter_and_says_so(self):
            q = "colleges in Zzyzxpur"
            hits = [hit(A, "Rajagiri College of Social Sciences", "Kochi")]
            r = answer("Nothing matched.", [], answered=False, reason="no match")
            p = should_retry(r, hits, {"city": "Zzyzxpur"}, q, budget=LoopBudget())
            self.assertEqual(p.why, WRONG_CONTEXT)
            self.assertNotIn("city", p.filters or {})
            self.assertIn("state that in one short clause", p.instruction)

        def test_zero_hits_drops_only_a_droppable_filter(self):
            q = "government BCA colleges in Kochi with hostel under 50000"
            r = answer("Nothing matched those constraints.", [], answered=False,
                       reason="no match")
            f = {"city": "Kochi", "course_terms": ["BCA"], "college_type": "Government",
                 "hostel_required": True, "max_tuition_inr": 50000}
            p = should_retry(r, [], f, q, budget=LoopBudget())
            self.assertEqual((p.action, p.why), ("retrieve", WRONG_CONTEXT))
            self.assertNotIn("hostel_required", p.filters)
            self.assertEqual(p.filters["max_tuition_inr"], 50000,
                             "money is never widened by this loop")
            self.assertEqual(p.filters["city"], "Kochi",
                             "geography is never widened by this loop")

        def test_zero_hits_with_only_money_and_place_does_not_loop(self):
            q = "colleges in Kochi under 20000"
            r = answer("Nothing in Kochi is that cheap.", [], answered=False,
                       reason="no match")
            b = LoopBudget()
            f = {"city": "Kochi", "max_tuition_inr": 20000}
            self.assertIsNone(should_retry(r, [], f, q, budget=b))
            self.assertEqual(b.passes, [])

        def test_refusal_without_adjacent_fact_regenerates(self):
            q = "what is the cutoff for Rajagiri College of Social Sciences?"
            r = answer("We don't have cutoff data.", [], answered=False,
                       reason="Admission cutoffs are not part of the catalogue.")
            p = should_retry(r, KOCHI, {}, q, budget=LoopBudget())
            self.assertEqual((p.action, p.why), ("regenerate", DEAD_END))
            self.assertIn("THE REFUSAL IS CORRECT", p.instruction)
            self.assertIn("cutoff", p.instruction)

        # -- ambiguous ---------------------------------------------------
        def test_wide_open_recommendation_asks_one_question(self):
            q = "which college is best?"
            r = answer("Here are some good ones:\n1. Rajagiri College of Social "
                       "Sciences, Kochi — Rs 45,000-90,000/year.\n2. Sacred Heart "
                       "College Thevara, Kochi — Rs 45,000-90,000/year.",
                       [A, B])
            p = should_retry(r, KOCHI, {}, q, budget=LoopBudget(),
                             question_kind="recommend", total=38700)
            self.assertEqual((p.action, p.why), ("clarify", AMBIGUOUS))
            self.assertEqual(p.ask_about, ("course",))
            self.assertEqual(p.clarify_question.count("?"), 1,
                             "exactly one question, or none of them gets answered")
            self.assertIn(p.clarify_question, p.instruction)

        def test_an_answer_that_already_asks_is_not_clarified(self):
            """Live pipeline output for the eval set's most ambiguous question:
            it named two colleges as examples but ended with a follow-up. A
            clarify pass would spend 2 s to rewrite a good question."""
            q = "help me choose a college"
            r = answer("No stress, let's figure this out together! Plenty of options "
                       "— which course or stream are you interested in?", [A, B])
            self.assertIsNone(should_retry(r, KOCHI, {}, q, budget=LoopBudget(),
                                           question_kind="recommend", total=37854))

        def test_help_me_choose_is_not_read_as_a_college_name(self):
            """'choose' looked like a discriminating name token until a live run
            showed it suppressing the ambiguity check entirely."""
            self.assertEqual(_named_institution("help me choose a college")[0], set())
            self.assertTrue(_is_vague("help me choose a college", {}))

        def test_specific_question_is_never_clarified(self):
            q = "which college is best for BCA in Kochi under 1 lakh?"
            r = answer("Rajagiri College of Social Sciences, Kochi leads on NAAC.",
                       [A, B])
            f = {"city": "Kochi", "course_terms": ["BCA"], "max_tuition_inr": 100000}
            self.assertIsNone(should_retry(r, KOCHI, f, q, budget=LoopBudget(),
                                           question_kind="recommend", total=41))

        def test_ambiguity_axis_prefers_the_missing_field(self):
            self.assertEqual(_ask_axes("which college is best?", {}, None),
                             ("course", "location", "budget"))
            self.assertEqual(_ask_axes("best college?", {"city": "Pune"}, None)[0],
                             "course")
            self.assertEqual(
                _ask_axes("best college?", {"course_terms": ["MBA"]}, None)[0],
                "location")

        # -- budget ------------------------------------------------------
        def test_budget_caps_at_two_passes(self):
            q = "Does Vellore Institute of Technology offer an M.Tech?"
            wrong = answer("Not listed.", [], answered=False, reason="no")
            bad = answer("I found related colleges but could not verify a fully "
                         "grounded answer.", [], answered=False,
                         reason="Grounding verification failed after retry.")
            b = LoopBudget()
            self.assertIsNotNone(should_retry(wrong, KOCHI, {}, q, budget=b))
            self.assertIsNotNone(should_retry(bad, KOCHI, {}, q, budget=b))
            self.assertIsNone(should_retry(wrong, KOCHI, {}, q, budget=b))
            self.assertEqual(b.blocked, "pass_limit")
            self.assertEqual(len(b.passes), MAX_EXTRA_PASSES)

        def test_same_fix_is_never_tried_twice(self):
            q = "what is the cutoff for Rajagiri College of Social Sciences?"
            r = answer("We don't have cutoff data.", [], answered=False,
                       reason="no cutoff")
            b = LoopBudget()
            self.assertIsNotNone(should_retry(r, KOCHI, {}, q, budget=b))
            self.assertIsNone(should_retry(r, KOCHI, {}, q, budget=b))
            self.assertEqual(b.blocked, "repeat")

        def test_wall_clock_ceiling_blocks_a_pass(self):
            q = "Does Vellore Institute of Technology offer an M.Tech?"
            r = answer("Not listed.", [], answered=False, reason="no")
            now = [1000.0]
            b = LoopBudget(clock=lambda: now[0])       # 0 s elapsed
            now[0] += 7.0                             # 7 s gone, 2 s left
            self.assertIsNone(should_retry(r, KOCHI, {}, q, budget=b))
            self.assertEqual(b.blocked, "wall_clock")
            self.assertAlmostEqual(b.remaining_s(), WALL_CEILING_S - 7.0, places=6)

        def test_a_pass_that_fits_is_issued(self):
            q = "Does Vellore Institute of Technology offer an M.Tech?"
            r = answer("Not listed.", [], answered=False, reason="no")
            now = [1000.0]
            b = LoopBudget(clock=lambda: now[0])
            now[0] += 4.0                             # 5 s left, retrieve needs 4
            p = should_retry(r, KOCHI, {}, q, budget=b)
            self.assertEqual(p.action, "retrieve")
            self.assertEqual(p.est_seconds, PASS_COST_S["retrieve"])

        def test_no_budget_means_exactly_one_pass(self):
            q = "Does Vellore Institute of Technology offer an M.Tech?"
            r = answer("Not listed.", [], answered=False, reason="no")
            p = should_retry(r, KOCHI, {}, q)          # caller brought no budget
            self.assertTrue(p.final, "an unbudgeted plan must be terminal")
            self.assertEqual(p.pass_no, 1)

        def test_clarify_happens_at_most_once(self):
            q = "which college is best?"
            r = answer("1. Rajagiri College of Social Sciences, Kochi\n"
                       "2. Sacred Heart College Thevara, Kochi", [A, B])
            b = LoopBudget()
            self.assertIsNotNone(should_retry(r, KOCHI, {}, q, budget=b,
                                              total=38700))
            self.assertIsNone(should_retry(r, KOCHI, {}, "kaunsa college best hai?",
                                           budget=b, total=38700))
            self.assertIn(b.blocked, ("already_clarified", "repeat"))

        def test_final_flag_marks_the_last_affordable_pass(self):
            q = "Does Vellore Institute of Technology offer an M.Tech?"
            r = answer("Not listed.", [], answered=False, reason="no")
            b = LoopBudget(max_passes=1)
            self.assertTrue(should_retry(r, KOCHI, {}, q, budget=b).final)

        def test_bad_input_never_raises(self):
            self.assertIsNone(should_retry(None, None, None, None))
            self.assertIsNone(should_retry({}, None, None, ""))

    suite = unittest.TestLoader().loadTestsFromTestCase(T)
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if res.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
