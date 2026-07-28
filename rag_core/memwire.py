"""memwire — the seam that makes durable memory usable from the request path.

memory.py owns the storage: a typed profile, a rolling summary, the turn log,
and a merge that never lets a null erase a known fact. It is deliberately
unaware of the pipeline. This module is the adapter — session id in, a
prompt-ready context out; finished answer in, persisted memory out — and it is
where the two decisions memory.py cannot make get made.

DECISION 1 — SUMMARISATION NEVER BLOCKS A REPLY.
memory.summarise() is an LLM call: 1-3 s, against a free tier measured at ~14
calls/min that the router and the generator are already competing for. Inline,
it would make one turn in six visibly slower than the other five and spend the
student's own rate budget on housekeeping. So after_turn() writes the turn and
the profile synchronously — three upserts, single-digit milliseconds — and
hands summarisation to a background thread that starts only after the answer
has been returned. At most _MAX_MAINT threads run at once and a session already
being summarised is skipped rather than queued: a missed summary costs a
slightly longer prompt next turn, a stolen LLM slot costs a student an answer.

DECISION 2 — WHICH REMEMBERED FACTS MAY BECOME HARD SQL FILTERS.
This is the whole value of the module and also its only real danger. A student
who said "80 thousand" on turn 2 must still get budget-filtered results on
turn 9 without repeating it (filters_from_profile). But a filter seeded from
memory is applied by code the student never sees, so a WRONG one is close to
undetectable from their side: the colleges simply are not there, and nothing in
the reply explains why. A stale hard filter is therefore treated as a
correctness bug, not a UX wrinkle, and the policy is:

  seeded              why it is safe to carry
  ------------------  ----------------------------------------------------
  max_tuition_inr     a budget is a property of the STUDENT, not of the
                      question. Only an unambiguous per-YEAR rupee amount
                      is carried; see parse_annual_budget for the seven
                      forms that are refused instead of guessed.
  course_terms        a stated field of study scopes every later turn (the
                      generator prompt already requires this). Only terms
                      from _FIELD_TERMS, which is closed and measured.
  program_level       closed vocabulary, and UG/PG is a fact about the
                      student's stage, not about one question.
  hostel_required     only True, and only when the student asserted it.

  NOT seeded          why not
  ------------------  ----------------------------------------------------
  state / city        OFF BY DEFAULT, and this is the one genuinely close
                      call in the module. Geography is the most volatile
                      constraint in a counselling chat and the one most often
                      a one-off ("what about Kerala?"); pipeline._BROADEN_RE
                      and _BROADEN_STATE_RE exist because sticky geography
                      produced "that's the only one there" turn after turn.
                      The two failures are not symmetric: colleges in the
                      wrong city are OBVIOUS to the student and one sentence
                      fixes them, while a city quietly still applied looks
                      identical to "there is nothing else" and cannot be
                      undone by someone who does not know it is there. So the
                      default is off — geography still reaches the generator
                      through the prompt block and ranking through the
                      question text, it just never becomes a WHERE. Set
                      SEED_LOCATION = True to carry it (validated against the
                      real place list first; see _profile_place).
  marks_pct           the corpus has no cutoff column, so marks can never be
                      a filter. Stated in memory.py and the router prompt too.
  category            no reservation data in the corpus to filter on.
  college_type        MEASURED BROKEN, not a policy call: retrieve lowercases
                      the value and Postgres LIKE is case-sensitive, so
                      {"college_type": "Government"} matches 0 of the 6,083
                      Government colleges. Seeding it would silently empty
                      every result set. Flip SEED_COLLEGE_TYPE once the
                      compare is case-insensitive.
  entrance_exam       MEASURED BROKEN: college_agg.entrance_exams is JSONB
                      and the filter does LIKE on it, which raises
                      "operator does not exist: jsonb ~~ text". Seeding it
                      would turn a remembered exam into a 500.

Anything seeded is also ANNOUNCED: apply_profile_filters returns notes and
filter_note() renders them for the generator, so a carried-forward constraint
is something the student is told about and can undo, never something that
silently shrinks their options.

HOW TO WIRE THIS IN (app.py / pipeline.py do not use memory yet):

    ctx = memwire.load_context(session_id)
    result = answer_question(question,
                             history=ctx["recent_turns"],       # the transcript
                             known_profile=ctx["profile"])      # the typed facts
    memwire.after_turn(session_id, question, result,
                       result.get("profile"), turn_no=ctx["n_turns"] + 1)

and inside the pipeline, right after _clean_filters() and right after the
CONTEXT block is assembled:

    filters, seeded = memwire.apply_profile_filters(
        filters, known_profile, question=question, question_kind=question_kind)
    ...
    user_msg = ctx["prompt_block"] + hist_text + f"CONTEXT:\\n{context}\\n\\n"
    user_msg += memwire.filter_note(seeded)   # note() it, so the verifier
                                              # accepts the numbers it states

Everything here returns rather than raises. Memory is an enhancement: a
student whose profile cannot be loaded must still get an answer to the question
in front of them, just without the history.
"""
from __future__ import annotations

import re
import sys
import threading

from . import memory

# Keys retrieve.py actually understands. Duplicated as a guard rather than
# imported: retrieve._KNOWN_FILTERS is private, and a key that does not appear
# there is dropped with a one-line warning deep in retrieval — which from the
# student's side looks exactly like memory having been ignored.
RETRIEVAL_FILTER_KEYS = frozenset({
    "state", "city", "college_type", "max_tuition_inr", "min_tuition_inr",
    "course_terms", "program_level", "hostel_required", "entrance_exam",
    "include_abroad",
})

# All three default False for the reasons in the module docstring. The first
# two are measured store defects rather than preferences; the third is a
# judgement call. Kept as flags so every one of them is a visible, one-line
# decision instead of a silent omission someone has to reverse-engineer.
SEED_COLLEGE_TYPE = False
SEED_ENTRANCE_EXAM = False
SEED_LOCATION = False

_MAX_MAINT = 2          # concurrent background summarisers; see DECISION 1


def _log(msg: str) -> None:
    print(f"[memwire] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Read side
# ---------------------------------------------------------------------------

def _recent_turns(session_id: str, limit: int) -> list[dict]:
    """The last `limit` messages, oldest-first, USER BEFORE ASSISTANT.

    Not memory.recent_turns(): that orders `turn_no DESC, role DESC` and then
    reverses, and because 'user' > 'assistant' the reverse puts the counsellor's
    reply BEFORE the question it answered. Measured against the live store, a
    two-turn session comes back as [A1, Q1, A2, Q2]. The pipeline renders this
    list verbatim as "CONVERSATION SO FAR", so every exchange would read
    backwards to the model — and pronoun resolution ("more details about it")
    is exactly what that block exists to support. `role ASC` before the reverse
    is the same query with the pairs the right way round.
    """
    from .store.backend import get_backend

    rows = get_backend().q(
        "SELECT role, content FROM conversation_turn WHERE session_id = ? "
        "ORDER BY turn_no DESC, role ASC LIMIT ?",
        [session_id, max(1, int(limit))],
    )
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def load_context(session_id: str, n_recent: int = memory.RECENT_TURNS) -> dict:
    """Everything the request path needs from memory, in one call.

    Returns {session_id, profile, summary, n_turns, recent_turns, prompt_block,
    profile_filters}. `recent_turns` is already in the [{role, content}] shape
    answer_question(history=...) wants, and `prompt_block` is ready to prepend
    to the generator's user message.

    Two round trips, ~2-4 ms warm against local Postgres. Never raises: memory
    is an enhancement, and a student whose store is unreachable must still get
    an answer to the question in front of them — just without the history.
    """
    ctx = {"session_id": session_id or "", "profile": {}, "summary": "",
           "n_turns": 0, "recent_turns": [], "prompt_block": "",
           "profile_filters": {}}
    if not session_id:
        return ctx
    try:
        mem = memory.load(session_id)
        ctx["profile"] = mem.get("profile") or {}
        ctx["summary"] = mem.get("summary") or ""
        ctx["n_turns"] = int(mem.get("n_turns") or 0)
        ctx["prompt_block"] = memory.as_prompt_block(mem)
        ctx["profile_filters"] = filters_from_profile(ctx["profile"])
    except Exception as exc:  # noqa: BLE001
        _log(f"profile load failed for {session_id!r}: {exc}")
    try:
        ctx["recent_turns"] = _recent_turns(session_id, n_recent)
    except Exception as exc:  # noqa: BLE001
        _log(f"turn log unavailable, falling back: {exc}")
        try:
            ctx["recent_turns"] = memory.recent_turns(session_id, n_recent)
        except Exception:  # noqa: BLE001
            pass
    return ctx


# ---------------------------------------------------------------------------
# Profile -> retrieval filters
# ---------------------------------------------------------------------------

# Money words a student actually types, including Hinglish. "l" is here because
# "1.5l" is common; it is only ever read as a SUFFIX on a number.
_UNITS = {
    "crore": 10_000_000, "cr": 10_000_000, "crores": 10_000_000,
    "lakh": 100_000, "lakhs": 100_000, "lac": 100_000, "lacs": 100_000,
    "l": 100_000,
    "thousand": 1_000, "k": 1_000, "hazaar": 1_000, "hazar": 1_000,
}

# Units this function refuses to carry forward instead of converting. The
# pipeline DOES convert a per-semester budget on the turn it is stated, and it
# says so in the answer ("60,000/semester = 120,000/year"). Silently reapplying
# that conversion six turns later, with no sentence explaining it, is a
# different act: house rule 3 says money is copied with its stated unit and
# never annualised, and a total-course budget cannot be hard-filtered at all
# because course lengths vary. So an ambiguous unit means no seeded filter.
_UNSAFE_UNIT_RE = re.compile(
    r"semester|\bsem\b|month|monthly|mahina|mahine|"
    r"total|whole|entire|full course|\bcourse\b|"
    r"\bper\s+(?:sem|semester|month)\b|pura|poora|puri", re.I)

_RANGE_RE = re.compile(r"\s(?:to|se)\s|-|–|—")
_AMOUNT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(crores?|cr|lakhs?|lacs?|l|thousand|k|hazaar|hazar)?\b",
    re.I)

# Rupee-per-year sanity band. The floor matches pipeline._clean_filters: a
# bound under Rs 1,000 can only be a unit slip ("2" meaning 2 lakh), and
# applying it tells a student that nothing fits their budget — the worst
# available wrong answer. The ceiling is a unit slip in the other direction
# ("80 lakh" for 80 thousand); the most expensive domestic tuition in this
# corpus is comfortably inside 1 crore.
MIN_BUDGET_INR = 1_000
MAX_BUDGET_INR = 10_000_000


def parse_annual_budget(value) -> int | None:
    """A remembered budget -> rupees per year, or None if it must not be used.

    Returns None (never a guess) for: a per-semester or per-month figure, a
    total-course figure, more than one amount with no range between them
    ("1 lakh 20 thousand" and a typo are indistinguishable), and anything
    outside the sanity band. A refused budget costs the student a wider result
    set; a wrong one costs them the right colleges without telling them.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        rupees = int(value)
    else:
        text = str(value).strip()
        if not text or len(text) > 120:
            return None
        if _UNSAFE_UNIT_RE.search(text):
            return None
        text = re.sub(r"(?:rs\.?|inr|₹|/-)", " ", text, flags=re.I).replace(",", "")
        found = [(float(m.group(1)), (m.group(2) or "").lower())
                 for m in _AMOUNT_RE.finditer(text) if m.group(1)]
        if not found:
            return None
        if len(found) > 1:
            if not _RANGE_RE.search(text):
                return None
            # A range is a ceiling: take the top end, and let a bare lower
            # bound inherit the unit that was only written once ("1-2 lakh").
            unit = next((u for _, u in reversed(found) if u), "")
            found = [(found[-1][0], found[-1][1] or unit)]
        amount, unit = found[0]
        rupees = int(round(amount * _UNITS.get(unit, 1)))

    if not MIN_BUDGET_INR <= rupees <= MAX_BUDGET_INR:
        _log(f"refusing implausible remembered budget {value!r} -> {rupees}")
        return None
    return rupees


# Field of study -> the course term EXACTLY as the corpus spells it, with the
# number of colleges each one matches, measured against the live store today.
#
# The casing is not cosmetic. Postgres LIKE is case-sensitive and retrieve.py
# passes course terms through unchanged, so the case decides whether a filter
# finds colleges at all: measured, "MBA" matches 3,863 colleges and "mba"
# matches 3, "Engineering" 5,180 and "engineering" 23, "Computer Science" 5,740
# and "computer science" 0. Any term added here must be verified with
# retrieve.count_matching({"course_terms": [term]}) before it is trusted.
#
# The map is closed on purpose. An unrecognised interest ("helping people",
# "something in tech") yields NO filter rather than a LIKE that matches nothing
# — the interest still reaches the generator via the prompt block and reaches
# ranking via the question text, neither of which can delete a result set.
_FIELD_TERMS: dict[str, str] = {
    # computing (5,740 / 5,798 / 1,827)
    "computer science": "Computer Science", "cs": "Computer Science",
    "cse": "Computer Science", "computers": "Computer Science",
    "computer": "Computer Science", "coding": "Computer Science",
    "programming": "Computer Science", "software": "Computer Science",
    "software engineering": "Computer Science",
    "computer applications": "Computer Applications",
    "bca": "Computer Applications", "mca": "Computer Applications",
    "it": "Information Technology",
    "information technology": "Information Technology",
    "data science": "Data Science",                        # 1,254
    "ai": "Artificial Intelligence",                       # 1,551
    "artificial intelligence": "Artificial Intelligence",
    "machine learning": "Artificial Intelligence",
    # engineering (5,180 for the generic term; branches are narrower)
    "engineering": "Engineering", "btech": "Engineering", "b tech": "Engineering",
    "be": "Engineering", "mtech": "Engineering", "m tech": "Engineering",
    "civil": "Civil Engineering", "civil engineering": "Civil Engineering",
    "mechanical": "Mechanical Engineering",
    "mechanical engineering": "Mechanical Engineering",
    "electrical": "Electrical Engineering",
    "electrical engineering": "Electrical Engineering",
    "electronics": "Electronics", "ece": "Electronics", "eee": "Electronics",
    # management and commerce (3,863 / 4,536 / 3,733 / 8,444)
    "mba": "MBA", "business administration": "MBA",
    "bba": "BBA",
    "management": "Management", "business": "Management",
    "commerce": "Commerce", "bcom": "Commerce", "b com": "Commerce",
    "accounts": "Commerce", "accounting": "Commerce",
    "economics": "Economics",                              # 2,973
    "statistics": "Statistics",                            # 736
    "finance": "Commerce",
    # health (559 / 3,847 / 2,671 / 324 / 558 / 98)
    "mbbs": "MBBS", "doctor": "MBBS", "medicine": "MBBS",
    "nursing": "Nursing", "nurse": "Nursing", "gnm": "Nursing", "bsc nursing": "Nursing",
    "pharmacy": "Pharmacy", "pharma": "Pharmacy", "bpharm": "Pharmacy",
    "b pharm": "Pharmacy", "dpharm": "Pharmacy",
    "dental": "Dental", "bds": "Dental", "dentist": "Dental",
    "physiotherapy": "Physiotherapy", "bpt": "Physiotherapy",
    "veterinary": "Veterinary",
    "biotechnology": "Biotechnology", "biotech": "Biotechnology",  # 1,179
    "microbiology": "Microbiology",                        # 968
    # law (1,459 / 1,340)
    "law": "Law", "llb": "Law", "llm": "Law", "lawyer": "Law", "advocate": "Law",
    # the broad streams (10,709 / 15,672 / 7,445)
    "arts": "Arts", "ba": "Arts", "humanities": "Arts",
    "fine arts": "Fine Arts",
    "science": "Science", "bsc": "Science", "b sc": "Science", "msc": "Science",
    "education": "Education", "teaching": "Education", "teacher": "Education",
    "bed": "Education", "b ed": "Education",
    # everything else, each measured above 200 colleges
    "design": "Design", "designing": "Design",
    "fashion": "Fashion", "fashion design": "Fashion",
    "animation": "Animation",
    "architecture": "Architecture",
    "journalism": "Journalism",
    "mass communication": "Mass Communication", "media": "Mass Communication",
    "psychology": "Psychology",
    "hotel management": "Hotel Management", "hospitality": "Hotel Management",
    "aviation": "Aviation",
    "social work": "Social Work", "msw": "Social Work",
    "agriculture": "Agriculture", "farming": "Agriculture",
}

# Longest first so "computer science" wins over "science" and "hotel
# management" over "management" when the interest arrives as a phrase.
_FIELD_KEYS = sorted(_FIELD_TERMS, key=len, reverse=True)

# retrieve._LEVEL_SYNONYMS lowercases the column, so level is the one text
# filter that is case-safe today. Vocabulary is closed by the ETL
# (sources/courses_csv.program_level).
_LEVELS = {
    "undergraduate": "Undergraduate", "ug": "Undergraduate",
    "bachelor": "Undergraduate", "bachelors": "Undergraduate",
    "postgraduate": "Postgraduate", "pg": "Postgraduate",
    "master": "Postgraduate", "masters": "Postgraduate",
    "doctorate": "Doctorate", "phd": "Doctorate", "doctoral": "Doctorate",
    "diploma": "Diploma/Certificate", "certificate": "Diploma/Certificate",
    "diploma/certificate": "Diploma/Certificate",
}

_COLLEGE_TYPES = {"government": "Government", "govt": "Government",
                  "sarkari": "Government", "private": "Private",
                  "pvt": "Private", "deemed": "Deemed", "public": "Public"}


def _norm_phrase(value) -> str:
    """Lowercase, drop dots/hyphens, keep word boundaries — 'B.Tech' -> 'btech',
    'Computer-Science' -> 'computer science'."""
    s = re.sub(r"[.\-_/]+", " ", str(value or "").lower())
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s)).strip()


def course_term(field_interest) -> str | None:
    """A remembered field of study -> one corpus-spelled course term, or None.

    Exact match first, then a whole-word scan so a phrase the router did not
    trim ("i love computer science") still resolves. One term, never several:
    retrieve ORs course_terms, so adding a second guess would WIDEN the filter
    into programmes the student never mentioned.
    """
    phrase = _norm_phrase(field_interest)
    if not phrase or len(phrase) > 60:
        return None
    if phrase in _FIELD_TERMS:
        return _FIELD_TERMS[phrase]
    for key in _FIELD_KEYS:
        if len(key) < 3:  # "cs"/"it"/"ba" only ever match the whole phrase
            continue
        if re.search(rf"\b{re.escape(key)}\b", phrase):
            return _FIELD_TERMS[key]
    return None


def _profile_place(value) -> str | None:
    """A remembered location -> the place name spelled the way the corpus spells
    it, or None if the corpus has no such place. Only used when SEED_LOCATION.

    Both halves matter. Postgres LIKE is case-sensitive, so "indore" matches 0
    colleges where "Indore" matches 253 — a seeded lowercase place is worse than
    no filter. And aliases.resolve() only returns a corrected value when the
    name exists in the real distinct city/state list, which doubles as the
    validity check: a hallucinated or misheard place seeds nothing.

    Which key it goes under does not matter: retrieve._where builds the same
    (state OR city OR district) LIKE clause for either one.
    """
    raw = str(value or "").strip()
    if not raw or len(raw) > 60:
        return None
    try:
        from .store.aliases import _load_places, _norm, resolve

        for kind in ("city", "state"):
            fixed, _note = resolve(raw, kind)
            places = _load_places().get(kind) or []
            if any(_norm(p) == _norm(fixed) for p in places):
                return fixed
    except Exception as exc:  # noqa: BLE001 - a place we cannot verify is not seeded
        _log(f"place check unavailable for {raw!r}: {exc}")
    return None


def filters_from_profile(profile: dict | None) -> dict:
    """Durable profile -> retrieval filters that are safe to apply unasked.

    Deterministic, and free of I/O in its default configuration, so it can run
    on every turn and be unit-tested without a store. Emits only keys
    retrieve.py understands, and only the ones the module docstring justifies.
    An unparseable fact produces no key at all.
    """
    out: dict = {}
    p = profile or {}
    if not isinstance(p, dict):
        return out

    budget = parse_annual_budget(p.get("budget"))
    if budget is not None:
        out["max_tuition_inr"] = budget

    term = course_term(p.get("field_interest"))
    if term:
        out["course_terms"] = [term]

    level = _LEVELS.get(_norm_phrase(p.get("program_level")))
    if level:
        out["program_level"] = level

    # True only. A False would be the student saying "no hostel needed", which
    # is not a constraint on the colleges — has_hostel is recorded for just 488
    # of 38,700 colleges, so filtering on its absence would be filtering on
    # missing data.
    if p.get("hostel_required") is True:
        out["hostel_required"] = True

    if SEED_COLLEGE_TYPE:
        ctype = _COLLEGE_TYPES.get(_norm_phrase(p.get("college_type")))
        if ctype:
            out["college_type"] = ctype
    if SEED_ENTRANCE_EXAM:
        exam = str(p.get("entrance_exam") or "").strip()
        if 2 <= len(exam) <= 20:
            out["entrance_exam"] = exam.upper()
    if SEED_LOCATION:
        # city, not state: the clause retrieve builds is identical, and `city`
        # is the key the router uses for the narrower of the two readings.
        place = _profile_place(p.get("city") or p.get("location") or p.get("state"))
        if place:
            out["city"] = place

    return {k: v for k, v in out.items() if k in RETRIEVAL_FILTER_KEYS}


# "Forget what I said about money / anywhere is fine" — a release must beat a
# remembered filter, or the student has no way out of a constraint they no
# longer want. Cheaper and more reliable than asking the router.
_RELEASE_BUDGET_RE = re.compile(
    r"any budget|no budget|budget (?:is )?(?:no|not a)\b|"
    r"money (?:is )?(?:no|not a)\b|cost (?:is )?(?:no|not a)\b|"
    r"price (?:is )?(?:no|not a)\b|whatever the (?:cost|fee)|"
    r"regardless of (?:cost|fee|budget|price)|forget the budget|"
    r"paisa (?:koi )?(?:issue|problem|dikkat) nahi|budget ki (?:tension|chinta) nahi|"
    r"kitna bhi (?:ho|lage)|expensive (?:is )?(?:ok|fine)", re.I)
# Deliberately narrow: it must name a course/stream/field. The looser phrases a
# student uses for "show me other COLLEGES" ("koi aur option", "kuch aur
# dikhao") mean the opposite of changing subject, and pipeline._BROADEN_RE
# already handles those against geography.
_RELEASE_COURSE_RE = re.compile(
    r"any course|any stream|any (?:field|subject)|other (?:course|stream|field)s?|"
    r"change (?:my )?(?:stream|field|course)|different (?:course|stream|field)|"
    r"koi bhi (?:course|stream|field|subject)|(?:course|stream|field) badal", re.I)
# Geography release, used only when SEED_LOCATION is on. Same intent as
# pipeline._BROADEN_RE / _BROADEN_STATE_RE, which exist because a sticky place
# filter made the counsellor repeat "that's the only one there".
_RELEASE_PLACE_RE = re.compile(
    r"near\s?by|nearby|near me|around here|other cit|another cit|different cit|"
    r"any (?:city|state|where)|anywhere|elsewhere|outside\s+\w+|"
    r"all over india|across india|anywhere in india|"
    r"kahi aur|kahin aur|aas paas|aaspaas|koi aur (?:city|shehar|jagah)", re.I)


def _seed_label(key, value) -> str:
    """How a carried-forward filter is described to the student."""
    if key == "max_tuition_inr":
        return f"your budget of up to Rs {int(value):,}/year"
    if key == "course_terms":
        return "your interest in " + ", ".join(value)
    if key == "hostel_required":
        return "your need for a hostel"
    if key == "program_level":
        return f"the {value} level you mentioned"
    if key == "city":
        return f"{value}, the place you mentioned"
    if key == "college_type":
        return f"the {value} colleges you asked for"
    if key == "entrance_exam":
        return f"the {value} exam you mentioned"
    return f"{key}={value}"


def apply_profile_filters(filters: dict | None, profile: dict | None,
                          question: str = "", question_kind: str | None = None
                          ) -> tuple[dict, list[str]]:
    """Fill in what THIS turn did not say from what earlier turns did.

    Returns (filters, seeded_labels). Three rules, each one a failure this
    would otherwise cause:

      1. The current message always wins. A key the router extracted this turn
         is never touched — "actually make it 2 lakh" must not be overruled by
         the 80k from turn 2.
      2. A lookup is never filtered. pipeline replaces filters with
         {"include_abroad": True} for a question about ONE named college,
         precisely so a budget cannot delete the college being asked about;
         seeding here would undo that.
      3. An explicit release wins over memory. "Budget is not a problem" must
         drop the remembered ceiling, not be outvoted by it.
    """
    out = dict(filters or {})
    if question_kind == "lookup":
        return out, []

    seed = filters_from_profile(profile)
    q = str(question or "")
    if _RELEASE_BUDGET_RE.search(q):
        seed.pop("max_tuition_inr", None)
    if _RELEASE_COURSE_RE.search(q):
        seed.pop("course_terms", None)
    if _RELEASE_PLACE_RE.search(q):
        seed.pop("city", None)

    # A ceiling below a floor the student just stated is not a memory hit, it
    # is an empty result set: "colleges above 2 lakh" + a remembered 80k max
    # matches nothing, and retrieval would report zero rather than the truth.
    stated_min = out.get("min_tuition_inr")
    if stated_min and seed.get("max_tuition_inr") and seed["max_tuition_inr"] <= stated_min:
        seed.pop("max_tuition_inr", None)

    # A place stated this turn beats a remembered one under EITHER key: the
    # student said "in Pune", so a remembered "Indore" in the other slot would
    # AND the two together and match nothing.
    if out.get("state") or out.get("city"):
        seed.pop("city", None)

    labels = []
    for key, value in seed.items():
        if out.get(key) not in (None, "", [], {}):
            continue
        out[key] = value
        labels.append(_seed_label(key, value))
    return out, labels


def filter_note(seeded: list[str] | None) -> str:
    """The prompt block that makes a remembered filter visible.

    Without this the module's own feature is its worst bug: results silently
    narrowed by something said six turns ago, with nothing in the reply to
    explain the gap or hint that it can be lifted. One short phrase in the
    answer turns an invisible constraint into a correctable one.
    """
    if not seeded:
        return ""
    return ("CARRIED FORWARD FROM EARLIER IN THIS CONVERSATION (applied as hard "
            "filters to CONTEXT, the student did NOT repeat them this turn): "
            + "; ".join(seeded)
            + ". Mention in ONE short natural phrase that you are still going by "
              "this (e.g. 'sticking to your Rs 80,000/year budget'), and offer to "
              "widen it if they want. Never present it as something they said "
              "just now.\n")


# ---------------------------------------------------------------------------
# Write side
# ---------------------------------------------------------------------------

# Facts about the STUDENT that may be harvested from a turn's filters when the
# caller passes them. Deliberately excludes state/city/college_type: those are
# how a student browses ("what about Kerala?"), not who they are, and
# merge_profile never forgets, so one browsing turn would pin them for the rest
# of the conversation.
_HARVEST_FROM_FILTERS = ("program_level", "hostel_required", "entrance_exam")


def after_turn(session_id: str, question: str, result: dict | None,
               router_profile: dict | None = None, *,
               filters: dict | None = None, turn_no: int | None = None,
               summarise: bool = True) -> dict:
    """Persist one exchange, merge the profile, and decide about summarising.

    Call it AFTER the answer has been handed to the student. The three upserts
    are synchronous — a fast follow-up must be able to see turn N when it asks
    turn N+1, and backgrounding them would lose that ordering for the sake of a
    few milliseconds — while summarisation, the only LLM cost here, is spawned
    and not waited on.

    `router_profile` is the profile the router emitted (result["profile"] is
    used if it is omitted). `filters` is optional and only harvests the handful
    of student-shaped keys in _HARVEST_FROM_FILTERS.

    `turn_no` avoids re-reading n_turns when the caller already has it from
    load_context. Concurrent turns in one session can collide on it; the
    schema's ON CONFLICT (session_id, turn_no, role) upsert is what makes that
    (and a retried request) overwrite rather than duplicate, which is the
    behaviour a summariser needs.
    """
    out = {"turn_no": None, "n_turns": 0, "profile": {}, "summarising": False}
    if not session_id:
        return out
    result = result or {}
    try:
        if turn_no is None:
            turn_no = int(memory.load(session_id).get("n_turns") or 0) + 1
        turn_no = max(1, int(turn_no))

        answer = str(result.get("answer") or "")
        citations = [c for c in (result.get("citations") or []) if isinstance(c, str)]
        memory.record_turn(session_id, turn_no, "user", str(question or ""))
        if answer:
            memory.record_turn(session_id, turn_no, "assistant", answer, citations)

        new_facts = dict(router_profile) if isinstance(router_profile, dict) else {}
        if not new_facts and isinstance(result.get("profile"), dict):
            new_facts = dict(result["profile"])
        for key in _HARVEST_FROM_FILTERS:
            value = (filters or {}).get(key)
            if value not in (None, "", [], False):
                new_facts.setdefault(key, value)

        # merge_profile drops unknown keys and never lets a null erase a known
        # fact, so this is a strict add — one turn cannot forget another's.
        memory.save(session_id, profile=new_facts, bump_turns=1)
        mem = memory.load(session_id)
        out["turn_no"] = turn_no
        out["n_turns"] = int(mem.get("n_turns") or 0)
        out["profile"] = mem.get("profile") or {}
    except Exception as exc:  # noqa: BLE001 - memory must never fail a request
        _log(f"after_turn failed for {session_id!r}: {exc}")
        return out

    if summarise and memory.should_summarise(out["n_turns"]):
        out["summarising"] = _spawn_summarise(session_id, out["n_turns"])
    return out


_maint_lock = threading.Lock()
_maint_active: set[str] = set()
_maint_threads: list[threading.Thread] = []


def _spawn_summarise(session_id: str, n_turns: int) -> bool:
    """Start the summariser on a background thread. True if it was started.

    Skipped rather than queued when busy. The LLM ceiling here is ~14
    calls/min shared with the router and the generator, so a queue of pending
    summaries would come due exactly when traffic is heaviest and take slots
    from students waiting on answers. The window it would have folded is still
    in the turn log, and the next trigger picks it up.
    """
    with _maint_lock:
        if session_id in _maint_active:
            return False
        if len(_maint_active) >= _MAX_MAINT:
            _log(f"summarisation skipped for {session_id!r}: "
                 f"{len(_maint_active)} already running")
            return False
        _maint_active.add(session_id)
        # daemon: a shutdown mid-summary loses a summary, which is recoverable.
        # Holding the process open to finish housekeeping is not what a deploy
        # or a Ctrl-C should wait on.
        t = threading.Thread(target=_summarise_now, args=(session_id, n_turns),
                             name=f"memwire-summarise-{session_id[:12]}", daemon=True)
        _maint_threads.append(t)
        _maint_threads[:] = [x for x in _maint_threads if x.is_alive() or x is t]
    try:
        t.start()
    except RuntimeError as exc:  # out of threads: release the slot, do not leak it
        _log(f"could not start summariser for {session_id!r}: {exc}")
        with _maint_lock:
            _maint_active.discard(session_id)
        return False
    return True


def _older_turns(session_id: str, n_turns: int) -> list[dict]:
    """The messages that have fallen out of the verbatim window, oldest-first.

    RECENT_TURNS is a count of MESSAGES and each exchange writes two, so the
    visible window is the last RECENT_TURNS // 2 exchanges. Capped at 40
    messages: everything older than that is already inside the previous
    summary, and summarise() truncates each message to 400 chars anyway.
    """
    from .store.backend import get_backend

    cutoff = int(n_turns) - max(1, memory.RECENT_TURNS // 2)
    if cutoff < 1:
        return []
    rows = get_backend().q(
        "SELECT role, content FROM conversation_turn "
        "WHERE session_id = ? AND turn_no <= ? "
        "ORDER BY turn_no DESC, role ASC LIMIT 40",
        [session_id, cutoff],
    )
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def _summarise_now(session_id: str, n_turns: int) -> None:
    """Fold the fallen-out turns into the rolling summary. Background only."""
    try:
        mem = memory.load(session_id)
        older = _older_turns(session_id, n_turns)
        if not older:
            return
        previous = mem.get("summary") or ""
        new = memory.summarise(session_id, older, previous)
        # summarise() returns `previous` on any failure, so an unchanged string
        # means nothing was learned — writing it would only touch updated_at.
        if new and new != previous:
            memory.save(session_id, summary=new)
            _log(f"summarised {session_id!r} at {n_turns} turns "
                 f"({len(older)} older messages -> {len(new)} chars)")
    except Exception as exc:  # noqa: BLE001
        _log(f"summarisation failed for {session_id!r}: {exc}")
    finally:
        with _maint_lock:
            _maint_active.discard(session_id)


def wait_for_maintenance(timeout: float = 30.0) -> bool:
    """Join any running summariser. For tests and orderly shutdown ONLY — the
    request path must never call this, which is the entire point of the split.
    True if nothing is still running."""
    with _maint_lock:
        threads = list(_maint_threads)
    deadline_each = max(0.1, timeout / max(1, len(threads)))
    for t in threads:
        t.join(deadline_each)
    with _maint_lock:
        return not _maint_active


if __name__ == "__main__":  # python -m rag_core.memwire
    # Pure-function check: no store, no model. These are the mappings a wrong
    # edit would break silently, so they assert rather than print.
    assert parse_annual_budget("80 thousand") == 80_000
    assert parse_annual_budget("1.5 lakh") == 150_000
    assert parse_annual_budget("Rs 1,20,000") == 120_000
    assert parse_annual_budget("50k-80k") == 80_000
    assert parse_annual_budget("1-2 lakh") == 200_000
    assert parse_annual_budget("60 thousand per semester") is None
    assert parse_annual_budget("4 lakh total for the course") is None
    assert parse_annual_budget("1 lakh 20 thousand") is None
    assert parse_annual_budget("80") is None
    assert course_term("computer science") == "Computer Science"
    assert course_term("i love computer science") == "Computer Science"
    assert course_term("hotel management") == "Hotel Management"
    assert course_term("helping people") is None
    assert filters_from_profile(
        {"budget": "80 thousand", "field_interest": "MBA", "marks_pct": 62,
         "location": "Indore", "college_type": "Government"}
    ) == {"max_tuition_inr": 80_000, "course_terms": ["MBA"]}
    f, seeded = apply_profile_filters(
        {"city": "Indore"}, {"budget": "80 thousand", "field_interest": "MBA"},
        question="any MBA colleges there?")
    assert f == {"city": "Indore", "max_tuition_inr": 80_000, "course_terms": ["MBA"]}, f
    assert seeded == ["your budget of up to Rs 80,000/year",
                      "your interest in MBA"], seeded
    # the current turn always wins over memory
    assert apply_profile_filters({"max_tuition_inr": 200_000},
                                 {"budget": "80 thousand"})[0]["max_tuition_inr"] == 200_000
    # ...and so does an explicit release, and a lookup is never filtered
    assert apply_profile_filters(
        {}, {"budget": "80 thousand"}, question="budget is not a problem")[1] == []
    assert apply_profile_filters(
        {}, {"budget": "80 thousand"}, question="fees of Alpha College?",
        question_kind="lookup") == ({}, [])
    # a floor stated now must not be capped by a remembered ceiling
    assert "max_tuition_inr" not in apply_profile_filters(
        {"min_tuition_inr": 200_000}, {"budget": "80 thousand"})[0]
    assert filter_note(seeded).startswith("CARRIED FORWARD")
    assert filter_note([]) == ""
    try:
        # Drift guard: a key retrieve does not know is dropped with a one-line
        # warning inside retrieval, which from the student's side is
        # indistinguishable from memory being ignored.
        from .retrieve import _KNOWN_FILTERS

        unknown = RETRIEVAL_FILTER_KEYS - _KNOWN_FILTERS
        assert not unknown, f"retrieve.py no longer accepts {sorted(unknown)}"
    except ImportError as exc:  # numpy-less environment: not worth failing over
        print(f"  (filter-key drift check skipped: {exc})")
    print("memwire self-check ok")
