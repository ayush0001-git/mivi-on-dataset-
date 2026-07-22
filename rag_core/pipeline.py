"""The answer pipeline: route -> retrieve -> generate (grounded) -> verify -> JSON.

Flow per query (mirrors the target production design):
  1. Router (small model): does this question need retrieval at all? Extract
     hard filters and normalise units (semester/lakh -> Rs per year).
  2. Hybrid retrieval: hard filters in code + semantic ranking (retrieve.py).
  3. Generation (large model): answer from the retrieved cards ONLY, with a
     [C0XX] citation after every factual claim.
  4. Deterministic verification: cited ids must exist in the retrieved context;
     ids mentioned in prose must appear in `citations`. One corrective retry,
     then fail closed (answered=false) rather than ship an ungrounded answer.
"""
import json
import random
import re
import sys
import threading
import time

from . import config
from .data import load_colleges
from .llm import chat
from .retrieve import encode_query, retrieve, to_context

CID_RE = re.compile(r"C0\d{2}")

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class _EmbedPrefetch:
    """Compute the query embedding on a daemon thread WHILE the router call is
    in flight. Warm, this overlaps the ~50ms encode with the ~2s router
    round-trip; cold (CLI), it overlaps the multi-second embedding-model load
    with the router's network wait — the single biggest cold-start win.
    Daemon: a smalltalk/out-of-scope turn never waits for (or is blocked by)
    an embedding it doesn't need."""

    def __init__(self, question):
        self._done = threading.Event()
        self._vec = None
        threading.Thread(target=self._run, args=(question,), daemon=True).start()

    def _run(self, question):
        try:
            self._vec = encode_query(question)
        except Exception as e:  # retrieval will retry the encode and surface it
            print(f"[prefetch] embed failed: {e}", file=sys.stderr)
        finally:
            self._done.set()

    def result(self, timeout=180):
        self._done.wait(timeout)
        return self._vec

# An explicit count in the question ("top 3", "just 2 options", "sirf 5").
# Numbers followed by a money/percent unit are excluded so "under 2 lakh" and
# "I got 78%" are never mistaken for a requested list size.
_COUNT_RE = re.compile(
    r"\b(?:top|best|first|show me|give me|only|just|sirf|bas)\s+"
    r"(\d{1,2})\b(?!\s*(?:lakh|lac|thousand|k\b|%|per))"
    r"|\b(\d{1,2})\b(?!\s*(?:lakh|lac|thousand|k\b|%|per))\s+"
    r"(?:colleges?|options?|choices?|colleg)", re.I)


def _requested_count(question):
    """The list size the student explicitly asked for, if any (2-15)."""
    m = _COUNT_RE.search(question)
    if not m:
        return None
    n = int(m.group(1) or m.group(2))
    return n if 2 <= n <= 15 else None


# "Show me something else" signals — the student wants options BEYOND the
# current city/course, so a sticky city filter must be released instead of
# answering "that's the only one here" over and over.
_BROADEN_RE = re.compile(
    r"near\s?by|nearby|near me|around here|other cit|another cit|different cit|"
    r"other than|anything else|something else|any other|alternative|elsewhere|"
    r"anywhere|kahi aur|kahin aur|aas paas|aaspaas|paas me|koi aur|"
    r"doosr[ae]|dusr[ae]", re.I)

# Obvious prompt-injection attempts. The generator already answers only from
# CONTEXT and all numbers come from code, so injection can't produce a fake
# fee — but a clear "ignore your rules" still gets a calm, on-brand deflection
# rather than a chance to derail the model.
_INJECTION_RE = re.compile(
    r"ignore\s+(?:all\s+|your\s+|the\s+|previous\s+|prior\s+)*"
    r"(?:instructions|rules|prompt|guidelines)"
    r"|disregard\s+(?:your\s+|the\s+|all\s+)*(?:instructions|rules|prompt)"
    r"|forget\s+(?:everything|your\s+instructions|all\s+previous)"
    r"|(?:reveal|show|print|repeat)\s+(?:me\s+)?(?:your|the)\s+"
    r"(?:system\s+)?prompt"
    r"|you\s+are\s+now\s+(?:a|an|in)\b"
    r"|developer\s+mode|jailbreak|\bDAN\b", re.I)

# Distress / crisis phrases. Two tiers: explicit self-harm language always
# triggers (high recall — a genuine crisis must never be missed); softer
# emotional words only trigger when they're self-referential ("I feel
# hopeless"), so "is 45% hopeless for engineering?" stays a normal question.
_DISTRESS_RE = re.compile(
    r"\b(kill myself|end my life|suicid|self[- ]?harm|want to die|"
    r"don'?t want to live|no reason to live|give up on life|can'?t go on|"
    r"marna (?:hai|chahta|chahti|h)|mar jau|jeena nahi|jina nahi|"
    r"khatam kar (?:du|dun|lu))\b"
    r"|\b(?:i(?:'?m| am| feel)|feeling|mai bahut|main bahut)\s+\w*\s*"
    r"(?:hopeless|worthless|empty|numb|done with everything)\b", re.I)

ROUTER_SYSTEM = """You are the query router for a college-search assistant.
The database contains exactly 15 colleges in Uttarakhand, India with fields:
college_id, name, city, state, type (Government/Private/Deemed), courses_offered,
annual_fees_inr (tuition per academic YEAR), last_year_cutoff_pct (hard minimum
aggregate %), total_seats, hostel_available, naac_grade, avg_placement_lpa
(0 = not reported), established_year, about (free text).

Classify the user's message and extract hard filters. Return ONLY JSON:
{
  "route": "data_query" | "smalltalk" | "out_of_scope",
  "question_kind": "enumerate" | "recommend" | "lookup" | "profile_share" | "other",
  "language": "<language of the message, e.g. English, Hindi, Hinglish>",
  "filters": {
    "max_annual_fee_inr": <int or null>,
    "student_score_pct": <int or null>,
    "college_type": "Government" | "Private" | "Deemed" | null,
    "hostel_required": <true or null>,
    "city": <city OR state preference as stated, string or null>
  },
  "course_terms": [<course words mentioned, e.g. "MBA", "B.Tech", "engineering">],
  "needs_all_records": <true if answering requires scanning ALL colleges:
    superlatives (cheapest, best, highest placement, oldest), counts
    ("how many..."), or ANY enumeration question ("which colleges offer/have
    X", "list the colleges that Y") — else false>,
  "unit_note": <string or null>,
  "profile": {"marks_pct": <int or null>, "field_interest": <string or null>,
    "budget": <string or null>, "location": <string or null>}
    — cumulative facts about the STUDENT gathered from the WHOLE conversation
    (not just the current message); null for anything not yet shared
}

Rules:
- "data_query": any question about colleges, courses, fees, admission, hostels,
  placements, scholarships — even if the answer may not be in the data.
  A student sharing their marks/percentage/budget (e.g. "I got 55% in class",
  "mera budget 1 lakh hai") is NEVER smalltalk — it is route "data_query"
  with question_kind "profile_share".
- "smalltalk": greetings, thanks, who-are-you.
- "out_of_scope": unrelated to colleges/admissions (poems, politics, coding...).
- question_kind: "enumerate" = user wants ALL colleges matching criteria
  ("which colleges...", "list...", "which fit my budget"); "recommend" = user
  wants advice/a pick ("best for me", "suggest"); "lookup" = a fact about one
  named college; "profile_share" = the student states facts about themselves
  (marks, budget, stream, interests) WITHOUT asking a concrete question, OR
  expresses that they're unsure/lost/don't know what to pick (e.g.
  "i got 55% in class 12", "i love computer science", "mujhe MBA karna hai",
  "confused hoon", "pata nahi kya karu", "I don't know what to do") — in all
  these cases the reply is conversational guidance, not a college lookup, so
  no citations are expected; "other" = anything else.
- Money: "1 lakh" = 100000, "60 thousand"/"60k" = 60000. Database fees are PER
  YEAR. If the user gives a budget per semester: annual = semester x 2, and set
  unit_note explaining the conversion (1 academic year = 2 semesters).
  If the user gives a total-course budget: set max_annual_fee_inr to null and
  put an explanation in unit_note (course lengths vary; cannot hard-filter).
  Examples:
    "budget of Rs 1.5 lakh/year"        -> max_annual_fee_inr = 150000
    "I can pay 60 thousand per semester" -> max_annual_fee_inr = 120000,
        unit_note = "60,000/semester = 120,000/year (2 semesters per year)"
    "I have 1 lakh per semester"         -> max_annual_fee_inr = 200000,
        unit_note = "1 lakh/semester = 2 lakh/year (2 semesters per year)"
- student_score_pct: set only when the user states their own marks/percentage.
- Set a filter ONLY if the user clearly stated it. Never guess.
- If CONVERSATION SO FAR is provided, interpret the CURRENT message in its
  context: a bare number after a marks discussion is their percentage, BUT a
  bare small number (1-15) right after the counsellor presented a NUMBERED
  LIST selects that list item — question_kind "lookup" about that college.
  A course interest stated earlier carries forward into course_terms; a
  message answering the counsellor's follow-up question is a data_query, not
  smalltalk. unit_note is ONLY for money/unit conversions — never put
  commentary or guesses there."""

GENERATOR_SYSTEM = """You are mimi, MakeMyEducation's AI college counsellor, answering from a
verified 15-college dataset (all in Uttarakhand, India). You will get CONTEXT
(college records) and a QUESTION. Answer using ONLY the CONTEXT.
Voice: say "I" when you act ("I'll check that"), "we" only for the
institution ("the colleges we work with").

EMPATHY (this is a nervous teenager making one of the biggest decisions of
their life, not a search query):
- Read the feeling behind the message and acknowledge it in ONE natural line
  before the facts — a low score deserves warmth not pity ("That's still
  very workable, don't count yourself out"), a tight budget deserves respect
  ("Totally fair to keep costs in mind — let's find real value"), excitement
  deserves matching energy. Never fake it, never over-do it, never a canned
  "I understand your concern."
- Empathy is a seasoning, not a prefix. Only open with it when there is a
  real feeling to meet (bad news, frustration, worry). A neutral question
  gets a useful answer, not a sympathy line. NEVER open two replies in a row
  with the same sympathy formula, and never reuse a stock opener like "I hear
  you" or "It's tough when..." — if your previous reply already opened that
  way, start this one somewhere else entirely (with the answer itself is
  usually best). Repeating the same emotional opener makes you sound like a
  script, which is worse than being blunt.
- If a student sounds genuinely distressed, hopeless, or mentions self-harm:
  drop the college talk entirely. Respond with brief, warm care, gently
  encourage them to talk to someone they trust or a professional, and share
  that India's free mental-health helpline KIRAN is 1800-599-0019. You are a
  counsellor for colleges, NOT a therapist — never try to handle a crisis
  alone. (Set answered:true, citations:[].)

Hard rules:
1. Every factual claim must come from CONTEXT. The supporting college_ids go
   ONLY in the "citations" array — NEVER write id codes (C001, C007...) inside
   the answer text itself. In the text, refer to colleges by full name (and
   city when helpful). Cite only ids that appear in CONTEXT.
2. Never use outside knowledge, never invent a college, fee, course or fact.
3. Fees are tuition per academic YEAR and EXCLUDE hostel/mess/kit/studio/lab
   charges. For cost/budget questions, mention relevant extra charges that the
   About text describes.
4. avg placement 0 means "not reported / not applicable" — NEVER describe it as
   low, zero or the worst. Explain why if the About text says (e.g. medical
   graduates proceed to internships, not campus placement).
5. The cutoff is a hard minimum: a student below it was not eligible.
6. Units: 1 academic year = 2 semesters. If the user's units differ from the
   data (per semester / total course), convert and STATE the assumption
   explicitly in the answer.
7. A Diploma is not a degree. If a diploma-only institution is relevant,
   you may include it, but only with an explicit note that it awards diplomas,
   not degrees.
8. Some college names look similar but are unrelated institutions in different
   cities — never confuse them. Identify every college by full name, city and
   id. Name ONLY colleges that qualify for the user's question: do not bring
   up a non-qualifying college even to disambiguate — precise naming (full
   name + city + id) IS the disambiguation.
8b. Course questions are EXACT-MATCH questions: include a college only if the
   asked program appears verbatim in its "Courses offered" list. Related
   programs are NOT the same program (BBA is not MBA, M.Com is not MBA,
   PGDM is not MBA — if PGDM is present alongside the question's program you
   may mention it as a separate note, clearly labelled).
9. Reply in the SAME language as the question (English / Hindi / Hinglish /
   other). Keep the warm, clear tone of a good counsellor; be concise.
9a. VOICE: you're the student's big brother/sister who happens to know every
   college inside out — not a customer-support bot, not a database. Warm,
   casual, encouraging, a little playful. Never say "dataset", "database",
   "CONTEXT", "records", "data", or "our system" — banned words, no
   exceptions, in every language you reply in. Say "we don't have <X> for
   <college>" or "<college> doesn't offer <X>, but here's what it does have"
   — never "not in our records" or any rephrasing of the same idea.
   If the student sounds unsure/lost ("I don't know what to do", "confused
   hoon", "pata nahi"), reassure first in ONE casual line ("No stress, let's
   figure this out together" / "koi na, chill — chalo shuru karte hain")
   before asking what interests them. Talk like this is a real conversation
   with someone you're rooting for, not a transaction.
9b. FORMAT: never a wall of text — short lines, not paragraphs. Lists: one
   short lead sentence that states the COUNT and speaks to the student's own
   constraints when they gave any — "There are 2 colleges in your range that
   offer an MBA:", "With your score, 3 colleges fit:" — then a numbered list,
   one college per line, separated by "\n", like:
   "1. <Name>, <City> — Rs <fee>/year; <the key facts asked>."
   Single-college details: one lead line, then short "- <label>: <value>"
   lines (fee, courses, cutoff, hostel, placement...), one per "\n".
   Never write more than 2 sentences without a line break. End with a
   one-line note (assumptions, extra charges) when needed. Single-fact
   answers may be one sentence. Plain text — no markdown bold/headers.
   ZERO filler: no "I hope this helps", "feel free to ask", "great question"
   or restating the question — every sentence must carry information.
9c. ANY question about 2-4 specific colleges TOGETHER — compare them, "details
   of both", "in dono ke baare me batao" — always uses a compact pipe table,
   one row per line separated by "\n", header first:
   "| Field | <College A> | <College B> |"
   "| Tuition/year | Rs 145,000 | Rs 118,000 |"
   Rows: Type, Courses, Tuition/year, Cutoff, Seats, Hostel, NAAC,
   Avg placement, Scholarships (summarise each college's scholarship/
   concession from its About in a few words; write "none mentioned" if its
   About lists none), Established. Every row starts AND ends with "|". Keep the
   long About text OUT of the table — after it, add at most ONE short line
   per college with its key About point, then ONE line with the key
   trade-off. (This is the only place any markdown-like syntax is allowed.)
10. If CONTEXT cannot support an answer, set "answered": false, briefly say
    what the data does and does not contain, and never guess. This includes
    ABSENCE questions: if asked whether a college offers a program/facility
    that its record does not list, set "answered": false and say the dataset
    does not record it. ALWAYS include one adjacent helpful fact from the
    record (what it does offer / the nearest alternative in CONTEXT, cited)
    so a refusal never feels like a dead end.
10b. Before ever saying a program is "not offered/listed", scan EVERY
    college's courses in CONTEXT — never deny a program that appears there.
    If the student's spelling looks like a typo of a listed program ("bse" ~
    B.Sc, "btech" ~ B.Tech, "b.pharma" ~ B.Pharm), assume the close match,
    answer for it, and note the assumption in one short phrase.
11. Enumeration questions ("which colleges offer/have X", "list...", "which
    fit my budget") must be COMPLETE: check EVERY record in CONTEXT and
    include ALL that qualify. If 8 colleges qualify, the answer names all 8
    and citations contains all 8 ids. A partial list is a wrong answer.
    EXCEPTION: if the student names a number ("top 3", "just 2 options"),
    that number wins — give exactly that many, ranked best-first. A REQUESTED
    COUNT line below states it explicitly when one was asked for.
12. Match the user's qualifier precisely. "Low-income / need-based" support
    means income- or need-based concessions only; merit, gender or
    region-based waivers are different (mention them only as clearly-labelled
    extras, and never count them as low-income support).
12a. COUNSEL like a real counsellor, don't just retrieve. If the student's
    message is underspecified for a confident recommendation — they only share
    their marks or budget, or ask for "a college" with no course/stream
    interest — do NOT dump a college list. Give at most a one-line sense of
    what's possible from CONTEXT, then ask ONE specific follow-up question
    (their course/stream interest, budget, or preferred location). Set
    "answered": true; cite only what you referenced. But when the question IS
    specific (a course, fee, hostel, comparison, eligibility list), answer it
    directly and completely — no unnecessary follow-ups.
12b. "All details" / full-profile questions about one college: present EVERY
    field from its record as a structured list — type, city, courses, tuition
    per year, last-year cutoff, total seats, hostel, NAAC grade, placement,
    established year — plus the key points from About (admission process,
    scholarships, extra charges). Leave nothing in the record out.
13. For "best"/recommendation questions: "best" is NOT "cheapest". Weigh
    placements (avg_placement_lpa), NAAC grade and cutoff feasibility against
    the budget, state the criteria you used, and lead with the strongest
    option(s) with their numbers. Offer 2-3 alternatives (e.g. per stream),
    note that "best" depends on the student's priorities, and never phrase a
    partial list as if it were the complete set of qualifying colleges.

Citations discipline: cite ONLY the colleges that form part of your actual
answer. For eligibility/budget questions, cite only colleges that satisfy the
stated constraints — never cite a college that fails them (you may say that
stretching the budget would open more options, without naming ids).

Worked examples of the required behaviour (names are generic placeholders):
- QUESTION: "Does Alpha College offer an M.Sc in Data Science?" and Alpha
  College's record lists only "B.Sc; B.Com" ->
  {"answer": "We don't have an M.Sc in Data Science listed at Alpha College —
  its courses with us are B.Sc and B.Com.",
  "citations": ["C0XX"], "answered": false,
  "reason_if_unanswered": "M.Sc Data Science is not recorded for Alpha College
  in the dataset."}
- QUESTION: "Which colleges offer a B.Sc, and what do they cost?" with two
  matches in CONTEXT ->
  {"answer": "Two colleges offer a B.Sc:\n1. Alpha College, Cityville — Rs
  50,000/year.\n2. Beta University, Townsburg — Rs 72,000/year.\nNote: fees
  are tuition only and exclude hostel/mess charges.",
  "citations": ["C0XX", "C0YY"], "answered": true, "reason_if_unanswered": null}

Return ONLY JSON, exactly this shape:
{"answer": "<string>", "citations": ["C0XX", ...], "answered": true|false,
 "reason_if_unanswered": null | "<string>"}

"answered" is true ONLY when the requested information is explicitly present
in CONTEXT. If the question asks about something the records do not list
(a course, a facility, a data field), you MUST set "answered": false — even
while the "answer" text helpfully explains what the records do list."""


def _parse_json(text):
    if not text:
        raise ValueError("empty model response")
    # Reasoning models may prepend <think>...</think>; strip it.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    # strict=False: some models emit literal newlines inside JSON strings
    # (we ask for newline-separated lists); accept them.
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        # Fall back to the first balanced {...} block in the text.
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0), strict=False)


def _heuristic_route(question):
    """Regex fallback if the router call fails: keep answering, log the incident."""
    q = question.lower()
    filters, terms, note = {}, [], None
    m = re.search(r"(\d+(?:\.\d+)?)\s*(lakh|lac|l\b)", q)
    amount = int(float(m.group(1)) * 100_000) if m else None
    if amount is None:
        m = re.search(r"(\d+(?:\.\d+)?)\s*(thousand|k\b)", q)
        if m:
            amount = int(float(m.group(1)) * 1_000)
    if amount:
        if re.search(r"\btotal\b|\bpura\b|\bpoora\b|whole course|entire course|puri padhai", q):
            # Total-course budgets cannot be hard-filtered (course lengths
            # vary) — the dictionary calls silently converting this the worst
            # failure available. Explain per-year costs instead.
            note = ("Budget given as TOTAL course cost; course lengths vary, so no "
                    "hard fee filter was applied — state this and explain per-year costs.")
        elif "semester" in q or re.search(r"\bsem\b", q):
            amount *= 2
            note = "Budget given per semester; assumed 1 academic year = 2 semesters."
            filters["max_annual_fee_inr"] = amount
        else:
            filters["max_annual_fee_inr"] = amount
    m = re.search(r"(\d{1,3}(?:\.\d+)?)\s*%", q)
    if m:
        pct = int(float(m.group(1)))
        if 0 < pct <= 100:
            filters["student_score_pct"] = pct
    if "government" in q or "sarkari" in q:
        filters["college_type"] = "Government"
    if "hostel" in q:
        filters["hostel_required"] = True
    for t in ("mba", "b.tech", "btech", "engineering", "mbbs", "law", "pharmacy", "design", "diploma"):
        if t in q:
            terms.append(t)
    aggregate = any(
        w in q
        for w in ("cheapest", "best", "worst", "highest", "lowest", "oldest", "newest",
                  "most ", "how many", "sabse", "kitne", "compare all", "top ",
                  "which colleges", "list the colleges", "list colleges", "kaun se college",
                  "kin colleges", "किन", "कौन")
    )
    kind = "other"
    unsure = any(w in q for w in ("confused", "don't know", "dont know", "not clear",
                                  "not sure", "pata nahi", "samaj nahi", "samajh nahi",
                                  "kya karu", "kya karoon"))
    if unsure:
        kind = "profile_share"
    elif any(w in q for w in ("which colleges", "list", "kaun se", "kin ", "किन", "कौन")):
        kind = "enumerate"
    elif any(w in q for w in ("best", "suggest", "recommend", "sabse accha")):
        kind = "recommend"
    elif ("?" not in q and filters
          and not any(w in q for w in ("which", "what", "how", "kaunsa", "kya", "kitna", "batao", "list"))):
        kind = "profile_share"
    return {
        "route": "data_query",
        "question_kind": kind,
        "language": "Hindi" if re.search(r"[ऀ-ॿ]", question) else "English",
        "filters": filters,
        "course_terms": terms,
        "needs_all_records": aggregate,
        "unit_note": note,
    }


def _computed_facts(rows):
    """Pre-compute superlatives in code so the model never does the math.

    LLMs reliably anchor on 'cheapest' when asked for 'best'; handing them the
    ranked facts (placement, fees) removes that failure mode the same way hard
    filters remove budget errors."""
    lines = []
    placed = sorted((r for r in rows if r["avg_placement_lpa"] > 0),
                    key=lambda r: -r["avg_placement_lpa"])[:3]
    if placed:
        lines.append("Highest reported avg placement: " + "; ".join(
            f"{r['college_id']} {r['name']} ({r['avg_placement_lpa']} LPA, "
            f"NAAC {r['naac_grade']}, Rs {r['annual_fees_inr']:,}/yr)" for r in placed))
    unreported = [r["college_id"] for r in rows if r["avg_placement_lpa"] == 0]
    if unreported:
        lines.append("Placement not reported / not applicable (never rank these as "
                     "worst): " + ", ".join(unreported))
    cheap = sorted(rows, key=lambda r: r["annual_fees_inr"])[:2]
    if cheap:
        lines.append("Lowest tuition: " + "; ".join(
            f"{r['college_id']} Rs {r['annual_fees_inr']:,}/yr" for r in cheap))
    costly = max(rows, key=lambda r: r["annual_fees_inr"])
    lines.append(f"Highest tuition: {costly['college_id']} Rs {costly['annual_fees_inr']:,}/yr")
    seats = max(rows, key=lambda r: r["total_seats"])
    lines.append(f"Most seats: {seats['college_id']} ({seats['total_seats']} seats)")
    oldest = min(rows, key=lambda r: r["established_year"])
    newest = max(rows, key=lambda r: r["established_year"])
    lines.append(f"Oldest: {oldest['college_id']} (est. {oldest['established_year']}); "
                 f"Newest: {newest['college_id']} (est. {newest['established_year']})")
    selective = max(rows, key=lambda r: r["last_year_cutoff_pct"])
    lines.append(f"Most selective (highest cutoff): {selective['college_id']} "
                 f"({selective['last_year_cutoff_pct']}%)")
    return ("COMPUTED FACTS (derived in code from CONTEXT — use these for any "
            "'best'/highest/lowest reasoning, and cite the ids). Every college "
            "in CONTEXT (including all listed below) ALREADY satisfies the "
            "user's stated budget/filters — never claim any of them exceeds "
            "the budget:\n- " + "\n- ".join(lines) + "\n")


def _clean_filters(filters):
    """Coerce router output to safe types; drop anything malformed (loudly)."""
    if not isinstance(filters, dict):
        if filters:
            print(f"[filter dropped] non-dict filters: {filters!r}", file=sys.stderr)
        return {}
    out = {}
    for key in ("max_annual_fee_inr", "student_score_pct"):
        v = filters.get(key)
        if v is not None:
            try:
                out[key] = int(float(v))
            except (TypeError, ValueError):
                print(f"[filter dropped] {key}={v!r}", file=sys.stderr)
    if filters.get("college_type") in ("Government", "Private", "Deemed"):
        out["college_type"] = filters["college_type"]
    elif filters.get("college_type"):
        print(f"[filter dropped] college_type={filters['college_type']!r}", file=sys.stderr)
    if filters.get("hostel_required") is True:
        out["hostel_required"] = True
    if isinstance(filters.get("city"), str) and filters["city"].strip():
        out["city"] = filters["city"].strip()
    return out


def _verify(result, context_ids, allow_uncited=False):
    """Deterministic grounding checks. Returns (ok, problems)."""
    problems = []
    citations = [c for c in result.get("citations", []) if isinstance(c, str)]
    bad = [c for c in citations if c not in context_ids]
    if bad:
        problems.append(f"citations not present in retrieved context: {bad}")
    # Any id code in the prose that never entered the context is a hallucination
    # signal (ids are scrubbed from the final text later either way).
    ghost = [c for c in set(CID_RE.findall(result.get("answer", ""))) if c not in context_ids]
    if ghost:
        problems.append(f"answer mentions ids outside retrieved context: {ghost}")
    # Counselling follow-ups (profile_share) legitimately state no facts.
    if result.get("answered") and not citations and not allow_uncited:
        problems.append("answered=true but citations empty")
    return (not problems), problems


def _norm(s):
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _norm_ws(s):
    """Lowercase, strip punctuation, KEEP word boundaries."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]+", " ", s.lower())).strip()


def _has_token(text_ws, token):
    """Word-boundary token match on space-normalized text. Prevents substring
    traps like 'LLB' hiding inside 'hiLLBlocks'."""
    return re.search(rf"\b{re.escape(_norm(token))}\b", text_ws) is not None


def _literal_programs(question, rows_by_id):
    """Program tokens from the dataset that the question literally names.

    Tokens come from each courses_offered entry + its leading word ("MBA",
    "B.Tech", "Diploma"). Matching is word-boundary and dot/space-insensitive
    ("btech" == "B.Tech"; "hill blocks" can never produce "llb")."""
    tokens = set()
    for r in rows_by_id.values():
        for tok in r["courses_offered"].split(";"):
            tok = tok.strip()
            if tok:
                tokens.add(tok)
                tokens.add(tok.split()[0])
    q_ws = _norm_ws(re.sub(r"[.\-]", "", question))
    return [t for t in tokens if len(_norm(t)) >= 3 and _has_token(q_ws, t)]


def _offers(r, token):
    return _has_token(_norm_ws(re.sub(r"[.\-]", "", r["courses_offered"])), token)


def _course_violations(result, question, rows_by_id):
    """Code-level exact-match check for course questions — no router involved.

    If the user's question literally names a program, every cited college must
    list it. Semantic asks ("engineering colleges") name no program token, so
    policy behaviour like diploma-with-disclosure is untouched."""
    if not result.get("answered"):
        return []
    named = _literal_programs(question, rows_by_id)
    if not named:
        return []
    problems = []
    for cid in result.get("citations", []):
        r = rows_by_id.get(cid)
        courses_ws = _norm_ws(re.sub(r"[.\-]", "", r["courses_offered"])) if r else ""
        if r and not any(_has_token(courses_ws, t) for t in named):
            problems.append(
                f"{cid} ({r['name']}) does not list {'/'.join(sorted(named))} in "
                f"its courses_offered — remove it from the answer: a college may "
                f"only be cited for a course question if it offers that course")
    return problems


def _resolve_list_selection(question, history, all_rows):
    """If the message picks an item ('1', '2 wala...') from the counsellor's
    MOST RECENT numbered list, resolve it in code — models grab the wrong
    (earlier) list when several are in the history window.

    Returns the matched college row, or None."""
    if not history:
        return None
    m = re.match(r"^\s*(\d{1,2})\b", question)
    if not m:
        return None
    n = int(m.group(1))
    # '1 lakh budget' / '75' (marks) / '2 colleges compare karo' (a count,
    # not a pick) are not selections
    if n > 15 or re.search(r"%|lakh|thousand|budget|fee|score|marks|साल|year"
                           r"|colleges|options|compare|list|which|kaun|kitne|how many",
                           question, re.I):
        return None
    for msg in reversed(history):
        if not (isinstance(msg, dict) and msg.get("role") == "assistant"):
            continue
        items = dict(re.findall(r"(?:^|\n)\s*(\d{1,2})\.\s*([^\n]+)",
                                str(msg.get("content", ""))))
        if not items:
            continue  # keep scanning for the latest message that HAS a list
        text = items.get(str(n), "")
        for r in all_rows:
            if _norm(r["name"]) in _norm(text):
                return r
        return None  # latest list found but item didn't name a college
    return None


def _resolve_list_items(history, all_rows):
    """All college rows named in the counsellor's most recent numbered list."""
    if not history:
        return []
    for msg in reversed(history):
        if not (isinstance(msg, dict) and msg.get("role") == "assistant"):
            continue
        items = re.findall(r"(?:^|\n)\s*\d{1,2}\.\s*([^\n]+)",
                           str(msg.get("content", "")))
        if not items:
            continue
        rows = []
        for text in items:
            for r in all_rows:
                if _norm(r["name"]) in _norm(text):
                    rows.append(r)
                    break
        return rows
    return []


_HINGLISH_WORDS = re.compile(
    r"\b(hai|hain|nahi|nahin|kya|kyu|kyun|kaise|kaun|kitna|kitni|kitne|mera|"
    r"mere|meri|mujhe|mujhko|aap|apka|aapka|bhaiya|didi|wala|wale|wali|"
    r"chahiye|batao|bata|karna|karu|karoon|sakta|sakti|milega|milegi|accha|"
    r"theek|thik|paisa|paise|sasta|sasti|padhai|hoon|hun|mein|aur|lekin|"
    r"magar|uska|uski|iske|iska|bhi|toh|na|yaar)\b", re.I)


def _lang_hint(question):
    """Deterministic language detection for the reply-language directive.

    Rule 9 asks the model to mirror the question's language, but live testing
    showed it drifting to English on short Hinglish turns mid-conversation
    ("hostel bhi chahiye mujhe" answered in English). A code-detected,
    per-turn directive is far more reliable than a buried prompt rule."""
    if re.search(r"[ऀ-ॿ]", question):
        return "Hindi (Devanagari script)"
    hits = {m.lower() for m in _HINGLISH_WORDS.findall(question)}
    if len(hits) >= 2:
        return "Hinglish (Roman-script Hindi-English mix, like the student's message)"
    return None


_BANNED_VOICE_WORDS = re.compile(
    r"\b(dataset|database|our records|the records|its records|our system)\b", re.I)


# Word-choice slips are cosmetic, not grounding failures — so they are
# rewritten in place rather than paid for with a whole extra LLM round trip.
# (Measured: voice slips caused ~4 of every 5 verification retries, and each
# retry added ~1.3-2 s to the answer. Substituting is instant and changes no
# fact.) Ordered longest-first so phrases win over the bare word.
_VOICE_FIXES = (
    # most specific first, so the result reads like a sentence a person wrote
    (re.compile(r"\b(?:the |our )?(?:dataset|database|records)\s+"
                r"(?:does not|doesn'?t)\s+(?:list|include|have|show|contain)\b", re.I),
     "we don't have"),
    (re.compile(r"\bin (?:our|the|its) (?:records|dataset|database)\b", re.I), "in our list"),
    (re.compile(r"\b(?:our|the|its) (?:records|dataset|database)\b", re.I), "our college list"),
    (re.compile(r"\bour system\b", re.I), "our college list"),
    (re.compile(r"\b(?:dataset|database)\b", re.I), "college list"),
    (re.compile(r"\brecords\b", re.I), "listings"),
)


def _repair_voice(text):
    """Rewrite database-speak into counsellor voice. Returns the fixed text."""
    for pat, repl in _VOICE_FIXES:
        text = pat.sub(repl, text)
    text = re.sub(r"\b(our|the)\s+(our|the)\b", r"\2", text, flags=re.I)  # "our our"
    # substitutions can start a sentence lowercase — restore capitalisation
    text = re.sub(r"(^|(?<=[.!?]\s)|(?<=\n))([a-z])",
                  lambda m: m.group(1) + m.group(2).upper(), text)
    return text


def _voice_violations(result):
    """Repair corporate/database-speak in place; only escalate if it survives.

    Returns [] after a successful repair — a regeneration is reserved for
    failures that actually threaten grounding."""
    answer = result.get("answer", "")
    if not _BANNED_VOICE_WORDS.search(answer):
        return []
    result["answer"] = _repair_voice(answer)
    if _BANNED_VOICE_WORDS.search(result["answer"]):  # repair didn't take
        return ["answer still uses database-speak — rephrase in warm, casual "
                "counsellor voice, never referencing 'records'/'dataset'"]
    print("[voice] repaired database-speak in place (no retry needed)", file=sys.stderr)
    return []


_MONEY_RE = re.compile(r"(?:rs\.?|₹|inr)\s*([\d][\d,]{3,})", re.I)
_LPA_RE = re.compile(r"(\d+(?:\.\d+)?)\s*lpa", re.I)
# cutoff / seats / year are matched ONLY when tied to their label, so a
# student's own "78%" or a career "next 5 years" never gets checked as a fact.
_CUTOFF_RE = re.compile(r"(\d{1,2})\s*%\s*(?:aggregate|cut[- ]?off)"
                        r"|cut[- ]?off\s*(?:of|is|was|:)?\s*(\d{1,2})\s*%", re.I)
_SEATS_RE = re.compile(r"(\d{2,4})\s*(?:total\s+)?seats", re.I)
_YEAR_RE = re.compile(r"(?:established|founded)\s*(?:in)?\s*(\d{4})", re.I)


def _numeric_violations(result, all_rows, ignore_money=None):
    """Every hard number in the answer that names a college fact — tuition,
    placement, cutoff, seats, founding year — must trace to a CITED college's
    actual field, checked against the data itself, not the model's word.
    Numbers are exactly where a wrong answer destroys trust and where LLMs are
    least reliable, so a stated figure that matches no cited record is blocked
    and regenerated. Each number is matched only in a form clearly attributed
    to that field ('82% cutoff', '420 seats', 'established in 2007'), so a
    student's own score ('I got 78%') or a conversational figure ('next 5
    years') is never mistaken for a fact-claim."""
    if not result.get("answered"):
        return []
    cited = [r for r in all_rows if r["college_id"] in set(result.get("citations", []))]
    if not cited:
        return []
    answer = result.get("answer", "")
    ignore = set(ignore_money or ())
    true_fees = {r["annual_fees_inr"] for r in cited}
    # allow per-semester conversions (fee/2) and the student's own budget number
    fee_ok = true_fees | {f * 2 for f in true_fees} | {f // 2 for f in true_fees} | ignore
    true_lpa = {round(r["avg_placement_lpa"], 1) for r in cited if r["avg_placement_lpa"] > 0}
    true_cutoffs = {r["last_year_cutoff_pct"] for r in cited}
    true_seats = {r["total_seats"] for r in cited}
    true_years = {r["established_year"] for r in cited}

    problems = []
    for m in _MONEY_RE.finditer(answer):
        val = int(m.group(1).replace(",", ""))
        if val >= 10000 and val not in fee_ok:
            problems.append(
                f"answer states Rs {val:,}, which is not the tuition of any cited "
                f"college (real fees: {sorted(true_fees)}) — correct it or drop it")
    for m in _LPA_RE.finditer(answer):
        val = round(float(m.group(1)), 1)
        # Checked even when NO cited college reports a figure — in that case
        # any stated LPA is by definition invented (e.g. a number for the
        # medical college whose placement is "not reported").
        if val not in true_lpa:
            problems.append(
                f"answer states {val} LPA, which is not the placement of any cited "
                f"college (real: {sorted(true_lpa) or 'none reported'}) — correct it or drop it")
    for m in _CUTOFF_RE.finditer(answer):
        val = int(m.group(1) or m.group(2))
        if val not in true_cutoffs:
            problems.append(
                f"answer states a {val}% cutoff, which matches no cited college "
                f"(real cutoffs: {sorted(true_cutoffs)}) — correct it or drop it")
    for m in _SEATS_RE.finditer(answer):
        val = int(m.group(1))
        if val not in true_seats:
            problems.append(
                f"answer states {val} seats, which matches no cited college "
                f"(real: {sorted(true_seats)}) — correct it or drop it")
    for m in _YEAR_RE.finditer(answer):
        val = int(m.group(1))
        if val not in true_years:
            problems.append(
                f"answer says established {val}, which matches no cited college "
                f"(real: {sorted(true_years)}) — correct it or drop it")
    return problems


def _name_violations(result, all_rows, context_ids):
    """Every college NAMED in the prose must be backed by a citation.

    Catches the citation-dodge failure mode: the model drops a flagged id from
    `citations` but leaves the college in the text with a disclaimer."""
    if not result.get("answer"):
        return []
    answer_norm = _norm(result["answer"])
    citations = set(result.get("citations", []))
    problems = []
    for r in all_rows:
        cid = r["college_id"]
        if _norm(r["name"]) in answer_norm and cid not in citations:
            if cid in context_ids:
                problems.append(
                    f"the answer names {r['name']} but {cid} is not in citations — "
                    f"either cite it (only if it truly qualifies for the question) "
                    f"or remove every mention of it from the answer")
            else:
                problems.append(
                    f"the answer names {r['name']} which is not even in CONTEXT — "
                    f"remove every mention of it")
    return problems


def _strip_ids(text):
    """Ids belong in the citations array; scrub any stray codes from prose."""
    text = re.sub(r"\s*[\(\[]\s*(?:id\s*)?C0\d{2}\s*[\)\]]", "", text)
    text = re.sub(r"\bC0\d{2}\b\s*,\s*", "", text)   # "(C014, 63%)" -> "(63%)"
    text = re.sub(r",?\s*\bC0\d{2}\b", "", text)
    text = re.sub(r"\(\s*,\s*", "(", text)           # leftover "(, ..." artifacts
    text = re.sub(r",\s*\)", ")", text)
    text = re.sub(r"\(\s*\)", "", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def _format_answer(text):
    """Clean prose + guaranteed structure: each numbered item on its own line
    (deterministic — independent of whether the model emitted newlines).

    Splits only when a genuine list exists (both "1." and "2." present), so a
    sentence that merely ends in a small number ("cutoff is 82. Hostel...")
    is never broken apart."""
    text = _strip_ids(text)
    nums = {int(m.group(1)) for m in re.finditer(r"(?:^|\s)(\d{1,2})\.\s", text)}
    if 1 in nums and 2 in nums:
        text = re.sub(r"\s+(\d{1,2}\.\s)", r"\n\1", text)
    return text.strip()


_METRICS_LOCK = threading.Lock()  # FastAPI serves from a threadpool


def _log_metrics(record):
    config.METRICS_FILE.parent.mkdir(exist_ok=True)
    with _METRICS_LOCK, open(config.METRICS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Zero-LLM fast path for pure greetings / thanks / goodbyes. These are the
# most common first messages, and burning two model calls (~3-4s) on "hi" is
# the single worst latency a student ever feels. Matching is deliberately
# strict — exact short phrases only — so anything substantive still takes the
# full pipeline. Replies rotate so a demo never looks canned.
_FAST_GREET_RE = re.compile(
    r"^(?:hi+|he+y+|hello+|helo|yo|hii+|hola|namaste+(?:\s*ji)?|namaskar|"
    r"good\s*(?:morning|afternoon|evening)|(?:hi+|hello+|he+y+)\s*(?:there|mimi|ji|bhaiya|didi))$")
_FAST_THANKS_RE = re.compile(
    r"^(?:ok(?:ay)?\s+)?(?:thanks+|thank\s*you+|thankyou|thanku|thnx|thx|ty|"
    r"dhanyawad|dhanyavad|shukriya)(?:\s*(?:a\s*lot|so\s*much|very\s*much|mimi|ji|bhaiya|didi))?$")
_FAST_BYE_RE = re.compile(
    r"^(?:bye+|good\s*bye|goodbye|bye\s*bye|alvida|tata|see\s*you|"
    r"bye\s*(?:mimi|ji|bhaiya|didi))$")

_GREET_REPLIES_EN = [
    "Hi! 👋 How are you doing? It's great that you've passed Class 12! You are now entering one of the critical phases of your life from where your real journey begins. I'm mimi, and I'm here to help. To get started, how much did you score in Class 12 (%)?",
    "Hello! 😊 How are you? Congratulations on passing Class 12! You are stepping into a crucial phase where your real journey begins. I'm mimi — let's find you a great college! How much did you score in Class 12 (%)?",
    "Hey there! 👋 How are you doing today? Passing Class 12 is a huge milestone, and now your real journey begins! I'm mimi, and I'll help you navigate this important phase. What did you score in Class 12 (%)?",
]
_GREET_REPLIES_HI = [
    "Namaste! 🙏 Main mimi hoon — aap kaise ho? Batao, Class 12 me kitne percent "
    "aaye? Wahin se apka perfect college dhoondhna shuru karte hain!",
    "Namaste ji! Main mimi — sab badhiya? Chalo shuru karein: Class 12 me aapka "
    "score kitna raha (%)?",
]
_THANKS_REPLIES = [
    "Anytime! 😊 That's what I'm here for — ask me whenever the next question pops up.",
    "You're most welcome! Rooting for you — ping me any time you need college info.",
    "Happy to help! Come back whenever — courses, fees, hostels, anything.",
]
_BYE_REPLIES = [
    "Bye! All the best — you've got this. 🎓 Come back any time you have questions!",
    "Take care! Best of luck with the admissions — I'm right here whenever you need me.",
]


def _fast_smalltalk(question, history, known_profile):
    """Return a canned result for trivial greetings/thanks/byes, else None."""
    q = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", question.lower(), flags=re.UNICODE)).strip()
    if not q or len(q) > 40:
        return None
    if _FAST_THANKS_RE.match(q):
        text = random.choice(_THANKS_REPLIES)
    elif _FAST_BYE_RE.match(q):
        text = random.choice(_BYE_REPLIES)
    elif _FAST_GREET_RE.match(q):
        # Mid-conversation greetings fall through to the pipeline — the canned
        # opener asks for marks, which must never be RE-asked once known.
        if history or (isinstance(known_profile, dict) and any(known_profile.values())):
            return None
        text = random.choice(
            _GREET_REPLIES_HI if re.search(r"namaste|namaskar|ji$|bhaiya|didi", q)
            else _GREET_REPLIES_EN)
    else:
        return None
    return {"answer": text, "citations": [], "answered": True,
            "reason_if_unanswered": None}


_CACHE = {}  # normalized question -> result (demo server only; see config)
_CACHE_LOCK = threading.Lock()
_CACHE_MAX = 300  # simple FIFO cap so a long-lived demo can't grow unbounded
_CACHE_FILE = config.ROOT / "data" / "answer_cache.json"


def _cache_store(key, result):
    with _CACHE_LOCK:
        _CACHE[key] = dict(result)
        while len(_CACHE) > _CACHE_MAX:
            _CACHE.pop(next(iter(_CACHE)))
        try:  # persist best-effort: server restarts keep their warm cache,
              # so prewarm costs real LLM calls exactly once, ever
            tmp = _CACHE_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(_CACHE, ensure_ascii=False), encoding="utf-8")
            tmp.replace(_CACHE_FILE)
        except Exception as e:
            print(f"[cache] persist failed: {e}", file=sys.stderr)


def _cache_load():
    if not config.CACHE_ENABLED:
        return
    try:
        if _CACHE_FILE.exists():
            data = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                _CACHE.update(data)
                print(f"[cache] loaded {len(data)} answers from disk", file=sys.stderr)
    except Exception as e:
        print(f"[cache] load failed: {e}", file=sys.stderr)


_cache_load()


def answer_question(question, history=None, known_profile=None):
    """Full pipeline. Returns the output dict for answer.py to print.

    `history` (optional): recent [{role, content}, ...] turns from the demo
    UI, giving the counsellor conversation memory. The graded CLI path
    (answer.py) never passes it — each published question stands alone.

    `known_profile` (optional): the student profile accumulated so far this
    session ({marks_pct, field_interest, budget, location}), as returned by a
    prior call's "profile" key. History alone is truncated to the last few
    turns for prompt size — a fact from turn 1 would silently "fall off" a
    long conversation without this; the client re-sends the running profile
    every turn so it survives regardless of how long the chat gets."""
    t_start = time.perf_counter()
    # Cache key: lowercase + whitespace-collapse ONLY. Do not be tempted to
    # use _norm() here — it strips non-Latin characters, which would collapse
    # every Hindi/Tamil question onto a single cache entry (a real bug we hit).
    cache_key = re.sub(r"\s+", " ", question.strip().lower())
    if config.CACHE_ENABLED and not history and cache_key in _CACHE:
        print("[cache] hit", file=sys.stderr)
        result = dict(_CACHE[cache_key])
        # The demo client tracks a running profile; a cache hit must not make
        # the "profile" key vanish (the client would keep its own copy, but
        # the contract stays consistent either way).
        if history is not None or known_profile is not None:
            result["profile"] = dict(known_profile) if isinstance(known_profile, dict) else {}
        return result
    calls = []  # per-LLM-call token/latency records

    # ---- 0. Code-level guards & fast paths (no LLM, no retrieval) ----------
    # Order matters: these run BEFORE the router so a greeting, an injection
    # attempt or a distress message never spends a model call (or its ~2s).
    fast = _fast_smalltalk(question, history, known_profile)
    if fast is not None:
        _log_metrics({"question": question, "route": "smalltalk", "calls": [],
                      "retrieved": [], "verified": True, "fast_path": True,
                      "total_latency_s": round(time.perf_counter() - t_start, 3)})
        if history is not None or known_profile is not None:
            fast["profile"] = dict(known_profile) if isinstance(known_profile, dict) else {}
        return fast

    # Defence-in-depth: an obvious "ignore your rules" attempt gets a calm,
    # on-brand deflection in code, before it can reach any model.
    if _INJECTION_RE.search(question):
        _log_metrics({"question": question, "route": "injection_blocked", "calls": calls,
                      "retrieved": [], "verified": True,
                      "total_latency_s": round(time.perf_counter() - t_start, 3)})
        result = {"answer": "Haha, nice try! But I'm just here to help you find the "
                            "right college — courses, fees, cutoffs, hostels, "
                            "scholarships. What would you like to know?",
                  "citations": [], "answered": True, "reason_if_unanswered": None}
        if history is not None or known_profile is not None:
            result["profile"] = dict(known_profile) if isinstance(known_profile, dict) else {}
        return result

    # Safety guard: a distressed student must never get an "I only help with
    # colleges" brush-off. Caught in code so it can't depend on the router.
    if _DISTRESS_RE.search(question):
        care = ("I'm really glad you told me, and I want you to know you're not "
                "alone in this. A score or a college is never worth more than "
                "you are. Please talk to someone you trust — a parent, teacher, "
                "or a friend — and if things feel too heavy, India's free "
                "helpline KIRAN (1800-599-0019) has kind people who listen, "
                "any time. I'm here whenever you're ready to talk colleges again.")
        _log_metrics({"question": question, "route": "distress", "calls": calls,
                      "retrieved": [], "verified": True,
                      "total_latency_s": round(time.perf_counter() - t_start, 3)})
        result = {"answer": care, "citations": [], "answered": True,
                  "reason_if_unanswered": None}
        if history is not None or known_profile is not None:
            result["profile"] = dict(known_profile) if isinstance(known_profile, dict) else {}
        return result

    hist_text = ""
    if history:
        lines = []
        for m in history[-6:]:
            if isinstance(m, dict) and m.get("content"):
                role = "Student" if m.get("role") == "user" else "Counsellor"
                lines.append(f"{role}: {str(m['content'])[:250]}")
        if lines:
            hist_text = "CONVERSATION SO FAR:\n" + "\n".join(lines) + "\n\n"

    # Start embedding the question NOW, in parallel with the router call —
    # by the time the router returns, the vector (and on cold starts, the
    # embedding model itself) is ready. See _EmbedPrefetch.
    prefetch = _EmbedPrefetch(question)

    # ---- 1. Route + extract filters (small, cheap model) -------------------
    try:
        raw = chat(
            [{"role": "system", "content": ROUTER_SYSTEM},
             {"role": "user", "content": f"{hist_text}CURRENT MESSAGE: {question}"}],
            model=config.FAST_MODEL, json_mode=True, temperature=0.0,
            max_tokens=300, usage_log=calls,
        )
        route_info = _parse_json(raw)
        if not isinstance(route_info, dict):
            raise ValueError(f"router returned {type(route_info).__name__}, not an object")
    except Exception as e:  # router must never kill the pipeline
        print(f"[router fallback] {e}", file=sys.stderr)
        route_info = _heuristic_route(question)

    route = route_info.get("route", "data_query")
    filters = _clean_filters(route_info.get("filters"))
    course_terms = [t for t in (route_info.get("course_terms") or [])
                    if isinstance(t, str)] if isinstance(route_info.get("course_terms"), list) else []
    needs_all = bool(route_info.get("needs_all_records"))
    # Merge: facts the client already knew (survives history truncation) +
    # anything the router freshly extracted this turn. Client-known facts are
    # the base so a fact from turn 1 is never lost by turn 15; the router
    # only ADDS, never overwrites an established fact with a blank one.
    turn_profile = route_info.get("profile") if isinstance(route_info.get("profile"), dict) else {}
    merged_profile = dict(known_profile) if isinstance(known_profile, dict) else {}
    for k, v in turn_profile.items():
        if v:
            merged_profile[k] = v
    unit_note = route_info.get("unit_note") if isinstance(route_info.get("unit_note"), str) else None
    # unit_note is strictly for money/unit conversions; routers occasionally
    # stuff commentary there, which must never leak into an answer.
    if unit_note and not re.search(r"\d|lakh|semester|year|annual|rs\b|₹", unit_note, re.I):
        unit_note = None

    # Lookup questions name a specific college ("can I get into X?", "what is
    # X's fee?") — hard-filtering by the student's score/budget would remove
    # the very college they asked about, making an honest "no, its cutoff is
    # higher than your score" impossible. Eligibility is computed in code and
    # handed to the generator as facts (see ELIGIBILITY FACTS below).
    raw_score = filters.get("student_score_pct")
    raw_budget = filters.get("max_annual_fee_inr")  # for the numeric guardrail's
                                                    # budget exclusion (below)
    if route_info.get("question_kind") == "lookup":
        filters = {}

    # "Any other college nearby?" / "something else?" means the student wants
    # to look BEYOND the current city — so the city filter must come off, not
    # stay on and produce "that's the only one here" three turns in a row.
    broadening = bool(_BROADEN_RE.search(question))
    if broadening and filters.get("city"):
        print(f"[broaden] dropping city filter {filters['city']!r}", file=sys.stderr)
        filters.pop("city", None)

    # Deterministic guard: marks/budget statements are counselling openers,
    # never smalltalk — small routers occasionally misfile them.
    if route == "smalltalk" and re.search(r"\d+\s*%|\blakh\b|budget|marks|score|percent", question, re.I):
        route = "data_query"
        route_info["question_kind"] = "profile_share"

    # ---- 2a. Non-data routes skip retrieval AND the big model entirely -----
    _GREETING_TONES = [
        "warm and supportive — like a friendly older mentor checking in",
        "upbeat and welcoming — genuinely happy to chat and help them out",
        "chill and reassuring — making them feel right at home before diving in",
    ]
    if route in ("smalltalk", "out_of_scope"):
        tone = random.choice(_GREETING_TONES)
        prompt = (
            f"The user said: {question!r}. You are mimi, a friendly, warm college counsellor "
            f"texting a student like a supportive older friend. Reply in the SAME language and script as "
            f"the user's message, 1-2 sentences. Never mention 'dataset' or 'database'. Never re-introduce yourself with "
            f"your full title ('your dedicated college counsellor here at MakeMyEducation'). "
            + (f"If the user is greeting you (like 'hi', 'hello', 'hey'), always greet them warmly back and ask how they are doing first (e.g., 'Hi! How are you?' or 'Hello! How are you doing today?'). "
               f"Then naturally ask how their Class 12 marks are looking or how you can help them find their dream college. Write this reply in a tone that is {tone}."
               if route == "smalltalk"
               else "Politely say you can only help with colleges and admissions, "
                    "and invite such a question.")
        )
        try:
            text = chat([{"role": "user", "content": prompt}], model=config.FAST_MODEL,
                        temperature=0.7, max_tokens=120, usage_log=calls)
        except Exception:
            text = ("Hi! How are you? Let me know how much you scored in Class 12 so we can start finding your dream college!")
        # A reasoning fallback model may prepend <think>…</think> — never let
        # that leak into a user-visible reply.
        text = _THINK_RE.sub("", text).strip()
        result = {
            "answer": text,
            "citations": [],
            "answered": route == "smalltalk",
            "reason_if_unanswered": None if route == "smalltalk"
            else "Out of scope: I only answer questions about the colleges in the dataset.",
        }
        _log_metrics({"question": question, "route": route, "calls": calls,
                      "retrieved": [], "verified": True,
                      "total_latency_s": round(time.perf_counter() - t_start, 3)})
        if config.CACHE_ENABLED:
            _cache_store(cache_key, result)
        # Preserve the running profile through a smalltalk turn — a "thanks!"
        # or "hi" mid-conversation must not reset what's already been learned.
        if history is not None or known_profile is not None:
            result["profile"] = merged_profile
        return result

    # ---- 2b. Hybrid retrieval ----------------------------------------------
    # Superlatives/counts need the full dataset in view: with only 15 records
    # the honest fix is to widen k to all of them, not to let top-k truncation
    # produce a confidently wrong "cheapest college".
    top_k = 15 if needs_all else config.TOP_K
    qv = prefetch.result()  # computed concurrently with the router call above
    all_rows = load_colleges()
    results = retrieve(question, filters=filters, course_terms=course_terms,
                       top_k=top_k, query_vec=qv)
    relaxed_note = ""
    if not results and any(filters.values() if filters else []):
        # Zero matches must not dead-end the student: relax the filters so the
        # counsellor can show the nearest real options — honestly labelled.
        stated = {k: v for k, v in filters.items() if v}
        results = retrieve(question, filters=None, course_terms=course_terms,
                           top_k=15, query_vec=qv)
        relaxed_note += (
            f"ZERO MATCHES: no listed college satisfies the stated constraints "
            f"{json.dumps(stated)}. Counsel honestly, never dead-end: say plainly "
            f"that none of the colleges we work with met these constraints last "
            f"year (e.g. every cutoff is above the student's score), then present "
            f"the 2-3 NEAREST options as future targets with their real numbers "
            f"(lowest cutoffs / closest fees), and one encouraging line about the "
            f"path forward. NEVER present any college as currently meeting the "
            f"stated constraints.\n")
        filters = {}
    # A named program must never be DENIED just because the student's own
    # filters (score/budget) removed every college that offers it. Live-tested
    # failure: "kya main LLB kar sakta hoon? mere 70% hain" -> the 70% filter
    # dropped Haldwani Law College (cutoff 74), the only LLB provider, and the
    # model honestly-but-wrongly said no college offers LLB. Detect the
    # conflict in code, widen the context, hand the model the honest framing.
    stated_now = {k: v for k, v in (filters or {}).items() if v}
    if stated_now and results:
        # The program can come from the CURRENT question ("can I do LLB?") or
        # from the running profile ("...LLB..." three turns ago, now just
        # "which colleges are nearby?"). Live-tested failure: the follow-up
        # question named no program, so this guard stayed silent and the model
        # said "we don't have any LLB programs" — contradicting its own
        # earlier, correct answer.
        prog_text = question + " " + str(
            (merged_profile or {}).get("field_interest") or "")
        named_prog = _literal_programs(prog_text, {r["college_id"]: r for r in all_rows})
        ctx_rows = [r for r, _ in results]
        missing = [t for t in named_prog
                   if not any(_offers(r, t) for r in ctx_rows)]
        offering = {t: [r for r in all_rows if _offers(r, t)] for t in missing}
        offering = {t: rs for t, rs in offering.items() if rs}
        if offering:
            results = retrieve(question, filters=None, course_terms=course_terms,
                               top_k=15, query_vec=qv)
            lines = []
            for t, rs in offering.items():
                lines.append(f"{t}: " + "; ".join(
                    f"{r['college_id']} {r['name']}, {r['city']} "
                    f"(cutoff {r['last_year_cutoff_pct']}%, Rs {r['annual_fees_inr']:,}/yr)"
                    for r in rs))
            relaxed_note += (
                "PROGRAM vs FILTER CONFLICT (computed in code): the asked program IS "
                "offered, but every college offering it failed the student's stated "
                f"constraints {json.dumps(stated_now)}. The colleges that DO offer it:\n- "
                + "\n- ".join(lines) +
                "\nAnswer honestly: NEVER say the program is not offered. Name these "
                "colleges with their real numbers, state plainly which constraint they "
                "miss (e.g. cutoff above the student's score — give both numbers), and "
                "add one encouraging line about the path forward (a compartment/"
                "improvement exam, or streams that do fit the constraints).\n")
            filters = {}
    context_ids = {r["college_id"] for r, _ in results}
    context = to_context(results) if results else "(no colleges matched the hard filters)"

    user_msg = hist_text + f"CONTEXT:\n{context}\n\n"
    if relaxed_note:
        user_msg += relaxed_note
    negated = re.search(r"\b(not|no|without|don'?t|doesn'?t|nahi|nahin|नहीं|बिना)\b",
                        question, re.I)
    if results and (needs_all or route_info.get("question_kind") == "recommend"):
        user_msg += _computed_facts([r for r, _ in results])
    if negated and results:
        # complement sets are computed in code — negated enumerations must be
        # complete, and models reliably under-count them
        rows_ctx0 = {r["college_id"]: r for r, _ in results}
        for t in _literal_programs(question, rows_ctx0)[:2]:
            non = [cid for cid in sorted(rows_ctx0) if not _offers(rows_ctx0[cid], t)]
            if non:
                user_msg += (f"NEGATION FACTS (computed in code): colleges NOT "
                             f"offering {t}: {', '.join(non)} — a complete "
                             f"'not offering' answer lists ALL of these.\n")
        if re.search(r"hostel|हॉस्टल", question, re.I):
            non = [cid for cid in sorted(rows_ctx0)
                   if rows_ctx0[cid]["hostel_available"].strip().lower() == "no"]
            if non:
                user_msg += (f"NEGATION FACTS (computed in code): colleges WITHOUT "
                             f"hostel: {', '.join(non)}.\n")
    if raw_score is not None and results:
        # Models mis-compare numbers ("78 is above 82") — verdicts come from code.
        verdicts = "; ".join(
            f"{r['college_id']} cutoff {r['last_year_cutoff_pct']}% -> "
            + ("ELIGIBLE" if raw_score >= r["last_year_cutoff_pct"] else "NOT eligible")
            for r, _ in results)
        user_msg += (f"ELIGIBILITY FACTS (computed in code from the student's score "
                     f"of {raw_score}%): {verdicts}. Use these verdicts verbatim — "
                     f"never recompute them yourself.\n")
    selected = _resolve_list_selection(question, history, all_rows)
    if selected:
        user_msg += (f"LIST SELECTION (resolved in code from your most recent "
                     f"numbered list): the student's '{question.strip()[:30]}' means "
                     f"item about {selected['name']} ({selected['college_id']}). "
                     f"Reply with the FULL profile of this college only — every "
                     f"field as short '- label: value' lines. Do not repeat the list.\n")
    elif history and re.search(r"compare|tulna|difference|versus|\bvs\b|फ़?र्क|अंतर",
                               question, re.I):
        # "compare all 3" / "in dono me kya farak hai" — when no college is
        # named, the referent is the most recent list; models grab the wrong
        # colleges if left to guess, so resolve it in code.
        if not any(_norm(r["name"]) in _norm(question) for r in all_rows):
            listed = _resolve_list_items(history, all_rows)
            if len(listed) >= 2:
                ids = "; ".join(f"{r['name']} ({r['college_id']})" for r in listed[:4])
                user_msg += (f"LIST REFERENCE (resolved in code): the student is "
                             f"referring to your most recent list — compare EXACTLY "
                             f"these colleges and NO others: {ids}.\n")

    if hist_text:
        # Feed back the previous reply's opening words so the model can't fall
        # into a sympathy loop ("I hear you..." three turns running).
        prev_open = ""
        for m in reversed(history or []):
            if isinstance(m, dict) and m.get("role") == "assistant" and m.get("content"):
                prev_open = " ".join(str(m["content"]).split()[:6])
                break
        if prev_open:
            user_msg += (f"YOUR PREVIOUS REPLY BEGAN: \"{prev_open}...\" — open this "
                         f"one in a completely different way. Do not reuse that "
                         f"phrasing or any equivalent sympathy formula.\n")
        known = {k: v for k, v in merged_profile.items() if v}
        missing = [f for f in ("marks_pct", "field_interest", "budget", "location")
                   if not merged_profile.get(f)]
        user_msg += (
            "STUDENT PROFILE (gathered across the whole conversation): "
            + (json.dumps(known, ensure_ascii=False) if known else "(nothing yet)")
            + (f". Not yet known: {', '.join(missing)}" if missing else "")
            + ".\nCOUNSELLOR FLOW: never re-ask anything already in the profile. "
              "NEVER ask permission to show options — if the student asks for "
              "other/nearby/alternative colleges, SHOW them straight away from "
              "CONTEXT (named, with fees), never reply 'let me know and I'll be "
              "happy to help' or 'if you're open to other cities' and stop "
              "there. Saying 'X is the only one in that city' is fine ONLY when "
              "you immediately follow it with the actual nearby alternatives in "
              "the same reply. Never give the same non-answer twice in a "
              "conversation — if a previous turn already said something is the "
              "only option, this turn must move forward with real options. "
              "Resolve pronouns ('it', 'that college', 'uska') from the "
              "conversation — 'more details about it' means the college discussed "
              "in your previous reply. A number/ordinal ('1', 'pehla') selects "
              "that item from your MOST RECENT numbered list (a LIST SELECTION "
              "note above resolves it authoritatively when present) — reply with "
              "that college's full details, never repeat or continue the list. "
              "If the current message asks a specific "
              "question, answer it. Otherwise run a step-by-step intake: warmly "
              "acknowledge, then ask for exactly ONE missing item per turn, in "
              "this order: marks -> field/stream -> budget. THE MOMENT all three "
              "(marks, field, budget) are known — typically the very turn the "
              "student states their budget — you MUST present the matching "
              "colleges IN THAT SAME REPLY as a numbered list with key facts. "
              "NEVER delay the list with another intake question: city, college "
              "type or hostel are optional refinements offered in ONE short "
              "line AFTER the list ('Want me to narrow these down by city or "
              "hostel?'), never a gate before it. Same when the student says "
              "budget is no constraint or asks to just see the options. "
              "E.g. when they share their field, reply: "
              "'Great choice! And what is your yearly budget for tuition, so I "
              "shortlist the best fits?' If a field/course interest is in the "
              "profile, keep every answer scoped to it unless the student "
              "changes topic. If the student is UNSURE/lost about their field "
              "('confused hoon', 'pata nahi', 'what should I do') — talk like a "
              "big brother, NOT a corporate FAQ:\n"
              "  Line 1: reassure, casual, ONE line — 'Koi na, that's normal — "
              "let's figure it out together.' / 'No stress, happens to everyone "
              "— let's chat it through.'\n"
              "  Line 2-3: 2-3 SHORT lines (not one dense paragraph — each on "
              "its own line, separated by \\n) of general career-trend guidance "
              "in plain talk — 'Tech/CS is hot right now, tons of hiring.' / "
              "'Medicine and pharmacy are always solid if you like science.' "
              "Clearly general chat, NOT college data — don't cite colleges for "
              "this part.\n"
              "  Last line: ONE casual question — 'What kind of stuff do you "
              "enjoy — building things, numbers, helping people, creative "
              "work?' Never dump the full stream list (Engineering/Management/"
              "Medical/...) as a menu; that reads like a form, not a chat. Never "
              "invent college facts to support trend talk, and don't cite "
              "college ids in this reassurance turn — save citations for once "
              "you're actually recommending colleges.\n"
              "Whenever a field/stream comes up — whether you're naming 2-3 "
              "options for an unsure student, or the student has already "
              "picked one — explicitly mention its outlook over the NEXT 5-10 "
              "YEARS and a possible LONG-TERM GOAL, by name, not just 'growing "
              "fast' (e.g. 'CS/tech — huge demand for the next 5-10 years with "
              "AI and data, long-term you could specialise, go into research, "
              "or start your own thing'; 'medicine — a long road but rock-"
              "solid demand for decades, long-term goal is usually your own "
              "practice or a specialisation'; 'commerce/management — steady "
              "demand, long-term goal is often a leadership role or your own "
              "business'). One such line per field mentioned. Keep it general "
              "career perspective, not college-specific data, phrased like a "
              "big sibling sharing real perspective, not a report.\n")
    if unit_note:
        user_msg += (f"UNIT ASSUMPTION (weave it naturally into the answer as one "
                     f"phrase — NEVER print a label like 'unit note'): {unit_note}\n")
    if filters:
        applied = {k: v for k, v in filters.items() if v}
        if applied:
            user_msg += (f"HARD FILTERS ALREADY APPLIED to CONTEXT: {json.dumps(applied)} "
                         f"(colleges failing them were removed before you saw the data). "
                         f"CONTEXT IS THEREFORE A FILTERED VIEW, NOT THE WHOLE LIST — "
                         f"never say 'we don't have any X' or 'X isn't offered anywhere' "
                         f"based on it. The honest phrasing is 'no college matching your "
                         f"score/budget offers X', which is a completely different "
                         f"statement, and never contradict something you already told "
                         f"the student earlier in this conversation.\n")
            if "max_annual_fee_inr" in applied:
                user_msg += ("REMINDER for budget answers: tuition figures EXCLUDE "
                             "hostel/mess/kit/lab charges described in About — mention "
                             "any such extra charges relevant to the colleges you list.\n")
            # Enumeration + structured filters + no course qualifier: the
            # qualifying set IS the context — hand the model the full id list
            # so a partial enumeration is impossible.
            if (route_info.get("question_kind") == "enumerate"
                    and not course_terms and results):
                id_list = ", ".join(sorted(context_ids))
                user_msg += (f"EVERY college in CONTEXT satisfies these filters. If the "
                             f"question asks which colleges fit/qualify, your answer and "
                             f"citations MUST include ALL of: {id_list}\n")
    # Targeted counselling instruction beats a rule buried in the system
    # prompt: when the student only shared their profile, ask before dumping.
    if route_info.get("question_kind") == "profile_share":
        user_msg += ("THE STUDENT HAS NOT ASKED A QUESTION — they only shared facts "
                     "about themselves. Respond like a real counsellor: warmly "
                     "acknowledge what they shared, give ONE encouraging line about "
                     "what is possible (no detailed college list, no fees, and never "
                     "say 'dataset' or 'database'). Speak qualitatively — never exact "
                     "counts ('13 colleges') and never 'all/every college': say "
                     "'most of the colleges we work with across India' when most "
                     "qualify, or 'several of the colleges we work with across "
                     "India' when fewer do — ALWAYS the phrase 'across India', "
                     "never a state name like Uttarakhand. Example "
                     "tone: 'That is a strong score! You are eligible for most of "
                     "the colleges we work with across India. To help me narrow down "
                     "the best options for your future, which field or stream are "
                     "you interested in pursuing?' Then ask ONE specific follow-up "
                     "question — their "
                     "course/stream interest, budget, or preferred location. "
                     "Reply in the SAME language as the student's message "
                     "(English→English, Hindi→Hindi, Hinglish→Hinglish — 'bhaiya "
                     "mere 80% aaye' deserves a Hinglish reply). "
                     "Keep it under 50 words.\n")
    # Note: the router's language guess is deliberately NOT passed here — the
    # generator mirrors the question's own language (rule 9). For Hindi and
    # Hinglish, a code-detected directive is added on top, because live tests
    # showed the model drifting to English on short Hinglish turns.
    lang = _lang_hint(question)
    if lang:
        user_msg += (f"LANGUAGE (detected in code): the student's current message is "
                     f"{lang} — write your ENTIRE reply in the same language and "
                     f"script. Do NOT reply in plain English.\n")
    # An explicit "top 3" beats the completeness rule — resolved in code so the
    # two instructions can never fight each other in the model's head.
    want_n = _requested_count(question)
    if want_n:
        user_msg += (
            f"REQUESTED COUNT: the student asked for exactly {want_n}. Show "
            f"EXACTLY {want_n} colleges — no more, no fewer (unless fewer than "
            f"{want_n} qualify at all, in which case say so). This OVERRIDES the "
            f"completeness rule above. Rank them best-first on the criterion "
            f"that matters for their question (placement, then NAAC, then fit "
            f"to their budget/score) and order the list and any table columns "
            f"in that same ranked order — a 'top {want_n}' presented out of "
            f"order is not a top {want_n}.\n")
    user_msg += f"\nCURRENT QUESTION (answer this, in the conversation's context): {question}"

    # ---- 3. Grounded generation + 4. verification (one corrective retry) ---
    messages = [{"role": "system", "content": config.GEN_SYSTEM_PREFIX + GENERATOR_SYSTEM},
                {"role": "user", "content": user_msg}]
    result, verified = None, False
    for attempt in range(3):
        try:
            raw = chat(messages, model=config.GEN_MODEL, json_mode=True,
                       temperature=0.0, max_tokens=1400, usage_log=calls)
            result = _parse_json(raw)
            result = {
                # strip stray markdown bold — the JSON answer is plain text
                "answer": str(result.get("answer", "")).replace("**", "").strip(),
                "citations": [c for c in (result.get("citations") or []) if isinstance(c, str)],
                "answered": bool(result.get("answered")),
                "reason_if_unanswered": result.get("reason_if_unanswered"),
            }
        except Exception as e:
            print(f"[generation error, attempt {attempt + 1}] {e}", file=sys.stderr)
            messages.append({"role": "user", "content":
                             "Your previous reply was not valid JSON. Return ONLY the JSON object."})
            continue

        ok, problems = _verify(
            result, context_ids,
            allow_uncited=route_info.get("question_kind") == "profile_share")
        rows_ctx = {r["college_id"]: r for r, _ in results}
        # Negation questions ("which colleges do NOT offer X") legitimately
        # cite colleges that lack the program — the exact-match check must
        # stand down there or it would flag every correct citation.
        ignore_money = {raw_budget, (raw_budget or 0) // 2, (raw_budget or 0) * 2} \
            if raw_budget else None
        problems = (problems
                    + ([] if negated else _course_violations(result, question, rows_ctx))
                    + _name_violations(result, all_rows, context_ids)
                    + _voice_violations(result)
                    + _numeric_violations(result, all_rows, ignore_money))
        if not problems:
            verified = True
            break
        print(f"[verify failed, attempt {attempt + 1}] {problems}", file=sys.stderr)
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content":
                         "Your answer failed grounding checks: " + "; ".join(problems)
                         + ". Rewrite the answer from scratch fixing these issues — keep counts "
                           "and wording consistent with the corrected list — and return ONLY "
                           "the corrected JSON. Cite only ids present in CONTEXT."})

    if result is None:  # every attempt failed to produce JSON
        result = {"answer": "Sorry — I hit a temporary issue (usually a busy server). "
                            "Please ask that again in a few seconds.",
                  "citations": [], "answered": False,
                  "reason_if_unanswered": "Transient provider error: no valid output after retries."}
    elif not verified:
        # Fail closed: an unverifiable answer must not be shipped as fact.
        result = {"answer": "I found related records but could not verify a fully grounded "
                            "answer, so I would rather not guess.",
                  "citations": [], "answered": False,
                  "reason_if_unanswered": "Grounding verification failed after retry."}

    if result["answered"]:
        result["reason_if_unanswered"] = None
    result["answer"] = _format_answer(result["answer"])

    _log_metrics({"question": question, "route": route, "calls": calls,
                  "retrieved": sorted(context_ids), "verified": verified,
                  "total_latency_s": round(time.perf_counter() - t_start, 3)})
    if config.CACHE_ENABLED and verified and not history:
        _cache_store(cache_key, result)
    # Only the conversational demo path gets a "profile" key — answer.py
    # never passes history/known_profile, so its stdout contract (exactly
    # answer/citations/answered/reason_if_unanswered) is untouched.
    if history is not None or known_profile is not None:
        result["profile"] = merged_profile
    return result
