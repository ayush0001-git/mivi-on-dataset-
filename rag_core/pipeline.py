"""The answer pipeline: route -> retrieve -> generate (grounded) -> verify -> JSON.

Flow per query:
  1. Router (small model): does this question need retrieval at all? Extract
     hard filters and normalise units (semester/lakh -> Rs per year).
  2. Hybrid retrieval over the production store (retrieve.py): hard filters in
     SQL + semantic/lexical ranking. ~35,300 colleges and 211k course
     offerings, so the retrieval layer returns a top-k of rendered cards and
     NEVER the corpus.
  3. Aggregates come from SQL, not from the model: counts and superlatives are
     computed by retrieve.count_matching()/superlatives() over the whole
     matching set, then handed to the generator as facts. At this scale the
     shown list is a sample of the matching set, never the matching set.
  4. Generation (large model): answer from the retrieved cards ONLY, citing
     college ObjectIds.
  5. Deterministic verification: cited ids must exist in the retrieved context;
     every number in the prose must trace back to a cited card. One corrective
     retry, then fail closed (answered=false) rather than ship an ungrounded
     answer.
"""
import json
import random
import re
import sys
import threading
import time

from . import config, retrieve, selfrag
from .llm import chat

# Citations are Mongo ObjectIds — the same 24-hex id that is the public URL
# (makemyeducation.com/college/<id>), so every citation resolves to a real page.
CID_RE = re.compile(r"\b[0-9a-f]{24}\b", re.I)

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
            self._vec = retrieve.encode_query(question)
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

# ...and a wider ask ("any state", "anywhere in India", "outside Kerala") has
# to release the STATE filter too, which only became a real filter once the
# corpus went national.
_BROADEN_STATE_RE = re.compile(
    r"other state|another state|different state|any state|anywhere in india|"
    r"all over india|across india|outside\s+\w+|pure india|puri india|"
    r"kisi bhi state", re.I)

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
#
# The stems below are spelled `suicid\w*` / `self[- ]?harm\w*` rather than bare
# stems for a reason that cost this guard its most important trigger: the whole
# alternation is followed by `\b`, and a bare `suicid` then requires a word
# boundary between "d" and "e" — which does not exist. "suicide" and "suicidal",
# the two words a student in crisis is most likely to type, silently failed to
# match and were answered as ordinary college questions. Any stem added here
# MUST either end the word or carry its own `\w*`; there is a test for exactly
# this in evals.
_DISTRESS_RE = re.compile(
    # "want to die" excludes "die-hard": erring toward the crisis path is the
    # right default, but "die-hard fan" is common enough in Indian English that
    # the false positive is worth one lookahead.
    r"\b(kill myself|end my life|suicid\w*|self[- ]?harm\w*|want to die(?![- ]?hard)|"
    r"don'?t want to live|no reason to live|give up on life|can'?t go on|"
    r"marna (?:hai|chahta|chahti|h)|mar jau|jeena nahi|jina nahi|"
    r"khatam kar (?:du|dun|lu))\b"
    r"|\b(?:i(?:'?m| am| feel)|feeling|mai bahut|main bahut)\s+\w*\s*"
    r"(?:hopeless|worthless|empty|numb|done with everything)\b", re.I)

ROUTER_SYSTEM = """You are the query router for MakeMyEducation's college-search assistant.
The catalogue holds ~35,300 colleges across EVERY state of India (plus a few
thousand overseas institutions, excluded unless the student asks for abroad)
and ~211,000 course offerings. Per college we know: name, city, district,
state, type (Government/Private/Deemed/Public), year established, NAAC grade,
NIRF ranking, affiliating university, the list of courses it offers with their
program levels and entrance exams, per-year tuition RANGE in rupees across
those courses, hostel availability, scholarships, and free-text about/placement
/facility notes. We do NOT have admission cutoffs, seat counts or placement
salary figures.

Classify the user's message and extract hard filters. Return ONLY JSON:
{
  "route": "data_query" | "smalltalk" | "out_of_scope",
  "question_kind": "enumerate" | "recommend" | "lookup" | "profile_share" | "other",
  "language": "<language of the message, e.g. English, Hindi, Hinglish>",
  "filters": {
    "state": <Indian state as written in full, e.g. "Uttarakhand", or null>,
    "city": <city/town only, e.g. "Dehradun", or null>,
    "college_type": "Government" | "Private" | "Deemed" | "Public" | null,
    "max_tuition_inr": <int, rupees per YEAR, or null>,
    "min_tuition_inr": <int, rupees per YEAR, or null>,
    "course_terms": [<course words mentioned, e.g. "MBA", "B.Tech", "nursing">],
    "program_level": "Undergraduate" | "Postgraduate" | "Diploma/Certificate"
        | "Doctorate" | null,
    "hostel_required": <true or null>,
    "entrance_exam": <exam named by the user, e.g. "NEET", "CUET", or null>,
    "include_abroad": <true ONLY if the user asks about studying abroad /
        names a foreign country or a foreign institution; else false>
  },
  "needs_all_records": <true if answering needs an AGGREGATE over the whole
    matching set rather than a handful of examples: counts ("how many
    colleges..."), superlatives (cheapest, oldest, most courses), or any
    "which/list all colleges that X" enumeration — else false>,
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
- question_kind: "enumerate" = user wants the colleges matching criteria
  ("which colleges...", "list...", "which fit my budget"); "recommend" = user
  wants advice/a pick ("best for me", "suggest"); "lookup" = a fact about one
  named college; "profile_share" = the student states facts about themselves
  (marks, budget, stream, interests) WITHOUT asking a concrete question, OR
  expresses that they're unsure/lost/don't know what to pick (e.g.
  "i got 55% in class 12", "i love computer science", "mujhe MBA karna hai",
  "confused hoon", "pata nahi kya karu", "I don't know what to do") — in all
  these cases the reply is conversational guidance, not a college lookup, so
  no citations are expected; "other" = anything else.
- Location: put a CITY in "city" and a STATE in "state" — never both fields
  from one word. "colleges in Kerala" -> state "Kerala"; "colleges in Kochi"
  -> city "Kochi". If the user names a country other than India, that is
  include_abroad = true, not a state.
- Money: "1 lakh" = 100000, "60 thousand"/"60k" = 60000, "1 crore" = 10000000.
  Fees in the catalogue are tuition PER YEAR. If the user gives a budget per
  semester: annual = semester x 2, and set unit_note explaining the conversion
  (1 academic year = 2 semesters). If the user gives a total-course budget:
  set max_tuition_inr to null and put an explanation in unit_note (course
  lengths vary; cannot hard-filter).
  Examples:
    "budget of Rs 1.5 lakh/year"        -> max_tuition_inr = 150000
    "I can pay 60 thousand per semester" -> max_tuition_inr = 120000,
        unit_note = "60,000/semester = 120,000/year (2 semesters per year)"
    "I have 1 lakh per semester"         -> max_tuition_inr = 200000,
        unit_note = "1 lakh/semester = 2 lakh/year (2 semesters per year)"
- The student's own marks go in profile.marks_pct, NEVER into filters: we hold
  no cutoff data, so marks can never be used as a filter.
- Set a filter ONLY if the user clearly stated it. Never guess.
- If CONVERSATION SO FAR is provided, interpret the CURRENT message in its
  context: a bare number after a marks discussion is their percentage, BUT a
  bare small number (1-15) right after the counsellor presented a NUMBERED
  LIST selects that list item — question_kind "lookup" about that college.
  A course interest stated earlier carries forward into course_terms; a
  message answering the counsellor's follow-up question is a data_query, not
  smalltalk. unit_note is ONLY for money/unit conversions — never put
  commentary or guesses there."""

GENERATOR_SYSTEM = """You are mimi, MakeMyEducation's AI college counsellor, answering from our
verified catalogue of Indian colleges. You will get CONTEXT (college records
selected for this question) and a QUESTION. Answer using ONLY the CONTEXT.
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
1. Every factual claim must come from CONTEXT. The supporting college ids
   (24-character codes like 652f1a9c4d3b2e7a91c0f8d1) go ONLY in the
   "citations" array — NEVER write an id, or a link containing one, inside the
   answer text. In the text, refer to colleges by full name and city. Cite
   only ids that appear in CONTEXT.
2. Never use outside knowledge, never invent a college, fee, course or fact.
3. FEES ARE FOR THE FULL PROGRAMME, NOT PER YEAR. This is the single easiest
   way to mislead a student, so read it carefully. Our fee figures are the
   whole-course cost and EXCLUDE hostel/mess/kit/exam charges. Never write
   "per year", "/year", "annually" or "p.a." next to one. Say "for the full
   course", "total for the programme", or "poore course ka".
   Verified against the state Fee Regulating Authority's approved ANNUAL fees:
   our figure divided by theirs equals the course length (2.0x for a two-year
   MBA, 3.6x for a four-year BTech). So a figure quoted as annual is inflated
   two-to-four-fold, and a student told a college costs Rs 2,39,000 a year when
   the approved annual fee is Rs 67,272 will rule out a college they can
   actually afford. We do NOT hold course durations, so you cannot compute the
   per-year figure — do not try. If the student asks "per year", say we hold the
   total course fee and they should divide by their course length or confirm
   with the college.
   A college usually offers many courses at different prices, so its fee is a
   RANGE ("Rs 45,000-1,20,000 for the full course, depending on the programme")
   — quote the range, or the figure for the specific course asked about, and
   never present the low end as "the fee" of the college.
3a. A "Fee Regulating Authority approved" line in CONTEXT is different: that
   one IS annual, it is the legally approved figure, and it carries a year.
   Quote it WITH its year ("approved fee for 2024-25 was ..."), say it is the
   government-approved figure, and tell the student to confirm the current
   year's fee — fees are revised annually and the published order may be older.
3b. ABROAD: an overseas institution's fees are in ITS OWN currency, never
   rupees. If CONTEXT says a college's tuition is in a local currency, say so
   plainly and NEVER convert it, compare it to a rupee budget, or write "Rs"
   in front of it.
4. "Not recorded" is not zero. A missing fee, NAAC grade or hostel entry means
   we simply do not have it — never describe it as free, zero, worst or a
   weakness. Say what we do have instead.
5. We hold NO admission cutoffs, NO seat counts and NO placement salary
   figures. If a student asks for a cutoff, a seat count or a package/LPA
   number, say plainly that we don't have that figure for the college(s) in
   question — do not guess, do not estimate, do not substitute a national
   average — then offer what IS known (courses, fees, entrance exams accepted,
   NAAC/NIRF, hostel, scholarships) and suggest checking the college's own
   admission page for the cutoff.
6. Units: 1 academic year = 2 semesters. If the user's units differ from the
   data (per semester / total course), convert and STATE the assumption
   explicitly in the answer.
7. A diploma or certificate is not a degree. If a Diploma/Certificate program
   is relevant you may include it, but only with an explicit note that it is a
   diploma, not a degree.
8. Thousands of colleges share similar names across different cities — never
   confuse them. Identify every college by full name AND city. Name ONLY
   colleges that qualify for the user's question: do not bring up a
   non-qualifying college even to disambiguate — precise naming (full name +
   city) IS the disambiguation.
8b. Course questions are EXACT-MATCH questions: include a college only if the
   asked program appears in its "Courses offered" list. Related programs are
   NOT the same program (BBA is not MBA, M.Com is not MBA, PGDM is not MBA —
   if PGDM is present alongside the question's program you may mention it as a
   separate note, clearly labelled). A course list ending in "(+N more)" is
   TRUNCATED: for such a college you may not claim a program is missing — say
   the listing shows more courses than fit here and offer to check that one.
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
   offer an MBA:", "With your budget, these 3 fit:" — then a numbered list,
   one college per line, separated by "\n", like:
   "1. <Name>, <City> — Rs <fee range>/year; <the key facts asked>."
   Single-college details: one lead line, then short "- <label>: <value>"
   lines (courses, tuition/year, hostel, NAAC, entrance exams, established...),
   one per "\n". Never write more than 2 sentences without a line break. End
   with a one-line note (assumptions, extra charges) when needed. Single-fact
   answers may be one sentence. Plain text — no markdown bold/headers.
   ZERO filler: no "I hope this helps", "feel free to ask", "great question"
   or restating the question — every sentence must carry information.
9c. ANY question about 2-4 specific colleges TOGETHER — compare them, "details
   of both", "in dono ke baare me batao" — always uses a compact pipe table,
   one row per line separated by "\n", header first:
   "| Field | <College A> | <College B> |"
   "| Tuition/year | Rs 145,000-190,000 | Rs 118,000 |"
   Rows: City, Type, Courses, Levels, Tuition/year, Hostel, NAAC, NIRF,
   Entrance exams, Scholarships (summarise each college's scholarship note in
   a few words; write "none mentioned" if it has none), Established. Skip a
   row entirely if neither college has that value — never fill it with a guess
   or a dash-with-commentary. Every row starts AND ends with "|". Keep the
   long About text OUT of the table — after it, add at most ONE short line
   per college with its key About point, then ONE line with the key
   trade-off. (This is the only place any markdown-like syntax is allowed.)
10. If CONTEXT cannot support an answer, set "answered": false, briefly say
    what we do and do not have, and never guess. This includes ABSENCE
    questions: if asked whether a college offers a program/facility that its
    record does not list, set "answered": false and say we don't have it
    listed for that college. ALWAYS include one adjacent helpful fact from the
    record (what it does offer / the nearest alternative in CONTEXT, cited)
    so a refusal never feels like a dead end.
10b. Before ever saying a program is "not offered/listed", scan EVERY
    college's courses in CONTEXT — never deny a program that appears there.
    If the student's spelling looks like a typo of a listed program ("bse" ~
    B.Sc, "btech" ~ B.Tech, "b.pharma" ~ B.Pharm), assume the close match,
    answer for it, and note the assumption in one short phrase.
11. SCALE — the single most important honesty rule. We list tens of thousands
    of colleges, so a "which colleges..." question often matches hundreds or
    thousands of them and CONTEXT holds only the top few. NEVER imply the
    colleges you show are all that qualify, and never write "these are the
    only ones" or "there are 5 colleges that...". When a TOTAL MATCHES line is
    given, state that exact total in your lead sentence, then show the top few
    and say plainly that more are available and you can narrow them down (by
    city, budget, course or level). Cite every college you actually name.
    If the student names a number ("top 3", "just 2 options"), give exactly
    that many, ranked best-first — a REQUESTED COUNT line below states it
    explicitly when one was asked for. Only when a line explicitly tells you
    CONTEXT holds ALL matches may you present the list as complete.
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
    specific (a course, fee, hostel, comparison, location list), answer it
    directly and completely — no unnecessary follow-ups.
12b. "All details" / full-profile questions about one college: present EVERY
    field its record actually carries as a structured list — type, city and
    state, established year, affiliating university, NAAC grade, NIRF
    ranking, courses and levels, total course fee, hostel, entrance exams
    accepted, scholarships, placements/facilities notes — plus the key points
    from About. Leave nothing in the record out, and simply omit the fields
    the record does not carry (do not list them as unknown one by one; one
    closing line naming what we don't have is enough).
13. For "best"/recommendation questions: "best" is NOT "cheapest". If COMPUTED
    FACTS contains a "NIRF ranking" block, THAT is the ranking — lead with it,
    name the discipline and year ("#3 in Engineering, NIRF 2024"), and say the
    ranking is the Government of India's, not ours. NIRF is the only external
    ranking available, so use it whenever it is present.
    If there is NO NIRF block, say plainly that we have no ranking for these
    colleges and that you are comparing on what IS known — breadth and level of
    relevant courses, fee fit against the stated budget, location. NEVER claim
    to have weighed "NAAC grade and NIRF ranking" for colleges whose records
    show neither: that invents a basis for the recommendation, which is worse
    than admitting the basis is limited. We hold no placement figures at all.
    Offer 2-3 alternatives, note that "best" depends on the student's
    priorities, and never phrase a partial list as the complete set.

14. TWO TIERS OF FACT. A card may end with a block headed "FROM THE COLLEGE'S
    OWN WEBSITE". Everything ABOVE that heading is MakeMyEducation's verified
    listing; everything under it was read off the college's own site and is NOT
    verified by us. You may absolutely use it — it is often the only place a
    scholarship or hostel charge exists — but you MUST attribute it in the same
    breath, naturally, e.g. "their own website lists a merit scholarship of…"
    or "as per the college's site". Add a short "worth confirming with the
    college directly" once, not after every line. NEVER present a website
    figure as if we had verified it, never merge it into a fee comparison
    table as though it were our tuition data, and never use it to contradict a
    verified figure — if the two disagree, say so plainly and give both.

Citations discipline: cite ONLY the colleges that form part of your actual
answer. For budget/eligibility questions, cite only colleges that satisfy the
stated constraints — never cite a college that fails them (you may say that
stretching the budget would open more options, without naming ids).

Worked examples of the required behaviour (names are generic placeholders):
- QUESTION: "Does Alpha College offer an M.Sc in Data Science?" and Alpha
  College's record lists only "BSc; BCom" ->
  {"answer": "We don't have an M.Sc in Data Science at Alpha College, Cityville
  — what it does offer is a B.Sc and a B.Com.",
  "citations": ["652f1a9c4d3b2e7a91c0f8d1"], "answered": false,
  "reason_if_unanswered": "M.Sc Data Science is not listed for Alpha College."}
- QUESTION: "Which colleges in Cityville offer a B.Sc under 1 lakh?" with a
  TOTAL MATCHES line saying 46 ->
  {"answer": "46 colleges in Cityville offer a B.Sc within that budget — here
  are the strongest few:\n1. Alpha College, Cityville — Rs 50,000-68,000/year;
  NAAC A.\n2. Beta University, Cityville — Rs 72,000/year; hostel available.\n
  I can narrow these down by hostel, college type or course if you tell me
  more.",
  "citations": ["652f1a9c4d3b2e7a91c0f8d1", "652f1a9c4d3b2e7a91c0f8d2"],
  "answered": true, "reason_if_unanswered": null}
- QUESTION: "What was last year's cutoff for Alpha College?" ->
  {"answer": "We don't have cutoff numbers for Alpha College, Cityville — that
  one's best checked on their own admissions page. What I can tell you: it
  takes CUET, tuition runs Rs 50,000-68,000/year, and hostel is available.",
  "citations": ["652f1a9c4d3b2e7a91c0f8d1"], "answered": false,
  "reason_if_unanswered": "Admission cutoffs are not part of the catalogue."}

Return ONLY JSON, exactly this shape:
{"answer": "<string>", "citations": ["<24-char college id>", ...],
 "answered": true|false, "reason_if_unanswered": null | "<string>"}

"answered" is true ONLY when the requested information is explicitly present
in CONTEXT. If the question asks about something the records do not list
(a course, a facility, a cutoff, a package figure), you MUST set
"answered": false — even while the "answer" text helpfully explains what the
records do list."""


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
    """Regex fallback if the router call fails: keep answering, log the incident.

    Deliberately extracts no city/state: there is no gazetteer in this module
    and guessing a place name out of free text would produce a hard SQL filter
    that silently returns nothing. The place name still reaches retrieval
    through the semantic/lexical query itself."""
    q = question.lower()
    filters, terms, note = {}, [], None
    m = re.search(r"(\d+(?:\.\d+)?)\s*(crore|cr\b)", q)
    amount = int(float(m.group(1)) * 10_000_000) if m else None
    if amount is None:
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
            filters["max_tuition_inr"] = amount
        else:
            filters["max_tuition_inr"] = amount
    if "government" in q or "sarkari" in q:
        filters["college_type"] = "Government"
    elif "private" in q or "praivet" in q:
        filters["college_type"] = "Private"
    if "hostel" in q:
        filters["hostel_required"] = True
    if re.search(r"abroad|overseas|foreign|videsh", q):
        filters["include_abroad"] = True
    for t in ("mba", "b.tech", "btech", "engineering", "mbbs", "law", "llb", "pharmacy",
              "nursing", "design", "diploma", "bca", "mca", "bba", "b.sc", "b.com"):
        if t in q:
            terms.append(t)
    if terms:
        filters["course_terms"] = terms
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
        "needs_all_records": aggregate,
        "unit_note": note,
    }


# --------------------------------------------------------------------------
# Code-computed facts handed to the generator.
# --------------------------------------------------------------------------
_SUP_LABELS = {
    # NIRF first: it is the only external, published ranking in the system and
    # therefore the only defensible answer to "which is best". Everything below
    # it is a proxy the student did not ask for.
    "best_ranked": "NIRF ranking (Govt of India, best first)",
    "cheapest": "Lowest total course fee",
    "most_expensive": "Highest total course fee",
    "oldest": "Oldest (earliest established)",
    "newest": "Newest (most recently established)",
    "most_courses": "Widest course range",
}


def _sup_line(row):
    """One superlative row -> one citable fact line, using only the fields the
    aggregate actually carried."""
    if not isinstance(row, dict) or not row.get("college_id"):
        return None
    bits = []
    # The rank leads, because for a NIRF row it IS the fact — everything else on
    # the line is context. Stated with its discipline and year: a bare "#3" gets
    # quoted as an overall national rank when it was third in Pharmacy.
    if row.get("nirf_rank"):
        cat = row.get("nirf_category") or "Overall"
        yr = row.get("nirf_year") or ""
        bits.append(f"NIRF {yr} #{int(row['nirf_rank'])} in {cat}".replace("  ", " "))
    where = ", ".join(str(row[k]) for k in ("city", "state") if row.get(k))
    if where:
        bits.append(where)
    lo, hi = row.get("min_tuition_inr"), row.get("max_tuition_inr")
    if lo and hi and lo != hi:
        # "total", not "/yr" — the corpus figure is whole-programme (verified
        # against the state authority's approved annual fees; see rule 3).
        bits.append(f"Rs {int(lo):,}-{int(hi):,} total")
    elif lo or hi:
        bits.append(f"Rs {int(hi or lo):,} total")
    if row.get("establish_year"):
        bits.append(f"est. {int(row['establish_year'])}")
    if row.get("n_courses"):
        bits.append(f"{int(row['n_courses'])} courses")
    return f"{row['college_id']} {row.get('name', '(unnamed)')} ({'; '.join(bits)})"


def _computed_facts(hits, sup=None, total=None):
    """State SQL-computed aggregates as facts so the model never does the math.

    This used to scan every record. With ~35,300 colleges that is impossible:
    counts and superlatives now come from retrieve.count_matching() /
    retrieve.superlatives(), which run over the WHOLE matching set in SQL, and
    the retrieved hits only supply the shortlist the model may quote from.
    LLMs reliably anchor on 'cheapest' when asked for 'best'; handing them the
    ranked facts removes that failure mode the same way hard filters remove
    budget errors."""
    lines = []
    if total is not None:
        lines.append(f"Total colleges matching the applied filters: {total:,} "
                     f"(CONTEXT shows the top {len(hits)} of them).")
    for key, rows in (sup or {}).items():
        if isinstance(rows, dict):
            rows = [rows]
        if not isinstance(rows, (list, tuple)):
            print(f"[superlatives] ignoring {key!r}: {type(rows).__name__}", file=sys.stderr)
            continue
        rendered = [s for s in (_sup_line(r) for r in rows) if s]
        if rendered:
            lines.append(f"{_SUP_LABELS.get(key, key.replace('_', ' ').capitalize())}: "
                         + "; ".join(rendered))
    if not lines:
        return ""
    return ("COMPUTED FACTS (derived in SQL over the full matching set, not just "
            "CONTEXT — use these for any count/'best'/highest/lowest reasoning, "
            "and cite the ids you name). Every college in CONTEXT ALREADY "
            "satisfies the user's stated budget/filters — never claim any of "
            "them exceeds the budget. A college with no tuition figure has none "
            "recorded; that never means free:\n- " + "\n- ".join(lines) + "\n")


_COLLEGE_TYPES = {"government": "Government", "private": "Private",
                  "deemed": "Deemed", "public": "Public", "govt": "Government",
                  "sarkari": "Government"}
_PROGRAM_LEVELS = {
    "undergraduate": "Undergraduate", "ug": "Undergraduate", "bachelor": "Undergraduate",
    "postgraduate": "Postgraduate", "pg": "Postgraduate", "master": "Postgraduate",
    "doctorate": "Doctorate", "phd": "Doctorate", "doctoral": "Doctorate",
    # the ETL derives this value with a slash (courses_csv.program_level)
    "diploma/certificate": "Diploma/Certificate", "diploma-certificate": "Diploma/Certificate",
    "diploma": "Diploma/Certificate", "certificate": "Diploma/Certificate",
}


def _clean_filters(filters):
    """Coerce router output to the retrieval contract's types; drop anything
    malformed (loudly). Keys must match retrieve.search()'s filter contract
    exactly — a typo here silently widens a query instead of failing."""
    if not isinstance(filters, dict):
        if filters:
            print(f"[filter dropped] non-dict filters: {filters!r}", file=sys.stderr)
        return {}
    out = {}
    for key in ("max_tuition_inr", "min_tuition_inr"):
        v = filters.get(key)
        if v is None:
            continue
        try:
            v = int(float(v))
        except (TypeError, ValueError):
            print(f"[filter dropped] {key}={v!r}", file=sys.stderr)
            continue
        # A rupee-per-year bound under Rs 1,000 can only be a unit slip
        # ("2" meaning 2 lakh). Applying it would return an empty list and
        # tell a student no college fits their budget — the worst possible
        # wrong answer, so it is dropped loudly instead.
        if v < 1000:
            print(f"[filter dropped] implausible {key}={v} (unit error?)", file=sys.stderr)
            continue
        out[key] = v
    if out.get("min_tuition_inr") and out.get("max_tuition_inr") \
            and out["min_tuition_inr"] > out["max_tuition_inr"]:
        print(f"[filter dropped] min_tuition {out['min_tuition_inr']} > max "
              f"{out['max_tuition_inr']}", file=sys.stderr)
        out.pop("min_tuition_inr")

    ctype = filters.get("college_type")
    if isinstance(ctype, str) and ctype.strip().lower() in _COLLEGE_TYPES:
        out["college_type"] = _COLLEGE_TYPES[ctype.strip().lower()]
    elif ctype:
        print(f"[filter dropped] college_type={ctype!r}", file=sys.stderr)

    level = filters.get("program_level")
    if isinstance(level, str) and level.strip().lower() in _PROGRAM_LEVELS:
        out["program_level"] = _PROGRAM_LEVELS[level.strip().lower()]
    elif level:
        print(f"[filter dropped] program_level={level!r}", file=sys.stderr)

    if filters.get("hostel_required") is True:
        out["hostel_required"] = True
    if filters.get("include_abroad") is True:
        out["include_abroad"] = True
    for key in ("state", "city", "entrance_exam"):
        v = filters.get(key)
        if isinstance(v, str) and v.strip():
            out[key] = v.strip()
        elif v:
            print(f"[filter dropped] {key}={v!r}", file=sys.stderr)
    terms = filters.get("course_terms")
    if isinstance(terms, list):
        terms = [t.strip() for t in terms if isinstance(t, str) and t.strip()]
        if terms:
            out["course_terms"] = terms
    elif terms:
        print(f"[filter dropped] course_terms={terms!r}", file=sys.stderr)
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


# render_card writes "Courses offered (12): BTech; MBA; ... (+3 more)."
_COURSE_LINE_RE = re.compile(r"^Courses offered \(\d+\):\s*(.*)$", re.M)


def _card_courses(card):
    """(course tokens, truncated?) from a card's course line.

    `truncated` matters: render_card caps the list at 28 titles, so a college
    whose line ends in "(+N more)" may well offer a program the card does not
    show. Asserting absence against a truncated list would block correct
    answers, which is why every absence check below is gated on it."""
    m = _COURSE_LINE_RE.search(card or "")
    if not m:
        return set(), True  # no course line at all: treat as unknown, not empty
    body = m.group(1)
    truncated = "more)" in body
    body = re.sub(r"\(\+\d+ more\)\.?$", "", body).strip().rstrip(".")
    tokens = set()
    for part in body.split(";"):
        part = part.strip()
        if part:
            tokens.add(part)
            tokens.add(part.split()[0])
    return tokens, truncated


# A "program" the student can be held to exactly: a degree-shaped token, not a
# broad field word. "engineering colleges" is a semantic ask and must never be
# turned into an exact-match assertion; "LLB" or "B.Pharm" must.
_DEGREE_TOKEN_RE = re.compile(
    r"^(?:b|m|d)\.?[a-z]{1,5}$"
    r"|^(?:mbbs|bds|bams|bhms|bpt|bsn|llb|llm|mba|pgdm|mca|bca|bba|bcom|bsc|"
    r"btech|mtech|msc|mcom|gnm|anm|phd|iti|diploma)$", re.I)


def _named_programs(question, course_terms, cards):
    """Degree-shaped programs the question literally names AND that at least
    one retrieved card actually lists.

    Both conditions matter: the router can over-extract ("engineering"), and a
    token no card lists gives the check nothing to verify against."""
    q_ws = _norm_ws(re.sub(r"[.\-]", "", question))
    vocab = set()
    for card in cards:
        toks, _ = _card_courses(card)
        vocab |= {_norm(t) for t in toks}
    named = []
    for t in course_terms or []:
        if not _DEGREE_TOKEN_RE.match(t.strip()) or len(_norm(t)) < 3:
            continue
        if _has_token(q_ws, t) and _norm(t) in vocab:
            named.append(t)
    return named


def _offers(card, token):
    """Card-only check — abbreviations only, so a MISS proves nothing.

    Kept for the cheap positive case ("MBA" really is in the card's list). Any
    caller acting on a NEGATIVE must use _offers_authoritative instead: the
    card holds deduped degree abbreviations, so field words like "nursing" or
    "engineering" are absent from it even when the college plainly offers them.
    """
    toks, _ = _card_courses(card)
    hay = _norm_ws(re.sub(r"[.\-]", "", " ; ".join(toks)))
    return _has_token(hay, token)


def _offers_authoritative(college_ids, token):
    """Set of ids that genuinely offer `token`, asked the same way retrieval
    asked it (title OR full_name in SQL). Falls back to the card check only if
    the store is unreachable, so a transient DB problem degrades to the old
    behaviour rather than erroring a student's answer."""
    try:
        from .retrieve import offers_program

        return offers_program(list(college_ids), token)
    except Exception as exc:  # noqa: BLE001
        print(f"[pipeline] offers_program unavailable ({exc}); card fallback",
              file=sys.stderr)
        return None


def _course_violations(result, question, hits_by_id, course_terms):
    """Code-level exact-match check for course questions — no router verdict
    involved. If the user's question literally names a program, every cited
    college must list it. Colleges whose card course list is truncated are
    skipped: their absence of the program is unproven, and blocking a correct
    answer is worse than missing a rare wrong one."""
    if not result.get("answered"):
        return []
    named = _named_programs(question, course_terms,
                            [h.get("card", "") for h in hits_by_id.values()])
    if not named:
        return []
    problems = []
    for cid in result.get("citations", []):
        h = hits_by_id.get(cid)
        if not h:
            continue
        _, truncated = _card_courses(h.get("card", ""))
        if truncated:
            continue
        if not any(_offers(h.get("card", ""), t) for t in named):
            problems.append(
                f"{h.get('name')} ({h.get('city')}) does not list "
                f"{'/'.join(sorted(named))} among its courses — remove it from the "
                f"answer: a college may only be cited for a course question if it "
                f"offers that course")
    return problems


def _list_items(history):
    """{n: item text} from the counsellor's MOST RECENT numbered list."""
    for msg in reversed(history or []):
        if not (isinstance(msg, dict) and msg.get("role") == "assistant"):
            continue
        items = dict(re.findall(r"(?:^|\n)\s*(\d{1,2})\.\s*([^\n]+)",
                                str(msg.get("content", ""))))
        if items:
            return items
    return {}


def _selected_item_text(question, history):
    """The text of the list item the student just picked ('2', '2 wala...').

    Resolved from the conversation BEFORE retrieval, because the item text
    (name + city) is the only usable retrieval query — embedding the bare
    message "2" against 35k colleges returns noise."""
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
    return _list_items(history).get(str(n))


def _match_hit(text, hits):
    """The retrieved college whose name appears in `text`, longest name first
    so 'X Institute of Technology' beats a bare 'X'."""
    for h in sorted(hits, key=lambda h: -len(h.get("name") or "")):
        if h.get("name") and _norm(h["name"]) in _norm(text):
            return h
    return None


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


# Matches a rupee figure AND the bare upper bound of a range. The prompt asks
# for fee ranges ("Rs 45,000-Rs 1,80,000", and models freely write
# "Rs 45,000-1,80,000"), so a currency-prefixed-only pattern verified the lower
# bound and let the upper one through unchecked — the half of the range a
# budget-conscious student actually acts on.
_MONEY_RE = re.compile(
    r"(?:rs\.?|₹|inr)\s*([\d][\d,]{3,})"
    r"|(?<=[\d])\s*(?:-|–|—|to)\s*(?:rs\.?|₹|inr)?\s*([\d][\d,]{3,})",
    re.I)
# year and course-count are matched ONLY when tied to their label, so a
# career "next 5 years" never gets checked as a stated fact.
_YEAR_RE = re.compile(r"(?:established|founded|est\.)\s*(?:in)?\s*(\d{4})", re.I)
# Comma-aware on purpose: "1,284 courses" must not be read as "284 courses"
# and then flagged as invented.
_NCOURSES_RE = re.compile(
    r"(\d[\d,]{0,7})\s+(?:different\s+)?(?:courses|programmes|programs)\b", re.I)


def _ints(text):
    """Every integer written in a piece of grounded text, comma-insensitive.

    ObjectIds are removed first: their timestamp prefix is often eight digits,
    and letting those into the allowed-number set would quietly widen the
    money check by a handful of arbitrary values."""
    text = CID_RE.sub(" ", text or "")
    return {int(m.replace(",", "")) for m in re.findall(r"\d[\d,]*", text)}


# A ranking claim: "#72 in Engineering", "NIRF 2025 rank 3", "ranked 12th".
_RANK_CLAIM_RE = re.compile(
    r"(?:nirf[^.\n]{0,40}?(?:rank(?:ed|ing)?)?\s*#?\s*(\d{1,4})"
    r"|#\s*(\d{1,4})\s*(?:in|for|among)\b"
    r"|rank(?:ed|ing)?\s*#?\s*(\d{1,4})(?:st|nd|rd|th)?\b)", re.I)


def _rank_violations(result, hits_by_id):
    """A ranking the answer states must appear on a CITED college's card.

    This exists because of a live failure: asked for the best engineering
    colleges, MIVI reported "IIT Bhilai ranked #72 in Engineering by NIRF 2025"
    for a college with zero registry facts and no NIRF line on its card — and
    the year was wrong too, since only 2024 was ingested. The general integer
    check could not catch it: 72 is small enough to occur incidentally in a
    card, so it counted as "seen".

    A ranking is exactly the kind of claim a student acts on and cannot easily
    disprove, so it is verified against the rendered ranking lines specifically
    rather than against any number that happens to appear.
    """
    if not result.get("answered"):
        return []
    answer = result.get("answer", "")
    claims = {int(next(g for g in m.groups() if g))
              for m in _RANK_CLAIM_RE.finditer(answer)}
    if not claims:
        return []

    cited = [hits_by_id[c] for c in result.get("citations", []) if c in hits_by_id]
    allowed: set[int] = set()
    for h in cited:
        for line in str(h.get("card", "")).split("\n"):
            if "NIRF" in line or "rank" in line.lower():
                allowed |= {int(n) for n in re.findall(r"\d{1,4}", line)}

    bad = sorted(claims - allowed)
    if not bad:
        return []
    return [f"the answer states rank(s) {bad} but no cited college's record "
            f"carries that ranking — remove the ranking entirely rather than "
            f"guessing it; we only hold NIRF ranks for a minority of colleges"]


def _numeric_violations(result, hits_by_id, extra_ok=None):
    """Every hard number in the answer that names a college fact — tuition,
    founding year, course count — must trace back to a CITED college's own
    card, checked against the data itself, not the model's word.

    The allowed set is exactly the integers that appear in the cited cards
    (plus code-computed facts we injected, and per-semester conversions of the
    tuition figures). The card IS what the model was shown, so this is the
    tightest possible rule that still lets it quote a hostel fee or a figure
    from the About text. Numbers are exactly where a wrong answer destroys
    trust and where LLMs are least reliable, so a stated figure that matches
    nothing cited is blocked and regenerated.

    Cutoff, seat-count and placement-LPA checks are deliberately absent: this
    corpus has no such columns, so there is nothing to check them against and
    asserting them would reject valid answers. Rule 5 of the generator prompt
    forbids stating those numbers at all.

    RANKINGS are checked separately and strictly (see _rank_violations): they
    DO have a source now, and the general integer test is too weak for them —
    "#72" is a small number that appears incidentally in almost any card, so it
    passed unchecked while being entirely invented."""
    if not result.get("answered"):
        return []
    cited = [hits_by_id[c] for c in result.get("citations", []) if c in hits_by_id]
    if not cited:
        return []
    answer = result.get("answer", "")

    allowed = set(extra_ok or ())
    fees, years, counts = set(), set(), set()
    for h in cited:
        allowed |= _ints(h.get("card", ""))
        for key in ("min_tuition_inr", "max_tuition_inr"):
            v = h.get(key)
            if v:
                fees.add(int(v))
        if h.get("n_courses"):
            counts.add(int(h["n_courses"]))
        m = re.search(r"^Established:\s*(\d{4})", h.get("card", ""), re.M)
        if m:  # establish_year is not in the retrieval hit; the card is
            years.add(int(m.group(1)))
    # per-semester / per-year restatements of a real tuition figure are honest
    allowed |= fees | {f * 2 for f in fees} | {f // 2 for f in fees} | counts | years

    problems = []
    for m in _MONEY_RE.finditer(answer):
        # group 1 = currency-prefixed figure, group 2 = the bare upper bound of
        # a range ("Rs 45,000-1,80,000"). Exactly one of them matches.
        val = int((m.group(1) or m.group(2)).replace(",", ""))
        if val not in allowed:
            problems.append(
                f"answer states Rs {val:,}, which appears nowhere in the cited "
                f"colleges' listings (their tuition figures: "
                f"{sorted(fees) or 'none recorded'}) — correct it or drop it")
    for m in _YEAR_RE.finditer(answer):
        val = int(m.group(1))
        if val not in allowed:
            problems.append(
                f"answer says established {val}, which matches no cited college "
                f"(real: {sorted(years) or 'none recorded'}) — correct it or drop it")
    for m in _NCOURSES_RE.finditer(answer):
        val = int(m.group(1).replace(",", ""))
        if val not in allowed:
            problems.append(
                f"answer states {val} courses, which matches no cited college "
                f"(real: {sorted(counts) or 'none recorded'}) — correct it or drop it")
    return problems


def _name_violations(result, hits):
    """Every retrieved college NAMED in the prose must be backed by a citation.

    Catches the citation-dodge failure mode: the model drops a flagged id from
    `citations` but leaves the college in the text with a disclaimer.

    Scope is the retrieved hits only. The old version scanned the whole corpus
    to also catch names that were never in CONTEXT; at ~35,300 colleges that
    scan is not affordable per answer, and a fabricated college is still caught
    by _verify (its id can't be in CONTEXT) and by _numeric_violations (its
    numbers can't be in a cited card)."""
    if not result.get("answer"):
        return []
    answer_norm = _norm(result["answer"])
    citations = set(result.get("citations", []))
    problems = []
    for h in hits:
        name = h.get("name") or ""
        # short names ("City College") collide with ordinary prose; require a
        # name substantial enough that a match is not a coincidence
        if len(_norm(name)) < 12:
            continue
        if _norm(name) in answer_norm and h["college_id"] not in citations:
            problems.append(
                f"the answer names {name} ({h.get('city')}) but its id is not in "
                f"citations — either cite it (only if it truly qualifies for the "
                f"question) or remove every mention of it from the answer")
    return problems


def _strip_ids(text):
    """Ids belong in the citations array; scrub any stray ObjectIds from prose."""
    text = re.sub(r"\s*[\(\[]\s*(?:id\s*)?[0-9a-f]{24}\s*[\)\]]", "", text, flags=re.I)
    text = re.sub(r"\b[0-9a-f]{24}\b\s*,\s*", "", text, flags=re.I)
    text = re.sub(r",?\s*\b[0-9a-f]{24}\b", "", text, flags=re.I)
    text = re.sub(r"\(\s*,\s*", "(", text)           # leftover "(, ..." artifacts
    text = re.sub(r",\s*\)", ")", text)
    text = re.sub(r"\(\s*\)", "", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def _format_answer(text):
    """Clean prose + guaranteed structure: each numbered item on its own line
    (deterministic — independent of whether the model emitted newlines).

    Splits only when a genuine list exists (both "1." and "2." present), so a
    sentence that merely ends in a small number ("established in 2007. Hostel...")
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
              # so prewarm costs real LLM calls exactly once, ever.
              # Stamped with the corpus build so a rebuild invalidates it —
              # see _cache_load.
            tmp = _CACHE_FILE.with_suffix(".tmp")
            tmp.write_text(
                json.dumps({"__corpus__": _corpus_stamp(), "answers": _CACHE},
                           ensure_ascii=False),
                encoding="utf-8")
            tmp.replace(_CACHE_FILE)
        except Exception as e:
            print(f"[cache] persist failed: {e}", file=sys.stderr)


def _corpus_stamp():
    """Identity of the corpus the cached answers were produced from."""
    try:
        from .store.db import get_conn, get_meta

        return str(get_meta(get_conn(), "built_at", "") or "")
    except Exception:  # noqa: BLE001 - no corpus yet: treat as unknown
        return ""


def _cache_load():
    """Load persisted answers, but ONLY if they came from the corpus we are
    about to serve.

    A cached answer is a frozen set of facts — fees, counts, the colleges it
    names. The ETL republishes the whole database, so an answer cached before a
    rebuild can cite a college that no longer exists or quote a fee that has
    since changed, and it would be served with full confidence and no
    retrieval. Keying the file on the corpus build stamp makes a stale cache
    impossible rather than merely unlikely.
    """
    if not config.CACHE_ENABLED:
        return
    try:
        if not _CACHE_FILE.exists():
            return
        data = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return
        stamp = _corpus_stamp()
        if data.get("__corpus__") != stamp:
            print("[cache] discarding on-disk answers: built from a different "
                  "corpus than the one being served", file=sys.stderr)
            return
        answers = data.get("answers")
        if isinstance(answers, dict):
            _CACHE.update(answers)
            print(f"[cache] loaded {len(answers)} answers from disk", file=sys.stderr)
    except Exception as e:
        print(f"[cache] load failed: {e}", file=sys.stderr)


_cache_load()


def _to_context(hits):
    """CONTEXT is exactly the cards retrieval returned — render_card is the
    single definition of what the model may see about a college, and the
    verifier checks the answer's numbers against these same strings.

    Budgeted: cards now carry web-harvested facts and long overviews, so an
    unbounded join grows with corpus richness rather than with top_k. Trimming
    drops whole lowest-ranked cards rather than truncating one mid-sentence —
    a half-card would let the model read a fee that belongs to another college.
    """
    budget = getattr(config, "MAX_CONTEXT_CHARS", 12_000)
    out, used = [], 0
    for h in hits:
        card = h.get("card")
        if not card:
            continue
        if out and used + len(card) + 2 > budget:
            break
        out.append(card)
        used += len(card) + 2
    return "\n\n".join(out)


def _search(question, filters, top_k, query_vec=None):
    """retrieve.search with the store's failure surfaced, not swallowed."""
    return retrieve.search(question, filters=filters or None, top_k=top_k,
                           query_vec=query_vec)


def _count(filters):
    """Exact total over the whole matching set. Never fatal: an aggregate we
    cannot compute costs a weaker prompt, not an unanswered student."""
    try:
        n = retrieve.count_matching(filters or None)
        return int(n) if n is not None else None
    except Exception as e:
        print(f"[count_matching failed] {e}", file=sys.stderr)
        return None


def _superlatives(filters, limit=5):
    try:
        sup = retrieve.superlatives(filters or None, limit=limit)
        return sup if isinstance(sup, dict) else {}
    except Exception as e:
        print(f"[superlatives failed] {e}", file=sys.stderr)
        return {}


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
                            "right college — courses, fees, hostels, entrance exams, "
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
            max_tokens=400, usage_log=calls,
        )
        route_info = _parse_json(raw)
        if not isinstance(route_info, dict):
            raise ValueError(f"router returned {type(route_info).__name__}, not an object")
    except Exception as e:  # router must never kill the pipeline
        print(f"[router fallback] {e}", file=sys.stderr)
        route_info = _heuristic_route(question)

    route = route_info.get("route", "data_query")
    filters = _clean_filters(route_info.get("filters"))
    course_terms = filters.get("course_terms", [])
    needs_all = bool(route_info.get("needs_all_records"))
    question_kind = route_info.get("question_kind")
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

    raw_budget = filters.get("max_tuition_inr")  # for the numeric guardrail's
                                                 # budget exclusion (below)
    # Lookup questions name a specific college ("what does X cost?") — hard
    # filters would remove the very college they asked about, making an honest
    # "that one is above your budget" impossible. Abroad is force-INCLUDED
    # here: a student naming a foreign institution must not be told we don't
    # have it just because the default view is domestic.
    if question_kind == "lookup":
        filters = {"include_abroad": True}

    # "Any other college nearby?" / "something else?" means the student wants
    # to look BEYOND the current city — so the city filter must come off, not
    # stay on and produce "that's the only one here" three turns in a row.
    if _BROADEN_RE.search(question) and filters.get("city"):
        print(f"[broaden] dropping city filter {filters['city']!r}", file=sys.stderr)
        filters.pop("city", None)
    if _BROADEN_STATE_RE.search(question) and filters.get("state"):
        print(f"[broaden] dropping state filter {filters['state']!r}", file=sys.stderr)
        filters.pop("state", None)

    # Deterministic guard: marks/budget statements are counselling openers,
    # never smalltalk — small routers occasionally misfile them.
    if route == "smalltalk" and re.search(r"\d+\s*%|\blakh\b|budget|marks|score|percent", question, re.I):
        route = "data_query"
        question_kind = "profile_share"

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
            else "Out of scope: I only answer questions about colleges and admissions.",
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
    default_k = int(getattr(config, "TOP_K", 12) or 12)
    # Aggregate questions get a wider shortlist, NOT the corpus: the count and
    # the superlatives come from SQL below, and the extra cards only give the
    # shown list room to be ranked and trimmed.
    top_k = min(default_k * 2, 24) if needs_all else default_k

    # A bare "2" or "compare these two" carries no retrievable signal against
    # 35k colleges — resolve the referent from the counsellor's last numbered
    # list FIRST and retrieve on that text instead.
    picked_text = _selected_item_text(question, history)
    compare_texts = []
    if not picked_text and history and re.search(
            r"compare|tulna|difference|versus|\bvs\b|फ़?र्क|अंतर", question, re.I):
        compare_texts = list(_list_items(history).values())[:4]
    search_query = question
    if picked_text:
        search_query = picked_text
    elif compare_texts:
        search_query = " | ".join(compare_texts)

    qv = prefetch.result()  # computed concurrently with the router call above
    if search_query is not question:
        qv = None  # the prefetched vector embeds the wrong text; re-encode
    hits = _search(search_query, filters, top_k, query_vec=qv)

    # ---- Self-RAG: repair retrieval BEFORE falling back to blunt relaxation --
    # Ordering is the point. The relaxation below rescues a dead end by DROPPING
    # the student's constraints, which answers a different question than the one
    # asked. A targeted repair — "Vizag" is spelled "Visakhapatnam", the level
    # filter was never stated — answers the SAME question correctly, so it has
    # to get first refusal. Gated on cheap code signals (see selfrag.needs_repair)
    # so an ordinary query never pays for the extra model call.
    repair_note = ""
    try:
        signal = selfrag.needs_repair(hits, filters, len(hits), question_kind)
        if signal:
            verdict = selfrag.grade(question, filters, _count(filters), signal, calls)
            repaired = selfrag.apply_repair(filters, verdict)
            if repaired:
                new_filters, what = repaired
                new_hits = _search(search_query, new_filters, top_k, query_vec=qv)
                # Only adopt a repair that actually helped. A relaxation that
                # returns the same nothing has cost latency and changed the
                # question for no gain, so it is discarded.
                if len(new_hits) > len(hits):
                    print(f"[selfrag] {signal}: {what} "
                          f"({len(hits)} -> {len(new_hits)} hits)", file=sys.stderr)
                    hits, filters = new_hits, new_filters
                    repair_note = selfrag.context_note(what, verdict)
    except Exception as exc:  # noqa: BLE001 - never let the critic break an answer
        print(f"[selfrag] skipped: {exc}", file=sys.stderr)

    relaxed_note = repair_note
    stated = {k: v for k, v in filters.items() if v and k != "include_abroad"}
    if not hits and stated:
        # Zero matches must not dead-end the student. Relax in two steps so the
        # subject of the question survives as long as possible: keep the course
        # /level the student cares about, drop the money and geography first.
        keep = {k: v for k, v in filters.items()
                if k in ("course_terms", "program_level", "include_abroad")}
        hits = _search(search_query, keep, top_k, query_vec=qv)
        if not hits:
            hits = _search(search_query, None, top_k, query_vec=qv)
        relaxed_note += (
            f"ZERO MATCHES: no college in the catalogue satisfies the stated "
            f"constraints {json.dumps(stated)}. Counsel honestly, never dead-end: "
            f"say plainly that nothing matched these exact constraints, then "
            f"present the 2-3 NEAREST options below as realistic alternatives "
            f"with their real numbers, name which constraint each one misses, "
            f"and add one encouraging line about the path forward. NEVER present "
            f"any college as currently meeting the stated constraints.\n")
        filters = {k: v for k, v in filters.items() if k == "include_abroad"}

    # A named program must never be DENIED just because the student's own
    # budget/location filters removed every college that offers it. Live-tested
    # failure on the old corpus: "kya main LLB kar sakta hoon?" + a budget
    # filter dropped every law college, and the model honestly-but-wrongly said
    # no college offers LLB. At this scale the check is a SQL count, not a scan:
    # ask how many colleges offer the program with the money/geography removed.
    stated_now = {k: v for k, v in filters.items() if v and k != "include_abroad"}
    if hits and course_terms and stated_now:
        # The "does any retrieved college offer this?" test MUST be asked the
        # way retrieval asked it (title OR full_name in SQL). Reading the card
        # instead is what made this branch misfire on every field-word term:
        # cards list abbreviations ("BSc; MD; MS"), so "nursing" looked absent
        # from colleges that plainly offer a Bachelor of Science in Nursing,
        # and the student's own state/budget filters were then thrown away and
        # replaced with a nationwide search plus a false "none of them
        # satisfied your constraints" note.
        hit_ids = [h["college_id"] for h in hits]
        missing = []
        for t in course_terms:
            offering = _offers_authoritative(hit_ids, t)
            if offering is None:  # store unreachable: fall back to the card
                offering = {h["college_id"] for h in hits
                            if _offers(h.get("card", ""), t)}
            if not offering:
                missing.append(t)
        if missing:
            course_only = {"course_terms": missing}
            if filters.get("include_abroad"):
                course_only["include_abroad"] = True
            n_offering = _count(course_only)
            if n_offering:
                hits = _search(" ".join(missing) + " " + question, course_only,
                               top_k, query_vec=None)
                relaxed_note += (
                    f"PROGRAM vs FILTER CONFLICT (counted in SQL): "
                    f"{n_offering:,} colleges DO offer {', '.join(missing)}, but none "
                    f"of them satisfied the student's other constraints "
                    f"{json.dumps(stated_now)}. CONTEXT now holds colleges that offer "
                    f"the program WITHOUT those constraints. Answer honestly: NEVER "
                    f"say the program is not offered. Give the real total, name a few "
                    f"of these colleges with their real numbers, state plainly which "
                    f"constraint they miss (give both numbers), and add one "
                    f"encouraging line about the path forward.\n")
                filters = course_only

    context_ids = {h["college_id"] for h in hits}
    hits_by_id = {h["college_id"]: h for h in hits}
    context = _to_context(hits) if hits else "(no colleges matched the hard filters)"

    # Every code-computed block is also collected here: the numeric verifier
    # must accept figures the model correctly repeats back from OUR arithmetic
    # (totals, superlative fees) as well as from the cited cards.
    code_notes = [relaxed_note]

    def note(text):
        code_notes.append(text)
        return text

    user_msg = hist_text + f"CONTEXT:\n{context}\n\n"
    if relaxed_note:
        user_msg += relaxed_note

    # ---- 2c. SQL aggregates: counts and superlatives ------------------------
    total = None
    if hits and (needs_all or question_kind in ("enumerate", "recommend")):
        total = _count(filters)
        # Superlatives are computed for recommendations too, not just
        # needs_all: "which is the best engineering college" is a recommend,
        # and it is exactly the question that needs the ranking.
        sup = _superlatives(filters, limit=5) if (
            needs_all or question_kind == "recommend") else {}

        # A superlative names colleges drawn from the WHOLE matching population,
        # which retrieval's top-k usually does not contain. Printing their ids in
        # COMPUTED FACTS while they are absent from CONTEXT is a trap: the model
        # is told to cite what it names, the verifier rejects ids that were never
        # retrieved, and the answer degrades to whatever happened to be in the
        # top-k instead. Observed live — "best engineering colleges in India"
        # returned five arbitrary private colleges while IIT Madras sat in the
        # NIRF block unciteable.
        #
        # So the superlative colleges are RETRIEVED and appended to the context
        # they are quoted in. Capped, and existing hits keep their order — this
        # adds the answer to the question actually asked, it does not reorder
        # everything else.
        extra_ids: list[str] = []
        for rows in sup.values():
            for row in (rows or []):
                cid = row.get("college_id")
                if cid and cid not in hits_by_id and cid not in extra_ids:
                    extra_ids.append(cid)
        if extra_ids:
            try:
                added = retrieve.hydrate_ids(extra_ids[:8])
                for h in added:
                    hits.append(h)
                    hits_by_id[h["college_id"]] = h
                    context_ids.add(h["college_id"])
                if added:
                    context = _to_context(hits)
                    user_msg = hist_text + f"CONTEXT:\n{context}\n\n"
                    if relaxed_note:
                        user_msg += relaxed_note
                    print(f"[pipeline] added {len(added)} superlative colleges "
                          f"to CONTEXT so they can be cited", file=sys.stderr)
            except Exception as exc:  # noqa: BLE001
                print(f"[pipeline] could not hydrate superlatives: {exc}",
                      file=sys.stderr)

        facts = _computed_facts(hits, sup, total)
        if facts:
            user_msg += note(facts)

    want_n = _requested_count(question)
    if total is not None and hits:
        if total > len(hits):
            show_n = want_n or min(len(hits), 10)
            user_msg += note(
                f"TOTAL MATCHES: {total:,} colleges match the applied filters; "
                f"CONTEXT holds only the top {len(hits)}. Lead with the exact total "
                f"({total:,}), then show at most {show_n} of them and say more are "
                f"available and you can narrow them down. It is FORBIDDEN to imply "
                f"the shown colleges are the only ones that qualify.\n")
        else:
            user_msg += note(
                f"TOTAL MATCHES: {total:,} — CONTEXT holds ALL of them, so an "
                f"enumeration here can and must be complete: name every qualifying "
                f"college and cite every one you name.\n")

    negated = re.search(r"\b(not|no|without|don'?t|doesn'?t|nahi|nahin|नहीं|बिना)\b",
                        question, re.I)
    if negated and hits:
        # "Which colleges do NOT offer X" has thousands of true answers here,
        # so the old code-computed complement set is meaningless. The honest
        # move is to scope the answer to the shortlist and say so.
        user_msg += note(
            "NEGATION QUESTION: a complete 'colleges that do NOT have X' list would "
            "run to thousands of colleges, so do not attempt one. Answer for the "
            "specific colleges the student asked about, or say plainly which of the "
            "colleges shown here lack it — and say that is a shortlist, not the "
            "full set.\n")

    if merged_profile.get("marks_pct") and hits:
        # The corpus has no cutoff column, so eligibility CANNOT be computed.
        # Saying so explicitly is what stops the model inventing a verdict.
        user_msg += note(
            f"MARKS NOTE: the student scored {merged_profile['marks_pct']}%. We hold "
            f"no admission cutoffs, so you must NOT say whether they qualify for any "
            f"college. Acknowledge the score warmly, mention entrance exams a college "
            f"accepts when that is listed, and point them to the college's own "
            f"admission page for cutoffs.\n")

    if picked_text:
        picked = _match_hit(picked_text, hits)
        if picked:
            user_msg += note(
                f"LIST SELECTION (resolved in code from your most recent numbered "
                f"list): the student's '{question.strip()[:30]}' means "
                f"{picked['name']}, {picked.get('city')}. Reply with the FULL profile "
                f"of this college only — every field as short '- label: value' lines. "
                f"Do not repeat the list.\n")
    elif compare_texts:
        listed = [h for h in (_match_hit(t, hits) for t in compare_texts) if h]
        if len(listed) >= 2:
            names = "; ".join(f"{h['name']}, {h.get('city')}" for h in listed[:4])
            user_msg += note(
                f"LIST REFERENCE (resolved in code): the student is referring to your "
                f"most recent list — compare EXACTLY these colleges and NO others: "
                f"{names}.\n")

    if filters.get("include_abroad"):
        user_msg += note(
            "ABROAD IN VIEW: some colleges here are overseas. Their tuition is in the "
            "institution's own currency — never write it as rupees, never convert it, "
            "and never compare it to an Indian budget. Say which country it is in.\n")

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
        user_msg += note(f"UNIT ASSUMPTION (weave it naturally into the answer as one "
                         f"phrase — NEVER print a label like 'unit note'): {unit_note}\n")
    applied = {k: v for k, v in filters.items() if v}
    if applied:
        user_msg += (f"HARD FILTERS ALREADY APPLIED to CONTEXT: {json.dumps(applied)} "
                     f"(colleges failing them were removed before you saw the data). "
                     f"CONTEXT IS THEREFORE A FILTERED VIEW, NOT THE WHOLE LIST — "
                     f"never say 'we don't have any X' or 'X isn't offered anywhere' "
                     f"based on it. The honest phrasing is 'no college matching your "
                     f"budget/location offers X', which is a completely different "
                     f"statement, and never contradict something you already told "
                     f"the student earlier in this conversation.\n")
        if "max_tuition_inr" in applied:
            user_msg += ("REMINDER for budget answers: tuition figures EXCLUDE "
                         "hostel/mess/kit/exam charges — mention any such extra "
                         "charges the listings actually describe.\n")
    # Targeted counselling instruction beats a rule buried in the system
    # prompt: when the student only shared their profile, ask before dumping.
    if question_kind == "profile_share":
        user_msg += ("THE STUDENT HAS NOT ASKED A QUESTION — they only shared facts "
                     "about themselves. Respond like a real counsellor: warmly "
                     "acknowledge what they shared, give ONE encouraging line about "
                     "what is possible (no detailed college list, no fees, and never "
                     "say 'dataset' or 'database'). Speak qualitatively — never exact "
                     "counts ('13 colleges') and never 'all/every college': say "
                     "'most of the colleges we work with across India' when most "
                     "qualify, or 'several of the colleges we work with across "
                     "India' when fewer do — ALWAYS the phrase 'across India', "
                     "never a single state name. Example "
                     "tone: 'That is a strong score! There is a lot open to you "
                     "among the colleges we work with across India. To help me "
                     "narrow down the best options for your future, which field or "
                     "stream are you interested in pursuing?' Then ask ONE specific "
                     "follow-up question — their "
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
    # An explicit "top 3" beats the scale rule's default — resolved in code so
    # the two instructions can never fight each other in the model's head.
    if want_n:
        user_msg += (
            f"REQUESTED COUNT: the student asked for exactly {want_n}. Show "
            f"EXACTLY {want_n} colleges — no more, no fewer (unless fewer than "
            f"{want_n} qualify at all, in which case say so). Rank them best-first "
            f"on the criterion that matters for their question (NAAC/NIRF standing, "
            f"course fit, then fit to their budget) and order the list and any table "
            f"columns in that same ranked order — a 'top {want_n}' presented out of "
            f"order is not a top {want_n}. Still state the true total when one is "
            f"given above.\n")
    user_msg += f"\nCURRENT QUESTION (answer this, in the conversation's context): {question}"

    # ---- 3. Grounded generation + 4. verification (one corrective retry) ---
    messages = [{"role": "system", "content": config.GEN_SYSTEM_PREFIX + GENERATOR_SYSTEM},
                {"role": "user", "content": user_msg}]
    # Figures we computed ourselves are legitimate for the model to repeat.
    extra_ok = _ints("".join(code_notes))
    if raw_budget:
        extra_ok |= {raw_budget, raw_budget // 2, raw_budget * 2}
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
            result, context_ids, allow_uncited=question_kind == "profile_share")
        # Negation questions ("which colleges do NOT offer X") legitimately
        # cite colleges that lack the program — the exact-match check must
        # stand down there or it would flag every correct citation.
        problems = (problems
                    + ([] if negated else
                       _course_violations(result, question, hits_by_id, course_terms))
                    + _name_violations(result, hits)
                    + _voice_violations(result)
                    + _rank_violations(result, hits_by_id)
                    + _numeric_violations(result, hits_by_id, extra_ok))
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
        result = {"answer": "I found related colleges but could not verify a fully grounded "
                            "answer, so I would rather not guess.",
                  "citations": [], "answered": False,
                  "reason_if_unanswered": "Grounding verification failed after retry."}

    if result["answered"]:
        result["reason_if_unanswered"] = None
    result["answer"] = _format_answer(result["answer"])

    _log_metrics({"question": question, "route": route, "calls": calls,
                  "retrieved": sorted(context_ids), "total_matching": total,
                  "verified": verified,
                  "total_latency_s": round(time.perf_counter() - t_start, 3)})
    if config.CACHE_ENABLED and verified and not history:
        _cache_store(cache_key, result)
    # Only the conversational demo path gets a "profile" key — answer.py
    # never passes history/known_profile, so its stdout contract (exactly
    # answer/citations/answered/reason_if_unanswered) is untouched.
    if history is not None or known_profile is not None:
        result["profile"] = merged_profile
    return result
