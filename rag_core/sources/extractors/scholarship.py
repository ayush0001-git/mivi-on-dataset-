"""Deterministic scholarship extractor: scheme names, eligibility, benefits.

WHY THIS EXISTS
The sweep's LLM extractor is capped at ~14 calls/minute, which is the only
thing bounding throughput across 10,175 queued colleges. Scholarship pages are
the highest-priority target in PAGE_TARGETS, so every page parsed here is one
LLM slot freed for a page that genuinely needs judgement.

WHAT THE INPUT ACTUALLY LOOKS LIKE
web_enrich.parse_html() emits visible text with tags stripped and newlines
preserved. Three properties of that output drive every design choice below,
and all three were measured on real pages (tiut.ac.in, chitkara.edu.in,
christuniversity.in, lpu.in, bncollegepatna.com):

1. TABLE STRUCTURE IS DESTROYED. `<td>` emits no newline, but the source-code
   newlines between cells survive, so a row like
       | Marks >= 95% | 100% on tuition fees |
   arrives as two separate lines - and on tiut.ac.in the whole marks COLUMN
   arrives before the whole waiver COLUMN. Pairing an eligibility band to a
   benefit across lines would therefore be inventing a fact. This module
   refuses to do it: it reports the criteria the page states and the benefits
   the page states, as two separate verbatim lists, and never claims which
   maps to which. A student reading "criteria: 95%+, 90-95%, 85-90%" and
   "waivers: 100%, 50%, 25%" loses nothing real; a student told the wrong
   pairing loses money.

2. THE PAGE IS MOSTLY NAVIGATION. On chitkara.edu.in the actual scholarship
   content is 7 lines out of 568 (and sits at ~char 17,000, past the 6,000-char
   budget the LLM fallback would ever see - so this extractor is not merely
   cheaper here, it is the only thing that finds that content at all).

3. NAV IS RENDERED TWICE. Desktop and mobile menus both land in the text, so
   nav labels appear as EXACT DUPLICATE short lines. That single observation is
   the strongest calibration lever in this file: bncollegepatna.com's fetched
   page contains the word "Scholarships" exactly twice, both nav, and dropping
   duplicated short lines takes it to zero topical evidence -> None.

WHAT IT DELIBERATELY REFUSES
- Cross-line pairing of a criterion to a benefit (see 1 above).
- Any figure without an explicit currency/percent marker in benefit context,
  which is what keeps phone numbers, PIN codes, years, Act numbers, visitor
  counters and room numbers out.
- Institutional totals ("allocates Rs 450 Lakhs annually") and disbursement
  statistics ("Rs 100 Cr disbursed last year") - true, but not money any
  individual student can claim, and a card that shows them reads as a promise.
- Self-declared rankings and superlatives (house rule 5).
- Anything at all below 0.5 confidence: it returns None so the LLM gets the
  page, rather than emitting a weak fact the caller would discard anyway.
"""
from __future__ import annotations

import re

__all__ = ["extract"]

# Bound the work on pathological pages. Real scholarship pages measured
# 9k-19k chars; 300k is ~15x the largest and still parses in single-digit ms.
_MAX_TEXT = 300_000
_MAX_LINES = 6_000

# Anchor words that can head a scheme name. The misspellings are not academic:
# bncollegepatna.com serves its page at /schorership.php, and "scholorship" is
# common enough on college CMSes to be worth three characters here.
_ANCHORS = {
    "scholarship", "scholarships", "freeship", "freeships", "fellowship",
    "fellowships", "bursary", "stipend", "anudaan", "chhatravritti",
    "chhatravriti", "chattarvriti", "chatravriti", "chhatravrati",
    "scholorship", "schorership", "scholarhip", "scholership",
}

_SCHOL_RE = re.compile(
    r"(?i)\b(?:scholarship|scholarships|scholorship|schorership|scholership"
    r"|freeship|free\s?ship|fee\s?waiver|fee\s?concession|tuition\s?waiver"
    r"|financial\s?aid|financial\s?assistance|bursary|stipend|anudaan"
    r"|chhatravritti|chhatravriti|chattarvriti|chatravriti)\b"
)

# A benefit is money/percentage the student RECEIVES. Requiring one of these on
# the line is what separates "50% concession" from "42%" (an LPU statistic
# sitting alone on its own line) and from "total fee is above Rs.10000/-".
_BENEFIT_CUE = re.compile(
    r"(?i)(waiv|concession|consession|rebate|remission|freeship|free\s?ship"
    r"|refund|reduction|discount|scholarship\s+(?:of|worth|amount|upto|up\s+to)"
    r"|on\s+tuition|of\s+tuition|tuition\s+fee|on\s+(?:the\s+)?fees?\b"
    r"|of\s+(?:the\s+)?(?:total\s+)?fees?\b|fee\s+waiver|stipend"
    r"|awarded|granted\s+to|assistance\s+of|scholarship\s+is\s+given)"
)

# Present -> the number is something the student must ACHIEVE, not receive.
_MERIT_CUE = re.compile(
    r"(?i)\b(marks?|score|scores|scoring|scored|aggregate|percentile|cgpa|gpa"
    r"|qualifying\s+exam(?:ination)?|rank)\b"
)

# Present -> the amount is an institutional total, a statistic, or a charge.
# Every one of these was observed on a real page masquerading as a benefit.
_DISQUALIFY = re.compile(
    r"(?i)(allocat|disburs|corpus|budget|total\s+amount|turnover|last\s+year"
    r"|instal?lment|penalt|\bfine\b|late\s+fee|application\s+fee|exam(?:ination)?\s+fee"
    r"|admission\s+fee|caution\s+money|security\s+deposit|refundable\s+deposit"
    r"|per\s+annum\s+income\s+of\s+the\s+institute)"
)

# House rule 5: no marketing, no self-declared rank.
_MARKETING = re.compile(
    r"(?i)\b(best|top|no\.?\s?1|#1|number\s+one|leading|premier|finest|largest"
    r"|award[- ]winning|ranked|world\s?class|unlock|dream)\b"
)

# UI chrome that looks like a title but is a button or a widget label.
_UI_WORDS = {
    "your", "you", "my", "me", "here", "now", "us", "we", "our", "predict",
    "check", "know", "click", "view", "apply", "download", "read", "see",
    "find", "get", "calculate", "explore", "enquire", "contact", "visit",
    "submit", "search", "login", "register", "back", "next", "more",
    "this", "that", "these", "those", "note", "notes",
}

# A name containing any of these is a SENTENCE, not a name. Without this,
# tiut.ac.in yields "This Scholarship is directly granted by the Hon'ble
# Chancellor" as a scheme - true prose, useless as a label.
_CLAUSE_WORDS = {
    "is", "are", "was", "were", "be", "been", "has", "have", "had", "will",
    "shall", "may", "can", "must", "should", "would", "requires", "require",
    "provides", "provide", "offers", "offer", "given", "granted", "awarded",
    "consists", "includes", "include", "covers", "applies",
}
# NOT in _CLAUSE_WORDS: "means". It is a verb, but "Merit cum Means" is one of
# the most common scheme-name idioms in India and blocking it lost
# "Chancellor Merit cum Means Scholarship" from tiut.ac.in.

# A generic descriptor immediately after the anchor makes the line a section
# heading or a table column header ("Scholarship Criteria (after Diploma...)").
_HEADING_TAIL = {
    "criteria", "criterion", "eligibility", "details", "detail", "information",
    "brackets", "bracket", "categories", "category", "phases", "phase",
    "amount", "amounts", "benefit", "benefits", "policy", "rules",
    "guidelines", "status", "faq", "faqs", "overview", "notice", "notices",
    "corner", "committee", "opportunities", "provisions", "highlights",
}

# An administrative unit is not a scheme. christuniversity.in otherwise yields
# "Office of Awards and Scholarships" alongside its actual scholarships.
_ADMIN_HEAD = {
    "office", "department", "cell", "committee", "section", "directorate",
    "bureau", "centre", "center", "desk", "wing", "board",
}

# A word of 3+ characters followed by ". " means a sentence boundary was
# swallowed ("scholarship for MBA. Eligibility"). Two-letter abbreviations
# ("Dr.", "St.", "No.") stay legal because real scheme names start with them.
_SENT_INSIDE = re.compile(r"\w{3,}\.\s")

# Tokens that carry no identifying information. A candidate name made only of
# these is a section heading ("Scholarship Schemes"), not a scheme.
_GENERIC = {
    "scholarship", "scholarships", "scholorship", "schorership", "scheme",
    "schemes", "programme", "programmes", "program", "programs", "details",
    "detail", "information", "info", "category", "categories", "brackets",
    "bracket", "phases", "phase", "policy", "our", "the", "types", "type",
    "list", "lists", "page", "home", "form", "forms", "online", "and", "for",
    "of", "cell", "office", "corner", "section", "other", "others", "all",
    "available", "related", "students", "student", "financial", "aid", "fund",
    "funds", "a", "an", "in", "at", "on", "to", "by", "is", "are", "be",
    "criteria", "criterion", "eligibility", "note", "notes", "terms",
    "conditions", "overview", "about", "amount", "amounts", "benefit",
    "benefits", "rules", "guidelines", "new", "latest",
}

_CONNECTORS = {"of", "for", "and", "cum", "&", "the", "de", "with"}

# Words that may follow the anchor and still belong to the name.
_TAIL_WORDS = {
    "scheme", "schemes", "programme", "programmes", "program", "programs",
    "yojna", "yojana", "fund", "portal", "scholarship", "scholarships",
    "awards", "award",
}

# "Merit cum Means" is deliberately ABSENT: it names a real central scheme, but
# colleges use the phrase for their own awards too ("Chancellor Merit cum Means
# Scholarship" at tiut.ac.in), and the genuine central one is always listed
# under a ministry heading that _GOV_SECTION already catches.
_GOV_NAME = re.compile(
    r"(?i)(post[- ]?matric|pre[- ]?matric|central\s+sector|centrally\s+sponsored"
    r"|national\s+scholarship|prime\s+minister|ministry\s+of|department\s+of"
    r"|state\s+government|govt\.?\s+of|government\s+of|\bugc\b|\baicte\b"
    r"|ishan\s+uday|pragati|saksham|ambedkar|vidya\s?la[kx]shmi|vidyalaxmi"
    r"|kalpana\s+chaw[la]+|indira\s+gandhi|\birdp\b|\bnsp\b"
    r"|national\s+means|\bcapf\b|assam\s+rifles|\brpf\b|beedi|minorit)"
)

# A HEADING that opens a government-scheme section. Once one of these is seen
# the rest of the page is government listings - on tiut.ac.in sections A/B/C
# ("Centrally Sponsored", "UGC Sponsored", "State Sponsored") run to the footer.
# The "offered by" branch demands a government-ish provider: lpu.in has
# "Scholarship offered by Corporate /Industry /NGO /Other Agencies", and a
# looser pattern flipped the flag there and relabelled LPU's own schemes as
# government for the whole rest of the page.
_GOV_SECTION = re.compile(
    r"(?i)(centrally\s+sponsored|state\s+sponsored|ugc\s+sponsored"
    r"|central\s+and\s+state\s+govt|central\s*/\s*state\s+govt"
    r"|(?:scholarships?|schemes?)\s+(?:offered|provided|sponsored|available)"
    r"\s+by\s+(?:the\s+)?(?:central|state|govt|government|ministry|department"
    r"|ugc|aicte|national)"
    r"|government\s+scholarship|govt\.?\s+scholarship)"
)

_GOV_URL = re.compile(r"(?i)\bgov\.in\b|scholarships?\.gov\.in|\bnsp\b")
_PORTAL = re.compile(r"(?i)((?:[a-z0-9-]+\.)*scholarships?[a-z0-9-]*\.gov\.in)")

# Indian money. Anchored on an explicit currency marker or an explicit magnitude
# word, which is precisely why "799004" (a PIN), "9230994080" (a phone) and
# "1956" (an Act year) cannot reach the money path at all.
_MONEY = re.compile(
    r"(?i)(?:(?:Rs\.?|INR|\u20b9|Rupees)\s*\d[\d,]*(?:\.\d+)?"
    r"\s*(?:lakhs?|lacs?|crores?|cr\b|k\b|thousand|million)?"
    r"|\d[\d,]*(?:\.\d+)?\s*(?:lakhs?|lacs?|crores?)\b"
    r"|\d[\d,]*(?:\.\d+)?\s*/-)"
)
_PCT = re.compile(r"\d{1,3}(?:\.\d{1,2})?\s*%")

_NUM_IN_MONEY = re.compile(r"\d[\d,]*(?:\.\d+)?")
_MULT = re.compile(
    r"(?i)\s*(lakhs?|lacs?|l|crores?|cr|k|thousand|million)\b")
_MULTIPLIERS = {
    "l": 1e5, "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5,
    "crore": 1e7, "crores": 1e7, "cr": 1e7,
    "k": 1e3, "thousand": 1e3, "million": 1e6,
}

_ENUM = re.compile(
    r"^\s*(?:\(?(?:\d{1,2}|[ivx]{1,4}|[IVX]{1,4}|[a-hA-H])\)"
    r"|(?:\d{1,2}|[ivx]{1,4}|[IVX]{1,4}|[a-hA-H])\.)\s+")
_BULLET = re.compile(r"^\s*[\u2022\u25cf\u00b7*\-\u2013\u2014]\s+")
# Sentence split that survives "Rs. 10,000" - the lookahead demands a capital,
# and a rupee figure is always followed by a digit.
_SENT = re.compile(r"(?<=[a-z%\)\]])[.;]\s+(?=[A-Z(])")
_ORPHAN = re.compile(r"^[A-Za-z][\w'’-]{0,14},\s+")

_AS_FOLLOWS = re.compile(
    r"(?i)(as\s+follows|as\s+under|as\s+given\s+below|as\s+below"
    r"|following\s+(?:concession|waiver|scholarship|slab|rate))")

# Acronyms stay CASE-SENSITIVE via a scoped (?-i:) group. Case-insensitive
# "\bST\b" matches "St." in "St. Xavier's College", which is a name, not a
# reservation category - measured on real listings, that alone would have
# tagged a large share of Christian-minority colleges with "ST".
_CATEGORIES: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"(?i)(?-i:\bSCs?\b)|scheduled\s+caste"), "SC"),
    (re.compile(r"(?i)(?-i:\bSTs?\b)|scheduled\s+tribe"), "ST"),
    (re.compile(r"(?i)(?-i:\bOBCs?\b)|other\s+backward"), "OBC"),
    (re.compile(r"(?i)(?-i:\bEWS\b)|economically\s+weaker"), "EWS"),
    (re.compile(r"(?i)(?-i:\bEBC\b)|economically\s+backward"), "EBC"),
    (re.compile(r"(?i)minorit"), "Minority"),
    (re.compile(r"(?i)single\s+girl\s+child"), "Single girl child"),
    (re.compile(r"(?i)\bgirls?\b|girl\s+students|\bwomen\b"), "Girl students"),
    (re.compile(r"(?i)disabilit|divyang|differently[- ]abled|\bPwD\b"
                r"|physically\s+(?:handicapped|challenged)"),
     "Persons with disabilities"),
    (re.compile(r"(?i)defence|armed\s+forces|ex[- ]servicemen|martyr"
                r"|paramilitary|(?-i:\bCAPF\b)|police\s+personnel"),
     "Defence/police wards"),
    (re.compile(r"(?i)\bsports?\b|sportsperson|athlet"), "Sports"),
    (re.compile(r"(?i)\borphan"), "Orphans"),
    (re.compile(r"(?i)(?-i:\bBPL\b)|below\s+poverty\s+line"), "BPL"),
    (re.compile(r"(?-i:\bNCC\b)"), "NCC"),
    (re.compile(r"(?i)\bwards?\s+of\s+(?:beedi|cine)"), "Wards of Beedi/Cine workers"),
)

_INCOME = re.compile(
    r"(?i)(?:family|annual|parental|parents'?|gross)?\s*income[^\n]{0,60}?"
    r"(?:Rs\.?|INR|\u20b9)\s*\d[\d,]*(?:\.\d+)?\s*(?:lakhs?|lacs?)?"
    r"|(?:Rs\.?|INR|\u20b9)\s*\d[\d,]*(?:\.\d+)?\s*(?:lakhs?|lacs?)?"
    r"[^\n]{0,40}?\bincome\b")


# ---------------------------------------------------------------------------
# Number parsing
# ---------------------------------------------------------------------------

def _parse_inr(s: str) -> float | None:
    """Rupee string -> float. Handles lakh grouping and magnitude suffixes.

    A plain int("1,20,000") is wrong twice over on Indian pages: the commas are
    not thousands-grouped, and half the figures carry a magnitude word instead
    of digits ("1.2 Lakh", "30K"). Used ONLY to sanity-check a candidate before
    it is emitted - the value written to `facts` is always the verbatim string.
    """
    m = _NUM_IN_MONEY.search(s)
    if not m:
        return None
    try:
        val = float(m.group(0).replace(",", ""))
    except ValueError:
        return None
    suf = _MULT.match(s, m.end())
    if suf:
        val *= _MULTIPLIERS.get(suf.group(1).lower(), 1.0)
    return val


def _plausible_amount(s: str) -> bool:
    """Reject figures too small or too large to be one student's scholarship.

    The floor is deliberately low (Rs 100) because small book/uniform grants are
    real; the ceiling (Rs 1 crore) excludes institutional totals that slipped
    past the disqualifier list.
    """
    v = _parse_inr(s)
    return v is not None and 100.0 <= v <= 1e7


def _plausible_pct(s: str) -> bool:
    m = re.search(r"\d{1,3}(?:\.\d{1,2})?", s)
    if not m:
        return False
    try:
        v = float(m.group(0))
    except ValueError:
        return False
    return 0.0 < v <= 100.0


# ---------------------------------------------------------------------------
# Line helpers
# ---------------------------------------------------------------------------

def _strip_lead(s: str) -> str:
    s = _ENUM.sub("", s, count=1)
    s = _BULLET.sub("", s, count=1)
    return s.strip()


def _finish(seg: str, width: int) -> str:
    out = _strip_lead(seg)
    # A wrapped source line often begins with the tail of the previous one
    # ("Group, will be offered 20% scholarship..."); drop that single orphan.
    if len(out) > 40:
        out = _ORPHAN.sub("", out, count=1)
    if len(out) > width:
        cut = out.rfind(" ", 0, width)      # never end mid-word
        out = out[:cut if cut > width // 2 else width]
    return out.strip(" ,;:-\u2013\u2014")


def _clause(line: str, start: int, end: int, width: int = 140) -> str:
    """Smallest verbatim span around a match that still reads as a statement.

    Offsets are rebased onto every substring taken; an earlier version stripped
    the leading enumerator first and then indexed with the original offsets,
    which silently sliced a sentence mid-word.
    """
    lead = len(line) - len(line.lstrip())
    s = line.strip()
    start = max(0, start - lead)
    end = max(start, end - lead)
    if len(s) <= width:
        return _finish(s, width)

    left, right = 0, len(s)
    for m in _SENT.finditer(s, 0, max(start, 1)):
        left = m.end()
    m = _SENT.search(s, min(end, len(s)))
    if m:
        right = m.start()
    seg = s[left:right]
    rel = start - left
    if len(seg) > width:
        # Grow outward from the comma-bounded piece holding the figure. On a
        # 170-char sentence this keeps "a remission of 50 % to 100 % of total
        # fees is waived for needy students" and drops the trailing clause.
        pieces: list[tuple[int, int]] = []
        pos = 0
        for m in re.finditer(r",\s+", seg):
            pieces.append((pos, m.start()))
            pos = m.end()
        pieces.append((pos, len(seg)))
        hit = next((i for i, (a, b) in enumerate(pieces) if a <= rel < b), 0)
        lo, hi = pieces[hit]
        i, j = hit - 1, hit + 1
        for _ in range(len(pieces)):
            grew = False
            if i >= 0 and hi - pieces[i][0] <= width:
                lo, i, grew = pieces[i][0], i - 1, True
            if j < len(pieces) and pieces[j][1] - lo <= width:
                hi, j, grew = pieces[j][1], j + 1, True
            if not grew:
                break
        seg = seg[lo:hi]
    return _finish(seg, width)


def _tokens_ok(name: str) -> bool:
    """True when `name` identifies a specific scheme rather than a heading."""
    if not (6 <= len(name) <= 120):
        return False
    if _MARKETING.search(name) or "?" in name or "|" in name:
        return False
    if name[0] in "*[(\u2022" or _SENT_INSIDE.search(name):
        return False
    if _GOV_SECTION.search(name):
        return False                       # section header, not a scheme
    raw = name.split()
    if not (2 <= len(raw) <= 16):
        return False
    upper = 0
    distinctive = 0
    cores: list[str] = []
    for t in raw:
        core = t.strip("()[],;:.\"'\u2018\u2019\u201c\u201d*/&-")
        if not core:
            continue
        low = core.lower()
        if low in _UI_WORDS or low in _CLAUSE_WORDS:
            return False
        cores.append(low)
        if core[0].isupper():
            upper += 1
        if low not in _GENERIC and low not in _CONNECTORS:
            distinctive += 1
    if not cores or cores[0] in _ADMIN_HEAD:
        return False
    for a, b in zip(cores, cores[1:]):
        if a in _ANCHORS and b in _HEADING_TAIL:
            return False
    # Two capitalised tokens is what separates a proper name
    # ("Alumni Association Scholarships") from a lowercase UI label
    # ("Scholarship/ fee benefits"); one distinctive token is what separates it
    # from a bare section heading ("Scholarship Schemes").
    return upper >= 2 and distinctive >= 1


def _clean_name(name: str) -> str:
    name = re.sub(r"\s+", " ", name).strip()
    if name.count("(") > name.count(")"):
        name = name[:name.index("(")].strip()
    return name.strip(" *:,;.-\u2013\u2014")


# ---------------------------------------------------------------------------
# Scheme-name detection
# ---------------------------------------------------------------------------

def _name_from_tokens(toks: list[str], i: int) -> str | None:
    """Grow a name outward from the anchor token at index `i`.

    Token walking rather than one big regex: the names on these pages run to
    eight words with slashes, ampersands and apostrophes, and a regex able to
    swallow that on unbounded input is exactly the nested-quantifier shape the
    house rules forbid. This is O(11) per anchor with no backtracking.
    """
    left: list[str] = []
    j, steps = i - 1, 0
    while j >= 0 and steps < 5:
        core = toks[j].strip("()[],;:.\"'\u2018\u2019\u201c\u201d*/&")
        if not core:
            break
        if core[0].isupper() or core.lower() in _CONNECTORS:
            left.append(toks[j])
            j -= 1
            steps += 1
        else:
            break
    left.reverse()
    while left and left[0].strip("()[],;:.").lower() in _CONNECTORS:
        left.pop(0)

    right: list[str] = []
    k = i + 1
    if k < len(toks) and toks[k].strip("()[],;:.").lower() in _TAIL_WORDS:
        right.append(toks[k])
        k += 1
    if k < len(toks) and toks[k].strip("()[],;:.").lower() == "for":
        tail = [toks[k]]
        k += 1
        n = 0
        while k < len(toks) and n < 6:
            core = toks[k].strip("()[],;:.\"'\u2018\u2019\u201c\u201d*/")
            if not core:
                break
            if core[0].isupper() or core.lower() in {"and", "&", "of", "the"}:
                tail.append(toks[k])
                k += 1
                n += 1
            else:
                break
        while tail and tail[-1].strip("()[],;:.").lower() in _CONNECTORS:
            tail.pop()
        if len(tail) > 1:
            right.extend(tail)

    name = _clean_name(" ".join(left + [toks[i]] + right))
    return name if _tokens_ok(name) else None


def _title_name(line: str) -> str | None:
    """A short standalone line that IS a scheme name."""
    s = line.strip()
    if not (6 <= len(s) <= 120) or not _SCHOL_RE.search(s):
        return None
    if _GOV_SECTION.search(s):
        return None            # section header; it sets the gov flag instead
    if s.endswith(":") and re.search(r"(?i)(offered|sponsored|provided)\s+by", s):
        return None            # provider header, not a scheme
    name = _clean_name(_strip_lead(s))
    return name if _tokens_ok(name) else None


def _colon_name(line: str) -> str | None:
    """"Merit Scholarship: the merit scholarship is..." -> the part before ':'."""
    i = line.find(":")
    if i < 4 or i > 90 or len(line) - i < 12:
        return None
    head = line[:i].strip()
    if not _SCHOL_RE.search(head) or _GOV_SECTION.search(head):
        return None
    name = _clean_name(_strip_lead(head))
    return name if _tokens_ok(name) else None


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _add(bucket: list[str], value: str, cap: int) -> None:
    """First-seen wins on substring collisions.

    Deliberately does NOT upgrade to the longer variant: on tiut.ac.in that
    replaced the clean "Chancellor Merit cum Means Scholarship" with the
    sentence that happened to contain it.
    """
    v = value.strip()
    if not v or len(bucket) >= cap:
        return
    low = v.lower()
    for existing in bucket:
        e = existing.lower()
        if low in e or e in low:
            return
    bucket.append(v)


def _join(bucket: list[str], limit: int, total: int = 420) -> str:
    out, used = [], 0
    for v in bucket[:limit]:
        if used + len(v) > total and out:
            break
        out.append(v)
        used += len(v) + 2
    extra = len(bucket) - len(out)
    s = "; ".join(out)
    if extra > 0:
        s += f" (and {extra} more listed)"
    return s


def _extract(text: str, url: str, college_name: str) -> dict | None:
    if not isinstance(text, str):
        return None
    if len(text) > _MAX_TEXT:
        text = text[:_MAX_TEXT]
    raw_lines = text.split("\n")[:_MAX_LINES]
    if len(raw_lines) < 3:
        return None

    # Desktop + mobile menus both reach the text, so nav labels appear as exact
    # duplicate short lines. Suppressing them is what makes a nav-only page
    # (bncollegepatna.com: "Scholarships" twice, nothing else) score zero.
    seen: dict[str, int] = {}
    for ln in raw_lines:
        s = ln.strip()
        if s:
            seen[s] = seen.get(s, 0) + 1
    lines = [ln.strip() for ln in raw_lines if ln.strip()]
    # Exempt lines carrying a figure: a table body legitimately repeats
    # "100% on tuition fees" once per scholarship tier (tiut.ac.in repeats it
    # six times), and suppressing those cost the entire benefit column.
    content = [s for s in lines
               if not (len(s) <= 60 and seen.get(s, 0) >= 2
                       and not _PCT.search(s) and not _MONEY.search(s))]
    if not content:
        return None

    topical = [s for s in content if _SCHOL_RE.search(s)]
    if len(topical) < 2:
        return None                        # nav mention only -> let the LLM look

    schemes: list[str] = []
    gov_schemes: list[str] = []
    waivers: list[str] = []
    amounts: list[str] = []
    merit: list[str] = []
    cats: list[str] = []
    income: list[str] = []
    portals: list[str] = []

    gov_section = False
    block_left = 0            # lines of "as follows" benefit context still open
    block_miss = 0
    cat_window = 0

    for idx, line in enumerate(content):
        low_len = len(line)
        if low_len > 1200:
            line = line[:1200]

        if _GOV_SECTION.search(line) and len(line) <= 120:
            gov_section = True

        m = _PORTAL.search(line)
        if m:
            _add(portals, m.group(1), 3)

        topical_line = _SCHOL_RE.search(line) is not None
        if topical_line:
            cat_window = 3
        elif cat_window:
            cat_window -= 1

        # --- scheme names -------------------------------------------------
        if topical_line and len(schemes) + len(gov_schemes) < 40:
            found: list[str] = []
            n = _colon_name(line) or _title_name(line)
            if n:
                found.append(n)
            if not found:
                toks = line.split()
                if len(toks) <= 60:
                    for i, t in enumerate(toks):
                        if t.strip("()[],;:.\"'\u2018\u2019\u201c\u201d*/&").lower() in _ANCHORS:
                            nm = _name_from_tokens(toks, i)
                            if nm:
                                found.append(nm)
                            if len(found) >= 2:
                                break
            for nm in found:
                ctx = " ".join(content[idx:idx + 3])
                is_gov = bool(
                    _GOV_NAME.search(nm) or gov_section or _GOV_URL.search(ctx))
                _add(gov_schemes if is_gov else schemes, nm, 20)

        # --- benefits -----------------------------------------------------
        value_line = False
        disq = _DISQUALIFY.search(line) is not None
        pm = _PCT.search(line)
        mm = _MONEY.search(line)

        if (pm or mm) and not disq:
            cue = _BENEFIT_CUE.search(line)
            in_block = block_left > 0 and len(line) <= 120
            if cue or in_block:
                if pm and _plausible_pct(pm.group(0)):
                    _add(waivers, _clause(line, pm.start(), pm.end()), 8)
                    value_line = True
                elif mm and _plausible_amount(mm.group(0)):
                    _add(amounts, _clause(line, mm.start(), mm.end()), 6)
                    value_line = True

        # --- eligibility --------------------------------------------------
        if pm and not value_line and not disq and len(line) <= 200:
            if _MERIT_CUE.search(line) and not _BENEFIT_CUE.search(line):
                _add(merit, _clause(line, pm.start(), pm.end(), 120), 8)

        if not disq:
            im = _INCOME.search(line)
            if im and len(line) <= 300:
                _add(income, _clause(line, im.start(), im.end(), 130), 2)

        if topical_line or cat_window:
            for rx, label in _CATEGORIES:
                if label not in cats and rx.search(line):
                    cats.append(label)

        # --- "...as follows:" opens a short benefit block ------------------
        if _AS_FOLLOWS.search(line) and _BENEFIT_CUE.search(line):
            block_left, block_miss = 8, 0
        elif block_left:
            block_left -= 1
            if value_line:
                block_miss = 0
            else:
                block_miss += 1
                if block_miss >= 3:
                    block_left = 0

    facts: dict[str, str] = {}
    for i, nm in enumerate(schemes[:8], 1):
        facts[f"scheme_{i}"] = nm
    if gov_schemes:
        facts["government_schemes"] = _join(gov_schemes, 6)
    if waivers:
        facts["fee_waiver"] = _join(waivers, 6)
    if amounts:
        facts["scholarship_amount"] = _join(amounts, 5)
    if merit:
        facts["merit_eligibility"] = _join(merit, 6)
    if income:
        facts["income_criterion"] = _join(income, 2)
    if cats:
        facts["eligible_categories"] = ", ".join(cats[:8])
    if portals:
        facts["apply_via"] = _join(portals, 2)

    if not facts:
        return None
    # Categories alone are too weak to stand as a fact: "SC" and "girl" appear
    # in plenty of prose that mentions scholarships without stating one.
    if set(facts) <= {"eligible_categories"}:
        return None

    conf = _confidence(schemes, gov_schemes, waivers, amounts, merit, cats,
                       portals, url, len(topical))
    if conf < 0.5:
        return None                        # hand it to the LLM instead

    summary = _summarise(college_name, schemes, gov_schemes, waivers, amounts,
                         merit, cats, portals)
    if not summary:
        return None
    return {"summary": summary, "facts": facts, "confidence": round(conf, 2)}


def _confidence(schemes, gov, waivers, amounts, merit, cats, portals,
                url, topical_hits) -> float:
    """Calibrated against the five real pages this was built on.

    Named schemes are cheap to find and therefore weighted low; an explicit
    benefit figure is the expensive, decisive signal and is weighted high. The
    0.70 ceiling when no benefit figure exists is the honest statement that the
    page named schemes but never said what they are worth - which is exactly
    lpu.in and chitkara.edu.in.
    """
    c = 0.25
    c += min(0.30, 0.12 * len(schemes))
    c += min(0.34, 0.14 * (len(waivers) + len(amounts)))
    c += min(0.12, 0.05 * len(merit))
    c += min(0.14, 0.07 * len(gov))
    c += min(0.08, 0.04 * len(cats))
    if portals:
        c += 0.04
    if re.search(r"(?i)scholar|freeship|financial[-_]?aid|fee[-_]?(waiver|concession)",
                 str(url or "")):
        c += 0.04
    if topical_hits >= 8:
        c += 0.04
    if not waivers and not amounts:
        c = min(c, 0.70)
    # 0.90 is the top of the scale in the house contract ("an explicit labelled
    # table"). Nothing here earns more than that: parse_html() has already
    # thrown the table structure away.
    return max(0.0, min(0.90, c))


def _summarise(college_name, schemes, gov, waivers, amounts, merit, cats,
               portals) -> str:
    """<=45 words of hedged, page-sourced prose. Verbs are 'lists'/'states' so
    the card never promises anything the college did not."""
    name = re.sub(r"\s+", " ", str(college_name or "")).strip()
    if not name or len(name) > 60 or _MARKETING.search(name):
        name = "The college"
    parts: list[str] = []
    if schemes:
        head = ", ".join(schemes[:3])
        more = f" and {len(schemes) - 3} more" if len(schemes) > 3 else ""
        parts.append(f"{name} lists its own scholarships: {head}{more}.")
    if waivers or amounts:
        vals = [v.rstrip(" .") for v in (waivers + amounts)[:2]]
        parts.append("Stated benefits include " + " and ".join(vals) + ".")
    if gov:
        parts.append("Government schemes listed: " + ", ".join(gov[:2]) + ".")
    if merit and not (waivers or amounts):
        parts.append("Stated criteria include " + merit[0].rstrip(" .") + ".")
    if cats and len(parts) < 3:
        parts.append("Eligibility mentioned for " + ", ".join(cats[:4]) + ".")
    if portals and len(parts) < 2:
        parts.append("Applications go through " + portals[0] + ".")
    if not parts:
        return ""
    out, words = [], 0
    for p in parts:
        w = len(p.split())
        if words + w > 45 and out:
            break
        out.append(p)
        words += w
    text = " ".join(out)
    if len(text.split()) > 45:
        text = " ".join(text.split()[:45]).rstrip(",;:") + "."
    return text[:600]


def extract(text: str, url: str, college_name: str) -> dict | None:
    """Page text -> {"summary", "facts", "confidence"} or None.

    Never raises: the sweep runs unattended over 10,000 third-party sites and
    one exception kills a whole batch. None is a correct, frequent answer.
    """
    try:
        return _extract(text, url, college_name)
    except Exception:      # noqa: BLE001 - a broken page must cost one page, not a batch
        return None


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import json
    import pathlib
    import sys
    import time

    def show(label: str, res, url="", elapsed=None) -> None:
        tag = f"{label}"
        if elapsed is not None:
            tag += f"  [{elapsed * 1000:.1f} ms]"
        print("=" * 78)
        print(tag)
        if url:
            print("  url:", url)
        if res is None:
            print("  -> None")
            return
        print(f"  confidence: {res['confidence']}")
        print(f"  summary ({len(res['summary'].split())}w): {res['summary']}")
        for k, v in res["facts"].items():
            print(f"    {k}: {v}")

    # --- number parser --------------------------------------------------
    print("### _parse_inr")
    cases = [
        ("Rs 45,000", 45000), ("1,20,000", 120000), ("Rs. 1,20,000/-", 120000),
        ("1.2 Lakh", 120000), ("1.2L", 120000), ("12000/-", 12000),
        ("Rs 45000 p.a.", 45000), ("\u20b930K", 30000), ("\u20b96.8 lakh", 680000),
        ("Rs 450 Lakhs", 45000000), ("100 Cr", 1000000000),
        ("Rs.10000/-", 10000), ("INR 2,50,000 per annum", 250000),
    ]
    bad = 0
    for s, want in cases:
        got = _parse_inr(s)
        ok = got is not None and abs(got - want) < 1
        bad += 0 if ok else 1
        print(f"  {'ok ' if ok else 'BAD'} {s!r:28} -> {got!r} (want {want})")
    print(f"  -> {len(cases) - bad}/{len(cases)} correct")

    # --- real pages -----------------------------------------------------
    here = pathlib.Path(
        r"C:\Users\ayush\AppData\Local\Temp\claude"
        r"\E--mivi-on-dataset\bf1f9a57-3d94-451b-aac3-60543b0d26a6"
        r"\scratchpad\sch_pages")
    names = {
        "tiut": "Techno India University, Tripura",
        "chitkara": "Chitkara University",
        "christ": "CHRIST (Deemed to be University)",
        "lpu": "Lovely Professional University",
        "bncollegepatna": "B. N. College, Patna",
    }
    print("\n### REAL PAGES (fetched live via web_enrich.parse_html)")
    if here.exists():
        for slug, cname in names.items():
            p = here / f"{slug}.txt"
            if not p.exists():
                continue
            body = p.read_text(encoding="utf-8")
            first, _, rest = body.partition("\n")
            url = first.replace("# URL:", "").strip()
            t0 = time.perf_counter()
            res = extract(rest, url, cname)
            dt = time.perf_counter() - t0
            show(f"REAL {slug} ({len(rest)} chars)", res, url, dt)
    else:
        print("  (captured pages not present; skipping)")

    # --- adversarial ----------------------------------------------------
    print("\n### ADVERSARIAL")
    adv: list[tuple[str, str, str]] = [
        ("empty", "", "https://x.ac.in/scholarship"),
        ("whitespace", "   \n \n\t\n  ", "https://x.ac.in/scholarship"),
        ("none-input", None, "https://x.ac.in/scholarship"),  # type: ignore[list-item]
        ("nav only", "Home\nAbout\nScholarships\nContact\nHome\nAbout\n"
                     "Scholarships\nContact\n", "https://x.ac.in/scholarship"),
        ("different topic (hostel)",
         "Hostel Facilities\nThe boys hostel has 240 rooms.\n"
         "Hostel fee is Rs 45,000 per year including mess charges.\n"
         "Mess charges Rs 3,500 per month.\nRooms are allotted on merit.\n",
         "https://x.ac.in/hostel"),
        ("phone/PIN/year/course-code trap",
         "Scholarship Cell\nContact the scholarship cell for details.\n"
         "Phone: +91 9230994080 | 81 | 82\nWomen Helpline: +91 700 532 3326\n"
         "Campus: Anandanagar, Agartala, Tripura - 799004\n"
         "Estd. under Act No. 4 of 2023 & UGC u/s 2(f) of UGC Act. 1956\n"
         "Affiliation No. 1-4567890123\nCourse code CSE-301, MBA-402\n"
         "Total Users: 630597\nUsers Today: 161\n"
         "Toll free 1800 267 1999\nRoom No :122, IV Block\n"
         "College established in 1889.\n"
         "The scholarship committee meets every semester.\n",
         "https://x.ac.in/scholarship"),
        ("institutional total only",
         "Scholarships\nThe university allocates a total amount of about "
         "Rs 450 Lakhs for fee concessions and scholarships annually.\n"
         "Scholarship amount disbursed last year: Rs 100 Cr.\n"
         "Our scholarship office is on the first floor.\n",
         "https://x.ac.in/scholarship"),
        ("installment threshold trap",
         "Scholarships and Installment Facility\n"
         "Such applications are processed through the office of scholarships. "
         "This facility applies only where the total fee is above Rs.10000/-.\n"
         "Students unable to pay fees in one installment may apply.\n"
         "The office of scholarships is open on weekdays.\n",
         "https://x.ac.in/scholarship"),
        ("marketing only",
         "Scholarships\nWe have the best scholarships in the region.\n"
         "Ranked No.1 for financial aid. Unlock your dreams with our "
         "world class scholarship support.\nApply now!\n",
         "https://x.ac.in/scholarship"),
        ("cutoff page mentioning the word once",
         "B.Tech Cut Off 2026\nGeneral 178 marks\nOBC 165 marks\nSC 140 marks\n"
         "Counselling starts in July. Scholarship details are on another page.\n",
         "https://x.ac.in/cutoff"),
        ("real but minimal",
         "Scholarship\nMerit Scholarship: a 25% tuition fee waiver is given to "
         "students scoring above 90% in the qualifying examination.\n"
         "Applications close in August.\n",
         "https://x.ac.in/scholarship"),
    ]
    for label, body, url in adv:
        t0 = time.perf_counter()
        res = extract(body, url, "Test College")  # type: ignore[arg-type]
        show(f"ADV {label}", res, url, time.perf_counter() - t0)

    # --- speed ----------------------------------------------------------
    print("\n### SPEED (6,000-char page, 200 runs)")
    sample = None
    if here.exists() and (here / "christ.txt").exists():
        sample = (here / "christ.txt").read_text(encoding="utf-8")[:6000]
    if sample:
        t0 = time.perf_counter()
        for _ in range(200):
            extract(sample, "https://x.ac.in/scholarship", "Test College")
        per = (time.perf_counter() - t0) / 200 * 1000
        print(f"  {per:.2f} ms/page  (budget 50 ms)")

    # --- shape ----------------------------------------------------------
    print("\n### CONTRACT SHAPE")
    if here.exists() and (here / "tiut.txt").exists():
        body = (here / "tiut.txt").read_text(encoding="utf-8")
        r = extract(body.partition("\n")[2], "https://www.tiut.ac.in/scholarship",
                    "Techno India University, Tripura")
        assert r is not None
        assert set(r) == {"summary", "facts", "confidence"}
        assert isinstance(r["summary"], str) and len(r["summary"].split()) <= 45
        assert isinstance(r["facts"], dict)
        assert all(isinstance(k, str) and isinstance(v, str)
                   for k, v in r["facts"].items())
        assert 0.0 <= r["confidence"] <= 1.0
        print("  ok: keys/types/word-limit/confidence range all hold")
        print("  json:", json.dumps(r, ensure_ascii=False)[:200], "...")
    sys.stdout.flush()
