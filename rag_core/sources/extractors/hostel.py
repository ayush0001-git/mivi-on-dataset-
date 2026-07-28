"""Deterministic hostel/accommodation extractor. No LLM, no network, stdlib only.

WHY THIS IS WORTH WRITING
The sweep is bounded by ~14 LLM calls/minute. Hostel pages are unusually
mechanical -- "There are two hostels one each for boys and girls", "Hostel Fee
Rs. 45,000 per annum" -- so most of them can be read by regex, and every page
read here is a page the quota never has to pay for.

WHAT MAKES THIS HARD, MEASURED ON REAL PAGES
1. Indian college pages are 80-95% navigation. On gpmanesar.ac.in/hostel.aspx
   the word "HOSTEL" appears exactly twice, both as menu labels; the page has no
   hostel content at all. On bncollegepatna.com/accomodation.php the nav names
   "Hostel Facility", "Our Hostel" and "Accommodation" eleven times and the body
   says "Yet we doesn't received any accomodation." Any extractor that triggers
   on keyword presence returns a fact for both. Both must return None.
   => Facts are only taken from sentences that make a STATEMENT (a verb, a
      number, an amount). Bare labels are structurally invisible here.
2. Prose is hard-wrapped mid-sentence. gcggn.ac.in breaks
   "There are 5 girls Hostel with a capacity of 750 / students." across two
   lines, so line-at-a-time matching truncates the fact. => reflow first.
3. Sentence scope is what keeps numbers honest. The same gcggn page carries
   "seating capacity of over 1000" (auditorium), "seating capacity at 200"
   (library), "44,000 books" and "250 KW"; srhu.edu.in carries "a 1200-bed
   teaching hospital" one sentence after a sentence about hostels. A number is
   only ever read out of a sentence that is itself about accommodation.

WHAT THIS DELIBERATELY REFUSES
- Marketing prose. The summary is always GENERATED from extracted values and
  never quoted from the page, so "Home Away from Home for 15,000 Students" and
  "I was staying in a hostel for the first time..." (both real, amity.edu)
  cannot leak into a student's card.
- Ambiguous fee tables. lpu.in prints five tables x two price columns
  (early-decision vs not) all labelled "Standard Room". When one label has more
  than one candidate amount the label is DROPPED, not guessed. That page falls
  through to the LLM for fees while still yielding availability and room types
  here, which is the correct division of labour.
- Bare numbers without a fee word next to them. Phone numbers, PIN codes,
  establishment years, visitor counters and "300 free electricity units" are all
  present on the pages tested and none of them are money.

Self-test:  python -m rag_core.sources.extractors.hostel
"""
from __future__ import annotations

import re

# The sweep hands over the whole page text, not the 6k slice it gives the LLM.
# Scanning is linear, but a runaway page (a 2 MB sitemap dump) should cost a
# bounded amount of CPU rather than a proportional one.
MAX_SCAN_CHARS = 200_000
MAX_FACTS = 14

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

# "accomodation" is not a typo on our side -- it is how bncollegepatna.com and a
# visible slice of Indian college sites spell it, including in the URL.
_HOSTEL = (r"hostel|hostels|hosteller|accommodation|accomodation|dormitor|"
           r"boarding|lodging|residence hall|residential hall|hall of residence|"
           r"student residence|paying guest")
HOSTEL_RE = re.compile(_HOSTEL, re.I)

# "residence"/"mess" alone are too loose to open a sentence (a principal has a
# residence; a college has a mess for day scholars), but they are fine as
# supporting context once a sentence already qualifies.
CONTEXT_RE = re.compile(_HOSTEL + r"|\bmess\b|\bwarden\b|\brooms?\b|\bbeds?\b", re.I)

# A sentence must assert something. This is the single filter that separates a
# nav label ("Hostel Details") from a fact ("There are two hostels...").
ASSERT_RE = re.compile(
    r"\b(is|are|was|were|has|have|had|provides?|provided|offers?|offering|"
    r"offered|available|availability|equipped|furnished|consists?|comprising|"
    r"comprises?|accommodates?|houses?|maintains?|allotted|allocated|charged|"
    r"charges|payable|includes?|including|inclusive|separate|separately|"
    r"can stay|may stay|there is|there are)\b", re.I)

# First person = testimonial, not an institutional fact (amity.edu carries one).
FIRSTPERSON_RE = re.compile(r"(?:^|[\s\"'(])(?:I|I'm|I've|my|My|me)\b")

GENDER_BOYS = re.compile(r"\b(boys?|men'?s?|male|gents?)\b", re.I)
GENDER_GIRLS = re.compile(r"\b(girls?|women'?s?|female|ladies|womens)\b", re.I)
SEPARATE_RE = re.compile(r"\bseparate(?:ly)?\b|\beach for\b|\bone each\b|\brespectively\b", re.I)

NEGATIVE_RE = re.compile(
    r"\b(no hostel|not available|does not (?:have|provide|offer)|"
    r"don'?t have|no accommodation|hostel is not|without hostel|"
    r"do not provide)\b", re.I)

# Numbers belonging to something that is not a hostel. Used to veto a capacity
# reading even when the sentence does mention accommodation somewhere.
NOT_HOSTEL_NOUN = re.compile(
    r"\b(hospital|auditorium|library|classroom|lecture|seminar|laborator|lab|"
    r"canteen seating|book|computer|vehicle|parking|kw|kva|sqm|sq\.?\s?ft|acre)\b",
    re.I)

PHONE_CTX = re.compile(
    r"\b(phone|ph\.?|tel|telephone|mobile|contact|helpline|fax|call|whatsapp|"
    r"pin|pincode|pin code|isbn|gst|reg(?:istration)? no)\b", re.I)

# ---------------------------------------------------------------------------
# Money
# ---------------------------------------------------------------------------

# Indian lakh grouping: 1,20,000 groups as 1 / 20 / 000, so the group size after
# the first comma is 2 OR 3. A plain int("1,20,000".replace(",","")) happens to
# work, but the PATTERN has to allow the 2-digit groups or the match stops at
# "1,20" and reports Rs 1,20 as the fee.
_NUM = r"\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?"
_CUR = r"(?:₹|Rs\.?|INR|Rupees|Rupee)"
_UNIT = r"(?:lakhs?|lacs?|lakh|crores?|[LK])"
_PERIOD = (r"(?:p\.?\s?a\.?|p\.?\s?m\.?|per\s+annum|per\s+year|per\s+month|"
           r"per\s+semester|per\s+sem\b|per\s+day|per\s+student|per\s+head|"
           r"annually|monthly|yearly|/\s?(?:yr|year|month|mo|sem|semester|annum|day))")

# Three concrete shapes, each anchored so there is no nested-quantifier blowup:
#   A  Rs. 83,000/-   |  Rs 45000 p.a.  |  INR 1,20,000 per annum  |  Rs.1.2 Lakh
#   B  1.2 Lakh       |  45,000 rupees  |  1.2L
#   C  12000/-        (the bare "/-" suffix is itself a rupee marker in India)
MONEY_RE = re.compile(
    rf"(?:{_CUR}\s*(?P<a>{_NUM})\s*(?:{_UNIT}\b)?"
    rf"|(?P<b>{_NUM})\s*{_UNIT}\b"
    rf"|(?P<c>{_NUM})\s*/\s?-)"
    rf"(?:\s*/?-)?(?:\s*(?:{_PERIOD}))?", re.I)

FEE_WORD = re.compile(
    r"\b(fee|fees|fees?structure|charge|charges|chargeable|rent|rental|tariff|"
    r"cost|amount|payable|deposit|dues|tuition)\b", re.I)

MESS_WORD = re.compile(r"\b(mess|dining|boarding|foods?|meals?|canteens?|diet)\b", re.I)

# "deposit" is a VERB at least as often as it is a noun on these pages.
# dscw45.ac.in: "Step 1: Kindly deposit an amount of Rs 750 (Cost of Prospectus)"
# was read as a hostel security deposit until this required a qualifier.
DEPOSIT_WORD = re.compile(
    r"\b(?:security|caution|refundable|hostel|room)[\s-]*(?:money[\s-]*)?deposits?\b"
    r"|\bcaution money\b"
    r"|\bdeposits?\s*\(\s*(?:refundable|one[- ]time|security)", re.I)
TOTAL_WORD = re.compile(r"\b(total|overall|all inclusive|altogether)\b", re.I)

# Two tiers on purpose. SPECIFIC names an actual room class and is safe to turn
# into a per-room-type rate. GENERIC ("room", "suite") is only a hint that the
# amount is accommodation-related. Both need word boundaries: an earlier draft
# used a bare "a/?c" and happily read "Academic fee Rs 50,000" as an AC room.
ROOMTYPE_SPECIFIC = re.compile(
    r"\b(?:single|double|triple|twin|quad|four|three|two|one)[- ]?"
    r"(?:seater|seated|sharing|occupancy|bedded|bed|room|seat)\b"
    r"|\b\d\s*(?:seater|seated|sharing|bedded|bed)\b"
    r"|\b(?:a\.?/?c|non[- ]?a\.?/?c|air[- ]?condition\w*|air[- ]?cooled)\b"
    r"|\b(?:deluxe|economy|standard|super\s?deluxe)\s+(?:room|suite)\b", re.I)
ROOMTYPE_GENERIC = re.compile(r"\b(rooms?|suites?|apartments?|cubicles?|"
                              r"dormitor\w+|seats?)\b", re.I)
HOSTEL_FEE_WORD = re.compile(
    r"\b(hostel|hostels|accommodation|accomodation|residence|residential|"
    r"lodging|boarding|room rent|seat rent)\b", re.I)

# ---------------------------------------------------------------------------
# Room types & facilities
# ---------------------------------------------------------------------------

ROOM_TYPES = (
    ("Single occupancy", r"\bsingle(?:\s+(?:room|seater|occupancy|sharing|bedded))?\b|"
                         r"\b1\s*(?:seater|sharing|bed(?:ded)?)\b|\bone[- ]seater\b"),
    ("Double occupancy", r"\bdouble(?:\s+(?:room|seater|occupancy|sharing|bedded))?\b|"
                         r"\btwin\s+sharing\b|\b2\s*(?:seater|sharing|bed(?:ded)?)\b|"
                         r"\btwo[- ]seater\b"),
    ("Triple occupancy", r"\btriple(?:\s+(?:room|seater|occupancy|sharing|bedded))?\b|"
                         r"\b3\s*(?:seater|sharing|bed(?:ded)?)\b|\bthree[- ]seater\b"),
    ("Four sharing", r"\b4\s*(?:seater|sharing|bed(?:ded)?)\b|\bfour[- ]seater\b|"
                     r"\bquad\b"),
    ("AC", r"\bair[- ]?condition(?:ed|ing)?\b|\ba\.?/?c\.?\s+room|\bac\s+room"),
    ("Non-AC", r"\bnon[- ]?a\.?/?c\.?\b|\bair[- ]?cooled\b|\bnon[- ]?air[- ]?condition"),
)
ROOM_TYPES = tuple((label, re.compile(pat, re.I)) for label, pat in ROOM_TYPES)

# Plurals matter more than they look: an early run missed "Geysers are available
# in the hostel bathrooms" (gcechd.ac.in) because \bgeyser\b cannot match
# "Geysers". Every entry below is pluralised for that reason.
FACILITIES = (
    ("Wi-Fi", r"\bwi-?fi\b|\binternet\b|\bbroadband\b"),
    ("Laundry", r"\blaundry\b|\bwashing\b|\bironing\b|\bdhobi\b"),
    ("Gym", r"\bgyms?\b|\bgymnasiums?\b|\bfitness\b"),
    ("Medical care", r"\bmedical\b|\bdispensar\w+|\bsick room\b|\bnurses?\b|"
                     r"\bdoctors?\b|\bfirst aid\b|\binfirmary\b|\bambulances?\b"),
    ("Mess/dining", r"\bmess\b|\bdining\b|\bcanteens?\b|\bcafeterias?\b"),
    ("Purified water", r"\bro (?:plant|system)\w*|\bwater coolers?\b|\bpurifiers?\b|"
                       r"\baquaguard\b|\bpurified water\b|\bdrinking water\b"),
    ("Hot water", r"\bgeysers?\b|\bhot water\b|\bsolar water\b"),
    ("Study room", r"\bstudy rooms?\b|\breading rooms?\b|\bstudy halls?\b"),
    ("Common room / TV", r"\bcommon rooms?\b|\btelevisions?\b|\bt\.?v\.?\b|"
                         r"\brecreation\w*|\bindoor games\b"),
    ("Power backup", r"\bpower back-?up\b|\bgenerators?\b|\binverters?\b|\bups\b"),
    ("Security / warden", r"\bcctv\b|\bsecurity\b|\bwardens?\b|\bbiometric\b"),
    ("Furnished rooms", r"\bfurnished\b|\balmirahs?\b|\bcupboards?\b|\bwardrobes?\b|"
                        r"\bstudy tables?\b|\bmattress\w*"),
)
FACILITIES = tuple((label, re.compile(pat, re.I)) for label, pat in FACILITIES)

# jmi.ac.in: "a combined capacity to house 2220 boys and 1900 girls". Reporting
# only the first half is a half-truth, so a two-gender figure is taken whole.
GENDER_CAPACITY_RE = re.compile(
    r"(\d{1,5})\s*(?:boys?|girls?|male|female|men|women)(?:\s+students?)?"
    r"\s*(?:and|,|&)\s*"
    r"(\d{1,5})\s*(?:boys?|girls?|male|female|men|women)(?:\s+students?)?", re.I)

CAPACITY_RE = re.compile(
    r"\b(?:capacity|intake|strength|accommodat\w*|houses?|houses|seats?)\D{0,20}?"
    r"(\d{1,5})\s*(students?|seats?|beds?|inmates?|boarders?|residents?|girls?|boys?)?"
    r"|\b(\d{1,5})\s*(students?|seats?|beds?|inmates?|boarders?|residents?)\b",
    re.I)
ROOMCOUNT_RE = re.compile(r"\b(\d{1,4})\s+(?:well[- ]?)?(?:furnished\s+)?rooms?\b", re.I)

MESS_SEPARATE = re.compile(
    r"\bmess\b[^.]{0,60}\b(?:extra|separate|additional|not included|excluded)\b|"
    r"\b(?:excluding|exclusive of|apart from|in addition to)\b[^.]{0,30}\bmess\b",
    re.I)
# "along with" was here and had to go: lpu.in's "The residential fee, along with
# charges for mess, laundry and gym, are all managed by Pro Bricks" says who
# collects the money, not that mess is bundled into the hostel fee.
MESS_INCLUDED = re.compile(
    r"\b(?:includ\w+|inclusive of)\b[^.]{0,40}\bmess\b|"
    r"\bmess\b[^.]{0,40}\b(?:is\s+)?(?:included|inclusive)\b", re.I)
MESS_ABSENT = re.compile(
    r"\bmess\b[^.]{0,40}\b(?:is\s+)?not\s+(?:available|provided)\b|"
    r"\bno\s+mess\b", re.I)


# ---------------------------------------------------------------------------
# Number parsing (used ONLY to sanity-check; output is always verbatim)
# ---------------------------------------------------------------------------

def _to_number(raw: str) -> float | None:
    """'1,20,000' -> 120000.0, '1.2 Lakh' -> 120000.0, '12000/-' -> 12000.0.

    Exists purely so an amount can be range-checked and compared against the
    year/PIN/phone shapes. The value that reaches the catalogue is always the
    original substring -- rule 4 of the brief, and the reason a student sees
    "Rs 1.2 Lakh per annum" rather than our arithmetic.
    """
    if not raw:
        return None
    s = raw.strip()
    mult = 1.0
    low = s.lower()
    if re.search(r"crores?\b", low):
        mult = 10_000_000.0
    elif re.search(r"lakhs?\b|lacs?\b", low):
        mult = 100_000.0
    elif re.search(r"\d\s*L\b", s):          # 1.2L -- capital L only; 'l' is litres
        mult = 100_000.0
    elif re.search(r"\d\s*K\b", s, re.I):
        mult = 1_000.0
    m = re.search(_NUM, s)
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", "")) * mult
    except ValueError:
        return None


def _digit_run(raw: str) -> int:
    """Longest run of consecutive digits, ignoring separators -- a 10-digit run
    is a phone number, never a fee."""
    return max((len(g) for g in re.findall(r"\d+", raw)), default=0)


def _plausible_money(raw: str, has_currency: bool) -> bool:
    """Reject the number shapes that pollute Indian college pages.

    Every rejection below was hit by a page in the test corpus: establishment
    years (gcwmadlauda "Establishment Year : 2014"), PIN codes (christuniversity
    "Bengaluru - 560029"), phone numbers (gpmanesar "0124-2337243"), visitor
    counters (bncollegepatna "350207") and unit counts (lpu "300 free
    electricity units").
    """
    n = _to_number(raw)
    if n is None:
        return False
    digits = _digit_run(raw)
    if digits >= 9:                      # phone / account / enrolment number
        return False
    if not has_currency:
        # A bare 4-digit number in the calendar range is a year, and a bare
        # 6-digit run is a PIN code. With "Rs." in front, neither reading holds.
        if 1900 <= n <= 2099 and digits == 4:
            return False
        if digits == 6:
            return False
    # Hostel money lives between a hundred rupees and fifty lakh. Outside that
    # band the token is measuring something else.
    return 100.0 <= n <= 5_000_000.0


# ---------------------------------------------------------------------------
# Text shaping
# ---------------------------------------------------------------------------

def _lines(text: str) -> list[str]:
    out = []
    for ln in text.split("\n"):
        ln = ln.strip()
        if ln:
            out.append(ln)
    return out


def _sentences(lines: list[str]) -> list[str]:
    """Rebuild sentences from hard-wrapped prose without gluing the nav together.

    A continuation line starts lowercase; a nav label starts capitalised. That
    one rule reflows gcggn's 'a capacity of 750 / students.' correctly while
    leaving 'Hostel / Field / Library / Computer lab / Mess' as five separate
    fragments, which is exactly the distinction the assertion filter needs.
    """
    paras: list[str] = []
    buf = ""
    joined = 0
    for ln in lines:
        cont = bool(re.match(r"[a-z(–—-]", ln))
        # A digit may also open a continuation ("...deposit of Rs" / "10,000/- is
        # payable"), but only after something that already reads as prose --
        # otherwise a numbered menu ("01.Home", "02.About Us") glues itself into
        # one giant pseudo-sentence.
        if not cont and ln[:1].isdigit() and len(buf) >= 45 and buf.count(" ") >= 6:
            cont = True
        if buf and cont and joined < 12 and len(buf) < 600 and buf[-1] not in ".!?:":
            buf += " " + ln
            joined += 1
            continue
        if buf:
            paras.append(buf)
        buf, joined = ln, 0
    if buf:
        paras.append(buf)

    sents: list[str] = []
    for p in paras:
        for s in re.split(r"(?<=[.!?;])\s+|\s*[•·●]\s*", p):
            s = re.sub(r"\s+", " ", s).strip()
            # Long fragments with no punctuation are concatenated menus, not
            # prose; short ones are labels. Both are useless as facts.
            if 12 <= len(s) <= 700:
                sents.append(s)
    return sents


def _slug(label: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return re.sub(r"_+", "_", s)[:44]


# ---------------------------------------------------------------------------
# Money extraction
# ---------------------------------------------------------------------------

def _classify(label: str) -> str | None:
    """Map the words sitting next to an amount onto a catalogue field name.

    There is deliberately NO generic "contains the word fee" fallback. The page
    classifier routes anything matching "facilities" to this topic, so a
    hostel-labelled page routinely also carries tuition tables; without a
    hostel/room/mess/deposit word next to the amount there is no evidence the
    figure is accommodation at all, and a tuition fee shown to a student as a
    hostel fee is a worse error than showing nothing.
    """
    if DEPOSIT_WORD.search(label):
        return "security_deposit"
    if MESS_WORD.search(label):
        return "mess_charges"
    if TOTAL_WORD.search(label) and (HOSTEL_FEE_WORD.search(label)
                                     or ROOMTYPE_GENERIC.search(label)):
        return "total_accommodation_cost"
    if ROOMTYPE_SPECIFIC.search(label):
        base = re.sub(r"\b(fee|fees|charges?|charge|rent|rental|tariff|per|"
                      r"amount|payable|type|category)\b", " ", label, flags=re.I)
        slug = _slug(base.strip(" :.-|–"))
        return ("room_rate_" + slug) if slug else "hostel_fee"
    if HOSTEL_FEE_WORD.search(label):
        return "hostel_fee"
    if ROOMTYPE_GENERIC.search(label) and FEE_WORD.search(label):
        return "hostel_fee"
    return None


def _is_bare_amount(line: str) -> bool:
    """A table cell holding nothing but a figure -- the tell-tale of a column."""
    if len(line) > 34:
        return False
    m = MONEY_RE.match(line.strip())
    return bool(m and len(m.group(0)) >= len(line.strip()) - 2)


def _money_in(chunk: str) -> list[tuple[str, str, bool]]:
    """(verbatim, prefix_text_before_it, had_currency_marker) for each amount."""
    out = []
    for m in MONEY_RE.finditer(chunk):
        raw = m.group(0).strip()
        has_cur = bool(re.match(_CUR, raw, re.I)) or "/-" in raw.replace(" ", "")
        if not _plausible_money(raw, has_cur):
            continue
        out.append((raw, chunk[:m.start()], has_cur))
    return out


def _harvest_money(lines: list[str], sents: list[str]) -> dict[str, str]:
    """Pull labelled amounts, refusing anything whose label is ambiguous.

    Two association paths, both deliberately short-range:
      same line   'Hostel Fee: Rs. 45,000 per annum'   (the common case)
      label above 'Standard Room' / 'AIR COOLED' / 'Rs. 83000/-'  (real tables)
    A label is only used when it collects exactly ONE amount. lpu.in prints the
    same 'Standard Room' label against ten different amounts across five tables
    and two price columns; there is no honest way to pick one, so every one of
    them is dropped and that page's fees go to the LLM.
    """
    candidates: dict[str, set[str]] = {}
    order: list[str] = []
    poisoned: set[str] = set()

    def offer(key: str, value: str) -> None:
        if key not in candidates:
            candidates[key] = set()
            order.append(key)
        candidates[key].add(value)

    # Path 1: amount and its label share a line or a sentence.
    for chunk in list(lines) + list(sents):
        if PHONE_CTX.search(chunk):
            continue
        for raw, before, _cur in _money_in(chunk):
            label = before[-90:]
            key = _classify(label)
            if key:
                offer(key, raw)

    # Path 2: fee tables that survive tag-stripping as label-line / value-line.
    for i, ln in enumerate(lines):
        if len(ln) > 120 or PHONE_CTX.search(ln):
            continue
        found = _money_in(ln)
        if not found:
            continue
        raw, before, _cur = found[0]
        if _classify(before[-90:]):
            continue                       # already handled by path 1
        label_parts: list[str] = []
        blocked = False
        for j in range(i - 1, max(-1, i - 4), -1):
            prev = lines[j]
            if len(prev) > 90 or _money_in(prev):
                blocked = True             # another amount in between: unreliable
                break
            label_parts.insert(0, prev)
            if _classify(" ".join(label_parts)):
                break
        if blocked or not label_parts:
            continue
        label = " ".join(label_parts)
        key = _classify(label)
        if not key:
            continue
        if len(found) > 1:
            # A multi-column row ("early decision" vs "standard" pricing). The
            # label cannot be attached to any single figure.
            poisoned.add(key)
            continue
        # Same thing spread down the page instead of across it. lpu.in prints
        #   One Meal | Rs. 55000/- | Rs. 28500/- | Rs. 5500/-
        # as four consecutive lines; taking the first would publish the yearly
        # figure, the per-semester figure or the monthly one with equal
        # confidence and no way to tell which.
        # Strictly the NEXT line: in a normal two-column table the following
        # line is the next row's label, and only a stacked column puts a second
        # bare figure immediately underneath.
        if i + 1 < len(lines) and _is_bare_amount(lines[i + 1]):
            poisoned.add(key)
            continue
        offer(key, raw)

    facts: dict[str, str] = {}
    for key in order:
        vals = candidates[key]
        # More than one distinct amount under one label means the page states
        # several and we cannot tell which. Silently drop: a guessed hostel fee
        # is the exact failure this whole module exists to avoid.
        if len(vals) == 1 and key not in poisoned:
            facts[key] = next(iter(vals))
    return facts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _extract(text: str, url: str, college_name: str) -> dict | None:
    if not isinstance(text, str):
        return None
    text = text[:MAX_SCAN_CHARS]
    if len(text.strip()) < 60:
        return None

    lines = _lines(text)
    sents = _sentences(lines)

    # Sentences that are actually about accommodation AND actually say something.
    topical = [s for s in sents
               if HOSTEL_RE.search(s) and ASSERT_RE.search(s)
               and not FIRSTPERSON_RE.search(s)]
    if not topical:
        # A page can still be a fee table with no prose at all, so money is
        # checked before giving up -- but only money that names a hostel.
        money_only = _harvest_money(lines, sents)
        # A stray deposit or mess figure on a page with no accommodation prose is
        # not evidence about hostels -- an admissions page charging a prospectus
        # fee will produce one. Require a figure whose own label says hostel.
        if not any(k == "hostel_fee" or k.startswith(("room_rate_", "total_"))
                   for k in money_only) or not HOSTEL_RE.search(text):
            return None
        facts = money_only
        summary = _summarise(college_name, facts, None, [], [])
        return _finish(summary, facts, 0.6)

    joined = " ".join(topical)

    facts: dict[str, str] = {}

    # --- availability -------------------------------------------------------
    availability = None
    if any(NEGATIVE_RE.search(s) for s in topical) and \
            not any(SEPARATE_RE.search(s) for s in topical):
        neg = next(s for s in topical if NEGATIVE_RE.search(s))
        if re.search(r"hostel|accommodation|accomodation", neg, re.I):
            facts["hostel_available"] = "No hostel stated on this page"
            availability = "no"
    if availability is None:
        boys = bool(GENDER_BOYS.search(joined))
        girls = bool(GENDER_GIRLS.search(joined))
        if boys and girls:
            facts["hostel_available"] = ("Yes - separate hostels for boys and girls"
                                         if SEPARATE_RE.search(joined) else
                                         "Yes - hostels for boys and girls")
            facts["boys_hostel"] = "Available"
            facts["girls_hostel"] = "Available"
            availability = "both"
        elif boys:
            facts["hostel_available"] = "Yes - boys hostel"
            facts["boys_hostel"] = "Available"
            availability = "boys"
        elif girls:
            facts["hostel_available"] = "Yes - girls hostel"
            facts["girls_hostel"] = "Available"
            availability = "girls"
        else:
            facts["hostel_available"] = "Yes"
            availability = "yes"

    # --- room types ---------------------------------------------------------
    room_scope = " ".join(s for s in sents if CONTEXT_RE.search(s))
    types = [label for label, pat in ROOM_TYPES if pat.search(room_scope)]
    # "Single occupancy" is worth reporting; a lone "AC" without any room word
    # nearby is usually about a seminar hall, so require a room noun in scope.
    if types and not re.search(r"\broom|\bseater|\bsharing|\boccupancy|\bsuite|"
                               r"\bapartment|\bbed", room_scope, re.I):
        types = []
    if types:
        facts["room_types"] = ", ".join(types[:6])

    # --- capacity / room count ---------------------------------------------
    for s in topical:
        if NOT_HOSTEL_NOUN.search(s):
            continue
        if "hostel_capacity" not in facts:
            g = GENDER_CAPACITY_RE.search(s)
            if g and 1 <= int(g.group(1)) <= 20000 and 1 <= int(g.group(2)) <= 20000:
                facts["hostel_capacity"] = g.group(0)
            else:
                m = CAPACITY_RE.search(s)
                if m:
                    num = m.group(1) or m.group(3)
                    unit = (m.group(2) or m.group(4) or "students").lower()
                    if num and 1 <= int(num) <= 20000:
                        facts["hostel_capacity"] = f"{num} {unit}"
        m2 = ROOMCOUNT_RE.search(s)
        if m2 and "number_of_rooms" not in facts and 1 <= int(m2.group(1)) <= 5000:
            facts["number_of_rooms"] = m2.group(1)

    # --- mess ---------------------------------------------------------------
    mess_scope = " ".join(s for s in sents if MESS_WORD.search(s))
    mess_absent = bool(MESS_ABSENT.search(mess_scope))
    if mess_absent:
        facts["mess"] = "Mess facility not available"
    elif MESS_SEPARATE.search(mess_scope):
        facts["mess"] = "Mess charges are separate from hostel charges"
    elif MESS_INCLUDED.search(mess_scope):
        facts["mess"] = "Mess charges included in hostel charges"
    elif re.search(r"\bmess\b", joined, re.I):
        facts["mess"] = "Mess facility available"

    # --- facilities ---------------------------------------------------------
    # Scoped to accommodation sentences on purpose: gcggn.ac.in lists a gym, a
    # dispensary and Wi-Fi for the whole campus, none of it hostel-specific.
    fac_scope = " ".join(topical)
    facilities = [label for label, pat in FACILITIES if pat.search(fac_scope)]
    # christuniversity.in says "mess facility is not available in the residence
    # halls" -- the word "mess" is present, so the keyword alone would list it
    # as an amenity and contradict the very sentence it came from.
    if mess_absent:
        facilities = [f for f in facilities if f != "Mess/dining"]
    if facilities:
        facts["facilities"] = ", ".join(facilities[:8])

    # --- money --------------------------------------------------------------
    facts.update(_harvest_money(lines, sents))

    money_keys = [k for k in facts
                  if k.startswith(("hostel_fee", "room_rate_", "mess_charges",
                                   "security_deposit", "total_"))]

    # --- confidence ---------------------------------------------------------
    # Anchored to how explicitly the page states things, not to how much we got.
    if len(money_keys) >= 2:
        conf = 0.9
    elif money_keys:
        conf = 0.8
    elif availability == "no":
        conf = 0.55
    elif availability in ("both", "boys", "girls"):
        conf = 0.7 if (types or "hostel_capacity" in facts
                       or len(facilities) >= 2) else 0.65
    else:
        # Availability with no gender, no rate, no room type: a single passing
        # sentence. Real, but thin -- and thin is where over-claiming starts.
        conf = 0.6 if (types or "hostel_capacity" in facts
                       or len(facilities) >= 2) else 0.5
    if len(topical) == 1 and not money_keys and conf > 0.6:
        conf -= 0.05

    if conf < 0.5:
        return None
    return _finish(_summarise(college_name, facts, availability, types, facilities),
                   facts, conf)


def _summarise(college_name: str, facts: dict, availability: str | None,
               types: list[str], facilities: list[str]) -> str:
    """Build the card text from extracted VALUES, never from page prose.

    This is the guard against marketing copy: nothing a college wrote about
    itself can reach a student through this path, only figures it published.
    """
    name = (college_name or "").strip()
    if len(name) > 46 or not name:
        name = "The college"
    parts: list[str] = []

    avail = facts.get("hostel_available", "")
    if availability == "no":
        parts.append(f"{name} states no hostel on this page.")
    elif availability == "both":
        parts.append(f"{name} has hostels for boys and girls"
                     + (" separately." if "separate" in avail else "."))
    elif availability == "boys":
        parts.append(f"{name} has a boys hostel.")
    elif availability == "girls":
        parts.append(f"{name} has a girls hostel.")
    elif availability:
        parts.append(f"{name} provides hostel accommodation.")

    if facts.get("hostel_capacity"):
        parts.append(f"Capacity {facts['hostel_capacity']}.")
    if types:
        parts.append("Room types: " + ", ".join(types[:4]) + ".")

    money_bits = []
    for key, label in (("hostel_fee", "Hostel fee"),
                       ("total_accommodation_cost", "Total"),
                       ("mess_charges", "Mess"),
                       ("security_deposit", "Security deposit")):
        if facts.get(key):
            money_bits.append(f"{label} {facts[key]}")
    for key in sorted(k for k in facts if k.startswith("room_rate_")):
        if len(money_bits) >= 3:
            break
        money_bits.append(f"{key[10:].replace('_', ' ')} {facts[key]}")
    if money_bits:
        parts.append("; ".join(money_bits) + ".")

    mess = facts.get("mess")
    if mess and not facts.get("mess_charges"):
        parts.append(mess + ".")
    if facilities and len(" ".join(parts).split()) < 30:
        parts.append("Facilities: " + ", ".join(facilities[:4]) + ".")

    out = re.sub(r"\s+", " ", " ".join(parts))
    out = re.sub(r"\.\s*\.+", ".", out)          # "Rs 75,000/- p.a.." -> "p.a."
    words = out.split()
    if len(words) > 45:
        out = " ".join(words[:45]).rstrip(",;:") + "."
    return out


def _finish(summary: str, facts: dict, conf: float) -> dict | None:
    clean: dict[str, str] = {}
    for k, v in facts.items():
        if not k or v is None:
            continue
        s = re.sub(r"\s+", " ", str(v)).strip()
        if s and len(s) <= 200:
            clean[str(k)[:60]] = s
        if len(clean) >= MAX_FACTS:
            break
    if not clean or not summary:
        return None
    return {"summary": summary[:600], "facts": clean,
            "confidence": round(max(0.0, min(1.0, conf)), 2)}


def extract(text: str, url: str, college_name: str) -> dict | None:
    """Hostel/accommodation facts from one page, or None.

    Never raises: the sweep runs unattended across 10,000 third-party sites and
    a single malformed page must not take a batch down with it.
    """
    try:
        return _extract(text, url, college_name)
    except Exception:  # noqa: BLE001 - deliberate: None is always a valid answer
        return None


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

_NAV_ONLY = """Home
About Us
Facilities
Hostel
Library
Computer lab
Mess
Boy's Hostel
Hostel Details
Accommodation
Contact Us
Govt. Polytechnic Education Society, Manesar
NH-8, Near NSG & NBRC
Manesar (Gurgaon)- 122051
phone : 0124-2337243
HOSTEL
Quick Links
Last Updated On :25 May 2026, 4:45 PM
(c) 2017 Govt. Polytechnic Manesar"""

_NOTHING = """Accomodation
hidden
Accomodation
Yet we doesn't received any accomodation.
About Us
Bihar National College, popularly known as B. N. College was established in 1889
B. N. College, Near Gandhi Maidan, Ashok Rajpath, Patna- 800004, Bihar [India]
0612-22677619
info@bncollegepatna.com
Visitor Counter :
 350207"""

_OTHER_TOPIC = """Principal's Message
It gives me immense pleasure to welcome you to our institution, established in 1959.
Our college has 48 classrooms and a library with a seating capacity at 200.
The auditorium seats over 1000 people. Generators having a capacity of 250 KW.
The library has around 44,000 books.
Contact No : 8168674047
Establishment Year : 2014"""

_NUMBERS_TRAP = """Facilities
Contact us on 0124-2337243 or 8168674047 for admission in 2026-27.
Our campus at Manesar (Gurgaon)- 122051 is spread over 17363 sqm.
Access to a 1200-bed teaching hospital strengthens the ecosystem.
The fee structure for 2025-26 was notified on 15/06/2025.
Visitor Counter : 350207"""

_REAL_GCECHD = """Hostel
There are two hostels one each for boys and girls. There are two wardens and one Nurse. Approximately each hostel has 14 rooms equipped with spacious almirah. Four, Five students can stay in one room.
Fourteen furnished rooms in each hostel (Bed, Table Chairs, Fan etc.)
There is study room, common room and dining hall. In common room, there is T.V. and Music system. Library is also there in each hostel.
Geysers are available in the hostel bathrooms.
Washing and Ironing of clothes facility available in the hostel.
Sick room is also in the hostel. Hostel Nurse is on the premises to look after the resident students."""

_REAL_GCGGN = """Facilities
The institution is fully
under the CCTV surveillance. There are 5 girls Hostel with a capacity of 750
students. The institution has installed various RO systems to provide pure
water to the students.
The State-of-the-art auditorium with seating capacity of over 1000
act as a common ground for students. The institution is
having a Central library with a seating capacity at 200.
The College Library has a collection of around 44,000 books."""

_REAL_GPAAMWALA = """Facilities
Building :
The institution is spread over an area of 17363 sqm consisting of administrative
block and teaching block, 1 multipurpose hall, 1 hostel, 1 drawing Hall, 10
workshop, 08 laboratories with adequate machinery, equipment, furniture, etc.
Hostel
Institute has 1 boys hostel with the capacity of 33 seats. Mess and common room
facility is available in the hostel."""

_REAL_CHRIST = """Hostel & Dining
About Us
STUDENT ACCOMMODATION SERVICES has well-furnished residence halls for non-resident students. CHRIST (Deemed to be University) offers student accommodation services on the first-come-first-serve basis.
Please note that mess facility is not available in the residence halls. However, students can avail the facilities of cafeteria and eateries on campus.
Admission to the hostel is for one year only.
Dharmaram College Post, Hosur Road, Bengaluru - 560029, Karnataka, India
Tel: +91 80 4012 9100 / 9600"""

_REAL_AMITY = """Accommodation
Home Away from Home for 15,000 Students
Amity offers comprehensive hostel facilities for boys and girls separately within the University Campus.
Caring wardens and a vigilant security ensures a pleasant stay allowing students to focus on academics. The air-conditioned residential apartment suites consist of 4 single rooms with an attached bathroom and sitting lounge equipped with sofa, cable TV and refrigerator. Normal Non-A/C hostel rooms are also available.
I was staying in a hostel for the first time, but the caring wardens, homely food and the friendly atmosphere made me feel so much at home."""

_REAL_ICITMKG = """Facilities
- Well maintained and well equipped Hostels for both boys & girls separately and also has quaters for the Faculty and staffs inside the Campus."""

_FEE_TABLE = """Hostel Fee Structure 2025-26
Room Category
Charges
Single Occupancy AC Room
Rs. 1,20,000 per annum
Double Occupancy Non-AC Room
Rs. 75,000/- p.a.
Mess Charges
Rs. 4,500 per month
Security Deposit (Refundable)
Rs. 10,000/-
Hostel is available for boys and girls separately."""

_FEE_PROSE = """Hostel
The college provides hostel facility for outstation students. The hostel fee is
Rs 45,000 p.a. including mess charges. A refundable security deposit of Rs
10,000/- is payable at the time of admission. Rooms are on twin sharing basis
with Wi-Fi and laundry."""

_LAKH = """Accommodation
Hostel charges are 1.2 Lakh per year for an air-conditioned single room.
The mess charge is Rs.3,500 per month, payable separately."""

_AMBIGUOUS_TABLE = """Residential Fee Charges
4 Seater
Standard Room
AIR COOLED
Rs. 83000/-
 300 free electricity units (per academic session)
Rs. 93000/-
3 Seater
Standard Room
AIR COOLED
Rs. 98000/-
 350 free electricity units (per academic session)
Rs. 108000/-
Hostel and apartment style housing is available for students."""


def _selftest() -> None:
    import json
    import os
    import time

    cases = [
        ("nav-only hostel page (gpmanesar.ac.in)", _NAV_ONLY, None),
        ("page that denies content (bncollegepatna)", _NOTHING, None),
        ("different topic: principal message (gcggn nav)", _OTHER_TOPIC, None),
        ("phones/PINs/years/sqm only", _NUMBERS_TRAP, None),
        ("real: gcechd.ac.in/hostel.php", _REAL_GCECHD, "dict"),
        ("real: gcggn.ac.in facilities", _REAL_GCGGN, "dict"),
        ("real: gpaamwala.org.in facilities", _REAL_GPAAMWALA, "dict"),
        ("real: christuniversity.in hostel-and-dining", _REAL_CHRIST, "dict"),
        ("real: amity.edu/hostel.aspx", _REAL_AMITY, "dict"),
        ("real: icitmkg.in/facilities.html", _REAL_ICITMKG, "dict"),
        ("synthetic fee table", _FEE_TABLE, "dict"),
        ("fee in prose", _FEE_PROSE, "dict"),
        ("lakh notation", _LAKH, "dict"),
        ("real: lpu.in ambiguous multi-column table", _AMBIGUOUS_TABLE, "dict"),
        ("empty", "", None),
        ("whitespace", "   \n\n  \t ", None),
        ("garbage", "\x00\x01" * 200, None),
    ]

    failures = 0
    for label, body, want in cases:
        t0 = time.perf_counter()
        got = extract(body, "https://example.edu/hostel", "Test College")
        ms = (time.perf_counter() - t0) * 1000
        ok = (got is None) if want is None else isinstance(got, dict)
        failures += 0 if ok else 1
        print(f"\n{'PASS' if ok else 'FAIL'}  {label}   [{ms:.1f} ms]")
        if got is None:
            print("      -> None")
        else:
            print(f"      conf={got['confidence']}  "
                  f"words={len(got['summary'].split())}")
            print(f"      summary: {got['summary']}")
            print("      facts:   "
                  + json.dumps(got["facts"], ensure_ascii=False, indent=None))

    # Number parser -- the lakh grouping is where a naive int() gets it wrong.
    print("\n-- _to_number --")
    for raw, want in [("1,20,000", 120000.0), ("45,000", 45000.0),
                      ("Rs. 83000/-", 83000.0), ("1.2 Lakh", 120000.0),
                      ("1.2L", 120000.0), ("₹12,00,000", 1200000.0),
                      ("2 Crore", 20000000.0), ("45k", 45000.0),
                      ("12000", 12000.0), ("1,234,567", 1234567.0)]:
        got = _to_number(raw)
        ok = got is not None and abs(got - want) < 0.5
        failures += 0 if ok else 1
        print(f"  {'ok ' if ok else 'BAD'} {raw!r:>16} -> {got} (want {want})")

    print("\n-- _plausible_money rejections --")
    for raw, cur, want in [("2026", False, False), ("560029", False, False),
                           ("8168674047", False, False), ("2014", False, False),
                           ("Rs. 45,000", True, True), ("300", False, True),
                           ("₹1,20,000", True, True), ("17363", False, True),
                           ("Rs 1,20,000", True, True)]:
        got = _plausible_money(raw, cur)
        ok = got is want
        failures += 0 if ok else 1
        print(f"  {'ok ' if ok else 'BAD'} {raw!r:>14} currency={cur} -> {got}")

    # Real cached pages, if the harvesting scratchpad is still around.
    for d in (os.environ.get("MIVI_HOSTEL_PAGES", ""),):
        if d and os.path.isdir(d):
            print(f"\n-- cached real pages in {d} --")
            for fn in sorted(os.listdir(d)):
                if not fn.endswith(".txt"):
                    continue
                raw = open(os.path.join(d, fn), encoding="utf-8",
                           errors="replace").read()
                head, _, body = raw.partition("===")
                got = extract(body, head.strip(), "Test College")
                print(f"  {fn[:52]:<54} "
                      + ("None" if got is None
                         else f"conf={got['confidence']} {got['summary'][:90]}"))

    print(f"\n{'ALL PASS' if not failures else str(failures) + ' FAILURES'}")


if __name__ == "__main__":
    _selftest()
