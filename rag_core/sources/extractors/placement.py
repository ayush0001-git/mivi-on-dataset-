"""Deterministic placement-statistics extractor. No LLM, no network, stdlib only.

WHY THIS IS WORTH WRITING BY HAND
Placement pages on Indian college sites are overwhelmingly built from the same
three shapes, and all three are parseable:

  1. A "stat card" strip where the NUMBER is its own line and the LABEL is the
     next line(s).  Measured on real pages -- srhu.edu.in, cgccoe.org,
     chitkara.edu.in, lpu.in all use it:

         36 LPA                 800+                       Rs1.1 Cr
         Highest Package        Companies for Placement     Highest Package

  2. "Label : value" on one line, or the same claim inside a sentence.
  3. The NAAC self-study table ("Average percentage of placement of outgoing
     students during the last five years"), whose header row of years and value
     row of numbers survive tag-stripping as two runs of short lines.

WHAT THIS DELIBERATELY REFUSES
The failure mode that matters is not a missed figure, it is an invented one, so
every rule below is a filter rather than a guess:

  * A number is only ever read through a LABEL. There is no "biggest number on
    the page wins" path, which is what keeps phone numbers, PIN codes, visitor
    counters, affiliation years and course codes out.
  * "100% Placement Support" / "placement assistance" / "placement guarantee"
    are marketing, not a placement rate -- assistance is not placement. Real
    pages state exactly this (srhu.edu.in shows "100% / Placement Support"),
    and the LLM fallback stored it as a fact. This module drops it.
  * "Rs 10.23 Lakh / average salary of top 25% placed students" (lpu.in) is a
    subset average. Reported as "average package" it would mislead, so a
    subset qualifier vetoes the fact entirely.
  * Individual student news ("Shweta Ranjan secures a Rs 58 LPA offer from
    Google") is not an institutional statistic. Nothing here reads a package
    that is not labelled highest/average/median/lowest.
  * Self-declared rankings and "best placements in the region" are never read.

CALIBRATION
An explicit labelled table scores highest, a stat card slightly lower, a
sentence lower still -- a single undated sentence lands at 0.48 and is therefore
DISCARDED by the caller, which is the intended behaviour. An attached batch year
is worth a bump because an undated package figure is a much weaker claim.
"""
from __future__ import annotations

import re

__all__ = ["extract"]

# ---------------------------------------------------------------------------
# Indian money grammar
# ---------------------------------------------------------------------------
# Digit runs are matched as EITHER comma-grouped or plain, because Indian sites
# group in lakhs ("1,20,000") and a western-only pattern silently truncates
# those to "1". Never int() a raw token; _to_number() below is the only parser.
_NUM = r"(?:\d{1,3}(?:,\d{2,3})+|\d+)(?:\.\d+)?"
_CUR = r"(?:Rs\.?|INR|Rupees|US\$|USD|\$|₹)"
# "L" and "K" are accepted only in the strict single-token line form; they are
# too collision-prone to allow loose inside prose.
_UNIT_STRICT = r"(?:LPA|L\.P\.A\.?|LPa|lakhs?|lacs?|crores?|cr|CPA|K|L)"
_UNIT_PROSE = r"(?:LPA|L\.P\.A\.?|lakhs?|lacs?|crores?|cr)"
_PERIOD = (r"(?:p\.?\s?a\.?|per\s+annum|per\s+year|per\s+month|p\.?\s?m\.?"
           r"|/\s?(?:yr|year|annum|month|mo))")

# A line that is nothing but a figure. Anchored on both ends on purpose: this is
# what separates a stat card from body text, and what stops "1800-2003575"
# (a real toll-free number on cgccoe.org) or "630597 users" (tiut.ac.in's
# visitor counter) from ever being considered.
_VALUE_LINE = re.compile(
    rf"^\s*(?P<cur>{_CUR})?\s*(?P<num>{_NUM})\s*(?P<plus>\+)?\s*"
    rf"(?P<unit>%|{_UNIT_STRICT})?\s*(?P<plus2>\+)?\s*(?P<period>{_PERIOD})?\s*$",
    re.I)

# Money inside a sentence. Requires a currency marker OR a lakh/crore/LPA unit;
# a bare number in prose is never money.
_MONEY_PROSE = re.compile(
    rf"(?:(?P<cur>{_CUR})\s{{0,2}}(?P<num>{_NUM})\s{{0,2}}(?P<unit>{_UNIT_PROSE})?"
    rf"|(?P<num2>{_NUM})\s{{0,2}}(?P<unit2>{_UNIT_PROSE}))"
    rf"\b\s{{0,2}}(?P<period>{_PERIOD})?",
    re.I)

_PCT_PROSE = re.compile(rf"(?P<num>{_NUM})\s?%")

_UNIT_CANON = {
    "lpa": "LPA", "l.p.a": "LPA", "l.p.a.": "LPA", "lpa.": "LPA",
    "lakh": "Lakh", "lakhs": "Lakh", "lac": "Lakh", "lacs": "Lakh",
    "l": "Lakh", "crore": "Crore", "crores": "Crore", "cr": "Crore",
    "cr.": "Crore", "k": "K", "cpa": "CPA", "%": "%",
}


def _to_number(raw: str) -> float | None:
    """'1,20,000' -> 120000.0, '45.5' -> 45.5. Commas are grouping, never decimal."""
    try:
        return float(raw.replace(",", ""))
    except (TypeError, ValueError, AttributeError):
        return None


def _canon_money(cur: str | None, num: str, unit: str | None,
                 period: str | None) -> tuple[str, float, str] | None:
    """Rebuild the amount with a canonical UNIT spelling and the number untouched.

    Returns (display, magnitude_in_rupees_for_sanity_only, unit_key) or None if
    the figure is not a plausible salary. The magnitude exists purely to reject
    nonsense (a "45 crore average package", a 10-digit phone number); it is
    never shown and never written back into facts.
    """
    n = _to_number(num)
    if n is None or n <= 0:
        return None
    u = _UNIT_CANON.get((unit or "").lower().strip()) if unit else None

    if u in ("LPA", "Lakh", "CPA"):
        # Plausible annual package band in lakhs. Below 0.3L is a stipend or a
        # typo, above 999L should have been written in crore.
        if not (0.3 <= n <= 999):
            return None
        rupees = n * 100_000
    elif u == "Crore":
        if not (0.01 <= n <= 100):
            return None
        rupees = n * 10_000_000
    elif u == "K":
        if not (10 <= n <= 100_000) or not cur:
            return None
        rupees = n * 1_000
    else:
        # Bare rupees: needs a currency marker, and a floor high enough that a
        # PIN code (812007), a phone number or a fee instalment cannot pass.
        if not cur:
            return None
        if not (10_000 <= n <= 100_000_000):
            return None
        rupees = n
        u = ""

    cur_disp = (cur or "").strip()
    if cur_disp.lower() in ("rupees",):
        cur_disp = "Rs"
    gap = "" if (not cur_disp or cur_disp[-1] in "₹$") else " "
    disp = f"{cur_disp}{gap}{num.strip()}"
    if u:
        disp += f" {u}"
    if period:
        disp += f" {period.strip()}"
    return disp[:48], rupees, (u or "INR")


# ---------------------------------------------------------------------------
# Label vocabulary
# ---------------------------------------------------------------------------
# Order is priority: "highest offer" must beat the generic "offers" counter.
_LABELS: tuple[tuple[str, str, str], ...] = (
    ("highest_package",
     r"(?:highest|top|maximum|max|best)\s+(?:ever\s+|annual\s+|salary\s+|international\s+"
     r"|domestic\s+|on[-\s]?campus\s+)?(?:package|salary|ctc|pay\s*package|pay|"
     r"compensation|offer\b|placement\s+package)", "money"),
    ("average_package",
     r"(?:average|avg\.?|mean)\s+(?:annual\s+|starting\s+)?"
     r"(?:package|salary|ctc|pay|compensation)", "money"),
    ("median_package",
     r"median\s+(?:annual\s+)?(?:package|salary|ctc|pay|compensation)", "money"),
    ("lowest_package",
     r"(?:lowest|minimum|min\.?)\s+(?:annual\s+)?"
     r"(?:package|salary|ctc|pay|compensation)", "money"),
    ("recruiters",
     r"(?:campus\s+|top\s+|total\s+|no\.?\s*of\s+|number\s+of\s+|placement\s+)?"
     r"recruiters?\b"
     r"|recruiting\s+(?:compan|organis|organiz|partner|firm)"
     r"|compan(?:y|ies)\s+(?:for\s+placement|visited|visit|recruited|recruiting|"
     r"participat|hiring)"
     r"|(?:companies|organisations|organizations|firms|mncs)\s+for\s+placement"
     r"|(?:hiring|placement|recruitment)\s+partners", "count"),
    ("students_placed",
     r"(?:students?\s+placed|placed\s+students?"
     r"|no\.?\s*of\s+students?\s+placed|number\s+of\s+students?\s+placed"
     r"|students?\s+(?:got|secured|received)\s+(?:placement|jobs?|offers?))", "pct_or_count"),
    # Before the percentage label: "Placement Offers" is a count, and a bare
    # "placement" pattern placed first would swallow it (measured on
    # cgccoe.org's "8500+ / Placement Offers" card).
    # No bare "offer" alternative: "we offer 100% placement assistance" would
    # otherwise read as 100 job offers, which is exactly the marketing-to-fact
    # laundering this module exists to prevent. Plural only, for the same
    # reason -- mitwpu.edu.in's admission funnel has a step numbered 10 captioned
    # "Receive Final Offer Letter", and singular "offer" made that 10 offers.
    ("placement_offers",
     r"(?:placement|job|campus|final|total|super\s*dream|dream)\s+offers\b"
     r"|offers\s+(?:made|received|bagged|secured|extended)"
     r"|no\.?\s*of\s+offers\b|number\s+of\s+offers\b", "count"),
    ("placement_percentage",
     r"(?:placement\s*(?:%|percentage|percent|rate|ratio)"
     r"|(?:percentage|percent|%)\s+of\s+(?:students?\s+)?place(?:d|ment)"
     r"|placement\s+of\s+(?:outgoing\s+)?students?)", "pct"),
    # Last resort: a bare "Placement" caption only ever yields a percentage,
    # never a headcount, because "800 / Placements" is unreadable.
    ("placement_percentage", r"placements?\b", "pct"),
)

_LABEL_RES = tuple((k, re.compile(p, re.I), kind) for k, p, kind in _LABELS)

# A subset/comparison qualifier makes the headline reading of the figure wrong.
# "average salary of top 25% placed students" (lpu.in) is the live example.
_QUALIFIER_VETO = re.compile(
    r"\btop\s*\d{1,3}\s*(?:%|percent)"
    r"|\btop\s+\d{1,4}\s+(?:students?|placements?|offers?|recruits?)"
    r"|\b(?:market|industry|national|global)\s+average"
    r"|\b(?:expected|projected|estimated|anticipated|likely|targeted|upto\s+expectation)"
    r"|\b(?:internship|stipend|apprentice)"
    r"|\b(?:fee|fees|tuition|scholarship|loan|refund|deposit|donation)"
    r"|\bhigher\s+salary\b", re.I)

# Marketing dressed as a placement rate. "Assistance" is not placement.
_PCT_VETO = re.compile(
    r"\b(?:assistance|assistanc\w*|assist\b|support|guarantee\w*|assured|assurance"
    r"|help|aid\b|opportunit\w*|training|satisfaction|record|eligib\w*|readiness"
    r"|potential|scope|commitment)\b", re.I)

# "6000+ offers from Fortune 500 companies" (lpu.in) counts a slice, not the
# total; an "offer letter" is an admission document, not a placement.
_OFFERS_VETO = re.compile(
    r"\boffers?\s+from\b|\boffers?\s+(?:letter|guide)|\bletter\s+of\s+offer\b"
    r"|\bprovisional\s+offer|\badmission\s+offer", re.I)

# Package labels that describe a slice of the cohort deserve their own key
# rather than being reported as the headline figure: lpu.in states a domestic
# high of Rs 62.72 Lakhs on the same page as an international CTC above Rs 3 Cr.
_SCOPE_KEYS = (
    ("international", r"\b(?:international|overseas|abroad|foreign)\b"),
    ("domestic", r"\b(?:domestic|national|in[-\s]?country)\b"),
    ("on_campus", r"\bon[-\s]?campus\b"),
)
_SCOPE_RES = tuple((n, re.compile(p, re.I)) for n, p in _SCOPE_KEYS)


def _refine_key(key: str, label_text: str) -> str:
    """Only the words INSIDE the label may narrow the key.

    Deliberately not the surrounding caption: chitkara.edu.in captions its
    headline figure "Highest Package offered to a student by global giant
    Amazon", and reading "global" there would have relabelled a plain highest
    package as an international one.
    """
    if not key.endswith("_package"):
        return key
    for name, rx in _SCOPE_RES:
        if rx.search(label_text):
            return key[:-len("_package")] + f"_{name}_package"
    return key


_PLACEMENT_WORD = re.compile(
    r"\b(?:placement|placements|placed|recruiter|recruiters|recruiting|recruitment"
    r"|campus\s+drive|campus\s+selection|package|ctc|lpa)\b", re.I)

# Faculty-vacancy pages get classified as "placement" by the crawler because the
# nav link says "Career". tiut.ac.in/career.php and manipal.edu careers are both
# real examples in the queue.
_VACANCY_WORD = re.compile(
    r"\b(?:applications?\s+are\s+invited|walk[-\s]?in\s+interview|vacanc\w+"
    r"|current\s+openings?|job\s+openings?|assistant\s+professor|associate\s+professor"
    r"|apply\s+for\s+the\s+post|recruitment\s+of\s+faculty|non[-\s]?teaching\s+post)\b",
    re.I)

_ACAD_YEAR = re.compile(r"\b(20[0-3]\d)\s*[-‐-―/]\s*(20[0-3]\d|[0-3]\d)\b")
# A bare year only counts as a batch when the page says it is one. Without this
# lpu.in's "National Employability Award 2025", two lines from a package figure,
# became the batch that package supposedly describes.
_BATCH_CTX = re.compile(
    r"(?:batch|class|session|placements?|drive|placed)\s*(?:of\s+|in\s+|for\s+|:\s*)?"
    r"(20[0-3]\d)\b"
    r"|\b(20[0-3]\d)\s*(?:batch|graduating|passing[-\s]?out|pass[-\s]?out|session)",
    re.I)

# Named recruiters. A closed vocabulary is the only way to name companies without
# inventing them: unknown names are simply not reported, which costs recall on
# regional employers and costs nothing in correctness.
_RECRUITERS = (
    "TCS", "Tata Consultancy Services", "Infosys", "Wipro", "Accenture", "Cognizant",
    "Capgemini", "HCL", "Tech Mahindra", "IBM", "Microsoft", "Amazon", "Google",
    "Deloitte", "KPMG", "PwC", "Adobe", "Oracle", "SAP", "Bosch", "Siemens",
    "Larsen & Toubro", "Reliance", "Flipkart", "Paytm", "Zomato", "Swiggy",
    "Samsung", "Maruti Suzuki", "Mahindra", "ITC", "HDFC", "ICICI", "Axis Bank",
    "Kotak", "Bajaj", "JP Morgan", "Goldman Sachs", "Morgan Stanley", "Nvidia",
    "Intel", "Qualcomm", "Cisco", "Dell", "Hexaware", "Mindtree", "LTIMindtree",
    "Mphasis", "Zoho", "Persistent Systems", "Virtusa", "Genpact", "Concentrix",
    "Vedanta", "Tata Steel", "JSW", "Ashok Leyland", "Hero MotoCorp", "TVS",
    "Asian Paints", "Nestle", "Nestlé", "Britannia", "Dabur", "Godrej",
    "Cipla", "Sun Pharma", "Biocon", "Apollo Hospitals", "Fortis", "Max Healthcare",
    "Byju", "Unacademy", "Ola", "Uber", "Myntra", "Nykaa", "Razorpay", "Juspay",
    "Salesforce", "VMware", "Philips", "Honeywell", "Schneider Electric", "ABB",
    "Vodafone", "Airtel", "Jio", "Amdocs", "Sopra Steria", "NTT Data", "Atos",
    "EXL", "WNS", "Wells Fargo", "Barclays", "HSBC", "Standard Chartered",
    "American Express", "Aditya Birla", "Berger Paints", "Hindustan Unilever",
    "Marico", "Emami", "Havells", "Voltas", "Blue Star", "Thermax", "Cummins",
)
_RECRUITER_RE = re.compile(
    r"(?<![A-Za-z0-9])(" + "|".join(re.escape(n) for n in _RECRUITERS) +
    r")(?![A-Za-z0-9])", re.I)
_RECRUITER_CANON = {n.lower(): n for n in _RECRUITERS}

# A famous name on a placement page is not automatically a recruiter, so the
# line it sits on has to say so. Measured on chitkara.edu.in, which names
# Microsoft in "CSE in AI & ML with Microsoft" and WNS in "WNS Centre of
# Excellence" -- both academic partnerships. Without these two filters the
# extractor reported them as companies that hire the college's graduates.
_RECRUITER_CTX = re.compile(
    r"\b(?:recruit\w*|placements?|placed|hiring|hired|employers?|campus\s+drive"
    r"|top\s+compan\w+|job\s+offers?|selected\s+in)\b", re.I)
_PARTNER_VETO = re.compile(
    r"(?:centre|center)\s+of\s+excellence"
    r"|\bin\s+(?:association|collaboration|partnership)\s+with"
    r"|\bpowered\s+by\b|\bmous?\b|\btie[-\s]?ups?\b|\bcurriculum\b"
    r"|\bcertification\b|\bacadem(?:y|ia)\b|\blab\b"
    r"|\b(?:B\.?\s?Tech|BBA|MBA|M\.?\s?Tech|B\.?\s?Sc|BCA|MCA|B\.?\s?Com)\b.{0,60}?\bwith\b",
    re.I)

# Evidence classes, strongest first. See module docstring for the reasoning.
_BASE_CONF = {"table": 0.72, "card": 0.62, "inline": 0.62, "prose": 0.48, "names": 0.42}
_CLASS_RANK = {"table": 4, "card": 3, "inline": 3, "prose": 1, "names": 0}

# Generous, because unlike the LLM path this extractor has no per-page cost and
# stat blocks routinely sit below the 6,000-char budget the LLM is given --
# chitkara.edu.in's "Rs1.1 Cr / Highest Package" card is at offset ~20,000.
# Still bounded so a pathological page cannot turn into an unbounded scan.
_MAX_LINES = 4000
_WINDOW_LINES = 3          # how far a stat card's caption may wrap
_WINDOW_CHARS = 90


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _match_label(window: str) -> tuple[str, str, str] | None:
    """(key, kind, matched_text) for the first vocabulary hit in a caption."""
    head = window[:_WINDOW_CHARS]
    for key, rx, kind in _LABEL_RES:
        m = rx.search(head)
        if m is not None:
            return key, kind, m.group(0)
    return None


def _veto(key: str, kind: str, window: str) -> bool:
    if _QUALIFIER_VETO.search(window):
        return True
    if key == "placement_offers" and _OFFERS_VETO.search(window):
        return True
    if kind in ("pct", "pct_or_count") and _PCT_VETO.search(window):
        return True
    return False


def _count_ok(key: str, n: float, tok_has_plus: bool, tok_raw: str) -> bool:
    """Sanity bands for headcount-style figures, plus a year trap."""
    if n <= 0 or n != int(n):
        return False
    # A bare four-digit integer with no '+', no comma and no unit is far more
    # likely a year than a count -- "2024 / Placements" must not become 2024
    # recruiters.
    if (not tok_has_plus and "," not in tok_raw and 1900 <= n <= 2035
            and len(tok_raw.strip()) == 4):
        return False
    # A zero-padded figure is a list index, a step number or a slide counter --
    # nobody writes "06 recruiters". Found by shuffling real pages and looking
    # at what accidental pairings survived.
    raw = tok_raw.strip()
    if len(raw) > 1 and raw[0] == "0":
        return False
    # Floors of 5: a single-digit "statistic" next to a caption is far more
    # often a step number, a list index or a slide counter than a real figure.
    if key == "recruiters":
        return 5 <= n <= 100_000
    if key == "placement_offers":
        return 5 <= n <= 2_000_000
    if key == "students_placed":
        return 5 <= n <= 500_000
    return False


def _caption(lines: list[str], i: int, forward: bool) -> str:
    """Join the caption lines that belong to the stat on line i.

    Stops at the next figure-only line so that a card strip like
    "36 LPA / Highest Package / 400+ / Campus Recruiters" never lets one card's
    caption bleed into the next card's number.
    """
    out: list[str] = []
    step = 1 if forward else -1
    j = i + step
    while 0 <= j < len(lines) and len(out) < _WINDOW_LINES:
        ln = lines[j].strip()
        if not ln or _VALUE_LINE.match(ln) or len(ln) > 70:
            break
        out.append(ln)
        if sum(len(x) for x in out) > _WINDOW_CHARS:
            break
        j += step
    if not forward:
        out.reverse()
    return " ".join(out).strip(" :–-—|•·")


def _card_direction(lines: list[str], value_idx: list[int]) -> bool:
    """True if captions follow their figure (the dominant Indian layout).

    Decided per page rather than per card: on srhu.edu.in reading backwards
    would pair "Highest Package" with the NEXT card's "400+" and report a
    highest package of 400. Choosing one direction for the whole page removes
    that failure mode; ties go forward because forward is what the card strips
    on every page measured actually use.
    """
    fwd = bwd = 0
    for i in value_idx:
        if _PLACEMENT_WORD.search(_caption(lines, i, True)):
            fwd += 1
        if _PLACEMENT_WORD.search(_caption(lines, i, False)):
            bwd += 1
    return fwd >= bwd


def _record(facts: dict, meta: dict, key: str, disp: str, cls: str,
            line: int, mag: float | None) -> None:
    """Keep the strongest evidence for a key; break ties the honest way."""
    old = meta.get(key)
    if old is not None:
        if _CLASS_RANK[cls] < _CLASS_RANK[old["cls"]]:
            return
        if _CLASS_RANK[cls] == _CLASS_RANK[old["cls"]]:
            # Same strength, different reading. "highest" is a max by
            # definition, so take the larger; anything else keeps the first
            # figure on the page, which is the headline one.
            if not (key.startswith("highest") and mag is not None
                    and old.get("mag") is not None and mag > old["mag"]):
                return
    facts[key] = disp
    meta[key] = {"cls": cls, "line": line, "mag": mag}


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def _scan_cards(lines: list[str], facts: dict, meta: dict) -> None:
    value_idx = [i for i, ln in enumerate(lines)
                 if ln.strip() and len(ln) <= 26 and _VALUE_LINE.match(ln)]
    if not value_idx:
        return
    forward = _card_direction(lines, value_idx)
    for i in value_idx:
        m = _VALUE_LINE.match(lines[i])
        if not m:
            continue
        cap = _caption(lines, i, forward)
        if not cap:
            continue
        hit = _match_label(cap)
        if hit is None:
            continue
        key, kind, label_text = hit
        if _veto(key, kind, cap):
            continue
        _apply_token(m, _refine_key(key, label_text), kind, facts, meta, "card", i)


def _apply_token(m: re.Match, key: str, kind: str, facts: dict, meta: dict,
                 cls: str, line: int) -> None:
    """Turn one matched figure + resolved label into a fact, or drop it."""
    num = m.group("num")
    unit = (m.groupdict().get("unit") or "")
    cur = m.groupdict().get("cur")
    period = m.groupdict().get("period")
    has_plus = bool(m.groupdict().get("plus") or m.groupdict().get("plus2"))
    n = _to_number(num)
    if n is None:
        return

    if unit == "%":
        if kind not in ("pct", "pct_or_count") or not (0 <= n <= 100):
            return
        if key == "students_placed":
            key = "placement_percentage"
        _record(facts, meta, key, f"{num}%", cls, line, n)
        return

    if kind == "money":
        money = _canon_money(cur, num, unit or None, period)
        if money is None:
            return
        disp, mag, _ = money
        _record(facts, meta, key, disp, cls, line, mag)
        return

    if kind in ("count", "pct_or_count"):
        if unit or cur:
            return          # "36 LPA recruiters" is not a headcount
        if kind == "pct_or_count":
            key = "students_placed"
        if not _count_ok(key, n, has_plus, num):
            return
        _record(facts, meta, key, num + ("+" if has_plus else ""), cls, line, n)


def _scan_lines(lines: list[str], facts: dict, meta: dict) -> None:
    """Label and figure on the SAME line: 'Highest Package: Rs 45 LPA' and the
    looser sentence form 'the highest package offered was Rs 45 LPA'.

    Every label on the line is tried, not just the first, because a stripped
    table row often collapses several columns onto one line
    ("Highest 24 LPA Average 6.5 LPA Median 5.2 LPA"). Each label's search span
    stops at the next label so columns cannot borrow each other's figures.
    """
    for i, raw in enumerate(lines):
        ln = raw.strip()
        if not (12 <= len(ln) <= 400) or not _PLACEMENT_WORD.search(ln):
            continue
        hits = []
        for prio, (key, rx, kind) in enumerate(_LABEL_RES):
            for lm in rx.finditer(ln):
                # Longest span at a given position wins, then vocabulary
                # priority: "Placement Percentage" must beat the bare
                # "Placement" fallback that starts at the same offset.
                hits.append((lm.start(), -(lm.end() - lm.start()), prio,
                             lm.end(), key, kind))
                if len(hits) > 16:
                    break
        if not hits:
            continue
        hits.sort()
        bounds = sorted({h[0] for h in hits})
        used: list[tuple[int, int]] = []
        for s, _neg, _prio, e, key, kind in hits:
            if any(s < ue and e > us for us, ue in used):
                continue        # overlapping vocabulary hit, keep the first
            used.append((s, e))
            nxt = next((b for b in bounds if b >= e), len(ln))
            tail = ln[e:min(nxt, e + 60)]
            prev_end = max((ue for us, ue in used[:-1]), default=0)
            head = ln[max(prev_end, s - 40):s]
            # A structured "label : value" is stronger evidence than the same
            # claim buried in a sentence, so the two are scored differently.
            structured = re.match(r"^\s*[:\-–—=|]\s*\S", tail) is not None
            window = ln[max(0, s - 60):e + 60]
            if _veto(key, kind, window):
                continue
            for side, txt in (("tail", tail), ("head", head)):
                if not txt.strip():
                    continue
                got = _first_figure(txt, kind, side)
                if got is None:
                    continue
                cls = "inline" if structured and side == "tail" else "prose"
                _apply_token(got, _refine_key(key, ln[s:e]), kind, facts, meta, cls, i)
                break


_GLUE = " :-=|–—.,\t"
_COUNT_PROSE = re.compile(rf"(?:over|more\s+than|above|around|nearly)?\s*({_NUM})\s*(\+)?",
                          re.I)


def _first_figure(txt: str, kind: str, side: str = "tail"):
    """First figure in a short span, wrapped to look like a _VALUE_LINE match.

    `side` says where the label sits: on the tail side the figure must butt up
    against the label's right edge, on the head side against its left. Without
    that anchor "recruiters visited the campus during the 2024 session" reads
    as 2024 recruiters.
    """
    def pick(rx, grp: int):
        """The candidate nearest the label, with nothing but glue in between."""
        ms = list(rx.finditer(txt))
        if not ms:
            return None
        m = ms[0] if side == "tail" else ms[-1]
        gap = txt[:m.start(grp)] if side == "tail" else txt[m.end():]
        if gap.strip(_GLUE) != "":
            return None
        # Reject a fragment of a larger token: the "24" of "2023-24" is not a
        # count of anything.
        i = m.start(grp)
        if i > 0 and txt[i - 1] in "-–—/.:0123456789":
            return None
        return m

    if kind in ("pct", "pct_or_count"):
        pm = pick(_PCT_PROSE, "num")
        if pm is not None:
            return _FakeMatch(num=pm.group("num"), unit="%")
    if kind == "money":
        # Money is allowed a few words of connective tissue ("the highest
        # package offered was Rs 45 LPA") because the currency or lakh/crore
        # unit is itself the evidence that the figure is a salary.
        mm = _MONEY_PROSE.search(txt) if side == "tail" else None
        if mm is None and side == "head":
            got = list(_MONEY_PROSE.finditer(txt))
            mm = got[-1] if got else None
        if mm is None:
            return None
        return _FakeMatch(num=mm.group("num") or mm.group("num2"),
                          unit=mm.group("unit") or mm.group("unit2") or "",
                          cur=mm.group("cur"), period=mm.group("period"))
    if kind in ("count", "pct_or_count"):
        cm = pick(_COUNT_PROSE, 1)
        if cm is None:
            return None
        # A figure wearing a percent sign is never a headcount.
        if txt[cm.end():cm.end() + 1] == "%" or "%" in cm.group(0):
            return None
        return _FakeMatch(num=cm.group(1), plus=cm.group(2))
    return None


class _FakeMatch:
    """Uniform accessor so prose hits and line hits share _apply_token."""

    __slots__ = ("_d",)

    def __init__(self, **kw):
        self._d = {"num": None, "unit": "", "cur": None, "period": None,
                   "plus": None, "plus2": None}
        self._d.update(kw)

    def group(self, k):
        return self._d.get(k)

    def groupdict(self):
        return dict(self._d)


_TABLE_ANCHOR = re.compile(
    r"(?:average\s+)?(?:percentage|%|number|no\.?)\s+of\s+"
    r"(?:students?\s+)?(?:placement|placed)"
    r"|placement\s+of\s+outgoing\s+students"
    r"|students?\s+placed\s+during", re.I)
_YEAR_CELL = re.compile(r"^\s*(20[0-3]\d)\s*[-‐-―/]\s*(20[0-3]\d|[0-3]\d)\s*$")
_NUM_CELL = re.compile(rf"^\s*({_NUM})\s*%?\s*$")


def _scan_naac_table(lines: list[str], facts: dict, meta: dict) -> str | None:
    """The NAAC five-year placement row, which tag-stripping flattens into a run
    of year cells followed by a run of value cells.

    Anchored on the criterion's standard wording, so the sibling tables on the
    same page ("students benefited by vocational education and training",
    "linkages for student exchange") cannot be picked up by accident.
    """
    latest = None
    for i, raw in enumerate(lines):
        if not _TABLE_ANCHOR.search(raw):
            continue
        is_pct = bool(re.search(r"percentage|%", raw, re.I))
        years: list[str] = []
        vals: list[str] = []
        for j in range(i + 1, min(i + 26, len(lines))):
            cell = lines[j].strip()
            if not cell:
                continue
            ym = _YEAR_CELL.match(cell)
            if ym and not vals:
                years.append(f"{ym.group(1)}-{ym.group(2)}")
                continue
            nm = _NUM_CELL.match(cell)
            if nm and years:
                vals.append(nm.group(1))
                if len(vals) == len(years):
                    break
                continue
            if years and vals:
                break
            if years and not vals and not nm:
                break
        if len(years) < 2 or len(years) != len(vals):
            continue
        for y, v in zip(years, vals):
            n = _to_number(v)
            if n is None:
                continue
            if is_pct:
                if not (0 <= n <= 100):
                    continue
                key = f"placement_percentage_{y.replace('-', '_')}"
                disp = f"{v}%"
            else:
                if not (0 <= n <= 500_000) or n != int(n):
                    continue
                key = f"students_placed_{y.replace('-', '_')}"
                disp = v
            _record(facts, meta, key, disp, "table", i, n)
        latest = max(years)
    return latest


def _find_batch(lines: list[str], stat_lines: list[int]) -> str | None:
    """The batch a figure describes, or None when the page is ambiguous.

    Ambiguity is resolved by silence rather than by picking one: cgccoe.org
    lists five placement-record ranges in its nav, and guessing which one the
    headline numbers belong to would be inventing a fact.
    """
    near, topical = set(), set()
    for i, ln in enumerate(lines):
        for m in _ACAD_YEAR.finditer(ln):
            y2 = m.group(2)
            tag = f"{m.group(1)}-{y2}"       # keep the page's own "2024-25" form
            if any(abs(i - s) <= 6 for s in stat_lines):
                near.add(tag)
            elif _PLACEMENT_WORD.search(ln):
                topical.add(tag)
    if len(near) == 1:
        return next(iter(near))
    if near:
        return None                          # two ranges beside the figures: ambiguous
    if len(topical) == 1:
        return next(iter(topical))
    if topical:
        return None
    plain = set()
    for i, ln in enumerate(lines):
        if not any(abs(i - s) <= 4 for s in stat_lines):
            continue
        for m in _BATCH_CTX.finditer(ln):
            plain.add(m.group(1) or m.group(2))
    return next(iter(plain)) if len(plain) == 1 else None


def _find_recruiters(lines: list[str]) -> list[str]:
    seen: list[str] = []
    for ln in lines:
        if len(ln) > 2000 or not _RECRUITER_CTX.search(ln):
            continue
        if _PARTNER_VETO.search(ln):
            continue
        for m in _RECRUITER_RE.finditer(ln):
            canon = _RECRUITER_CANON.get(m.group(1).lower(), m.group(1))
            if canon not in seen:
                seen.append(canon)
            if len(seen) >= 12:
                return seen
    return seen


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

_PHRASE = (
    ("placement_percentage", "{v} of students placed"),
    ("students_placed", "{v} students placed"),
    ("recruiters", "{v} recruiters"),
    ("placement_offers", "{v} placement offers"),
)
_PKG_ORDER = ("highest", "average", "median", "lowest")


def _fact_kind(key: str) -> str:
    base = key.split("_20")[0]
    return base.split("_")[0] + "_package" if base.endswith("_package") else base


def _package_bits(facts: dict) -> list[str]:
    """'highest_domestic_package' -> 'highest domestic package Rs 62.72 Lakh'."""
    keys = [k for k in facts if k.endswith("_package")]
    keys.sort(key=lambda k: (_PKG_ORDER.index(k.split("_")[0])
                             if k.split("_")[0] in _PKG_ORDER else 9, k))
    out = []
    for k in keys:
        words = k[:-len("_package")].replace("on_campus", "on-campus").split("_")
        out.append(f"{' '.join(words)} package {facts[k]}")
    return out


def _summarise(college: str, facts: dict, batch: str | None) -> str:
    who = (college or "").strip()[:60] or "The college"
    bits = _package_bits(facts)
    bits += [tmpl.format(v=facts[k]) for k, tmpl in _PHRASE if k in facts]
    dated = False
    for prefix, noun in (("placement_percentage_", "of outgoing students placed"),
                         ("students_placed_", "students placed")):
        years = sorted(k[len(prefix):].replace("_", "-")
                       for k in facts if k.startswith(prefix))
        if not years:
            continue
        latest = years[-1]
        val = facts[prefix + latest.replace("-", "_")]
        span = f", with figures published for {len(years)} years" if len(years) > 1 else ""
        bits.append(f"{val} {noun} in {latest}{span}")
        dated = True
    if not bits:
        names = facts.get("recruiters_mentioned", "")
        if names:
            return f"{who}'s placement page lists recruiters including {names}."[:600]
        return ""
    body = ", ".join(bits[:4])
    tail = "" if dated else (f" for {batch}" if batch else "")
    names = facts.get("recruiters_mentioned")
    extra = ""
    if names and len(bits) <= 2:
        extra = f" Recruiters listed include {', '.join(names.split(', ')[:4])}."
    out = f"{who} reports {body}{tail} on its placement page.{extra}"
    words = out.split()
    if len(words) > 45:
        out = " ".join(words[:45]).rstrip(" ,;") + "."
    return out[:600]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def extract(text: str, url: str, college_name: str) -> dict | None:
    """Placement statistics from one page of visible text, or None.

    None is the expected answer for nav pages, faculty-vacancy pages and
    placement pages that make only qualitative claims -- which is most of them.
    """
    try:
        return _extract(text, url, college_name)
    except Exception:          # noqa: BLE001 - one broken page must never stop a sweep
        return None


def _extract(text: str, url: str, college_name: str) -> dict | None:
    if not text or len(text) < 120:
        return None

    lines = [ln.strip() for ln in text.split("\n")][:_MAX_LINES]
    body = "\n".join(lines)

    # Topic gate. Costs nothing and removes the whole class of pages the crawler
    # mislabels through a "Career" nav link -- tiut.ac.in/career.php and
    # manipal.edu's vacancy pages contain the word "placement" zero times.
    signals = _PLACEMENT_WORD.findall(body)
    if len(signals) < 2:
        return None
    if len(signals) < 3 and len({s.lower() for s in signals}) < 2:
        return None
    # A faculty-vacancy page can still mention placements in its nav; require
    # the page to be substantially about student placement before trusting it.
    if _VACANCY_WORD.search(body) and len(signals) < 6:
        return None

    facts: dict[str, str] = {}
    meta: dict[str, dict] = {}

    table_year = _scan_naac_table(lines, facts, meta)
    _scan_cards(lines, facts, meta)
    _scan_lines(lines, facts, meta)

    stat_lines = [m["line"] for m in meta.values()]
    batch = table_year if table_year else _find_batch(lines, stat_lines)

    names = _find_recruiters(lines)
    if len(names) >= 4:
        facts["recruiters_mentioned"] = ", ".join(names)

    numeric = {k: v for k, v in facts.items() if k != "recruiters_mentioned"}
    if not numeric and "recruiters_mentioned" not in facts:
        return None

    # ---- confidence -------------------------------------------------------
    if numeric:
        best = max((meta[k]["cls"] for k in numeric), key=lambda c: _CLASS_RANK[c])
        conf = _BASE_CONF[best]
        # Distinct fact TYPES, not per-year columns and not scope variants: a
        # five-year NAAC table is one piece of evidence stated five times, and
        # highest/highest-domestic are one claim, not two.
        kinds = {_fact_kind(k) for k in numeric}
        conf += min(0.24, 0.08 * (len(kinds) - 1))
        if "recruiters_mentioned" in facts:
            conf += 0.03
        if batch:
            conf += 0.06
    else:
        conf = 0.52 if len(names) >= 6 else 0.42

    conf = round(max(0.0, min(0.92, conf)), 2)

    if batch:
        facts["batch"] = batch
    summary = _summarise(college_name, facts, batch)
    if not summary:
        return None
    return {"summary": summary, "facts": facts, "confidence": conf}


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":          # pragma: no cover
    import os
    import time

    # Verbatim excerpts of pages fetched on 2026-07-27 through
    # web_enrich.parse_html(). Kept inline so the self-test proves something
    # without a network call.
    SRHU = """Placements
From Classroom to Workplace
36 LPA
Highest Package
400+
Campus Recruiters
100%
Placement Support
#FromCampusToCorporate
At SRHU, placements are the outcome of a carefully designed ecosystem.
Placements Record by School
2025-26
School of
Management Studies"""

    CGCCOE = """Home
Applied Science
Placement Record
PLACEMENT RECORD 2016-2020
PLACEMENT RECORD 2017-2021
PLACEMENT RECOED 2020-2024
Unparalleled Placement Records
800+
Companies for Placement
8500+
Placement Offers
45.5Lacs
Highest Package
Testimonials
It's a famous saying atAdobe that "The future belongs to those who create it".
Megha Kwatra
Adobe Systems India Pvt. Ltd
CGC College of Engineering
Landran, Kharar-Banur Highway,
Sector 112, Greater Mohali,
Punjab 140307 (INDIA)
1800-2003575
admissions@cgc.edu.in
+91-95922-14444"""

    CHITKARA = """Record Breaking Placements
Our stellar placements ensure the perfect launchpad for careers with global
giants like Google, Deloitte, and Amazon.
Our Impact in Numbers
₹1.1 Cr
Highest Package
offered to a student
by global giant Amazon
₹10 LPA
870+ Students
received Super Dream
Offers from Blue-chip MNCs
2000+
Recruiting Companies
visit our campus annually
for student placements
72%
Chitkara Graduates
earn higher salary
than market average
Industry Partnerships & MOUs
Our robust industry engagement model is built on 300+ strategic MOUs."""

    LPU = """LPU students who secured packages ranging from ₹10 lakh to ₹2.5 crore
Know More
2225+
recruiters hired
LPU students
6000+
offers from
Fortune 500 companies
400+
recruiters of IITs/ IIMs/
NITs also hire from LPU
₹ 10.23 Lakh
average salary of
top 25% placed students
Recognition
Highly ranked for placements
10th best training and placement institutions in India
Shweta Ranjan secures an exceptional ₹58 LPA offer from Google India
J Nitesh has secured an impressive package of 53.41 LPA with Microsoft
Manav Bafna bags a placement at Amazon with a package of Rs. 48.64 LPA
Four B.Tech. students secured positions at SAP Labs India with ₹20.75 LPA"""

    BNC = """Training And Placement
A student placement cell has been formed to provide guidance and provide
facility for placement of students in various industries, company, etc.
Number of linkages for student exchange, internship, field trip, on-the-job
training, research, etc year-wise during the last five years:-
2017-18
2016-17
2015-16
2014-15
2013-14
7
3
3
4
0
Average percentage of students benefited by vocational education and training
(VET) during the last five years:-
2017-18
2016-17
2015-16
2014-15
2013-14
358
438
456
467
470
Average percentage of placement of outgoing students during the last five years:-
2017-18
2016-17
2015-16
2014-15
2013-14
0
0
0
0
0"""

    MARWARI = """About Placement
Training & Placement
Placement Brochure
Placement List
About Placement
WILP - Wipro's Work Integrated Learning Program 2019 Campus Drive
for 2019 graduates on 20th May ,2019 at the College premises
Registration Link
Training & Placement Cell
Location:
MARWARI COLLEGE, BHAGALPUR
BHAGALPUR - 812 007
Bihar-812 007
Call us on:
0641-2422768
VISITORS: 4551805"""

    TIUT = """Career Opportunity
Calling all legal experts! Join Techno India University, Tripura, as a
Professor/Associate Professor and inspire the next generation of legal minds
Techno India University, Tripura, is hiring Assistant Professors in Aquaculture
Other Opportunities
Download Advertisement
Estd. under The Tripura Act No. 4 of 2023 & UGC u/s 2(f) of UGC Act. 1956
Campus: Maheshkhola, Anandanagar, Agartala, Tripura - 799004
+91 9230994080 | 81 | 82 | 83 | 84
Office Hours: 8AM - 8PM
Our Visitor
630597 users
Users Today: 161
Users This Week: 7083
Users This Year: 426961
Total Users: 630597"""

    CHRIST = """CHRIST (Deemed to be University) | Central Campus
Placements
Empowering careers, building futures.
Hostel & Dining
Comfortable living
NAAC A+
Accredited
100+
Programmes
Global
Exposure
Admission
Examination
School of Engineering and Technology"""

    FEES_PAGE = """Fee Structure 2024-25
B.Tech Computer Science
Tuition Fee Rs. 1,20,000 per annum
Hostel Fee Rs 65,000 per year
Examination Fee Rs 2,500 per semester
Caution Deposit Rs 10,000 one time
Admission Fee Rs 25,000
Total Fee Rs 2,22,500 per annum
Contact: 0141-2345678, Jaipur - 302017"""

    CUTOFF_PAGE = """Cut Off 2024
Opening and closing ranks for B.Tech admission
CSE Opening Rank 12045 Closing Rank 45890
ECE Opening Rank 45900 Closing Rank 78000
JEE Main percentile required 92.5
Roll No 220156789 Application No 2024001234"""

    LABEL_VALUE = """Training and Placement Cell
Placement Highlights 2023-24
Highest Package : Rs. 24,00,000 per annum
Average Package : Rs 6,50,000 p.a.
Median Salary: 5.2 LPA
Number of students placed: 412
Total Recruiters: 96
Placement Percentage: 87%
Top recruiters: TCS, Infosys, Wipro, Cognizant, Capgemini, Accenture"""

    PROSE_PAGE = """Training and Placement Cell
The placement cell of the college works round the year. During the 2023-24
placement season the highest package offered was Rs 18 LPA and the average
package stood at 5.4 LPA. Over 60 companies visited the campus."""

    MARKETING = """Placement Cell
We offer 100% placement assistance to all our students. Our students enjoy
excellent placement records and the best placements in the region. We provide
100% placement support and guaranteed job assistance. Ranked No. 1 college
for placements in the state."""

    # Every number here is a phone, PIN, roll number, year, course code or
    # affiliation number sitting next to placement vocabulary on purpose.
    NUMBER_TRAP = """Training and Placement Cell
Contact the Placement Officer
Phone: 0641-2422768, Mobile +91 9230994080
Toll Free 1800-2003575
Address: Sector 112, Greater Mohali, Punjab 140307 (INDIA)
BHAGALPUR - 812 007, Bihar - 812 007
Affiliated to AICTE vide approval NO./06/KER/ENGG/2002/91 dated 12/05/03
Course code CSE-401 Placement Training, credits 4
Placement Cell established in 1998 under Act No. 4 of 2023
Roll No 220156789 placed in list
2024
Placements
VISITORS: 4551805
Users This Year: 426961"""

    # Numbered step lists sit next to placement captions on real admission and
    # process pages; zero-padded indices must never read as counts.
    STEP_LIST = """Placement Process
Step
04
Campus Recruiters
Step
06
Placement Offers
Step
07
Students Placed
Step
08
Attend the placement drive and complete the recruitment formalities"""

    STIPEND_TRAP = """Placements and Internships
Highest internship stipend Rs 25,000 per month
Average stipend Rs 12,000 per month
Placement assistance is provided to all students
Tuition fee Rs 1,20,000 per annum"""

    CASES = [
        ("srhu.edu.in/placements (real)", SRHU, "Swami Rama Himalayan University", True),
        ("cgccoe.org/placement-record (real)", CGCCOE, "CGC College of Engineering", True),
        ("chitkara.edu.in/career-advancement (real)", CHITKARA, "Chitkara University", True),
        ("lpu.in/placements.php (real)", LPU, "Lovely Professional University", True),
        ("bncollegepatna.com NAAC table (real)", BNC, "Bihar National College", True),
        ("label:value table (synthetic)", LABEL_VALUE, "Government Engineering College", True),
        ("prose paragraph (synthetic)", PROSE_PAGE, "St Xavier's College", True),
        ("-- marwaricollegebgp placement, no stats (real)", MARWARI, "Marwari College", False),
        ("-- tiut.ac.in/career, faculty vacancy (real)", TIUT, "Techno India University", False),
        ("-- christuniversity.in/placement, nav only (real)", CHRIST, "CHRIST University", False),
        ("-- fees page (wrong topic)", FEES_PAGE, "Some Institute", False),
        ("-- cutoff page (wrong topic)", CUTOFF_PAGE, "Some Institute", False),
        ("-- pure marketing", MARKETING, "Some Institute", False),
        ("-- phones/PINs/years next to placement words", NUMBER_TRAP, "Some Institute", False),
        ("-- stipends and fees, not packages", STIPEND_TRAP, "Some Institute", False),
        ("-- numbered step list beside placement captions", STEP_LIST, "Some Institute", False),
        ("-- empty page", "", "X", False),
        ("-- whitespace only", "   \n \n  \n", "X", False),
        ("-- nav only", "Home\nAbout\nAdmission\nContact\nGallery\n" * 6, "X", False),
        ("-- None-ish input", None, "X", False),
    ]

    # --- number parser: int() on an Indian-grouped figure is silently wrong ---
    NUM_CASES = [("1,20,000", 120000.0), ("12,00,000", 1200000.0),
                 ("1,200,000", 1200000.0), ("45000", 45000.0),
                 ("45.5", 45.5), ("36", 36.0), ("10.23", 10.23),
                 ("2,25,000", 225000.0), ("bad", None), ("", None), (None, None)]
    bad = [(raw, _to_number(raw), want) for raw, want in NUM_CASES
           if _to_number(raw) != want]
    print(f"number parser: {len(NUM_CASES) - len(bad)}/{len(NUM_CASES)} ok"
          + (f"  MISMATCHES {bad}" if bad else ""))

    MONEY_CASES = [
        ("45.5Lacs", None, "45.5", "Lacs", None, "45.5 Lakh"),
        ("Rs 12 Lakh PA", "Rs", "12", "Lakh", "PA", "Rs 12 Lakh PA"),
        ("₹1.1 Cr", "₹", "1.1", "Cr", None, "₹1.1 Crore"),
        ("INR 6.5 LPA", "INR", "6.5", "LPA", None, "INR 6.5 LPA"),
        ("Rs 45000 p.a.", "Rs", "45000", None, "p.a.", "Rs 45000 p.a."),
        ("Rs. 1,20,000", "Rs.", "1,20,000", None, None, "Rs. 1,20,000"),
        ("12000/- (no currency, no unit)", None, "12000", None, None, None),
        ("9230994080 (phone)", None, "9230994080", None, None, None),
        ("812007 (PIN, no currency)", None, "812007", None, None, None),
        ("2000 Cr (absurd package)", None, "2000", "Cr", None, None),
    ]
    print("money canonicaliser:")
    for note, cur, num, unit, per, want in MONEY_CASES:
        got_m = _canon_money(cur, num, unit, per)
        got_s = got_m[0] if got_m else None
        print(f"   {'ok ' if got_s == want else 'BAD'} {note:34} -> {got_s!r}")

    print("=" * 78)
    fp = fn = 0
    t_total = 0.0
    for label, txt, name, expect in CASES:
        t0 = time.perf_counter()
        got = extract(txt, "https://example.edu/placements", name)
        dt = (time.perf_counter() - t0) * 1000
        t_total += dt
        ok = (got is not None) == expect
        if not ok:
            if got is not None:
                fp += 1
            else:
                fn += 1
        print(f"\n[{'PASS' if ok else 'FAIL'}] {label}   ({dt:.1f} ms)")
        if got is None:
            print("   -> None")
        else:
            print(f"   conf {got['confidence']}")
            print(f"   sum  {got['summary']}")
            for k, v in got["facts"].items():
                print(f"        {k} = {v}")

    print("\n" + "=" * 78)
    print(f"false positives: {fp}   false negatives: {fn}   "
          f"total {t_total:.1f} ms over {len(CASES)} cases")

    # --- must never raise, and must never blow up on adversarial shapes ------
    import random
    random.seed(7)
    alphabet = "0123456789,.%+-/:₹ \nRsLPAlakhcroreplacementhighestaverage<>&;"
    crashes = 0
    t0 = time.perf_counter()
    for _ in range(400):
        junk = "".join(random.choice(alphabet) for _ in range(600))
        try:
            extract(junk, "u", "c")
        except BaseException as exc:     # noqa: BLE001 - that is the point
            crashes += 1
            print("   CRASH", type(exc).__name__, exc)
    # Shapes that kill a naive regex: long digit runs, deep comma groups,
    # thousands of repeated separators.
    for pathological in ("9" * 6000, ("1," * 3000), ("Rs " * 2000),
                         ("highest package " * 400), ("," * 6000),
                         ("2024-25 " * 800), ("100% placement " * 400)):
        try:
            extract(pathological, "u", "c")
        except BaseException as exc:     # noqa: BLE001
            crashes += 1
            print("   CRASH", type(exc).__name__, exc)
    print(f"fuzz + pathological inputs: {crashes} crashes, "
          f"{(time.perf_counter() - t0) * 1000:.0f} ms for 407 inputs")

    # --- the size the sweep actually hands over ----------------------------
    page6k = (CGCCOE + "\n") * 3
    page6k = page6k[:6000]
    runs = 200
    t0 = time.perf_counter()
    for _ in range(runs):
        extract(page6k, "https://example.edu/placements", "CGC")
    per = (time.perf_counter() - t0) * 1000 / runs
    print(f"6,000-char placement page: {per:.2f} ms/call over {runs} calls")

    # Full captured pages, when they are lying around from the fetch session.
    corpus = os.getenv("MIVI_PAGE_CORPUS")
    if corpus and os.path.isdir(corpus):
        print("\n" + "=" * 78)
        print(f"FULL PAGES from {corpus}")
        hits = total = 0
        worst = (0.0, "")
        for fn_ in sorted(os.listdir(corpus)):
            if not fn_.endswith(".txt") or fn_.endswith(".links.txt"):
                continue
            with open(os.path.join(corpus, fn_), encoding="utf-8") as fh:
                page = fh.read()
            total += 1
            t0 = time.perf_counter()
            got = extract(page, "https://example.edu/", "Test College")
            dt = (time.perf_counter() - t0) * 1000
            if dt > worst[0]:
                worst = (dt, f"{fn_[:50]} ({len(page)} chars)")
            if got is None:
                continue
            hits += 1
            print(f"\n{fn_[:70]}  ({len(page)} chars, {dt:.1f} ms)")
            print(f"   conf {got['confidence']}")
            print(f"   sum  {got['summary']}")
            print(f"   facts {got['facts']}")
        print(f"\n{hits} of {total} pages produced facts; "
              f"slowest {worst[0]:.1f} ms on {worst[1]}")
