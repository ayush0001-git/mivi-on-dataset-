"""Gate every harvested fact before it can reach a student.

WHY THIS IS THE MOST IMPORTANT MODULE IN THE HARVEST
A wrong fee in the database is far more expensive than a missing one. Missing
data is visible — MIVI says "we don't have that" and the student checks the
college site. Wrong data is invisible: it is stated confidently, cited, and
acted on, and nobody discovers it until a student has already made a decision
on it. And once it is in the store, finding it again among tens of thousands of
rows is much harder than never letting it in.

So nothing extracted goes straight into the served corpus. Everything passes
through here first and lands in one of three places:

    CONFIRMED  -> the corpus, usable in answers
    UNCERTAIN  -> quarantine, never shown, reviewable
    REJECTED   -> dropped, with the reason recorded

THE LOAD-BEARING CHECK: VERBATIM ANCHORING
Every number a fact asserts must appear LITERALLY in the source page text. If
an extractor reports "Rs 1,20,000 per year" then "1,20,000" (or 120000, or
1.2 lakh) has to be findable on the page it read. A number that is not on the
page was invented — by a regex that matched across two table cells, or by a
model that filled a gap — and no confidence score it reports about itself can
rescue it.

This check is deterministic, costs microseconds, needs no model, and cannot be
argued with. It is strictly stronger than a confidence threshold: an extractor
that is confidently wrong still fails it.

THE SECOND CHECK: IS THIS PAGE EVEN ABOUT THIS COLLEGE
A redirect, a parked domain, or a wrong guess from site discovery can hand us a
page belonging to someone else entirely. Facts harvested from it would be
attributed to the wrong college — the single worst failure in this pipeline,
because the number is real, just not theirs. So the page has to corroborate the
college's identity before anything it says is believed.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
import sys
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Plausibility envelopes. Deliberately WIDE — this is a fabrication and
# unit-error catcher, not an opinion about what a college should charge. Every
# bound below is well outside anything real in the Indian market, so a value
# that trips one is a data error rather than an unusual college.
# ---------------------------------------------------------------------------
BOUNDS = {
    # annual tuition in rupees: below 1k is a placeholder, above 5cr is a typo
    "money": (1_000, 50_000_000),
    # A percentage cutoff or placement rate. The floor is 0, not 1: a college
    # reporting 0% placement is a REAL and unusually informative datum (Bihar
    # National College publishes exactly that for five consecutive years), and
    # rejecting it would delete the honest disclosures while keeping the
    # flattering ones.
    "percent": (0.0, 100.0),
    # LPA package: 0.5 LPA is implausibly low, 500 LPA is a decimal slip
    "lpa": (0.5, 500.0),
    "year": (1800, 2100),
}

_NUM_RE = re.compile(r"\d[\d,.]*")
_WORD_RE = re.compile(r"[a-z0-9]+")

# Words that make a "fee" not an annual tuition. Presence near a money figure
# means the unit is being asserted, and an extractor that dropped it has
# changed the meaning of the number.
_PERIOD_WORDS = ("per year", "per annum", "p.a", "pa", "annual", "yearly",
                 "per semester", "semester", "per month", "monthly",
                 "total", "one time", "one-time", "lifetime", "course")


@dataclass
class VerifyResult:
    verdict: str                      # CONFIRMED | UNCERTAIN | REJECTED
    confidence: float                 # adjusted, not the extractor's own claim
    reasons: list[str] = field(default_factory=list)
    unanchored: list[str] = field(default_factory=list)
    anchors: dict = field(default_factory=dict)   # value -> snippet it came from

    @property
    def ok(self) -> bool:
        return self.verdict == "CONFIRMED"


# ---------------------------------------------------------------------------
# Number normalisation, for COMPARISON ONLY
# ---------------------------------------------------------------------------

def _variants(token: str) -> set[str]:
    """Every way a page might legitimately write the same number.

    Indian grouping is the whole reason this exists: 120000 is written
    "1,20,000" locally and "120,000" internationally, and "1.2 Lakh" means the
    same thing again. Anchoring must accept all of them or it would reject
    correct extractions constantly and be switched off — a check nobody can
    live with is worse than no check.
    """
    raw = token.strip()
    digits = raw.replace(",", "").replace(" ", "")
    out = {raw, raw.replace(",", ""), digits}
    try:
        val = float(digits)
    except ValueError:
        return {v for v in out if v}

    if val == int(val):
        n = int(val)
        out.add(str(n))
        # Indian grouping: last 3 digits, then pairs.
        s = str(n)
        if len(s) > 3:
            head, tail = s[:-3], s[-3:]
            groups = []
            while len(head) > 2:
                groups.insert(0, head[-2:])
                head = head[:-2]
            if head:
                groups.insert(0, head)
            out.add(",".join(groups + [tail]))
        # International grouping.
        out.add(f"{n:,}")
        # Lakh / crore forms, both "1.2" and "1.20".
        if n >= 1000:
            for unit, div in (("lakh", 100_000), ("crore", 10_000_000)):
                q = n / div
                if 0.1 <= q < 1000:
                    for text in (f"{q:g}", f"{q:.1f}", f"{q:.2f}".rstrip("0").rstrip(".")):
                        out.add(text)
                        out.add(f"{text} {unit}")
    return {v for v in out if v}


def _numbers_in(text: str) -> list[str]:
    return [m.group(0).strip(".,") for m in _NUM_RE.finditer(text or "")]


def _page_index(page_text: str) -> set[str]:
    """All numeric tokens on the page, in every normalised form, for O(1) lookup."""
    idx: set[str] = set()
    for tok in _numbers_in(page_text):
        idx |= _variants(tok)
    return idx


def _snippet(page_text: str, token: str, width: int = 90) -> str | None:
    """The text around a token, so a confirmed fact carries its own evidence.

    Stored with the fact: when someone later asks "where did this fee come
    from?", the answer is a quotation from the page, not a URL they have to
    re-read and hope has not changed.
    """
    for form in sorted(_variants(token), key=len, reverse=True):
        i = page_text.find(form)
        if i >= 0:
            lo = max(0, i - width)
            return re.sub(r"\s+", " ", page_text[lo:i + len(form) + width]).strip()
    return None


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def anchor_numbers(fact: dict, page_text: str) -> tuple[list[str], dict]:
    """Which asserted numbers are NOT on the page. Empty list means all anchored.

    Years and small counts are excluded from rejection: an extractor writing
    "3 dated notices" is summarising, not quoting, and "2026-27" may be
    assembled from "2026" and "27" appearing separately. Only numbers that
    carry a claim about money, marks or packages must anchor.
    """
    idx = _page_index(page_text)
    unanchored: list[str] = []
    anchors: dict = {}

    haystack = [str(fact.get("summary") or "")]
    for k, v in (fact.get("facts") or {}).items():
        haystack.append(f"{k}: {v}")

    for text in haystack:
        for tok in _numbers_in(text):
            # A dotted or slashed token is very often a DATE ("17.11.2026",
            # "2026-27") that the page writes differently from how the summary
            # renders it. Anchoring those produced false rejections, and a date
            # is not the kind of number a wrong value can mislead a student
            # about — money and marks are. So only plain integers are anchored.
            if "." in tok:
                continue
            digits = tok.replace(",", "")
            if not digits.isdigit():
                continue
            n = int(digits)
            # Small integers and bare years are summary artefacts, not claims.
            if n < 100:
                continue
            if 1800 <= n <= 2100 and len(digits) == 4:
                continue
            # A 3-digit run is usually a fragment of a year range ("2026-27" ->
            # "202") or a page number; too weak to reject a whole fact on.
            if len(digits) <= 3:
                continue
            if _variants(tok) & idx:
                snip = _snippet(page_text, tok)
                if snip:
                    anchors[tok] = snip
            else:
                unanchored.append(tok)
    return unanchored, anchors


def page_is_about(college_name: str, city: str | None, page_text: str) -> bool:
    """Does this page corroborate the college's identity?

    Guards against a redirect, a parked domain, or a wrong site guess. Matching
    is on distinctive name words — the generic ones ("college", "institute",
    "university") are stripped, because every college page contains them and
    they would make this check pass on any page at all.
    """
    stop = {"college", "institute", "university", "school", "of", "and", "the",
            "for", "government", "govt", "national", "indian", "india", "centre",
            "center", "academy", "education", "studies", "science", "sciences",
            "technology", "management", "group", "campus", "society", "trust",
            "degree", "arts", "commerce", "engineering"}
    name = (college_name or "").lower()
    hay = (page_text or "").lower()

    words = [w for w in _WORD_RE.findall(name) if len(w) > 3 and w not in stop]
    if any(w in hay for w in set(words)):
        return True

    # Acronyms. Indian institutions are overwhelmingly known by initials —
    # MVGR, SRM, BITS, VIT, KIIT, LPU — and a dotted form in the catalogue
    # ("M.V.G.R. Degree College") leaves only single letters after tokenising,
    # so a word-based check finds nothing and would reject the college's own
    # homepage. Rebuild the acronym and look for that instead.
    letters = re.findall(r"\b([a-z])\.", name)
    if len(letters) >= 2:
        acronym = "".join(letters)
        if acronym in hay or ".".join(letters) in hay:
            return True
    # Also the initials of the significant words, for undotted names.
    initials = "".join(w[0] for w in _WORD_RE.findall(name)
                       if len(w) > 2 and w not in {"of", "and", "the"})
    if len(initials) >= 3 and initials in hay:
        return True

    if city and len(city) > 3 and city.lower() in hay:
        return True
    # Nothing distinctive matched. If the name offered NOTHING to match on
    # (e.g. bare "Government College" with no city), we cannot judge — and
    # rejecting on an inability to check would throw away good data.
    return not words and not city


# A money figure's envelope depends entirely on the unit written next to it,
# and ignoring that unit is how a bounds check starts rejecting correct data.
# Measured against the real harvest, a unit-blind check falsely rejected:
#   "highest package Rs 22.57 LPA"  -> read as Rs 22        (a real placement figure)
#   "Rs. 300/- p.m. stipend"        -> read as annual fee   (a real SC/BC stipend)
#   "Rs 5,000 per semester"         -> compared to annual bounds
# So each unit gets its own envelope, and an amount with NO unit is judged
# loosely — an unlabelled number is a units problem (check_units handles it),
# not a fabrication.
_UNIT_ENVELOPES = (
    # (regex of the unit that FOLLOWS the amount, low, high, label)
    (r"(?:lpa|lakhs?\s*per\s*annum|lakhs?\s*p\.?a)", 0.5, 500.0, "LPA package"),
    (r"crores?", 0.01, 500.0, "crore amount"),
    (r"lakhs?", 0.05, 500.0, "lakh amount"),
    (r"(?:p\.?\s*m\.?|per\s*month|monthly|/\s*month)", 100, 500_000, "monthly amount"),
)

# Amount, optionally decimal, optionally followed by a unit word. The decimal is
# the important part: "22.57" must not be read as "22".
_AMOUNT_RE = re.compile(
    r"(?:rs\.?|inr|₹)\s*([\d][\d,]*(?:\.\d+)?)\s*(?:/-)?\s*([a-z./ ]{0,18})")


def check_bounds(fact: dict) -> list[str]:
    """Amounts that are impossible FOR THE UNIT THEY ARE WRITTEN IN."""
    problems: list[str] = []
    blob = json.dumps(fact.get("facts") or {}, ensure_ascii=False) + " " + str(
        fact.get("summary") or "")
    low = blob.lower()

    for m in _AMOUNT_RE.finditer(low):
        try:
            val = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        trailing = (m.group(2) or "").strip()
        matched = False
        for pat, lo, hi, label in _UNIT_ENVELOPES:
            if re.match(pat, trailing):
                if not (lo <= val <= hi):
                    problems.append(f"{label} {val:g} outside [{lo:g}, {hi:g}]")
                matched = True
                break
        if matched:
            continue
        # No recognised unit: judge as a rupee amount, but only flag the
        # genuinely absurd. A bare "Rs 500" may be an exam form fee or a
        # per-month figure whose unit sits further away in the sentence —
        # rejecting it as a fabricated tuition would be the wrong call, and
        # check_units already flags unlabelled money separately.
        lo, hi = BOUNDS["money"]
        if val > hi or val < 1:
            problems.append(f"money {val:,g} outside [1, {hi:,}]")

    # Percentages: skip anything that is plainly an ordinal ("20th percentile")
    # or a decimal share of a package.
    for m in re.finditer(r"([\d.]+)\s*(?:%|percent\b)", low):
        try:
            val = float(m.group(1))
        except ValueError:
            continue
        lo, hi = BOUNDS["percent"]
        if not (lo <= val <= hi):
            problems.append(f"percent {val:g} outside [{lo:g}, {hi:g}]")

    # LPA written without a currency symbol ("36 LPA", "6.5 lpa").
    for m in re.finditer(r"([\d.]+)\s*lpa\b", low):
        try:
            val = float(m.group(1))
        except ValueError:
            continue
        lo, hi = BOUNDS["lpa"]
        if not (lo <= val <= hi):
            problems.append(f"package {val:g} LPA outside [{lo:g}, {hi:g}]")
    return problems


def check_units(fact: dict, page_text: str) -> list[str]:
    """A money figure whose PERIOD was dropped.

    "Rs 45,000" means nothing on its own — per year and per semester differ by
    2x and total-course by four. If the page states a period near the figure and
    the fact does not repeat it, the fact has silently changed the number's
    meaning, which is a unit error rather than a fabrication and belongs in
    quarantine rather than the corpus.
    """
    blob = (str(fact.get("summary") or "") + " " +
            json.dumps(fact.get("facts") or {}, ensure_ascii=False)).lower()
    if not re.search(r"(?:rs\.?|inr|₹)\s*\d", blob):
        return []
    if any(w in blob for w in _PERIOD_WORDS):
        return []
    page_low = (page_text or "").lower()
    if any(w in page_low for w in _PERIOD_WORDS):
        return ["a money figure is reported with no period, but the page states one"]
    return []


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------

# A CONFIRMED fact needs this much adjusted confidence. Above the extractor's
# own bar on purpose: passing every structural check is necessary, not
# sufficient — a well-formed fact off a page that barely mentions the topic is
# still a guess.
CONFIRM_AT = 0.65


def verify(fact: dict, page_text: str, college_name: str = "",
           city: str | None = None) -> VerifyResult:
    """Gate one extracted fact. Never raises — a verifier that crashes would
    take the whole sweep with it, and its failure mode must be 'quarantine
    this', never 'lose everything'."""
    try:
        return _verify(fact, page_text, college_name, city)
    except Exception as exc:  # noqa: BLE001
        print(f"[verify] crashed, quarantining: {exc}", file=sys.stderr)
        return VerifyResult("UNCERTAIN", 0.0, [f"verifier error: {exc}"])


def _verify(fact, page_text, college_name, city) -> VerifyResult:
    reasons: list[str] = []
    claimed = float(fact.get("confidence") or 0.0)

    if not fact.get("summary") and not fact.get("facts"):
        return VerifyResult("REJECTED", 0.0, ["empty fact"])
    if not page_text or len(page_text.strip()) < 120:
        return VerifyResult("REJECTED", 0.0, ["no source text to verify against"])

    # 1. Wrong-entity check FIRST. If the page is not about this college, every
    #    other check would be validating someone else's numbers.
    if not page_is_about(college_name, city, page_text):
        return VerifyResult(
            "REJECTED", 0.0,
            [f"page does not corroborate the identity of {college_name!r} — "
             f"possible redirect, parked domain or wrong site match"])

    # 2. Verbatim anchoring. The strongest check we have.
    unanchored, anchors = anchor_numbers(fact, page_text)
    if unanchored:
        return VerifyResult(
            "REJECTED", 0.0,
            [f"numbers not present on the source page: {unanchored[:5]} — "
             f"these were not read off this page"],
            unanchored=unanchored, anchors=anchors)

    # 3. Plausibility.
    out_of_bounds = check_bounds(fact)
    if out_of_bounds:
        return VerifyResult("REJECTED", 0.0, out_of_bounds, anchors=anchors)

    # 4. Units — a real defect, but the value itself is genuine, so it is
    #    reviewable rather than binned.
    unit_problems = check_units(fact, page_text)
    adjusted = claimed
    if unit_problems:
        reasons += unit_problems
        adjusted *= 0.6

    # 5. Corroboration bonus: a fact whose every number is anchored WITH
    #    surrounding context is better evidenced than one where anchoring
    #    merely succeeded. Capped so it can never manufacture a pass on its own.
    if anchors:
        adjusted = min(1.0, adjusted + 0.05)

    verdict = "CONFIRMED" if adjusted >= CONFIRM_AT else "UNCERTAIN"
    if verdict == "UNCERTAIN" and not reasons:
        reasons.append(f"adjusted confidence {adjusted:.2f} below {CONFIRM_AT}")
    return VerifyResult(verdict, round(adjusted, 3), reasons, anchors=anchors)


def cross_check(facts_by_page: dict[str, dict]) -> tuple[str, list[str]]:
    """Agreement across two pages that reported the SAME kind for one college.

    Returns ('AGREE'|'CONFLICT'|'SINGLE', notes). A conflict does not pick a
    winner: two pages on the same site disagreeing about a fee means the site
    itself is ambiguous, and choosing one silently would hide that from the
    student. reconcile.py decides what to do with it.
    """
    if len(facts_by_page) < 2:
        return "SINGLE", []
    sets = []
    for url, fact in facts_by_page.items():
        nums = set()
        for tok in _numbers_in(str(fact.get("summary") or "")):
            digits = tok.replace(",", "").replace(".", "")
            if digits.isdigit() and int(digits) >= 1000:
                nums.add(int(digits))
        sets.append((url, nums))
    a, b = sets[0][1], sets[1][1]
    if not a or not b:
        return "SINGLE", []
    if a & b:
        return "AGREE", [f"{len(a & b)} figure(s) corroborated across two pages"]
    return "CONFLICT", [
        f"{sets[0][0]} and {sets[1][0]} report different figures "
        f"({sorted(a)[:3]} vs {sorted(b)[:3]}) — surface both, pick neither"]


# ---------------------------------------------------------------------------
# how old is this figure?
# ---------------------------------------------------------------------------

# "2025-26", "2025-2026", "2025/26", "AY 2024-25", "Session 2023-24"
_AY_RE = re.compile(
    r"(?:a\.?y\.?|academic\s+year|session|batch|admission)?\s*"
    r"\b(20[0-4]\d)\s*[-/–]\s*(\d{2}|20[0-4]\d)\b", re.I)
# A bare year, only when a fee/admission word vouches for it. "2019" alone on a
# page is as likely to be a copyright footer or an establishment year.
#
# \b on every vouching word is load-bearing: unanchored, "fee" matched inside
# "Feedback", which appears in the footer of nearly every Indian college site, so
# any stray year on the page could vouch for itself.
_BARE_YEAR_RE = re.compile(
    r"\b(?:fees?|tuition|charges|admissions?|session|academic)\b[^.\n]{0,40}?"
    r"\b(20[0-4]\d)\b|\b(20[0-4]\d)\b[^.\n]{0,20}?"
    r"\b(?:fee\s*structure|fees?|tuition)\b", re.I)

# Words that make a year the year OF THE FIGURES rather than just a year on the
# page. An academic-year token can appear anywhere — a news item, an affiliation
# notice, next year's prospectus link — and taking the maximum across 60k
# characters attributed the newest of those to the fees.
_NEAR_FEE = re.compile(
    r"\b(fees?|tuition|charges|fee\s*structure|admissions?|session|academic"
    r"|scholarship|cut\s*off|hostel|semester|annual)\b", re.I)
NEAR_WINDOW = 220        # characters either side of the year token


def stated_year(text: str, *, now_year: int | None = None) -> str | None:
    """The academic year a page says its figures are FOR, or None.

    This is not the same fact as when we fetched the page, and conflating the
    two is the specific way a stale fee looks fresh: the card used to say only
    "checked 2026-07-28", which a reader takes as "this fee is from 2026" when
    the page it came from is headed "Fee Structure 2019-20".

    The latest year on the page wins. Fee pages accumulate — last year's table is
    usually still there below this year's — so the earliest match would
    systematically under-date every figure.

    Returns None rather than guessing. An undated fee must be presented as
    undated; inventing a year here would be worse than admitting we do not know.
    """
    if not text:
        return None
    cap = (now_year or datetime.now(timezone.utc).year) + 1
    best: tuple[int, str] | None = None

    window = text[:60_000]
    for m in _AY_RE.finditer(window):
        start = int(m.group(1))
        tail = m.group(2)
        end = int(tail) if len(tail) == 4 else int(str(start)[:2] + tail)
        # An academic year spans one or two calendar years, never more: "2015-30"
        # is a validity range or a phone number, not a session.
        if not 2000 <= start <= cap or end - start not in (0, 1):
            continue
        # The year has to be NEAR something it could be the year of. Without
        # this, the maximum year anywhere on the page won — a news item, an
        # affiliation notice, or a link to next year's prospectus would re-date
        # a fee table published years earlier, which is the exact staleness this
        # function exists to expose.
        around = window[max(0, m.start() - NEAR_WINDOW):m.end() + NEAR_WINDOW]
        if not _NEAR_FEE.search(around):
            continue
        label = f"{start}-{str(end)[-2:]}"
        if best is None or start > best[0]:
            best = (start, label)

    if best:
        return best[1]

    for m in _BARE_YEAR_RE.finditer(text[:60_000]):
        y = int(m.group(1) or m.group(2))
        if 2000 <= y <= cap and (best is None or y > best[0]):
            best = (y, str(y))
    return best[1] if best else None
