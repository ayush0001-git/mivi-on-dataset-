"""Course-level detail for tier-1 colleges, read off the college's OWN pages.

WHAT IT FILLS
    courses.duration_years      the priority field, and the reason this exists
    courses.total_seats         sanctioned intake PER YEAR, never enrolment
    courses.study_mode          Full Time / Part Time / Distance, as printed
    courses.fees_by_category    General / OBC / SC-ST / NRI, as printed

plus one `college_field` row per (college, course, field) so the client can see
when each value was read and from where. A value written only to `courses` is
undated and invisible to the nightly refresh, so both writes happen together in
one transaction or neither happens.

WHY duration_years IS THE PRIORITY
`courses.tuition_*_inr` is the WHOLE-PROGRAMME fee. That was established against
Maharashtra's Fee Regulating Authority over 564 same-stream pairs: the ratio of
catalogue fee to legally-approved ANNUAL fee is the course length (MBA 2.00x,
MCA 2.39x, BTech 3.55x). MIVI therefore cannot say "per year" about any
catalogue fee, because course duration is present on 10 of 211,071 rows and the
annual figure cannot be derived. Every duration recovered here converts one
college's fee from an unusable total into a number a student can plan with.

WHAT IT REFUSES TO DO — each of these is a way this job goes wrong quietly

- It will not name a course the page did not name. A programme label is matched
  to a catalogue row on degree family FIRST and specialisation SECOND, and an
  ambiguous match writes nothing. Writing a real duration onto the wrong course
  is worse than writing nothing: the number is correct, so nothing downstream
  can detect it.
- It will not treat a MODIFIED intake route as the degree. This one was a real
  bug, caught by hand-checking what a first run had already written:
  sharda.ac.in publishes "B.Tech. (Lateral Entry) — 3 Year", the matcher read
  "(Lateral Entry)" as noise, and 3 years landed on every regular four-year
  B.Tech row at that university. Lateral-entry, integrated, dual, executive,
  part-time, distance, online and honours variants are different programmes with
  different lengths — see QUALIFIER_ONLY.
- It will not resolve a page that contradicts itself. If one page says B.Tech is
  4 years and another line says 3, the whole family is dropped. Two figures on
  one site means the site is ambiguous, and picking one silently hides that.
- It will not read a bare "Duration: 4" as years unless THAT page also writes
  "N years" in the same labelled-field shape elsewhere. The convention has to be
  demonstrated by the page, not assumed by us.
- It will not confuse intake with enrolment. `total_seats` is only taken from a
  cell whose own label says intake / seats / sanctioned strength.
- It will not convert currency, annualise a fee, or sum categories.
- It will not touch `duration_value` / `duration_unit` / `tuition_*`. Those are
  curated catalogue columns; nothing scraped overwrites the catalogue.

SEMESTERS. "8 semesters" is stored as 4.0 years. An Indian semester is half an
academic year without exception, and the verbatim "8 semesters" is kept in
college_field.value so the arithmetic is auditable. A page that says
"trimester" is NOT converted — that ratio is not fixed.

WHAT THIS DOES NOT COVER, AND WHY IT IS THE NEXT THING TO BUILD
The third source for sanctioned intake — the affiliating university and the
state counselling authority — is NOT read here. It is the richest source by
far: a state seat matrix lists every approved intake in the state in one
document, against a college site that usually publishes none. It is also a
separate job, because those documents need name-matching back to the corpus
(registries.match_one) and several are font-encoded regional-language PDFs:
Karnataka's KEA matrix, already on disk from an earlier session, comes out of
pypdf as Kannada mojibake and needs OCR or a font map before any parser can see
it. Building that on the end of a per-college crawler would have hidden the
matching risk inside a fetch loop.

Run:
    python -m rag_core.sources.harvest_courses --selftest        # parser only
    python -m rag_core.sources.harvest_courses --limit 20 --dry-run
    python -m rag_core.sources.harvest_courses --limit 200 --concurrency 8
    python -m rag_core.sources.harvest_courses --report
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass, field as dc_field
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse

import httpx

# config BEFORE backend: rag_core.config is the only module that calls
# load_dotenv(), and store.backend reads MIVI_PG_DSN at ITS import time. Get the
# order wrong and a run lands in the built-in localhost default instead of the
# configured server. Same trap as web_enrich.
from .. import config  # noqa: F401
from ..store.backend import get_backend
from .pdf_harvest import fetch_pdf, find_pdf_links, pdf_to_text
from .verify_facts import stated_year
from .verify_facts import verify as verify_fact
from .web_enrich import HEADERS, FirecrawlBudget, Politeness, fetch, normalise_site, parse_html

SOURCE = "college_site"

# ---------------------------------------------------------------------------
# Budgets. Every one of these is a third-party host we are a guest on.
# ---------------------------------------------------------------------------
MAX_PAGES_PER_COLLEGE = int(os.getenv("MIVI_COURSE_PAGES", "6"))
MAX_PDFS_PER_COLLEGE = int(os.getenv("MIVI_COURSE_PDFS", "2"))
PAGE_TEXT_BUDGET = 200_000     # a programme list is long; a 1 MB page is not one
MAX_LINES = 6_000
MAX_ROWS_PER_PAGE = 400        # a seat matrix can be genuinely huge; bound it

# Anchor text or path fragments worth a request, most specific first. Matching
# looks at BOTH, because Indian college sites routinely link "Courses Offered"
# to /page.php?id=42 where the URL alone gives nothing away.
COURSE_PAGE_KEYS = (
    "courses-offered", "courses offered", "programmes-offered", "programmes offered",
    "programs-offered", "programs offered", "course-offered", "programme-offered",
    "seat-matrix", "seat matrix", "sanctioned-intake", "sanctioned intake",
    "intake", "course-structure", "programme-structure", "academic-programmes",
    "academic programmes", "list-of-courses", "courses-and-fees", "courses & fees",
    "course-fee", "fee-structure", "fee structure",
    "courses", "programmes", "programs", "academics", "admission", "admissions",
)
# Paths that look right but never carry a programme table. Cheap to exclude and
# each one is a request not spent.
COURSE_PAGE_DENY = re.compile(
    r"(?:news|notice|circular|event|gallery|photo|video|alumni|contact|login"
    r"|career|job|recruit|tender|result|exam[\s_-]?schedule|feedback|grievance"
    r"|privacy|sitemap|blog|faculty|staff|download[\s_-]?centre)", re.I)


# ---------------------------------------------------------------------------
# Degree families
#
# The catalogue's `title` is a clean short code (124 distinct values across
# tier 1: BTech, BSc, MTech, BA LLB, PG Diploma, …) and `full_name` carries the
# specialisation ("Bachelor of Technology in Mechanical Engineering"). A page
# writes the same programme any of a dozen ways, so both sides are normalised to
# one family code and matched on that.
#
# ORDER IS LOAD-BEARING: the first pattern that matches wins, so every compound
# degree (BA LLB, BSc Ag, MSc) must precede its prefix (BA, BSc, MS).
# ---------------------------------------------------------------------------

def _fam_key(s: str) -> str:
    return re.sub(r"[^a-z0-9&]", "", (s or "").lower())


FAMILY_PATTERNS: tuple[tuple[str, str], ...] = (
    # --- compound / dual degrees, before their prefixes -------------------
    ("ballb",  r"\bb\.?\s?a\.?[\s.]*ll\.?\s?b\b|bachelor\s+of\s+arts.{0,20}law"),
    ("bballb", r"\bb\.?\s?b\.?\s?a\.?[\s.]*ll\.?\s?b\b"),
    ("bcomllb", r"\bb\.?\s?com\.?[\s.]*ll\.?\s?b\b"),
    ("bvsc&ah", r"\bb\.?\s?v\.?\s?sc\b.{0,12}\ba\.?\s?h\b|bachelor\s+of\s+veterinary"),
    ("bscag",  r"\bb\.?\s?sc\b.{0,6}\bag(?:ri|riculture)?\b|"
               r"bachelor\s+of\s+science\s+in\s+agricultur"),
    ("mscag",  r"\bm\.?\s?sc\b.{0,6}\bag(?:ri|riculture)?\b|"
               r"master\s+of\s+science\s+in\s+agricultur"),
    ("bel ed", r"\bb\.?\s?el\.?\s?ed\b"),
    ("del ed", r"\bd\.?\s?el\.?\s?ed\b"),
    # --- doctoral / research ---------------------------------------------
    ("phd",    r"\bph\.?\s?d\b|doctor\s+of\s+philosophy"),
    ("dphil",  r"\bd\.?\s?phil\b"),
    ("mphil",  r"\bm\.?\s?phil\b"),
    ("fpm",    r"\bf\.?\s?p\.?\s?m\b|fellow\s+programme\s+in\s+management"),
    # --- medicine ---------------------------------------------------------
    ("mbbs",   r"\bm\.?\s?b\.?\s?b\.?\s?s\b"),
    ("bds",    r"\bb\.?\s?d\.?\s?s\b|bachelor\s+of\s+dental"),
    ("mds",    r"\bm\.?\s?d\.?\s?s\b|master\s+of\s+dental"),
    ("bams",   r"\bb\.?\s?a\.?\s?m\.?\s?s\b"),
    ("bhms",   r"\bb\.?\s?h\.?\s?m\.?\s?s\b"),
    ("bnys",   r"\bb\.?\s?n\.?\s?y\.?\s?s\b"),
    ("pharmd", r"\bpharm\.?\s?d\b|doctor\s+of\s+pharmacy"),
    ("mch",    r"\bm\.?\s?ch\b"),
    ("dm",     r"\bd\.?\s?m\b(?!\w)"),
    ("dnb",    r"\bd\.?\s?n\.?\s?b\b"),
    ("bpt",    r"\bb\.?\s?p\.?\s?t\b|bachelor\s+of\s+physiotherapy"),
    ("mpt",    r"\bm\.?\s?p\.?\s?t\b|master\s+of\s+physiotherapy"),
    ("gnm",    r"\bg\.?\s?n\.?\s?m\b|general\s+nursing"),
    ("anm",    r"\ba\.?\s?n\.?\s?m\b"),
    # --- pharmacy / architecture / planning / design ----------------------
    ("bpharm", r"\bb\.?\s?pharm\b|bachelor\s+of\s+pharmacy"),
    ("mpharm", r"\bm\.?\s?pharm\b|master\s+of\s+pharmacy"),
    ("dpharm", r"\bd\.?\s?pharm\b|diploma\s+in\s+pharmacy"),
    ("barch",  r"\bb\.?\s?arch\b|bachelor\s+of\s+architecture"),
    ("march",  r"\bm\.?\s?arch\b|master\s+of\s+architecture"),
    ("bplan",  r"\bb\.?\s?plan\b|bachelor\s+of\s+planning"),
    ("mplan",  r"\bm\.?\s?plan\b|master\s+of\s+planning"),
    ("bdes",   r"\bb\.?\s?des\b|bachelor\s+of\s+design"),
    ("mdes",   r"\bm\.?\s?des\b|master\s+of\s+design"),
    # --- law / education --------------------------------------------------
    ("llb",    r"\bll\.?\s?b\b|bachelor\s+of\s+laws?"),
    ("llm",    r"\bll\.?\s?m\b|master\s+of\s+laws?"),
    ("bed",    r"\bb\.?\s?ed\b|bachelor\s+of\s+education"),
    ("med",    r"\bm\.?\s?ed\b|master\s+of\s+education"),
    ("bped",   r"\bb\.?\s?p\.?\s?ed\b"),
    ("mped",   r"\bm\.?\s?p\.?\s?ed\b"),
    # --- engineering / technology ----------------------------------------
    # "b.e" must not swallow "b.ed": \b after e fails on the 'd'.
    ("btech",  r"\bb\.?\s?tech\b|bachelor\s+of\s+technology|\bb\.?\s?e\b"
               r"|bachelor\s+of\s+engineering"),
    ("mtech",  r"\bm\.?\s?tech\b|master\s+of\s+technology|\bm\.?\s?e\b"
               r"|master\s+of\s+engineering"),
    ("bca",    r"\bb\.?\s?c\.?\s?a\b|bachelor\s+of\s+computer\s+application"),
    ("mca",    r"\bm\.?\s?c\.?\s?a\b|master\s+of\s+computer\s+application"),
    ("bit",    r"\bb\.?\s?i\.?\s?t\b"),
    # --- management -------------------------------------------------------
    ("emba",   r"\be\.?\s?m\.?\s?b\.?\s?a\b|executive\s+m\.?\s?b\.?\s?a"
               r"|executive\s+master\s+of\s+business"),
    ("mba",    r"\bm\.?\s?b\.?\s?a\b|master\s+of\s+business\s+administration"),
    ("bba",    r"\bb\.?\s?b\.?\s?a\b|bachelor\s+of\s+business\s+administration"),
    ("bbm",    r"\bb\.?\s?b\.?\s?m\b"),
    ("bms",    r"\bb\.?\s?m\.?\s?s\b(?!c)"),
    ("mms",    r"\bm\.?\s?m\.?\s?s\b(?!c)"),
    ("ipm",    r"\bi\.?\s?p\.?\s?m\b|integrated\s+programme?\s+in\s+management"),
    ("pgpm",   r"\bp\.?\s?g\.?\s?p\.?\s?m\b"),
    ("mha",    r"\bm\.?\s?h\.?\s?a\b|master\s+of\s+hospital\s+administration"),
    ("mhrm",   r"\bm\.?\s?h\.?\s?r\.?\s?m\b"),
    # --- core arts / science / commerce, LAST because they are prefixes ---
    ("bsc",    r"\bb\.?\s?sc\b|bachelor\s+of\s+science"),
    ("msc",    r"\bm\.?\s?sc\b|master\s+of\s+science"),
    ("bcom",   r"\bb\.?\s?com\b|bachelor\s+of\s+commerce"),
    ("mcom",   r"\bm\.?\s?com\b|master\s+of\s+commerce"),
    ("bfa",    r"\bb\.?\s?f\.?\s?a\b|bachelor\s+of\s+fine\s+arts"),
    ("mfa",    r"\bm\.?\s?f\.?\s?a\b|master\s+of\s+fine\s+arts"),
    ("bsw",    r"\bb\.?\s?s\.?\s?w\b|bachelor\s+of\s+social\s+work"),
    ("msw",    r"\bm\.?\s?s\.?\s?w\b|master\s+of\s+social\s+work"),
    ("bvoc",   r"\bb\.?\s?voc\b"),
    ("mvoc",   r"\bm\.?\s?voc\b"),
    ("ba",     r"\bb\.?\s?a\b|bachelor\s+of\s+arts"),
    ("ma",     r"\bm\.?\s?a\b|master\s+of\s+arts"),
    ("ms",     r"\bm\.?\s?s\b(?!c)"),
    ("md",     r"\bm\.?\s?d\b(?!\w)"),
    ("pgdiploma", r"\bp\.?\s?g\.?\s*diploma\b|post[\s-]?graduate\s+diploma"),
    ("diploma", r"\bdiploma\b"),
    ("certification", r"\bcertificate\b|\bcertification\b"),
)
_FAMILY_RE = tuple((fam, re.compile(pat, re.I)) for fam, pat in FAMILY_PATTERNS)


def family_of(label: str) -> str | None:
    """The degree family a free-text programme label names, or None."""
    if not label:
        return None
    s = re.sub(r"\s+", " ", label)[:160]
    for fam, rx in _FAMILY_RE:
        if rx.search(s):
            return fam
    return None


# Words that survive in a specialisation but say nothing distinctive about it.
# Kept in the token set (they help a weak match hold together) but a match must
# share at least one token from OUTSIDE this set.
GENERIC_SPEC = {
    "engineering", "technology", "science", "sciences", "studies", "management",
    "arts", "commerce", "general", "regular", "programme", "program", "course",
    "degree", "and", "with", "in", "of", "the", "for", "specialisation",
    "specialization", "branch", "stream", "full", "time", "self", "financed",
    "aided", "unaided", "shift", "morning", "day", "new", "revised", "applied",
}
# A label whose only extra words are these is still a DEGREE-level statement:
# "MBA (Full Time)" and "B.Tech (Regular)" name no specialisation, so what they
# say about duration or intake is true of every row of that degree.
#
# WHAT IS DELIBERATELY *NOT* HERE, and why this list is the most dangerous one
# in the file. An earlier version included "lateral" and "entry", which made
# "B.Tech. (Lateral Entry) — 3 Year" on sharda.ac.in read as a statement about
# B.Tech, and 3 years was written onto every regular four-year B.Tech row at
# that university. The number was real; the course was not. A lateral-entry,
# integrated, dual, executive, part-time, distance, online or honours variant is
# a DIFFERENT programme with a DIFFERENT length, so a label carrying one of
# those words is never degree-level. It either matches a catalogue row that
# carries the same word, or it matches nothing.
QUALIFIER_ONLY = {
    "regular", "full", "time", "general", "programme", "program", "course",
    "degree", "shift", "morning", "day", "self", "financed", "aided", "unaided",
    "sfs", "new", "revised",
}
_TOKEN_RE = re.compile(r"[a-z][a-z0-9]{1,}")
# Grammar words and degree nouns that carry no specialisation. Deliberately
# SHORT: an earlier version also stripped discipline nouns (computer, science,
# engineering, management…), which deleted the very words that tell one
# specialisation from another — "B.Tech Computer Science and Engineering" and
# "B.Tech Mechanical Engineering" both reduced to nothing and every B.Tech row
# at a college matched every other.
_SPEC_STOP = {
    "bachelor", "master", "doctor", "doctorate", "philosophy", "the", "and",
    "for", "with", "degree", "programme", "program", "course", "courses",
    "hons", "honours", "honors", "specialisation", "specialization",
}


def spec_tokens(label: str, family: str | None) -> set[str]:
    """Distinctive words of a programme label, with the degree phrase removed.

    The degree's OWN pattern does the stripping — "Bachelor of Technology in
    Computer Science" loses "Bachelor of Technology" because that is what the
    btech pattern matched — so nothing but the specialisation survives.
    """
    s = (label or "").lower()
    if family:
        for fam, rx in _FAMILY_RE:
            if fam == family:
                s = rx.sub(" ", s)
                break
    return {t for t in _TOKEN_RE.findall(s)
            if len(t) > 2 and t not in _SPEC_STOP}


# ---------------------------------------------------------------------------
# Cell parsing
# ---------------------------------------------------------------------------

# "4 Years", "3 Yrs", "8 Semesters", "6 months", "4.5 years", "2 Year"
_DURATION_RE = re.compile(
    r"^\s*(\d{1,2}(?:\.\d)?)\s*(years?|yrs?|yr|semesters?|sems?|months?|mos?)\b"
    r"[\s.\-–(]*(?:\(?\s*\d{1,2}\s*(?:semesters?|sems?)\s*\)?)?\s*$", re.I)
_DURATION_INLINE_RE = re.compile(
    r"\b(\d{1,2}(?:\.\d)?)\s*(years?|yrs?|semesters?|sems?|months?)\b", re.I)
_TRIMESTER_RE = re.compile(r"trimester|quarter", re.I)

DURATION_MIN_YEARS = 0.25
DURATION_MAX_YEARS = 8.0

# Sanctioned intake per year. 2,000 is far above anything real for one
# programme (the largest single sanctioned intake in Indian engineering is a few
# hundred) and well below a PIN code or a phone fragment.
SEATS_MIN = 1
SEATS_MAX = 2_000


def parse_duration(cell: str) -> tuple[float, str] | None:
    """(years, verbatim) from a duration cell, or None.

    A semester is half an academic year in Indian higher education without
    exception, so "8 semesters" is 4.0 years. A TRIMESTER is not — that ratio
    varies by institution — so a cell mentioning one is refused outright.
    """
    if not cell or _TRIMESTER_RE.search(cell):
        return None
    m = _DURATION_RE.match(cell.strip())
    if m is None:
        return None
    return _duration_from(m.group(1), m.group(2), m.group(0).strip())


def _duration_from(num: str, unit: str, verbatim: str) -> tuple[float, str] | None:
    try:
        n = float(num)
    except ValueError:
        return None
    u = unit.lower()
    if u.startswith(("year", "yr")):
        years = n
    elif u.startswith(("sem", "semester")):
        years = n / 2.0
    elif u.startswith(("month", "mo")):
        years = n / 12.0
    else:
        return None
    if not (DURATION_MIN_YEARS <= years <= DURATION_MAX_YEARS):
        return None
    return round(years, 2), re.sub(r"\s+", " ", verbatim).strip()


_SEATS_CELL_RE = re.compile(r"^\s*(\d{1,4})\s*(?:students?|seats?|nos?\.?)?\s*$", re.I)


def parse_seats(cell: str) -> tuple[int, str] | None:
    m = _SEATS_CELL_RE.match((cell or "").strip())
    if m is None:
        return None
    n = int(m.group(1))
    # A bare 4-digit 1900-2100 is a session year, not an intake.
    if 1900 <= n <= 2100:
        return None
    if not (SEATS_MIN <= n <= SEATS_MAX):
        return None
    return n, re.sub(r"\s+", " ", (cell or "").strip())


# Study modes, as Indian colleges actually print them. The page's own phrase is
# stored; this only decides whether the cell IS a mode.
_MODE_RE = re.compile(
    r"^\s*(full[\s-]?time|part[\s-]?time|regular|distance|correspondence|open"
    r"|online|blended|hybrid|weekend|evening|executive|self[\s-]?paced"
    r"|full time / part time)\s*(?:mode)?\s*$", re.I)


def parse_mode(cell: str) -> str | None:
    m = _MODE_RE.match((cell or "").strip())
    if m is None:
        return None
    return re.sub(r"\s+", " ", m.group(1)).strip().title()[:40]


# A money cell, kept VERBATIM. Nothing here converts or sums.
_MONEY_CELL_RE = re.compile(
    r"^\s*(?:rs\.?|inr|₹|\$|us\$)?\s*\d{1,3}(?:[,\d]{0,12})(?:\.\d{1,2})?"
    r"\s*(?:/-|/=)?\s*(?:lakhs?|lacs?|crores?|inr|rs\.?|₹)?\s*$", re.I)
_MONEY_DIGITS_RE = re.compile(r"\d")


def parse_money(cell: str) -> str | None:
    s = (cell or "").strip()
    if not s or len(s) > 32 or not _MONEY_DIGITS_RE.search(s):
        return None
    if not _MONEY_CELL_RE.match(s):
        return None
    digits = re.sub(r"[^\d]", "", s)
    # Below 1,000 a bare number in a fee column is far more often a serial number
    # or a count than a price; above 5 crore it is a typo.
    if not digits or not (1_000 <= int(digits) <= 500_000_000):
        return None
    return re.sub(r"\s+", " ", s)


# Reservation / quota categories, as printed. Only these become
# fees_by_category keys — a column headed "Total" or "Hostel" is a fee but not a
# category, and belongs to the fee harvester, not here.
_CATEGORY_RE = re.compile(
    r"\b(general|gen|open|obc|obc[\s-]?nc|sc|st|sc[\s/-]*st|ews|nri|foreign"
    r"|international|management(?:\s+quota)?|government(?:\s+quota)?|govt"
    r"|merit|minority|pwd|ph\b|defence|tfw|supernumerary|payment(?:\s+seat)?)\b",
    re.I)


def category_of(header: str) -> str | None:
    h = re.sub(r"\s+", " ", (header or "")).strip(" :|")
    if not h or len(h) > 44:
        return None
    if not _CATEGORY_RE.search(h):
        return None
    # Refuse a header that is really something else wearing a category word.
    # "General Fund" and "Management Committee" were anticipated; "School of
    # Management" was not, and gdgoenkauniversity.com's fee page turned that
    # column heading into a reservation category called "School of Management".
    # An ORGANISATIONAL UNIT is never a quota.
    if re.search(r"\b(fund|committee|body|office|council|trust|society|school"
                 r"|faculty|department|dept|institute|college|centre|center"
                 r"|university|campus|programme|program)\b", h, re.I):
        return None
    return h


# ---------------------------------------------------------------------------
# Column headers
# ---------------------------------------------------------------------------
_HDR_PROGRAMME = re.compile(
    r"^\s*(?:name\s+of\s+(?:the\s+)?)?(?:course|courses|programme|programmes"
    r"|program|programs|degree|specialisation|specialization|branch|discipline"
    r"|subject|department|stream)\s*(?:name|offered|title)?\s*$", re.I)
_HDR_DURATION = re.compile(r"^\s*duration\s*(?:\(.{0,12}\))?\s*$|^\s*course\s+duration\s*$",
                           re.I)
_HDR_SEATS = re.compile(
    r"^\s*(?:approved\s+|sanctioned\s+|total\s+|annual\s+)?"
    r"(?:intake|seats?|no\.?\s*of\s+seats|number\s+of\s+seats|seat\s+intake"
    r"|sanctioned\s+strength|intake\s+capacity)\s*(?:\(.{0,12}\))?\s*$", re.I)
_HDR_MODE = re.compile(r"^\s*(?:study\s+)?mode(?:\s+of\s+study)?\s*$|^\s*type\s*$", re.I)

MAX_HEADER_WIDTH = 8
MIN_TABLE_ROWS = 2


# ---------------------------------------------------------------------------
# What one programme row on a page asserts
# ---------------------------------------------------------------------------

@dataclass
class RowFact:
    label: str
    shape: str                      # 'table' | 'labelled'
    duration_years: float | None = None
    duration_raw: str | None = None          # the cell AS PRINTED, for anchoring
    duration_inferred: bool = False          # unit came from the page's own habit
    seats: int | None = None
    seats_raw: str | None = None
    mode: str | None = None                  # normalised, may not be verbatim
    mode_raw: str | None = None              # the cell AS PRINTED, for anchoring
    fees_by_category: dict = dc_field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return (self.duration_years is None and self.seats is None
                and self.mode is None and not self.fees_by_category)

    def evidence(self) -> str:
        bits = []
        if self.duration_raw:
            note = " (page writes durations in years elsewhere)" if self.duration_inferred else ""
            bits.append(f"Duration: {self.duration_raw}{note}")
        if self.seats_raw:
            bits.append(f"Intake: {self.seats_raw}")
        if self.mode_raw:
            bits.append(f"Mode: {self.mode_raw}")
        for k, v in list(self.fees_by_category.items())[:6]:
            bits.append(f"{k}: {v}")
        return "; ".join(bits)[:300]


# ---------------------------------------------------------------------------
# Shape A — labelled fields
#
#   Bachelor of Architecture
#   Offered By : Department Of Architecture
#   Duration: 5 years
#   Intake: 40
#
# jadavpuruniversity.in/courses publishes its entire programme list this way,
# and it is by far the most common shape once a site drops the table markup.
# ---------------------------------------------------------------------------

_LBL_DURATION = re.compile(r"^\s*(?:course\s+)?duration\s*[:\-–]\s*(.{1,40})$", re.I)
_LBL_SEATS = re.compile(
    r"^\s*(?:approved\s+|sanctioned\s+|total\s+|annual\s+)?"
    r"(?:intake|seats?|no\.?\s*of\s+seats|number\s+of\s+seats|intake\s+capacity"
    r"|sanctioned\s+strength)\s*[:\-–]\s*(.{1,30})$", re.I)
_LBL_MODE = re.compile(r"^\s*(?:study\s+)?mode(?:\s+of\s+study)?\s*[:\-–]\s*(.{1,30})$",
                       re.I)
# A line that is itself a labelled field, and therefore never a programme name.
_ANY_LABEL = re.compile(
    r"^\s*(?:course\s+)?(?:duration|intake|seats?|mode|offered\s+by|eligibility"
    r"|fees?|department|level|faculty|syllabus|curriculum|apply|click)\b", re.I)

LABEL_LOOKBACK = 6          # lines above a "Duration:" that may hold the name
_BARE_NUM_RE = re.compile(r"^\s*(\d{1,2})\s*$")


def _plausible_label(line: str) -> bool:
    """Could this line be a programme name?

    The lower length bound is 3, not 6, because "MBA" / "BCA" / "LLB" are whole
    programme names and a 6-character floor silently dropped every one of them.
    Anything that short must still name a degree family, so a stray "2000" or
    "USD" cannot become a course.
    """
    s = (line or "").strip(" \t|:;-–—")
    if not (3 <= len(s) <= 120):
        return False
    if len(s) < 6 and not family_of(s):
        return False
    if _ANY_LABEL.match(s) or COURSE_PAGE_DENY.search(s):
        return False
    if "http" in s.lower() or "@" in s:
        return False
    letters = sum(c.isalpha() for c in s)
    return letters >= 3 and letters / len(s) > 0.45


def parse_labelled(lines: list[str]) -> list[RowFact]:
    """Shape A. Returns one RowFact per programme block found."""
    out: list[RowFact] = []
    # Does this page establish that a bare "Duration: 4" means years? Only its
    # OWN convention counts — an explicit "N years" elsewhere in the same shape.
    bare_unit_ok = any(_DURATION_RE.match((m.group(1) or "").strip())
                       for m in (_LBL_DURATION.match(l) for l in lines) if m)

    i = 0
    n = len(lines)
    while i < n and len(out) < MAX_ROWS_PER_PAGE:
        md = _LBL_DURATION.match(lines[i])
        ms = _LBL_SEATS.match(lines[i])
        mm = _LBL_MODE.match(lines[i])
        if not (md or ms or mm):
            i += 1
            continue
        # Walk back to the nearest line that could be a programme name.
        label = None
        for j in range(i - 1, max(-1, i - 1 - LABEL_LOOKBACK), -1):
            if _plausible_label(lines[j]):
                label = re.sub(r"\s+", " ", lines[j]).strip(" \t|:;-–—")
                break
        if label is None:
            i += 1
            continue
        row = RowFact(label=label, shape="labelled")
        # Consume the consecutive labelled fields belonging to this block.
        k = i
        while k < n and (k - i) < 8:
            d, s, mo = (_LBL_DURATION.match(lines[k]), _LBL_SEATS.match(lines[k]),
                        _LBL_MODE.match(lines[k]))
            if d and row.duration_years is None:
                cell = d.group(1).strip()
                got = parse_duration(cell)
                if got is None and bare_unit_ok:
                    b = _BARE_NUM_RE.match(cell)
                    if b:
                        got = _duration_from(b.group(1), "years", cell)
                        row.duration_inferred = got is not None
                if got:
                    row.duration_years, row.duration_raw = got
            elif s and row.seats is None:
                got = parse_seats(s.group(1).strip())
                if got:
                    row.seats, row.seats_raw = got
            elif mo and row.mode is None:
                raw = mo.group(1).strip()
                row.mode = parse_mode(raw)
                row.mode_raw = raw if row.mode else None
            elif not (d or s or mo) and _plausible_label(lines[k]) and k > i:
                break            # next programme block has started
            k += 1
        if not row.empty:
            out.append(row)
        i = max(k, i + 1)
    return out


# ---------------------------------------------------------------------------
# Shape B — a column table flattened to one cell per line
#
#   Programmes / Duration / Tuition Fee (USD)      <- header, k lines
#   B.Tech. (Lateral Entry) / 3 Year / 3800        <- row, k lines
#
# www.sharda.ac.in publishes exactly this. The header is what makes the columns
# readable at all; a table whose header cannot be read is skipped rather than
# labelled by position, which is the same rule extractors/fees.py enforces.
# ---------------------------------------------------------------------------

_HDR_FILLER_RE = re.compile(
    r"^(?:sl\.?\s*no\.?|s\.?\s*no\.?|sr\.?\s*no\.?|serial|#|no\.?|year|eligibility"
    r"|remarks?|fees?|tuition|total|amount|annual\s+fees?|fee\s+structure"
    r"|syllabus|curriculum|details?|action|apply)\s*(?:\(.{0,14}\))?$", re.I)


def _is_data_cell(cell: str) -> bool:
    """A cell that carries a VALUE, which a header cell never does.

    This is what bounds the header scan. Anything else — including a column
    caption this module does not recognise, like "Tuition Fee (USD)" — is
    tolerated inside the header, because refusing an unknown caption is what
    made the width come out one short and paired every programme name with the
    previous row's fee.
    """
    return bool(parse_duration(cell) or parse_seats(cell)
                or parse_money(cell) or parse_mode(cell))


def _header_at(lines: list[str], i: int) -> tuple[dict, list, int, int] | None:
    """(roles, categories, min_width, max_width) if a header run starts at i.

    The WIDTH is not decided here. A header cell this module does not recognise
    is invisible to it, so the recognised roles only establish a LOWER bound;
    the true period of the table is settled by parse_tables against the data
    rows, which is evidence rather than a guess.
    """
    first = re.sub(r"\s+", " ", lines[i]).strip(" :|")
    if not first or len(first) > 60:
        return None
    if not (_HDR_PROGRAMME.match(first) or _HDR_DURATION.match(first)
            or _HDR_SEATS.match(first) or _HDR_MODE.match(first)
            or _HDR_FILLER_RE.match(first) or category_of(first)):
        return None

    roles: dict[str, int] = {}
    cats: list[tuple[int, str]] = []
    w = 0
    while w < MAX_HEADER_WIDTH and i + w < len(lines):
        cell = re.sub(r"\s+", " ", lines[i + w]).strip(" :|")
        if not cell or len(cell) > 60 or _is_data_cell(cell):
            break
        if _HDR_PROGRAMME.match(cell) and "label" not in roles:
            roles["label"] = w
        elif _HDR_DURATION.match(cell) and "duration" not in roles:
            roles["duration"] = w
        elif _HDR_SEATS.match(cell) and "seats" not in roles:
            roles["seats"] = w
        elif _HDR_MODE.match(cell) and "mode" not in roles:
            roles["mode"] = w
        else:
            cat = category_of(cell)
            if cat:
                cats.append((w, cat))
        w += 1
    if "label" not in roles:
        return None
    typed = {"duration", "seats", "mode"} & set(roles)
    # Two or more category columns make a category breakdown; one is just a fee,
    # and a fee is extractors/fees.py's topic, not this module's.
    if not typed and len(cats) < 2:
        return None
    used = list(roles.values()) + [o for o, _ in cats]
    return roles, cats, max(used) + 1, max(w, max(used) + 1)


def _read_rows(lines: list[str], start: int, width: int, roles: dict,
               cats: list, limit: int) -> list[RowFact]:
    """Data rows of one table at a given width. Empty if the width is wrong."""
    rows: list[RowFact] = []
    n = len(lines)
    j = start
    misses = 0
    while j + width <= n and len(rows) < limit:
        cells = [re.sub(r"\s+", " ", lines[j + t]).strip(" |") for t in range(width)]
        label = cells[roles["label"]]
        row = RowFact(label=label, shape="table")
        typed_ok = False
        if "duration" in roles:
            got = parse_duration(cells[roles["duration"]])
            if got:
                row.duration_years, row.duration_raw = got
                typed_ok = True
        if "seats" in roles:
            got = parse_seats(cells[roles["seats"]])
            if got:
                row.seats, row.seats_raw = got
                typed_ok = True
        if "mode" in roles:
            m = parse_mode(cells[roles["mode"]])
            if m:
                row.mode = m
                row.mode_raw = cells[roles["mode"]]
                typed_ok = True
        for idx, cat in cats:
            money = parse_money(cells[idx])
            if money:
                row.fees_by_category[cat] = money
        if len(row.fees_by_category) >= 2:
            typed_ok = True
        if typed_ok and _plausible_label(label):
            rows.append(row)
            misses = 0
        else:
            misses += 1
            # Two consecutive unreadable rows: the table has ended and what
            # follows is page furniture. Reading on would pair a programme name
            # with a cell from somewhere else entirely.
            if misses >= 2:
                break
        j += width
    return rows


def parse_tables(lines: list[str]) -> list[RowFact]:
    out: list[RowFact] = []
    i = 0
    n = len(lines)
    while i < n and len(out) < MAX_ROWS_PER_PAGE:
        found = _header_at(lines, i)
        if found is None:
            i += 1
            continue
        roles, cats, w_min, w_max = found
        # Try every width the header could have had and keep the one the DATA
        # supports. A table whose period cannot be established this way is
        # skipped, never read by position.
        best: tuple[int, list[RowFact]] | None = None
        for width in range(w_min, min(w_max, MAX_HEADER_WIDTH) + 1):
            rows = _read_rows(lines, i + width, width, roles, cats,
                              MAX_ROWS_PER_PAGE - len(out))
            if best is None or len(rows) > len(best[1]):
                best = (width, rows)
        if best and len(best[1]) >= MIN_TABLE_ROWS:
            out.extend(best[1])
            i += best[0] + best[0] * len(best[1])
        else:
            i += 1
    return out


# ---------------------------------------------------------------------------

def extract_course_rows(text: str) -> list[RowFact]:
    """Every programme row one page asserts. Never raises."""
    try:
        if not text:
            return []
        lines = text[:PAGE_TEXT_BUDGET].split("\n")
        if len(lines) > MAX_LINES:
            lines = lines[:MAX_LINES]
        rows = parse_tables(lines) + parse_labelled(lines)
        return [r for r in rows if not r.empty]
    except Exception as exc:  # noqa: BLE001 - one page must never kill a run
        print(f"[courses] parser raised: {str(exc)[:120]}", file=sys.stderr)
        return []


# ---------------------------------------------------------------------------
# Matching a page label to catalogue rows
# ---------------------------------------------------------------------------

SPEC_MIN_SCORE = 0.6          # containment, not Jaccard — see _score
SPEC_MARGIN = 0.15            # a winner must beat the runner-up by this much
MAX_DEGREE_FANOUT = 60        # one page line may not rewrite an entire faculty


@dataclass
class Entry:
    course_id: str
    family: str
    spec: set[str]
    display: str


def _score(a: set[str], b: set[str]) -> float:
    """Containment, so "CSE" matching "Computer Science and Engineering (CSE)"
    is not punished for the longer side carrying more words."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


class CourseIndex:
    """One college's catalogue rows, indexed by degree family.

    `title` is the authority on the family: it is a clean short code across the
    corpus (BTech, BSc, MTech, …). `full_name` supplies the specialisation, but
    only when the two AGREE — the catalogue contains rows whose title says MBA
    and whose full_name says "Master of Computer Applications", and a row that
    contradicts itself cannot be matched safely by either half.
    """

    def __init__(self, rows: list[dict]):
        self.by_family: dict[str, list[Entry]] = {}
        self.ambiguous = 0
        for r in rows:
            title, full = (r.get("title") or "").strip(), (r.get("full_name") or "").strip()
            fam = _fam_key(title)
            if not fam:
                continue
            if full:
                fam_full = family_of(full)
                if fam_full and fam_full != fam and _fam_key(fam_full) != fam:
                    self.ambiguous += 1
                    continue
            spec = spec_tokens(full or title, family_of(full or title))
            self.by_family.setdefault(fam, []).append(
                Entry(r["course_id"], fam, spec, full or title))

    @property
    def n_courses(self) -> int:
        return sum(len(v) for v in self.by_family.values())

    def match(self, label: str) -> tuple[list[str], str]:
        """(course_ids, kind). kind is 'exact', 'degree', or a refusal reason."""
        fam = family_of(label)
        if not fam:
            return [], "no_degree_in_label"
        cands = self.by_family.get(fam)
        if not cands:
            return [], "degree_not_in_catalogue"

        spec = spec_tokens(label, fam)
        distinctive = spec - GENERIC_SPEC
        if distinctive and not distinctive <= QUALIFIER_ONLY:
            scored = sorted(((_score(spec, e.spec), e) for e in cands),
                            key=lambda x: -x[0])
            best = scored[0]
            if best[0] < SPEC_MIN_SCORE:
                return [], "specialisation_not_in_catalogue"
            if not (distinctive & best[1].spec):
                return [], "only_generic_words_matched"
            # DUPLICATE catalogue rows are not an ambiguity. The client's bulk
            # import left several identical rows per programme at some colleges
            # (IIM Kashipur carries three rows all reading "Executive Master of
            # Business Administration in Analytics"), and refusing a tie there
            # would leave every one of them empty forever while the page states
            # the answer plainly. Rows that tie AND describe the same programme
            # all get the value; a tie between DIFFERENT programmes still
            # refuses, because then we genuinely cannot tell which was meant.
            tied = [e for score, e in scored if best[0] - score < SPEC_MARGIN]
            if len(tied) > 1 and any(e.spec != best[1].spec for e in tied):
                return [], "ambiguous_specialisation"
            return [e.course_id for e in tied], "exact"

        # The label names the degree and nothing else ("B.Tech", "MBA (Regular)").
        # That IS a statement about every row of that degree at this college —
        # provided the page never states a second value for the same degree,
        # which the caller checks before anything is written.
        if len(cands) > MAX_DEGREE_FANOUT:
            return [], "degree_fanout_too_large"
        return [e.course_id for e in cands], "degree"


# ---------------------------------------------------------------------------
# One college's harvest, reconciled across its pages
# ---------------------------------------------------------------------------

FIELDS = ("duration_years", "total_seats", "study_mode", "fees_by_category")


@dataclass
class Claim:
    value: object
    verbatim: str
    url: str
    stated_year: str | None
    confidence: float
    kind: str                      # 'exact' | 'degree'


class Reconciler:
    """Collects claims per (course_id, field) and refuses to pick a winner.

    Two pages on one site disagreeing about a duration means the SITE is
    ambiguous. Choosing one silently would hide that from the student, so the
    field is dropped and counted. An exact-match claim does beat a degree-level
    one for the same course — that is not a conflict, it is a more specific
    statement about the same thing.
    """

    def __init__(self):
        self.claims: dict[tuple[str, str], list[Claim]] = {}
        self.conflicts = 0

    def add(self, course_id: str, field: str, claim: Claim) -> None:
        self.claims.setdefault((course_id, field), []).append(claim)

    def resolve(self) -> dict[tuple[str, str], Claim]:
        out: dict[tuple[str, str], Claim] = {}
        for key, claims in self.claims.items():
            exact = [c for c in claims if c.kind == "exact"]
            pool = exact or claims
            distinct = {json.dumps(c.value, sort_keys=True, default=str) for c in pool}
            if len(distinct) > 1:
                self.conflicts += 1
                continue
            # Prefer the claim carrying a stated year, then the highest
            # confidence: same value, better provenance.
            pool.sort(key=lambda c: (c.stated_year is not None, c.confidence),
                      reverse=True)
            out[key] = pool[0]
        return out


# ---------------------------------------------------------------------------
# Page selection
# ---------------------------------------------------------------------------

def pick_course_pages(base_url: str, links: list[tuple[str, str]],
                      limit: int = MAX_PAGES_PER_COLLEGE) -> list[str]:
    """Same-origin pages likely to carry a programme table, best first."""
    origin = urlparse(base_url).netloc
    ranked: list[tuple[int, str]] = []
    seen: set[str] = set()
    for href, anchor in links:
        if not href or href.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue
        absolute = urljoin(base_url, href).split("#")[0]
        p = urlparse(absolute)
        if p.scheme not in ("http", "https") or p.netloc != origin:
            continue
        if p.path.lower().endswith(".pdf") or absolute in seen:
            continue
        hay = f"{p.path.lower()} {(anchor or '').lower()}"
        if COURSE_PAGE_DENY.search(hay):
            continue
        for rank, key in enumerate(COURSE_PAGE_KEYS):
            if key in hay:
                seen.add(absolute)
                ranked.append((rank, absolute))
                break
    ranked.sort(key=lambda x: x[0])
    return [u for _, u in ranked[:limit]]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

DDL = (
    "ALTER TABLE courses ADD COLUMN IF NOT EXISTS total_seats INTEGER",
    "ALTER TABLE courses ADD COLUMN IF NOT EXISTS study_mode TEXT",
    "ALTER TABLE courses ADD COLUMN IF NOT EXISTS duration_years REAL",
    "ALTER TABLE courses ADD COLUMN IF NOT EXISTS fees_by_category JSONB",
    """CREATE TABLE IF NOT EXISTS course_crawl_state (
           college_id   TEXT PRIMARY KEY,
           status       TEXT,
           last_attempt TIMESTAMPTZ,
           attempts     INTEGER DEFAULT 0,
           pages        INTEGER DEFAULT 0,
           courses_hit  INTEGER DEFAULT 0)""",
)

# COALESCE(%s, col): a run never nulls a column it found nothing for, so a
# second pass over a site that has since dropped its table cannot erase what the
# first pass read. It DOES overwrite with a newer reading, which is the point.
COURSE_UPDATE = """
UPDATE courses SET
    duration_years   = COALESCE(%s, duration_years),
    total_seats      = COALESCE(%s, total_seats),
    study_mode       = COALESCE(%s, study_mode),
    fees_by_category = COALESCE(%s::jsonb, fees_by_category)
WHERE course_id = %s"""

# One row per (college, course, field). The PK is (college_id, field, source),
# so the course id has to live inside `field` — that is what keeps two courses
# at one college from overwriting each other, and it keeps the nightly
# "fetched_at < now() - 180 days" scan working unchanged.
FIELD_UPSERT = """
INSERT INTO college_field
    (college_id, field, value, value_num, source, source_url, stated_year,
     fetched_at, confidence)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (college_id, field, source) DO UPDATE SET
    value       = EXCLUDED.value,
    value_num   = EXCLUDED.value_num,
    source_url  = EXCLUDED.source_url,
    stated_year = EXCLUDED.stated_year,
    fetched_at  = EXCLUDED.fetched_at,
    confidence  = EXCLUDED.confidence"""

STATE_UPSERT = """
INSERT INTO course_crawl_state
    (college_id, status, last_attempt, attempts, pages, courses_hit)
VALUES (%s, %s, %s, 1, %s, %s)
ON CONFLICT (college_id) DO UPDATE SET
    status       = EXCLUDED.status,
    last_attempt = EXCLUDED.last_attempt,
    attempts     = course_crawl_state.attempts + 1,
    pages        = EXCLUDED.pages,
    courses_hit  = EXCLUDED.courses_hit"""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _pool():
    pool = getattr(get_backend(), "_pool", None)
    if pool is None:
        raise RuntimeError("harvest_courses needs the Postgres connection pool "
                           "and the active backend does not expose one")
    return pool


class CourseWriter:
    """Buffers a batch and lands courses + college_field + state in ONE
    transaction, for the same reason web_enrich.HarvestWriter does: a value in
    `courses` without its `college_field` row is an undated number that no
    refresh will ever revisit, which is precisely the failure this table exists
    to prevent."""

    def __init__(self, every: int = 20, dry_run: bool = False):
        self.every = every
        self.dry_run = dry_run
        self._courses: list[tuple] = []
        self._fields: list[tuple] = []
        self._state: list[tuple] = []
        self._done = 0

    def course(self, course_id: str, dur, seats, mode, fees) -> None:
        self._courses.append(
            (dur, seats, mode,
             json.dumps(fees, ensure_ascii=False) if fees else None, course_id))

    def field(self, cid: str, name: str, value: str, num: float | None,
              url: str, year: str | None, conf: float) -> None:
        self._fields.append((cid, name, value[:400], num, SOURCE, url[:500],
                             year, _now(), round(float(conf), 3)))

    def mark(self, cid: str, status: str, pages: int, hit: int) -> None:
        self._state.append((cid, status, _now(), pages, hit))
        self._done += 1

    @property
    def due(self) -> bool:
        return self._done >= self.every

    async def flush(self) -> None:
        courses, self._courses = self._courses, []
        fields, self._fields = self._fields, []
        state, self._state = self._state, []
        self._done = 0
        if self.dry_run or not (courses or fields or state):
            return
        await asyncio.to_thread(self._commit, courses, fields, state)

    @staticmethod
    def _commit(courses, fields, state) -> None:
        with _pool().connection() as c:      # commits on exit, rolls back on error
            cur = c.cursor()
            if courses:
                cur.executemany(COURSE_UPDATE, courses)
            if fields:
                cur.executemany(FIELD_UPSERT, fields)
            if state:
                cur.executemany(STATE_UPSERT, state)


# ---------------------------------------------------------------------------
# The harvest
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def anchored(verbatim: str, page_norm: str) -> bool:
    """Is the cell this value was read from literally on the page?

    verify_facts.anchor_numbers deliberately ignores integers under 100 and
    tokens of three digits or fewer — they are summary artefacts on a prose
    page, and anchoring them produced false rejections. But an intake of "8" and
    a duration of "4 years" are exactly that shape, so the project's headline
    rule would go UNCHECKED for the two fields this module exists to fill.
    This closes that gap: the cell text itself, not the number derived from it,
    has to be findable in the page. It is true by construction today, which is
    the point — a future parser that computes instead of reading fails here
    instead of writing a number nobody published.
    """
    if not verbatim:
        return False
    return _WS_RE.sub(" ", verbatim).strip().lower() in page_norm


def _fact_for_verifier(rows: list[RowFact], college: str, url: str) -> dict:
    """Shape the page's rows the way verify_facts.verify expects.

    Everything the gate needs is here: the numbers to anchor verbatim against
    the page text, and enough prose for the identity check to work on.
    """
    facts = {r.label[:90]: r.evidence() for r in rows[:40] if r.evidence()}
    return {
        "summary": (f"{college[:60]} lists course-level detail for "
                    f"{len(facts)} programme(s) on this page."),
        "facts": facts,
        "confidence": 0.85 if len(facts) >= 3 else 0.7,
    }


async def harvest_one(client, pol, writer: CourseWriter, college: dict,
                      index: CourseIndex, fc=None) -> dict:
    """Fetch one college's course pages, match, reconcile, queue the writes."""
    cid, name = college["college_id"], college["name"]
    site = normalise_site(college.get("website_url"))
    stats = {"pages": 0, "rows": 0, "matched": 0, "courses": 0, "rejected": 0,
             "unanchored": 0, "conflicts": 0, "uncertain": 0, "unmatched": {},
             "status": "empty"}
    if not site:
        stats["status"] = "no_site"
        writer.mark(cid, "no_site", 0, 0)
        return stats

    home = await fetch(client, pol, site, fc)
    if home is None:
        stats["status"] = "blocked"
        writer.mark(cid, "blocked", 0, 0)
        return stats
    stats["pages"] += 1
    home_text, links = parse_html(home)

    pages: list[tuple[str, str]] = [(site, home_text)]      # (url, text)
    for url in pick_course_pages(site, links):
        # No Firecrawl for sub-pages. web_enrich measured that spending credits
        # here costs 11 per 25 colleges and collapses throughput; a credit is
        # only worth it when the SITE blocks outright, which the homepage fetch
        # above already handles.
        html = await fetch(client, pol, url, None)
        if html is None:
            continue
        stats["pages"] += 1
        text, _ = parse_html(html)
        pages.append((url, text))

    # Prospectus / seat-matrix PDFs. pdf_harvest measured that 28% of college
    # sites link a PDF about fees or the prospectus against 14% carrying a rupee
    # figure in HTML, and an intake table is a document far more often than a
    # web page. Only run it when the HTML pass came up short — a PDF is the most
    # expensive request in this job.
    if sum(len(extract_course_rows(t)) for _, t in pages) < 3:
        try:
            for kind, pdf_url in find_pdf_links(site, links, limit=4):
                if kind not in ("courses", "admission", "fees"):
                    continue
                data, _note = await fetch_pdf(client, pol, pdf_url)
                if data is None:
                    continue
                stats["pages"] += 1
                text, _note = pdf_to_text(data)
                if text:
                    pages.append((pdf_url, text))
                if sum(1 for u, _ in pages if u.lower().endswith(".pdf")) \
                        >= MAX_PDFS_PER_COLLEGE:
                    break
        except Exception as exc:  # noqa: BLE001 - a PDF must not kill a college
            print(f"[courses] pdf step failed for {name[:30]}: {str(exc)[:90]}",
                  file=sys.stderr)

    rec = Reconciler()
    # Degree-level claims are collected per (family, field) FIRST, so a page
    # that states two durations for one degree drops that degree entirely
    # instead of writing whichever line came last.
    degree_claims: dict[tuple[str, str], list[tuple[Claim, list[str]]]] = {}

    for url, text in pages:
        rows = extract_course_rows(text)
        if not rows:
            continue
        stats["rows"] += len(rows)

        # THE GATE. Same verifier every other harvest in this project goes
        # through: every number must appear verbatim in this page's text, and
        # the page must corroborate the college's identity. A course page on a
        # domain that turns out to belong to someone else is the wrong-entity
        # failure — the numbers would be real and simply not this college's.
        fact = _fact_for_verifier(rows, name, url)
        checked = verify_fact(fact, text, name, college.get("city"))
        if checked.verdict == "REJECTED":
            stats["rejected"] += 1
            print(f"[courses] REJECTED {url[:60]}: "
                  f"{'; '.join(checked.reasons)[:100]}", file=sys.stderr)
            continue
        # UNCERTAIN costs this page its MONEY only. The verifier's remaining
        # doubt is almost always check_units — a rupee figure whose period the
        # page states and the fact dropped — and a fee whose period is unclear
        # is the exact defect that made every catalogue fee wrong by 2-4x. A
        # duration or an intake carries no period, so that same doubt says
        # nothing about them and dropping the whole page would discard the
        # field this module exists to fill over an unrelated complaint.
        money_ok = checked.verdict == "CONFIRMED"
        if not money_ok:
            stats["uncertain"] = stats.get("uncertain", 0) + 1
        year = stated_year(text)
        conf = checked.confidence
        page_norm = _WS_RE.sub(" ", text).strip().lower()

        for row in rows:
            ids, kind = index.match(row.label)
            if not ids:
                stats["unmatched"][kind] = stats["unmatched"].get(kind, 0) + 1
                continue
            stats["matched"] += 1
            payload = [
                ("duration_years", row.duration_years, row.duration_raw),
                ("total_seats", row.seats, row.seats_raw),
                ("study_mode", row.mode, row.mode_raw),
                # TWO categories or none. One is not a breakdown — it is a fee,
                # and a fee belongs to extractors/fees.py, which knows how to
                # keep its period attached. The header check already demanded
                # two category COLUMNS; this demands two that actually parsed,
                # which is what stopped a lone "Management" column from being
                # published as a category breakdown of one.
                ("fees_by_category",
                 row.fees_by_category if len(row.fees_by_category) >= 2 else None,
                 row.evidence()),
            ]
            for fname, value, verbatim in payload:
                if value is None:
                    continue
                # Rule 1, checked rather than assumed. fees_by_category anchors
                # per amount, because its evidence string is assembled from
                # several cells and is not itself a run of page text.
                if fname == "fees_by_category":
                    if not money_ok:
                        continue
                    if not all(anchored(v, page_norm) for v in value.values()):
                        stats["unanchored"] = stats.get("unanchored", 0) + 1
                        continue
                elif not anchored(verbatim, page_norm):
                    stats["unanchored"] = stats.get("unanchored", 0) + 1
                    print(f"[courses] UNANCHORED {fname} {verbatim!r} not on "
                          f"{url[:60]}", file=sys.stderr)
                    continue
                claim = Claim(value, verbatim or "", url, year, conf, kind)
                if kind == "degree":
                    fam = family_of(row.label) or ""
                    degree_claims.setdefault((fam, fname), []).append((claim, ids))
                else:
                    rec.add(ids[0], fname, claim)

    # Fan a degree-level claim out only when the whole site agreed on it.
    for (fam, fname), pairs in degree_claims.items():
        distinct = {json.dumps(c.value, sort_keys=True, default=str) for c, _ in pairs}
        if len(distinct) > 1:
            rec.conflicts += 1
            continue
        claim, ids = pairs[0]
        for course_id in ids:
            rec.add(course_id, fname, claim)

    resolved = rec.resolve()
    stats["conflicts"] = rec.conflicts
    if not resolved:
        writer.mark(cid, "empty", stats["pages"], 0)
        return stats

    per_course: dict[str, dict] = {}
    for (course_id, fname), claim in resolved.items():
        per_course.setdefault(course_id, {})[fname] = claim
    for course_id, got in per_course.items():
        writer.course(
            course_id,
            got["duration_years"].value if "duration_years" in got else None,
            got["total_seats"].value if "total_seats" in got else None,
            got["study_mode"].value if "study_mode" in got else None,
            got["fees_by_category"].value if "fees_by_category" in got else None,
        )
        for fname, claim in got.items():
            num = None
            if fname == "duration_years":
                num = float(claim.value)
            elif fname == "total_seats":
                num = float(claim.value)
            value = (json.dumps(claim.value, ensure_ascii=False)
                     if isinstance(claim.value, dict) else str(claim.value))
            # The verbatim page text is carried alongside the parsed value, so a
            # "where did 4 years come from" question is answered by a quotation
            # rather than a URL someone has to re-read and hope has not changed.
            writer.field(cid, f"course.{course_id}.{fname}",
                         f"{value} | {claim.verbatim}"[:400], num,
                         claim.url, claim.stated_year,
                         claim.confidence * (1.0 if claim.kind == "exact" else 0.9))
    stats["courses"] = len(per_course)
    stats["status"] = "ok"
    writer.mark(cid, "ok", stats["pages"], len(per_course))
    return stats


TIER1_SQL = """
SELECT p.college_id, c.name, c.city, c.state, c.website_url
FROM college_priority p JOIN colleges c USING(college_id)
WHERE p.tier = 1
  AND c.website_url IS NOT NULL AND TRIM(c.website_url) <> ''
  {resume}
ORDER BY p.score DESC, c.name
{limit}"""


def load_targets(limit: int | None, redo: bool) -> list[dict]:
    resume = "" if redo else (
        "AND p.college_id NOT IN (SELECT college_id FROM course_crawl_state "
        "WHERE status IN ('ok', 'empty', 'no_site'))")
    sql = TIER1_SQL.format(resume=resume,
                           limit=f"LIMIT {int(limit)}" if limit else "")
    return [dict(r) for r in get_backend().q(sql)]


def load_index(college_ids: list[str]) -> dict[str, CourseIndex]:
    if not college_ids:
        return {}
    rows = get_backend().q(
        "SELECT college_id, course_id, title, full_name FROM courses "
        "WHERE college_id = ANY(?)", [college_ids])
    grouped: dict[str, list[dict]] = {}
    for r in rows:
        grouped.setdefault(r["college_id"], []).append(dict(r))
    return {cid: CourseIndex(rs) for cid, rs in grouped.items()}


async def run(limit: int | None = None, concurrency: int = 6, delay: float = 1.5,
              dry_run: bool = False, redo: bool = False,
              firecrawl_reserve: int = 100) -> dict:
    _pool()                                   # fail now, not after an hour
    # Runs on a dry run too. These are all IF NOT EXISTS and create no data —
    # and load_targets' resume clause selects FROM course_crawl_state, so
    # skipping them would make --dry-run fail on a store that has never run
    # this job, which is exactly the store a dry run is for.
    be = get_backend()
    for stmt in DDL:
        be.execute(stmt)

    targets = load_targets(limit, redo)
    if not targets:
        print("[courses] nothing to do — every tier-1 college is already "
              "attempted (use --redo to force)", file=sys.stderr)
        return {}
    index = load_index([t["college_id"] for t in targets])
    print(f"[courses] {len(targets)} tier-1 colleges, "
          f"{sum(i.n_courses for i in index.values()):,} catalogue courses, "
          f"{sum(i.ambiguous for i in index.values()):,} rows skipped as "
          f"self-contradictory", file=sys.stderr)

    pol = Politeness(delay=delay)
    writer = CourseWriter(dry_run=dry_run)
    fc = FirecrawlBudget(reserve=firecrawl_reserve)
    totals = {"colleges": 0, "nothing": 0, "ok": 0, "blocked": 0, "no_site": 0,
              "pages": 0, "rows": 0, "matched": 0, "courses": 0, "conflicts": 0,
              "rejected": 0, "unanchored": 0, "uncertain": 0}
    unmatched: dict[str, int] = {}
    samples: list[str] = []
    lock = asyncio.Lock()
    queue: asyncio.Queue = asyncio.Queue()
    for t in targets:
        queue.put_nowait(t)

    async with httpx.AsyncClient(headers=HEADERS, timeout=25.0,
                                 follow_redirects=True, verify=False) as client:
        await fc.prime(client)

        async def worker() -> None:
            while True:
                try:
                    college = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    idx = index.get(college["college_id"]) or CourseIndex([])
                    st = await harvest_one(client, pol, writer, college, idx,
                                           fc if fc.enabled else None)
                except Exception as exc:  # noqa: BLE001 - one college, not the run
                    print(f"[courses] {college['name'][:34]} failed: "
                          f"{str(exc)[:110]}", file=sys.stderr)
                    st = {"status": "error", "pages": 0, "rows": 0, "matched": 0,
                          "courses": 0, "unmatched": {}}
                async with lock:
                    totals["colleges"] += 1
                    for k in ("pages", "rows", "matched", "courses", "conflicts",
                              "rejected", "unanchored", "uncertain"):
                        totals[k] += st.get(k, 0)
                    totals[st["status"]] = totals.get(st["status"], 0) + 1
                    if not st.get("courses"):
                        totals["nothing"] += 1
                    for k, v in (st.get("unmatched") or {}).items():
                        unmatched[k] = unmatched.get(k, 0) + v
                    if st.get("courses") and len(samples) < 40:
                        samples.append(f"{college['name'][:44]}: "
                                       f"{st['courses']} course(s) filled from "
                                       f"{st['pages']} page(s)")
                    if totals["colleges"] % 10 == 0:
                        print(f"[courses] {totals['colleges']}/{len(targets)} "
                              f"colleges · {totals['courses']} course rows filled "
                              f"· {totals['nothing']} yielded nothing",
                              file=sys.stderr)
                    if writer.due:
                        await writer.flush()

        await asyncio.gather(*(worker() for _ in range(max(1, concurrency))))
    await writer.flush()

    totals["unmatched_reasons"] = unmatched
    totals["samples"] = samples
    return totals


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report() -> None:
    be = get_backend()
    print("=" * 70)
    print("COURSE-LEVEL DETAIL — tier 1")
    print("=" * 70)
    t1 = be.one("SELECT COUNT(*) FROM courses cr JOIN college_priority p "
                "USING(college_id) WHERE p.tier = 1") or 0
    for col in FIELDS:
        n = be.one(f"SELECT COUNT(*) FROM courses cr JOIN college_priority p "
                   f"USING(college_id) WHERE p.tier = 1 AND cr.{col} IS NOT NULL") or 0
        pct = 100.0 * n / t1 if t1 else 0.0
        print(f"  {col:<18} {n:>7,} / {t1:,}  {pct:5.1f}%")
    cols = be.one("SELECT COUNT(DISTINCT college_id) FROM college_field "
                  "WHERE field LIKE 'course.%'") or 0
    rows = be.one("SELECT COUNT(*) FROM college_field WHERE field LIKE 'course.%'") or 0
    print(f"\n  college_field    {rows:,} tracked values over {cols:,} colleges")
    try:
        st = be.q("SELECT status, COUNT(*) n FROM course_crawl_state GROUP BY 1 "
                  "ORDER BY 2 DESC")
        print("  crawl outcome:   " + "  ".join(f"{r['status']}={r['n']}" for r in st))
    except Exception:  # noqa: BLE001 - table may not exist before the first run
        pass
    print("\n  sample of what was written:")
    for r in be.q("""SELECT c.name, cr.title, cr.full_name, cr.duration_years,
                            cr.total_seats, cr.study_mode, f.value, f.source_url,
                            f.stated_year
                     FROM courses cr
                     JOIN colleges c USING(college_id)
                     LEFT JOIN college_field f
                            ON f.college_id = cr.college_id
                           AND f.field = 'course.' || cr.course_id || '.duration_years'
                     WHERE cr.duration_years IS NOT NULL
                     ORDER BY random() LIMIT 12"""):
        print(f"    {r['name'][:32]:<34} {(r['full_name'] or r['title'])[:38]:<40} "
              f"{r['duration_years']}y seats={r['total_seats']} "
              f"mode={r['study_mode']}")
        if r["value"]:
            print(f"        {str(r['value'])[:96]}")
            print(f"        {str(r['source_url'])[:88]}  year={r['stated_year']}")


# ---------------------------------------------------------------------------
# Self-test — real captured page text, plus the cases that decide calibration.
# Run: python -m rag_core.sources.harvest_courses --selftest
# ---------------------------------------------------------------------------

REAL_JADAVPUR = """Programs Offered
Choose Faculty
Faculty of Arts
Choose Department
Choose Level
Advanced Diploma in Industrial Safety & Environmental Management
Offered By : School Of Environmental Studies
Intake: 30
Click here for Syllabus and Curriculam
B.Ed. in Special Education (Locomotor Impairment and Neuro-Muscular Disorder)
Offered By : Department of Adult, Continuing Education and Extension
Duration: 1 year
Intake: 28 students
Click here for Syllabus and Curriculam
BA Arts (H) Sociology
Offered By : Department of Sociology
Duration: 4 years
Click here for Syllabus and Curriculam
Bachelor of Architecture
Offered By : Department Of Architecture
Duration: 5 years
Intake: 40
Click here for Syllabus and Curriculam
Bachelor of Arts (Hons) in Bengali
Offered By : Department Of Bengali
Duration: 4
Intake: 55
Click here for Syllabus and Curriculam"""

REAL_SHARDA = """CERTIFICATE PROGRAMMES
Allied Health Sciences
Programmes
 Duration
 Tuition Fee (USD)
Certificate in Nursing Care Assistant
 1 Year
 2000
Certificate in Emergency Medical Technician
 1 Year
 2000
Certificate in Medical Imaging Technician
 1 Year
 2000
Lateral Entry Programs
Programmes
 Duration
 Tuition Fee (USD)
B.Tech. (Lateral Entry)
 3 Year
 3800
BBA (Lateral Entry)
 2 Year
 3200
BCA (Lateral Entry)
 2 Year
 3000
BMIT (Lateral Entry)
 2.5 Year
 2700
Note
**Including Accommodation & Food"""

SYNTH_SEAT_MATRIX = """Sanctioned Intake 2025-26
Course
 Duration
 Sanctioned Intake
 Mode
B.Tech Computer Science and Engineering
 4 Years
 180
 Full Time
B.Tech Mechanical Engineering
 4 Years
 60
 Full Time
M.Tech Structural Engineering
 2 Years
 18
 Full Time
MBA
 2 Years
 120
 Full Time"""

SYNTH_CATEGORY_FEES = """Fee Structure 2026-27
Course
 General
 OBC
 SC/ST
 NRI
B.Tech Civil Engineering
 1,25,000
 1,25,000
 62,500
 3,50,000
B.Tech Computer Science and Engineering
 1,45,000
 1,45,000
 72,500
 4,00,000"""

NEG_ENROLMENT = """Student Strength
Total students enrolled in the institute: 4,250
Boys 2,800 Girls 1,450
Faculty 310
Our campus is spread over 25 acres."""

NEG_TIMETABLE = """Academic Calendar
Duration: 6 weeks
Orientation Programme
Duration: 3 days
Industrial Visit
Duration: 2 days"""

NEG_NAV = """Home
About Us
Courses
Programmes
Admission
Contact Us
Fee Structure"""


def _selftest() -> int:  # pragma: no cover - developer tool
    cases = [
        ("REAL jadavpuruniversity.in/courses (labelled fields)", REAL_JADAVPUR, 5),
        ("REAL sharda.ac.in short-term programmes (column table)", REAL_SHARDA, 7),
        ("SYNTH seat matrix with mode", SYNTH_SEAT_MATRIX, 4),
        ("SYNTH category fee table", SYNTH_CATEGORY_FEES, 2),
        ("NEG enrolment counts are not intake", NEG_ENROLMENT, 0),
        ("NEG event durations are not course durations", NEG_TIMETABLE, 0),
        ("NEG nav only", NEG_NAV, 0),
    ]
    bad = 0
    for name, text, want in cases:
        rows = extract_course_rows(text)
        ok = len(rows) == want
        bad += not ok
        print(f"\n{'PASS' if ok else 'FAIL'}  {name}  -> {len(rows)} row(s), "
              f"expected {want}")
        for r in rows[:8]:
            print(f"      [{r.shape}] {r.label[:52]!r} dur={r.duration_years} "
                  f"({r.duration_raw}) seats={r.seats} mode={r.mode} "
                  f"cats={list(r.fees_by_category)}")

    # --- matching -------------------------------------------------------
    print("\n" + "-" * 66 + "\nMATCHER\n" + "-" * 66)
    idx = CourseIndex([
        {"course_id": "c1", "title": "BTech",
         "full_name": "Bachelor of Technology in Computer Science and Engineering"},
        {"course_id": "c2", "title": "BTech",
         "full_name": "Bachelor of Technology in Mechanical Engineering"},
        {"course_id": "c3", "title": "MBA",
         "full_name": "Master of Business Administration"},
        {"course_id": "c4", "title": "MBA",
         "full_name": "Master of Computer Applications"},   # self-contradictory
        {"course_id": "c5", "title": "BArch",
         "full_name": "Bachelor of Architecture"},
        # The bulk import's duplicate rows: same programme, two ids.
        {"course_id": "c6", "title": "EMBA",
         "full_name": "Executive Master of Business Administration in Analytics"},
        {"course_id": "c7", "title": "EMBA",
         "full_name": "Executive Master of Business Administration in Analytics"},
        {"course_id": "c8", "title": "MTech",
         "full_name": "Master of Technology in Structural Engineering"},
        {"course_id": "c9", "title": "MTech",
         "full_name": "Master of Technology in Thermal Engineering"},
    ])
    match_cases = [
        ("B.Tech Computer Science and Engineering", ["c1"], "exact"),
        ("Executive MBA in Analytics", ["c6", "c7"], "exact"),
        # "Engineering" distinguishes nothing, so this is a statement about the
        # degree, not about one of its specialisations.
        ("M.Tech Engineering", ["c8", "c9"], "degree"),
        ("B.Tech Mechanical Engineering", ["c2"], "exact"),
        ("Bachelor of Architecture", ["c5"], "degree"),
        # sharda.ac.in publishes "B.Tech. (Lateral Entry) — 3 Year". A regular
        # B.Tech is four years, so this must NOT reach the B.Tech rows.
        ("B.Tech. (Lateral Entry)", [], "specialisation_not_in_catalogue"),
        ("MBA (Full Time)", ["c3"], "degree"),
        ("Integrated M.Tech", [], "specialisation_not_in_catalogue"),
        ("B.A. (Hons)", [], "degree_not_in_catalogue"),
        ("B.Tech Aerospace Engineering", [], "specialisation_not_in_catalogue"),
        ("Diploma in Yoga", [], "degree_not_in_catalogue"),
        ("Orientation Programme", [], "no_degree_in_label"),
    ]
    for label, want_ids, want_kind in match_cases:
        ids, kind = idx.match(label)
        ok = sorted(ids) == sorted(want_ids) and kind == want_kind
        bad += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {label[:44]:<46} -> {kind} {ids}")
    print(f"      (index skipped {idx.ambiguous} self-contradictory catalogue row)")

    # --- conversions ----------------------------------------------------
    print("\n" + "-" * 66 + "\nCONVERSIONS\n" + "-" * 66)
    conv = [("4 Years", 4.0), ("8 Semesters", 4.0), ("6 months", 0.5),
            ("2.5 Year", 2.5), ("3 Yrs", 3.0), ("6 trimesters", None),
            ("12 years", None), ("Rs 40,000", None)]
    for cell, want in conv:
        got = parse_duration(cell)
        val = got[0] if got else None
        ok = val == want
        bad += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {cell:<16} -> {val} (want {want})")

    print(f"\n{'ALL PASS' if not bad else str(bad) + ' FAILURE(S)'}")
    return 1 if bad else 0


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m rag_core.sources.harvest_courses",
        description="Course-level detail for tier-1 colleges from their own sites.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=6,
                    help="workers ACROSS domains; pacing within a domain is fixed")
    ap.add_argument("--delay", type=float, default=1.5,
                    help="seconds between requests to one host")
    ap.add_argument("--dry-run", action="store_true",
                    help="crawl and match, write nothing")
    ap.add_argument("--redo", action="store_true",
                    help="re-attempt colleges already recorded in course_crawl_state")
    ap.add_argument("--firecrawl-reserve", type=int, default=100)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args(argv)

    if a.selftest:
        return _selftest()
    if a.report:
        report()
        return 0

    totals = asyncio.run(run(limit=a.limit, concurrency=a.concurrency,
                             delay=a.delay, dry_run=a.dry_run, redo=a.redo,
                             firecrawl_reserve=a.firecrawl_reserve))
    if not totals:
        return 0
    print("\n" + "=" * 70)
    print("DRY RUN — nothing was written" if a.dry_run else "WRITTEN")
    print("=" * 70)
    print(f"  colleges attempted     {totals['colleges']:,}")
    print(f"  yielded NOTHING        {totals['nothing']:,}   "
          f"<- the honest number")
    print(f"  blocked                {totals.get('blocked', 0):,}")
    print(f"  pages fetched          {totals['pages']:,}")
    print(f"  programme rows parsed  {totals['rows']:,}")
    print(f"  rows matched to course {totals['matched']:,}")
    print(f"  course rows filled     {totals['courses']:,}")
    print(f"  pages rejected by gate {totals['rejected']:,}")
    print(f"  values not anchored    {totals['unanchored']:,}")
    print(f"  pages UNCERTAIN (fees  {totals['uncertain']:,}")
    print(f"    dropped, rest kept)")
    print(f"  dropped as conflicting {totals['conflicts']:,}")
    if totals.get("unmatched_reasons"):
        print("\n  why rows did not match a catalogue course:")
        for k, v in sorted(totals["unmatched_reasons"].items(), key=lambda x: -x[1]):
            print(f"    {k:<34} {v:,}")
    for s in totals.get("samples", [])[:20]:
        print(f"    {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
