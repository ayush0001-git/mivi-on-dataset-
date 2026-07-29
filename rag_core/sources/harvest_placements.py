"""Harvest PLACEMENT statistics for tier-1 colleges, from the college's own site.

WHAT IT FILLS
    colleges.placement_pct, placement_year, package_highest_inr,
    package_average_inr, package_median_inr, top_recruiters, placement_source
    college_field   one dated, sourced row per field per college

WHY THIS MODULE IS THIN
Almost none of the hard work is here. The placement grammar already exists and
is self-tested (`extractors/placement.py`: 20 cases, 0 false positives) — it
reads a number only through a LABEL, vetoes "100% placement assistance"
marketing, vetoes "average salary of the top 25%" subset claims, vetoes faculty
vacancy pages reached through a "Career" nav link, and refuses to guess a batch
year when a page shows several. Fetching, robots.txt, per-domain pacing and the
Firecrawl rescue already exist in `web_enrich`. The gate already exists in
`verify_facts`. PDF reading and two-DPI OCR already exist in `pdf_harvest`/`ocr`.

So what this module actually does is the one job none of them do: turn a
DISPLAY STRING into a stored rupee integer, attach the year that dates it, and
refuse to store anything undated.

THE TWO WAYS THIS GOES WRONG, AND WHAT STOPS EACH

1. THE 100,000x ERROR.  An Indian placement page writes "45 LPA" and means
   Rs 45,00,000. Writing 45 into package_highest_inr is not a small error, it is
   a five-order-of-magnitude one, and it looks perfectly reasonable in a table.
   This module never does that arithmetic itself. `money_to_rupees()` runs the
   display string back through the extractor's OWN `_MONEY_PROSE` /
   `_canon_money` pair — the same code that produced it — so the LPA / Lakh /
   Crore conversion exists in exactly ONE place in the codebase and is exercised
   by that module's self-test rather than by a second copy written here.

2. THE UNDATED FIGURE.  "Highest package Rs 1.1 Cr" with no year reads as this
   year's number, gets quoted as this year's number, and may be from 2013 — the
   marketing trap named in the brief. Every rule below follows from that:

     * placement_pct and every package REQUIRE a year. No year, no write. The
       figure is not "stored without a date", it is not stored.
     * The year is the page's own batch token, or the year column of the NAAC
       five-year table, and never today's date. `verify_facts.stated_year()` is
       NOT used here: it is vouched by fee/admission vocabulary and returns
       None on placement pages (measured: None on all 57 tier-1 placement facts
       already in the corpus, several of which carry a real batch year).
     * A year dates a figure only if it is WRITTEN BESIDE IT. A page-level batch
       token is not a licence to date everything on the page — see
       `_batch_dates_figure`, which is the check four of the first eleven
       results needed and which no amount of reading the extracted values
       would have revealed.
     * A NAAC table dates the percentage it sits in, and dates NOTHING else. A
       package card elsewhere on the same page is a different claim from a
       different year, so it keeps its own year — see ROW CONSISTENCY below.
     * A figure covering SEVERAL batches ("last five years", "batches passed
       between 2022-2026") gets no year at all, because it has none: pinning it
       to the range's end year would republish a five-year record as this
       year's number.
     * An OLD year is stored, not suppressed. A 2013 record package labelled
       2013 is honest; the same figure labelled with nothing is the bug. Nirma
       University's Rs 88.20 Lakh is stored dated 2022 for exactly this reason.

ROW CONSISTENCY — why colleges and college_field can differ
`colleges` has ONE placement_year for the whole row, so everything written into
that row must share it: the row is one observation, of one batch, from one page.
A figure dated differently is dropped from the row rather than silently filed
under the wrong year. `college_field` has a stated_year PER FIELD and so keeps
the full truth, including the differently-dated figure. When the two disagree,
college_field is the complete record and the colleges row is the safe summary.

VERBATIM
`verify_facts.verify()` is the gate, unchanged and not reimplemented. But its
anchoring deliberately skips tokens containing "." and tokens under four digits
(dates and page numbers would otherwise cause false rejections) — which is
exactly the shape of a package figure: "45", "6.5", "88.20". So before a number
becomes a stored column, `_missing_tokens()` asserts that the token as written
is present on the page, reusing `verify_facts._variants` for the normalisation
rather than writing a second one. This does not replace the gate; it closes the
gate's known blind spot for the specific tokens this module writes.

POLITENESS
robots.txt via web_enrich.Politeness, 1.5 s between requests to one host,
concurrency ACROSS hosts only, and a hard ceiling of 1 homepage + 3 placement
pages + 2 PDFs per college.

Run:
    python -m rag_core.sources.harvest_placements --limit 40            # dry run
    python -m rag_core.sources.harvest_placements --limit 40 --sample 12
    python -m rag_core.sources.harvest_placements --commit
    python -m rag_core.sources.harvest_placements --report
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from dataclasses import dataclass, field as dc_field
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse

import httpx

# config BEFORE the backend, for the reason web_enrich documents: config is the
# only module that calls load_dotenv(), and store.backend reads MIVI_PG_DSN at
# import time. Wrong order and a long crawl writes into the wrong database.
from .. import config  # noqa: F401
from ..store.backend import get_backend
from . import verify_facts as vf
from .extractors import placement as pex
from .verify_facts import verify as verify_fact
from .web_enrich import (FACT_UPSERT, FETCH_TIMEOUT, HEADERS, FirecrawlBudget,
                         Politeness, _pool, fetch, normalise_site, parse_html)

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

TIER1_SQL = """
SELECT p.college_id, c.name, c.city, c.state, c.website_url, p.score
FROM college_priority p JOIN colleges c USING(college_id)
WHERE p.tier = 1
  AND c.website_url IS NOT NULL AND TRIM(c.website_url) <> ''
ORDER BY p.score DESC
"""

# Placement pages already located by an earlier sweep. Re-using a known-good URL
# costs nothing and skips a homepage-link guess that may not repeat.
KNOWN_URL_SQL = """
SELECT f.college_id, f.source_url
FROM college_web_facts f JOIN college_priority p USING(college_id)
WHERE f.kind = 'placement' AND p.tier = 1 AND f.source_url IS NOT NULL
"""

# `?`, not `%s`: this one goes through backend.q(), which rewrites `?` and
# escapes `%`. The UPDATE/UPSERT statements below go straight to a psycopg
# cursor and therefore use `%s`. Mixing the two conventions up is a run-time
# error, not a silent one, which is why they are commented where they differ.
DONE_SQL = """
SELECT DISTINCT college_id FROM college_field
WHERE source = 'college_site'
  AND field IN ('placement_pct','package_highest_inr','package_average_inr',
                'package_median_inr','top_recruiters')
  AND fetched_at > now() - make_interval(days => ?)
"""

# Link vocabulary. Bare "career" is deliberately ABSENT: on Indian college sites
# a "Careers" nav link is a faculty-vacancy page far more often than a student
# placement page, which is why placement.py carries a vacancy veto at all. No
# need to spend a request discovering that.
PLACEMENT_KEYS = (
    "placement", "placements", "training-placement", "training_and_placement",
    "training-and-placement", "tnp", "t&p", "campus-placement", "placement-cell",
    "placement-record", "placement-statistic", "placement-report",
    "placement-highlight", "our-recruiters", "recruiters", "career-services",
)

MAX_PLACEMENT_PAGES = 3      # per college, on top of the homepage
MAX_PDFS = 2                 # placement reports are usually one file
PAGE_TEXT_BUDGET = 60_000    # the extractor caps at 4,000 lines anyway

# A package figure's plausible rupee band, derived from the LPA envelope the
# verifier already owns (0.5 - 500 LPA) so there is one opinion about it.
PKG_MIN_INR = int(vf.BOUNDS["lpa"][0] * 100_000)      # Rs 50,000
PKG_MAX_INR = int(vf.BOUNDS["lpa"][1] * 100_000)      # Rs 5,00,00,000

PACKAGE_COLUMNS = (
    ("package_highest_inr", "highest_package"),
    ("package_average_inr", "average_package"),
    ("package_median_inr", "median_package"),
)

FIELD_COLUMNS = ("placement_pct", "package_highest_inr", "package_average_inr",
                 "package_median_inr", "top_recruiters")


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Display string -> stored number
# ---------------------------------------------------------------------------

def money_to_rupees(display: str) -> tuple[int, str] | None:
    """'INR 88.20 Lakh' -> (8_820_000, '88.20'). None if it is not a package.

    Returns the rupee integer AND the numeric token exactly as the page wrote
    it, because that token — not the converted value — is what has to be found
    verbatim on the page.

    The conversion is delegated to the extractor's own money grammar. That is
    the whole point: "45 LPA" means Rs 45,00,000 and re-deriving that here would
    put the 100,000x error in a second place where a second test would have to
    catch it.
    """
    m = pex._MONEY_PROSE.search(display or "")
    if m is None:
        return None
    g = m.groupdict()
    num = g.get("num") or g.get("num2")
    unit = g.get("unit") or g.get("unit2")
    if not num:
        return None
    got = pex._canon_money(g.get("cur"), num, unit, g.get("period"))
    if got is None:
        return None
    _disp, rupees, _unit = got
    rupees = int(round(rupees))
    if not (PKG_MIN_INR <= rupees <= PKG_MAX_INR):
        return None
    return rupees, num


def pct_to_number(display: str) -> tuple[float, str] | None:
    """'95.5%' -> (95.5, '95.5'), bounded by the verifier's percent envelope."""
    m = re.search(r"(\d{1,3}(?:\.\d+)?)\s*%?", str(display or ""))
    if m is None:
        return None
    tok = m.group(1)
    try:
        val = float(tok)
    except ValueError:
        return None
    lo, hi = vf.BOUNDS["percent"]
    if not (lo <= val <= hi):
        return None
    return val, tok


_YEAR_TOKEN = re.compile(r"^(20[0-4]\d)(?:\s*[-/]\s*(20[0-4]\d|\d{2}))?$")

# How long a degree can run. "Batch 2020-2023" is a three-year cohort, not a
# malformed academic year, and treating it as malformed is what made SR
# University's CONFIRMED "51 LPA / 6.5 LPA" page read as undated and get
# discarded. Beyond this a span is a validity range or an accreditation period,
# not a graduating batch.
MAX_BATCH_SPAN = 6


def normalise_year(raw: str | None) -> str | None:
    """'2024-2025' -> '2024-25', '2020-2023' -> '2023', '2022' -> '2022'.

    Two different tokens mean two different things on an Indian placement page,
    and collapsing them would lose a year or invent one:

      * A ONE-year span is an ACADEMIC YEAR ("Placements 2024-25"). Kept in the
        page's own shape, because the client's refresh compares it against what
        the college publishes next cycle and "2024" would lose which half of the
        pair was meant.
      * A LONGER span is a COHORT ("Batch 2020-2023" — admission year to
        graduation year). Those students were placed in the FINAL year, so the
        end year is the placement year. Storing "2020-2023" here would date the
        figure to 2020 for anything that sorts or compares it, which is three
        years too early.

    The page's exact token is never lost: it stays in the college_web_facts
    payload as `batch`, so the raw wording is always auditable.
    """
    if raw is None:
        return None
    m = _YEAR_TOKEN.match(str(raw).strip())
    if m is None:
        return None
    start = int(m.group(1))
    cap = _now().year + 1
    if not (2000 <= start <= cap):
        return None
    tail = m.group(2)
    if not tail:
        return str(start)
    end = int(tail) if len(tail) == 4 else int(str(start)[:2] + tail)
    span = end - start
    if span < 0 or span > MAX_BATCH_SPAN:
        return None
    if not (2000 <= end <= cap):
        return None
    if span <= 1:
        return f"{start}-{str(end)[-2:]}" if span == 1 else str(start)
    return str(end)


# How far from a figure the batch token may sit and still be its year.
#
# CALIBRATED, NOT GUESSED. Measured on four tier-1 pages that this module was
# about to write:
#   nirmauni.ac.in   "placement in 2022. | INR 88.20 Lakhs"      same line block
#   upes.ac.in       "92% | Placements in 2024-25*"              1 line
#   sru.edu.in       "Placements AY 2020-2023 | ... | 51 LPA"    2 lines
#   sit.ac.in        "46.54 LPA" vs the nearest year token       183 lines  <-- REJECT
#
# sit.ac.in is why this exists. Its placement figures carry no year at all, and
# placement.py's LAST-RESORT batch fallback dated them "2026-27" — a token that
# appears only in "Faculty Recruitment Updates ... Admissions open for 2026-27",
# a faculty-vacancy notice. `_PLACEMENT_WORD` matches "Recruitment" there, so an
# unrelated hiring advert 183 lines away became the year of the placement
# statistics. The same page-level batch was also being stamped onto upes.ac.in's
# "1.3Cr Highest Package*" — footnoted "*secured by an Alumnus" — which sits 19
# lines from the only year on the page.
#
# Both are the failure rule 2 exists to prevent, and neither is a wrong NUMBER:
# they are correct numbers wearing a year the page never gave them, which is the
# form of this error that survives review because every figure checks out.
BATCH_NEAR_LINES = 8

# A figure summarising SEVERAL batches has no single placement_year, and forcing
# one on it is the "decade-old record package" trap wearing this year's date.
# juet.ac.in states "Placement Highlights - Last Five Years (Batches passed
# between 2022-2026)" directly above "44.14 LPA / Highest Package": a five-year
# record, which this module was about to store as the 2026 figure.
#
# There is no honest single year to give these, and placement_year cannot hold a
# range, so they are refused rather than flattened.
_AGGREGATE_VETO = re.compile(
    r"\blast\s+(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+years?\b"
    r"|\b(?:over|during)\s+the\s+(?:last|past)\s+\w+\s+years?\b"
    r"|\bbatch(?:es)?\s+(?:passed|graduat\w+)\s+(?:between|from)\b"
    r"|\b(?:cumulative|aggregate|combined|all[-\s]time|ever\s+recorded)\b"
    r"|\b(?:five|5|three|3|ten|10)[-\s]year\s+(?:placement\s+)?"
    r"(?:record|summary|highlight|average|trend)", re.I)

# What must appear ON the figure's own line for that line to BE the figure
# rather than a coincidence. walchandsangli.ac.in is why: the average package
# "Rs 12 LPA" was dated 2026-27 because "12" also occurs in an unrelated notice,
# "AICTE ATAL FDP ... from 7-12 September 2026", which sits beside an admissions
# guideline headed "(2026-27)". A bare numeric substring match found that date
# and treated it as the package. The unit is what distinguishes a package from
# a day of the month.
_UNIT_HINTS = (
    (re.compile(r"\b(?:lpa|l\.p\.a|cpa)\b", re.I), ("lpa", "l.p.a", "cpa")),
    (re.compile(r"\bcrores?\b|\bcr\b", re.I), ("crore", "cr")),
    (re.compile(r"\blakh?s?\b|\blacs?\b", re.I), ("lakh", "lac")),
)


# A year token sitting on an ADMISSIONS, FEE, COURSE or VACANCY line cannot date
# a placement figure, however close it is. Proximity alone is not enough because
# tag-stripping flattens a homepage's stat strip and its news ticker into
# neighbouring lines, so an admissions banner lands within a few lines of the
# placement cards it has nothing to do with.
#
# All three of these were COMMITTED before this veto existed, and every one of
# them is a correct number wearing a year the page never gave it:
#   drait.edu.in   "NOTIFICATION - Fee Structure ... for the Academic year 2026-27"
#   juet.ac.in     "95% Placement Rate | ... | Admissions 2026-27 Open"
#   svecw.edu.in   "59.28 LPA Highest Package | ... | SVECW Courses AY 2026-27"
#
# "Placements in 2026", "Batch 2024-25" and "Class of 2025" carry none of these
# words and are unaffected — checked against every row this module had written.
_YEAR_CONTEXT_VETO = re.compile(
    r"\badmission\w*\b|\bapply\b|\benrol\w*\b|\bprospectus\b"
    r"|\bfee\s*structure\b|\bfees?\b|\btuition\b|\bscholarship\b|\bquota\b"
    r"|\bcourses?\b|\bprogramme?s?\b|\bsyllab\w*\b|\bcurricul\w*\b"
    r"|\bexam\w*\b|\btime\s*table\b|\bresults?\b|\bsemester\b"
    r"|\bnotification\b|\bcircular\b|\btender\b"
    r"|\bfaculty\s+recruit\w*\b|\brecruitment\s+of\s+faculty\b|\bvacanc\w*\b"
    r"|\bassistant\s+professor\b|\bwalk[-\s]?in\b", re.I)


def _hints_for(display: str) -> tuple[str, ...]:
    """Marks that must sit on the same line as the number for it to be THE figure."""
    d = str(display or "")
    if "%" in d:
        return ("%",)
    for rx, hints in _UNIT_HINTS:
        if rx.search(d):
            return hints
    return ("rs", "inr", "₹", "$")


def _figure_lines(lines: list[str], fig_token: str,
                  hints: tuple[str, ...]) -> list[int]:
    """Lines that carry this figure — the token AND its unit, not the token alone."""
    out = []
    for i, ln in enumerate(lines):
        if fig_token not in ln:
            continue
        low = ln.lower()
        if any(h in low for h in hints):
            out.append(i)
    return out


def _batch_dates_figure(lines: list[str], fig_token: str, raw_batch: str,
                        display: str) -> tuple[bool, str]:
    """(ok, why_not). Does the page's batch actually sit NEXT TO this figure?

    Searched as the page's own raw token ("2020-2023", "2026-27", "2022") rather
    than a normalised year, so this is a verbatim proximity test: the exact
    string the extractor read has to appear within BATCH_NEAR_LINES of a line
    that carries the figure WITH its unit.

    Applied ONLY to the page-level batch, never to a NAAC table year. In that
    table the year is the column header of the value and the two are paired
    positionally by `_scan_naac_table`, so the year is bound to the figure by
    structure and has nothing to prove by proximity.
    """
    if not raw_batch:
        return False, "no batch on the page"
    at = _figure_lines(lines, fig_token, _hints_for(display))
    if not at:
        return False, "figure and its unit are not on one line"
    saw_close = False
    for i, ln in enumerate(lines):
        if raw_batch not in ln:
            continue
        close = [j for j in at if abs(i - j) <= BATCH_NEAR_LINES]
        if not close:
            continue
        saw_close = True
        # The year's OWN line must not be about admissions, fees, courses or
        # vacancies. Checked on that line alone, not the window: the placement
        # figures themselves sit in the window, and widening this would let a
        # single "fees" link in a nav bar veto a perfectly good batch heading.
        if _YEAR_CONTEXT_VETO.search(ln):
            continue
        lo = max(0, min(i, min(close)) - 2)
        hi = max(i, max(close)) + 3
        window = " ".join(lines[lo:hi])
        if _AGGREGATE_VETO.search(window):
            return False, "figure covers several batches, not one year"
        return True, ""
    if saw_close:
        return False, (f"the only nearby {raw_batch!r} is on an admissions/fee/"
                       f"course line, which does not date a placement figure")
    return False, f"batch {raw_batch!r} is more than {BATCH_NEAR_LINES} lines away"


def _missing_tokens(tokens: list[str], page_text: str) -> list[str]:
    """Which of these numeric tokens are NOT on the page, in any Indian form.

    Closes the gate's documented blind spot for short and decimal tokens. Uses
    verify_facts' own normalisation — "1,20,000" / "120,000" / "1.2 lakh" are
    the same number — so there is no second opinion about what a number looks
    like. A short token is a weak assertion (a bare "80" is easy to find on any
    page) and this is NOT claimed to be strong evidence on its own; the strong
    evidence is that placement.py only ever reads a number through a label. This
    catches the case that check cannot: a value that reached the column from a
    stale payload rather than from the page in front of us.
    """
    idx = vf._page_index(page_text or "")
    return [t for t in tokens if not (vf._variants(t) & idx)]


# ---------------------------------------------------------------------------
# One page's facts -> typed values
# ---------------------------------------------------------------------------

@dataclass
class Reading:
    """Everything one page yielded, already converted and already dated."""
    url: str = ""
    confidence: float = 0.0
    verdict: str = ""
    batch: str | None = None                       # dates the package figures
    values: dict = dc_field(default_factory=dict)  # column -> number | list
    display: dict = dc_field(default_factory=dict) # column -> page's own wording
    years: dict = dc_field(default_factory=dict)   # column -> the year for THAT figure
    notes: list = dc_field(default_factory=list)

    @property
    def numeric_columns(self) -> list[str]:
        return [k for k in self.values if k != "top_recruiters"]

    def score(self) -> tuple:
        """Rank competing pages for one college. More dated figures wins."""
        dedicated = "placement" in self.url.lower()
        return (len(self.numeric_columns), int(bool(self.batch)),
                int(dedicated), self.confidence)


def _latest_naac_pct(facts: dict) -> tuple[float, str, str, str] | None:
    """(value, token, display, year) from the newest column of the NAAC table.

    The five-year table is the best-dated placement evidence an Indian college
    publishes: it is a regulatory return, each figure sits under its own year,
    and placement.py already refuses to read it unless the criterion's standard
    wording is present. The NEWEST column is taken because a student asks about
    now; the older ones stay in the college_web_facts payload.
    """
    best: tuple[float, str, str, str] | None = None
    prefix = "placement_percentage_"
    for key, disp in facts.items():
        if not key.startswith(prefix):
            continue
        year = normalise_year(key[len(prefix):].replace("_", "-"))
        if not year:
            continue
        got = pct_to_number(disp)
        if got is None:
            continue
        val, tok = got
        if best is None or year > best[3]:
            best = (val, tok, str(disp), year)
    return best


def read_page(facts: dict, page_text: str, url: str,
              confidence: float, verdict: str) -> Reading:
    """Convert one verified extraction into dated, stored-shape values."""
    r = Reading(url=url, confidence=confidence, verdict=verdict)
    raw_batch = str(facts.get("batch") or "").strip()
    batch = normalise_year(raw_batch)
    r.batch = batch
    lines = [ln.strip() for ln in (page_text or "").split("\n")][:4000]

    def dated_by_batch(token: str, display: str) -> tuple[bool, str]:
        """The batch may date THIS figure only if it is written beside it."""
        if not batch:
            return False, "the page states no batch year"
        return _batch_dates_figure(lines, token, raw_batch, display)

    # ---- placement percentage -------------------------------------------
    # A NAAC year beats a batch year: it is printed in the same table row as
    # the figure, whereas a batch token is inferred from proximity.
    dated = _latest_naac_pct(facts)
    if dated is not None:
        val, tok, disp, year = dated
        if not _missing_tokens([tok], page_text):
            r.values["placement_pct"] = val
            r.display["placement_pct"] = disp
            r.years["placement_pct"] = year
        else:
            r.notes.append(f"placement_pct {tok} not found on page")
    elif facts.get("placement_percentage"):
        disp = str(facts["placement_percentage"])
        got = pct_to_number(disp)
        tok = got[1] if got else None
        ok, why = dated_by_batch(tok, disp) if got else (False, "unparseable")
        if got is None:
            pass
        elif _missing_tokens([tok], page_text):
            r.notes.append(f"placement_pct {tok} not found on page")
        elif not ok:
            r.notes.append(f"placement % {tok} present but undated - "
                           f"not stored ({why})")
        else:
            r.values["placement_pct"] = got[0]
            r.display["placement_pct"] = disp
            r.years["placement_pct"] = batch

    # ---- packages --------------------------------------------------------
    # Only the UNSCOPED keys. placement.py deliberately files a scope-limited
    # figure under its own key (highest_domestic_package,
    # highest_international_package) because lpu.in states a Rs 62.72 Lakh
    # domestic high on the same page as a 3 Cr international CTC. Neither is
    # "the highest package", so neither becomes package_highest_inr.
    for column, key in PACKAGE_COLUMNS:
        disp = facts.get(key)
        if not disp:
            continue
        got = money_to_rupees(str(disp))
        if got is None:
            r.notes.append(f"{key} {disp!r} not a plausible package")
            continue
        rupees, tok = got
        if _missing_tokens([tok], page_text):
            r.notes.append(f"{key} {tok} not found on page")
            continue
        ok, why = dated_by_batch(tok, str(disp))
        if not ok:
            r.notes.append(f"{key} present but undated - not stored ({why})")
            continue
        r.values[column] = rupees
        r.display[column] = str(disp)
        r.years[column] = batch

    # A crossed label is a parse error, not an unusual college: an average above
    # the highest means the two were read off each other's captions. Drop the
    # whole package set rather than pick a winner.
    hi = r.values.get("package_highest_inr")
    for other in ("package_average_inr", "package_median_inr"):
        if hi is not None and r.values.get(other) is not None and r.values[other] > hi:
            r.notes.append(f"{other} exceeds highest - packages dropped as crossed labels")
            for column, _ in PACKAGE_COLUMNS:
                r.values.pop(column, None)
                r.display.pop(column, None)
                r.years.pop(column, None)
            break

    # ---- recruiters ------------------------------------------------------
    # A closed vocabulary in placement.py, filtered by a line-level context
    # check, so every name here was printed on the page next to a recruiting
    # word. Not a numeric claim, so it is storable without a year — but it gets
    # one when the page states one.
    names = facts.get("recruiters_mentioned")
    if names:
        parts = [n.strip() for n in str(names).split(",") if n.strip()][:12]
        if parts:
            r.values["top_recruiters"] = parts
            r.display["top_recruiters"] = ", ".join(parts)
            # A recruiter list has no single token to sit beside, so it cannot
            # prove proximity the way a figure can. It inherits the batch only
            # when that batch already proved itself against a figure on this
            # page; otherwise the list is stored undated, which it is.
            if batch and r.years:
                r.years["top_recruiters"] = batch
    return r


def choose_row_year(r: Reading) -> str | None:
    """The single year the denormalised colleges row will carry.

    The year that dates the most figures wins; ties go to the package year,
    because that is the figure a student reads first and the one a stale date
    misleads hardest.
    """
    counts: dict[str, int] = {}
    for column, year in r.years.items():
        if column == "top_recruiters":
            continue
        counts[year] = counts.get(year, 0) + 1
    if not counts:
        return None
    best = max(counts.values())
    tied = [y for y, n in counts.items() if n == best]
    if len(tied) > 1 and r.batch in tied:
        return r.batch
    return sorted(tied)[-1]


# ---------------------------------------------------------------------------
# Crawl
# ---------------------------------------------------------------------------

def pick_placement_pages(base_url: str, links: list[tuple[str, str]],
                         limit: int = MAX_PLACEMENT_PAGES) -> list[str]:
    """Same-origin placement URLs from the homepage's links.

    Same-origin is not a nicety: a placement PDF or page on someone else's
    domain is usually an aggregator or a government circular, and attributing it
    to this college is the wrong-entity error the whole pipeline guards against.
    """
    origin = urlparse(base_url).netloc
    out: list[str] = []
    seen: set[str] = set()
    for href, anchor in links:
        if not href or href.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue
        absolute = urljoin(base_url, href.split("#")[0])
        p = urlparse(absolute)
        if p.scheme not in ("http", "https") or p.netloc != origin:
            continue
        if absolute in seen:
            continue
        hay = f"{p.path.lower()} {(anchor or '').lower()}"
        if any(k in hay for k in PLACEMENT_KEYS):
            seen.add(absolute)
            out.append(absolute)
        if len(out) >= limit:
            break
    return out


def _run_extractor(name: str, city: str | None, url: str,
                   text: str) -> tuple[dict, object] | None:
    """placement.extract -> the verify_facts gate. Returns (fact, result)."""
    got = pex.extract(text[:PAGE_TEXT_BUDGET], url, name)
    if not got:
        return None
    checked = verify_fact(got, text, name, city)
    return got, checked


async def harvest_one(client, pol, college: dict, fc=None,
                      use_ocr: bool = True) -> dict:
    """One college: homepage + placement pages + placement PDFs -> best Reading."""
    cid, name = college["college_id"], college["name"]
    site = normalise_site(college["website_url"])
    out = {"college_id": cid, "name": name, "city": college.get("city"),
           "pages": 0, "status": "empty", "reading": None, "rejected": 0,
           "uncertain": 0, "notes": []}
    if not site:
        out["status"] = "no_site"
        return out

    home = await fetch(client, pol, site, fc)
    if home is None:
        out["status"] = "blocked"
        return out
    out["pages"] += 1
    home_text, links = parse_html(home)

    # A 200 with an empty body is a JS shell, and `fetch` cannot see it: it only
    # falls back to Firecrawl when the REQUEST failed, and this request
    # succeeded. Measured on the first 10 tier-1 colleges, two of them
    # (iul.ac.in, dpsru.edu.in) parsed to 0 characters and 0 links this way, so
    # the whole college was lost without a single error being logged.
    #
    # This is the sanctioned use of a credit — one call unlocks a whole college
    # — and it stays on the HOMEPAGE only, never subpages, for the reason
    # web_enrich measured: passing the budget down spent 11 credits on 25
    # colleges and cut throughput by 18x.
    if len(home_text) < 500 and not links and fc is not None:
        rescued = await fc.scrape(client, site)
        if rescued:
            out["pages"] += 1
            home_text, links = parse_html(rescued)
            out["notes"].append(f"firecrawl rescued an empty homepage "
                                f"({len(home_text)} chars)")
    if len(home_text) < 500 and not links:
        out["status"] = "blocked"
        return out

    candidates: list[tuple[str, str]] = [(site, home_text)]
    targets = list(dict.fromkeys(
        (college.get("known_urls") or []) + pick_placement_pages(site, links)))
    for url in targets[:MAX_PLACEMENT_PAGES]:
        if url == site:
            continue
        html = await fetch(client, pol, url, None)   # no credit on subpages
        if html is None:
            continue
        out["pages"] += 1
        text, sub_links = parse_html(html)
        candidates.append((url, text))
        # Resolve against the page they were found ON, not the homepage. A
        # relative "reports/placement-2024.pdf" on /placements/ points at
        # /placements/reports/..., and handing the raw href to find_pdf_links
        # (which joins against `site`) would resolve it to /reports/... — a
        # different document, or more often a 404 that silently costs the
        # college its placement report.
        links.extend((urljoin(url, href), anchor) for href, anchor in sub_links
                     if href and not href.startswith(
                         ("mailto:", "tel:", "javascript:", "#")))

    readings: list[Reading] = []
    for url, text in candidates:
        res = await asyncio.to_thread(_run_extractor, name, college.get("city"),
                                      url, text)
        if res is None:
            continue
        got, checked = res
        if checked.verdict == "REJECTED":
            out["rejected"] += 1
            out["notes"].append(f"REJECTED {url[:60]}: {'; '.join(checked.reasons)[:110]}")
            continue
        if checked.verdict == "UNCERTAIN":
            # Same policy as the rest of the harvest: quarantine, never shown.
            # A typed column IS shown, so an UNCERTAIN fact must not reach one.
            out["uncertain"] += 1
            continue
        r = read_page(got["facts"], text, url, checked.confidence, checked.verdict)
        # Notes travel whether or not the page yielded a value: "present but
        # undated - not stored" is the single most important thing this module
        # reports, and it happens most often on pages that DID yield something
        # else. Losing it would hide the size of the dating problem.
        out["notes"].extend(r.notes)
        if r.values:
            readings.append(r)
            out.setdefault("facts_payload", []).append((url, got, checked))

    # ---- PDFs, and OCR only for the scans --------------------------------
    # web_enrich deliberately excludes placement from its OCR set: across the
    # whole corpus a scanned placement brochure is not worth the CPU. Here it
    # is, because tier 1 is 863 colleges rather than 38,700 and the annual
    # placement report is exactly the document that carries a batch year next
    # to its figures. Still last, still only when nothing was found in HTML.
    if not readings:
        try:
            from .pdf_harvest import fetch_pdf, find_pdf_links, pdf_to_text

            # limit=8 is every topic pdf_harvest knows. It stops collecting once
            # it has `limit` KINDS, so a lower cap lets six unrelated topics
            # (fees, scholarship, cutoff, admission, courses, hostel) crowd out
            # the one kind this module came for.
            for kind, pdf_url in find_pdf_links(site, links, limit=8):
                if kind != "placement":
                    continue
                data, note = await fetch_pdf(client, pol, pdf_url)
                if data is None:
                    continue
                out["pages"] += 1
                text, note = pdf_to_text(data)
                ceiling = 1.0
                if not text and use_ocr:
                    from .ocr import confidence_ceiling, ocr_pdf

                    text, rep = await asyncio.to_thread(ocr_pdf, data)
                    ceiling = confidence_ceiling(rep)
                    if text:
                        out["notes"].append(
                            f"OCR {pdf_url[:56]}: {rep.get('corroborated', 0)} "
                            f"figures corroborated, ceiling {ceiling}")
                if not text:
                    continue
                res = await asyncio.to_thread(_run_extractor, name,
                                              college.get("city"), pdf_url, text)
                if res is None:
                    continue
                got, checked = res
                # An OCR page can never publish at full confidence however sure
                # the parser is - the ceiling reflects how well the two DPI
                # passes agreed that the digits are even legible.
                conf = min(checked.confidence, ceiling)
                if checked.verdict != "CONFIRMED" or conf < vf.CONFIRM_AT:
                    out["rejected"] += int(checked.verdict == "REJECTED")
                    continue
                r = read_page(got["facts"], text, pdf_url, conf, checked.verdict)
                if r.values:
                    got["confidence"] = conf
                    readings.append(r)
                    out.setdefault("facts_payload", []).append((pdf_url, got, checked))
                if len(readings) >= MAX_PDFS:
                    break
        except Exception as exc:  # noqa: BLE001 - a PDF must never lose a college
            out["notes"].append(f"pdf/ocr step failed: {str(exc)[:90]}")

    if readings:
        out["reading"] = max(readings, key=lambda r: r.score())
        out["status"] = "ok"
    return out


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

# Every column set together, from ONE page, for ONE year. Not COALESCE'd onto
# whatever was there before: mixing this run's packages with a previous run's
# percentage under a single placement_year would produce a row that no page ever
# stated. A refresh replaces the observation wholesale.
COLLEGE_UPDATE = """
UPDATE colleges SET
    placement_pct       = %s,
    placement_year      = %s,
    package_highest_inr = %s,
    package_average_inr = %s,
    package_median_inr  = %s,
    top_recruiters      = %s::jsonb,
    placement_source    = %s
WHERE college_id = %s"""

# A page that yielded ONLY a recruiter list must not blank the dated figures a
# previous run read off the same site. A recruiter list is not an observation of
# the placement statistics, so it updates its own column and leaves the year and
# the packages alone. Without this, one refresh against a redesigned placement
# page silently deletes every figure this module worked for.
RECRUITERS_ONLY_UPDATE = """
UPDATE colleges SET
    top_recruiters   = %s::jsonb,
    placement_source = COALESCE(placement_source, %s)
WHERE college_id = %s"""

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

SOURCE = "college_site"

# Re-examine exactly the colleges this module has already written, and nothing
# else. Needed because tightening a dating rule can only ever REMOVE a stored
# figure, and a college that no longer qualifies produces no reading — so a
# plain re-run would leave its old row untouched and the correction would never
# land. `--recheck` clears this module's writes for those colleges first, then
# re-harvests them, so a row that no longer earns its place disappears instead
# of going stale. It touches ~24 sites rather than 777, which is also the polite
# way to correct a mistake.
RECHECK_SQL = """
SELECT p.college_id, c.name, c.city, c.state, c.website_url, p.score
FROM college_priority p JOIN colleges c USING(college_id)
WHERE p.tier = 1 AND c.placement_source IS NOT NULL
ORDER BY p.score DESC
"""

CLEAR_COLLEGES = """
UPDATE colleges SET
    placement_pct = NULL, placement_year = NULL, package_highest_inr = NULL,
    package_average_inr = NULL, package_median_inr = NULL,
    top_recruiters = NULL, placement_source = NULL
WHERE college_id = ANY(%s)"""

CLEAR_FIELDS = """
DELETE FROM college_field
WHERE source = %s AND college_id = ANY(%s)
  AND field IN ('placement_pct','package_highest_inr','package_average_inr',
                'package_median_inr','top_recruiters')"""


def clear_written(college_ids: list[str]) -> None:
    """Undo this module's writes for these colleges, in one transaction."""
    if not college_ids:
        return
    with _pool().connection() as c:
        cur = c.cursor()
        cur.execute(CLEAR_COLLEGES, (college_ids,))
        cur.execute(CLEAR_FIELDS, (SOURCE, college_ids))


def build_writes(result: dict) -> tuple[tuple | None, list[tuple], list[tuple]]:
    """(colleges row, college_field rows, college_web_facts rows) for one college.

    Nothing here decides truth; every value was already converted, dated, found
    verbatim on the page and passed by the gate. This only decides SHAPE.
    """
    r: Reading | None = result.get("reading")
    if r is None or not r.values:
        return None, [], []
    cid = result["college_id"]
    now = _now()
    row_year = choose_row_year(r)

    # The colleges row carries only figures dated by ITS year. Anything else is
    # dropped from the row and kept in college_field, which dates per field.
    def for_row(column):
        if column not in r.values:
            return None
        year = r.years.get(column)
        if column == "top_recruiters":
            return r.values[column]
        return r.values[column] if year == row_year else None

    pct = for_row("placement_pct")
    hi = for_row("package_highest_inr")
    avg = for_row("package_average_inr")
    med = for_row("package_median_inr")
    rec = for_row("top_recruiters")

    if pct is None and hi is None and avg is None and med is None and not rec:
        return None, [], []

    if pct is None and hi is None and avg is None and med is None:
        # Recruiters only — a narrower statement, so a narrower write.
        college_row = ("recruiters_only",
                       json.dumps(rec, ensure_ascii=False), r.url, cid)
    else:
        college_row = ("full", pct, row_year, hi, avg, med,
                       json.dumps(rec, ensure_ascii=False) if rec else None,
                       r.url, cid)

    fields: list[tuple] = []
    for column in FIELD_COLUMNS:
        if column not in r.values:
            continue
        value = r.values[column]
        if column == "top_recruiters":
            value_text = json.dumps(value, ensure_ascii=False)
            value_num = None
        else:
            value_text = r.display.get(column)
            value_num = float(value)
        fields.append((cid, column, value_text, value_num, SOURCE, r.url,
                       r.years.get(column), now, round(float(r.confidence), 3)))

    # Also keep the raw extraction where every other harvested fact lives, so
    # the card renderer, reverify and the nightly refresh see it the same way
    # they see the rest of the sweep. The typed columns are a promotion of this,
    # never a replacement for it.
    # The payload must be the page the typed columns actually came from. Taking
    # whichever page was crawled first would file the chosen page's figures
    # under a different page's URL and summary — the provenance would point at
    # a document that does not contain them.
    facts_rows: list[tuple] = []
    for url, got, checked in (result.get("facts_payload") or []):
        if url != r.url:
            continue
        facts_rows.append(
            (cid, "placement", got["summary"],
             json.dumps(got["facts"], ensure_ascii=False), url, now,
             "placement-parser", float(got.get("confidence") or checked.confidence),
             r.years.get("placement_pct") or r.batch))
    return college_row, fields, facts_rows


def commit(batch: list[dict]) -> tuple[int, int]:
    """Land one batch. Colleges, fields and raw facts commit together."""
    full, recruiters_only, fields, facts = [], [], [], []
    for result in batch:
        crow, frows, wrows = build_writes(result)
        if crow is None:
            continue
        (full if crow[0] == "full" else recruiters_only).append(crow[1:])
        fields.extend(frows)
        facts.extend(wrows)
    n = len(full) + len(recruiters_only)
    if not n:
        return 0, 0
    with _pool().connection() as c:
        cur = c.cursor()
        if full:
            cur.executemany(COLLEGE_UPDATE, full)
        if recruiters_only:
            cur.executemany(RECRUITERS_ONLY_UPDATE, recruiters_only)
        if fields:
            cur.executemany(FIELD_UPSERT, fields)
        if facts:
            cur.executemany(FACT_UPSERT, facts)
    return n, len(fields)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

async def sweep(limit: int | None = None, concurrency: int = 6,
                delay: float = 1.5, commit_writes: bool = False,
                sample: int = 8, skip_done_days: int | None = 180,
                use_ocr: bool = True, firecrawl_reserve: int = 100,
                recheck: bool = False) -> dict:
    be = get_backend()
    rows = be.q(RECHECK_SQL if recheck else TIER1_SQL)

    known: dict[str, list[str]] = {}
    for row in be.q(KNOWN_URL_SQL):
        known.setdefault(row["college_id"], []).append(row["source_url"])

    skip: set[str] = set()
    if skip_done_days and not recheck:
        skip = {r["college_id"] for r in be.q(DONE_SQL, (int(skip_done_days),))}

    targets = [dict(r, known_urls=known.get(r["college_id"], []))
               for r in rows if r["college_id"] not in skip]
    if limit:
        targets = targets[:limit]

    print(f"[placements] {'RECHECK of already-written' if recheck else 'tier-1 with a website'}"
          f": {len(rows)}; already fresh: {len(skip)}; attempting: {len(targets)}",
          file=sys.stderr)
    if not targets:
        return {"attempted": 0}

    # Clear BEFORE re-harvesting, so a college that no longer qualifies ends up
    # empty rather than keeping the row this run has decided it cannot justify.
    if recheck and commit_writes:
        ids = [t["college_id"] for t in targets]
        await asyncio.to_thread(clear_written, ids)
        print(f"[placements] cleared previous writes for {len(ids)} colleges",
              file=sys.stderr)

    pol = Politeness(delay=delay)
    sem = asyncio.Semaphore(concurrency)
    stats = {"attempted": 0, "ok": 0, "nothing": 0, "blocked": 0, "no_site": 0,
             "rejected": 0, "uncertain": 0, "fields": 0, "colleges": 0,
             "undated_figures": 0, "undated_colleges": 0}
    done: list[dict] = []
    pending: list[dict] = []

    async with httpx.AsyncClient(headers=HEADERS, timeout=FETCH_TIMEOUT,
                                 follow_redirects=True, verify=False) as client:
        fc = FirecrawlBudget(reserve=firecrawl_reserve)
        try:
            await fc.prime(client)
        except Exception:  # noqa: BLE001 - no credits is not a failure
            fc = None

        async def one(college):
            async with sem:
                try:
                    return await harvest_one(client, pol, college, fc, use_ocr)
                except Exception as exc:  # noqa: BLE001
                    return {"college_id": college["college_id"],
                            "name": college["name"], "status": "error",
                            "reading": None, "notes": [str(exc)[:120]],
                            "rejected": 0, "uncertain": 0}

        for i in range(0, len(targets), concurrency * 4):
            chunk = targets[i:i + concurrency * 4]
            for result in await asyncio.gather(*(one(c) for c in chunk)):
                stats["attempted"] += 1
                stats["rejected"] += result.get("rejected", 0)
                stats["uncertain"] += result.get("uncertain", 0)
                # The honest number, and the one worth reporting: how much real
                # placement data this run REFUSED because the page never said
                # which year it was for. Counted per figure and per college,
                # because "18 undated" is meaningless without knowing whether
                # that is 18 colleges or one college with 18 figures.
                lost = sum(1 for n in (result.get("notes") or [])
                           if "undated" in n)
                stats["undated_figures"] += lost
                stats["undated_colleges"] += int(lost > 0)
                status = result.get("status")
                if status in ("blocked", "no_site"):
                    stats[status] += 1
                if result.get("reading") is not None:
                    stats["ok"] += 1
                    done.append(result)
                    pending.append(result)
                else:
                    stats["nothing"] += 1
            # Flush once per CHUNK, not once per N successes. The yield here is
            # about 4%, so a "flush every 25 results" rule needs ~600 colleges
            # crawled before its first commit — an hour of politeness-paced
            # fetching held in memory, and lost entirely if the process is
            # killed. Chunk-flushing bounds the loss to one chunk.
            if commit_writes and pending:
                c, f = await asyncio.to_thread(commit, pending)
                stats["colleges"] += c
                stats["fields"] += f
                pending = []
            print(f"[placements] {stats['attempted']}/{len(targets)} attempted, "
                  f"{stats['ok']} with figures, {stats['nothing']} yielded nothing",
                  file=sys.stderr)

    if commit_writes and pending:
        c, f = await asyncio.to_thread(commit, pending)
        stats["colleges"] += c
        stats["fields"] += f
    if not commit_writes:
        # Dry run still computes exactly what WOULD be written, so the numbers
        # reported are the numbers a --commit run produces.
        for result in done:
            crow, frows, _ = build_writes(result)
            if crow is not None:
                stats["colleges"] += 1
                stats["fields"] += len(frows)

    _report_sample(done, sample, commit_writes)
    return stats


def _report_sample(done: list[dict], n: int, committed: bool) -> None:
    """The hand-check table. Nothing goes out of here without one."""
    if not done or n <= 0:
        return
    print("\n" + "=" * 78, file=sys.stderr)
    print(f"HAND-CHECK SAMPLE ({'committed' if committed else 'DRY RUN'}) — "
          f"every figure below must be readable on its source URL", file=sys.stderr)
    print("=" * 78, file=sys.stderr)
    for result in done[:n]:
        r: Reading = result["reading"]
        print(f"\n{result['name'][:60]}  [{result['college_id']}]", file=sys.stderr)
        print(f"  source : {r.url[:100]}", file=sys.stderr)
        print(f"  year   : {choose_row_year(r)}   (batch token: {r.batch})",
              file=sys.stderr)
        for column in FIELD_COLUMNS:
            if column not in r.values:
                continue
            val = r.values[column]
            shown = f"{val:,}" if isinstance(val, int) else val
            print(f"  {column:22} {str(shown)[:52]:52} <- page said "
                  f"{str(r.display.get(column))[:34]!r} ({r.years.get(column)})",
                  file=sys.stderr)
        for note in r.notes[:3]:
            print(f"  note   : {note[:100]}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Self-test. No network, no database.
# ---------------------------------------------------------------------------

# Every case below is a real page shape this module met on a tier-1 college, and
# the four DROP cases are defects it actually committed before the check named
# beside them existed. They are here so that raising coverage later cannot
# quietly reintroduce one.
_CONVERSIONS = [
    # (display, expected rupees) -- the 100,000x error lives here
    ("45 LPA", 4_500_000), ("Rs 45 LPA", 4_500_000), ("6.5 LPA", 650_000),
    ("INR 88.20 Lakh", 8_820_000), ("1.5 Crore", 15_000_000),
    ("Rs1.1 Cr", 11_000_000), ("Rs 12,00,000", 1_200_000),
    ("Rs 62.72 Lakhs", 6_272_000), ("8.5 lacs", 850_000),
    ("Rs 300 p.m.", None),          # a stipend, not a package
    ("Rs 5,000", None),             # below any real package
    ("900 LPA", None),              # a decimal slip
    ("US$ 7000.00", None),          # not rupees
]

_YEARS = [
    ("2024-25", "2024-25"), ("2024-2025", "2024-25"),
    ("2020-2023", "2023"),          # a 3-year cohort: placed in the final year
    ("2022", "2022"), ("2015-30", None), ("2000-2010", None),
    ("1998", None), ("abc", None), (None, None),
]

_PAGES = [
    ("keeps a figure whose batch is beside it",
     "Placements Overview Batch 2025\n700+\nCompanies Visited\n62 LPA\n"
     "Highest Package\nplacement drive placed recruiters\n"
     "Our placement cell coordinates campus recruitment for placed students.",
     {"package_highest_inr": (6_200_000, "2025")}),
    ("drops a figure whose only year is 183 lines away",
     "Admissions open for 2026-27 Faculty Recruitment Updates\n"
     + "\n".join(f"filler {i}" for i in range(190))
     + "\n90%+\nPlacement rate\n46.54 LPA\nHighest Package\nplacement placed",
     {}),
    ("drops a five-year aggregate rather than dating it to the range end",
     "Placement Highlights - Last Five Years (Batches passed between 2022-2026)\n"
     "44.14 LPA\nHighest Package\nplacement placed recruiters drive",
     {}),
    ("does not let an unrelated date anchor the year",
     "Guidelines for New Admissions (2026-27)\n"
     "AICTE ATAL FDP from 7-12 September 2026\n"
     + "\n".join(f"notice {i}" for i in range(40))
     + "\n12 LPA\nAverage Package\nplacement placed recruiters",
     {}),
    ("refuses a year that belongs to an admissions banner",
     "From Classroom to Career\n95%\nPlacement Rate\nGlobal Industry Exposure\n"
     "Admissions 2026-27 Open\nplacement placed recruiters drive\n"
     "Our placement cell coordinates campus recruitment for placed students.",
     {}),
    ("refuses a year that belongs to a fee notification",
     "7LPA\nAvg. Salary Package\n30LPA\nHighest Salary Package\n"
     "NOTIFICATION - Fee Structure for I Year B.E for the Academic year 2026-27\n"
     "placement placed recruiters drive\n"
     "Our placement cell coordinates campus recruitment for placed students.",
     {}),
    ("keeps a year stated as a placement year",
     "Placements\n1256+\nPlacements in 2026\n38 Lakhs\nhighest package\n"
     "placement placed recruiters drive\n"
     "Our placement cell coordinates campus recruitment for placed students.",
     {"package_highest_inr": (3_800_000, "2026")}),
]


def self_test() -> int:
    """Deterministic checks for the arithmetic and the dating rules."""
    bad = 0
    for disp, want in _CONVERSIONS:
        got = money_to_rupees(disp)
        val = got[0] if got else None
        if val != want:
            bad += 1
            print(f"  FAIL money {disp!r}: got {val}, want {want}")
    for raw, want in _YEARS:
        got = normalise_year(raw)
        if got != want:
            bad += 1
            print(f"  FAIL year {raw!r}: got {got!r}, want {want!r}")
    for label, text, want in _PAGES:
        got = pex.extract(text, "https://t.edu/placements", "Test College")
        facts = (got or {}).get("facts", {})
        r = read_page(facts, text, "https://t.edu/placements", 0.8, "CONFIRMED")
        actual = {k: (v, r.years.get(k)) for k, v in r.values.items()
                  if k != "top_recruiters"}
        if actual != want:
            bad += 1
            print(f"  FAIL page [{label}]\n       got  {actual}\n"
                  f"       want {want}\n       notes {r.notes}")
    print(f"self-test: {'FAILED ' + str(bad) if bad else 'all cases pass'} "
          f"({len(_CONVERSIONS)} conversions, {len(_YEARS)} years, "
          f"{len(_PAGES)} pages)")
    return bad


def report() -> None:
    """What is actually in the columns now."""
    be = get_backend()
    print("-- tier-1 placement coverage --")
    row = be.q("""
        SELECT COUNT(*) tier1,
               COUNT(c.placement_pct) pct,
               COUNT(c.package_highest_inr) hi,
               COUNT(c.package_average_inr) avg,
               COUNT(c.package_median_inr) med,
               COUNT(c.top_recruiters) rec,
               COUNT(c.placement_year) yr
        FROM college_priority p JOIN colleges c USING(college_id)
        WHERE p.tier = 1""")[0]
    for k, v in row.items():
        print(f"  {k:8} {v:>6,}")
    print("-- college_field rows written by this module --")
    for r in be.q("""SELECT field, COUNT(*) n, COUNT(stated_year) dated,
                            ROUND(AVG(confidence)::numeric, 2) conf
                     FROM college_field
                     WHERE field IN ('placement_pct','package_highest_inr',
                        'package_average_inr','package_median_inr','top_recruiters')
                     GROUP BY field ORDER BY n DESC"""):
        print(f"  {r['field']:22} {r['n']:>5,}  dated {r['dated']:>5,}  "
              f"avg confidence {r['conf']}")
    print("-- placement_year spread --")
    for r in be.q("""SELECT placement_year y, COUNT(*) n FROM colleges
                     WHERE placement_year IS NOT NULL
                     GROUP BY y ORDER BY y DESC LIMIT 15"""):
        print(f"  {r['y']:10} {r['n']:>5,}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=6,
                    help="colleges in flight; pacing within a host is separate")
    ap.add_argument("--delay", type=float, default=1.5,
                    help="seconds between requests to the SAME domain")
    ap.add_argument("--commit", action="store_true",
                    help="actually write. Without this it is a DRY RUN that "
                         "reports exactly what it would have written.")
    ap.add_argument("--sample", type=int, default=8,
                    help="colleges to print in full for hand-checking")
    ap.add_argument("--refresh-days", type=int, default=180,
                    help="re-harvest a college whose placement fields are older "
                         "than this; 0 to re-harvest everything")
    ap.add_argument("--no-ocr", action="store_true",
                    help="skip OCR of scanned placement reports")
    ap.add_argument("--recheck", action="store_true",
                    help="re-harvest ONLY the colleges this module has already "
                         "written, clearing them first. Use after tightening a "
                         "rule, so rows that no longer qualify are removed "
                         "rather than left stale.")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--self-test", action="store_true",
                    help="offline checks for the LPA arithmetic and the dating "
                         "rules; no network, no database")
    a = ap.parse_args()

    if a.self_test:
        sys.exit(1 if self_test() else 0)
    if a.report:
        report()
        return
    stats = asyncio.run(sweep(limit=a.limit, concurrency=a.concurrency,
                              delay=a.delay, commit_writes=a.commit,
                              sample=a.sample,
                              skip_done_days=a.refresh_days or None,
                              use_ocr=not a.no_ocr, recheck=a.recheck))
    print("\n" + "=" * 78, file=sys.stderr)
    print("DRY RUN - nothing written" if not a.commit else "COMMITTED",
          file=sys.stderr)
    for k in ("attempted", "ok", "nothing", "blocked", "no_site", "rejected",
              "uncertain", "undated_figures", "undated_colleges", "colleges",
              "fields"):
        print(f"  {k:10} {stats.get(k, 0):>6,}", file=sys.stderr)
    if stats.get("attempted"):
        print(f"  yield      {stats.get('ok', 0) / stats['attempted']:.1%} of "
              f"colleges attempted produced a stored figure", file=sys.stderr)


if __name__ == "__main__":
    main()
