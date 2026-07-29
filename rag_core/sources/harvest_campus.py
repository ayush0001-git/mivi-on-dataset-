"""Harvest FACILITIES, MEDIA and CONTACT from each tier-1 college's own site.

WHY THIS IS A SEPARATE HARVESTER, AND WHY IT USES NO LLM
`web_enrich.py` reads *prose* — fees, scholarships, cutoffs — where the value is
a sentence a model has to understand. Everything here is *structural*: a mailto:
on a contact page, a same-origin PDF called "Prospectus", a facebook.com link in
a footer, the word "Library" under a heading called Infrastructure. None of that
needs a language model, and the LLM is the measured bottleneck on this account
(15 req/min on the tighter provider — see the provider-budget note). So this
module is entirely deterministic: it runs at fetch speed, spends zero quota, and
its output is reproducible from the same page, which matters because every value
it writes has to be defensible field by field.

WHAT IT WILL NOT DO
- It will not decide a college has a girls' hostel because the page said
  "hostel". The vocabulary distinguishes boys_hostel from girls_hostel and a
  bare, ungendered "hostel" claims NEITHER. A female student choosing a college
  on a girls' hostel we invented is the kind of harm this corpus exists to
  avoid, so the ambiguous case is dropped and counted, not resolved.
- It will not take a phone number or an email just because it sits on the page.
  On a template-built college site the footer number frequently belongs to the
  web designer. Every contact value has to sit within a window that corroborates
  the COLLEGE'S OWN identity, and windows carrying "designed by" / "powered by"
  are refused outright.
- It will not follow a link off the college's domain. A social profile, a
  brochure or a gallery image hosted elsewhere is unattributable, and mis-
  attribution is the worst failure mode in this pipeline.
- It will not invent a facility from a negation. "Hostel facility is not
  available" and "swimming pool (proposed)" both contain the vocabulary word.

HOW IT IS GATED
`verify_facts.verify()` is the gate, the same one the fee harvest uses — not a
parallel one. It contributes two checks here:

  * the wrong-entity check, which is the load-bearing one for this module. The
    catalogue's `website_url` can point at the wrong institution, and if it does
    then every link, image and phone number on it would be filed under a college
    that never published them. So the homepage text has to corroborate the
    college's identity BEFORE any of its links are believed, and the whole
    college is abandoned when it does not.
  * verbatim anchoring, for the fields whose value is text read off the page:
    campus_size, phone, email, pincode, facilities.

Anchoring is deliberately NOT applied to the URL-valued fields (brochure, video,
gallery, social). A URL lives in an `href`, and `parse_html` returns visible
text — so anchoring "prospectus-2024-1187.pdf" would reject every correct
brochure for being absent from prose it was never in. Those fields are gated by
same-origin plus the homepage identity check instead, which is a stronger claim
than anchoring anyway: the file is on the college's own server.

THE PINCODE IS MOSTLY NOT A CRAWL
`colleges.address` is 100% populated from the catalogue and carries a 6-digit
PIN for 843 of the 863 tier-1 colleges. Reading it out of a column we already
own beats fetching a contact page for it, per the standing finding that the
client's own data held more than the open web did. It goes through the SAME
gate, with the address as the source document rather than a page, and is
additionally checked against the PIN region map: a Telangana college whose
"pincode" starts with 7 has had a phone fragment or a house number parsed as a
PIN, and the region digit catches that for free.

Run:
    python -m rag_core.sources.harvest_campus --dry-run --limit 25   # inspect
    python -m rag_core.sources.harvest_campus --limit 25 --sample 8  # hand-check
    python -m rag_core.sources.harvest_campus --tier 1               # the job
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse

import httpx

# config first, then the backend — rag_core.config is the only module that calls
# load_dotenv() and store.backend reads MIVI_PG_DSN at import time. The wrong
# order silently targets localhost instead of the configured server.
from .. import config  # noqa: F401
from ..store.backend import get_backend
from .verify_facts import _AY_RE, anchor_numbers, page_is_about, stated_year
from .verify_facts import verify as verify_fact
from .web_enrich import (HEADERS, FirecrawlBudget, Politeness, fetch,
                         normalise_site, parse_html)
from .web_enrich import _pool  # the harvest's own "one transaction" connection

# Homepage + at most this many subpages. A hard ceiling per host, because
# politeness is not only about the 1.5s pace — it is also about not walking a
# small college's shared hosting looking for a gallery.
MAX_SUBPAGES = 5
MAX_GALLERY_IMAGES = 12
IDENTITY_WINDOW = 300        # chars either side of a contact value
FACILITY_WINDOW = 130        # chars either side of a vocabulary word


# ---------------------------------------------------------------------------
# Which subpages are worth a request
# ---------------------------------------------------------------------------
# Same (kind, keywords) shape as web_enrich.PAGE_TARGETS and matched the same
# way — against BOTH the href and the anchor text, because Indian college sites
# routinely link "Contact Us" to /page.php?id=7 where the URL says nothing.
# A separate list rather than a reuse of that one: the fee sweep wants
# fees/cutoff/admission pages and would spend its four-page budget on them.
CAMPUS_TARGETS = (
    ("contact", ("contact", "contact-us", "contactus", "reach-us", "reach us",
                 "get-in-touch", "enquiry", "how-to-reach")),
    ("facilities", ("facilit", "infrastructure", "amenit", "campus-life",
                    "campus life", "our-campus", "hostel", "library")),
    ("about", ("about", "about-us", "aboutus", "profile", "overview",
               "the-institute", "our-college", "history")),
    # "media" is deliberately NOT a keyword here. It matches press-and-media
    # pages, and the first dry run duly filled a gallery with news-item
    # thumbnails and an accessibility icon.
    ("gallery", ("gallery", "photo", "photos", "campus-tour", "campus tour",
                 "photo-gallery", "image-gallery")),
    ("admission", ("prospectus", "brochure", "admission", "admissions")),
)


def pick_campus_pages(base_url: str, links: list[tuple[str, str]]
                      ) -> list[tuple[str, str]]:
    """[(kind, absolute_url)] — at most one page per kind, same-origin only."""
    origin = urlparse(base_url).netloc
    chosen: dict[str, str] = {}
    for href, anchor in links:
        if not href or href.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue
        absolute = urljoin(base_url, href.split("#")[0])
        p = urlparse(absolute)
        if p.scheme not in ("http", "https") or p.netloc != origin:
            continue
        if p.path.lower().endswith(".pdf"):
            continue
        path_hay = p.path.lower()
        both_hay = f"{path_hay} {(anchor or '').lower()}"
        for kind, keys in CAMPUS_TARGETS:
            if kind in chosen:
                continue
            # A gallery has to live at a gallery URL. Matching on anchor text as
            # well put IIT Bombay's /research-highlight/... news item into its
            # gallery, and the "photographs" harvested from it were the images
            # inside a research article. Every other kind still matches on the
            # anchor, because "Contact Us" really is often linked to /page.php.
            hay = path_hay if kind == "gallery" else both_hay
            if any(k in hay for k in keys):
                chosen[kind] = absolute
                break
        if len(chosen) >= MAX_SUBPAGES:
            break
    return list(chosen.items())


# ---------------------------------------------------------------------------
# FACILITIES — a bounded vocabulary, and the reasons a match does not count
# ---------------------------------------------------------------------------
# The keys are the client's vocabulary and nothing else may be written. Each
# carries the ways a college actually spells it; plurals are explicit because a
# \b-anchored singular misses "Laboratories" and "Canteens", which is how a
# facilities page that lists everything yields nothing.
FACILITY_VOCAB: tuple[tuple[str, str], ...] = (
    ("boys_hostel", r"\b(?:boys?|gents?|men'?s|male)[\s'’-]*hostel\w*\b"
                    r"|\bhostel\w*\s+for\s+(?:boys|gents|men)\b"),
    ("girls_hostel", r"\b(?:girls?|ladies|women'?s|female)[\s'’-]*hostel\w*\b"
                     r"|\bhostel\w*\s+for\s+(?:girls|ladies|women)\b"),
    ("library", r"\blibrar(?:y|ies)\b|\bbook\s*bank\b|\bdigital\s+librar\w+\b"),
    ("wifi", r"\bwi[\s-]?fi\b|\bwifi\b|\bwireless\s+(?:internet|campus|network)\b"),
    ("labs", r"\blaborator(?:y|ies)\b|\blabs?\b|\bcomputer\s+cent(?:re|er)\b"),
    ("auditorium", r"\bauditorium\w*\b|\bseminar\s+hall\w*\b|\bconference\s+hall\w*\b"),
    ("gym", r"\bgym\b|\bgymnasium\w*\b|\bfitness\s+(?:cent(?:re|er)|room)\b"),
    ("sports_ground", r"\bplay\s*ground\w*\b|\bsports\s+(?:ground|complex|field|"
                      r"facilit\w+)\b|\bathletic\s+track\b|\bcricket\s+ground\b"
                      r"|\bfootball\s+(?:ground|field)\b|\bbasketball\s+court\b"),
    ("swimming_pool", r"\bswimming\s+pool\w*\b|\bswimming\s+facilit\w+\b"),
    ("cafeteria", r"\bcafeteria\w*\b|\bcanteen\w*\b|\bfood\s+court\w*\b|\bcafe\b"
                  r"|\bmess\s+(?:facilit\w+|hall)\b"),
    ("medical", r"\bmedical\s+(?:facilit\w+|cent(?:re|er)|room|aid)\b"
                r"|\bhealth\s+(?:cent(?:re|er)|care\s+cent(?:re|er))\b"
                r"|\bdispensar(?:y|ies)\b|\binfirmar(?:y|ies)\b|\bsick\s+bay\b"),
    ("transport", r"\btransport(?:ation)?\s+facilit\w+\b|\bbus\s+(?:facilit\w+|"
                  r"service|fleet)\b|\bcollege\s+buses\b|\bshuttle\s+service\b"),
    ("atm", r"\batm\b|\bautomated\s+teller\b"),
)
FACILITY_VOCAB = tuple((k, re.compile(p, re.I)) for k, p in FACILITY_VOCAB)

# A vocabulary word inside one of these is not a claim that the facility exists.
# "Proposed", "under construction" and "coming soon" are the honest ones; the
# rest are outright denials. Checked over the whole window, both sides, because
# Indian college prose puts the negation on either end ("no swimming pool",
# "swimming pool is not available").
FACILITY_NEGATION = re.compile(
    r"\b(?:no|not|none|without|non)\b[^.]{0,40}$"          # negation just before
    r"|\bnot\s+(?:available|provided|present|yet)\b"
    r"|\bis\s+no\b|\bare\s+no\b|\bdoes\s+not\b|\bdo\s+not\b|\bwill\s+be\b"
    r"|\bpropos\w+\b|\bunder\s+construction\b|\bcoming\s+soon\b|\bplanned\b"
    r"|\bshortly\b|\bin\s+future\b|\bnearest\b|\bnearby\s+hospital\b"
    r"|\boutside\s+the\s+campus\b", re.I)

# On a page that is NOT about facilities, a bare vocabulary word is usually
# navigation ("Library" in a menu appears on every page of the site). There it
# only counts when the sentence actually asserts the facility.
FACILITY_AFFIRMATIVE = re.compile(
    r"\b(?:has|have|having|provides?|provided|offers?|offering|equipped|"
    r"available|avails?|facilit\w+|maintains?|houses?|boasts?|"
    r"well[\s-]?(?:stocked|equipped|furnished)|spacious|separate|own)\b", re.I)

# A "hostel" with no gender attached. Counted and reported, never stored: the
# vocabulary has no ungendered hostel key and picking one would be a guess.
_BARE_HOSTEL = re.compile(r"\bhostel\w*\b", re.I)


# Where one claim stops and the next begins. Scoping to the SENTENCE rather
# than to a character window is the difference between reading a facilities page
# and misreading it: at ±130 characters, "Swimming pool is not available." sits
# inside the window of the very next line, "Library is open from 9 am", and
# cancels a library that is genuinely claimed. Measured on a real page shape,
# a fixed window dropped every facility on a page that denied one.
_SEGMENT_BREAK = re.compile(r"[.\n;•|!?]")


def _evidence(text: str, lo: int, hi: int, limit: int = 240) -> str:
    """A quote from [lo:hi] that never begins or ends inside a number.

    Slicing text at a fixed offset invents numbers: cutting "411057" produced
    "4110", which `anchor_numbers` then correctly reported as absent from the
    source, rejecting a pincode that was perfectly good. Every evidence string
    in this module is number-anchored downstream, so all of them come through
    here. The boundaries only ever move INWARD, so a quote can lose a digit but
    can never grow to include text the caller did not ask for.
    """
    lo, hi = max(0, lo), min(len(text), hi)
    while lo < hi and lo > 0 and text[lo].isdigit() and text[lo - 1].isdigit():
        lo += 1
    while hi > lo and hi < len(text) and text[hi - 1].isdigit() and text[hi].isdigit():
        hi -= 1
    out = re.sub(r"\s+", " ", text[lo:hi]).strip()
    if len(out) > limit:
        # The same hazard again, one level down: hard-cutting the collapsed
        # string can split a number, so drop any partial trailing figure.
        out = re.sub(r"\d[\d,.]*$", "", out[:limit]).rstrip()
    return out


def _segment(text: str, start: int, end: int, span: int = 200
             ) -> tuple[str, str]:
    """(whole segment, the part before the match) for the claim at [start:end]."""
    lo = max(0, start - span)
    left = text[lo:start]
    breaks = list(_SEGMENT_BREAK.finditer(left))
    seg_lo = lo + (breaks[-1].end() if breaks else 0)
    right = text[end:end + span]
    m = _SEGMENT_BREAK.search(right)
    seg_hi = end + (m.start() if m else len(right))
    return text[seg_lo:seg_hi], text[seg_lo:start]


def facilities_from(text: str, kind: str, name: str | None = None,
                    city: str | None = None) -> dict[str, str]:
    """{facility_key: the verbatim sentence that claims it}.

    `kind` is the page's role, and it decides how much a mention has to prove.
    On a facilities/infrastructure page a bare mention IS the claim — those
    pages are overwhelmingly bullet lists, and demanding a verb there would
    discard the single richest source. Anywhere else the sentence has to assert
    something, because a vocabulary word in a nav menu appears on every page of
    the site and claims nothing.

    Off the facilities page, a claim must ALSO be locally attributable to this
    college when `name` is given. Same wrong-entity risk that put another
    university's acreage into `campus_size`, one step milder: a homepage
    carries news about partner institutions, and "UCR has a swimming pool and
    a gym" is a perfectly affirmative sentence about somebody else.
    """
    found: dict[str, str] = {}
    if not text:
        return found
    on_facilities_page = kind in ("facilities", "gallery")
    for key, rx in FACILITY_VOCAB:
        for m in rx.finditer(text):
            segment, before = _segment(text, m.start(), m.end())
            # Negation is checked against the text BEFORE the word for the
            # "no <facility>" shape, and against the sentence for the rest.
            if FACILITY_NEGATION.search(before) or FACILITY_NEGATION.search(segment):
                continue
            if not on_facilities_page:
                if not FACILITY_AFFIRMATIVE.search(segment):
                    continue
                if name is not None and not _attributable(text, m.start(),
                                                          name, city):
                    continue
            found[key] = _evidence(segment, 0, len(segment), 220)
            break
    return found


def bare_hostel_mentions(text: str) -> int:
    """Ungendered 'hostel' occurrences that deliberately claimed nothing."""
    if not text:
        return 0
    gendered = 0
    for key, rx in FACILITY_VOCAB:
        if key.endswith("_hostel"):
            gendered += len(rx.findall(text))
    return max(0, len(_BARE_HOSTEL.findall(text)) - gendered)


# ---------------------------------------------------------------------------
# CAMPUS SIZE
# ---------------------------------------------------------------------------
# Acres and hectares only. Square feet is deliberately excluded: an Indian
# college quoting "2,00,000 sq ft" is almost always stating BUILT-UP AREA, which
# is a different quantity from campus size, and mixing the two into one column
# makes every comparison across colleges meaningless. Skipped ones are counted.
_AREA_RE = re.compile(
    r"([\d]{1,3}(?:,[\d]{2,3})*(?:\.\d+)?)\s*"
    r"(acres?|hectares?|hect\.?|ha\b|sq\.?\s*(?:ft|feet)|square\s+feet)", re.I)

# The number has to be near something that says it is the CAMPUS. Without this,
# "12 acres under the trust's mango orchard" and "5 acres donated in 1972"
# become the campus size.
_CAMPUS_CONTEXT = re.compile(
    r"\b(?:campus|spread|sprawl\w*|sprawling|land|premises|area|estate|"
    r"situated|located|built|extends?|covering|lush\s+green)\b", re.I)

# Wide, like every other envelope here: a fabrication and unit-error catcher,
# not an opinion. India's agricultural universities really do run to thousands
# of acres; nothing real is a 60,000-acre college.
_AREA_BOUNDS = {"acre": (0.25, 25_000.0), "hectare": (0.1, 10_000.0)}
_HECTARE_TO_ACRE = 2.4710538


def campus_size_from(text: str, name: str | None = None,
                     city: str | None = None
                     ) -> tuple[str, float | None, str] | None:
    """(verbatim value, acres if derivable, evidence quote) or None.

    When `name` is given, the acreage must also sit in a window that
    corroborates THIS college. That check is not optional in production and it
    exists because of a specific, caught failure: Sanskriti University's
    homepage carries a news item about an MoU with the University of
    California, Riverside, reading "The main campus sits on 1,900 acres in a
    suburban district of Riverside". Every other test passed — it is a real
    acreage, in bounds, and the sentence genuinely contains the word "campus" —
    and Sanskriti was recorded as a 1,900-acre campus in Mathura.

    A single number describing somebody else's institution is the worst error
    this pipeline can make, because the value is real and only the attribution
    is wrong, so nothing downstream looks anomalous.
    """
    if not text:
        return None
    for m in _AREA_RE.finditer(text):
        unit_raw = m.group(2).lower()
        if unit_raw.startswith("sq") or unit_raw.startswith("square"):
            continue                       # built-up area, not campus size
        unit = "hectare" if unit_raw.startswith(("hect", "ha")) else "acre"
        try:
            val = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        lo, hi = _AREA_BOUNDS[unit]
        if not (lo <= val <= hi):
            continue
        lo_i = max(0, m.start() - 160)
        window = text[lo_i:m.end() + 160]
        if not _CAMPUS_CONTEXT.search(window):
            continue
        # Whose campus? See the docstring — this is the check that keeps another
        # university's acreage out of this college's row.
        if name is not None and not _attributable(text, m.start(), name, city):
            continue
        # The stored string keeps the page's own number; only the unit word is
        # normalised, so "25" never silently becomes "25.0".
        label = f"{m.group(1)} {'hectares' if unit == 'hectare' else 'acres'}"
        acres = val if unit == "acre" else round(val * _HECTARE_TO_ACRE, 2)
        return label, acres, _evidence(text, lo_i, m.end() + 160, 220)
    return None


def sqft_skipped(text: str) -> int:
    """Area figures refused for being built-up area rather than campus size."""
    return sum(1 for m in _AREA_RE.finditer(text or "")
               if m.group(2).lower().startswith(("sq", "square")))


# ---------------------------------------------------------------------------
# CONTACT — email, phone, pincode
# ---------------------------------------------------------------------------
# The number/address on a template-built site's footer belongs to whoever built
# the site often enough that it has to be excluded by name.
_VENDOR = re.compile(
    r"\b(?:designed?|developed?|maintained?|powered|hosted|created)\s+(?:and\s+"
    r"\w+\s+)?by\b|\bwebsite\s+by\b|\bweb\s+(?:design|solutions?|development)\b"
    r"|\ball\s+rights\s+reserved\s+to\b", re.I)

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")

# Placeholder addresses that ship with themes, and asset filenames that parse as
# an address ("logo@2x.png").
_EMAIL_JUNK = re.compile(
    r"@(?:example|domain|yourdomain|email|test|sample|mydomain)\.|"
    r"\.(?:png|jpe?g|gif|webp|svg|css|js)$|^(?:your|name|email|user)@", re.I)

# Indian phone shapes: +91 98765 43210 / 0891-2755222 / (040) 2345 6789 /
# 1800-425-1234. Kept loose here and validated numerically below, because the
# separators colleges use are genuinely arbitrary.
_PHONE_RE = re.compile(
    r"(?<![\d])(?:\+?91[\s.\-]?)?(?:\(?0\d{1,4}\)?[\s.\-]?)?"
    r"\d{3,5}[\s.\-]?\d{3,5}(?![\d])")

_PHONE_CUE = re.compile(
    r"\b(?:phone|ph\.?|tel|telephone|mobile|mob\.?|contact|call|helpline|"
    r"landline|fax)\b", re.I)

_PIN_RE = re.compile(r"(?<!\d)([1-9]\d{5})(?!\d)")

# The first digit of an Indian PIN is its postal region, and every state sits in
# exactly one. A six-digit run whose region contradicts the college's own state
# is a house number, a phone fragment or an amount — not a PIN — and this map
# rejects it for free, without a network call or a model.
#
# Written out state by state rather than matched by substring: substring
# matching silently fails on "Jammu and Kashmir" (the region string spells it
# "jammu kashmir") and silently passes on any state containing the word
# "pradesh", which spans three different regions. An explicit table is
# auditable; a clever one is not. Every state present in tier 1 is listed, and
# anything absent — a new spelling, a foreign state — yields NO pincode rather
# than a guessed one.
PIN_REGION: dict[str, str] = {
    # region 1 — north
    "delhi": "1", "new delhi": "1", "haryana": "1", "punjab": "1",
    "chandigarh": "1", "himachal pradesh": "1", "jammu and kashmir": "1",
    "jammu & kashmir": "1", "ladakh": "1",
    # region 2 — UP and Uttarakhand
    "uttar pradesh": "2", "uttarakhand": "2", "uttaranchal": "2",
    # region 3 — west
    "rajasthan": "3", "gujarat": "3", "daman and diu": "3",
    "dadra and nagar haveli": "3",
    "dadra and nagar haveli and daman and diu": "3",
    # region 4 — central and Maharashtra
    "maharashtra": "4", "madhya pradesh": "4", "chhattisgarh": "4",
    "chattisgarh": "4", "goa": "4",
    # region 5 — the Deccan
    "andhra pradesh": "5", "telangana": "5", "telengana": "5",
    "karnataka": "5",
    # region 6 — the south
    "tamil nadu": "6", "tamilnadu": "6", "kerala": "6", "puducherry": "6",
    "pondicherry": "6", "lakshadweep": "6",
    # region 7 — east and north-east
    "west bengal": "7", "odisha": "7", "orissa": "7", "assam": "7",
    "sikkim": "7", "arunachal pradesh": "7", "manipur": "7",
    "meghalaya": "7", "mizoram": "7", "nagaland": "7", "tripura": "7",
    "andaman and nicobar islands": "7", "andaman and nicobar": "7",
    # region 8 — Bihar and Jharkhand
    "bihar": "8", "jharkhand": "8",
}


def pin_region_of(state: str | None) -> str | None:
    """The PIN first digit this state's addresses must start with, or None."""
    if not state:
        return None
    key = re.sub(r"\s+", " ", str(state).strip().lower())
    return PIN_REGION.get(key)


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s or "")


def normalise_phone(raw: str) -> str | None:
    """The page's number in a dialable form, or None if it is not a phone.

    Rejecting is the common case and the important one: the loose regex above
    happily matches a pincode, a year range and a fee, and every one of those
    would be stored as a phone number nobody answers.
    """
    d = _digits(raw)
    if d.startswith("91") and len(d) == 12:
        d = d[2:]
    elif d.startswith("0") and len(d) in (11, 12):
        d = d.lstrip("0")
    if len(d) == 10 and d[0] in "6789":
        return f"+91-{d}"                       # mobile
    if len(d) == 11 and d.startswith("1800"):
        return d                                # toll-free
    if 10 <= len(d) <= 11 and d[0] in "23458":
        return f"+91-{d}"                       # landline with STD code
    return None


# Another institution, named. The pattern deliberately requires a proper noun
# next to the institution word, so "The College campus is spread over 25 acres"
# — a college describing itself generically, which is extremely common — is not
# mistaken for a reference to somebody else. Articles and possessives are
# excluded for the same reason.
_OTHER_INSTITUTION = re.compile(
    r"\b(?!The\b|A\b|An\b|Our\b|This\b|Its\b)[A-Z][A-Za-z]{2,}\s+"
    r"(?:University|Institute|College|Academy|Vidyapeeth|Vishwavidyalaya)\b"
    r"|\b(?:University|Institute|College|Academy)\s+of\s+[A-Z][A-Za-z]{2,}")


def _attributable(text: str, at: int, name: str, city: str | None) -> bool:
    """Is the claim at `at` about THIS college, or about someone else's campus?

    An identity WINDOW alone is not enough, and Sanskriti University proved it
    twice. Its homepage carries an MoU news item about the University of
    California, Riverside: "The main campus sits on 1,900 acres ... with a
    branch campus of 20 acres in Palm Desert." Rejecting the 1,900 on a window
    check simply promoted the 20 — both belong to UCR, and both had a stray
    "SU students" close enough to look corroborated.

    So the question is not "is our name near this?" but "is anyone ELSE's name
    near this?", asked in that order:

        our identity in the sentence, or its neighbours   -> ours
        another institution named there instead           -> NOT ours
        nobody named at all                               -> ours

    The last branch is the common and correct case: "The campus is spread over
    25 acres" on a college's own site is about that college. The middle branch
    is the one that costs us data — a page whose acreage sentence sits next to
    an affiliation notice loses its campus size — and that is the right way to
    be wrong.
    """
    seg, _ = _segment(text, at, at, span=260)
    i = text.find(seg)
    if i < 0:
        i, seg = max(0, at - 260), text[max(0, at - 260):at + 260]
    prev, _ = _segment(text, max(0, i - 2), max(0, i - 1), span=260)
    nxt, _ = _segment(text, min(len(text), i + len(seg) + 1),
                      min(len(text), i + len(seg) + 2), span=260)
    if page_is_about(name, city, f"{prev} {seg} {nxt}"):
        return True
    return not _OTHER_INSTITUTION.search(f"{prev} {seg}")


VENDOR_LOOKBACK = 140        # chars before a contact value


def _vendor_before(text: str, at: int) -> bool:
    """Is this value the web designer's rather than the college's?

    Checked only in the text BEFORE the value, and only just before it. A
    vendor always credits itself first and prints its contact details after —
    "Designed and developed by Webnet Solutions, Kochi. Ph: +91 99887 76655".
    Looking AFTER the value instead condemns the college's own phone number for
    having a designer credit further down the same footer, which is how this
    check first managed to reject every correct contact on a short page.
    """
    return bool(_VENDOR.search(text[max(0, at - VENDOR_LOOKBACK):at]))


def _identity_near(name: str, city: str | None, text: str, at: int) -> str | None:
    """The window around position `at`, if it corroborates THIS college.

    Reuses `page_is_about` rather than re-deriving what a distinctive name word
    is — the same judgement, applied to 600 characters instead of a whole page.
    Returns the window so it can be stored as the fact's evidence.
    """
    if _vendor_before(text, at):
        return None
    lo = max(0, at - IDENTITY_WINDOW)
    window = text[lo:at + IDENTITY_WINDOW]
    if not page_is_about(name, city, window):
        return None
    return _evidence(text, lo, at + IDENTITY_WINDOW)


def checkable_identity(name: str, city: str | None) -> bool:
    """Can this college be identified in text at all?

    `page_is_about` returns True when the name offers nothing distinctive to
    match on — correct for a whole page (refusing to judge beats rejecting good
    data), but wrong for a 600-character window, where it would wave through the
    web designer's phone number. Probing it with empty text is exactly how to
    ask "is there anything to match on", without duplicating its stopword list.
    """
    return not page_is_about(name, city, "")


def _site_host(site: str) -> str:
    return (urlparse(site).netloc or "").lower().removeprefix("www.")


# Mailboxes that speak for the institution rather than for one desk inside it.
# Without this, IIT Bombay's contact page yielded "jeeadv@iitb.ac.in" — a real
# address, but the JEE Advanced help desk, not the institute. The first
# same-domain address on a page is whichever one the layout happened to put
# first, and that is not a property worth storing.
_EMAIL_PREFERRED = re.compile(
    r"^(?:info|contact|enquiry|enquiries|inquiry|office|admin|administration|"
    r"registrar|principal|director|dean|admission|admissions|mail|helpdesk|"
    r"reception|college|institute|university)\b", re.I)


def email_from(text: str, name: str, city: str | None, site: str
               ) -> tuple[str, float, str] | None:
    """(email, confidence, evidence). Same-domain first, identity window second.

    Among same-domain addresses an institutional mailbox wins over a
    departmental one; among the rest, only an address the surrounding text ties
    to this college is eligible at all.
    """
    host = _site_host(site)
    fallback: tuple[str, float, str] | None = None
    on_domain: tuple[str, float, str] | None = None
    for m in _EMAIL_RE.finditer(text or ""):
        addr = m.group(0)
        if _EMAIL_JUNK.search(addr) or len(addr) > 90:
            continue
        low = addr.lower()
        domain = low.split("@", 1)[1]
        # An address on the college's own domain needs no further argument, and
        # deliberately gets no vendor check either: the college controls that
        # domain, so the mailbox is the college's whatever text surrounds it.
        if host and (domain == host or domain.endswith("." + host)
                     or host.endswith("." + domain)):
            quote = _evidence(text, m.start() - 120, m.end() + 120)
            if _EMAIL_PREFERRED.match(low.split("@", 1)[0]):
                return addr, 0.85, quote     # institutional: settled
            if on_domain is None:
                on_domain = (addr, 0.80, quote)
            continue
        # A free-mail or third-party address is only this college's if the
        # surrounding text says so. Held back as a fallback so a same-domain
        # address anywhere on the page always wins.
        if fallback is None:
            window = _identity_near(name, city, text, m.start())
            if window:
                fallback = (addr, 0.72, window)
    return on_domain or fallback


def phone_from(text: str, name: str, city: str | None
               ) -> tuple[str, float, str] | None:
    """(phone as printed, confidence, evidence) from a contact-page window."""
    for m in _PHONE_RE.finditer(text or ""):
        raw = m.group(0).strip()
        if normalise_phone(raw) is None:
            continue
        # A phone cue nearby separates "0891-2755222" from a stray number pair.
        lo = max(0, m.start() - 60)
        if not _PHONE_CUE.search(text[lo:m.end() + 40]):
            continue
        window = _identity_near(name, city, text, m.start())
        if window:
            return raw, 0.75, window
    return None


def pincode_from(text: str, state: str | None, name: str = "",
                 city: str | None = None, *, need_identity: bool = False
                 ) -> tuple[str, str] | None:
    """(pincode, evidence) — six digits whose region agrees with the state.

    Used for both sources. From the catalogue address the identity is already
    established by the row itself; from a fetched page it is not, so the caller
    asks for the identity window.
    """
    # No mappable state means the region check cannot run, and a PIN we cannot
    # check is a PIN we do not write. Rule 3: an empty column is a true
    # statement.
    region = pin_region_of(state)
    if region is None:
        return None
    for m in _PIN_RE.finditer(text or ""):
        pin = m.group(1)
        if pin[0] != region:
            continue
        if need_identity:
            window = _identity_near(name, city, text, m.start())
            if not window:
                continue
        else:
            window = _evidence(text, m.start() - 90, m.end() + 60)
        return pin, window
    return None


# ---------------------------------------------------------------------------
# MEDIA — social profiles, gallery images, brochure, video
# ---------------------------------------------------------------------------
# A share widget is not a profile. facebook.com/sharer.php?u=... appears on tens
# of thousands of pages and points at nothing the college owns, so the platform
# rules below demand a real profile path and refuse the sharing endpoints.
_SOCIAL_RULES: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    # (key, host suffixes, first-path-segments that are NOT profiles)
    ("facebook", ("facebook.com", "fb.com", "fb.me"),
     ("sharer", "sharer.php", "share.php", "dialog", "plugins", "login",
      "tr", "help", "policies", "profile.php")),
    ("twitter", ("twitter.com", "x.com"),
     ("intent", "share", "home", "search", "hashtag", "i", "privacy", "tos")),
    ("instagram", ("instagram.com",),
     ("p", "reel", "reels", "explore", "accounts", "tv", "stories")),
    ("linkedin", ("linkedin.com",),
     ("shareArticle", "sharing", "share-offsite", "login", "signup", "feed",
      "pulse", "jobs")),
    ("youtube", ("youtube.com", "youtu.be"),
     ("watch", "embed", "results", "playlist", "shorts", "signin")),
    ("telegram", ("t.me", "telegram.me"), ("share", "joinchat")),
)
# Any of these in the query string means the link is a share action carrying the
# current page, not a destination the college owns.
_SHARE_QS = re.compile(r"(?:^|&)(?:u|url|text|link|via|mini)=", re.I)


def _social_key(url: str) -> str | None:
    p = urlparse(url)
    host = p.netloc.lower().removeprefix("www.")
    for key, hosts, bad_first in _SOCIAL_RULES:
        if host != hosts[0] and not any(host == h or host.endswith("." + h)
                                        for h in hosts):
            continue
        if _SHARE_QS.search(p.query or ""):
            return None
        segs = [s for s in p.path.split("/") if s]
        if not segs:
            return None                     # bare facebook.com — not a profile
        if segs[0] in bad_first:
            return None
        if key == "youtube":
            # A channel is a social profile; a /watch is a video and is handled
            # as one. Distinguishing them is why both fields can be filled from
            # the same footer without either being wrong.
            if not (segs[0] in ("channel", "c", "user") or segs[0].startswith("@")
                    or host.endswith("youtu.be")):
                return None
        if key == "linkedin" and segs[0] not in ("company", "school", "in",
                                                 "edu", "groups"):
            return None
        return key
    return None


def social_from(base_url: str, links: list[tuple[str, str]]) -> dict[str, str]:
    """{platform: profile_url}, first occurrence wins, share widgets refused."""
    out: dict[str, str] = {}
    for href, anchor in links:
        if not href:
            continue
        absolute = urljoin(base_url, href)
        if not absolute.lower().startswith(("http://", "https://")):
            continue
        key = _social_key(absolute)
        if key and key not in out and not _VENDOR.search(anchor or ""):
            out[key] = absolute.split("#")[0][:400]
    return out


_VIDEO_RE = re.compile(
    r"^https?://(?:www\.)?(?:youtube\.com/(?:watch\?[^\"']*v=|embed/)"
    r"|youtu\.be/|player\.vimeo\.com/video/|vimeo\.com/\d)", re.I)


def video_from(base_url: str, links: list[tuple[str, str]], html: str
               ) -> str | None:
    """The first embedded/linked video on the college's own page."""
    for href, _ in links:
        if not href:
            continue
        absolute = urljoin(base_url, href)
        if _VIDEO_RE.match(absolute):
            return absolute.split("&")[0][:400]
    for m in re.finditer(r"""<iframe[^>]+src=["']([^"']+)["']""", html or "",
                         re.I):
        src = urljoin(base_url, m.group(1))
        if _VIDEO_RE.match(src):
            return src.split("&")[0][:400]
    return None


_IMG_RE = re.compile(r"""<img[^>]+?src\s*=\s*["']([^"']+)["']""", re.I)
# Photographs, not graphics. PNG is excluded deliberately: on these sites it is
# overwhelmingly the format for logos, seals, accreditation badges and UI
# chrome, and a gallery with a bank logo in it is visibly broken to anyone who
# looks. The occasional real PNG photograph is a fair price for that.
_IMG_EXT = re.compile(r"\.(?:jpe?g|webp)(?:\?|$)", re.I)
# Site furniture, not campus photographs. A gallery page's <img> tags include
# the header logo, the social icons and the loading spinner, and storing those
# as "gallery_urls" would make the field worthless the first time anyone looked.
_IMG_JUNK = re.compile(
    r"logo|icon|favicon|sprite|placeholder|loader|loading|spinner|avatar|"
    r"arrow|bullet|btn|button|banner|whatsapp|facebook|twitter|insta|"
    r"youtube|linkedin|blank|spacer|pixel|1x1|thumb[-_]?nav|captcha|"
    r"accessib|header|footer|watermark|emblem|ashoka|naac[-_]?logo", re.I)


def gallery_from(base_url: str, html: str, limit: int = MAX_GALLERY_IMAGES
                 ) -> list[str]:
    """Same-origin campus images from a gallery page, chrome filtered out."""
    origin = urlparse(base_url).netloc
    out: list[str] = []
    seen: set[str] = set()
    for m in _IMG_RE.finditer(html or ""):
        src = (m.group(1) or "").strip()
        if not src or src.startswith("data:"):
            continue
        absolute = urljoin(base_url, src).split("#")[0]
        p = urlparse(absolute)
        if p.scheme not in ("http", "https") or p.netloc != origin:
            continue
        if not _IMG_EXT.search(p.path) or _IMG_JUNK.search(absolute):
            continue
        if absolute in seen or len(absolute) > 400:
            continue
        seen.add(absolute)
        out.append(absolute)
        if len(out) >= limit:
            break
    return out


_BROCHURE_RE = re.compile(
    r"prospect\w*|brochure|e[-\s]?brochure|information\s*(?:bulletin|brochure)"
    r"|admission\s*booklet", re.I)


def brochure_candidates(base_url: str, links: list[tuple[str, str]]
                        ) -> list[tuple[str, str]]:
    """[(url, anchor)] — same-origin PDFs that call themselves a prospectus."""
    origin = urlparse(base_url).netloc
    out: list[tuple[str, str]] = []
    for href, anchor in links:
        if not href:
            continue
        clean = href.split("#")[0]
        if not clean.lower().split("?")[0].endswith(".pdf"):
            continue
        absolute = urljoin(base_url, clean)
        p = urlparse(absolute)
        if p.scheme not in ("http", "https") or p.netloc != origin:
            continue
        if _BROCHURE_RE.search(f"{p.path} {anchor or ''}"):
            out.append((absolute[:400], (anchor or "").strip()[:120]))
    return out


def brochure_year(anchor: str, url: str) -> str | None:
    """The academic year of the prospectus, from its own link text or filename.

    `stated_year()` returns None here and is RIGHT to: it only trusts a year
    that sits near a fee/admission word, precisely so that a link to next
    year's prospectus cannot re-date a fee table published years earlier. That
    guard is load-bearing for the fee harvest and is not being relaxed.

    This is the opposite situation, and it needs saying explicitly: the
    document being dated IS the prospectus. A year in "Prospectus/2025-26.pdf"
    is that file's own edition, not a neighbour's date, so the page-proximity
    rule has nothing to bite on.

    It reuses `_AY_RE`, so "an academic year" means exactly what it means
    everywhere else in this project — including the span rule that rejects
    WordPress upload paths like "/uploads/2026/04/", where 2026-04 would
    otherwise read as a session.
    """
    for text in (anchor or "", url or ""):
        for m in _AY_RE.finditer(text):
            start = int(m.group(1))
            tail = m.group(2)
            end = int(tail) if len(tail) == 4 else int(str(start)[:2] + tail)
            if 2000 <= start <= datetime.now(timezone.utc).year + 2 \
                    and end - start in (0, 1):
                return f"{start}-{str(end)[-2:]}"
    return None


async def head_is_pdf(client: httpx.AsyncClient, pol: Politeness,
                      url: str) -> bool:
    """Confirm the brochure link actually resolves to a PDF.

    A link labelled "Prospectus" that 404s is a wrong value, not a missing one —
    the client's directory would show a broken download. One HEAD is cheap
    enough to make the difference, and it goes through the same robots and
    per-domain pacing as everything else.
    """
    if not await pol.allowed(client, url):
        return False
    await pol.wait(url)
    try:
        r = await client.head(url, follow_redirects=True)
        if r.status_code in (405, 501):       # HEAD refused: do not guess
            return False
        if r.status_code != 200:
            return False
        ctype = r.headers.get("content-type", "").lower()
        return "pdf" in ctype or url.lower().endswith(".pdf") and "html" not in ctype
    except Exception:  # noqa: BLE001 - unreachable is "do not store it"
        return False


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
# COALESCE(new, existing) in every branch: this harvester must never blank a
# column it happened not to find on a given run. Re-running it is therefore
# safe, and a later, better run can only add.
COLLEGE_UPDATE = """
UPDATE colleges SET
    facilities   = COALESCE(%s::jsonb, facilities),
    gallery_urls = COALESCE(%s::jsonb, gallery_urls),
    social_links = COALESCE(%s::jsonb, social_links),
    brochure_url = COALESCE(%s, brochure_url),
    video_url    = COALESCE(%s, video_url),
    campus_size  = COALESCE(%s, campus_size),
    email        = COALESCE(%s, email),
    phone        = COALESCE(%s, phone),
    pincode      = COALESCE(%s, pincode)
WHERE college_id = %s"""

# The row that makes the value refreshable. A value written only to `colleges`
# is undated and invisible to the nightly pass, which is the whole reason this
# table exists — so nothing is written above without a row here.
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

# Column name -> the `field` recorded in college_field. Kept explicit so the two
# writes cannot drift apart silently.
FIELD_ORDER = ("facilities", "gallery_urls", "social_links", "brochure_url",
               "video_url", "campus_size", "email", "phone", "pincode")


def _now() -> datetime:
    return datetime.now(timezone.utc)


class CampusWriter:
    """Buffers colleges and lands each batch in ONE transaction.

    Same reasoning as HarvestWriter in web_enrich: crawling a college takes
    seconds under the politeness pace, so a transaction held open across one
    would pin an XID for minutes. And the two statements belong together — a
    `colleges` column filled without its `college_field` row is exactly the
    undated, un-refreshable value the client asked us to stop producing.
    """

    def __init__(self, every: int = 25, dry_run: bool = False):
        self.every = every
        self.dry_run = dry_run
        self._cols: list[tuple] = []
        self._fields: list[tuple] = []
        self._done = 0
        self.written_fields = 0
        self.written_colleges = 0

    def add(self, cid: str, values: dict, provenance: dict) -> None:
        """`values` is column -> value; `provenance` is field -> row tuple tail."""
        if values:
            self._cols.append(tuple(values.get(c) for c in FIELD_ORDER) + (cid,))
            self.written_colleges += 1
        for field, (value, value_num, source, url, year, conf) in provenance.items():
            self._fields.append((cid, field, value, value_num, source, url, year,
                                 _now(), conf))
            self.written_fields += 1
        self._done += 1

    @property
    def due(self) -> bool:
        return self._done >= self.every

    async def flush(self) -> None:
        cols, self._cols = self._cols, []
        fields, self._fields = self._fields, []
        self._done = 0
        if self.dry_run or not (cols or fields):
            return
        await asyncio.to_thread(self._commit, cols, fields)

    @staticmethod
    def _commit(cols: list[tuple], fields: list[tuple]) -> None:
        with _pool().connection() as c:      # commits on exit, rolls back on error
            cur = c.cursor()
            if cols:
                cur.executemany(COLLEGE_UPDATE, cols)
            if fields:
                cur.executemany(FIELD_UPSERT, fields)


# ---------------------------------------------------------------------------
# One college
# ---------------------------------------------------------------------------

def _gate(fact_value: str, page_text: str, name: str, city: str | None,
          conf: float, quote: str) -> tuple[bool, str]:
    """Run one text-valued finding through verify_facts. (accepted, reason).

    The fact is built so the verifier sees exactly what the page said: the value
    verbatim plus the window it came from. Every number in either is a literal
    substring of `page_text`, so anchoring can only fail if the extractor
    fabricated something — which is the case it exists to catch.
    """
    fact = {"summary": _evidence(quote, 0, len(quote), 200),
            "facts": {"value": fact_value},
            "confidence": conf}
    res = verify_fact(fact, page_text, name, city)
    if res.verdict == "REJECTED":
        return False, "; ".join(res.reasons)[:160]
    if res.verdict == "UNCERTAIN":
        return False, f"uncertain: {'; '.join(res.reasons)[:140]}"
    return True, ""


def _gate_address(value: str, address: str, name: str, city: str | None,
                  state: str | None) -> tuple[bool, str]:
    """The same gate, for a value read out of `colleges.address`.

    Two of `verify()`'s checks do not transfer from a fetched page to a column
    we already own, and forcing them produced 786 false rejections out of 863
    before this function existed. Both are documented here rather than relaxed
    inside verify_facts, because that floor protects every fee the web sweep
    harvests and must not move for this module's convenience.

    1. THE LENGTH FLOOR. `verify()` rejects any source under 120 characters as
       "no source text to verify against". Correct about a page — a 40-character
       page is a stub or an error document — and wrong about an address, which
       is 60-110 characters by nature. Dropped here.

    2. `page_is_about`. Its job is to catch a page belonging to someone else,
       which is a live risk when a URL was guessed. It is not a risk here: the
       address is a column of THIS college's row, so identity comes from the
       primary key, not from text matching. Demanding it also fails on the many
       addresses that simply do not repeat the institution's name — "Taleigao
       Plateau, Goa 403206" is Goa University's real address and mentions
       neither "Goa University" nor anything distinctive. It rejected 210
       correct pincodes.

    What replaces them is a corroboration an address CAN actually offer: it has
    to name the college's own city or state. Measured across tier 1, 862 of 863
    addresses do, so this is a real check that costs almost nothing — and it
    still catches the failure that matters, a row carrying another place's
    address. `anchor_numbers` is kept unchanged, and the PIN region digit has
    already been checked against the college's state by `pincode_from`.
    """
    if not address or not address.strip():
        return False, "no address to verify against"
    low = address.lower()
    place = [p.lower() for p in (city, state) if p and len(p) > 2]
    if not place:
        return False, "college has no city or state to corroborate the address"
    if not any(p in low for p in place):
        return False, (f"address names neither the college's city nor its "
                       f"state ({place})")
    # The address goes in WHOLE, never truncated. Slicing it to 200 characters
    # cut "411057" in half, and `anchor_numbers` then correctly reported that
    # "4110" is not on the page — a self-inflicted rejection of a correct
    # pincode. Any truncation of text that is about to be number-anchored can
    # manufacture a number that was never there.
    fact = {"summary": address, "facts": {"value": value}, "confidence": 0.9}
    unanchored, _anchors = anchor_numbers(fact, address)
    if unanchored:
        return False, f"not present in the address: {unanchored[:3]}"
    return True, ""


async def harvest_one(client: httpx.AsyncClient, pol: Politeness,
                      college: dict, fc: FirecrawlBudget | None = None) -> dict:
    """Fetch a college's campus pages and return everything defensible.

    Returns {"values": {column: value}, "provenance": {field: row}, "stats": {}}.
    """
    cid = college["college_id"]
    name = college["name"] or ""
    city = college.get("city")
    state = college.get("state")
    site = normalise_site(college.get("website_url"))
    values: dict = {}
    prov: dict = {}
    stats = {"pages": 0, "status": "empty", "rejected": 0, "bare_hostel": 0,
             "sqft_skipped": 0, "no_identity": 0}

    # ---- the pincode we already own -------------------------------------
    # Done before any network call: the address is a column, and the standing
    # finding on this project is that the client's own data holds more than the
    # open web does. The address IS the source document, so it goes through the
    # same gate rather than around it.
    address = college.get("address") or ""
    hit = pincode_from(address, state)
    if hit:
        pin, quote = hit
        ok, why = _gate_address(pin, address, name, city, state)
        if ok:
            values["pincode"] = pin
            prov["pincode"] = (pin, float(pin), "catalogue_address", None,
                               None, 0.90)
        else:
            stats["rejected"] += 1

    if not site:
        stats["status"] = "no_site"
        return {"values": values, "provenance": prov, "stats": stats}

    home = await fetch(client, pol, site, fc)
    if home is None:
        stats["status"] = "blocked"
        return {"values": values, "provenance": prov, "stats": stats}
    stats["pages"] += 1
    home_text, home_links = parse_html(home)

    # ---- THE GATE FOR EVERY LINK ON THIS SITE ---------------------------
    # If the homepage does not corroborate this college, the catalogue's
    # website_url points somewhere else — a rebrand, a parked domain, a typo, a
    # shared trust portal. Every social profile, image and brochure below would
    # then be filed under a college that never published it, so the college is
    # abandoned rather than partially trusted.
    #
    # "Cannot tell" and "wrong college" are reported as DIFFERENT statuses. A
    # JS-rendered homepage parses to zero visible text (dpsru.edu.in does), and
    # calling that `wrong_site` would tell the client we had found someone
    # else's website when all we found was an empty shell. Both stop the
    # harvest; only one of them is an accusation.
    if len(home_text.strip()) < 200:
        stats["status"] = "no_text"
        return {"values": values, "provenance": prov, "stats": stats}
    if not page_is_about(name, city, home_text):
        stats["status"] = "wrong_site"
        return {"values": values, "provenance": prov, "stats": stats}

    pages: list[tuple[str, str, str, str]] = [("home", site, home, home_text)]
    for kind, url in pick_campus_pages(site, home_links):
        html = await fetch(client, pol, url, None)   # no Firecrawl on subpages
        if html is None:
            continue
        stats["pages"] += 1
        text, _ = parse_html(html)
        pages.append((kind, url, html, text))

    all_links: list[tuple[str, str]] = list(home_links)
    for kind, url, html, text in pages[1:]:
        _, sub_links = parse_html(html)
        all_links.extend((urljoin(url, h), a) for h, a in sub_links)

    # ---- facilities ------------------------------------------------------
    facilities: dict[str, str] = {}
    fac_url = None
    fac_conf = 1.0
    for kind, url, _html, text in pages:
        stats["bare_hostel"] += bare_hostel_mentions(text)
        got = facilities_from(text, kind, name, city)
        if not got:
            continue
        accepted: dict[str, str] = {}
        conf = 0.80 if kind in ("facilities", "gallery") else 0.70
        for key, quote in got.items():
            if key in facilities:
                continue
            ok, why = _gate(key.replace("_", " "), text, name, city, conf, quote)
            if ok:
                accepted[key] = quote
            else:
                stats["rejected"] += 1
        if accepted:
            facilities.update(accepted)
            # The recorded confidence is the WEAKEST evidence in the list, not
            # a constant. It was hardcoded at 0.80 — the facilities-page score —
            # which stamped that on lists assembled entirely from homepage
            # prose, overstating them. A list is only as good as its weakest
            # member, because the client reads one number for the whole field.
            fac_conf = min(fac_conf, conf)
            if fac_url is None or kind == "facilities":
                fac_url = url
    if facilities:
        items = sorted(facilities)
        values["facilities"] = json.dumps(items)
        prov["facilities"] = (json.dumps(items), float(len(items)),
                              "college_site", fac_url, None, round(fac_conf, 2))

    # ---- campus size -----------------------------------------------------
    for kind, url, _html, text in pages:
        stats["sqft_skipped"] += sqft_skipped(text)
        if "campus_size" in values:
            continue
        got = campus_size_from(text, name, city)
        if not got:
            continue
        label, acres, quote = got
        ok, why = _gate(label, text, name, city, 0.80, quote)
        if ok:
            values["campus_size"] = label
            prov["campus_size"] = (label, acres, "college_site", url,
                                   stated_year(text), 0.80)
        else:
            stats["rejected"] += 1

    # ---- contacts --------------------------------------------------------
    # Only from the contact/about pages, and only for a college whose name
    # gives the identity check something to work with.
    if checkable_identity(name, city):
        # Contact page first, homepage last. The homepage of a large university
        # carries a departmental address ("cit@jmi.ac.in" — the IT centre) while
        # the contact page carries the institutional one, and whichever is read
        # first wins. Ordering by how authoritative the page is, rather than by
        # the order it happened to be fetched in, is the whole fix.
        contact_order = {"contact": 0, "about": 1, "home": 2}
        for kind, url, _html, text in sorted(
                pages, key=lambda p: contact_order.get(p[0], 9)):
            if kind not in contact_order:
                continue
            if "email" not in values:
                got = email_from(text, name, city, site)
                if got:
                    addr, conf, quote = got
                    ok, _why = _gate(addr, text, name, city, conf, quote)
                    if ok:
                        values["email"] = addr
                        prov["email"] = (addr, None, "college_site", url,
                                         None, conf)
                    else:
                        stats["rejected"] += 1
            if "phone" not in values and kind in ("contact", "about"):
                got = phone_from(text, name, city)
                if got:
                    raw, conf, quote = got
                    ok, _why = _gate(raw, text, name, city, conf, quote)
                    if ok:
                        values["phone"] = raw
                        prov["phone"] = (raw, None, "college_site", url,
                                         None, conf)
                    else:
                        stats["rejected"] += 1
            if "pincode" not in values and kind == "contact":
                got = pincode_from(text, state, name, city, need_identity=True)
                if got:
                    pin, quote = got
                    ok, _why = _gate(pin, text, name, city, 0.75, quote)
                    if ok:
                        values["pincode"] = pin
                        prov["pincode"] = (pin, float(pin), "college_site",
                                           url, None, 0.75)
    else:
        stats["no_identity"] = 1

    # ---- media -----------------------------------------------------------
    # URL-valued: gated by the homepage identity check above and by same-origin,
    # not by verbatim anchoring — a URL lives in an href and is not in the
    # visible text the anchor check reads.
    socials = social_from(site, all_links)
    if socials:
        values["social_links"] = json.dumps(socials)
        prov["social_links"] = (json.dumps(socials), float(len(socials)),
                                "college_site", site, None, 0.75)

    for kind, url, html, _text in pages:
        if kind != "gallery":
            continue
        imgs = gallery_from(url, html)
        if imgs:
            values["gallery_urls"] = json.dumps(imgs)
            prov["gallery_urls"] = (json.dumps(imgs), float(len(imgs)),
                                    "college_site", url, None, 0.70)
        break

    for kind, url, html, _text in pages:
        vid = video_from(url, all_links if kind == "home" else [], html)
        if vid:
            values["video_url"] = vid
            prov["video_url"] = (vid, None, "college_site", url, None, 0.70)
            break

    for url, anchor in brochure_candidates(site, all_links)[:2]:
        if await head_is_pdf(client, pol, url):
            stats["pages"] += 1
            # The prospectus year is the one date that matters for this field:
            # a 2019-20 prospectus offered as current is exactly the staleness
            # this project has already been burned by once.
            year = brochure_year(anchor, url)
            values["brochure_url"] = url
            prov["brochure_url"] = (url, None, "college_site", url, year, 0.80)
            break

    stats["status"] = "ok" if prov else "empty"
    return {"values": values, "provenance": prov, "stats": stats}


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

SELECT_TIER = """
SELECT p.college_id, c.name, c.city, c.state, c.address, c.website_url
FROM college_priority p JOIN colleges c USING(college_id)
WHERE p.tier = ?
ORDER BY p.score DESC
LIMIT ? OFFSET ?"""


async def sweep(tier: int = 1, limit: int | None = None, offset: int = 0,
                concurrency: int = 8, delay: float = 1.5,
                dry_run: bool = False, sample: int = 0,
                ) -> dict:
    _pool()   # fail now, not after an hour of politely rate-limited crawling
    todo = get_backend().q(SELECT_TIER, [tier, limit or 1_000_000, offset])
    if not todo:
        print(f"[campus] tier {tier} is empty", file=sys.stderr)
        return {}

    print(f"[campus] {len(todo)} tier-{tier} colleges "
          f"(concurrency {concurrency}, {delay}s per domain"
          f"{', DRY RUN' if dry_run else ''})", file=sys.stderr)

    pol = Politeness(delay=delay)
    sem = asyncio.Semaphore(concurrency)
    writer = CampusWriter(every=25, dry_run=dry_run)
    counters: dict[str, int] = {}
    per_field: dict[str, int] = {}
    nothing = 0
    web_nothing = 0
    shown = 0
    done = 0
    t0 = time.perf_counter()
    lock = asyncio.Lock()

    # No Firecrawl. `harvest_one` accepts a budget so a caller can rescue a
    # JS-rendered site, but this sweep never spends one: the account's ~1,000
    # credits/month are the fee harvest's rescue path, and a campus gallery is
    # not worth a credit that could unblock a fee table. Blocked and
    # JS-only sites are reported as such and left empty.
    limits = httpx.Limits(max_connections=concurrency * 2,
                          max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(headers=HEADERS, timeout=20.0, limits=limits,
                                 follow_redirects=True, verify=False) as client:

        async def one(row: dict):
            nonlocal done, nothing, web_nothing, shown
            async with sem:
                try:
                    got = await harvest_one(client, pol, dict(row), None)
                except Exception as exc:  # noqa: BLE001 - one site must not stop the run
                    print(f"[campus] {row['college_id']}: {type(exc).__name__}: "
                          f"{str(exc)[:110]}", file=sys.stderr)
                    got = {"values": {}, "provenance": {},
                           "stats": {"status": "error"}}
                async with lock:
                    done += 1
                    st = got["stats"]
                    counters[st["status"]] = counters.get(st["status"], 0) + 1
                    for k in ("rejected", "bare_hostel", "sqft_skipped",
                              "no_identity"):
                        counters[k] = counters.get(k, 0) + st.get(k, 0)
                    for f in got["provenance"]:
                        per_field[f] = per_field.get(f, 0) + 1
                    if not got["provenance"]:
                        nothing += 1
                    # Reported separately, because the pincode comes out of a
                    # column and would otherwise make every college look like a
                    # successful crawl. This is the number that says how the
                    # WEB half of the job actually went.
                    if not any(p[2] != "catalogue_address"
                               for p in got["provenance"].values()):
                        web_nothing += 1
                    writer.add(row["college_id"], got["values"],
                               got["provenance"])
                    if sample and shown < sample and got["provenance"]:
                        shown += 1
                        _print_sample(dict(row), got)
                    if writer.due:
                        await writer.flush()
                        rate = done / max(time.perf_counter() - t0, 1e-6)
                        print(f"[campus] {done}/{len(todo)} "
                              f"ok={counters.get('ok',0)} "
                              f"empty={counters.get('empty',0)} "
                              f"blocked={counters.get('blocked',0)} "
                              f"wrong_site={counters.get('wrong_site',0)} "
                              f"fields={writer.written_fields} | {rate:.2f}/s",
                              file=sys.stderr)

        batch = concurrency * 20
        for i in range(0, len(todo), batch):
            await asyncio.gather(*(one(r) for r in todo[i:i + batch]))
    await writer.flush()

    mins = (time.perf_counter() - t0) / 60
    print(f"\n[campus] {'DRY RUN ' if dry_run else ''}finished in {mins:.1f}m",
          file=sys.stderr)
    n = max(len(todo), 1)
    print(f"[campus] colleges: {len(todo)} | yielded NOTHING AT ALL: {nothing} "
          f"({100*nothing/n:.0f}%) | nothing from the WEB: {web_nothing} "
          f"({100*web_nothing/n:.0f}%)", file=sys.stderr)
    print(f"[campus] status: {counters}", file=sys.stderr)
    print(f"[campus] fields written: {writer.written_fields}", file=sys.stderr)
    for f, n in sorted(per_field.items(), key=lambda kv: -kv[1]):
        print(f"[campus]   {f:<14} {n:>5}", file=sys.stderr)
    return {"colleges": len(todo), "nothing": nothing,
            "web_nothing": web_nothing, "fields": writer.written_fields,
            "per_field": per_field, "status": counters}


def _print_sample(row: dict, got: dict) -> None:
    """One college, printed with its evidence, so a human can hand-check it."""
    print(f"\n--- {row['name'][:64]} ({row.get('city')}) "
          f"{row.get('website_url') or ''}")
    for field, (value, num, source, url, year, conf) in got["provenance"].items():
        v = value if len(str(value)) < 130 else str(value)[:127] + "..."
        print(f"    {field:<13} = {v}")
        print(f"    {'':<13}   [{source} conf={conf} "
              f"{'year=' + year + ' ' if year else ''}{(url or '-')[:70]}]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tier", type=int, default=1,
                    help="college_priority tier to work (default 1)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8,
                    help="parallel COLLEGES; pacing is still per-domain")
    ap.add_argument("--delay", type=float, default=1.5,
                    help="seconds between requests to one host")
    ap.add_argument("--dry-run", action="store_true",
                    help="extract and report, write nothing")
    ap.add_argument("--sample", type=int, default=0,
                    help="print this many colleges with their evidence")
    args = ap.parse_args()
    asyncio.run(sweep(tier=args.tier, limit=args.limit, offset=args.offset,
                      concurrency=args.concurrency, delay=args.delay,
                      dry_run=args.dry_run, sample=args.sample))


if __name__ == "__main__":
    main()
