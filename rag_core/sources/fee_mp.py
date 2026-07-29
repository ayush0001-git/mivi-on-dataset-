"""Madhya Pradesh AFRC — the fee a private MP college is LEGALLY allowed to charge.

WHY THIS SOURCE
23,864 colleges (61.7%) in the catalogue have no fee at all, and what the
catalogue does hold is a whole-programme figure with no year attached. The
Admission and Fee Regulatory Committee (AFRC), a statutory body under the
Madhya Pradesh Niji Vyavsayik Shikshan Sanstha (Pravesh Ka Viniyaman Avam
Shulk Ka Nirdharan) Adhiniyam 2007, DETERMINES the chargeable fee for every
private professional institution in the state and publishes it:

    https://web.afrcmp.org/FeesInformation/Frm_ShowInstitutes.aspx

That is not somebody's estimate of a fee. It is the ceiling the institution may
lawfully collect — the order text says an institution charging above it is
"charging capitation fee for which legal action ... shall be initiated". It is
also PERIODIC (per semester or per year), which is the only annualisable fee
data this project has found anywhere.

WHAT THE HARVEST HOLDS
    2,144 rows across 22 programmes and three streams
    T technical (B.E./B.Tech, M.B.A., B.Pharm., Diploma, M.C.A., ...)
    H higher-ed  (B.Ed., M.Ed., B.P.Ed., the integrated law degrees)
    M medical    (M.B.B.S., B.D.S., B.A.M.S., B.H.M.S., nursing)

THE PERIOD IS READ, NEVER ASSUMED
Each row carries `fee_basis` — the AFRC portal's own column header, either
"Semester wise Fees (in Rs)" or "Year wise Fees (in Rs)". 1,155 rows are per
SEMESTER and 989 per YEAR, so treating the printed number as annual would
have inflated 54% of this state by 2x. That is the exact bug documented in
RESUME.md as the most consequential this project has had (an empty period
defaulted to "Year" and every catalogue fee came out 2-4x too high), so:

  * a semester basis is annualised as fee x 2, and the payload states the
    printed figure, the period, the multiplier and the fact that it was
    annualised — the conversion is auditable, not silent;
  * a basis this module does not recognise DROPS the row. There is no default.

The x2 is confirmed against the two sample orders on disk. The B.E. order for
Gwalior Institute of Information Technology prints "Tuition Fee (Per Semester)
(Rs.) 22800.00", and the harvest's GIIT row reads exactly 22800 with a semester
basis; the B.Ed. order is per year and the harvest's B.Ed. rows are year-basis.
Every AICTE/PCI/NCTE programme in this table runs two semesters to an academic
session, so 2 is the multiplier, not an average.

MATCHING IS registries.match_one, UNCHANGED
The family gate, token containment, dotted-acronym gate and leading-brand gate
each exist because of a real mis-match, and a second matcher would repeat them.
Candidates are drawn from Madhya Pradesh only — AFRC has no jurisdiction
outside the state, so a candidate elsewhere is wrong by definition.

City comes from `place` when the catalogue knows that place, and otherwise is
recovered out of the free-text `address` with naac._city_from_address, which
normalises punctuation to whitespace and refuses to let a STATE outrank a CITY.

The one thing this adapter must do to the NAME before handing it over is strip
the town AFRC appends to it. 1,519 of the 2,085 usable rows end with their own
place — "Acropolis Institute of Technology Research, Indore" — while the
catalogue holds "Acropolis Institute of Technology and Research". match_one's
token-containment gate tolerates one missing token of at most four letters, so
a six-letter town on the registry side vetoed the match outright and the first
dry run refused 1,766 rows with NO candidate surviving the gates at all.
registries.match_one already forgives the mirror image of this (a place appended
on the CATALOGUE side scores 0.85); the trailing place is removed here only when
it is the same place the row's own `place`/address resolved to, so this is the
same accommodation, not a looser one.

CONTESTED COLLEGES GET NOTHING
AFRC lists "Arihant College, Indore" three times for M.B.A. at 19,500 and
32,500, and "Radha Krishna College, Datia" twice for B.Ed. at 32,000 and
35,000. When two source rows land on the same college and programme, at least
one fee is wrong for it, and the higher match score is not a tie-break — so
both are refused and logged to registry_unmatched. Telling a student a fee that
belongs to a different institution is worse than telling them nothing.

Run:
    python -m rag_core.sources.fee_mp --dry-run
    python -m rag_core.sources.fee_mp
    python -m rag_core.sources.fee_mp --report
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .. import config  # noqa: F401 - loads .env before the backend reads it
from ..store.backend import get_backend
from .naac import _city_from_address
from .registries import (_index_by_city, _index_by_token, _load_place_aliases,
                         _norm, _place_key, _sig, ensure_tables, match_one)

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "raw" / "registries" / "fees" / "afrc_mp_2024_sample.json"

REGISTRY = "fra_mp"
STATE = "Madhya Pradesh"
SOURCE_URL = "https://web.afrcmp.org/FeesInformation/Frm_ShowInstitutes.aspx"
AUTHORITY = "Admission and Fee Regulatory Committee, Madhya Pradesh"

# The portal's own column header -> (period, periods per academic year).
#
# This mapping is the whole safety mechanism of this module. A basis string that
# is not in here is NOT annualised and NOT stored: an unrecognised period is
# exactly the condition under which the 2-4x inflation bug happened, and the
# only safe response is to drop the row and count it.
_BASIS = {
    "semester wise fees (in rs)": ("semester", 2),
    "year wise fees (in rs)": ("year", 1),
}

# Plausibility, applied to the figure AS PRINTED and again after annualising.
# The harvest carries 59 rows whose institute name is literally "2" and whose
# fee is "5" — a column-index leak in the source table — plus 7 rows printed as
# "0" where AFRC has determined no fee. Both are noise, not cheap colleges.
MIN_PERIOD_INR = 1_000
MAX_ANNUAL_INR = 5_000_000

# Text that carries digits which are NOT money: appeal numbers, writ-petition
# numbers, order dates, academic sessions. Scrubbed before any figure is read,
# so "Appeal No. 14/2023 order dated 04.09.2023" cannot contribute a rupee
# amount.
_SCRUB = re.compile(
    r"appeal\s*no\.?\s*\S+|w\.?\s*p\.?\s*no\.?\s*\S+|dated\s*[0-9./-]+"
    r"|\b\d+\s*/\s*\d{4}\b|\b20\d{2}\s*-\s*\d{2,4}\b", re.I)
_LEADING = re.compile(r"^\s*(?:rs\.?\s*)?([0-9][0-9,]*(?:\.\d{1,2})?)", re.I)
_FIGURE = re.compile(r"(?:rs\.?\s*)?([0-9][0-9,]{3,}(?:\.\d{1,2})?)", re.I)
_ORDER_DATE = re.compile(r"dated\s*(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{4})", re.I)
_ORDER_REF = re.compile(r"(appeal\s*no\.?\s*[\w/\-]+|w\.?\s*p\.?\s*no\.?\s*[\w/\-]+)",
                        re.I)

# AFRC spellings the catalogue writes differently. Deliberately tiny and
# hand-checked against the catalogue's own city list: a wrong equivalence here
# would let one town's college corroborate another's and drop the match floor
# from 0.92 to 0.72 on the strength of a typo.
_MP_PLACE_FORMS = {
    "chhindwada": "chhindwara",
    "narsingpur": "narsinghpur",
    "badwani": "barwani",
    "chhattarpur": "chhatarpur",
    "singroli": "singrauli",
    "hosangabad": "hoshangabad",
}


def _num(text: str) -> float | None:
    try:
        return float(str(text).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def parse_fee(fee_text: str) -> dict | None:
    """The rupee figure(s) an AFRC row prints, as printed — no annualising here.

    Returns {"low", "high", "varies_by_branch"} or None when the row carries no
    usable amount.

    2,118 of the 2,144 rows are a bare number. The 26 that are not fall into
    exactly two shapes, both verified by reading every one of them:

      "39200Revised as per Appeal No. 14/2023 order dated 04.09.2023"
          one fee, plus the order that revised it

      "36800 For non-accredited branches and Rs. 41800 For Accredited branches
       Computer Science and Engineering, ..."
          TWO legally approved fees for one programme at one college: a base
          rate and a higher rate for NAAC/NBA-accredited branches

    A second figure is only read when the text says "accredit". Without that
    gate a stray number in prose becomes a fee, which is the failure mode this
    whole module exists to avoid.
    """
    raw = str(fee_text or "").strip()
    if not raw:
        return None
    lead = _LEADING.match(raw)
    if not lead:
        return None
    low = high = _num(lead.group(1))
    if low is None:
        return None

    if re.search(r"accredit", raw, re.I):
        figures = [f for f in (_num(x) for x in _FIGURE.findall(_SCRUB.sub(" ", raw)))
                   if f is not None and f >= MIN_PERIOD_INR]
        if len(figures) >= 2:
            low, high = min(figures), max(figures)

    if low < MIN_PERIOD_INR:
        return None
    return {"low": low, "high": high, "varies_by_branch": high > low}


def parse_order(fee_text: str) -> tuple[str | None, str | None]:
    """(ISO order date, order reference) when the row names the order revising it.

    AFRC writes dd.mm.yyyy. Only rows that were revised on appeal or by a court
    carry one; the rest are covered by the block order for the academic year,
    whose date the harvest does not include. Those get None rather than a
    guessed date — an invented order date would make a fee look verified to a
    day it was never verified on.
    """
    raw = str(fee_text or "")
    date = None
    m = _ORDER_DATE.search(raw)
    if m:
        d, mo, y = (int(x) for x in m.groups())
        try:
            date = datetime(y, mo, d).date().isoformat()
        except ValueError:
            date = None
    ref = _ORDER_REF.search(raw)
    return date, (" ".join(ref.group(1).split()) if ref else None)


def load_rows(path: Path = SRC) -> tuple[list[dict], dict]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    rows = doc.get("rows") or []
    meta = {
        "academic_year": doc.get("academic_year"),
        "harvested": doc.get("harvested"),
        "authority": doc.get("authority") or AUTHORITY,
        "source": doc.get("source") or SOURCE_URL,
    }
    return rows, meta


def _mp_catalogue() -> list[dict]:
    """Madhya Pradesh only. AFRC regulates nothing outside the state, so a
    candidate anywhere else is not a near miss — it is a wrong answer."""
    return get_backend().q(
        "SELECT college_id, name, city, district, state FROM colleges "
        "WHERE NOT is_abroad AND state ILIKE ?", [STATE])


# A trailing state marker, which AFRC appends to 45 institute names.
_TRAIL_STATE = re.compile(r"\s+(?:m\s*p|madhya\s+pradesh)$", re.I)

# How close a token has to be to count as the SAME word spelt differently.
# "iess" vs "ies" scores 0.86; the four wrong matches this rule was written
# against scored 0.50, 0.33, 0.33 and 0.29, so the gap is wide and the exact
# threshold is not load-bearing.
BRAND_SIM_MIN = 0.75

# Abbreviations AFRC writes where the catalogue writes the word out. These are
# NOT brands and refusing on them threw away 11 correct matches in the dry run
# ("Truba Institute of Engg. Information Technology" against the catalogue's
# "...of Engineering and Information Technology"). Kept as an explicit list
# rather than a prefix rule, because a prefix rule would also equate "sam" with
# "samaj" and "nri" with "nripendra".
_ABBREV = {
    "engg": "engineering", "engng": "engineering", "engr": "engineering",
    "tech": "technology", "techno": "technology",
    "mgmt": "management", "mgt": "management", "mangement": "management",
    "sci": "science", "scs": "science",
    "inst": "institute", "instt": "institute", "insti": "institute",
    "coll": "college", "univ": "university", "vidhyalaya": "vidyalaya",
    "edu": "education", "educ": "education", "res": "research",
    "pharm": "pharmacy", "poly": "polytechnic", "info": "information",
    "intl": "international", "natl": "national", "prof": "professional",
}

# Discipline words a college name carries when that IS what the college is.
# Medical is deliberately absent: medical colleges genuinely run nursing
# schools, so treating "Medical" as incompatible with B.Sc. Nursing would refuse
# correct matches to prevent none.
_DISCIPLINE = {
    "pharmacy": (r"\bpharmac(?:y|eutical)\b", r"\bpharma\b"),
    "nursing": (r"\bnursing\b",),
    "education": (r"\beducation\b", r"\bshiksha\b", r"\bteachers?\b",
                  r"\bb\.?\s?ed\b", r"\bted\b"),
    "polytechnic": (r"\bpolytechnic\b",),
    "law": (r"\blaw\b", r"\bl\.?\s?l\.?\s?b\b", r"\bvidhi\b"),
    "management": (r"\bmanagement\b", r"\bprabandh\w*\b"),
    "engineering": (r"\bengineering\b", r"\btechnology\b", r"\btechnical\b"),
}

# The discipline each AFRC programme belongs to. Only these programmes are
# discipline-checked, and that asymmetry is the point: a B.Ed. or a B.Pharm. fee
# landing on the trust's engineering or management college is a real error seen
# in the dry run, while an M.B.A. or M.C.A. fee landing on an engineering
# college is usually right, because engineering colleges run those courses.
#
# "Diploma (3 Years)" is the polytechnic diploma and is deliberately NOT mapped:
# in MP it is taught inside engineering colleges, so requiring "Polytechnic" in
# the name would refuse most of them.
_PROGRAM_DISCIPLINE = {
    "B.Pharm.": "pharmacy", "M.Pharmacy": "pharmacy",
    "D.Pharmacy": "pharmacy", "Diploma Pharmacy (2 Years)": "pharmacy",
    "B.Sc. Nursing": "nursing", "M.Sc. Nursing": "nursing",
    "B.Ed.": "education", "M.Ed.": "education", "B.P.Ed.": "education",
    "B.B.A. L.L.B.": "law", "B.Com. L.L.B.": "law",
}


def _disciplines(name: str) -> set[str]:
    low = " " + _norm(name) + " "
    raw = " " + str(name or "").lower() + " "
    return {d for d, pats in _DISCIPLINE.items()
            if any(re.search(p, low) or re.search(p, raw) for p in pats)}


def veto(afrc_name: str, college_name: str, program: str) -> str | None:
    """Why this match must be thrown away, or None to keep it.

    These are REFUSALS layered on top of registries.match_one, never a
    relaxation of it. Both were written from wrong matches found by hand-reading
    the lowest-scoring accepted pairs in the dry run.

    1. A DIFFERENT BRAND, not a spelling of the same one. match_one tolerates
       one missing token of up to four letters, because honorifics and
       abbreviations drop out of catalogue names routinely. Indian college
       brands are frequently three or four letters too, so that tolerance let
       through:

           BRD College of Pharmacy       -> Dr APJ Abdul Kalam College of Pharmacy
           NRI Institute of Pharmacy     -> R.K.D.F. Institute of Pharmacy Science
           EKTA Mahavidhyalaya           -> Maa Sharda Mahavidhyalaya
           Sun Institute of Teachers Ed. -> Om Institute of Technical and Teachers Ed.

       all four with the right city and the right discipline — a fee attached to
       a real, precise, entirely different college. A tolerated miss is only
       allowed when some catalogue token is the same word spelt differently
       ("IESs" for "IES"), which every one of those four fails.

    2. THE WRONG FACULTY OF THE RIGHT TRUST. One trust runs several colleges
       under one brand, and the name gates cannot separate them because the
       brand is what they share:

           Vidyasagar College   [B.Ed.] -> Vidyasagar College of Pharmacy
           Malhotra College     [B.Ed.] -> Malhotra College of Pharmacy
           Satpuda B.Ed College [B.Ed.] -> Satpuda Polytechnic College

       The B.Ed. fee is real; the pharmacy college is not the one it belongs to.
    """
    afrc_sig, cat_sig = _sig(afrc_name), _sig(college_name)
    # The catalogue name with every separator removed, so a dotted acronym still
    # reads as a word: "Dr. A.P.J. Abdul Kalam" -> "drapjabdulkalam" contains
    # "apj", which _sig shreds into three single letters and drops.
    squashed = re.sub(r"[^a-z0-9]", "", str(college_name or "").lower())
    # AFRC appends the programme to an institute name when a trust runs the same
    # brand across wings ("Shri Ram Institute of Technology MCA"). That names the
    # COURSE, which this fact already records as its category, so it is not a
    # brand the catalogue is missing.
    prog_marker = re.sub(r"[^a-z]", "", str(program or "").lower())

    for token in afrc_sig - cat_sig:
        if _ABBREV.get(token) in cat_sig or token == prog_marker:
            continue
        if len(token) >= 3 and token in squashed:
            continue
        if max((difflib.SequenceMatcher(None, token, t).ratio() for t in cat_sig),
               default=0.0) < BRAND_SIM_MIN:
            return (f"AFRC name carries {token!r}, which is nothing the "
                    f"catalogue name spells — a different institution")

    want = _PROGRAM_DISCIPLINE.get(program)
    if want:
        found = _disciplines(college_name)
        # Only a discipline the CATALOGUE name asserts and the AFRC name does
        # not. Without that subtraction the check fires on trust names —
        # "Evergreen Education Society College" and "Shri Gajanan Shiksha Samiti
        # College" both read as "education colleges" while being the exact
        # institution AFRC named, matched at 1.00.
        extra = found - _disciplines(afrc_name)
        if want not in found and extra:
            return (f"catalogue college is a {'/'.join(sorted(extra))} college "
                    f"but the AFRC row is {program} — wrong faculty of the "
                    f"same trust")
    return None


def strip_trailing_place(name: str, *places: str) -> str:
    """Remove the town AFRC tacks onto the end of an institute name.

    ONLY from the end, and ONLY when it is a place the row itself resolved to.
    Both restrictions matter:

      * a LEADING place is the brand ("Bhopal Institute of Technology" is not
        "Institute of Technology"), and registries.match_one leans on the first
        significant word to reject a foreign trust — removing it would defeat
        that gate;
      * removing an arbitrary town name would let "Nagpur Institute of
        Technology, Wardha" lose the wrong half.

    The result must keep at least two distinctive words, otherwise the name has
    been reduced to something too generic to identify anything and the original
    is used instead.

    Punctuation is preserved on purpose. registries.match_one has a DOTTED
    ACRONYM gate that reads "S.G.B.M." straight out of the raw string, and an
    earlier version of this function normalised the name first — which silently
    disabled that gate and let "S.G.B.M. Institute of Technology Science" match
    "PRT Institute of Technology and Science" at 0.82, two unrelated Jabalpur
    colleges whose only shared words are "Institute of Technology Science".
    """
    cleaned = _TRAIL_STATE.sub("", str(name or "")).strip()
    changed = True
    while changed:
        changed = False
        for place in places:
            p = _norm(place)
            if not p:
                continue
            trimmed = re.sub(r"[\s,.\-]*" + re.escape(p) + r"[\s,.]*$", "",
                             cleaned, flags=re.I)
            if trimmed != cleaned and len(
                    [w for w in _norm(trimmed).split() if len(w) > 2]) >= 2:
                cleaned, changed = trimmed.strip(), True
    return cleaned or str(name or "")


def _place_for(row: dict, by_city: dict, aliases: dict, city_keys: set[str],
               state_keys: set[str]) -> tuple[str, str]:
    """(place key the catalogue recognises, where it came from)."""
    key = _place_key(row.get("place"), aliases)
    key = _MP_PLACE_FORMS.get(key, key)
    if key and key in by_city:
        return key, "place"
    # `place` is often a village or tehsil the catalogue has never heard of
    # ("BhelaKalan", "kachhua"). The address usually names the district town as
    # well, so it is scanned the same way NAAC addresses are.
    found = _city_from_address(row.get("address") or "", city_keys,
                               row.get("institute") or "", state_keys)
    if found:
        return found, "address"
    return "", "none"


def build(path: Path = SRC, limit: int | None = None) -> tuple[list[dict], list[tuple], dict]:
    """Match every usable AFRC row. Returns (writes, misses, stats)."""
    rows, meta = load_rows(path)
    if limit:
        rows = rows[:limit]
    year = meta["academic_year"] or "2024-2025"

    colleges = _mp_catalogue()
    aliases = _load_place_aliases()
    by_city = _index_by_city(colleges, aliases)
    by_token = _index_by_token(colleges)
    # CITY keys only for the address scan — by_city also holds districts, and a
    # district bucket is every college in it (see naac._city_from_address).
    city_keys = {k for k in (_place_key(c.get("city"), aliases) for c in colleges) if k}
    state_keys = {k for k in (_place_key(c.get("state"), aliases) for c in colleges) if k}
    # Catalogue names that are not unique inside their own city. match_one
    # returns one college and no notion of a tie, so a name held by two colleges
    # in one town would be resolved by whichever difflib saw first — a coin
    # flip. Only 4 MP colleges are affected, but a coin flip on a fee is exactly
    # what this module refuses to do.
    name_counts: dict[tuple[str, str], int] = defaultdict(int)
    for c in colleges:
        name_counts[(_place_key(c.get("city"), aliases), _norm(c["name"]))] += 1

    print(f"[fra-mp] {len(rows):,} AFRC rows for {year} vs {len(colleges):,} "
          f"{STATE} colleges ({len(by_city):,} place keys)", file=sys.stderr)

    stats = {"rows": len(rows), "junk": 0, "no_fee": 0, "bad_basis": 0,
             "over_ceiling": 0, "matched": 0, "unmatched": 0, "no_city": 0,
             "annualised": 0, "contested_rows": 0, "contested_colleges": 0,
             "ambiguous_catalogue": 0, "vetoed": 0}
    now = datetime.now(timezone.utc)
    claims: dict[tuple[str, str], list] = defaultdict(list)
    misses: list[tuple] = []
    samples: list[str] = []

    for r in rows:
        institute = " ".join(str(r.get("institute") or "").split())
        program = str(r.get("program") or "").strip() or "unknown"
        sno = str(r.get("sno") or "")
        ref_id = f"{r.get('stream') or '?'}/{program}/{sno}"

        # Column-index leak in the source table: institute "2", address "3",
        # place "4", fee "5". Dropped by name, not by fee, so a genuinely cheap
        # college is never mistaken for one of these.
        if len(_norm(institute)) < 4 or institute.isdigit():
            stats["junk"] += 1
            continue

        basis_raw = str(r.get("fee_basis") or "").strip()
        basis = _BASIS.get(basis_raw.lower())
        if basis is None:
            # NO DEFAULT. See the module docstring.
            stats["bad_basis"] += 1
            misses.append((REGISTRY, ref_id, program, year, institute,
                           r.get("place"), STATE, 0.0,
                           f"unrecognised fee basis {basis_raw!r} — refused "
                           f"rather than assumed annual"))
            continue
        period, per_year = basis

        fee = parse_fee(r.get("fee_text"))
        if fee is None:
            stats["no_fee"] += 1
            continue

        annual_low = int(round(fee["low"] * per_year))
        annual_high = int(round(fee["high"] * per_year))
        if annual_high > MAX_ANNUAL_INR:
            stats["over_ceiling"] += 1
            misses.append((REGISTRY, ref_id, program, year, institute,
                           r.get("place"), STATE, 0.0,
                           f"annualised fee Rs {annual_high:,} above the "
                           f"Rs {MAX_ANNUAL_INR:,} ceiling — treated as a "
                           f"source error"))
            continue
        if per_year != 1:
            stats["annualised"] += 1

        city, city_src = _place_for(r, by_city, aliases, city_keys, state_keys)
        if not city:
            stats["no_city"] += 1
        lookup = strip_trailing_place(institute, city, r.get("place") or "")
        college, score, why = match_one({"name": lookup, "city": city},
                                        by_city, by_token, aliases)
        if college is None:
            stats["unmatched"] += 1
            misses.append((REGISTRY, ref_id, program, year, institute,
                           city or r.get("place"), STATE, round(score, 3), why))
            continue
        killed = veto(lookup, college["name"], program)
        if killed:
            stats["vetoed"] += 1
            misses.append((REGISTRY, ref_id, program, year, institute, city,
                           STATE, round(score, 3),
                           f"{killed} (scored {score:.2f} against "
                           f"{college['name']!r})"))
            continue
        if name_counts[(_place_key(college.get("city"), aliases),
                        _norm(college["name"]))] > 1:
            stats["ambiguous_catalogue"] += 1
            misses.append((
                REGISTRY, ref_id, program, year, institute, city, STATE,
                round(score, 3),
                f"catalogue holds more than one {college['name']!r} in "
                f"{college.get('city')} — cannot tell which one AFRC means"))
            continue

        order_date, order_ref = parse_order(r.get("fee_text"))
        note = str(r.get("fee_text") or "").strip()
        payload = {
            # ANNUAL. The name says so because the ambiguity is what caused the
            # 2-4x bug; `approved_tuition_inr` is the same number under the key
            # fee_promote.py already reads for every fra_* registry.
            "approved_tuition_annual_inr": annual_high,
            "approved_tuition_inr": annual_high,
            "approved_tuition_annual_min_inr": annual_low,
            "period_fee_inr": int(round(fee["high"])),
            "period_fee_min_inr": int(round(fee["low"])),
            "period": period,
            "periods_per_year": per_year,
            "annualised": per_year != 1,
            "fee_basis": basis_raw,
            "fee_varies_by_branch": fee["varies_by_branch"],
            "program": program,
            # fee_promote.py renders `stream` into the label a student reads
            # ("...approved tuition [ENGG]" for Maharashtra). AFRC's own stream
            # is a bare letter — T/H/M — which says nothing, so the programme
            # goes here and the letter is kept beside it under its own name.
            "stream": program,
            "stream_code": r.get("stream"),
            "academic_year": year,
            "order_date": order_date,
            "order_ref": order_ref,
            "fee_note": note if not note.replace(",", "").replace(".", "").isdigit() else None,
            "authority": meta["authority"],
            "authority_name": institute,
            "authority_place": r.get("place"),
            "authority_address": " ".join(str(r.get("address") or "").split()),
            "matched_city": city,
            "city_source": city_src,
            "as_on": meta["harvested"],
        }
        claims[(college["college_id"], program)].append(
            (college, score, why, payload, institute, ref_id, city,
             annual_high, order_date))

    # ---- refuse contested (college, programme) pairs -----------------------
    # One college has ONE approved fee per programme per session. Two AFRC rows
    # claiming the same pair means at least one is attached to the wrong
    # college, and the top score is not evidence of which — measured here,
    # "Arihant College, Indore" claims M.B.A. three times at 19,500 and 32,500
    # per semester, a 2x spread on a number a student plans around.
    #
    # Rows that are byte-identical in institute name AND annualised fee are the
    # same listing repeated, not a contest, so one of them is kept.
    writes: list[dict] = []
    for (cid, program), group in claims.items():
        if len(group) > 1:
            distinct = {(_norm(g[4]), g[7]) for g in group}
            if len(distinct) > 1:
                stats["contested_colleges"] += 1
                stats["contested_rows"] += len(group)
                fees = sorted({g[7] for g in group})
                for college, score, why, payload, institute, ref_id, city, _a, _d in group:
                    misses.append((
                        REGISTRY, ref_id, program, year, institute, city, STATE,
                        round(score, 3),
                        f"{len(group)} AFRC rows matched the same college for "
                        f"{program} (annual fees {fees}) — refused rather than "
                        f"guessed"))
                continue
        college, score, why, payload, institute, ref_id, city, annual, odate = group[0]
        stats["matched"] += 1
        writes.append({
            "row": (cid, REGISTRY, ref_id, program, year,
                    json.dumps(payload, ensure_ascii=False), SOURCE_URL,
                    SOURCE_URL, round(score, 3), why, now),
            "sample": (f"  Rs {annual:>9,}/yr  {payload['period_fee_inr']:>7,}"
                       f"/{payload['period'][:3]}  {program:<26} "
                       f"{institute[:36]:38} -> {college['name'][:34]:36} "
                       f"({score:.2f} {city or 'no-city'})"),
        })

    for w in writes[:14]:
        samples.append(w["sample"])
    stats["writes"] = len(writes)
    return writes, misses, stats, samples


def run(*, dry_run: bool = False, limit: int | None = None,
        path: Path = SRC) -> dict:
    ensure_tables()
    writes, misses, stats, samples = build(path, limit)

    print("\n".join(samples), file=sys.stderr)
    print(f"\n[fra-mp] {stats['rows']:,} rows: "
          f"{stats['junk']:,} junk, {stats['no_fee']:,} no usable fee, "
          f"{stats['bad_basis']:,} unrecognised basis, "
          f"{stats['over_ceiling']:,} above the ceiling", file=sys.stderr)
    print(f"[fra-mp] {stats['annualised']:,} per-semester figures annualised x2 "
          f"(basis recorded on every row)", file=sys.stderr)
    print(f"[fra-mp] matched {stats['matched']:,} college-programme fees, "
          f"{stats['unmatched']:,} refused by the matcher, "
          f"{stats['no_city']:,} had no recoverable city", file=sys.stderr)
    print(f"[fra-mp] vetoed after matching: {stats['vetoed']:,} wrong "
          f"brand/faculty, {stats['ambiguous_catalogue']:,} duplicated "
          f"catalogue name", file=sys.stderr)
    if stats["contested_colleges"]:
        print(f"[fra-mp] REFUSED {stats['contested_colleges']:,} "
              f"college-programme pairs claimed by {stats['contested_rows']:,} "
              f"competing AFRC rows", file=sys.stderr)

    if dry_run:
        print("[fra-mp] DRY RUN — nothing written", file=sys.stderr)
        return stats

    be = get_backend()
    # Clean replace, like naac.py: everything here is re-derived from the file,
    # and an upsert alone would strand a fee that this run now REFUSES.
    removed = be.execute(f"DELETE FROM registry_fact WHERE registry = '{REGISTRY}'")
    if removed:
        print(f"[fra-mp] cleared {removed:,} facts from the previous run",
              file=sys.stderr)
    be.execute(f"DELETE FROM registry_unmatched WHERE registry = '{REGISTRY}'")

    be.executemany(
        "INSERT INTO registry_fact (college_id, registry, ref_id, category, "
        "year, payload, source_url, detail_url, match_score, match_note, "
        "fetched_at) VALUES (?,?,?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT (college_id, registry, category, year) DO UPDATE SET "
        "payload = EXCLUDED.payload, match_score = EXCLUDED.match_score, "
        "match_note = EXCLUDED.match_note, fetched_at = EXCLUDED.fetched_at",
        [w["row"] for w in writes])
    try:
        be.executemany(
            "INSERT INTO registry_unmatched (registry, ref_id, category, year, "
            "name, city, state, best_score, note) VALUES (?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT (registry, ref_id, category, year) DO UPDATE SET "
            "best_score = EXCLUDED.best_score, note = EXCLUDED.note", misses)
    except Exception as exc:  # noqa: BLE001 - the misses log is diagnostic only
        print(f"[fra-mp] could not record misses ({str(exc)[:80]})", file=sys.stderr)

    n = be.one(f"SELECT COUNT(*) FROM registry_fact WHERE registry='{REGISTRY}'") or 0
    c = be.one("SELECT COUNT(DISTINCT college_id) FROM registry_fact "
               f"WHERE registry='{REGISTRY}'") or 0
    print(f"[fra-mp] stored {n:,} approved fees across {c:,} colleges",
          file=sys.stderr)
    stats["stored"], stats["colleges"] = n, c
    return stats


def report() -> None:
    be = get_backend()
    n = be.one(f"SELECT COUNT(*) FROM registry_fact WHERE registry='{REGISTRY}'") or 0
    c = be.one("SELECT COUNT(DISTINCT college_id) FROM registry_fact "
               f"WHERE registry='{REGISTRY}'") or 0
    print(f"-- Madhya Pradesh AFRC --\n   {n:,} approved annual fees "
          f"across {c:,} colleges")
    if not n:
        return
    print(f"\n   {'programme':<28} {'n':>5} {'period':<9} "
          f"{'median annual':>14}  {'range':>26}")
    for r in be.q(
            "SELECT category, COUNT(*) n, MIN(payload->>'period') period, "
            "PERCENTILE_CONT(0.5) WITHIN GROUP "
            "(ORDER BY (payload->>'approved_tuition_annual_inr')::numeric) med, "
            "MIN((payload->>'approved_tuition_annual_inr')::numeric) lo, "
            "MAX((payload->>'approved_tuition_annual_inr')::numeric) hi "
            f"FROM registry_fact WHERE registry='{REGISTRY}' "
            "GROUP BY category ORDER BY n DESC"):
        print(f"   {str(r['category']):<28} {r['n']:>5} {str(r['period']):<9} "
              f"Rs {int(r['med']):>11,}  Rs {int(r['lo']):>9,} - Rs {int(r['hi']):>9,}")
    gained = be.one(
        "SELECT COUNT(DISTINCT f.college_id) FROM registry_fact f "
        "LEFT JOIN college_agg a USING(college_id) "
        f"WHERE f.registry='{REGISTRY}' AND a.min_tuition_inr IS NULL") or 0
    print(f"\n   colleges gaining a fee they had NONE for: {gained:,}")
    ref = be.one("SELECT COUNT(*) FROM registry_unmatched "
                 f"WHERE registry='{REGISTRY}'") or 0
    print(f"   AFRC rows refused (kept as leads): {ref:,}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="match and report, write nothing")
    ap.add_argument("--limit", type=int, help="only the first N AFRC rows")
    ap.add_argument("--file", type=Path, default=SRC)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
        return
    run(dry_run=a.dry_run, limit=a.limit, path=a.file)


if __name__ == "__main__":
    main()
