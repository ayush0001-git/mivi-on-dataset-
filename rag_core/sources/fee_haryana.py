"""Haryana AFRC — the fee a private college is LEGALLY PERMITTED to charge.

WHY THIS SOURCE
Same argument as the Maharashtra adapter next door (sources/fee_authority.py):
the Admission & Fee Regulating Committee under the Haryana Private Technical
Educational Institutions Act must APPROVE the fee a private technical college
may charge, and it publishes the approved figure with the order that fixed it.
That beats a college's own website on every axis that matters — it is the
legally binding number, it is per YEAR (the catalogue's own fee is
whole-programme), and it comes with a date.

INPUT IS ON DISK, ALREADY HARVESTED
    data/raw/registries/fees/haryana_afrc_feechart_2026-07-29.json
    1,255 rows: course, sr, institute, tuition, development, total,
                order_url, remarks
The per-course HTML the rows were parsed from sits in fees/haryana_afrc/ and is
what the column-shift note below was checked against. Nothing here touches the
network.

--------------------------------------------------------------------------
THE YEAR RULE, AND WHY 598 REAL FEES ARE THROWN AWAY
--------------------------------------------------------------------------
An undated fee is the failure this project spent a day fixing, so every stored
row must carry an academic session that the SOURCE states. Haryana states it
two ways and only one of them is usable:

    "...for the session 2025-26 onwards allowed vide letter No.318/A&FRC
     dated 28.05.2025."                                    <- session STATED
    "The fee was fixed vide order dated 08.06.2022."       <- date only, 466 rows

It is tempting to derive the session from the order date. That was MEASURED
against the 653 rows which state both, tabulating (session year - order year)
by the month of the order:

    month  1: {-1: 12, 0: 17}        month  7: {-5: 4, -1: 1, 0: 62}
    month  2: {-1: 8, 0: 133}        month  8: {-1: 2, 0: 24, 1: 8}
    month  3: {0: 69}                month  9: {-1: 2, 0: 15, 1: 7}
    month  4: {-2: 2, -1: 5, 0: 62}  month 10: {0: 4, 1: 12}
    month  5: {-1: 5, 0: 60, 1: 2}   month 11: {-1: 1, 0: 4, 1: 14, 3: 2}
    month  6: {-1: 4, 0: 59, 1: 1}   month 12: {0: 1, 1: 51}

A January order is a 12-vs-17 coin flip between two different sessions, and a
November order lands on four different ones. There is no rule to extract. So an
order date is recorded as PROVENANCE on rows that are kept, and is never
promoted into a year. Rows with only a date are dropped and logged.

--------------------------------------------------------------------------
"Closed in the year 2019-20" IS NOT AN ACADEMIC SESSION
--------------------------------------------------------------------------
121 rows carry a YYYY-YY that is the year the institution SHUT DOWN:

    "Closed in the year 2019-20."
    "AICTE approved progressive closure from academic year 2013-14"
    "HSBTE issued NOC for closure in academic year 2016- 17"

Two of those phrase the closure with the word "session"/"academic year", so a
year regex alone reads them as a live fee for a dead college — the single worst
row this file could write, since it sends a student to an institution that no
longer admits anyone. Closure is therefore checked FIRST, before any year is
parsed, and a closed row is dropped whatever else it says.

--------------------------------------------------------------------------
A COLUMN SHIFT THAT LOOKS LIKE DATA
--------------------------------------------------------------------------
The Hotel Management table has two extra columns the harvest did not allow for:

    B.Tech   Sr | Institute | Tution | Development | Total | Order | Remarks
    BHMCT    Sr | Institute | Tution | Development | Training & Other |
                                       Operational | Total | Order | Remarks

so on those 7 rows `total` actually holds "Training & Other Fee" and `remarks`
holds the total. tuition + development == total holds on 1,248 of 1,255 rows and
fails on exactly those 7, so that identity is used as the parse check and the
shifted rows are refused rather than published with a wrong total.

--------------------------------------------------------------------------
MATCHING
--------------------------------------------------------------------------
Reuses registries.match_one unchanged — family gate, token containment, dotted
acronym, leading brand — and naac._city_from_address for the place, because the
`institute` field is a name welded to a postal address:

    "PKG College of Engineering & Technology, Village Mohidinpur Thirana,
     Tehsil Madlauda, Assand Road, Distt. Panipa (formerly R.N. College...)"

Feeding that whole string in as a name fails token containment on every row, so
the name is cut at the address and the address half supplies the city. Candidates
are restricted to Haryana, which is itself a gate: the AFRC only regulates
Haryana institutions.

The AFRC spells several districts the way the state does and the catalogue uses
the renamed forms (Gurgaon/Gurugram, Sonepat/Sonipat, Hissar/Hisar). Those go in
as place ALIASES passed to the existing matcher, not as a new matcher.

Run:
    python -m rag_core.sources.fee_haryana --dry-run
    python -m rag_core.sources.fee_haryana
    python -m rag_core.sources.fee_haryana --report
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from .. import config  # noqa: F401 - loads .env before the backend is imported
from ..store.backend import get_backend
from .naac import _city_from_address
from .registries import (_STOP, _acronym, _index_by_city, _index_by_token,
                         _load_place_aliases, _norm, _place_key, _sig,
                         ensure_tables, match_one)

ROOT = Path(__file__).resolve().parents[2]
FEE_JSON = (ROOT / "data" / "raw" / "registries" / "fees"
            / "haryana_afrc_feechart_2026-07-29.json")

REGISTRY = "fra_hr"
STATE = "Haryana"
SOURCE_URL = "https://afrchry.techeduhry.gov.in/feechart"

# Approved fees are annual and in rupees; the observed range on this file is
# 14,000-122,700. The guard is the same one the Maharashtra adapter uses —
# anything outside it is a parse error, not a cheap or expensive college.
FEE_MIN, FEE_MAX = 1_000, 50_000_000

# How the AFRC spells a place vs how the catalogue does. Passed to the existing
# matcher as aliases rather than edited into registries._CITY_FORMS, which is
# shared with NIRF and NAAC. Every entry is a district RENAME or a spelling of
# the same town — no two distinct places are merged here.
HR_PLACE_ALIASES = {
    "gurgaon": "gurugram",
    "sonepat": "sonipat",
    "hissar": "hisar",
    "yamuna nagar": "yamunanagar",
    "yamunnagar": "yamunanagar",
    "mahendergarh": "mahendragarh",
    "mohindergarh": "mahendragarh",
    "kurukshetara": "kurukshetra",
    "panchkulla": "panchkula",
    "jhajjhar": "jhajjar",
    "faridbad": "faridabad",
    "mewat": "nuh",
    "charkhi dadri": "dadri",
}


# ---------------------------------------------------------------------------
# Field parsing
# ---------------------------------------------------------------------------

def _int(v) -> int | None:
    s = re.sub(r"[^\d]", "", str(v or ""))
    if not s:
        return None
    n = int(s)
    return n if FEE_MIN <= n <= FEE_MAX else None


# Checked before anything else — see the module docstring. "closure" catches
# "progressive closure", "NOC for closure"; the rest are the other ways a row
# says the institution is not running.
_CLOSED_RE = re.compile(
    r"clos(?:ed|ure|ing)|de-?affiliat|disaffiliat|discontinu|surrender(?:ed)?|"
    r"withdraw(?:n|al)|shut\s*down", re.I)

# A session only counts when the source CALLS it one. A bare "2019-20" is far
# more often a closure year than a fee year on this file.
_SESSION_RE = re.compile(
    r"(?:session|academic\s+year)s?\s*[:\-]?\s*(20\d{2})\s*-+\s*(\d{2,4})", re.I)

_ORDER_DATE_RE = re.compile(
    r"dated\s*:?\s*(\d{1,2})\s*[.\-/]\s*(\d{1,2})\s*[.\-/]\s*(\d{2,4})", re.I)

# Address-ish opener: tells the name/address split where the postal part begins.
_ADDR_LEAD_RE = re.compile(
    r"^(?:village|vill|vpo|v\.?p\.?o|p\.?o|post|tehsil|teh|distt|dist|district|"
    r"near|opp|opposite|behind|sector|sec|plot|khasra|nh|gt\s+road|"
    r"\d+\s*(?:km|k\.?m\.?)|\d+(?:st|nd|rd|th)?\s+(?:milestone|mile)|c/o|"
    r"back\s+side|main\s+road|bypass)\b", re.I)

# A first segment that names a FACULTY rather than the institution: the brand is
# in the next segment ("Faculty of Hotel Management, Galaxy Global Educational
# Trust's Group of Institutions, ..."). Extending one segment recovers it.
_GENERIC_HEAD_RE = re.compile(
    r"^(?:faculty|school|department|dept|centre|center|institute|college)\s+of\s+"
    r"[a-z&.\s]+$", re.I)


def _clean_name(seg: str) -> str:
    """Drop parentheticals and trailing punctuation from a name segment.

    Parentheticals on this file are qualifiers and aliases, never identity:
    "(Technical Campus)", "(SKITM)", "(124507)", "(formerly R.N. College...)",
    "(Civil and Mech Engg)". Left in, their words have to be present in the
    catalogue name for token containment to pass, which refuses good rows.
    """
    seg = re.sub(r"\([^)]*\)", " ", seg)
    seg = re.sub(r"\s+", " ", seg).strip(" .,-;:&")
    return seg


def split_name(institute: str) -> str:
    """The institution name out of `institute`, which is name + postal address."""
    parts = [p.strip() for p in str(institute or "").split(",") if p.strip()]
    if not parts:
        return ""
    name = _clean_name(parts[0])
    # Only extend when the head is a bare "<something> of <something>" faculty
    # phrase AND the next segment is not already the address.
    if (len(parts) > 1 and _GENERIC_HEAD_RE.match(name)
            and not _ADDR_LEAD_RE.match(parts[1])):
        nxt = _clean_name(parts[1])
        if nxt:
            name = f"{name}, {nxt}"
    return name


def session_of(remarks: str) -> tuple[str | None, str]:
    """(academic session 'YYYY-YY', why-not) stated by the source itself."""
    text = str(remarks or "")
    if not text.strip():
        return None, "no remarks, so no academic session is stated"
    found: set[str] = set()
    for m in _SESSION_RE.finditer(text):
        start, tail = int(m.group(1)), m.group(2)
        end = int(tail) if len(tail) == 4 else 2000 + int(tail[-2:])
        # A session is one academic year. "2026-29" is a fee band, not a
        # session, and would misdate everything under it.
        if end != start + 1:
            continue
        found.add(f"{start}-{str(end)[-2:]}")
    if not found:
        return None, "no academic session stated (order date alone cannot fix one)"
    if len(found) > 1:
        return None, (f"remarks name {len(found)} sessions {sorted(found)} — "
                      f"cannot tell which one this fee is for")
    return found.pop(), ""


def order_date_of(remarks: str) -> str | None:
    """The order date, as provenance only. Never used to derive a year."""
    m = _ORDER_DATE_RE.search(str(remarks or ""))
    if not m:
        return None
    d, mth, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if y < 100:
        y += 2000
    if not (1 <= d <= 31 and 1 <= mth <= 12 and 2000 <= y <= 2100):
        return None
    return f"{y:04d}-{mth:02d}-{d:02d}"


# ---------------------------------------------------------------------------
# Place text
# ---------------------------------------------------------------------------

_ALIAS_SUBS = sorted(HR_PLACE_ALIASES.items(), key=lambda kv: -len(kv[0]))


def place_text(s: str) -> str:
    """Address text with AFRC spellings rewritten to the catalogue's.

    naac._city_from_address finds a city by looking for catalogue city keys as
    LITERAL substrings of the address, so the alias map passed to match_one does
    nothing for it — aliases are applied to structured city columns, not to free
    text. Measured consequence on this file: the catalogue key is "yamunanagar"
    and every AFRC address writes "Yamuna Nagar", so none of them resolved, and
    "L.N Institute of Technology, Madhogarh, distt Mahendergarh" (key is
    "mahendragarh") fell to the no-city path and matched "L.R. Institute of
    Technology" in Palwal — a different college in a different district.

    Only the hand-checked Haryana map is applied here, not the nationwide
    place_alias table: a nationwide alias fired against a free-text address is
    how a village name becomes some other state's city.
    """
    txt = re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()
    for alias, canonical in _ALIAS_SUBS:
        txt = re.sub(rf"\b{re.escape(alias)}\b", canonical, txt)
    return txt


# ---------------------------------------------------------------------------
# Extra refusal gates, layered ON TOP of registries.match_one
# ---------------------------------------------------------------------------
# match_one does the matching; nothing below can make it accept a pair it
# rejected, only refuse one it accepted. Each gate is here because the dry run
# was hand-checked and produced the wrong match quoted with it.

def brand(name: str) -> str:
    """The leading brand token, with initials joined: "M.R. DAV" -> "mrdav".

    Single letters are kept even when they are stop words. "D.A.V College"
    normalises to "d a v college", and dropping "a" as a stop word joined the
    rest into "dv" — which then read as a different brand from the catalogue's
    "dav" and vetoed a correct match.
    """
    words = [w for w in _norm(name).split() if len(w) == 1 or w not in _STOP]
    joined: list[str] = []
    for w in words:
        if len(w) == 1 and joined and len(joined[-1]) <= 3 and joined[-1].isalpha():
            joined[-1] += w          # D.A.V -> dav, M.R.D.A.V -> mrdav
        else:
            joined.append(w)
    return joined[0] if joined else ""


def _is_acronym_of(short: str, full_name: str) -> bool:
    """Is `short` built from the initials of `full_name`, in order?

    Exact equality against registries._acronym is too rigid: it takes the first
    letter of EVERY word, so "Seth Jai Parkash Mukand Lal Institute of
    Engineering & Technology" yields "sjpmliet" while the catalogue writes the
    real-world acronym "JMIT". Requiring a SUBSEQUENCE of those initials accepts
    the restatement and still refuses a different brand: "dvm" and "dbm" are not
    subsequences of B(M) C(ollege) o(f) P(harmacy)'s initials.
    """
    if len(short) < 2 or not short.isalpha():
        return False
    initials = [w[0] for w in _norm(full_name).split() if w]
    it = iter(initials)
    return all(ch in it for ch in short)


def _brand_ok(entry_name: str, cat_name: str) -> bool:
    """Refuse when two SHORT leading brands disagree.

    registries' leading-brand gate only fires on a catalogue head longer than 3
    characters, so a three-letter brand walks straight past it. Measured
    mis-matches, all of which passed every existing gate:

        BM College of Pharmacy (Gurgaon)   -> DVM College of Pharmacy, Gurugram
        BM College of Pharmacy (Gurgaon)   -> DBM College of Pharmacy, Sonipat
        L.N Institute of Technology        -> L.R. Institute of Technology

    "BM" is two characters, so _sig drops it entirely and containment compares
    {college, pharmacy} against {college, pharmacy} — a perfect score on a
    different college. Prefix containment is allowed because the two sides
    legitimately write the same initials at different lengths ("M.R. D.A.V." ->
    mrdav vs "MR DAV" -> mr).
    """
    a, b = brand(entry_name), brand(cat_name)
    if not a or not b or a == b:
        return True
    # The catalogue routinely prefixes the college's OWN acronym ("PIET -
    # Panipat Institute of Engineering and Technology", "JMIT - Seth Jai
    # Parkash Mukand Lal Institute of..."). That is the same name restated, not
    # a rival brand — the same allowance registries.match_one makes in its
    # leading-brand gate.
    if _is_acronym_of(b, entry_name) or _is_acronym_of(a, cat_name):
        return True
    if len(a) > 4 and len(b) > 4:
        return True                  # two real words; existing gates own this
    return a.startswith(b) or b.startswith(a)


_WOMEN_RE = re.compile(r"\b(?:girls?|womens?|woman|mahila|kanya|balika|ladies)\b")
_MEN_RE = re.compile(r"\b(?:boys?|mens?)\b")


def _gender(name: str) -> str | None:
    n = _norm(name)
    if _WOMEN_RE.search(n):
        return "women"
    if _MEN_RE.search(n):
        return "men"
    return None


def _gender_ok(entry_name: str, cat_name: str) -> bool:
    """A boys' college is not the girls' college of the same trust.

    Measured: "Shah Satnam Ji Boys College, Sirsa" matched "Shah Satnam Ji Girls
    College" in Sirsa at 0.87. The trust runs both, the catalogue holds only the
    girls' one, and every gate passed because "boys" is four characters and
    containment tolerates one short missing token. Only an EXPLICIT conflict
    refuses — a catalogue name that simply omits the marker is not evidence.
    """
    a, b = _gender(entry_name), _gender(cat_name)
    return a is None or b is None or a == b


def _head_ok(entry_name: str, cat_name: str) -> bool:
    """The AFRC name's own leading word must survive into the candidate.

    registries checks the CATALOGUE head against the registry name but not the
    reverse, and containment forgives one missing token of four characters or
    fewer. Together those let the leading brand vanish:

        Seth Jai Parkash Mukand Lal Institute of Engineering & Technology, Radaur
          -> Jai Parkash Mukand Lal Innovative Engineering and Technology Institute

    Both are real colleges in Radaur, and the catalogue holds the right one
    ("JMIT - Seth Jai Parkash Mukand Lal Institute of Engineering and
    Technology"). The only difference is the dropped word "Seth", which is
    exactly the brand. Short heads are skipped here because _sig drops them;
    _brand_ok covers those.
    """
    words = [w for w in _norm(entry_name).split() if w not in _STOP]
    if not words:
        return True
    head = words[0]
    if len(head) <= 3:
        return True
    if head in _sig(cat_name):
        return True
    # An acronym the catalogue spells with dots is still the same word: "RKSD
    # College" against "R.K.S.D. (PG) College" tokenises to {college} and loses
    # "rksd" entirely. Compare against the letters-only form, which is how
    # registries.match_one runs its own dotted-acronym check.
    return head in re.sub(r"[^a-z]", "", _norm(cat_name))


def agrees_both_ways(entry: dict, by_city, by_city_rev, by_token, by_token_rev,
                     aliases) -> tuple[dict | None, float, str]:
    """match_one run over the same candidates in both orders; refuse if they differ.

    match_one keeps the FIRST candidate at the best score (`if score >
    best_score`), so a tie is settled by whatever order the candidate list
    happens to be in — and that order is not stable. Two sources of drift were
    measured on this file: Postgres reorders rows after a write (the dry run
    matched 298 and the write run that followed matched 295), and match_one's
    no-city fallback iterates a SET of name words, whose order changes with
    Python's per-process hash seed (two consecutive dry runs then gave 298 and
    306).

    Rather than reach into the shared matcher, the same candidates are offered
    in reverse and the two answers compared. A name that really identifies one
    college gives the same answer either way. A name that gives two different
    answers is a tie between distinct colleges, which is the one-row form of the
    contention rule: refuse, do not let an arbitrary order pick the fee.
    """
    a, sa, wa = match_one(entry, by_city, by_token, aliases)
    b, _, _ = match_one(entry, by_city_rev, by_token_rev, aliases)
    ida = a["college_id"] if a else None
    idb = b["college_id"] if b else None
    if ida != idb:
        return None, sa, ("tie: the same name matches "
                          f"{(a or {}).get('name', '-')!r} and "
                          f"{(b or {}).get('name', '-')!r} equally well")
    return a, sa, wa


def verify(entry_name: str, cat_name: str) -> str:
    """"" if the pair survives every extra gate, else the reason it did not."""
    if not _head_ok(entry_name, cat_name):
        return f"AFRC name leads with {brand(entry_name)!r}, absent from the candidate"
    if not _brand_ok(entry_name, cat_name):
        return (f"leading brand {brand(entry_name)!r} != {brand(cat_name)!r} "
                f"— different institutions sharing a generic tail")
    if not _gender_ok(entry_name, cat_name):
        return (f"{_gender(entry_name)}'s college matched the "
                f"{_gender(cat_name)}'s college of the same trust")
    return ""


def load_rows(path: Path = FEE_JSON) -> list[dict]:
    if not path.exists():
        print(f"[fra-hr] missing input {path}", file=sys.stderr)
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [r for r in data if isinstance(r, dict)]


# ---------------------------------------------------------------------------
# Row gates
# ---------------------------------------------------------------------------

def vet(row: dict) -> tuple[dict | None, str]:
    """(usable record, refusal reason). Every gate here drops rather than guesses."""
    remarks = str(row.get("remarks") or "")

    # 1. CLOSED FIRST. A closed institution's fee is the worst row this file
    #    could write, and 2 of the 121 closure rows phrase the closure with the
    #    word "academic year", so a year parse would read them as live.
    if _CLOSED_RE.search(remarks):
        return None, f"institution is closed/withdrawn: {remarks[:90]}"

    tuition = _int(row.get("tuition"))
    development = _int(row.get("development")) or 0
    total = _int(row.get("total"))

    if tuition is None:
        return None, f"no usable approved tuition (raw {row.get('tuition')!r})"

    # 2. PARSE CHECK. tuition + development == total on 1,248/1,255 rows; the 7
    #    failures are the Hotel Management column shift, where `total` is really
    #    "Training & Other Fee". Publishing that as a total would understate the
    #    bill a student has to plan for.
    if total is None or tuition + development != total:
        return None, (f"column shift: tuition {tuition} + development "
                      f"{development} != total {total}")

    # 3. THE YEAR. Non-negotiable; see the module docstring.
    session, why = session_of(remarks)
    if not session:
        return None, why

    return {
        "name": split_name(row.get("institute")),
        "institute_raw": " ".join(str(row.get("institute") or "").split()),
        "course": str(row.get("course") or "").strip(),
        "sr": str(row.get("sr") or "").strip(),
        "tuition": tuition,
        "development": development or None,
        "total": total,
        "session": session,
        "order_date": order_date_of(remarks),
        "order_url": str(row.get("order_url") or "").strip() or None,
        "remarks": " ".join(remarks.split())[:400],
    }, ""


# ---------------------------------------------------------------------------
# Catalogue side
# ---------------------------------------------------------------------------

def _load_haryana() -> list[dict]:
    """Haryana colleges, in a STABLE order.

    ORDER BY is not cosmetic here. match_one keeps the first candidate at the
    best score (`if score > best_score`), so when two colleges tie the winner is
    whichever the database happened to return first — and Postgres reorders rows
    freely after a write. Measured: the dry run matched 298 rows and the write
    run that followed it matched 295, because three ties flipped to a candidate
    the extra gates then vetoed. Same input, different answer, which makes a
    hand-check meaningless.
    """
    return get_backend().q(
        "SELECT college_id, name, city, district, state FROM colleges "
        "WHERE NOT is_abroad AND state ILIKE ? ORDER BY college_id", [STATE])


def _now():
    return datetime.now(timezone.utc)


def run(*, dry_run: bool = False, limit: int | None = None) -> dict:
    ensure_tables()
    raw = load_rows()
    if limit:
        raw = raw[:limit]
    if not raw:
        return {}

    stats = {"rows": len(raw), "closed": 0, "shifted": 0, "no_year": 0,
             "no_fee": 0, "matched": 0, "unmatched": 0, "vetoed": 0, "tied": 0,
             "contested_rows": 0, "contested_keys": 0, "duplicate_rows": 0}
    misses: list[tuple] = []

    records: list[dict] = []
    for r in raw:
        rec, why = vet(r)
        ref = f"{str(r.get('course') or '?')}#{r.get('sr') or '?'}"
        if rec is None:
            if why.startswith("institution is closed"):
                stats["closed"] += 1
            elif why.startswith("column shift"):
                stats["shifted"] += 1
            elif "tuition" in why and "column shift" not in why:
                stats["no_fee"] += 1
            else:
                stats["no_year"] += 1
            misses.append((REGISTRY, ref, str(r.get("course") or ""), "unknown",
                           split_name(r.get("institute")) or None, None, STATE,
                           None, why[:400]))
            continue
        records.append(rec)

    colleges = _load_haryana()
    aliases = dict(_load_place_aliases())
    aliases.update(HR_PLACE_ALIASES)
    by_city = _index_by_city(colleges, aliases)
    by_token = _index_by_token(colleges)
    by_city_rev = {k: list(reversed(v)) for k, v in by_city.items()}
    by_token_rev = {k: list(reversed(v)) for k, v in by_token.items()}
    # CITY keys only — _city_from_address must not be allowed to resolve to a
    # district, whose bucket is every college in it. See naac.py.
    #
    # Sorted, not a set: _city_from_address scans these with a longest-match
    # rule that keeps the first key at the winning length, so two keys of equal
    # length would otherwise be separated by set iteration order, which moves
    # with the hash seed.
    place_keys = sorted({k for k in (_place_key(c.get("city"), aliases)
                                     for c in colleges) if k})
    state_keys = {k for k in (_place_key(c.get("state"), aliases)
                              for c in colleges) if k}

    print(f"[fra-hr] {len(raw):,} chart rows -> {len(records):,} dated, "
          f"live, parse-clean rows, matched against {len(colleges):,} Haryana "
          f"colleges ({len(place_keys):,} city keys)", file=sys.stderr)

    # (college_id, course, session) -> the rows claiming it.
    claims: dict[tuple, list] = {}
    no_city = 0

    for rec in records:
        city = _city_from_address(place_text(rec["institute_raw"]), place_keys,
                                  place_text(rec["name"]), state_keys)
        if not city:
            no_city += 1
        college, score, why = agrees_both_ways(
            {"name": rec["name"], "city": city},
            by_city, by_city_rev, by_token, by_token_rev, aliases)
        ref = f"{rec['course']}#{rec['sr']}"
        if why.startswith("tie:"):
            stats["tied"] += 1
        if college is not None:
            bad = verify(rec["name"], college["name"])
            if bad:
                stats["vetoed"] += 1
                college, why = None, f"vetoed: {bad}"
        if college is None:
            stats["unmatched"] += 1
            misses.append((REGISTRY, ref, rec["course"], rec["session"],
                           rec["name"], city or None, STATE,
                           round(score, 3), why[:400]))
            continue
        claims.setdefault(
            (college["college_id"], rec["course"], rec["session"]), []
        ).append((rec, college, score, why, city))

    # ---- resolve contention by REFUSING ------------------------------------
    # Two chart rows landing on the same college for the same course and session
    # means at least one is attached to the wrong college. The higher match score
    # is not a safe tie-break — that was measured wrong on real pairs — so a
    # disagreeing pair yields NO fee.
    #
    # The one exception is when the contending rows carry the IDENTICAL figures:
    # then whichever row is the mis-match, the number written is the same, so
    # there is nothing to get wrong. Those are collapsed to one fact and counted
    # separately rather than silently.
    writes: list[tuple] = []
    samples: list[str] = []
    now = _now()

    for (cid, course, session), group in sorted(claims.items()):
        figures = {(g[0]["tuition"], g[0]["development"], g[0]["total"])
                   for g in group}
        if len(group) > 1 and len(figures) > 1:
            stats["contested_keys"] += 1
            stats["contested_rows"] += len(group)
            for rec, college, score, why, city in group:
                misses.append((
                    REGISTRY, f"contested:{rec['course']}#{rec['sr']}",
                    rec["course"], rec["session"], rec["name"], city or None,
                    STATE, round(score, 3),
                    f"{len(group)} chart rows claim {college['name'][:60]!r} for "
                    f"{course} {session} with different fees "
                    f"{sorted(figures)} — refused rather than guessed"[:400]))
            continue
        if len(group) > 1:
            stats["duplicate_rows"] += len(group) - 1

        rec, college, score, why = group[0][0], group[0][1], group[0][2], group[0][3]
        city = group[0][4]
        payload = {
            # Key names are the contract fee_promote.py and store/db.py read.
            "approved_tuition_inr": rec["tuition"],
            "approved_total_inr": rec["total"],
            "development_fees_inr": rec["development"],
            "stream": rec["course"],
            "order_date": rec["order_date"],
            "district": (city or "").title() or None,
            "academic_year": rec["session"],
            "year_source": "session stated in the AFRC order remarks",
            "authority": "Admission & Fee Regulating Committee, Haryana",
            "authority_name": rec["institute_raw"][:300],
            "chart_as_on": "2026-07-29",
            "remarks": rec["remarks"],
        }
        writes.append((college["college_id"], REGISTRY,
                       f"{rec['course']}#{rec['sr']}", rec["course"],
                       rec["session"], json.dumps(payload, ensure_ascii=False),
                       SOURCE_URL, rec["order_url"], round(score, 3),
                       why[:400], now))
        stats["matched"] += 1
        if len(samples) < 20:
            samples.append(
                f"  Rs {rec['tuition']:>7,}/yr (Rs {rec['total']:>7,} incl dev) "
                f"{rec['session']}  {rec['course'][:22]:24} "
                f"{rec['name'][:40]:42} -> {college['name'][:38]:40} "
                f"[{city or '-'}] ({score:.2f})")

    print("\n".join(samples), file=sys.stderr)
    print(f"\n[fra-hr] dropped before matching: {stats['closed']:,} closed, "
          f"{stats['shifted']:,} column-shifted, {stats['no_year']:,} with no "
          f"stated session, {stats['no_fee']:,} with no usable fee",
          file=sys.stderr)
    print(f"[fra-hr] matched {stats['matched']:,} facts, "
          f"{stats['unmatched']:,} refused by the matcher "
          f"({stats['vetoed']:,} vetoed by the extra gates, "
          f"{stats['tied']:,} tied between two colleges), "
          f"{no_city:,} had no recoverable city", file=sys.stderr)
    if stats["contested_keys"]:
        print(f"[fra-hr] REFUSED {stats['contested_keys']:,} college/course/year "
              f"keys claimed by {stats['contested_rows']:,} disagreeing chart "
              f"rows", file=sys.stderr)
    if stats["duplicate_rows"]:
        print(f"[fra-hr] collapsed {stats['duplicate_rows']:,} duplicate rows "
              f"that agreed on every figure", file=sys.stderr)
    print(f"[fra-hr] {len({w[0] for w in writes}):,} distinct colleges would "
          f"gain an approved fee"
          + ("   (DRY RUN — nothing written)" if dry_run else ""),
          file=sys.stderr)

    if dry_run:
        stats["colleges"] = len({w[0] for w in writes})
        return stats

    be = get_backend()
    # Clean replace, like naac.py: an upsert alone would strand facts that this
    # run now REFUSES (a newly-contested key would keep the coin-flip fee a
    # previous run wrote), which is the exact failure the refusal exists to stop.
    removed = be.execute("DELETE FROM registry_fact WHERE registry = ?", [REGISTRY])
    if removed:
        print(f"[fra-hr] cleared {removed:,} facts from the previous run",
              file=sys.stderr)
    be.executemany(
        "INSERT INTO registry_fact (college_id, registry, ref_id, category, "
        "year, payload, source_url, detail_url, match_score, match_note, "
        "fetched_at) VALUES (?,?,?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT (college_id, registry, category, year) DO UPDATE SET "
        "payload = EXCLUDED.payload, match_score = EXCLUDED.match_score, "
        "match_note = EXCLUDED.match_note, fetched_at = EXCLUDED.fetched_at",
        writes)
    try:
        be.execute("DELETE FROM registry_unmatched WHERE registry = ?", [REGISTRY])
        be.executemany(
            "INSERT INTO registry_unmatched (registry, ref_id, category, year, "
            "name, city, state, best_score, note) VALUES (?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT (registry, ref_id, category, year) DO UPDATE SET "
            "best_score = EXCLUDED.best_score, note = EXCLUDED.note", misses)
    except Exception as exc:  # noqa: BLE001 - the misses log is diagnostic only
        print(f"[fra-hr] could not record misses ({str(exc)[:80]})", file=sys.stderr)

    n = be.one("SELECT COUNT(*) FROM registry_fact WHERE registry = ?",
               [REGISTRY]) or 0
    c = be.one("SELECT COUNT(DISTINCT college_id) FROM registry_fact "
               "WHERE registry = ?", [REGISTRY]) or 0
    print(f"[fra-hr] stored {n:,} approved-fee facts across {c:,} colleges",
          file=sys.stderr)
    stats["stored"], stats["colleges"] = n, c
    return stats


def report() -> None:
    be = get_backend()
    n = be.one("SELECT COUNT(*) FROM registry_fact WHERE registry = ?",
               [REGISTRY]) or 0
    c = be.one("SELECT COUNT(DISTINCT college_id) FROM registry_fact "
               "WHERE registry = ?", [REGISTRY]) or 0
    print(f"-- Haryana AFRC --\n   {n:,} approved-fee facts across {c:,} colleges")
    if not n:
        return
    for r in be.q(
            "SELECT category, year, COUNT(*) n, "
            "ROUND(AVG((payload->>'approved_tuition_inr')::numeric)) avg_fee "
            "FROM registry_fact WHERE registry = ? "
            "GROUP BY category, year ORDER BY n DESC LIMIT 15", [REGISTRY]):
        print(f"   {str(r['category'])[:34]:36} {r['year']:8} {r['n']:>4}  "
              f"avg approved tuition Rs {int(r['avg_fee'] or 0):,}")
    gained = be.one("""SELECT COUNT(DISTINCT f.college_id) FROM registry_fact f
                       LEFT JOIN college_agg a USING(college_id)
                       WHERE f.registry = ? AND a.min_tuition_inr IS NULL""",
                    [REGISTRY]) or 0
    print(f"\n   colleges gaining a fee they had NONE for: {gained:,}")
    print("\n-- refused (leads, not losses) --")
    for r in be.q("SELECT COUNT(*) n FROM registry_unmatched WHERE registry = ?",
                  [REGISTRY]):
        print(f"   {r['n']:,} rows logged with the reason they were refused")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="match and report, write nothing")
    ap.add_argument("--limit", type=int, help="only the first N chart rows")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
        return
    run(dry_run=a.dry_run, limit=a.limit)


if __name__ == "__main__":
    main()
