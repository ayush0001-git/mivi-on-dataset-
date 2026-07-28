"""Deterministic extraction of admission cutoffs, closing ranks and minimum
eligibility marks from a college's own web page. No LLM, no network.

WHY THIS IS MOSTLY A REFUSAL ENGINE
I fetched and parsed ~25 real `cutoff`-classified pages (JIIT, IPU, SVCE, DU
admissions, Chitkara, SRCC Gobichettipalayam, DAV Chandigarh, Loyola, Stella
Maris, Rajagiri, Mar Athanasius, BN College Patna, Sharda, Nirma, Vidya
Academy) before writing a single pattern. What the sweep actually hands this
function is, in order of measured frequency:

  1. Student-counselling-cell and M.Sc-Counselling-Psychology pages. The
     `cutoff` classifier in web_enrich matches on "counselling", and on small
     college sites that keyword finds a mental-health service far more often
     than it finds admission counselling. This is the largest single source of
     false positives, so "counselling" is deliberately NOT an extraction
     anchor anywhere in this module.
  2. Link lists whose numbers live in a linked PDF or JPG. Every Delhi
     University category-wise cut-off rank is a PDF; BN College Patna
     publishes merit lists as .jpg. The HTML holds a title and nothing else.
     SVCE's page is literally headed "Cut Off Marks in TNEA 2025" and the only
     numbers on it are a phone number, a PIN code and a counselling code.
  3. Collapsed HTML tables. parse_html() emits a newline for <tr> but not for
     <td>, so a real JIIT cutoff row arrives as
         JIIT62B.TechCSE129488091094.6799.83
     which is 1294|880910|94.67|99.83 or 12948|80910|94.679|9.83 — genuinely
     ambiguous. Guessing a split is how a student gets told a wrong cutoff, so
     digit-soup and dense tabular lines are skipped wholesale rather than
     mined. The LLM fallback can have those pages.
  4. Prose statements of minimum eligibility ("a minimum of 50% aggregate
     marks"). This is the one shape that parses reliably, and it is the bulk
     of what this module actually returns.

CUTOFF vs ELIGIBILITY
These live in separate fact keys and are never merged. A minimum eligibility
mark is a floor for *applying*; a cutoff is the mark the last admitted
candidate actually had. Telling a student who clears the floor that they clear
the cutoff is exactly the harm this pipeline exists to avoid, so
`min_eligibility_*` and `cutoff_*` are distinct keys and the summary says
which is which in words. Ranks and percentages are likewise never mixed: a
rank only ever lands in a `*_rank` key.

Contract: extract(text, url, college_name) -> {"summary","facts","confidence"}
or None. Never raises.
"""
from __future__ import annotations

import re

# Past this it is navigation, footers and boilerplate. The real content on
# every page measured sat well inside it, and the cap bounds the worst case on
# the 3 MB pages web_enrich still lets through.
MAX_CHARS = 40_000
MAX_LINES = 1_200
MAX_FACTS = 14

# Below this the LLM fallback should get the page instead. Returning a
# sub-threshold dict would be worse than returning None: the caller discards
# the facts AND skips the fallback, so the page yields nothing at all.
MIN_CONFIDENCE = 0.55


# ---------------------------------------------------------------------------
# Indian number parsing
# ---------------------------------------------------------------------------

_LAKH = re.compile(r"^(\d+(?:\.\d+)?)\s*(?:lakhs?|lacs?|l)$", re.I)
_CRORE = re.compile(r"^(\d+(?:\.\d+)?)\s*(?:crores?|cr)$", re.I)


def _to_number(raw: str) -> float | None:
    """Parse Indian numeric notation. int('1,20,000') raises; this does not.

    Handles lakh grouping (2,2,3 from the right) as well as western 3,3
    grouping, and the "1.2 Lakh" / "1.2L" shorthand Indian sites use
    interchangeably with digits. Used only to sanity-check a value — the
    string stored as a fact is always the page's own wording.
    """
    if not raw:
        return None
    s = re.sub(r"\s+", " ", raw.strip().replace(" ", " ")).strip()
    m = _LAKH.match(s)
    if m:
        try:
            return float(m.group(1)) * 100_000
        except ValueError:
            return None
    m = _CRORE.match(s)
    if m:
        try:
            return float(m.group(1)) * 10_000_000
        except ValueError:
            return None
    s = s.replace(",", "").replace(" ", "")
    if not re.fullmatch(r"\d+(?:\.\d+)?", s):
        return None
    try:
        return float(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Line-level gates
# ---------------------------------------------------------------------------

# 9+ unbroken digit/dot characters. That is what a collapsed <td> run looks
# like ("129488091094.6799.83") and also what a mobile number looks like
# ("9840861987"); neither may reach the value patterns.
_SOUP = re.compile(r"\d[\d.]{8,}")

_HAS_DIGIT = re.compile(r"\d")


def _looks_tabular(line: str) -> bool:
    """A collapsed table row, whose column boundaries are unrecoverable.

    Measured: JIIT's real cut-off row "JIIT62B.TechBIO-115686-129809--" is 45%
    digits and has no sentence structure; the prose eligibility statements on
    Chitkara and SRCC measured 2-6% digits. The threshold has a wide margin on
    both sides, and a legitimate one-line category list
    ("General 96.5%, OBC 94.25%, SC 90%") measures ~20%.
    """
    if len(line) < 25:
        return False
    digits = sum(1 for ch in line if ch.isdigit())
    return digits / len(line) > 0.35


# Context that disqualifies a number outright. Every entry here was observed
# producing a plausible-looking wrong answer on a page I actually fetched:
#   "85% seats are reserved for UT pool"            DAV Chandigarh
#   "27 % (17) seats are reserved for OBC category" SRCC Gobichettipalayam
#   "50% weight-age is given to OCET exam"          DAV Chandigarh
#   "financial support up to 100% of the tuition"   Nirma
#   "65% to 100%" above "Placement program wise"    Sharda
#   "Vidya ranks 29th in KIRF 2025"                 Vidya Academy
#   "Counseling Code - 1219", "Mobile: 9840861987"  SVCE cut-off page
#   "round 1 cut off dated 30.06.2026"              IPU admission notices
_FORBID_SRC = r"""
      seats?\b[^.\n]{0,30}?(?:reserv|earmark|supernumerar|allot|availab)
    | (?:reserv|earmark|supernumerar)\w*[^.\n]{0,30}?\bseats?\b
    | \bquota\b | supernumerar | weight\s*-?\s*age
    | attendance
    | \bfees?\b | tuition | scholarship | waiver | concession | stipend
    | refund | discount | \bloan\b | fellowship
    | placement | placed | package | salary | \bctc\b | recruit | hiring
    | nirf | naac | \bnba\b | \bkirf\b | ranking | ranked
    | \branks?\s+\d{1,3}\w{0,2}\s+(?:in|among)
    | phone | mobile | contact | helpline | landline | \btel\b | \bfax\b
    | whatsapp | e-?mail
    | \bcodes?\b | \bpin\b | \bext\b | \bdated?\b | \bregd?\. | roll\s*no
    | \bgst\b | interest | growth | increase | hike | turnover
"""
_FORBID = re.compile(_FORBID_SRC + r"| \brelax | \bgrace\b", re.I | re.X)
# The relaxation rule is *about* relaxation, so it cannot forbid the word.
_FORBID_NO_RELAX = re.compile(_FORBID_SRC, re.I | re.X)


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

_PCT = r"(?P<val>\d{1,3}(?:\.\d{1,3})?)\s*(?:%|per\s?cent(?:age)?\b|percent(?:age)?\b)"
_PCTILE_POST = r"(?P<val>\d{1,3}(?:\.\d{1,4})?)\s*(?:%\s*ile\b|percentile\b)"
_PCTILE_PRE = (r"(?:percentile|%\s*ile)\s*(?:of\s+|:\s*|is\s+|was\s+|-\s*)?"
               r"(?P<val>\d{1,3}(?:\.\d{1,4})?)")
# Two digits minimum, or comma-grouped. "closing rank 7" is not a thing on
# these pages, and bare single digits are overwhelmingly list numbering.
_RANK_NUM = r"(?P<val>\d{1,3}(?:,\d{2,3})+|\d{2,7})"

_CUT_ANCHOR = re.compile(
    r"cut[\s\-]?off|closing\s+rank|last\s+rank|opening\s+rank|"
    r"closing\s+(?:score|marks?|percent)|merit\s+cut|shortlist|"
    r"minimum\s+allocation\s+score|last\s+admitted",
    re.I,
)

_ELIG_ANCHOR = re.compile(
    r"eligib|qualifying\s+exam|aggregate|10\s*\+\s*2|class\s*(?:xii|12)\b|"
    r"\bxii\b|intermediate|higher\s+secondary|\bhsc\b|senior\s+secondary|"
    r"bachelor|graduation|degree\s+exam|marks\b",
    re.I,
)

_CATEGORY = re.compile(
    r"\b(General|Gen|UR|Un-?reserved|Open|OBC(?:-NCL)?|SC|ST|EWS|PwD|PH|"
    r"NRI|Management|Minority|Defence|Sports)\b"
)

_EXAM = re.compile(
    r"\b(NEET(?:\s*UG|\s*PG)?|JEE\s*(?:Main|Advanced)?|CUET(?:-UG|-PG)?|CAT|"
    r"XAT|MAT|CMAT|GATE|CLAT|TNEA|KEAM|EAMCET|MHT[\s-]?CET|CET|WBJEE|COMEDK|"
    r"BITSAT|VITEEE|SRMJEEE|LPUNEST|KIITEE|NATA|GPAT|NIFT|UCEED)\b",
    re.I,
)
_EXAM_NAMES = (r"NEET|JEE|CUET|CAT|GATE|CLAT|TNEA|KEAM|EAMCET|MHT[\s-]?CET|"
               r"CET|WBJEE|COMEDK|BITSAT|VITEEE|KIITEE|NATA|GPAT")


# ---------------------------------------------------------------------------
# Rules: (key, pattern, requires, kind, weight)
# ---------------------------------------------------------------------------
# `requires` is matched against a +/-90 character window, which on the real
# pages is the enclosing clause. Every quantifier is explicitly bounded and
# none is nested, so no input can trigger catastrophic backtracking.

_RULES: tuple[tuple[str, re.Pattern[str], re.Pattern[str] | None, str, float], ...] = (
    # --- minimum eligibility: a floor to APPLY, never a cutoff -------------
    (
        "min_eligibility",
        re.compile(
            r"(?:minimum|min\.|at\s*least|atleast|not\s+less\s+than|"
            r"no\s+less\s+than)[^.\n]{0,25}?" + _PCT,
            re.I,
        ),
        _ELIG_ANCHOR,
        "pct",
        0.72,
    ),
    (
        "min_eligibility",
        re.compile(_PCT + r"[^.\n]{0,30}?(?:or\s+above|or\s+more|and\s+above)", re.I),
        _ELIG_ANCHOR,
        "pct",
        0.62,
    ),
    (
        "min_eligibility",
        re.compile(
            r"(?:secur\w+|obtain\w+|scor\w+|pass\w+)\s+(?:a\s+)?"
            r"(?:minimum\s+(?:of\s+)?)?" + _PCT,
            re.I,
        ),
        _ELIG_ANCHOR,
        "pct",
        0.60,
    ),
    (
        "eligibility_relaxation",
        re.compile(r"(?:relaxation|relaxed|concession)\s+(?:of\s+)?" + _PCT, re.I),
        re.compile(r"marks|aggregate|categor|\bSC\b|\bST\b|OBC|PwD", re.I),
        "pct_relax",
        0.60,
    ),
    # --- actual cutoffs ---------------------------------------------------
    (
        "cutoff_percentage",
        re.compile(
            r"(?:cut[\s\-]?off|closing|last\s+admitted|merit)[^.\n]{0,40}?" + _PCT,
            re.I,
        ),
        None,
        "pct",
        0.80,
    ),
    (
        "cutoff_percentage",
        re.compile(_PCT + r"[^.\n]{0,25}?cut[\s\-]?off", re.I),
        None,
        "pct",
        0.78,
    ),
    (
        "cutoff_percentile",
        re.compile(
            r"(?:cut[\s\-]?off|closing|shortlist\w*|minimum|required)"
            r"[^.\n]{0,40}?" + _PCTILE_POST,
            re.I,
        ),
        None,
        "pctile",
        0.78,
    ),
    (
        "cutoff_percentile",
        re.compile(_PCTILE_PRE, re.I),
        _CUT_ANCHOR,
        "pctile",
        0.74,
    ),
    (
        "closing_rank",
        re.compile(
            r"(?:closing|last|cut[\s\-]?off|final)\s+rank[^.\n]{0,40}?" + _RANK_NUM,
            re.I,
        ),
        None,
        "rank",
        0.82,
    ),
    (
        "closing_rank",
        re.compile(
            r"rank\s+(?:of\s+|upto\s+|up\s+to\s+)?" + _RANK_NUM
            + r"[^.\n]{0,25}?(?:admitted|allotted|closed|cut[\s\-]?off)",
            re.I,
        ),
        None,
        "rank",
        0.72,
    ),
    (
        "opening_rank",
        re.compile(r"opening\s+rank[^.\n]{0,40}?" + _RANK_NUM, re.I),
        None,
        "rank",
        0.78,
    ),
    # --- entrance-exam thresholds -----------------------------------------
    (
        "exam_cutoff_marks",
        re.compile(
            r"(?:" + _EXAM_NAMES + r")[^.\n]{0,30}?(?:score|marks?)\s*"
            r"(?:of\s+|above\s+|at\s*least\s+|is\s+|was\s+|:\s*|-\s*)"
            r"(?P<val>\d{2,4})",
            re.I,
        ),
        _CUT_ANCHOR,
        "marks",
        0.72,
    ),
    (
        "exam_cutoff_marks",
        re.compile(r"(?P<val>\d{2,4})\s*(?:marks|score)\b", re.I),
        re.compile(r"(?:" + _EXAM_NAMES + r")", re.I),
        "marks",
        0.66,
    ),
    (
        "exam_cutoff_rank",
        re.compile(r"(?:\bAIR\b|all\s+india\s+rank)[^.\n]{0,25}?" + _RANK_NUM, re.I),
        _CUT_ANCHOR,
        "rank",
        0.70,
    ),
)

# The second exam-marks rule is loose on its own, so it additionally requires a
# cutoff anchor in the window; encoded here rather than in the tuple because
# `requires` already carries the exam-name check.
_NEEDS_CUT_ANCHOR = {"exam_cutoff_marks"}

# Sanity bounds per value kind. A percentage above 100, a rank of 0 or a NEET
# score of 2025 is a parsing accident, not a fact. The marks ceiling of 800 is
# what rejects a year that slipped into a score pattern.
_BOUNDS = {
    "pct": (1.0, 100.0),
    "pct_relax": (1.0, 25.0),
    "pctile": (1.0, 100.0),
    "rank": (10.0, 2_000_000.0),
    "marks": (10.0, 800.0),
}

_SESSION = re.compile(
    r"(?:academic\s+(?:year|session)|\bA\.?Y\.?\b|\bA\.?S\.?\b|session|batch)"
    r"\s*:?\s*(20\d{2}\s*[-–]\s*\d{2,4})",
    re.I,
)
_SESSION_BARE = re.compile(r"\b(20[1-3]\d\s*[-–]\s*\d{2,4})\b")
_YEAR_ONLY = re.compile(r"\b(20[1-3]\d)\b")
_ROUND = re.compile(r"\bround\s*[-–:]?\s*(\d{1,2}|I{1,3}V?|IV|VI{0,3})\b", re.I)


# ---------------------------------------------------------------------------
# Sentence handling
# ---------------------------------------------------------------------------

# Abbreviations that end in a period mid-sentence on Indian college pages.
# Splitting "M.Sc. Physics" or "Tk. - 602 117" into sentences would cut the
# quoted statement in half.
_ABBREV = {
    "sc", "no", "dr", "mr", "mrs", "ms", "tk", "ext", "st", "dist", "govt",
    "univ", "approx", "viz", "etc", "fig", "vol", "dept", "engg", "tech",
    "edn", "hrs", "rs", "ph", "jr", "sr", "prof", "hons", "com", "a", "b",
    "e", "d", "m", "p", "u", "s", "i", "ii", "iii", "iv",
}
_BOUNDARY = re.compile(r"([A-Za-z0-9)\]%]+)\.\s+(?=[A-Z(])")


def _boundaries(line: str) -> list[int]:
    cuts = [0]
    for m in _BOUNDARY.finditer(line):
        if m.group(1).lower().rstrip(".") not in _ABBREV:
            cuts.append(m.end())
    cuts.append(len(line))
    return cuts


def _sentence(line: str, start: int, end: int, cuts: list[int]) -> str:
    """The page's own sentence around a match, verbatim.

    Quoting the source sentence rather than rebuilding one guarantees nothing
    is ever paraphrased into a stronger claim than the page made. Boundaries
    are computed once per line by the caller: recomputing them per match made
    a single pathological 40k-character line cost 220 ms.
    """
    lo, hi = 0, len(line)
    for i in range(len(cuts) - 1):
        if cuts[i] <= start < cuts[i + 1]:
            lo, hi = cuts[i], cuts[i + 1]
            break
    if hi <= end:                       # match straddles a boundary
        hi = min(len(line), end + 60)
    frag = re.sub(r"\s+", " ", line[lo:hi]).strip(" ,;:-–—")
    if len(frag) > 220:                 # keep a card-sized quote
        rel = start - lo
        a = max(0, rel - 90)
        frag = re.sub(r"\s+", " ", line[lo + a: lo + a + 220]).strip(" ,;:-–—")
    return frag


def _category_near(window: str, at: int) -> str | None:
    """The category label governing a number, if one is close enough.

    Only the nearest label within 30 characters counts; on a line listing four
    categories a looser radius attaches the wrong one.
    """
    best, best_d = None, 31
    for m in _CATEGORY.finditer(window):
        d = at - m.end() if m.end() <= at else m.start() - at
        if 0 <= d < best_d:
            best, best_d = m.group(1), d
    return best


def _verbatim(raw: str, kind: str, line: str, val_end: int) -> str:
    """The amount exactly as the page wrote it, carrying its own unit.

    `val_end` is the end of the value group, not of the whole match, so the
    unit that follows the digits is still in view ("610 marks", "50 per cent").
    """
    tail = line[val_end:val_end + 14]
    if kind in ("pct", "pct_relax"):
        m = re.match(r"\s*(%|per\s?cent(?:age)?|percent(?:age)?)", tail, re.I)
        unit = m.group(1).strip() if m else "%"
        return f"{raw}{unit}" if unit == "%" else f"{raw} {unit}"
    if kind == "pctile":
        return f"{raw} percentile"
    if kind == "marks":
        return f"{raw} marks" if re.match(r"\s*marks?\b", tail, re.I) else raw
    return raw


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def _collect(lines: list[str]) -> list[dict]:
    found: list[dict] = []
    for line in lines:
        if not _HAS_DIGIT.search(line):
            continue
        if _SOUP.search(line) or _looks_tabular(line):
            continue
        has_cut = bool(_CUT_ANCHOR.search(line))
        for key, pat, requires, kind, weight in _RULES:
            for m in pat.finditer(line):
                raw = m.group("val")
                num = _to_number(raw)
                if num is None:
                    continue
                lo, hi = _BOUNDS[kind]
                if not (lo <= num <= hi):
                    continue
                s, e = m.span()
                wlo = max(0, s - 90)
                window = line[wlo: e + 90]
                forbid = _FORBID_NO_RELAX if kind == "pct_relax" else _FORBID
                if forbid.search(window):
                    continue
                if requires is not None and not requires.search(window):
                    continue
                if key in _NEEDS_CUT_ANCHOR and not (
                        has_cut or _CUT_ANCHOR.search(window)):
                    continue
                found.append({
                    "key": key,
                    "kind": kind,
                    "value": _verbatim(raw, kind, line, e),
                    "num": num,
                    "weight": weight,
                    "statement": _sentence(line, s, e),
                    "category": _category_near(window, s - wlo),
                    "exam": (_EXAM.search(window).group(1).upper().replace(" ", "")
                             if _EXAM.search(window) else None),
                    "line": line,
                })
                if len(found) >= 80:        # bail out of pathological pages
                    return found
        # A single line that lists several categories against a cutoff is the
        # one tabular shape that survives parse_html intact, e.g.
        # "First cut-off list: General 96.5%, OBC 94.25%, SC 90%". The rules
        # above only capture the first of those, so sweep the rest here.
        if has_cut:
            for m in re.finditer(_PCT, line):
                s, e = m.span()
                wlo = max(0, s - 60)
                window = line[wlo: e + 60]
                if _FORBID.search(window):
                    continue
                cat = _category_near(window, s - wlo)
                if not cat:
                    continue
                num = _to_number(m.group("val"))
                if num is None or not 1.0 <= num <= 100.0:
                    continue
                found.append({
                    "key": "cutoff_percentage",
                    "kind": "pct",
                    "value": _verbatim(m.group("val"), "pct", line, e),
                    "num": num,
                    "weight": 0.80,
                    "statement": _sentence(line, s, e),
                    "category": cat,
                    "exam": None,
                    "line": line,
                })
    return found


def _context(lines: list[str], hit_lines: set[str]) -> dict[str, str]:
    """Session and round labels, taken only from a line that produced a fact
    or from a line that names a cutoff. A year lifted from anywhere on the
    page would as often be a copyright notice as an admission session."""
    out: dict[str, str] = {}
    for line in lines:
        near = line in hit_lines
        cut = bool(_CUT_ANCHOR.search(line))
        if not (near or cut):
            continue
        if "session" not in out:
            m = _SESSION.search(line) or (_SESSION_BARE.search(line) if cut else None)
            if m:
                out["session"] = re.sub(r"\s+", "", m.group(1))
            elif cut:
                y = _YEAR_ONLY.search(line)
                if y:
                    out["session"] = y.group(1)
        if "round" not in out and cut:
            m = _ROUND.search(line)
            if m:
                out["round"] = f"Round {m.group(1).upper()}"
        if len(out) == 2:
            break
    return out


_CUT_PREFIX = ("cutoff", "closing_rank", "opening_rank", "exam_cutoff")


def _summarise(college: str, shorts: list[tuple[str, str]],
               session: str | None) -> str:
    """<=45 words of plain prose. Says 'minimum eligibility' in words whenever
    that is all the page gave, so a card can never read as a cutoff."""
    name = re.sub(r"\s+", " ", (college or "").strip())[:45] or "This college"
    cuts = [t for k, t in shorts if k.startswith(_CUT_PREFIX)]
    elig = [t for k, t in shorts if k.startswith("min_eligibility")]
    relax = [t for k, t in shorts if k.startswith("eligibility_relaxation")]
    parts: list[str] = []
    if cuts:
        parts.append("cut-offs listed " + ", ".join(dict.fromkeys(cuts))[:150])
    if elig:
        parts.append("a stated minimum eligibility of "
                     + ", ".join(dict.fromkeys(elig))[:80])
    if relax:
        parts.append("relaxation of " + ", ".join(dict.fromkeys(relax))[:40]
                     + " for reserved categories")
    if not parts:
        return ""
    text = f"{name}: this page shows " + "; ".join(parts) + "."
    if session:
        text = text[:-1] + f" ({session})."
    if not cuts:
        text += " These are minimum eligibility marks, not admission cutoffs."
    return " ".join(re.sub(r"\s+", " ", text).split(" ")[:45])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def extract(text: str, url: str, college_name: str) -> dict | None:
    """Cutoffs / closing ranks / minimum eligibility from one page's text.

    Returns None whenever the page states no concrete, unambiguous figure —
    which on the pages measured is the overwhelming majority of them.
    """
    try:
        return _extract(text, url, college_name)
    except Exception:  # noqa: BLE001 - one broken page must never kill a batch
        return None


def _extract(text: str, url: str, college_name: str) -> dict | None:
    if not isinstance(text, str) or not text:
        return None
    body = text[:MAX_CHARS]
    if len(body.strip()) < 60:
        return None

    lines = [ln.strip() for ln in body.split("\n")]
    lines = [ln for ln in lines if ln][:MAX_LINES]
    if not lines:
        return None

    hits = _collect(lines)
    if not hits:
        return None

    # Most explicit evidence first, so the payload keeps the best statement of
    # a value and the confidence reflects the strongest rule that fired.
    hits.sort(key=lambda h: -h["weight"])
    facts: dict[str, str] = {}
    shorts: list[tuple[str, str]] = []
    statements: list[str] = []
    hit_lines: set[str] = set()
    seen: set[tuple[str, str, str]] = set()
    top = 0.0

    for h in hits:
        sig = (h["key"], h["category"] or "", h["value"])
        if sig in seen:
            continue
        seen.add(sig)
        key = h["key"]
        if h["category"]:
            key = f"{key}_{h['category'].lower()}"
        elif h["exam"] and key.startswith("exam_"):
            key = f"{key}_{h['exam'].lower()}"
        base, n = key, 2
        while key in facts:
            key = f"{base}_{n}"
            n += 1
        facts[key] = h["value"]
        label = (f"{h['category']} {h['value']}" if h["category"] else h["value"])
        if h["key"] in ("closing_rank", "opening_rank", "exam_cutoff_rank"):
            label = f"{h['key'].replace('_', ' ')} {h['value']}"
        shorts.append((h["key"], label))
        if h["statement"] and h["statement"] not in statements:
            statements.append(h["statement"])
        hit_lines.add(h["line"])
        top = max(top, h["weight"])
        if len(facts) >= MAX_FACTS:
            break

    ctx = _context(lines, hit_lines)
    facts.update(ctx)
    # The verbatim source sentences, so a reviewer can audit any card without
    # refetching the page. Capped at two: this is evidence, not the payload.
    for i, st in enumerate(statements[:2], start=1):
        facts["statement" if i == 1 else f"statement_{i}"] = st

    confidence = top
    values = [k for k in facts
              if k not in ("session", "round", "statement", "statement_2")]
    if len(values) >= 3:
        confidence += 0.06          # a real table or criteria block, not a stray line
    if ctx:
        confidence += 0.03
    if re.search(r"cut[\-_]?off|merit[\-_]?list|closing[\-_]?rank", url or "", re.I):
        confidence += 0.05          # the site itself filed the page under cutoffs
    confidence = round(min(confidence, 0.95), 2)
    if confidence < MIN_CONFIDENCE:
        return None

    summary = _summarise(college_name, shorts, ctx.get("session"))
    if not summary:
        return None
    return {"summary": summary[:600], "facts": facts, "confidence": confidence}


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import json
    import os
    import sys
    import time

    def show(label: str, out: dict | None) -> None:
        if out is None:
            print(f"  None      | {label}")
        else:
            print(f"  conf {out['confidence']:<5}| {label}")
            print(f"      summary: {out['summary']}")
            print(f"      facts:   {json.dumps(out['facts'], ensure_ascii=False)}")

    print("=" * 74)
    print("1. Indian number parser")
    print("=" * 74)
    bad_parse = 0
    for raw, want in [("1,20,000", 120000.0), ("12,345", 12345.0),
                      ("1.2 Lakh", 120000.0), ("1.2L", 120000.0),
                      ("45000", 45000.0), ("96.5", 96.5), ("2 Cr", 20000000.0),
                      ("", None), ("abc", None), ("12-34", None)]:
        got = _to_number(raw)
        ok = got == want
        bad_parse += not ok
        print(f"  {'ok ' if ok else 'FAIL'} {raw!r:<12} -> {got!r} (want {want!r})")

    print()
    print("=" * 74)
    print("2. Adversarial pages - every one MUST be None")
    print("=" * 74)
    NEG = [
        ("empty", ""),
        ("whitespace only", "   \n\n  \t "),
        ("nav only", "Home\nAbout Us\nAdmissions\nContact\nAlumni\nLibrary\n"
                     "Research\nPlacement\nAlumni Login\nGallery\nCircular"),
        # verbatim from https://www.svce.ac.in/admission/cut-off-marks/
        ("SVCE cut-off page: numbers are phone / PIN / counselling code",
         "Cut Off Marks in TNEA 2025\n"
         "Minimum cutoff Marks and Maximum Rank allotted through TNEA\n"
         "View the complete Report\nCounseling Code - 1219\n"
         "Admission Coordinator: Dr. N.R. Sheela\nMobile: 9840861987\n"
         "Landline: 044-27152000 Ext: 123/126\n"
         "Sriperumbudur (off Chennai) Tk. - 602 117\nPhone No: 044-27152000"),
        # verbatim from https://www.ipu.ac.in/adm2026/adm2026notices.php
        ("IPU notices: 'cut off' beside course codes and dates",
         "BBA ROUND 01 CUT OFF DATED 02.06.2026 A.S. 2026-27\n"
         "B.A (economics) (code-197) round 1 cut off dated 30.06.2026\n"
         "B.com (h) (code-146) round 1 cut off dated 30.06.2026\n"
         "LE-B.TECH FOR DIPLOMA HOLDERS (CODE-128)_CUT OFF IN ALLOTMENT IN "
         "ROUND 01 FOR 2026-27. dated- 19.06.2026"),
        # verbatim from https://www.davchd.ac.in/admission
        ("DAV Chandigarh: weightage % and reserved-seat %",
         "However, for admission in M.Sc. Physics, M.Sc. Chemistry, M.Sc. "
         "Zoology and M.Sc. Bio-technology 50% weight-age is given to OCET "
         "exam conducted by Panjab University.\n"
         "85% seats are reserved for UT pool."),
        # verbatim from https://www.srccgbo.edu.in/Admissions.php
        ("SRCC Gobi: reservation percentages",
         "The total number of seats available are 67 including reservation. "
         "27 % (17) seats are reserved for OBC category, 15 % (9) for SC "
         "category, 7.5 % (5) for ST category and 3 % (2-Supernumerary) for "
         "Persons with Disability (PwD) 5% Foreign National (3 Supernumerary "
         "seats) category candidates."),
        # verbatim from https://www.nirmauni.ac.in/admissions/
        ("Nirma: fee-waiver % and supernumerary-seat %",
         "Every year the university extends financial support up to 100% of "
         "the tuition fee to meritorious students in the form of scholarships "
         "based on merit.\nA total 15% of supernumerary seats in each "
         "programme is available for admission to international students, out "
         "of which 5% seats are earmarked for CIWGC-SEA category, while 10% "
         "seats are for the OCI/FN category."),
        # verbatim from https://www.vidyaacademy.ac.in/
        ("Vidya: institutional ranking + a student's rank",
         "Vidya ranks 29th in KIRF 2025 among Kerala engineering colleges and "
         "16th among self-financing colleges in Kerala\n"
         "Reshma P R secures KTU first rank in MCA with a CGPA of 9.72\n"
         "Vidya leaps towards sustainability with a 235 kW Solar Plant "
         "meeting over 85% of demand"),
        # verbatim from https://www.sharda.ac.in/admission/eligibility-criteria
        ("Sharda: placement % on an /eligibility-criteria URL",
         "Top companies are hiring from sharda\n65% to 100% \n"
         "Placement program wise\n34000+\nSuccessful Alumni worldwide"),
        # verbatim from macollege.ac.in/Student-Support/Student-Counselling-Cell
        ("student counselling cell (matched 'counselling', wrong topic)",
         "Student Counselling Cell\nCounselling is a process that aims to "
         "facilitate physical and mental well-being of the students. The "
         "Guidance and Counselling Cell of the College provides guidance to "
         "the students by helping them build potential within themselves to "
         "combat the distress. The Cell provides individual as well as group "
         "counselling to students as and when needed. 2 counsellors are "
         "available on all 6 working days."),
        # verbatim from jiit.ac.in/prospective-student/admission/cut-off-merit
        ("JIIT: real cut-off table, columns collapsed by parse_html",
         "Cut-Off Merit : AY 2026-27 (Updated on 23 July)\n"
         "CampusProgramBranchJEE AIR10+2\nMINMAXMINMAX\n"
         "JIIT62B.TechCSE129488091094.6799.83\n"
         "JIIT128B.TechCSE4985816368988.6796.33\n"
         "JIIT62B.TechIT565049260694.0097.00\n"
         "JIIT62B.TechBIO-115686-129809--\n"
         "Note: Above is the summary of cut-offs after all rounds."),
        ("different topic: hostel charges",
         "Hostel Charges 2026-27\nSingle occupancy room Rs 1,20,000 per "
         "annum. Double occupancy Rs 85,000 per annum including mess charges "
         "of Rs 45,000 p.a. Caution money Rs 10,000 refundable."),
        ("different topic: address block with PIN and affiliation number",
         "Sri Venkateswara College of Engineering, Post Bag No.1, Pennalur "
         "Village, Sriperumbudur Tk. - 602 117, Tamil Nadu, India. "
         "Phone No: 044-27152000. Estd. 1985. Affiliation No. 1219."),
        ("attendance rule, not a cutoff",
         "Students must maintain a minimum of 75% attendance in each course "
         "to be permitted to appear for the end semester examination."),
        ("marketing claim with a number",
         "We are proud to be among the best colleges in the region with 100% "
         "placement assistance and the highest package of Rs 45 LPA."),
        ("cutoff page that only links to a PDF",
         "Category-wise Cut-off Ranks\nFirst Round - B.Tech. (2026-27) - "
         "Category-wise Cut-off Ranks\nSecond Round - B.Tech. (2026-27) - "
         "Category-wise Cut-off Ranks\nMinimum Allocation Score for Round II "
         "of UG CSAS\nDownload PDF"),
    ]
    neg_fp = 0
    for label, txt in NEG:
        out = extract(txt, "https://example.ac.in/admission/cut-off-marks/",
                      "Test College")
        neg_fp += out is not None
        show(label, out)

    print()
    print("=" * 74)
    print("3. Pages that DO state something concrete")
    print("=" * 74)
    POS = [
        # verbatim from Chitkara (a *counselling* URL that really does state a
        # minimum eligibility)
        ("REAL Chitkara M.Sc: eligibility prose",
         "Eligibility Criteria\nCandidates must possess any one of the "
         "following qualifications from a recognised university with a "
         "minimum of 50% aggregate marks.",
         "https://www.chitkara.edu.in/healthscience/m-sc-reproductive-health-"
         "fertility-counselling/", "Chitkara University"),
        # verbatim from SRCC Gobichettipalayam
        ("REAL SRCC Gobi: eligibility + category relaxation",
         "The eligibility for admission is minimum 50% marks in the degree "
         "examination (10+2+3) of a recognized University. Candidates who are "
         "appearing for their qualifying examination are also eligible to "
         "apply. A relaxation of 5% in total aggregate marks is admissible to "
         "candidates belonging to reserved categories of OBC,SC, ST and PwD.",
         "https://www.srccgbo.edu.in/Admissions.php", "SRCC Gobichettipalayam"),
        # shapes seen in the wild whose numbers lived in a PDF, written the way
        # a site that keeps them in HTML writes them
        ("category-wise cut-off percentages on one line",
         "B.Com (Hons.) Admission 2026-27\nFirst cut-off list: General 96.5%, "
         "OBC 94.25%, SC 90%, ST 88.75%, EWS 95%.",
         "https://example.ac.in/admissions/cut-off-list", "Example College"),
        ("closing and opening rank in prose",
         "Round 2 counselling result for B.Tech Computer Science and "
         "Engineering: the closing rank was 12,345 and the opening rank was "
         "8,970 for the academic session 2025-26.",
         "https://example.ac.in/closing-rank", "Example Institute"),
        ("lakh-grouped closing rank",
         "For B.Tech Civil Engineering the last rank admitted in the final "
         "round was 1,20,450 in the 2025 counselling.",
         "https://example.ac.in/admission/last-rank", "Example Institute"),
        ("NEET marks cutoff",
         "Admission to MBBS is through NEET UG only. The cut-off NEET score "
         "for the General category in the 2025 session was 610 marks.",
         "https://example.ac.in/mbbs-cutoff", "Example Medical College"),
        ("CAT percentile cutoff",
         "MBA admissions 2026-27: the shortlisting cut-off was a CAT "
         "percentile of 92 for the General category.",
         "https://example.ac.in/mba/cut-off", "Example B School"),
        ("minimum eligibility, class 12",
         "Eligibility: A candidate must have passed the 10+2 examination "
         "with at least 45% marks in aggregate. For SC/ST candidates the "
         "requirement is 40%.",
         "https://example.ac.in/eligibility", "Example College"),
    ]
    for label, txt, u, name in POS:
        show(label, extract(txt, u, name))

    print()
    print("=" * 74)
    print("4. Every real page captured (MIVI_EXTRACTOR_PAGES=<dir of .txt>)")
    print("=" * 74)
    pages_dir = os.getenv("MIVI_EXTRACTOR_PAGES")
    fired = total = 0
    if pages_dir and os.path.isdir(pages_dir):
        t0 = time.perf_counter()
        for fn in sorted(os.listdir(pages_dir)):
            if not fn.endswith(".txt"):
                continue
            with open(os.path.join(pages_dir, fn), encoding="utf-8",
                      errors="replace") as fh:
                raw = fh.read()
            first, _, rest = raw.partition("\n")
            src = first[4:].strip() if first.startswith("URL:") else fn
            rest = rest.split("=" * 70, 1)[-1]
            total += 1
            out = extract(rest, src, "Sample College")
            if out is not None:
                fired += 1
                show(src, out)
        dt = (time.perf_counter() - t0) * 1000
        print(f"\n  {fired}/{total} real pages produced a fact; "
              f"{dt:.0f} ms total, {dt / max(total, 1):.2f} ms/page")
    else:
        print("  (skipped: MIVI_EXTRACTOR_PAGES not set)")

    print()
    print("=" * 74)
    print("5. Speed and robustness")
    print("=" * 74)
    big = ("Minimum 50% marks in aggregate are required in class 12. "
           "Contact 9840861987. Cut-off 2025. " * 90)[:6000]
    t0 = time.perf_counter()
    for _ in range(200):
        extract(big, "https://x.ac.in/cut-off", "X College")
    print(f"  6,000-char page: {(time.perf_counter() - t0) * 5:.3f} ms/call")
    for name, bad in (
        ("20k digits + commas", "9" * 20000 + "%" + "," * 5000 + "cut off rank " * 2000),
        ("one 100k line", ("cut off 95% rank 12,345 " * 4000)),
        ("no newlines, all punctuation", "." * 50000),
        ("unicode soup", "कटऑफ 95% – " * 2000),
    ):
        t0 = time.perf_counter()
        extract(bad, "https://x.ac.in", "X")
        print(f"  {name:<30} {(time.perf_counter() - t0) * 1000:6.1f} ms, no hang")
    for bad in (None, 123, b"bytes", [], {"a": 1}, object()):
        assert extract(bad, "u", "n") is None, bad          # type: ignore[arg-type]
    print("  non-string input -> None, no exception")

    print()
    print(f"  parser failures:                  {bad_parse}")
    print(f"  false positives on adversarial:   {neg_fp}/{len(NEG)}")
    sys.exit(1 if (neg_fp or bad_parse) else 0)
