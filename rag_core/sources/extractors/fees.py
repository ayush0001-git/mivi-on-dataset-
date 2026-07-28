"""Deterministic fee / tuition extraction from a college's own fee page.

WHY THIS EXISTS
Every extraction the sweep makes is one call against a 14/min free-tier quota,
and fees are the single most-requested missing field (the catalogue has a fee on
58% of courses and nothing at all for hostel, mess or exam charges). Fee pages
are also the most machine-readable thing a college publishes: they are almost
always a table, and a table survives tag-stripping as a very recognisable line
pattern. So this parses them on local CPU and leaves the LLM for the pages that
genuinely need reading.

WHAT IT REFUSES TO DO (every one of these was a real page, most of them a bug
this module actually had before the page was tested against it)
- A number with no currency mark, unless it sits in a multi-column table whose
  own adjacent header declares amount columns. This is the load-bearing rule:
  it is what stops PIN codes ("Chandigarh 160 002"), phone numbers
  ("Admission Helpline9501105714"), visitor counters ("Users This Year: 426961"),
  statute numbers ("Tripura Act No. 4 of 2023"), amity.edu's marketing strip
  ("150,000 / Students") and jmi.ac.in's hostel bed capacities from becoming
  fees.
- Guessing what an unlabelled column means. A table whose header cannot be read
  is skipped entirely rather than labelled by position, and a row glued into one
  separator-less string ("B.Tech. Hons. CSE62,000/-96,000/-36,000/-…") is
  dropped because which column survived is unrecoverable.
- Bank details. The LLM fallback, on uimt.org's fee-payment page, emitted an
  account number and IFSC code as "fees". Those lines are filtered outright.
- Scholarship money. Award amounts are money and sit under headers like
  "Amount", but they are not prices — and they are another extractor's topic.
- Converting, annualising or summing. The period is whatever the page's own
  column header or trailing phrase says, copied verbatim beside the amount.

Self-test:  python -m rag_core.sources.extractors.fees
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Limits. The sweep hands us whatever parse_html() produced from a third-party
# page, so every loop here is bounded.
# ---------------------------------------------------------------------------
# Measured over 128 real college pages: the largest was 1,314 lines / 57 KB and
# the median 307 lines. These caps leave better than 2x headroom over anything
# real while bounding a pathological page — the sweep runs unattended over
# 10,000 third-party sites and must not stall on one of them.
MAX_TEXT = 120_000
MAX_LINES = 3_000
MAX_LINE = 2_000        # scan only the head of a pathological single-line blob
# A real fee table runs to 60+ programmes (tiut.ac.in publishes 65) and the
# whole point of this field is that a student can find THEIR course, so the cap
# is generous. ~90 bytes a row keeps the stored payload around 5 KB.
MAX_FACTS = 60
PROSE_SLOTS = 8         # reserved so a huge table cannot crowd out hostel/mess
MIN_CONFIDENCE = 0.5    # below this the caller discards it, so return None instead

# Indian grouping first: "1,20,000" must match whole, not just the leading "1".
_NUM = r"\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?|\d{1,9}(?:\.\d{1,2})?"

# `L` and `Cr` only directly after digits, so "LL.B." and "Cr." in prose cannot
# turn a nearby number into lakhs.
_UNIT = r"(?:\s{0,2}(?:lakhs?|lacs?|crores?)\b|(?<=\d)\s?(?:L|Cr)\b)"

# \b before Rs matters: without it re.I matches the "rs" inside "Courses".
_CUR_PRE = r"(?:\bRs\.?|\bINR\b|\bRupees\b|₹|₨|\bUS\$|\$|\bUSD\b|€|£|\bAED\b)"
# "only" is deliberately NOT a suffix marker: it would promote "in 2023 only"
# into an explicit amount.
_CUR_POST = r"(?:\bINR\b|₹|\bRs\.?|/\-|/=)"

_MONEY_RE = re.compile(
    r"(?P<pre>" + _CUR_PRE + r")?\s{0,3}"
    r"(?P<n1>" + _NUM + r")(?P<u1>" + _UNIT + r")?"
    # A range is one fact, not two: "Rs 45,000 - 1,20,000" stays whole.
    r"(?:\s{0,3}(?:-|–|—|to)\s{0,3}(?:" + _CUR_PRE + r")?\s{0,3}"
    r"(?P<n2>" + _NUM + r")(?P<u2>" + _UNIT + r")?)?"
    r"\s{0,3}(?P<post>" + _CUR_POST + r")?",
    re.I)

# Kept verbatim on the end of an amount so the basis is never lost or converted.
_PERIOD_RE = re.compile(
    r"\s{0,2}(?:per\s+(?:annum|year|yr|semester|sem|month|head|student|candidate|"
    r"course|paper|subject|installment|instalment)"
    r"|p\.\s?a\.?|/\s?(?:yr|year|sem|semester|month|annum)\b|annually|monthly"
    r"|one[\s-]?time|for\s+(?:the\s+)?(?:entire|whole|full)\s+course)", re.I)

# A line matching this is never a fee, whatever numbers it carries.
_BAD_LINE_RE = re.compile(
    r"\b(?:phone|telephone|mobile|fax|whats\s?app|whatsapp|helpline|toll[\s-]?free"
    r"|contact\s+(?:no|number)|pin\s?code|zip|postal|ifsc|micr|swift|a/c"
    r"|account\s+(?:no|number|name)|beneficiary|branch\s+code|gstin|udise"
    r"|visitors?|users?\b|hits\b|followers?"
    # The trailing "s?" matters: \bscholarship\b does not match "Scholarships",
    # which let a whole table of donor-named awards through as fees.
    r"|scholarships?|stipends?|fellowships?|freeships?|awards?|prizes?|packages?"
    r"|salary|ctc|lpa\b|donations?|endowments?|bursary|bursaries"
    r"|placements?|recruiters?|percentile|cut[\s-]?off|marks?\b|rank\b|seats?\b"
    r"|intake\b|copyright|all\s+rights\s+reserved)\b", re.I)

# A prose line must carry one of these to be read as a fee at all.
# "tution" is in here because that misspelling is genuinely common on Indian
# college sites and costs nothing to accept.
_FEE_WORD_RE = re.compile(
    r"\b(?:fees?|tuition|tution|charges?|chargeable|deposit|caution|instal?lments?"
    r"|hostel|mess|exam|examination|registration|admission|security|payable|pays?"
    r"|paid|payment|cost|dues|refundable|fine|library|transport|bus|convocation"
    r"|migration|development|enrol?ment|prospectus|application|semester|annum"
    r"|annual)\b", re.I)

# A table column header that declares an amount.
_AMOUNT_HDR_RE = re.compile(
    r"(?:\b(?:fees?|tuition|tution|amount|charges?|cost|deposit|instal?lment|total"
    r"|payable|rs|inr)\b|₹|\$)", re.I)

# Stricter: a header must say this before a column of BARE numbers is believed
# to be money. "Total" or "Instalment" alone is not enough.
_CURRENCY_HDR_RE = re.compile(
    r"(?:\b(?:fees?|tuition|tution|amount|charges?|cost|deposit|rs|inr)\b|₹|\$)", re.I)

# Generic table furniture: never a programme name on its own.
_GENERIC_LABEL = {
    "item", "items", "nature", "particulars", "particular", "description", "desc",
    "course", "courses", "programme", "program", "programs", "programmes",
    "duration", "amount", "remarks", "remark", "type", "category", "sl", "sl.",
    "sl no", "sl.no", "s.no", "sno", "sr no", "sr.no", "serial", "total", "year",
    "years", "semester", "semesters", "fee", "fees", "class", "name", "details",
    "sn", "no", "#", "-", "head", "heads", "subject", "subjects", "branch",
    "stream", "level", "session",
}

# Fee-nature descriptors that belong to a trailing table column and leak into
# the NEXT row's label block once the table is flattened to lines.
_NATURE_RE = re.compile(
    r"^(?:one[\s-]?time|per\s+(?:annum|year|semester|sem|month|head)|refundable"
    r"|non[\s-]?refundable|yearly|annually|payable\b|optional|as\s+applicable"
    r"|lump\s?sum|at\s+the\s+time|every\s+(?:year|semester))", re.I)

# Words that make a line a duration cell rather than a programme name. A line
# whose words are MOSTLY from this set is dropped from labels. The ratio (not
# "all") and the prefix rule together handle tiut.ac.in's ragged
# "(6 Semester+ 1" / "YearInternship/Training) semesters" split, where
# tag-stripping glued "Year" onto "Internship" into one token.
_DURATION_WORDS = {
    "year", "years", "yr", "yrs", "month", "months", "semester", "semesters",
    "sem", "sems", "internship", "training", "na", "nil", "and", "or", "to",
    "the", "of", "duration", "approx", "week", "weeks", "day", "days", "hour",
    "hours", "full", "part", "time",
}
_DURATION_PREFIX = ("year", "semester", "month", "week", "sem")
_DURATION_RATIO = 0.6

_WORD_RE = re.compile(r"[A-Za-z]{2,}")
_SPACE_RE = re.compile(r"\s+")
_PAGE_FEE_RE = re.compile(r"\b(?:fees?|tuition)\b", re.I)
_PAGE_SCHOLARSHIP_RE = re.compile(
    r"\b(?:scholarships?|stipends?|fellowships?|freeships?|bursary|bursaries"
    r"|grant-in-aid|merit-cum-means)\b", re.I)
_URL_FEE_RE = re.compile(r"(?:^|[/_\-.?=&])(?:fee|fees|fee-structure|feestructure"
                         r"|tuition|feestruct)(?:$|[/_\-.?=&])", re.I)

# An explicitly marked amount may be 0 ("0 INR" is a real cell on tiut.ac.in for
# certificate courses with no per-semester component). A BARE number must clear
# 100 to be believed at all.
_SANE_MIN_EXPLICIT = 0.0
_SANE_MIN_BARE = 100.0
_SANE_MAX = 200_000_000.0   # 20 crore; a 10-digit phone number is ~9.5e9

# "Not applicable" cells. They keep a flattened table row intact instead of
# splitting it in two, and they are worth emitting: "Total Course Fee: N/A INR"
# is what the page says.
_PLACEHOLDER_RE = re.compile(
    r"^(?:n\.?\s?/?\s?a\.?|nil|none|not\s+applicable|-{1,3}|—|–|\*)"
    r"(?:\s*(?:INR|Rs\.?|₹|\$))?$", re.I)


# ---------------------------------------------------------------------------
# Number handling
# ---------------------------------------------------------------------------

def _to_number(raw: str, unit: str | None) -> float | None:
    """Numeric value of an amount, for sanity checks ONLY.

    Nothing derived from this is ever emitted: the fact always carries the
    page's own text. This exists so a phone number cannot pass as a fee.
    """
    try:
        v = float(raw.replace(",", ""))
    except ValueError:
        return None
    if unit:
        u = unit.strip().lower()
        if u.startswith(("lakh", "lac", "l")):
            v *= 100_000
        elif u.startswith(("crore", "cr")):
            v *= 10_000_000
    return v


def _money_at(line: str, pos: int = 0):
    """First amount in `line` at/after `pos`.

    Returns (start, end, explicit, value) or None. `explicit` means the page
    itself marked it as currency (Rs / INR / ₹ / /- / lakh); a bare number is
    reported too, but callers may only trust it inside a declared amount column.
    """
    m = _MONEY_RE.search(line, pos)
    while m is not None:
        n1, u1 = m.group("n1"), m.group("u1")
        val = _to_number(n1, u1)
        if val is not None:
            explicit = bool(m.group("pre") or m.group("post") or u1 or m.group("u2"))
            # A range of adjacent years ("Session 2025-26") is not an amount.
            n2 = m.group("n2")
            year_range = (n2 is not None and not explicit
                          and _looks_like_year(n1) and len(n2.replace(",", "")) <= 4)
            floor = _SANE_MIN_EXPLICIT if explicit else _SANE_MIN_BARE
            if not year_range and floor <= val <= _SANE_MAX:
                return m.start(), m.end(), explicit, val
        nxt = m.end() if m.end() > m.start() else m.start() + 1
        if nxt >= len(line):
            return None
        m = _MONEY_RE.search(line, nxt)
    return None


def _looks_like_year(raw: str) -> bool:
    s = raw.replace(",", "")
    return len(s) == 4 and s.isdigit() and 1900 <= int(s) <= 2099


def _verbatim(line: str, start: int, end: int) -> str:
    """The amount exactly as printed, plus anything that qualifies it.

    An immediately following parenthetical or period phrase is part of the fact
    — "Rs. 5990/- (Including Refundable Hostel Security of Rs. 350/-)" loses its
    meaning if trimmed to the number.
    """
    out_end = end
    # Up to three, because "per head per semester" (tiut.ac.in bus fee) is two
    # phrases and dropping the second changes what the number means.
    for _ in range(3):
        p = _PERIOD_RE.match(line, out_end)
        if not p or p.end() == out_end:
            break
        out_end = p.end()
    tail = line[out_end:out_end + 200]
    stripped = tail.lstrip()
    if stripped.startswith("("):
        off = len(tail) - len(stripped)
        depth = 0
        for i, ch in enumerate(tail[off:off + 170]):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    out_end = out_end + off + i + 1
                    break
    return _SPACE_RE.sub(" ", line[start:out_end]).strip(" .,;:")


# ---------------------------------------------------------------------------
# Line classification
# ---------------------------------------------------------------------------

def _strip_cell(line: str) -> str:
    # Leading bullets go; a trailing "-" must NOT, it is half of "12,000/-".
    return line.strip().lstrip("*•·-–— \t").rstrip("*• \t.")


def _money_only(line: str):
    """(text, explicit) if the whole line is one amount, else None."""
    s = _strip_cell(line)
    if not s or len(s) > 60 or not any(c.isdigit() for c in s):
        return None
    m = _MONEY_RE.match(s)
    if m is None:
        return None
    # The cell may carry its own basis — "Rs 1,20,000 per annum",
    # "12,000/- per semester" — which is extremely common and must stay part of
    # the cell, not be mistaken for the next row's label. Nothing else may
    # follow: "4 years" and "3-5 years" still have to read as text.
    end = m.end()
    for _ in range(3):
        p = _PERIOD_RE.match(s, end)
        if not p or p.end() == end:
            break
        end = p.end()
    if end != len(s):
        return None
    val = _to_number(m.group("n1"), m.group("u1"))
    if val is None:
        return None
    explicit = bool(m.group("pre") or m.group("post") or m.group("u1") or m.group("u2"))
    floor = _SANE_MIN_EXPLICIT if explicit else _SANE_MIN_BARE
    if not (floor <= val <= _SANE_MAX):
        return None
    # A bare 4-digit 1900-2099 in a table cell is a session year far more often
    # than a fee. Losing a genuine "Rs 2000" is the cheaper mistake, and that
    # one still survives because its "Rs" makes it explicit.
    if not explicit and _looks_like_year(m.group("n1")):
        return None
    return s, explicit


def _is_placeholder(line: str) -> bool:
    return bool(_PLACEHOLDER_RE.match(line.strip()))


def _label_words(line: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(line)]


def _is_duration_cell(line: str) -> bool:
    words = _label_words(line)
    if not words:
        return False
    dur = sum(1 for w in words if w in _DURATION_WORDS or w.startswith(_DURATION_PREFIX))
    return dur / len(words) >= _DURATION_RATIO


_NOTE_PREFIX_RE = re.compile(r"^(?:note|n\.?b\.?|nb|important|please\s+note)\s*[:.\-–—]\s*",
                             re.I)


def _clean_label(lines: list[str]) -> str | None:
    """Row label from the text lines that preceded the amounts.

    A flattened table row reads "<school> / <programme>" or
    "<programme> / <degree>", so the last two usable lines are the useful ones;
    anything earlier is leftover header or the previous row's trailing column.
    The two are only joined when the result stays readable — otherwise the more
    specific tail line alone beats an 80-character truncation.
    """
    keep: list[str] = []
    for raw in lines:
        s = _SPACE_RE.sub(" ", raw).strip(" \t|:;-–—")
        s = _NOTE_PREFIX_RE.sub("", s)
        if not s or len(s) > 120:
            continue
        low = s.lower().strip(" .:")
        if low in _GENERIC_LABEL or _NATURE_RE.match(s) or _is_duration_cell(s):
            continue
        if _BAD_LINE_RE.search(s) or "http" in low or "@" in s:
            continue
        # An amount is never a programme name. Without this, a cell the run
        # scanner declined ("0 INR") becomes a fact key.
        if _money_only(s) is not None or _is_placeholder(s):
            continue
        words = _WORD_RE.findall(s)
        if not words or all(w.lower() in _CUR_TOKENS for w in words):
            continue
        keep.append(s)
    if not keep:
        return None
    tail = keep[-1].strip()
    if len(keep) >= 2:
        joined = _SPACE_RE.sub(" ", f"{keep[-2]} {tail}").strip()
        label = joined if len(joined) <= 80 else (tail if len(tail) >= 8 else joined[:80])
    else:
        label = tail
    label = _SPACE_RE.sub(" ", label).strip(" \t|:;-–—,")
    if len(label) < 3 or not _WORD_RE.search(label):
        return None
    if len(label) > 90:                       # cut on a word, never mid-token
        label = label[:90].rsplit(" ", 1)[0]
    return label.strip(" \t|:;-–—,")


# ---------------------------------------------------------------------------
# Shape B: column table flattened to one cell per line
# ---------------------------------------------------------------------------

_CUR_TOKENS = {"rs", "inr", "usd", "aed", "rupees", "only", "na", "nil", "n"}

# A header must sit within this many lines of the row it labels. amity.edu's
# admission page put "Fee related queries (1st Sem):" — a phone-number caption —
# 975 lines above a strip of marketing counters (150,000 Students, 6000
# Faculties). Unbounded header search adopted it and published the counters as
# fees. Proximity is what makes a header a header.
HEADER_WINDOW = 16

# Header-shaped text that is really a contact prompt or a link.
_HDR_NEGATIVE_RE = re.compile(
    r"\b(?:quer(?:y|ies)|enquir|inquir|contact|helpdesk|help\s?desk|support|call"
    r"|email|mail|apply|click|login|log\s?in|register|link|form|download|brochure"
    r"|read\s+more|know\s+more|view|faq)\b", re.I)


def _is_amount_header(line: str) -> bool:
    """A column header that names an amount.

    The descriptive-word requirement is the important half: without it the cell
    "N/A INR" (a real tiut.ac.in table cell) matches on "INR" and gets adopted
    as a column name, which then labels a genuine amount with nonsense.
    """
    s = line.strip()
    if not s or len(s) > 60 or _BAD_LINE_RE.search(s) or not _AMOUNT_HDR_RE.search(s):
        return False
    if _HDR_NEGATIVE_RE.search(s) or _money_only(s) is not None or _is_placeholder(s):
        return False
    # A column header is a noun phrase, never a sentence. Prose such as "Hostel
    # Fee and other fees are notified separately." sits happily inside the
    # header window and will otherwise be adopted as a column name — which then
    # labels a table of hostel bed capacities as fees.
    words = s.split()
    if len(words) > 8 or (s.endswith(".") and len(words) >= 5):
        return False
    return any(w.lower() not in _CUR_TOKENS for w in _WORD_RE.findall(s))


def _find_header(block: list[str], k: int) -> tuple[list[str], int, int] | None:
    """The k consecutive lines that name this table's amount columns.

    Returns (header, index_after_header, headers_seen_in_window). Only the tail
    of the block is searched — see HEADER_WINDOW — and the search runs backwards
    so the real header beats the "Fee Structure" nav link every one of these
    pages repeats twice.
    """
    base = max(0, len(block) - HEADER_WINDOW)
    window = block[base:]
    idx = [i for i, l in enumerate(window) if _is_amount_header(l)]
    for p in range(len(idx) - k, -1, -1):
        first, last = idx[p], idx[p + k - 1]
        if last - first != k - 1:
            continue
        if last >= len(window) - 1:     # a header needs at least one row after it
            continue
        header = [_SPACE_RE.sub(" ", window[i]).strip(" :|") for i in range(first, last + 1)]
        return header, base + last + 1, len(idx)
    return None


def _tables(lines: list[str]) -> list[dict]:
    """Group flattened table rows: runs of text lines followed by k amount lines."""
    kinds: list[tuple[str, str, bool]] = []
    for raw in lines:
        head = raw[:MAX_LINE]
        mo = _money_only(head)
        if mo is not None and not _BAD_LINE_RE.search(head):
            kinds.append(("M", mo[0], mo[1]))
        elif _is_placeholder(head):
            kinds.append(("P", head.strip(), True))
        else:
            kinds.append(("T", raw, False))

    rows, i, prev_end, n = [], 0, 0, len(kinds)
    while i < n:
        if kinds[i][0] == "M":
            # A run may contain "N/A" cells but must be mostly real amounts,
            # so a stray dash after a table cannot inflate a row's width.
            j = i
            while j < n and kinds[j][0] in ("M", "P") and (j - i) < 8:
                j += 1
            span = list(range(i, j))
            if sum(1 for t in span if kinds[t][0] == "P") > len(span) // 2:
                while span and kinds[span[-1]][0] == "P":
                    span.pop()
                j = span[-1] + 1
            rows.append({"span": (prev_end, i),
                         "money": [kinds[t][1] for t in span],
                         "explicit": all(kinds[t][2] for t in span)})
            prev_end = j
            i = j
        else:
            i += 1

    groups, cur = [], []
    for r in rows:
        gap = r["span"][1] - r["span"][0]
        if cur and len(cur[-1]["money"]) == len(r["money"]) and gap <= 10:
            cur.append(r)
        else:
            if len(cur) >= 2:
                groups.append(cur)
            cur = [r]
    if len(cur) >= 2:
        groups.append(cur)
    return groups


def _facts_from_tables(lines: list[str]) -> tuple[dict[str, str], int, bool]:
    facts: dict[str, str] = {}
    best_rows, headed = 0, False
    for group in _tables(lines):
        k = len(group[0]["money"])
        a, b = group[0]["span"]
        block = lines[a:b]
        found = _find_header(block, k) if block else None
        if found is None:
            # No readable header. Guessing what column 2 of 3 means is exactly
            # the kind of invention this module exists to avoid.
            if k != 1 or not group[0]["explicit"] or len(block) > 12:
                continue
            header, first_label_at, hdr_seen = None, 0, 0
        else:
            header, first_label_at, hdr_seen = found

        # A single column of BARE numbers under a header located by proximity is
        # the weakest evidence on any page, and it is where every false positive
        # in the captured corpus came from: amity.edu's "150,000 / Students"
        # counter strip and jmi.ac.in's table of hostel bed capacities. Refuse
        # it. The cost is real but small — tiut.ac.in's "Additional Fees" list
        # prints bare numbers under "Amount (₹)" and is lost, while the nine
        # three-column programme rows on the same page are unaffected — and
        # ordinary "Course | Fee" tables survive because they print a currency.
        if k == 1 and not all(r["explicit"] for r in group):
            continue

        # Even with explicit amounts, several competing amount headers in the
        # window mean the segmentation is ambiguous: lpu.in's residence table
        # has "Charges with…"/"Charges without…" plus a continuation line under
        # every cell, so k collapses to 1 and row two's "label" is really the
        # amenity note from row one. Then demand the row name the fee itself.
        need_fee_label = (k == 1 and (header is None or hdr_seen > 1))
        if header is None and not all(r["explicit"] for r in group):
            continue
        # Bare numbers are only money because a header said so. Without a
        # currency-ish header they stay numbers — this is the rule that keeps
        # PIN codes, phone digits and visitor counters out of the payload.
        if header is not None and not all(r["explicit"] for r in group) \
                and not any(_CURRENCY_HDR_RE.search(h) for h in header):
            continue

        rows_done = 0
        for gi, r in enumerate(group):
            if len(facts) >= MAX_FACTS:
                break
            s, e = r["span"]
            blk = lines[s:e]
            if gi == 0 and found is not None:
                blk = blk[first_label_at:]
            elif gi == 0 and len(blk) > 12:
                blk = blk[-4:]
            label = _clean_label(blk)
            if label is None:
                continue
            if need_fee_label and not _FEE_WORD_RE.search(label):
                continue
            if header is not None:
                value = "; ".join(f"{h}: {m}" for h, m in zip(header, r["money"]))
            else:
                value = r["money"][0]
            _put(facts, label, value)
            rows_done += 1
        if rows_done:
            best_rows = max(best_rows, rows_done)
            headed = headed or header is not None
    return facts, best_rows, headed


# ---------------------------------------------------------------------------
# Shape A: label and amount glued on one line by tag-stripping
#   "B.A-I (1st Semester (Without any Practical Subject))10347/-"
# ---------------------------------------------------------------------------

_GLUED_RE = re.compile(r"^(?P<label>.{3,90}?)(?P<money>(?:" + _CUR_PRE + r")?\s{0,3}"
                       + r"(?:" + _NUM + r")(?:" + _UNIT + r")?"
                       + r"(?:\s{0,3}(?:-|–|—|to)\s{0,3}(?:" + _CUR_PRE + r")?\s{0,3}"
                       + r"(?:" + _NUM + r")(?:" + _UNIT + r")?)?"
                       + r"\s{0,3}(?:" + _CUR_POST + r"))$", re.I)


def _facts_from_glued(lines: list[str], used: set[int]) -> tuple[dict[str, str], int]:
    facts: dict[str, str] = {}
    hits = 0
    for i, raw in enumerate(lines):
        if i in used:
            continue
        if len(facts) >= MAX_FACTS:     # nothing further can be emitted
            break
        s = raw.strip().rstrip(" .")[:MAX_LINE]
        # A line with no digit cannot hold an amount; on a 40k-char page most
        # lines are nav and this skips the regex entirely.
        if len(s) < 6 or not any(c.isdigit() for c in s) or _BAD_LINE_RE.search(s):
            continue
        m = _GLUED_RE.match(s)
        if m is None:
            continue
        # A whole multi-column row can arrive with no separators at all:
        # "B.Tech. Hons. CSE62,000/-96,000/-36,000/-1,32,000/-1,15,000/-…".
        # The label then swallows every column but the last, and which column
        # the survivor is cannot be recovered — so refuse the row rather than
        # publish one arbitrary number under a mangled name.
        raw_label = m.group("label")
        got = _money_at(raw_label)
        while got is not None and not got[2]:
            got = _money_at(raw_label, got[1])
        if got is not None:
            continue
        label = _clean_label([raw_label])
        if label is None:
            continue
        money = m.group("money").strip()
        mv = _money_only(money)
        if mv is None:
            continue
        used.add(i)
        hits += 1
        _put(facts, label, money)
    return facts, hits


# ---------------------------------------------------------------------------
# Shape C: prose — "Hostel fee 1st Installment:- Rs. 5990/- (…)"
# ---------------------------------------------------------------------------

_LABEL_TAIL_RE = re.compile(r"[\s:;=@\-–—]+(?:is|are|of|at|rs|inr|only)?[\s:;=@\-–—]*$", re.I)

# A label left dangling on a function word is a sentence fragment, not the name
# of a charge. lpu.in's library blurb — "There are more than 20 lakh books" —
# carries the fee word "library" and an explicit "20 lakh", and this is what
# stops it becoming a fee.
_DANGLING_RE = re.compile(
    r"\b(?:than|more|over|above|below|about|around|nearly|approx(?:imately)?|of|for"
    r"|with|and|the|is|are|was|were|to|in|on|at|by|from|up|upto|has|have|had|be"
    r"|been|a|an|as|that|which|there|it|its|than)$", re.I)


def _facts_from_prose(lines: list[str], used: set[int]) -> tuple[dict[str, str], int]:
    facts: dict[str, str] = {}
    hits = 0
    for i, raw in enumerate(lines):
        if len(facts) >= PROSE_SLOTS:
            break
        if i in used:
            continue
        line = raw.strip()[:MAX_LINE]
        if len(line) < 8 or len(line) > 400 or not any(c.isdigit() for c in line):
            continue
        if _BAD_LINE_RE.search(line) or not _FEE_WORD_RE.search(line):
            continue
        hit = _money_at(line)
        while hit is not None and not hit[2]:
            nxt = _money_at(line, hit[1])
            hit = nxt
        if hit is None:
            continue
        start, end, _explicit, _val = hit
        value = _verbatim(line, start, end)
        prefix = line[:start]
        label = _clean_label([_LABEL_TAIL_RE.sub("", prefix)])
        if label is None:
            # Amount leads the sentence ("Rs. 30/- extra facilitation charges…").
            # The clause after it is the label; the amount stays as printed.
            after = line[end:end + 90].strip(" .:;-–—()")
            after = after.split(".")[0]
            label = _clean_label([after])
            if label is None:
                continue
        if _DANGLING_RE.search(label.rstrip(" .,;:")):
            continue
        used.add(i)
        hits += 1
        _put(facts, label, value)
    return facts, hits


# ---------------------------------------------------------------------------

def _put(facts: dict[str, str], label: str, value: str) -> None:
    value = _SPACE_RE.sub(" ", value).strip(" .,;:")
    if not value or len(value) > 220:
        return
    if label in facts:
        if facts[label] == value:
            return
        for n in range(2, 6):
            alt = f"{label} ({n})"
            if alt not in facts:
                facts[alt] = value
                return
        return
    facts[label] = value


def _summary(college: str, facts: dict[str, str], headed: bool) -> str:
    college = _SPACE_RE.sub(" ", (college or "The college")).strip()[:60] or "The college"
    n = len(facts)
    noun = "fee" if n == 1 else "programmes and charges"
    words = [f"{college} publishes fee amounts for {n} {noun} on this page."]
    for label, value in list(facts.items())[:1]:
        words.append(f"Example — {label[:60]}: {value[:110]}.")
    if headed:
        words.append("Amounts are as printed on the page.")
    text = " ".join(words)
    parts = text.split()
    if len(parts) > 45:
        text = " ".join(parts[:45]).rstrip(",;:") + "…"
    return text[:600]


def extract(text: str, url: str, college_name: str) -> dict | None:
    """Fee facts for one page, or None. Never raises."""
    try:
        return _extract(text or "", url or "", college_name or "")
    except Exception:      # noqa: BLE001 - one broken page must not kill a batch
        return None


def _extract(text: str, url: str, college_name: str) -> dict | None:
    if len(text) > MAX_TEXT:
        text = text[:MAX_TEXT]
    if len(text.strip()) < 40:
        return None
    # Cheap gate. "feedback" and "coffee" do not match \bfees?\b, which is the
    # whole point — the crawler classifies pages by substring and hands us
    # /stakeholder-feedback.php believing it is a fee page.
    n_fee = len(_PAGE_FEE_RE.findall(text))
    if n_fee < 2:
        return None
    # A scholarship page is full of money and full of the word "fee" (schemes
    # reimburse fees), but every amount on it is an award, not a price. Measured
    # over the captured corpus the two classes separate cleanly: real fee pages
    # run 2-3 scholarship mentions against 7-40 fee mentions, scholarship pages
    # run 24-92 against 9-49. Whichever term dominates decides the topic.
    if len(_PAGE_SCHOLARSHIP_RE.findall(text)) > n_fee:
        return None

    lines = text.split("\n")
    if len(lines) > MAX_LINES:
        lines = lines[:MAX_LINES]

    table_facts, table_rows, headed = _facts_from_tables(lines)
    used: set[int] = set()
    glued_facts, glued_hits = _facts_from_glued(lines, used)
    prose_facts, prose_hits = _facts_from_prose(lines, used)

    facts: dict[str, str] = {}
    for src in (table_facts, glued_facts):
        for k, v in src.items():
            if len(facts) >= MAX_FACTS - PROSE_SLOTS:
                break
            _put(facts, k, v)
    for k, v in prose_facts.items():
        if len(facts) >= MAX_FACTS:
            break
        _put(facts, k, v)
    if not facts:
        return None

    if headed and table_rows >= 3:
        conf = 0.90
    elif headed and table_rows >= 2:
        conf = 0.75
    elif glued_hits >= 5:
        conf = 0.85
    elif glued_hits >= 3:
        conf = 0.72
    elif table_rows >= 2:
        conf = 0.68
    elif prose_hits >= 4:
        conf = 0.65
    elif prose_hits >= 2:
        conf = 0.55
    else:
        # A single passing mention is a real fact but a poor page-level claim;
        # hand it to the LLM rather than store a fee we half-understand.
        return None

    if _URL_FEE_RE.search(url or ""):
        conf += 0.02
    conf = max(0.0, min(0.92, conf))
    if conf < MIN_CONFIDENCE:
        return None

    return {"summary": _summary(college_name, facts, headed),
            "facts": facts,
            "confidence": round(conf, 2)}


# ---------------------------------------------------------------------------
# Self-test. Real page text (captured from the live sites via
# web_enrich.parse_html) plus the adversarial cases that decide calibration.
# Run:  python -m rag_core.sources.extractors.fees [--pages DIR]
# ---------------------------------------------------------------------------

REAL_TIUT_TABLE = """Fee Structure
Course
Duration
Admission Fee
Semester Fee
Total Course Fee
Computer Science & Engineering
 B.Tech.
4 years
8 semesters
15,000 INR
50,000 INR
4,15,000 INR
BCA
4 years
8 semesters
15,000 INR
30,000 INR
2,55,000 INR
MCA
2 years
4 semesters
15,000 INR
30,000 INR
1,35,000 INR
Computer Science & Engineering (AI and Machine Learning)
 B. Tech. (AI and Machine Learning)
4 years
8 semesters
15,000 INR
60,000 INR
4,95,000 INR
HR/Marketing/Finance/Healthcare
 MBA
2 years
4 semesters
15,000 INR
65,000 INR
2,75,000 INR
Bachelor of Medical Laboratory Science
 BMLS
4 years
(6 Semester+ 1
YearInternship/Training) semesters
15,000 INR
35,000 INR
2,25,000 INR
Certificate course in Proficiency in English
0 years
1 semesters
20,000 INR
0 INR
20,000 INR
Get In Touch
+91 9230994080 | 81 | 82 | 83 | 84
Campus: Techno India University, Tripura, Maheshkhola, Anandanagar, Agartala, Tripura - 799004
Our Visitor
630597 users
Users Today: 161
Users This Year: 426961"""

REAL_TIUT_PHD = """Ph.D. Fee Structure
Ph.D. Programme
Admission Fees
Sem. wise Tuition Fees
Caution Deposit (Refundable)
School of Engineering & Architecture
 Computer Science & Engineering
15,000 INR
55,000 INR
8,000 INR
School of Humanities & Social Sciences
 English
15,000 INR
40,000 INR
5,000 INR
School of Law
 Law
15,000 INR
55,000 INR
5,000 INR
Additional Fees (Applicable to ALL Ph.D. Programmes)
Item
 Amount (₹)
 Nature
Registration Fee
 10,000
 One-time (Payable in Semester II)
Pre-Submission Fee
 5,000
 One-time (Payable in Semester V)
Convocation Fee
 3,000
 One-time (Payable in Semester VI)
Note: University Bus facility is optional and payable @ Rs. 7,360/- per head per semester.
Estd. under The Tripura Act No. 4 of 2023 & UGC u/s 2(f) of UGC Act. 1956"""

REAL_GCG11 = """Fee Structure
Fee Structure for Session 2025-26
Fee Structure for Session 2023-24
Class1st Installment
B.A-I (1st Semester (Without any Practical Subject))10347/-
B.A-I (1st Semester (With One Practical Subject))12187/-
B.SC-I (1st Semester (Medical))13747/-
BCA-I (1st Semester)26264/-
B.Com-I (1st Semester)12187/-
M.Sc-I (1st Semester (IT))34583/-
PGDCA-I (1st Semester)18059/-
Hostel fee 1st Installment:- Rs. 5990/- (Including Refundable Hostel Security of Rs. 350/- and Mess Security of Rs. 1600/-)
Hostel Fee 2nd Installment:- Rs. 4040/-
The Following amount will be charged extra as above said fees.
Rs. 30/- extra facilitation charges on per receipt. Fees should be deposit in the Sampark Centers on the same day till 8:00 PM after generate the receipt.
Foreign Students extra Rs. 1177/- charged.
Late fee with principal permission Rs. 1000/-.
Migration fee of Rs. 451/- will be charged if student comes from any other board except Punjab, Haryana, Himachal, CBSE.
Contact Us
Phone: 0172-2740614
Address: Sector 11 , Chandigarh, UT, India
Zip Code: 160011"""

# christuniversity.in/fee-structure renders its fees with JavaScript: the static
# HTML the crawler gets is nav only. A very common real negative.
REAL_CHRIST_JS = """CHRIST (Deemed to be University) | Central Campus
Admission
Examination
Online Programmes
Doctoral Programmes
Hostel & Dining
Comfortable living
Fee Structure
Online Payment Portal
Secure, quick & convenient payments.
Scholarship and Fellowship Support Cell
Student Accommodation
Academic Calendar 2026-27
Dharmaram College Post, Hosur Road, Bengaluru - 560029, Karnataka, India
Tel: +91 80 4012 9100 / 9600
Fax: +91 80 4012 9000
Copyright © CHRIST (Deemed to be University) 2025 | Privacy Policy"""

# chitkara.edu.in links "Fee Structure" to /admissions/, which has no fees —
# only helpline numbers, PIN codes and programme durations.
REAL_CHITKARA = """Admissions for Academic Year 2026
give a miss call on1800 267 1999
Admission Helpline9501105714
9501105715
Delhi & NCR Helpline9599368734
WhatsApp9859000000
2-Year Post Graduate Programs
2-Year MBA with AI in Business Minor Specialisations:
4-Year Engineering Programs
5-Year B.A. LL.B. (Hons.)
98759 98712
98824 64749
Chandigarh-Patiala National Highway, Punjab 140 401
Unit No. A 201-202, Elante Mall Office Complex
Industrial Area Phase 1, Chandigarh 160 002
Copyright 2026 Chitkara Educational Trust
Fee Structure
Fees and payment"""

# bncollegepatna.com: the crawler matched "fee" inside "feedback".
REAL_FEEDBACK = """Teachers' Feedback Form
B.N. College, Patna University, Patna.
Please indicate your level of satisfaction for each item given in below by ticking Ed against a score between 1 and 4 ( 1.Excellent 2.Very Good 3.Good 4.Poor).
1. Syllabus is suitable to the course and need based.
2. Course content is followed by corresponding
3. Sufficient number of prescribed books is available in
11. Experience related to the adoption of new techniques
13. The administration is teacher-friendly.
Student feedback
Alumni Feedback"""

# amity.edu/admission: a marketing counter strip. The only amount-ish caption on
# the page ("Fee related queries") is ~975 lines above it, next to phone numbers.
# An early build published "150,000 Students" as a fee.
REAL_STAT_STRIP = """Admission helpline: 7303-399-399
 Fee related queries (1st Sem):
 8770479789 / 7024291885
Greater Noida Campus
Landline - 0120 - 5065308 / 09
Tuition fees and other fees are given in the prospectus.
WHY AMITY
Grooming leaders who are not only thorough professionals but also good human beings
This is why we are
consistently ranked
no.1
Read More About Amity
27 Years'
of Education Experience
24 Years'
of Management Education Experience
150,000
Students
6000
Faculties
45000
paper published
5000
Books Authored
1200
acres of Campuses"""

# jmi.ac.in/Hostels: the numbers are bed capacities. "Fee Structure" appears in
# the nav far above, and an early build turned 275 beds into a Rs 275 fee.
REAL_CAPACITY = """Fee Structure
Hostel Fee and other fees are notified separately.
Girls Hostels in Jamia
Campus - A:
Girls Hostels - Campus A
Sr. No
 Name of the Hostel
 Capacity
1
 Aruna Asaf Ali Hostel (AAA)
 61
2
 Begum Anis Kidwai Hostel (BAK)
 275
3
 Bi Amma Hostel
 385
4
 Bi Amma Hostel - Annexe
 24"""

# A whole multi-column fee row with no separators at all. Which column the last
# number belongs to is unrecoverable, so the row must be refused outright.
REAL_GLUED_MULTICOL = """Fee Structure 2026-27
Below is the detailed fee structure for various programs offered at the University.
ROORKEE COLLEGE OF SMART COMPUTING
CoursesTuition Fees1st Year2nd Year3rd Year4th Year
Semester 1Semester 2Total
B.Tech. Hons. CSE62,000/-96,000/-36,000/-1,32,000/-1,15,000/-1,15,000/-1,15,000/-
B.Tech. Hons. AI & ML62,000/-1,31,000/-31,000/-1,62,000/-1,45,000/-1,45,000/-1,45,000/-
B.Tech. Hons. Cyber Security62,000/-1,06,000/-36,000/-1,42,000/-1,25,000/-1,25,000/-1,25,000/-"""

# Donor-named awards under an "Amount" header, on a page whose fee mentions come
# from "fee concession" prose. Money, but not a price.
REAL_SCHOLARSHIP_TABLE = """Scholarships and Fee Concession
The following scholarships are awarded. Fee concession is separate.
Name of the Scholarship
 Amount
Virasat, Alumni Association Six Scholarships
 12,000.00
Late Mrs V Bhargava
 1500.00
Mrs Renu Anand
 1500.00
Late Sh Mohinder Singh Mrs Tejinder Kaur
 2000.00
Smt Kailash Jain Dr Punam Gupta
 2500.00
Scholarship applications and fee concession forms are available.
Merit scholarship and freeship details follow the scholarship policy."""

# lpu.in/student-services/residence.php: a genuine one-column charge table with
# a single unambiguous header. The two-header variant higher up the same page
# ("with"/"without Early Decision Benefit") is correctly refused instead.
REAL_LPU_CHARGES = """Gym, Sports & Swimming Fee Charges
 Academic Session 2026-2027
X
For Hostellers
For Day Scholars
For Hostellers
For Hostellers
Charges with Early Decision Benefit
(Till 31st July 2026)
 Charges without Early Decision Benefit
 (After 31st July 2026)
GYM Facility
 Rs. 4000/-
 Rs. 5000/-
Sports Facilities
Badminton
 Rs. 2000/-
 Rs. 2500/-
Combative Sports
Shooting
 Rs. 4000/-
 Rs. 5000/-
For Day Scholars
For Day Scholars
Charges
GYM Facility
 Rs. 6000/-
Sports Facilities
Badminton
 Rs. 3000/-
Volleyball
Basketball
Squash
Combative Sports
Shooting
 Rs. 6000/-
Important Note:
 (Gym Facility can be allocated after depositing Rs. 4000)
The residential fee, along with charges for mess, laundry and gym, are all managed by Pro Bricks.
There are more than 20 lakh books and e-books on all subject fields. Library system subscribes journals."""

SYNTH_PROSE = """Fee Structure 2025-26
The tuition fee for the B.Tech. programme is Rs. 1,20,000 per annum.
Hostel charges are Rs 45,000 - 1,20,000 per year depending on room type.
Mess charges: Rs. 3,500 per month.
A one-time caution deposit of Rs 10,000/- (refundable) is payable at admission.
Examination fee Rs 2,500 per semester.
Our NIRF rank improved to 45 this year and we have the best placements in the region.
Average package offered was 8.5 LPA."""

# The most common table shape after tiut's: two columns, amounts carrying their
# own basis. "4 years" / "3-5 years" in the same block must stay text.
SYNTH_PER_ANNUM_TABLE = """Fee Structure 2026-27
Course
Duration
Tuition Fee
Hostel Fee
B.Tech. Computer Science
4 years
Rs 1,20,000 per annum
Rs 45,000 per annum
MBA
2 years
Rs 2,50,000 per annum
Rs 60,000 per annum
B.Sc. Nursing
3-5 years
Rs 90,000 per semester
Rs 45,000 per annum"""

SYNTH_LAKH = """Fee Structure
Total course fee for MBBS is Rs 1.2 Lakh per year.
BDS total tuition fee: 12.5 lakhs for the entire course.
B.Sc Nursing fee is 1.2L per annum.
Development charges Rs. 25,000/- one-time.
International students pay $12,000/semester."""

NEG_ADDRESS = """Contact Us
Admission Office, Sector 62, Noida - 201309, Uttar Pradesh
Phone: 0120-4392000, Mobile: 9810355724
Established under Act No. 12 of 1956, affiliated to University Code 27051
Our courses: B.Tech (240 seats), MBA (120 seats)
Cut off 2024: 96.5 percentile. Last rank 12045.
Fee structure and fees are available in the prospectus."""

NEG_EMPTY = ""
NEG_NAV = """Home
About Us
Admissions
Fee Structure
Contact
Fees and Payment"""
NEG_OTHER_TOPIC = """Placement Highlights
Highest package 45 LPA offered by Microsoft.
Average CTC 8.2 LPA in 2024. 1,240 students placed.
Top recruiters: TCS, Infosys, Wipro.
Scholarship of Rs. 50,000 is awarded to the top 10 rank holders.
Fees concession available for meritorious students."""


def _selftest() -> None:  # pragma: no cover - developer tool
    import argparse, json, os, time

    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default=None,
                    help="directory of captured parse_html() dumps to also run")
    args = ap.parse_args()

    cases = [
        ("REAL tiut.ac.in/fee_structure", REAL_TIUT_TABLE,
         "https://www.tiut.ac.in/fee_structure", "Techno India University, Tripura", True),
        ("REAL tiut.ac.in/phd_fee_structure", REAL_TIUT_PHD,
         "https://www.tiut.ac.in/phd_fee_structure", "Techno India University, Tripura", True),
        ("REAL gcg11.ac.in/admissions/fee-structure", REAL_GCG11,
         "https://gcg11.ac.in/admissions/fee-structure/", "PG Govt College for Girls", True),
        ("REAL christuniversity.in/fee-structure (JS-only)", REAL_CHRIST_JS,
         "https://www.christuniversity.in/fee-structure", "CHRIST University", False),
        ("REAL chitkara.edu.in/admissions (phones+PINs)", REAL_CHITKARA,
         "https://www.chitkara.edu.in/admissions/", "Chitkara University", False),
        ("REAL bncollegepatna feedback ('fee' in feedback)", REAL_FEEDBACK,
         "https://www.bncollegepatna.com/staff_admin/feedback_alumni3.php", "B.N. College", False),
        ("REAL amity.edu stat counters (150,000 Students)", REAL_STAT_STRIP,
         "https://www.amity.edu/admission/", "Amity University", False),
        ("REAL jmi.ac.in hostel bed CAPACITY not fee", REAL_CAPACITY,
         "https://www.jmi.ac.in/Hostels", "Jamia Millia Islamia", False),
        ("REAL glued multi-column row (unrecoverable)", REAL_GLUED_MULTICOL,
         "https://example.ac.in/fees", "Haridwar University", False),
        ("REAL scholarship amounts under 'Amount' header", REAL_SCHOLARSHIP_TABLE,
         "https://example.ac.in/scholarship", "Example College", False),
        ("REAL lpu.in one-column charges table", REAL_LPU_CHARGES,
         "https://www.lpu.in/student-services/residence.php", "LPU", True),
        ("SYNTH two-column table, per-annum cells", SYNTH_PER_ANNUM_TABLE,
         "https://example.ac.in/fee-structure", "Example Institute", True),
        ("SYNTH prose per-annum/range/deposit", SYNTH_PROSE,
         "https://example.ac.in/fee-structure", "Example Institute", True),
        ("SYNTH lakh / L / foreign currency", SYNTH_LAKH,
         "https://example.ac.in/fees.php", "Example Institute", True),
        ("NEG address/phone/PIN/seats/percentile", NEG_ADDRESS,
         "https://example.ac.in/contact", "Example Institute", False),
        ("NEG empty page", NEG_EMPTY, "https://example.ac.in/fees", "Example", False),
        ("NEG nav only", NEG_NAV, "https://example.ac.in/fees", "Example", False),
        ("NEG different topic (placements/scholarship)", NEG_OTHER_TOPIC,
         "https://example.ac.in/placement", "Example", False),
    ]

    bad = 0
    for name, text, url, college, want in cases:
        t0 = time.perf_counter()
        got = extract(text, url, college)
        ms = (time.perf_counter() - t0) * 1000
        ok = (got is not None) == want
        bad += not ok
        print(f"\n{'PASS' if ok else 'FAIL'}  {name}   [{len(text)} chars, {ms:.1f} ms]")
        if got is None:
            print("      -> None")
        else:
            print(f"      confidence {got['confidence']}")
            print(f"      summary    {got['summary']}")
            print(f"      facts      ({len(got['facts'])})")
            for k, v in list(got["facts"].items())[:12]:
                print(f"         {k!r}: {v!r}")
            if len(got["facts"]) > 12:
                print(f"         … {len(got['facts']) - 12} more")

    if args.pages and os.path.isdir(args.pages):
        print("\n" + "=" * 70 + "\nCAPTURED PAGES\n" + "=" * 70)
        for fn in sorted(os.listdir(args.pages)):
            if not fn.endswith(".txt"):
                continue
            raw = open(os.path.join(args.pages, fn), encoding="utf-8").read()
            head, _, body = raw.partition("=" * 60)
            t0 = time.perf_counter()
            got = extract(body.strip(), head.strip(), "College")
            ms = (time.perf_counter() - t0) * 1000
            print(f"\n{head.strip()}  [{len(body)} chars, {ms:.1f} ms]")
            if got is None:
                print("   -> None")
            else:
                print(f"   conf {got['confidence']} | {len(got['facts'])} facts")
                print(f"   {got['summary']}")
                print("   " + json.dumps(dict(list(got["facts"].items())[:6]),
                                         ensure_ascii=False)[:600])

    print(f"\n{'ALL PASS' if not bad else str(bad) + ' FAILURES'}")


if __name__ == "__main__":  # pragma: no cover
    _selftest()
