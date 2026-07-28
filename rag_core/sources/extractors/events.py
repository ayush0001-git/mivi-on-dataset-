"""Dated, actionable notices from a college's own notice board / news page.

WHY THIS IS DELIBERATELY NARROW
`events` is the least valuable of the six harvested topics and by far the
easiest to pollute. The LLM fallback, measured on the events rows already in
college_web_facts, returned things like {"speech competition": "third place"}
and a list of campus cities — text that is arguably true and answers no
question a student asked. So this parser refuses far more than it accepts: an
item survives only if the page states BOTH a full calendar date AND an
admission-cycle word (admission, exam, result, counselling, merit list, last
date...). A dated hackathon, a convocation, an "Annual Fest" — none of those
are events in the sense MIVI needs, and None is the correct answer for them.

WHAT THE PAGES ACTUALLY LOOK LIKE
Measured against real notice boards (srhu.edu.in, tiut.ac.in, davchd.ac.in,
chitkara.edu.in, bncollegepatna.com), a notice list arrives from parse_html()
as a flat run of lines where the date sits on its OWN line beside its title, in
one of two orders:

    Third Counseling for Admission to M.Sc. Medical Physics, 2026   <- title
    17 July, 2026                                                   <- date
    View                                                            <- chrome

    2026-06-15                                                      <- date
    Notification on Admit Cards                                     <- title
    Download                                                        <- chrome

Both orders are common in the wild and a given page is consistent with itself,
so orientation is decided by voting across all of the page's date lines rather
than guessed per item. Guessing per item is what turns srhu.edu.in's
"16 July, 2026" — which has a valid title on BOTH sides — into a fact attached
to the wrong notice.

WHAT IT REFUSES, ON PURPOSE
- Undated anything. bncollegepatna.com publishes real titles ("3rd Merit List
  for Admission - B.Sc. (Math) Part 1") with no date anywhere near them; a
  merit list with no date is not actionable, so the page yields nothing.
- Fests, hackathons, conferences, convocations, prize distributions, sports
  days, achievements — dated or not.
- Tenders, quotations and staff recruitment. They dominate small-college notice
  boards and are irrelevant to an applicant.
- Month-year with no day ("Examinations- July 2026"). Not a date a student can
  act on, and it collides with academic-session strings like "2026-27".

The verbatim date text is what gets stored, never a normalised one: pages write
07/10/2024 and 04-March-2025 and 2026-07-14, and DD/MM vs MM/DD is genuinely
ambiguous on some of them. The parsed date is used only for ordering and for
the staleness check; the student sees exactly what the college wrote.
"""
from __future__ import annotations

import re
from datetime import date

# Bounds. web_enrich hands over the WHOLE parsed page (chitkara.edu.in's home
# page measures ~36k chars), not the 6k slice the LLM prompt gets, so the work
# has to be bounded here rather than assumed small.
MAX_TEXT_CHARS = 80_000
MAX_LINES = 4_000
MAX_ITEMS = 6              # a card cannot show more, and older ones are noise
MAX_KEY_CHARS = 130

# Overridable so the self-test does not rot as the wall clock moves. None means
# "use today"; recency is a real property of a notice board, not a fixture.
TODAY: date | None = None

_MONTH_RE = (r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
             r"jul(?:y)?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|"
             r"dec(?:ember)?)")
_MONTH_NUM = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
              "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}

# Every pattern is fenced with (?<![\d/-]) / (?![\d/-]) so it cannot bite a
# chunk out of a longer number. That guard is what stops "+91-135-2471102"
# (srhu.edu.in's phone), "0612-22677619", "1849-1898 AD" and the academic
# session strings "2019-2020" / "Session-2026-27" from parsing as dates —
# measured on those exact strings, see the self-test.
_P_ISO = re.compile(r"(?<![\d/-])(20\d{2})-(\d{1,2})-(\d{1,2})(?![\d/-])")
_P_DMY_NUM = re.compile(r"(?<![\d/.-])(\d{1,2})([/.-])(\d{1,2})\2((?:19|20)\d{2}|\d{2})(?![\d/.-])")
_P_DMY_TXT = re.compile(r"(?<![\w/-])(\d{1,2})(?:st|nd|rd|th)?[\s.\-]{1,3}" + _MONTH_RE +
                        r"\.?,?[\s\-]{1,3}((?:19|20)\d{2})(?!\d)", re.I)
_P_MDY_TXT = re.compile(r"(?<![\w/-])" + _MONTH_RE + r"\.?[\s\-]{1,3}(?<!\d)(\d{1,2})"
                        r"(?:st|nd|rd|th)?,?[\s\-]{1,3}((?:19|20)\d{2})(?!\d)", re.I)

# An item must name something a student can act on. "notice"/"notification" are
# the standard words on Indian college sites for exactly this, so they are in
# despite being generic; _JUNK below is what keeps them honest.
_ACTIONABLE = re.compile(r"""(?xi)
    \b(?:
        admissions?|admit\s*cards?|hall\s*tickets?|prospectus|
        applications?|apply|registrations?|enrol(?:l?ment|ment)?|counsell?ing|
        last\s*date|due\s*date|deadline|closing\s*date|last\s*day|
        exams?|examinations?|entrance|date\s*-?\s*sheet|
        results?|answer\s*keys?|merit\s*list|rank\s*list|cut\s*-?\s*offs?|
        seat\s*allotment|allotment|
        notifications?|notices?|circulars?|
        interviews?|viva|scholarships?|
        semester\s*fee|fee\s*(?:payment|submission|deposit)
    )\b
""")

# Notice boards of small colleges are mostly tenders and staff hiring. These
# carry dates and the word "notice", so they clear every other gate; they are
# useless to an applicant and would crowd out the six slots on the card.
_JUNK = re.compile(r"""(?xi)
    \b(?:
        e?\s*-?\s*tenders?|quotations?|quatations?|bids?|auctions?|
        empanel\w*|procure\w*|vacanc\w*|walk\s*-?\s*in|
        post\s+of|appointment\s+of|recruitment\s+(?:of|for)|
        repair|painting|roof|furnitures?|civil\s+work|
        obituary|condolence
    )\b
""")

# Chrome that sits between a date and its title in list markup.
_CHROME = re.compile(r"""(?xi)^\s*(?:
    view|download|read\s*more|more(?:\s+updates?)?|click\s*here|know\s*more|
    apply\s*now|details?|next|previous|prev|last|home|notice\s*board|
    view\s*(?:all|more|details?)|see\s*more|archives?
)\b""")

# Only these phrasings promote a sentence-level date to a fact. A bare date in
# running prose is almost always a past event being described.
_DEADLINE_CUE = re.compile(r"""(?xi)
    \b(?:
        last\s+date|last\s+day|due\s+date|deadline|closing\s+date|
        will\s+be\s+held|to\s+be\s+held|scheduled\s+(?:on|for)|
        commenc\w{0,4}|begins?\s+on|starts?\s+on|opens?\s+on|closes?\s+on|
        date\s+of\s+(?:exam\w{0,7}|interview|counsell?ing|results?)|
        registration\s+(?:opens|closes|ends|begins)
    )\b
""")

# A page under one of these paths is not a student events page whatever it says.
# No \b after the stem: the live path is /tenders/, /careers/, /jobs/ far more
# often than the singular, and \b between "tender" and "s" never fires.
_BAD_URL = re.compile(r"(?i)/(?:tender|quotation|quotaion|career|recruit|"
                      r"vacanc|job|alumni)")

_WORD = re.compile(r"[A-Za-zऀ-ॿ][\w'&./’-]*")
_SPACEY = re.compile(r"[  -​   ﻿]")


def _today() -> date:
    return TODAY or date.today()


def _norm(s: str) -> str:
    """parse_html collapses [ \\t] only, so &nbsp; survives as \\xa0 — measured
    on chitkara.edu.in, whose event dates read '10-Oct,\\xa02026'."""
    return re.sub(r"[ \t]+", " ", _SPACEY.sub(" ", s)).strip(" \t▶⏭•·|>-")


def _mk_date(y: int, mo: int, d: int) -> date | None:
    if y < 100:
        y += 2000
    # Anything outside this window is an act number, a founding year or a
    # misparse, never a notice date.
    if not (1990 <= y <= _today().year + 5):
        return None
    try:
        return date(y, mo, d)
    except ValueError:      # 31 Feb, month 13, and friends
        return None


def _find_dates(line: str) -> list[tuple[int, int, date, str]]:
    """All non-overlapping dates in one line: (start, end, parsed, verbatim)."""
    found: list[tuple[int, int, date, str]] = []
    for m in _P_ISO.finditer(line):
        d = _mk_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if d:
            found.append((m.start(), m.end(), d, m.group(0)))
    for m in _P_DMY_NUM.finditer(line):
        a, b, y = int(m.group(1)), int(m.group(3)), int(m.group(4))
        # Indian sites write DD/MM. Only flip when the first field cannot be a
        # month; when both fit, day-first is the local convention. The stored
        # value is verbatim either way, so this only affects ordering.
        d = _mk_date(y, b, a) if a > 12 or b <= 12 else _mk_date(y, a, b)
        if d:
            found.append((m.start(), m.end(), d, m.group(0)))
    for m in _P_DMY_TXT.finditer(line):
        d = _mk_date(int(m.group(3)), _MONTH_NUM[m.group(2)[:3].lower()], int(m.group(1)))
        if d:
            found.append((m.start(), m.end(), d, m.group(0)))
    for m in _P_MDY_TXT.finditer(line):
        d = _mk_date(int(m.group(3)), _MONTH_NUM[m.group(1)[:3].lower()], int(m.group(2)))
        if d:
            found.append((m.start(), m.end(), d, m.group(0)))
    found.sort(key=lambda t: (t[0], -(t[1] - t[0])))
    out: list[tuple[int, int, date, str]] = []
    end = -1
    for st, en, d, v in found:
        if st >= end:
            out.append((st, en, d, v))
            end = en
    return out


def _strip_dates(line: str, dates: list[tuple[int, int, date, str]]) -> str:
    if not dates:
        return line
    keep, prev = [], 0
    for st, en, _d, _v in dates:
        keep.append(line[prev:st])
        prev = en
    keep.append(line[prev:])
    return " ".join("".join(keep).split())


def _is_date_line(line: str, dates: list[tuple[int, int, date, str]]) -> bool:
    """A line that exists to carry a date, not to say anything."""
    if not dates:
        return False
    covered = sum(en - st for st, en, _d, _v in dates)
    return len(line) <= 26 or covered * 2 >= len(line)


def _title_ok(line: str) -> bool:
    """Is this line a notice title rather than a nav label or a timestamp?

    The word count is taken AFTER removing dates, which is what separates the
    real title "Notice for Counselling on 15 June 2026" from srhu.edu.in's page
    banner "Circulars And Notices March 17, 2025" — strip the date from the
    banner and only three words are left, so it was never a notice at all.
    """
    if not (18 <= len(line) <= 220) or _CHROME.match(line):
        return False
    if _JUNK.search(line) or not _ACTIONABLE.search(line):
        return False
    body = _strip_dates(line, _find_dates(line))
    return len(_WORD.findall(body)) >= 4


def _add(items: dict[str, tuple[date, str, str]], title: str, d: date, verbatim: str) -> None:
    """Keyed on the lowercased title so the same notice cannot land twice.
    tiut.ac.in renders its board as a table whose Title and Content columns hold
    identical text, so every item genuinely does arrive twice."""
    key = title if len(title) <= MAX_KEY_CHARS else title[:MAX_KEY_CHARS - 1].rstrip() + "…"
    items.setdefault(key.lower(), (d, verbatim, key))


def _extract(text: str, url: str, college_name: str) -> dict | None:
    if _BAD_URL.search(url or ""):
        return None

    lines = [_norm(x) for x in text[:MAX_TEXT_CHARS].split("\n")[:MAX_LINES]]
    lines = [x for x in lines if x]
    if len(lines) < 3:
        return None

    dated = [(i, _find_dates(x)) for i, x in enumerate(lines)]
    date_rows = [(i, ds) for i, ds in dated if ds and _is_date_line(lines[i], ds)]
    found: dict[str, tuple[date, str, str]] = {}

    if date_rows:
        # Vote on orientation across the whole page. Only date lines with a
        # valid title on exactly ONE side get a vote; ambiguous ones cancel.
        before = after = 0
        for i, _ds in date_rows:
            prev_ok = i > 0 and _title_ok(lines[i - 1])
            next_ok = i + 1 < len(lines) and _title_ok(lines[i + 1])
            if prev_ok and not next_ok:
                before += 1
            elif next_ok and not prev_ok:
                after += 1
        order = () if not (before or after) else (-1, 1) if before >= after else (1, -1)

        for i, ds in date_rows:
            d, verbatim = ds[0][2], ds[0][3]
            for step in order:
                j = i + step
                if 0 <= j < len(lines) and _title_ok(lines[j]):
                    _add(found, lines[j], d, verbatim)
                    break

    # Sentence-level deadlines ("Last date for submission of application form
    # is 30 June 2026"). Rare next to a notice list but the highest-value line
    # on a page when it appears, so it is worth a second pass. Both a cue and
    # an actionable word are required — "the convocation will be held on ..."
    # has the cue and is not something a student applies to.
    labelled = 0
    for i, ds in dated:
        line = lines[i]
        if not ds or _is_date_line(line, ds) or not (20 <= len(line) <= 300):
            continue
        if _JUNK.search(line) or not (_DEADLINE_CUE.search(line) and _ACTIONABLE.search(line)):
            continue
        body = _strip_dates(line, ds)
        # Three words, not the four demanded of a bare title: this line already
        # cleared two independent gates (a deadline cue AND an actionable word),
        # and real deadlines are terse. christuniversity.in states the whole
        # thing in "Applications close on 3 August 2026" — three words once the
        # date is removed, and the single most useful line on the page.
        if len(_WORD.findall(body)) < 3:
            continue
        labelled += 1
        _add(found, line, ds[0][2], ds[0][3])

    if not found:
        return None

    items = sorted(found.values(), key=lambda t: t[0], reverse=True)[:MAX_ITEMS]
    facts = {title: verbatim for _d, verbatim, title in items}

    newest, newest_v, newest_t = items[0]
    oldest_v = items[-1][1]
    name = (college_name or "This college").strip()[:70]
    head = newest_t if len(newest_t) <= 90 else newest_t[:89].rstrip() + "…"
    # A deadline sentence already carries its own date; repeating it in
    # parentheses reads like two different dates on the card.
    stamp = "" if newest_v in head else f" ({newest_v})"
    if len(items) == 1:
        summary = f"{name} lists one dated notice on this page. {head}{stamp}."
    else:
        summary = (f"{name} lists {len(items)} dated notices on this page, "
                   f"{oldest_v} to {newest_v}. Most recent: {head}{stamp}.")
    summary = " ".join(summary.split()[:45])

    n = len(items)
    conf = 0.55 if n == 1 else 0.72 if n == 2 else 0.85
    if labelled:
        # An explicit "last date ..." sentence is the least ambiguous evidence
        # a page can offer; it does not depend on the line-adjacency guess.
        conf = max(conf, 0.8)
    # Staleness folded into confidence deliberately. The contract says
    # confidence is about how explicitly the page states the fact, and a 2019
    # merit list is stated perfectly clearly — but the number the renderer
    # gates on is this one, and a seven-year-old deadline shown to an applicant
    # as current is the exact harm this pipeline exists to avoid. Dropping an
    # archive below the threshold sends it to the LLM instead of onto a card.
    age = _today().year - newest.year
    if age >= 4:
        conf *= 0.55
    elif age >= 2:
        conf *= 0.85
    conf = round(max(0.0, min(1.0, conf)), 2)

    return {"summary": summary, "facts": facts, "confidence": conf}


def extract(text: str, url: str, college_name: str) -> dict | None:
    """Public contract. Never raises: the sweep runs unattended across 10,000
    third-party sites and a single exception would kill a whole batch."""
    try:
        if not text or not isinstance(text, str):
            return None
        return _extract(text, str(url or ""), str(college_name or ""))
    except Exception:  # noqa: BLE001 - a broken page must cost nothing
        return None


# ---------------------------------------------------------------------------
# Self-test: real page text first, adversarial pages second.
#   python -m rag_core.sources.extractors.events [dir-of-parsed-page-dumps]
# ---------------------------------------------------------------------------

# Verbatim excerpts of text produced by web_enrich.parse_html() on the live
# sites named below, trimmed only in length. Kept inline so the test still runs
# once the scratchpad dumps are gone.
REAL = {
"srhu.edu.in/admission-notices (title-then-date)": ("""
Home
Admission Notices
Admission Notices March 17, 2025
 2025-09-05 6:05
Admission Notices
Amended Answer Keys of Ph.D. Entrance Examinations- July 2026
21 July, 2026
View
Third Counseling for Admission to M.Sc. Medical Physics, 2026
17 July, 2026
View
Admission Notification - MBBS programme, Academic Session-2026-27
16 July, 2026
Notice for Admission MBBS Academic Session 2026-27
View
Caution Notice for Public
View
Answer Keys of Ph.D. Entrance Examination (Ph.DEE) - July 2026
15 July, 2026
View
Notification for Second Counselling with Merit List - 2026 (Allied & Healthcare UG Programs)
14 July, 2026
View
123Next Last
Careers
NIRF
Academic Calendar
Pay your Fee
Admission Notices
Events
Examination
University Results
Circular and Notices
SWAMI RAMA HIMALAYAN UNIVERSITY
Swami Ram Nagar, Jolly Grant
Dehradun - 248016, Uttarakhand, India
Contact
+91-135-2471102 / 2471140
Email
info@srhu.edu.in
""", "https://srhu.edu.in/admission-notices/", "Swami Rama Himalayan University"),

"srhu.edu.in/circulars-and-notices (exam schedules)": ("""
Home
Circulars And Notices
Circulars And Notices March 17, 2025
 2025-08-29 10:33
Circulars And Notices
Weeding of the used Answer Scripts of University End Semester/ Year End / continuous assessment Examinations (Result declared on or before December 31, 2024)
20 May, 2026
View
Examination schedule for Allied and Healthcare (UG) Supplementary(Yearly Pattern) Examinations, November-2025
22 October, 2025
Exam Schedule Allied and Healthcare (Supplementary-UG) YEARLY Pattern, November-2025
View
Special Re-Examination, September 2025 - School of Pharmaceutical Sciences
29 August, 2025
View
Special End Semester Examination, August 2025 - School of Yoga Sciences
28 August, 2025
View
Schedule of Special End Semester Examination, September 2025 - School of Science & Technology
20 August, 2025
View
123Next Last
""", "https://srhu.edu.in/circulars-and-notices/", "Swami Rama Himalayan University"),

"tiut.ac.in/notice_board (date-then-title, ISO, table)": ("""
Admission
Admission Form
Ph.D. Admission
Fee Structure
Notice Board
Contact Info
Techno India University, Tripura, Maheshkhola, Anandanagar, Agartala, Tripura - 799004
+91 9230994080 | 81 | 82 | 83 | 84
info@tiut.ac.in
Estd. under The Tripura Act No. 4 of 2023 & UGC u/s 2(f) of UGC Act. 1956
Notice Board
Date
Title
Content
Action
2026-07-14
Notification for Semester Fee for 2nd, 3rd and 4th year students, 2026-2027
Notification for Semester Fee for 2nd, 3rd and 4th year students, 2026-2027
Download
2026-06-17
Notification on Backlog Examination
Notification on Backlog Examination
Download
2026-06-15
Notification on Admit Cards
Notification on Admit Cards
Download
2026-06-12
Notification for ESA, Spring 2026 Schedule
Notification for ESA, Spring 2026 Schedule
Download
2026-06-12
Ph.D. Admission Notification: Academic Session 2026-27(Summer)
Ph.D. Admission Notification: Academic Session 2026-27(Summer)
Download
2026-06-08
Notification for Review Results
Notification for Review Results
Download
2026-06-04
Notification for Admit Card
Notification for Admit Card
Download
""", "https://tiut.ac.in/notice_board", "Techno India University Tripura"),

"davchd.ac.in/news-1 (news list, mostly NOT actionable)": ("""
Dav College in News
Latest News and Updates
Jun 9, 2026
Chandigarh Releases Admission Prospectus 2026-27
Read More
Apr 23, 2026
PRIZE DISTRIBUTION FUNCTION
Read More
Apr 22, 2026
DAV College, Sector 10, Chandigarh, held its Annual Convocation
Read More
Nov 2, 2025
Outstanding Achievements by Our Students
Read More
Aug 15, 2025
15th August Celebration at DAV College, Chandigarh
Read More
Apr 11, 2025
Convocation Function 11/04/2025
Read More
Sep 20, 2023
Library Orientation Programme
Read More
Sep 16, 2023
Workshop on Vedic Mathematics
Read More
DAV College Chandigarh
Sector 10, Chandigarh
mail@davchd.ac.in
0172-2754400
Copyright 1958- 2026- DAV College, Sector - 10, UT. Chandigarh, India.
""", "https://www.davchd.ac.in/news-1", "DAV College Chandigarh"),

"chitkara.edu.in/events (dated FESTS - must stay empty)": ("""
Events
10-Oct, 2026
WECON 2026
16-Oct, 2026
17th Global Week 2026
24-Jul, 2026
28-Jun, 2026
10-May, 2026
Cyber Xcelerate 2026: A 24 Hour Hackathon
30-Apr, 2026
20-Apr, 2026
12-Mar, 2026
Chitkara International Animation Festival 2026
28-Feb, 2026
SHAPE 2026-Embracing the SDG vision through Architecture
10-Jan, 2026
20-Dec, 2025
""", "https://www.chitkara.edu.in/events/", "Chitkara University"),

"christuniversity.in home (deadline sentence among dated FESTS)": ("""
Admission
Applications are invited for the MBA programme
at Pune Lavasa Campus, The Analytical Hub
Applications close on 3 August 2026
Apply Now
Campus Events
Jul 27 2026
SharpEdge 2026 - Portfolio Optimisation Workshop & Challenge
Bangalore Central Campus
Jul 27 2026
ZERO to ONE - A Two Day Workshop on Entrepreneurship
Bangalore Central Campus
Jul 27 2026
Inauguration of LABYRINTH - Students Club
Bangalore Central Campus
Jul 27 2026
Alumni Interactive Session on Succeeding in Theatre: The Power of Being a Multi-hyphenate
Bangalore Central Campus
Jul 28 2026
Vidya Daan - Pass It Forward, a Book Collection Drive
Read Full Story
Dec 31 2026
World University Rankings for Innovation (WURI) 2026
Mar 31 2027
India Today-MDRA Best Colleges Ranking 2026
""", "https://christuniversity.in/", "CHRIST (Deemed to be University)"),

"bncollegepatna.com notices (real titles, NO dates)": ("""
News & Notice
Invite quotation for Books
Repair and Painting of Pariksha Bhawan (Department of BBA) BNC
Roof Treatment of Pariksha Bhawan, BNC
3rd Merit List for Admission - B.Sc. (Math) Part 1
3rd Merit List for Admission - B.A. Part 1
Admission Notice, 2019 - Vocation Education in Biotechnology (2019-2020)
2nd Merit List for Admission - B.Sc. (Math) Part 1
Important Notice - Admission in B.A./B.Sc (Math/Bio) 2019-22 | Download
Schedule for Admission in B.A. (Voc.) in Computer Applications - Download
Admission 2019-22 UG Notice Part-I
Notice Archive
Apply for admission in 2026-30 sesssion
Our Founders
Babu Bisheshwar Narayan Singh
(1849-1898 AD)
B. N. College, Near Gandhi Maidan, Ashok Rajpath, Patna- 800004, Bihar [India]
0612-22677619
Visitor Counter :
 350646
Copyright 2026 B. N. College Patna - All Rights Reserved
""", "https://bncollegepatna.com/view_all_notice.php", "Bihar National College"),
}

ADVERSARIAL = {
"empty": ("", "https://x.ac.in/events", "X College"),
"whitespace only": ("   \n\n \t \n", "https://x.ac.in/events", "X College"),
"nav only": ("""
Home
About Us
Academics
Admission
Admission Form
Fee Structure
Notice Board
Events
Examination
University Results
Circular and Notices
Placements
Contact
""", "https://x.ac.in/notice_board", "X College"),

"tiut.ac.in/fee_structure (money, NOT dates)": ("""
Fee Structure
PH.D. Program
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
""", "https://tiut.ac.in/fee_structure", "Techno India University Tripura"),

"tiut.ac.in/contact (phones, PIN, act numbers)": ("""
Contact Info
Techno India University, Tripura, Maheshkhola, Anandanagar, Agartala, Tripura - 799004
+91 9230994080 | 81 | 82 | 83 | 84
info@tiut.ac.in
Estd. under The Tripura Act No. 4 of 2023 & UGC u/s 2(f) of UGC Act. 1956
Admission
Admission Form
Ph.D. Admission
Admission Process
Fee Structure
Notice Board
""", "https://tiut.ac.in/contact", "Techno India University Tripura"),

"numbers that are NOT dates, beside real notice words": ("""
Admission Notice Board
Helpline for admission queries 0612-22677619 and +91-135-2471102
Registrar office PIN Patna- 800004, Dehradun - 248016
Affiliation No. 1-4703261 granted under UGC Act. 1956
Course code CS-12-2026 and BA-11-2025 for the admission portal
Founders lived 1849-1898 AD
Academic Session-2026-27 admission notification details
Visitor Counter : 350646
Total seats 1,20,000 across 12/24 batches
""", "https://x.ac.in/notice_board", "X College"),

"placement page (different topic)": ("""
Training and Placement Cell
Our students have been placed in leading companies.
Highest package 45 LPA
Average package Rs 6,50,000 per annum
Top recruiters TCS, Infosys, Wipro, Amazon
The placement cell has best placements in the region.
960 students placed in 2025
""", "https://x.ac.in/placement", "X College"),

"undated fest listing": ("""
Events and Activities
Annual Fest
Cultural Night
Sports Day
Freshers Party
Technical Symposium
Alumni Meet
""", "https://x.ac.in/events", "X College"),

"tender notice board (dated, junk)": ("""
Tender Notices
2026-06-11
Invite quotation for Books for the central library
Download
2026-05-02
Notice for repair and painting of the examination hall
Download
2026-04-18
Walk-in interview for the post of Assistant Professor
Download
""", "https://x.ac.in/notice_board", "X College"),

"tender URL beats content": ("""
Notice Board
2026-06-11
Admission Notification for the B.Tech programme 2026-27
Download
2026-05-02
Notification for the semester examination schedule 2026
Download
""", "https://x.ac.in/tenders/notice", "X College"),

"labelled deadline sentence": ("""
Admission 2026
The last date for submission of the online application form is 30 June 2026.
Counselling for the first merit list will be held on 15/07/2026 at the campus.
Candidates should note that the convocation will be held on 5 March 2026.
""", "https://x.ac.in/admission-notices", "X College"),

"stale archive (2018 notices)": ("""
Notice Archive
12 June, 2018
Admission Notification for B.A. Part 1 session 2018-21
View
04 May, 2018
Merit list for admission to B.Sc. Mathematics Part 1
View
21 March, 2018
Notification for the annual examination results 2018
View
""", "https://x.ac.in/notice_board", "X College"),

"malformed / hostile": ("\x00\x01" + "a" * 5000 + "\n" + "12/13/9999\n" * 40 +
                        "31/02/2026\n" * 20 + "😀" * 10,
                        "https://x.ac.in/events", "X College"),
}


# What each adversarial page must do. "none" = nothing at all; "dropped" =
# None or a confidence the caller discards (<0.5); "kept" = trusted without an
# LLM call (>=0.6, web_enrich's DETERMINISTIC_MIN_CONFIDENCE). Distinguishing
# "none" from "dropped" matters: a stale archive SHOULD still parse, it just
# must not reach a student.
EXPECT = {"labelled deadline sentence": "kept",
          "stale archive (2018 notices)": "dropped"}


def _run(label: str, case, expect: str | None = None) -> bool:
    import time
    text, url, name = case
    t0 = time.perf_counter()
    got = extract(text, url, name)
    ms = (time.perf_counter() - t0) * 1000
    conf = got["confidence"] if got else 0.0
    ok = True
    if expect == "none":
        ok = got is None
    elif expect == "dropped":
        ok = got is None or conf < 0.5
    elif expect == "kept":
        ok = got is not None and conf >= 0.6
    print(f"\n{'-' * 74}\n{label}\n  chars={len(text)}  {ms:.2f} ms"
          + ("" if expect is None else f"  want={expect} [{'PASS' if ok else 'FAIL'}]"))
    if got is None:
        print("  -> None")
    else:
        print(f"  -> confidence {got['confidence']}")
        print(f"     summary: {got['summary']}")
        for k, v in got["facts"].items():
            print(f"       {k}\n         = {v}")
    return ok


# Strings lifted from the live pages that a naive date regex eats. Every one of
# these must yield NO date; getting this wrong is how a phone number becomes a
# counselling deadline.
NOT_DATES = [
    "+91-135-2471102 / 2471140", "0612-22677619", "0172-2754400",
    "+91 9230994080 | 81 | 82", "Dehradun - 248016, Uttarakhand",
    "Patna- 800004, Bihar", "Tripura - 799004", "Visitor Counter : 350646",
    "UGC Act. 1956", "The Tripura Act No. 4 of 2023", "(1849-1898 AD)",
    "Academic Session-2026-27", "Biotechnology (2019-2020)",
    "Admission 2019-22 UG Notice Part-I", "15,000 INR total 4,15,000 INR",
    "Highest package 45 LPA", "Course code CS-12-2026",
    "Affiliation No. 1-4703261", "Examinations- July 2026",
    "November-2025 schedule", "31/02/2026", "12/13/9999",
    "8 semesters 4 years", "Rs 1,20,000 per annum", "2026-27",
]

# Every date format observed across the six sites, plus the ISO normalisation
# each must produce. DD/MM is assumed when both fields could be a month.
REAL_DATES = [
    ("21 July, 2026", "2026-07-21"), ("2026-07-14", "2026-07-14"),
    ("Jun 9, 2026", "2026-06-09"), ("10-Oct, 2026", "2026-10-10"),
    ("Jul 27 2026", "2026-07-27"), ("07/10/2024", "2024-10-07"),
    ("04-March-2025", "2025-03-04"), ("7th March 2025", "2025-03-07"),
    ("15th June, 2026", "2026-06-15"), ("3 August 2026", "2026-08-03"),
    ("30/06/26", "2026-06-30"), ("11/04/2025", "2025-04-11"),
    ("December 31, 2024", "2024-12-31"), ("25-February-2025", "2025-02-25"),
]


def _date_tests() -> list[str]:
    bad = []
    for s in NOT_DATES:
        got = _find_dates(s)
        if got:
            bad.append(f"read a date in {s!r}: {[g[3] for g in got]}")
    for s, want in REAL_DATES:
        got = _find_dates(s)
        if not got or got[0][2].isoformat() != want:
            bad.append(f"{s!r} -> {[g[2].isoformat() for g in got] or None}, want {want}")
    print(f"  not-a-date strings rejected : "
          f"{len(NOT_DATES) - sum(1 for b in bad if 'read a date' in b)}/{len(NOT_DATES)}")
    print(f"  real dates parsed correctly : "
          f"{len(REAL_DATES) - sum(1 for b in bad if 'read a date' not in b)}/{len(REAL_DATES)}")
    for b in bad:
        print(f"  !! {b}")
    return bad


if __name__ == "__main__":
    import sys

    print("=" * 74)
    print("DATE PARSER (strings taken off the live pages)")
    print("=" * 74)
    date_fails = _date_tests()

    print("\n" + "=" * 74)
    print("REAL PAGES (text as produced by web_enrich.parse_html)")
    print("=" * 74)
    for label, case in REAL.items():
        _run(label, case)

    print("\n" + "=" * 74)
    print("ADVERSARIAL - calibration cases")
    print("=" * 74)
    fails = []
    for label, case in ADVERSARIAL.items():
        if not _run(label, case, expect=EXPECT.get(label, "none")):
            fails.append(label)

    # Throughput on a realistic page, the number that decides whether this is
    # free relative to a 20s fetch.
    import time
    page = (REAL["srhu.edu.in/admission-notices (title-then-date)"][0] * 4)[:6000]
    t0 = time.perf_counter()
    for _ in range(200):
        extract(page, "https://srhu.edu.in/admission-notices/", "Swami Rama Himalayan University")
    per = (time.perf_counter() - t0) / 200 * 1000
    print(f"\n{'=' * 74}\n6,000-char page: {per:.2f} ms/call  (budget 50 ms)")

    # Optional: run over raw parse_html dumps in a directory.
    if len(sys.argv) > 1:
        import os
        d = sys.argv[1]
        print(f"\n{'=' * 74}\nFULL PAGE DUMPS in {d}\n{'=' * 74}")
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".txt"):
                continue
            with open(os.path.join(d, fn), encoding="utf-8") as fh:
                _run(fn, (fh.read(), "https://" + fn.split("__")[0].replace("_", "."), "College"))

    print(f"\n{'=' * 74}")
    print(f"date-parser failures: {len(date_fails)}")
    print("page failures: " + (", ".join(fails) if fails else "none"))
    sys.exit(1 if (fails or date_fails) else 0)
