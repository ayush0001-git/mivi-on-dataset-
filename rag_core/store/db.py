"""The canonical retrieval card, plus Postgres-backed reads of the corpus.

This module used to own SQLite: connections, PRAGMAs, schema bootstrap. It owns
none of that now — there is one database, one pool, and it lives in
`store.backend`. What is left here is the part that was never about storage.

`render_card` is the single definition of "what the model is allowed to see
about a college". The ETL stores it, the dense index embeds it, and retrieval
returns it — one function, so those three can never drift apart. Its currency
rules are load-bearing, not cosmetic: an overseas listing must never render a
rupee figure, because a student comparing it against a rupee budget would be
comparing 30,000 GBP to 30,000 INR.
"""
from __future__ import annotations

import html
import json
import re
from datetime import date as _date

from .backend import DEFAULT_PG_DSN, get_backend


def _loads(v, fallback):
    """Accept a JSON column whether it arrives decoded or as text.

    Postgres hands back JSONB already parsed (`course_titles` is a real list),
    while the same value read from a CSV row or an older dump is a string. The
    isinstance check is not defensive noise: without it every JSONB column
    silently fell through to the fallback, and a card lost its entire course
    list, level list and exam list without raising anything.
    """
    if not v:
        return fallback
    if isinstance(v, (list, dict)):
        return v
    try:
        return json.loads(v)
    except (TypeError, ValueError):
        return fallback


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


# Mojibake from the source: UTF-8 bytes that were decoded as CP1252 somewhere
# upstream, so a curly quote arrives as "â€™" and an en-dash as "â€“". Visible
# to students ("âInstitute of Eminence'"), and cheap to repair here rather than
# asking the model to read around it.
_MOJIBAKE = (
    ("â€™", "'"), ("â€˜", "'"), ("â€œ", '"'), ("â€", '"'),
    ("â€“", "-"), ("â€”", "-"), ("â€¦", "..."), ("Â ", " "), ("Â", ""),
    ("â", "'"),   # bare 'â' before a quote is the same truncation
)


def clean_prose(text: str | None) -> str:
    """Strip the HTML that ~0.5% of college overviews carry.

    The `details`/`aboutCollege` fields are authored in a rich-text admin, so a
    minority arrive as `<p>...</p>` with `&nbsp;` entities. That is a small
    share of 38k colleges but a visibly broken answer for every student who
    hits one — the generator will happily echo markup into the chat.
    """
    if not text:
        return ""
    text = _TAG_RE.sub(" ", str(text))
    text = html.unescape(text)
    for bad, good in _MOJIBAKE:
        if bad in text:
            text = text.replace(bad, good)
    return _WS_RE.sub(" ", text).strip()


def _rupees(n) -> str:
    return f"Rs {int(n):,}"


def _fee_phrase(lo, hi, is_abroad: bool) -> str:
    """Fee text that never claims a period the data does not state.

    It used to say "per year". That was wrong, and provably so: checked against
    Maharashtra's Fee Regulating Authority (the legally approved ANNUAL fee),
    the catalogue figure divided by the approved fee tracks course length —
    2.00 for two-year MBAs, 3.55 for four-year BTechs across 564 pairs. These
    are whole-programme fees. The source never stated a period at all; "per
    year" was an assumption in the ETL that reached every answer MIVI gave.

    Course duration is recorded on 10 of 211,071 rows, so the annual figure
    cannot be computed. Saying "for the full programme" is therefore both the
    accurate reading and the only one the data supports — and it errs the safe
    way: a student who thinks a course costs less per year than it does will
    check, whereas one told it costs four times more simply walks away.
    """
    if is_abroad:
        return (
            "Tuition: listed in the institution's local currency — not directly "
            "comparable to an Indian rupee budget."
        )
    if not lo and not hi:
        return "Tuition: not recorded."
    span = (f"{_rupees(lo)}–{_rupees(hi)}" if (lo and hi and lo != hi)
            else _rupees(hi or lo))
    return (f"Tuition: {span} for the full programme across its courses "
            f"(the source does not state a per-year figure; do NOT present this "
            f"as an annual fee).")


# ---------------------------------------------------------------------------
# Is this token an entrance exam, a school board, or neither?
#
# Lives here rather than in sources/ because BOTH the catalogue renderer below
# and sources/admission_ingest.py need the same answer, and sources already
# depends on store (never the other way round).
#
# It is needed because the catalogue's `entrance_exams` field is mislabelled at
# source. Measured over 95,048 tokens across 29,951 colleges, its most common
# values are CBSE 12th (20,760), ISC (20,219), UP 12th, Karnataka 2nd PUC, GSEB
# HSC and ICSE — school boards, not exams. Rendering them under "Entrance exams
# accepted" told a student their own board was an entrance exam, which is the
# opposite of useful when they are asking which exam to sit.
# ---------------------------------------------------------------------------

_BOARD_HINT = re.compile(
    r"\b(cbse|icse|isc|nios|hsc|puc|hsslc|ahsec|wbchse|bseb|rbse|hbse|cgbse"
    r"|mpbse|upmsp|pseb|gseb|kseeb|tnbse|bieap|tsbie|jkbose|cisce"
    r"|hssc|chse|bse\b|dge|pue|mbose|nbse"
    r"|board|10th|12th|xii|matric|intermediate|senior\s*secondary"
    r"|higher\s*secondary|state\s*board|equivalent|diploma|graduation"
    r"|bachelor|degree)\b", re.I)

_EXAM_NAMES = re.compile(
    r"\b(jee|neet|gate|cat|mat|xat|cmat|cuet|clat|nift|nata|bitsat|viteee"
    r"|srmjeee|comedk|kcet|eamcet|emcet|cet|wbjee|kiitee|net|gpat|ielts"
    r"|toefl|pte|duolingo|gmat|gre|sat|nmat|snap|iift|tissnet|ugat|aiapget"
    r"|inicet|jexpo|reap|ubter|jeep|jenpas|hpcet|dcece|atma|bcece|gujcet"
    r"|hstes|uptac|aieea)\b", re.I)
_EXAM_WORDS = re.compile(
    r"entrance|counsell?ing|admission\s*(test|process)|selection\s*test"
    r"|\bexams?\b|\btest\b", re.I)
_EXAM_ACRONYM = re.compile(r"^[A-Z]{3,8}(\s+[A-Z]{2,8})?$")

# Placeholders and column headers the source DB emits. Matched by CONTAINMENT
# for the placeholder family, because it appears as "Course Common", "Course Name
# Common" and inside longer strings — an exact-match rule let every variant but
# one through, and the placeholder reached students as an accepted exam.
_JUNK_CONTAINS = re.compile(r"courses?(\s+name)?\s*common|common\s+course", re.I)
_JUNK_EXACT = re.compile(
    r"^(common|others?|n\.?a\.?|none|nil|any|all|test|default|tbd|xxx+|-+|\.+"
    r"|merit[\s-]?based|merit|direct|management\s*quota|lateral"
    r"|courses?(\s+name)?|marks?|percentage|interview"
    r"|[\d\s.,%/-]+)$", re.I)


def token_is_junk(token: str) -> bool:
    t = " ".join(str(token or "").split())
    return not t or bool(_JUNK_EXACT.match(t) or _JUNK_CONTAINS.search(t))


def split_exams_and_boards(tokens) -> tuple[list[str], list[str]]:
    """(exams, boards) from one mixed list, junk discarded.

    Boards are tested BEFORE the acronym rule, so WBCHSE and AHSEC stay boards
    rather than being read as exam acronyms. A token matching neither is dropped:
    an unrecognised value is not evidence of anything, and a wrong label on a
    real-looking value is worse than leaving it out.
    """
    exams: list[str] = []
    boards: list[str] = []
    for raw in (tokens or []):
        t = " ".join(str(raw or "").split())
        if not (2 <= len(t) <= 60) or token_is_junk(t):
            continue
        if _EXAM_NAMES.search(t):
            exams.append(t)
        elif _BOARD_HINT.search(t):
            boards.append(t)
        elif _EXAM_WORDS.search(t) or _EXAM_ACRONYM.match(t):
            exams.append(t)
    # Preserve order, drop repeats.
    return (list(dict.fromkeys(exams)), list(dict.fromkeys(boards)))


def render_registry_lines(reg: list | None) -> list[str]:
    """NIRF rankings as citable prose.

    A separate helper because these are a DIFFERENT class of fact from both the
    catalogue and the website harvest: published by a government body, about
    the college rather than by it. The catalogue has a `nirf_ranking` column
    populated on exactly 1 of 38,700 colleges, so for anyone asking "which is
    the best engineering college" this is the only real ranking evidence in the
    system — and it is worth naming the source and year, because a 2024 rank
    stated bare will still be quoted as current in 2027.
    """
    if not reg:
        return []
    out = []
    for r in reg:
        payload = r.get("payload")
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except (TypeError, ValueError):
                payload = {}
        payload = payload or {}
        cat, year = r.get("category"), r.get("year")

        # State Fee Regulating Authority: the legally APPROVED fee, which is a
        # different and stronger claim than a fee scraped off a page — but it is
        # dated, and the year is not optional. Maharashtra's latest published
        # order is 2024-25 while fees are revised annually, so a bare figure
        # here would be quoted as current and be wrong by two cycles.
        approved = payload.get("approved_tuition_inr")
        total = payload.get("approved_total_inr")
        if approved or total:
            amt = _rupees(approved or total)
            what = "tuition" if approved else "total fees"
            line = (f"Fee Regulating Authority ({payload.get('district') or 'state'}) "
                    f"approved {what} for {year}: {amt} per year")
            if approved and total and total != approved:
                line += f" ({_rupees(total)} including development fees)"
            if payload.get("stream"):
                line += f" [{payload['stream']}]"
            order = payload.get("order_date")
            line += (f". Order dated {order}. This is the approved figure for "
                     f"{year}, NOT necessarily the current year — say so and tell "
                     f"the student to confirm with the college.")
            out.append(line)
            continue

        # NAAC accreditation. The catalogue's own naac_grade column is empty on
        # 38,698 of 38,700 rows, so without this MIVI cannot answer one of the
        # three questions students ask most. The declaration date is not
        # decoration: a NAAC cycle runs about five years, so a grade declared in
        # 2019 may have been re-assessed, and a bare "A+" would be read as
        # today's grade.
        grade = payload.get("grade")
        if grade:
            line = f"NAAC accreditation grade {grade}"
            if payload.get("cgpa"):
                line += f" (CGPA {payload['cgpa']}/4.00)"
            if payload.get("cycle"):
                line += f", cycle {payload['cycle']}"
            declared = payload.get("declared")
            if declared:
                line += f", declared {declared}"
            line += (f". From NAAC's list of valid accreditations as on "
                     f"{payload.get('as_on') or year}; a NAAC cycle lasts about "
                     f"five years, so give the date and tell the student to "
                     f"confirm the current grade with the college.")
            out.append(line)
            continue

        rank, score = payload.get("rank"), payload.get("score")
        if not rank:
            continue
        line = f"NIRF {year} rank #{rank} in {cat}"
        if score:
            line += f" (score {score}/100)"
        # Graduation Outcomes is the closest public proxy to placement, which the
        # catalogue lacks entirely — worth surfacing, but named for what it is
        # rather than passed off as a placement rate.
        go = (payload.get("sub_scores") or {}).get("GO")
        if go:
            line += f"; Graduation Outcomes sub-score {go}/100"
        out.append(line + ".")
    return out


def render_card(college: dict, agg: dict | None = None, facts: dict | None = None,
                web_facts: dict | None = None, registry: list | None = None,
                admission: list | None = None) -> str:
    """The single retrieval unit: one college rendered as grounded prose.

    One college = one card (not one course) because students ask about
    institutions — "which colleges offer BTech under 2 lakh in Dehradun" — and
    a college-level chunk keeps every citation resolvable to a real page on
    makemyeducation.com/college/<id>. Course detail is folded in as a rollup.

    Only recorded values are written. A missing NAAC grade produces no NAAC
    sentence at all, rather than "NAAC: unknown" — absent text cannot be
    misread by the generator as a stated fact.
    """
    agg = agg or {}
    facts = facts or {}
    is_abroad = bool(college.get("is_abroad"))

    where = ", ".join(
        p for p in (college.get("city"), college.get("district"), college.get("state"))
        if p and str(p).strip()
    )
    # district often repeats city — collapse the duplicate rather than print it twice
    seen, parts = set(), []
    for p in where.split(", "):
        if p and p.lower() not in seen:
            seen.add(p.lower())
            parts.append(p)
    where = ", ".join(parts)

    head = f"{college['name']} (id {college['college_id']})"
    bits = [b for b in (college.get("type"), f"in {where}" if where else "") if b]
    lines = [f"{head} — {' institution '.join(bits) if len(bits) > 1 else (bits[0] if bits else '')}".rstrip(" —")]

    if college.get("establish_year"):
        lines.append(f"Established: {college['establish_year']}.")
    if college.get("affiliated_university"):
        lines.append(f"Affiliated to: {college['affiliated_university']}.")
    if college.get("naac_grade"):
        lines.append(f"NAAC grade: {college['naac_grade']}.")
    if college.get("nirf_ranking"):
        lines.append(f"NIRF ranking: {college['nirf_ranking']}.")

    # The dated fee held in the database itself (sources/fee_promote.py). Placed
    # here, above the catalogue rollup, because when a legally approved annual
    # figure exists it is the number a student should plan around — and unlike
    # the catalogue's, it carries the year it applies to.
    vf = college.get("verified_fee_inr")
    if vf:
        per = college.get("verified_fee_period") or "unspecified"
        per_txt = {"year": "per year", "semester": "per semester"}.get(
            per, "period not stated on the source")
        yr = college.get("verified_fee_year") or "an unstated year"
        basis = college.get("verified_fee_basis")
        who = ("the state Fee Regulating Authority (legally approved)"
               if basis == "fra_approved" else "the college's own website")
        line = (f"Verified fee on record: {_rupees(vf)} {per_txt}, for {yr}, "
                f"from {who}")
        if college.get("verified_fee_label"):
            line += f" — {college['verified_fee_label']}"
        try:
            age = _date.today().year - int(str(yr)[:4])
        except (TypeError, ValueError):
            age = 0
        if age >= 2:
            line += (f". This is {age} years old and fees are revised annually, "
                     f"so quote it as the last approved figure, not the current "
                     f"one")
        lines.append(line + ".")

    # Government-published rankings, matched to this college by
    # rag_core/sources/registries.py. Placed here with the other verified
    # identity facts rather than under the website-harvest heading: NIRF is a
    # public registry, not the college's own marketing.
    lines.extend(render_registry_lines(registry))

    titles = _loads(agg.get("course_titles"), [])
    levels = _loads(agg.get("program_levels"), [])
    if titles:
        shown = titles[:28]
        more = f" (+{len(titles) - len(shown)} more)" if len(titles) > len(shown) else ""
        lines.append(f"Courses offered ({agg.get('n_courses', len(titles))}): "
                     f"{'; '.join(shown)}{more}.")
    if levels:
        lines.append(f"Levels: {', '.join(levels)}.")

    lines.append(_fee_phrase(agg.get("min_tuition_inr"), agg.get("max_tuition_inr"), is_abroad))

    if agg.get("has_hostel"):
        # Same currency trap as tuition: a hostel charge on an overseas listing
        # is not rupees, so the amount is withheld rather than mislabelled.
        h = agg.get("min_hostel_inr")
        cost = f" from {_rupees(h)} per year" if h and not is_abroad else ""
        lines.append(f"Hostel: available{cost}.")
    # The source field is called entrance_exams but holds boards and exams mixed
    # together, so it is split before rendering rather than printed under one
    # label. See split_exams_and_boards.
    exams, boards = split_exams_and_boards(_loads(agg.get("entrance_exams"), []))
    if exams:
        lines.append(f"Entrance exams accepted: {', '.join(exams[:10])}.")
    if boards:
        lines.append(f"School qualifications accepted: {', '.join(boards[:10])}.")

    for kind, label in (("placement", "Placements"), ("scholarship", "Scholarships"),
                        ("accommodation", "Accommodation"), ("facility", "Facilities")):
        text = facts.get(kind)
        if text:
            lines.append(f"{label}: {str(text)[:400]}")

    overview = clean_prose(college.get("details") or college.get("about"))
    if overview:
        lines.append(f"About: {overview[:900]}")

    # Facts read off the college's OWN website, kept visibly separate from the
    # catalogue above. The attribution is not decoration: the generator is told
    # to repeat the "college's own website" framing, so a student can weigh a
    # scraped scholarship figure differently from a verified tuition figure.
    # Low-confidence extractions are dropped entirely rather than hedged —
    # a vague fact in a counselling answer is worse than no fact.
    if web_facts:
        rendered = []
        for kind in ("scholarship", "fees", "cutoff", "hostel", "placement",
                     "admission", "events"):
            item = web_facts.get(kind)
            if not item:
                continue
            summary = str(item.get("summary") or "").strip()
            if not summary or float(item.get("confidence") or 0) < 0.5:
                continue
            line = f"  - {kind.capitalize()}: {summary[:320]}"
            # The year the PAGE says the figures are for — a different fact from
            # the fetch date below, and the one that decides whether a fee is
            # usable. A "Fee Structure 2019-20" table fetched this month is a
            # 2019 fee, and saying only "checked 2026-07" invites the student to
            # read it as current. Age is spelled out rather than left as
            # arithmetic on two dates.
            year = str(item.get("stated_year") or "").strip()
            if year:
                head = year[:4]
                age = _date.today().year - int(head) if head.isdigit() else 0
                stale = (" — OUT OF DATE, treat as indicative only and tell the "
                         "student to get the current figure from the college"
                         if age >= 2 else "")
                line += f" [the page states these are for {year}{stale}]"
            elif kind in ("fees", "scholarship", "cutoff", "hostel"):
                # Absence is informative here and must not be silent: an undated
                # figure cannot be aged, so it cannot be trusted as current.
                line += " [the page gives no year for these figures — undated]"
            rendered.append(line)
        if rendered:
            when = ""
            for kind in ("scholarship", "fees", "cutoff", "hostel", "placement",
                         "admission", "events"):
                if web_facts.get(kind, {}).get("fetched_at"):
                    when = f" (checked {str(web_facts[kind]['fetched_at'])[:10]})"
                    break
            lines.append(
                f"FROM THE COLLEGE'S OWN WEBSITE{when} — not verified by "
                f"MakeMyEducation; tell the student to confirm on the official site:")
            lines.extend(rendered)

    lines.extend(render_admission_lines(admission))

    if is_abroad:
        lines.append(f"Location note: this is an overseas institution "
                     f"({college.get('country') or 'outside India'}).")

    return "\n".join(l for l in lines if l.strip())


# Rendered per course, capped: a college with 40 programmes would otherwise
# push everything else out of the generator's context window.
MAX_ADMISSION_COURSES = 8


def render_admission_lines(admission: list | None) -> list[str]:
    """MakeMyEducation's own admission records, from `college_admission`.

    Kept as its own block with its own attribution because it is a third tier:
    not the course catalogue, not scraped from the college's site, but the
    client's records — which are patchy enough that a student must be told to
    confirm. See sources/admission_ingest.py for what was thrown away to get
    here (72% of the source eligibility field was website navigation text).

    Deliberately NOT rendered: admissionStatus and admissionMode. They read
    "Open" and "Merit Based" on essentially every row in the country, so
    printing them would manufacture a fact out of a schema default — and
    "admissions are open" is precisely the claim a student would act on.
    """
    if not admission:
        return []

    by_course: dict[str, dict[str, dict]] = {}
    for row in admission:
        kind = row.get("kind")
        payload = row.get("payload")
        if isinstance(payload, str):
            payload = _loads(payload, None)
        if not kind or not isinstance(payload, dict):
            continue
        by_course.setdefault(str(row.get("course_title") or "").strip(),
                             {})[kind] = payload

    out: list[str] = []
    # Courses carrying a qualifying mark first: it is the fact a student can
    # actually check themselves against.
    order = sorted(by_course.items(),
                   key=lambda kv: (0 if "min_marks" in kv[1] else 1, kv[0]))
    for title, kinds in order[:MAX_ADMISSION_COURSES]:
        bits: list[str] = []
        mm = kinds.get("min_marks")
        if mm and mm.get("min_percent") is not None:
            stage = mm.get("stage") or "qualifying exam"
            bits.append(f"minimum {mm['min_percent']:g}% in {stage}")
        af = kinds.get("accepted_from") or {}
        if af.get("exams"):
            bits.append("accepts " + ", ".join(af["exams"][:6]))
        if af.get("boards"):
            bits.append("boards: " + ", ".join(af["boards"][:6]))
        q = kinds.get("qualification") or {}
        if q.get("text"):
            bits.append(str(q["text"])[:200])
        d = kinds.get("documents") or {}
        if d.get("text"):
            bits.append("documents: " + str(d["text"])[:160])
        if bits:
            label = title or "General"
            out.append(f"  - {label}: {'; '.join(bits)}.")

    if not out:
        return []
    more = len(by_course) - len(out)
    head = ("ADMISSION CRITERIA on record at MakeMyEducation — incomplete for "
            "many colleges; tell the student to confirm with the college:")
    if more > 0:
        head = head[:-1] + f" (+{more} more programme(s) on record):"
    return [head, *out]


# ---------------------------------------------------------------------------
# Corpus reads
# ---------------------------------------------------------------------------

def fetch_colleges(ids, _legacy_ids=None) -> dict[str, dict]:
    """Hydrate college rows + aggregates for a retrieved id list, in one round
    trip. Retrieval returns ids; this turns them into citable records.

    The old signature was `fetch_colleges(conn, ids)`; there is no connection to
    pass any more, but app.py still calls it the old way, so a second positional
    argument is read as the id list and the first is discarded.
    """
    if _legacy_ids is not None:
        ids = _legacy_ids
    ids = list(ids or [])
    if not ids:
        return {}
    qs = ",".join("?" * len(ids))
    rows = get_backend().q(
        f"""SELECT c.*, a.n_courses, a.program_levels, a.course_titles,
                   a.departments, a.entrance_exams, a.min_tuition_inr,
                   a.max_tuition_inr, a.min_hostel_inr, a.has_hostel,
                   a.has_scholarship, a.card
            FROM colleges c LEFT JOIN college_agg a USING(college_id)
            WHERE c.college_id IN ({qs})""",
        ids,
    )
    return {r["college_id"]: dict(r) for r in rows}


def _drop_legacy_conn(args: tuple) -> tuple:
    """Callers written against the SQLite API pass a connection first. There is
    one pool now, so it is dropped rather than honoured — quietly, because the
    alternative is editing four modules this migration does not own."""
    if args and not isinstance(args[0], str):
        return args[1:]
    return args


def get_meta(*args):
    """build_meta lookup. `get_meta(key[, default])`, or the legacy
    `get_meta(conn, key[, default])`."""
    args = _drop_legacy_conn(args)
    if not args:
        raise TypeError("get_meta() needs a key")
    key = args[0]
    default = args[1] if len(args) > 1 else None
    value = get_backend().one("SELECT value FROM build_meta WHERE key = ?", [key])
    return default if value is None else value


def set_meta(*args) -> None:
    """`set_meta(key, value)`, or the legacy `set_meta(conn, key, value)`."""
    args = _drop_legacy_conn(args)
    if len(args) < 2:
        raise TypeError("set_meta() needs a key and a value")
    key, value = args[0], args[1]
    get_backend().execute(
        "INSERT INTO build_meta(key, value) VALUES(?, ?) "
        "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
        [key, str(value)],
    )


# ---------------------------------------------------------------------------
# Legacy names, kept alive deliberately
#
# `DB_FILE` and `get_conn()` should have died with SQLite. They are still here
# because retrieve.py and app.py were out of scope for this migration and are
# not being edited, and both still ask the store two questions in SQLite's
# vocabulary: "does the DB file exist?" and "which tables does it have?"
# (retrieve.py `_get_conn`, app.py `_corpus_probe`).
#
# Deleting the names would do more than break an import. retrieve.warm() calls
# `_get_conn()` inside a try/except and returns early on failure, so a broken
# probe would silently skip the model + index preload and hand the cold start
# to the first student of the day, while logging a "store unavailable" that is
# not true.
#
# So: nothing below opens a file. These are three thin adapters over the same
# pool everything else uses. They should be deleted together with those two
# call sites, which is a ~10-line change in each.
# ---------------------------------------------------------------------------

_PG_CATALOG = ("SELECT table_name AS name FROM information_schema.tables "
               "WHERE table_schema = 'public'")


class _Rows(list):
    """sqlite3 cursors are iterable AND answer fetchone()/fetchall(); the two
    call sites above use both shapes."""

    def fetchone(self):
        return self[0] if self else None

    def fetchall(self) -> list:
        return list(self)


class _Store:
    """What `DB_FILE` and `get_conn()` now return."""

    @property
    def name(self) -> str:
        """The database name, and only that. /api/health publishes this field
        publicly; it used to be a filename precisely so the server's layout was
        not exposed, and a DSN with a host, port and password would be a far
        worse leak than the path ever was."""
        return DEFAULT_PG_DSN.rsplit("/", 1)[-1].split("?")[0] or "postgres"

    def exists(self) -> bool:
        """Is there a corpus to serve?

        Never raises. app.py reads this while assembling its health payload,
        outside its own try block — a health endpoint that 500s when the
        database is down is a health endpoint that reports nothing at the exact
        moment someone needs it to.
        """
        try:
            return bool(get_backend().one(
                "SELECT to_regclass('public.colleges') IS NOT NULL"))
        except Exception:  # noqa: BLE001
            return False

    def execute(self, sql: str, params=()) -> _Rows:
        # retrieve.py's boot check asks SQLite's catalog which tables exist.
        # The question is still exactly right — refusing to serve a half-loaded
        # corpus is a hard invariant — so it is answered, in Postgres's terms.
        if "sqlite_master" in sql:
            sql, params = _PG_CATALOG, ()
        return _Rows(get_backend().q(sql, params))

    def commit(self) -> None:
        """No-op: every write commits inside backend.execute()."""

    def close(self) -> None:
        """No-op: connections belong to the pool, not to the caller."""

    def __str__(self) -> str:
        # Reads sanely inside the messages these callers build, e.g.
        # "MIVI store not found at postgres:postgres".
        return f"postgres:{self.name}"

    __repr__ = __str__


_STORE = _Store()
DB_FILE = _STORE


def get_conn(*_args, **_kwargs) -> _Store:
    """Deprecated. There are no per-thread connections any more, only the pool;
    this returns a handle that answers the few questions old callers ask."""
    return _STORE
