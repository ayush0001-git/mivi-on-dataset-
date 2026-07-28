"""Government registries: the tier-2 source that does not need a college website.

WHY THIS IS THE ONLY REMAINING LEVER
The website harvest is capped by arithmetic, not effort. Measured on the real
corpus: 62% of the colleges it reaches come back with nothing usable, and a
purpose-built page-discovery pass (rag_core/sources/pagefind.py) moved that
number by almost zero — 16/40 before, 16/40 after — because those colleges
simply do not publish fees or placements in parseable form. Worse, 28,491
colleges have no known website at all, so no crawler can ever reach them.

NIRF is different in the way that matters: it is keyed on the INSTITUTION, not
on whether that institution runs a good website. A college with no site at all
still appears in the rankings, and the data is a plain HTML table published for
public use — no login, no paywall, no scraping of a competitor's product.

WHAT NIRF ACTUALLY GIVES (verified against the live 2024 pages)
    Institute ID   IR-E-U-0456        stable across years, which makes it a
                                      real join key rather than a name
    Name, City, State                 independent corroboration of the
                                      catalogue's own city/state
    Score, Rank                       NIRF ranking is 0.0% populated in the
                                      catalogue today
    TLR / RPC / GO / OI / PERCEPTION  the five sub-scores. GO (Graduation
                                      Outcomes) is the closest public proxy to
                                      the placement data the catalogue lacks
    A per-institute PDF               .../pdf/Engineering/IR-E-U-0456.pdf holds
                                      median salary and placed counts; parsing
                                      it needs a PDF dependency this project
                                      does not have, so it is recorded as a
                                      source_url for a later pass rather than
                                      guessed at

THE HARD PART IS MATCHING, NOT FETCHING
NIRF keys on its own ids; the catalogue keys on Mongo ObjectIds. Nothing links
them but the name, and Indian college names vary wildly — "IIT Roorkee",
"Indian Institute of Technology Roorkee", "I.I.T. Roorkee" are one institution,
while "Presidency College" is FOUR different ones in four cities (measured: the
corpus has 364 duplicate names covering 757 rows).

So matching requires the city to agree AND the distinctive name words to
overlap strongly, and it refuses rather than guesses. A wrong match here
attributes one college's national ranking to another — a number that is real,
precise, and about somebody else, which is the hardest kind of error for anyone
downstream to catch.

Run:
    python -m rag_core.sources.registries --nirf 2024 --dry-run
    python -m rag_core.sources.registries --nirf 2024
    python -m rag_core.sources.registries --report
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from .. import config  # noqa: F401 - loads .env before the backend is imported
from ..store.backend import get_backend
from .web_enrich import HEADERS

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "raw" / "registries"

NIRF_BASE = "https://www.nirfindia.org"

# The NIRF disciplines worth pulling. Overall + the ones that map onto what this
# catalogue actually contains; the long tail (Dental, Agriculture…) can be added
# by extending this tuple, since the page shape is identical for all of them.
NIRF_CATEGORIES = (
    "Overall", "University", "Engineering", "Management", "Pharmacy",
    "Medical", "Law", "College", "Architecture", "Research",
)

# Matching thresholds. Deliberately strict — see the module docstring on why a
# wrong match is the worst outcome here.
NAME_SIM_MIN = 0.72        # difflib ratio on normalised names
NAME_SIM_NO_CITY = 0.92    # required when the city does NOT corroborate

# ONLY grammatical filler. This list was originally much longer — it stripped
# "indian", "institute", "technology", "college", "university" — and that turned
# the matcher into a city matcher: "Indian Institute of Technology Roorkee" and
# "Roorkee Institute of Technology" both reduced to {roorkee} and scored a
# perfect 1.00. Measured on the dry run, that mis-assigned IIT Madras to
# University of Madras, IIT Kharagpur to Kharagpur College, and IIT Guwahati to
# Guwahati University.
#
# Those words are exactly what distinguishes an IIT from a local institute of
# the same town, so they stay in. The type words carry real signal here and
# removing them destroyed it.
_STOP = {"the", "of", "and", "for", "in", "at", "a", "an", "&"}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", str(s or "").lower())).strip()


def _sig(s: str) -> set[str]:
    """Distinctive words — what actually identifies an institution."""
    return {w for w in _norm(s).split() if len(w) > 2 and w not in _STOP}


# Institution FAMILIES. This is the check that string similarity cannot do.
#
# "Indian Institute of Technology Roorkee" and "Roorkee Institute of Technology"
# are 78% similar as strings and share their only distinctive token, but one is
# an IIT and the other is a private college in the same town. Likewise IIT
# Madras scored 0.80 against "Hindustan Institute of Technology" purely on the
# shared phrase "Institute of Technology".
#
# These families are a small, closed, nationally-known set, and membership is
# the strongest identity signal available. If NIRF says IIT and the candidate is
# not an IIT, no similarity score should be allowed to override that.
_FAMILIES = (
    ("iit", (r"\bindian institutes? of technology\b", r"\biit\b", r"\bi\.i\.t\b")),
    ("nit", (r"\bnational institutes? of technology\b", r"\bnit\b", r"\bn\.i\.t\b")),
    ("iiit", (r"\bindian institutes? of information technology\b", r"\biiit\b")),
    ("iim", (r"\bindian institutes? of management\b", r"\biim\b", r"\bi\.i\.m\b")),
    ("iiser", (r"\bindian institutes? of science education\b", r"\biiser\b")),
    ("iisc", (r"\bindian institute of science\b(?! education)", r"\biisc\b")),
    ("aiims", (r"\ball india institutes? of medical sciences\b", r"\baiims\b")),
    ("nift", (r"\bnational institutes? of fashion technology\b", r"\bnift\b")),
    ("nlu", (r"\bnational law (?:school|university|institute)\b", r"\bnlu\b")),
    ("bits", (r"\bbirla institute of technology and science\b", r"\bbits\b")),
    ("nid", (r"\bnational institutes? of design\b", r"\bnid\b")),
    ("spa", (r"\bschools? of planning and architecture\b",)),
    ("nimhans", (r"\bnimhans\b", r"\bnational institute of mental health\b")),
    ("pgimer", (r"\bpgimer\b", r"\bpostgraduate institute of medical education\b")),
    ("jipmer", (r"\bjipmer\b",)),
    ("icar", (r"\bindian (?:agricultural|veterinary) research institute\b",)),
)


def family(name: str) -> str | None:
    """Which national institution family this name belongs to, if any."""
    low = " " + _norm(name) + " "
    raw = " " + str(name or "").lower() + " "
    for fam, pats in _FAMILIES:
        for p in pats:
            if re.search(p, low) or re.search(p, raw):
                return fam
    return None


def _acronym(s: str) -> str:
    """IIT / NIT / BITS-style initials, which is how these are usually written."""
    dotted = re.findall(r"\b([a-z])\.", str(s or "").lower())
    if len(dotted) >= 2:
        return "".join(dotted)
    return "".join(w[0] for w in _norm(s).split() if w not in {"of", "and", "the"})


# ---------------------------------------------------------------------------
# NIRF scraping
# ---------------------------------------------------------------------------

_ID_RE = re.compile(r"IR-[A-Z]-[A-Z]-\d+")
_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)
_CELL_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.S)
# The name cell carries nested <div>s (a "More Details" panel and a hidden
# sub-score table). Everything from the first nested <div> onward is chrome, so
# the name is whatever precedes it.
_NAME_CUT_RE = re.compile(r"<div", re.I)


def _text(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


_INNER_TABLE_RE = re.compile(
    r"<table[^>]*>((?:(?!<table)[\s\S])*?)</table>", re.I)


# Every ranking row nests a second <table> inside its name cell (the "More
# Details" panel), and that nested table has its own </tr> — which terminated a
# non-greedy <tr>...</tr> match early, leaving the outer row with ONE cell.
# Result: every discipline except University parsed to zero institutions. The
# University page happened to lack the nesting, which is exactly how a bug like
# this hides. Sub-scores are read out first, then the nested tables removed.


def parse_nirf(html: str, category: str, year: str) -> list[dict]:
    """Rows out of a NIRF ranking page.

    Parsing is anchored on the institute id rather than on cell position: the
    pages carry several tables, so a positional parser silently picks up
    sub-score rows as institutions.
    """
    # Sub-scores are read from the ORIGINAL html (per id), then the nested
    # tables are removed so the outer rows split correctly.
    sub_by_id: dict[str, dict] = {}
    for m in _INNER_TABLE_RE.finditer(html):
        before = html[max(0, m.start() - 1200):m.start()]
        ids = _ID_RE.findall(before)
        if not ids:
            continue
        block = m.group(0)
        heads = [_text(h) for h in re.findall(r"<th[^>]*>(.*?)</th>", block, re.S)]
        vals = [_text(v) for v in re.findall(r"<td[^>]*>(.*?)</td>", block, re.S)]
        row = {re.sub(r"\s*\(\d+\)\s*", "", h).strip(): v
               for h, v in zip(heads, vals) if h and v}
        if row:
            sub_by_id[ids[-1]] = row

    html = _INNER_TABLE_RE.sub("", html, count=0)

    out: list[dict] = []
    for row in _ROW_RE.findall(html):
        cells = _CELL_RE.findall(row)
        if len(cells) < 6:
            continue
        inst_id = _text(cells[0])
        if not _ID_RE.fullmatch(inst_id):
            continue  # a sub-score row, or a header
        name_html = _NAME_CUT_RE.split(cells[1], 1)[0]
        name = _text(name_html)
        city, state = _text(cells[2]), _text(cells[3])
        score, rank = _text(cells[4]), _text(cells[5])
        if not name:
            continue

        sub = sub_by_id.get(inst_id, {})

        out.append({
            "institute_id": inst_id,
            "name": name,
            "city": city,
            "state": state,
            "score": score,
            "rank": rank,
            "category": category,
            "year": year,
            "sub_scores": sub,
            "detail_pdf": (f"{NIRF_BASE}/nirfpdfcdn/{year}/pdf/{category}/{inst_id}.pdf"),
            "source_url": f"{NIRF_BASE}/Rankings/{year}/{category}Ranking.html",
        })
    return out


def fetch_nirf(year: str = "2024", categories=NIRF_CATEGORIES,
               refresh: bool = False) -> list[dict]:
    """All configured NIRF categories, cached on disk.

    Cached hard on purpose: these pages change once a year. Re-fetching them on
    every run would be both pointless and rude to a government server.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with httpx.Client(headers=HEADERS, timeout=40.0, follow_redirects=True,
                      verify=False) as client:
        for cat in categories:
            cache = CACHE_DIR / f"nirf_{year}_{cat}.html"
            if cache.exists() and not refresh:
                html = cache.read_text(encoding="utf-8", errors="replace")
            else:
                url = f"{NIRF_BASE}/Rankings/{year}/{cat}Ranking.html"
                try:
                    r = client.get(url)
                except Exception as exc:  # noqa: BLE001
                    print(f"[nirf] {cat}: {exc}", file=sys.stderr)
                    continue
                if r.status_code != 200 or len(r.text) < 5_000:
                    print(f"[nirf] {cat}: HTTP {r.status_code} "
                          f"({len(r.text)}b) — skipping", file=sys.stderr)
                    continue
                html = r.text
                cache.write_text(html, encoding="utf-8")
                time.sleep(1.5)   # one government host, paced
            got = parse_nirf(html, cat, year)
            print(f"[nirf] {cat:14} {len(got):>4} institutions", file=sys.stderr)
            rows += got
    return rows


# ---------------------------------------------------------------------------
# Matching NIRF -> catalogue
# ---------------------------------------------------------------------------

def _load_catalogue() -> list[dict]:
    return get_backend().q(
        "SELECT college_id, name, city, district, state FROM colleges "
        "WHERE NOT is_abroad")


def _load_place_aliases() -> dict[str, str]:
    """alias -> canonical, from the curated place_alias table.

    The matcher was ignoring this table, and city agreement is what decides
    which floor a candidate must clear (0.72 with it, 0.92 without). So a place
    written two ways was silently costing real matches: NIRF's "New Delhi"
    against a catalogue "Delhi" is the same city, and treating it as a
    disagreement is a normalisation bug, not caution.
    """
    try:
        rows = get_backend().q("SELECT alias, canonical FROM place_alias")
    except Exception:  # noqa: BLE001 - table absent on a store built before it
        return {}
    return {_norm(r["alias"]): _norm(r["canonical"]) for r in rows
            if r.get("alias") and r.get("canonical")}


# Registry-side spellings that place_alias does not carry because they are not
# what students type -- these come from how the registries themselves write a
# place. Kept small and explicit rather than inferred: a wrong equivalence here
# would let one city's college corroborate another's.
_CITY_FORMS = {
    "new delhi": "delhi",
    "gautam budh nagar": "noida",
    "gautam buddha nagar": "noida",
    "greater noida": "noida",
    "trivandrum": "thiruvananthapuram",
    "gurgaon": "gurugram",
    "mysore": "mysuru",
    "pondicherry": "puducherry",
    "kanpur nagar": "kanpur",
    "prayagraj": "allahabad",
    "varanasi cantt": "varanasi",
    # The loosest entry here: Navi Mumbai is its own city, not a spelling of
    # Mumbai. Kept because registries file the same institution under both, and
    # because city agreement only lowers the floor -- the family, token
    # containment, acronym and leading-brand gates still have to pass.
    "navi mumbai": "mumbai",
}


def _place_key(value, aliases: dict[str, str]) -> str:
    """One canonical spelling for a city or district name."""
    v = _norm(value)
    if not v:
        return ""
    v = _CITY_FORMS.get(v, v)
    return aliases.get(v, v)


def _index_by_city(colleges: list[dict],
                   aliases: dict[str, str] | None = None) -> dict[str, list[dict]]:
    """place -> colleges, keyed by BOTH city and district.

    District matters because registries often report the administrative district
    while the catalogue holds the town: NIRF files Shiv Nadar University under
    "Gautam Budh Nagar" and the catalogue calls it Greater Noida. Without the
    district key those rows fall to the no-city path and need 0.92.
    """
    aliases = aliases or {}
    idx: dict[str, list[dict]] = {}
    for c in colleges:
        for field in ("city", "district"):
            key = _place_key(c.get(field), aliases)
            if not key:
                continue
            bucket = idx.setdefault(key, [])
            # A college whose city and district are the same word must not be
            # listed twice — it would be scored twice for no reason.
            if not bucket or bucket[-1] is not c:
                bucket.append(c)
    return idx


def _index_by_token(colleges: list[dict]) -> dict[str, list[dict]]:
    """word -> colleges whose name contains it.

    Without this the no-city fallback compares one NIRF row against all 37,854
    colleges with difflib — 29 million string alignments for 780 rows, which is
    minutes of CPU for an answer that a token intersection reaches in
    milliseconds. Candidates still have to clear the same similarity floor
    afterwards, so this narrows the search without loosening it.
    """
    idx: dict[str, list[dict]] = {}
    for c in colleges:
        for w in _sig(c["name"]):
            idx.setdefault(w, []).append(c)
    return idx


def match_one(entry: dict, by_city: dict, by_token: dict,
              aliases: dict[str, str] | None = None) -> tuple[dict | None, float, str]:
    """Best catalogue match for one NIRF row: (college, score, why).

    Candidates are drawn from the NIRF city first. That is the whole defence
    against the duplicate-name problem — "Presidency College" is four different
    institutions, and only the city separates them, so a name-first search would
    confidently pick the wrong one.
    """
    nirf_city = _place_key(entry.get("city"), aliases or {})
    nirf_sig = _sig(entry["name"])
    nirf_ac = _acronym(entry["name"])

    candidates = list(by_city.get(nirf_city, []))
    city_backed = True
    if not candidates:
        # No city agreement available. Fall back to colleges that share at least
        # one distinctive name word, and demand a far higher similarity — the one
        # signal that would have caught a same-name collision is missing.
        seen = set()
        candidates = []
        for w in nirf_sig:
            for c in by_token.get(w, ()):
                if c["college_id"] not in seen:
                    seen.add(c["college_id"])
                    candidates.append(c)
        city_backed = False

    nirf_fam = family(entry["name"])

    best, best_score, why = None, 0.0, ""
    for c in candidates:
        cat_sig = _sig(c["name"])
        if not nirf_sig or not cat_sig:
            continue
        # FAMILY GATE, before any scoring. An IIT can only match an IIT. This is
        # not a tie-break — it is a hard veto, because it encodes real-world
        # identity that no amount of string similarity can infer.
        cat_fam = family(c["name"])
        if nirf_fam != cat_fam:
            continue

        # TOKEN CONTAINMENT GATE. Every distinctive word in the NIRF name must
        # appear EXACTLY in the candidate. Fuzzy scoring cannot do this job:
        # hand-checking 16 matches found "Birla Institute of Technology" paired
        # with "Birsa Institute of Technology" at 0.97 (one letter apart, two
        # different institutions) and "Gujarat Technological University" with
        # "Gujarat University" at 0.72 (a dropped word that changes which
        # university it is). Both die here.
        #
        # Containment, not equality: catalogue names are routinely longer
        # ("IIT Madras - Indian Institute of Technology"), so extra words on the
        # candidate side are fine. Missing or altered words are not.
        missing = nirf_sig - cat_sig
        if missing:
            # One tolerated miss, and only for a short token — abbreviations and
            # honorifics ("M.", "Dr", "Shri") drop out of catalogue names
            # routinely and are not identity-bearing.
            if len(missing) > 1 or len(next(iter(missing))) > 4:
                continue

        cat_norm = _norm(c["name"])
        # DOTTED ACRONYM GATE. Tokenising destroys the very part that identifies
        # the institution: "S.R.M. Institute of Science and Technology" reduces
        # to {institute, science, technology}, which is contained by
        # "Sathyabama Institute of Science and Technology" — a completely
        # different university that this matched at 0.87. If NIRF spells an
        # acronym, the candidate has to carry it.
        dotted = "".join(re.findall(r"\b([a-z])\.", entry["name"].lower()))
        if len(dotted) >= 2 and dotted not in re.sub(r"[^a-z]", "", cat_norm):
            continue

        # LEADING-BRAND GATE. On Indian college names the first significant word
        # is usually the trust or founder — the brand. A candidate that opens
        # with a distinctive word NIRF never mentions is a different institution
        # that happens to share a generic tail: "Maestro School of Planning and
        # Architecture, Vijayawada" contains every token of the national "School
        # of Planning and Architecture Vijayawada" and is not it.
        cat_words = [w for w in cat_norm.split() if len(w) > 2 and w not in _STOP]
        if cat_words:
            head = cat_words[0]
            # Tolerated when the catalogue simply prefixes the institution's own
            # acronym ("IIT Roorkee - Indian Institute…", "PAU - Punjab
            # Agricultural…"): that is the same name restated, not a new brand.
            is_restatement = head == _acronym(entry["name"]) or head == dotted
            if len(head) > 3 and head not in nirf_sig and not is_restatement:
                continue
        # FULL-NAME similarity is the primary signal, not token overlap. Overlap
        # on small token sets is worthless — a one-word intersection scores 1.00
        # and cheerfully equates "IIT Roorkee" with "Roorkee Institute of
        # Technology", which is how the first run mis-assigned six of the top ten
        # engineering colleges in the country.
        ratio = difflib.SequenceMatcher(None, _norm(entry["name"]),
                                        _norm(c["name"])).ratio()
        overlap = len(nirf_sig & cat_sig) / max(1, len(nirf_sig | cat_sig))
        score = ratio
        # Overlap may only NUDGE a decision the name already supports, and only
        # when both names are substantial enough for the intersection to mean
        # something. It can never carry a match on its own.
        if len(nirf_sig) >= 3 and len(cat_sig) >= 3 and overlap >= 0.8:
            score = max(score, 0.5 * ratio + 0.5 * overlap)
        # An exact acronym hit is strong evidence for Indian institutions, which
        # are usually written "BITS Pilani" rather than in full — but only when
        # the acronym is long enough to be distinctive. Three letters like "gcs"
        # collide constantly.
        if nirf_ac and len(nirf_ac) >= 4 and nirf_ac == _acronym(c["name"]):
            score = max(score, 0.88)
        if score > best_score:
            best, best_score, why = c, score, (
                f"name~{ratio:.2f} overlap~{overlap:.2f} "
                f"city={'yes' if city_backed else 'no'}")

    floor = NAME_SIM_MIN if city_backed else NAME_SIM_NO_CITY
    if best is None or best_score < floor:
        return None, best_score, f"below floor {floor} ({why})"
    return best, best_score, why


# ---------------------------------------------------------------------------
# Persist
# ---------------------------------------------------------------------------

DDL = """
CREATE TABLE IF NOT EXISTS registry_fact (
    college_id    TEXT NOT NULL,
    registry      TEXT NOT NULL,          -- nirf | aicte | ugc
    ref_id        TEXT NOT NULL,          -- the registry's own key, e.g. IR-E-U-0456
    category      TEXT,
    year          TEXT,
    payload       JSONB NOT NULL,
    source_url    TEXT,
    detail_url    TEXT,
    match_score   REAL,
    match_note    TEXT,
    fetched_at    TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (college_id, registry, category, year)
);
CREATE INDEX IF NOT EXISTS idx_regfact_registry ON registry_fact(registry);

-- Rows the matcher REFUSED. Kept because an unmatched national ranking is a
-- lead, not a dead end: with a name-normalisation fix or a manual pairing it
-- becomes usable, and throwing it away would hide how much is being left on
-- the table.
CREATE TABLE IF NOT EXISTS registry_unmatched (
    registry     TEXT NOT NULL,
    ref_id       TEXT NOT NULL,
    category     TEXT,
    year         TEXT,
    name         TEXT,
    city         TEXT,
    state        TEXT,
    best_score   REAL,
    note         TEXT,
    payload      JSONB,
    PRIMARY KEY (registry, ref_id, category, year)
);
"""


def ensure_tables() -> None:
    be = get_backend()
    for stmt in [s for s in DDL.split(";") if s.strip()]:
        be.execute(stmt)


def _now():
    return datetime.now(timezone.utc)


def ingest_nirf(year: str = "2024", dry_run: bool = False,
                refresh: bool = False) -> dict:
    ensure_tables()
    rows = fetch_nirf(year, refresh=refresh)
    if not rows:
        print("[nirf] nothing fetched", file=sys.stderr)
        return {}

    colleges = _load_catalogue()
    aliases = _load_place_aliases()
    by_city = _index_by_city(colleges, aliases)
    by_token = _index_by_token(colleges)
    print(f"[nirf] matching {len(rows)} ranked institutions against "
          f"{len(colleges)} catalogue colleges "
          f"({len(by_city):,} place keys, {len(aliases):,} aliases)",
          file=sys.stderr)

    be = get_backend()
    stats = {"rows": len(rows), "matched": 0, "unmatched": 0}
    samples: list[str] = []

    for e in rows:
        college, score, why = match_one(e, by_city, by_token, aliases)
        payload = {
            "rank": e["rank"], "score": e["score"], "category": e["category"],
            "sub_scores": e["sub_scores"], "nirf_name": e["name"],
            "nirf_city": e["city"], "nirf_state": e["state"],
        }
        if college is None:
            stats["unmatched"] += 1
            if not dry_run:
                be.execute(
                    "INSERT INTO registry_unmatched"
                    "(registry, ref_id, category, year, name, city, state,"
                    " best_score, note, payload) VALUES(?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT (registry, ref_id, category, year) DO UPDATE SET "
                    "best_score=EXCLUDED.best_score, note=EXCLUDED.note",
                    ["nirf", e["institute_id"], e["category"], year, e["name"],
                     e["city"], e["state"], score, why,
                     json.dumps(payload, ensure_ascii=False)])
            continue

        stats["matched"] += 1
        if len(samples) < 8:
            samples.append(f"  #{e['rank']:>4} {e['name'][:44]:46} -> "
                           f"{college['name'][:40]:42} ({score:.2f})")
        if not dry_run:
            be.execute(
                "INSERT INTO registry_fact"
                "(college_id, registry, ref_id, category, year, payload,"
                " source_url, detail_url, match_score, match_note, fetched_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT (college_id, registry, category, year) DO UPDATE SET "
                "payload=EXCLUDED.payload, match_score=EXCLUDED.match_score, "
                "match_note=EXCLUDED.match_note, fetched_at=EXCLUDED.fetched_at",
                [college["college_id"], "nirf", e["institute_id"], e["category"],
                 year, json.dumps(payload, ensure_ascii=False), e["source_url"],
                 e["detail_pdf"], score, why, _now()])

    print(f"\n[nirf] matched {stats['matched']}/{stats['rows']} "
          f"({100 * stats['matched'] / stats['rows']:.1f}%), "
          f"{stats['unmatched']} refused"
          f"{'  (DRY RUN — nothing written)' if dry_run else ''}", file=sys.stderr)
    print("\n".join(samples), file=sys.stderr)
    return stats


def report() -> None:
    ensure_tables()
    be = get_backend()
    print("-- registry_fact --")
    for r in be.q("SELECT registry, category, COUNT(*) n, "
                  "ROUND(AVG(match_score)::numeric,2) s FROM registry_fact "
                  "GROUP BY registry, category ORDER BY n DESC LIMIT 12"):
        print(f"   {r['registry']:6} {str(r['category']):14} {r['n']:>5}  avg match {r['s']}")
    tot = be.one("SELECT COUNT(DISTINCT college_id) FROM registry_fact") or 0
    print(f"\n   distinct colleges gaining a registry fact: {tot:,}")
    print("\n-- refused (leads, not losses) --")
    for r in be.q("SELECT registry, COUNT(*) n, ROUND(AVG(best_score)::numeric,2) s "
                  "FROM registry_unmatched GROUP BY registry"):
        print(f"   {r['registry']:6} {r['n']:>5} refused, avg best score {r['s']}")
    print("\n-- NIRF ranking now available where the catalogue had none --")
    n = be.one("SELECT COUNT(*) FROM colleges c JOIN registry_fact f "
               "USING(college_id) WHERE c.nirf_ranking IS NULL "
               "OR TRIM(c.nirf_ranking) = ''") or 0
    print(f"   {n:,} colleges")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nirf", metavar="YEAR", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--refresh", action="store_true",
                    help="re-fetch instead of using the on-disk cache")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
        return
    if a.nirf:
        ingest_nirf(a.nirf, dry_run=a.dry_run, refresh=a.refresh)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
