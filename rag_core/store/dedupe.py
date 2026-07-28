"""Duplicate and conflict DETECTION for the college catalogue. Reports only.

Nothing in this module writes to `colleges`. That is the whole design premise,
not caution for its own sake: measured against the live corpus, name collision
is overwhelmingly LEGITIMATE. All 364 duplicate-name groups (757 rows) contain
at least two different cities, and not one group has two rows in the same city.
"Presidency College" is four real, separate institutions in Bangalore, Raisen,
Meerut and Alwar. An auto-deduper keyed on the name would have deleted three of
them, and a deleted college is unrecoverable — the client decides, this module
only assembles the evidence.

WHAT THE MEASUREMENTS CHANGED IN THIS DESIGN

1. EXACT NAME IS THE WRONG KEY. It finds 364 groups and zero same-city pairs,
   i.e. zero duplicate candidates. The real duplicates hide behind punctuation
   and encoding variants, which exact matching cannot see:
     "B M College, Barari"          vs "B. M. College, Barari"
     "Government College of Education, Chandigarh" vs "...Education,Chandigarh"
     "Alva's Homoeopathic ..."      vs "Alva\ufffds Homoeopathic ..."  (mojibake)
     "Patna Dental College & Hospital" vs "... College And Hospital"
   Normalising punctuation, `&`/`and` and the U+FFFD replacement character
   raises this to 574 groups / 1201 rows, of which 108 groups DO share a city.
   Those 108 are the actual duplicate candidates; the exact-name view of the
   problem could not see any of them.

2. A SHARED WEBSITE DOMAIN IS MOSTLY NOT A DUPLICATE. 1,037 hosts are shared by
   more than one distinct name, covering 2,736 rows — but the big ones are state
   admission portals (hte.rajasthan.gov.in serves 56 unrelated colleges),
   university umbrellas (lnmu.ac.in for its affiliated colleges), franchise
   chains (Lakme Academy, NIELIT) and social-media placeholders (facebook.com,
   instagram.com). So the host count itself is the filter: a host carrying more
   than MAX_NAMES_PER_HOST distinct names is an umbrella and is worth nothing as
   duplicate evidence, while a host with two names is real evidence. No
   hand-maintained blocklist, so new portals classify themselves.

3. "establish_year BEFORE 1800" IS A 95% FALSE-POSITIVE RULE. It matches 88 rows
   and 84 of them are `is_abroad` institutions with correct founding dates —
   Bologna 1088, Oxford 1096, Cambridge 1209. The floor is therefore applied
   only to Indian rows, which leaves 4. Likewise "year in the future" matches
   nothing at all: max(establish_year) is 2025. Both checks are kept, because a
   later ingest will reintroduce them, but they are reported against the right
   denominator rather than as 88 alarms.

4. THE ADDRESS IS THE ARBITER FOR CITY/STATE, NOT THE MAJORITY VOTE. Counting
   which state a city "usually" sits in flags 79 rows, and most are homonyms of
   genuinely different real places: Bilaspur exists in both Chhattisgarh and
   Himachal Pradesh, Pratapgarh in UP and Rajasthan, Aurangabad in Bihar and
   Maharashtra, Srinagar in J&K and Uttarakhand. Every row carries a full
   `address` (38,700/38,700), and Indian addresses end "..., City, State PIN,
   India" — so the state named in the address TAIL is checked against the state
   column instead. That is evidence rather than a vote, and it finds the one
   provable case the vote also found ("Government Autonomous College,
   Sambalpur": state column Chhattisgarh, address "Sambalpur, Odisha 768001").

WHY FOUR VERDICTS AND NOT TWO
`SUBUNIT_OR_CAMPUS` exists because of what the brief's own example turned out to
be. "IIM Kashipur" does appear twice, and the second row's city is Dehradun, but
that row is *named* "IIM Kashipur - Indian Institute of Management - Dehradun
Campus" and IIM Kashipur really does run a Dehradun campus. It is a satellite
listing, not a stray copy — merging it would delete a real campus. What is
genuinely wrong with it is its `type`: Private, against Government on the parent.
So the pair is reported as a subunit WITH a type conflict, which is the finding
the client can act on. Collapsing that into TRUE_DUPLICATE would have proposed
exactly the destructive merge this module exists to avoid.

KNOWN BLIND SPOTS, so nobody reads a clean report as a clean catalogue
  * ACRONYM vs EXPANSION. "A.B.M. College" and "Abdul Bari Memorial College" are
    one college on one domain, and no amount of string similarity will pair them
    (measured name_sim 0.58). They surface only in the `shared_domain` high tier,
    which is the safety net for this class — reviewing that tier is not optional.
  * TRANSLITERATION. Mahavidyalaya/Mahavidyalay, Shri/Sri/Shree are not folded,
    so those pairs are missed unless a shared domain catches them.
  * The 28,491 rows with no website_url get no domain evidence at all, so a
    duplicate among them is only found if its name normalises identically.
  * Course overlap is weak here (Jaccard 0.14-0.33 on confirmed duplicates,
    because the two rows hold different subsets of one catalogue). It corroborates
    a decision already made on name and place; it never drives one.

Read-only, idempotent, no network, no LLM: it issues two SELECTs and writes only
CSVs, so it is safe to run against the live database while the harvest sweep is
writing. Measured 13.8 s end to end over 38,700 colleges and 211,071 courses,
almost all of it the course scan; two consecutive runs produce byte-identical
CSVs.

    python -m rag_core.store.dedupe --out reports/
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import re
import sys
from difflib import SequenceMatcher

from .backend import get_backend

# ---------------------------------------------------------------------------
# Thresholds, every one of them set against a measured distribution
# ---------------------------------------------------------------------------

# A host shared by more than this many distinct names is an umbrella (state
# portal, affiliating university, franchise) rather than a duplicate. Measured:
# iimkashipur.ac.in carries 2 names and is real evidence; manipal.edu carries
# 10, sharda.ac.in 10, nielit.gov.in 12, hte.rajasthan.gov.in 56. The gap
# between "a college and its own duplicate" and "a portal" is wide, so the exact
# value is not delicate — 3 sits inside it.
MAX_NAMES_PER_HOST = 3

# Hosts nobody can own institutionally. These are not umbrellas to be inferred;
# a college listing its Facebook page as its website tells us nothing about
# identity, so they are dropped before the host rule ever runs.
NON_INSTITUTIONAL_HOSTS = frozenset({
    "facebook.com", "m.facebook.com", "instagram.com", "twitter.com", "x.com",
    "linkedin.com", "youtube.com", "sites.google.com", "blogspot.com",
    "wordpress.com", "wixsite.com", "google.com", "goo.gl", "bit.ly",
})

# Name similarity floor for pairing two rows that share an institutional domain
# but NOT a normalised name. Deliberately high: below it the pair is far more
# likely to be two colleges under one trust than one college twice.
NAME_SIM_MIN = 0.62

# Indian establishment years. The floor applies to domestic rows only (see the
# docstring) and the ceiling is "next year", since a college announced for the
# coming session is a legitimate listing.
YEAR_FLOOR_IN = 1800
YEAR_CEIL_SLACK = 1

# Words that carry no identity. Single letters are NOT stopwords: stripping "a"
# from "A.D.V.P College" would leave "d v p college" and risk colliding with a
# genuinely different "D V P College".
_STOPWORDS = frozenset({"the", "of", "and", "for", "at", "an"})

# A qualifier past a shared name prefix means subunit, not copy. Measured on the
# real collisions: Heidelberg University vs "... Medical Faculty Mannheim",
# IIM Kashipur vs "... Dehradun Campus", Sharda University vs "School of
# Pharmacy (SOP), Sharda University".
_SUBUNIT_MARKERS = frozenset({
    "campus", "branch", "unit", "centre", "center", "school", "faculty",
    "department", "dept", "institute", "college", "wing", "extension",
    "offcampus", "constituent", "annexe", "annex",
    # Delivery arms listed separately from the parent, e.g. "Arka Jain
    # University" alongside "Arka Jain Online". Same trust, different product,
    # different fees — merging them would misprice one of the two.
    "online", "distance", "openlearning", "odl",
})

# Ownership labels. Government/Public are the same claim in this catalogue's
# vocabulary (6,083 Government, 640 Public) and must not read as a contradiction.
# "Institute" is a category, not an ownership, so it contradicts nothing.
_OWNERSHIP = {"government": "public", "public": "public", "private": "private"}

# Indian states and UTs, spelled as the corpus spells them plus the aliases that
# appear in address text. NOT taken from SELECT DISTINCT state: that column holds
# 428 values including Adana, Ankara, Alberta and Beijing (the 846 `is_abroad`
# rows), and substring-matching those against Indian addresses is what turned
# "Sankar Ashram" into Ankara and "Punjabi Bagh" into Punjab.
_IN_STATES = frozenset({
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh",
    "goa", "gujarat", "haryana", "himachal pradesh", "jharkhand", "karnataka",
    "kerala", "madhya pradesh", "maharashtra", "manipur", "meghalaya",
    "mizoram", "nagaland", "odisha", "punjab", "rajasthan", "sikkim",
    "tamil nadu", "telangana", "tripura", "uttar pradesh", "uttarakhand",
    "west bengal", "delhi", "jammu and kashmir", "ladakh", "puducherry",
    "chandigarh", "andaman & nicobar islands", "andaman and nicobar islands",
    "dadra & nagar haveli and daman & diu", "daman & diu", "lakshadweep",
})
_STATE_ALIAS = {
    "orissa": "odisha", "pondicherry": "puducherry", "new delhi": "delhi",
    "nct of delhi": "delhi", "uttaranchal": "uttarakhand",
    "jammu & kashmir": "jammu and kashmir", "j&k": "jammu and kashmir",
}

# Conurbations that straddle a state line, where the state column may honestly
# record the region a college brands itself with rather than the polygon its
# building stands in. Measured: 8 of the 65 address/state disagreements are
# Chandigarh-labelled colleges physically in Mohali, Landran or Rajpura
# (Chitkara University "Chandigarh" is in Rajpura, Punjab). Real, but a naming
# convention rather than a data error — so these are reported at low severity
# instead of being hidden or shouted about.
_ADJACENT_STATES = {
    frozenset({"chandigarh", "punjab"}),
    frozenset({"chandigarh", "haryana"}),
    frozenset({"delhi", "haryana"}),
    frozenset({"delhi", "uttar pradesh"}),
    frozenset({"puducherry", "tamil nadu"}),
}


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _fold(s: str | None) -> str:
    """Lowercase, de-punctuate, collapse whitespace.

    U+FFFD is stripped along with punctuation, which is what lets
    "Alva\ufffds Homoeopathic Medical College" meet "Alva's ...". Roughly 0.5%
    of names carry that replacement character from a bad decode upstream, and it
    is invisible in a report — it just silently prevents the match.
    """
    s = (s or "").lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def name_key(name: str | None) -> str:
    """The blocking key: folded name, stopwords dropped, initial runs glued.

    Gluing consecutive single-character tokens is what makes initials stable.
    Folding alone turns "B.M. College, Katihar" into "b m college katihar" but
    "BM College, Katihar" into "bm college katihar", so the two never met — a
    real duplicate pair in the corpus that the first version of this module
    missed. Runs are glued rather than all single letters removed, so two
    genuinely different initials still produce different keys.
    """
    out: list[str] = []
    glued = False  # is out[-1] an initial run, and so open to absorbing more?
    for tok in _fold(name).split():
        if tok in _STOPWORDS:
            continue
        if len(tok) == 1 and glued:
            out[-1] += tok
        else:
            out.append(tok)
            glued = len(tok) == 1
    return " ".join(out)


def place_key(value: str | None) -> str:
    return _fold(value).replace(" ", "")


def host_of(url: str | None) -> str:
    """Registrable-ish host from a website_url. Deliberately string-only.

    Not urllib.parse: the column holds bare hosts, scheme-less values and one
    Google-Business URL with a 120-character utm_ tail, and urlparse returns an
    empty netloc for the scheme-less ones — which would silently drop them from
    every domain check.
    """
    u = (url or "").strip().lower()
    if not u:
        return ""
    u = re.sub(r"^[a-z][a-z0-9+.-]*://", "", u)
    u = re.split(r"[/?#]", u, 1)[0].split(":")[0]
    if u.startswith("www."):
        u = u[4:]
    return u if "." in u else ""


def _tokens(name: str | None) -> set[str]:
    return set(name_key(name).split())


def name_similarity(a: str | None, b: str | None) -> float:
    """max(character ratio, token containment).

    Containment carries the pairs that matter here, where one name is the other
    plus qualifiers: "iim kashipur indian institute management" is fully
    contained in "... dehradun campus", which SequenceMatcher scores at only
    ~0.85 and a plain Jaccard at 0.71.
    """
    ka, kb = name_key(a), name_key(b)
    if not ka or not kb:
        return 0.0
    ta, tb = set(ka.split()), set(kb.split())
    containment = len(ta & tb) / min(len(ta), len(tb))
    return max(SequenceMatcher(None, ka, kb).ratio(), containment)


def state_in_address_tail(address: str | None, segments: int = 3) -> set[str]:
    """Indian states named in the last few comma-separated address segments.

    Only the tail, because the head is street furniture: matching anywhere reads
    "Punjabi Bagh" as Punjab and "Sankar Ashram Chowk" as (of all things) Ankara.
    Measured, restricting to the tail plus word boundaries cut the candidate set
    from 56 noisy rows to 65 precise ones — more findings AND fewer false ones,
    because the noise was masking real disagreements elsewhere.
    """
    if not address:
        return set()
    segs = [s for s in re.split(r"[,\n]", address) if s.strip()]
    found: set[str] = set()
    for seg in segs[-segments:] if len(segs) > segments else segs:
        n = _fold(seg)
        n = re.sub(r"\b\d{5,6}\b", " ", n)
        n = re.sub(r"\bindia\b", " ", n).strip()
        n = re.sub(r"\s+", " ", n)
        if not n:
            continue
        direct = _STATE_ALIAS.get(n, n)
        if direct in _IN_STATES:
            found.add(direct)
            continue
        for st in _IN_STATES:
            if re.search(rf"(?:^|\s){re.escape(st)}(?:$|\s)", n):
                found.add(st)
        for alias, canon in _STATE_ALIAS.items():
            if re.search(rf"(?:^|\s){re.escape(alias)}(?:$|\s)", n):
                found.add(canon)
    return found


def _canon_state(value: str | None) -> str:
    n = _fold(value)
    return _STATE_ALIAS.get(n, n)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_rows(backend=None) -> list[dict]:
    """Every college plus its course count, in one round trip.

    NOTE for anyone extending the SQL here: backend._q_to_pct rewrites `?` to
    `%s` and only guards against a lone '?' string constant, so a regex literal
    like '^https?://' inside SQL becomes '^https%s://' and the query dies with a
    placeholder-count error. Host and state parsing therefore happen in Python,
    which is also where they are testable.
    """
    be = backend or get_backend()
    rows = be.q(
        """SELECT c.college_id, c.name, c.city, c.district, c.state, c.country,
                  c.address, c.type, c.establish_year, c.website_url,
                  c.source_level, c.is_abroad, c.logo_url, c.details,
                  COALESCE(a.n_courses, 0) AS n_courses
           FROM colleges c
           LEFT JOIN college_agg a USING (college_id)"""
    )
    out = []
    for r in rows:
        d = dict(r)
        d["_key"] = name_key(d["name"])
        d["_city"] = place_key(d["city"])
        d["_host"] = host_of(d["website_url"])
        out.append(d)
    return out


def load_course_signatures(backend=None) -> dict[str, set[str]]:
    """college_id -> set of normalised course signatures.

    Uses full_name, not title. `title` is degenerate in this corpus — one
    college's 20 courses are all titled "BA" — so title overlap would score
    every arts college against every other. full_name carries the real
    programme ("bachelor of science hons in zoology").

    Honest caveat on the strength of this signal: on the confirmed duplicate
    pairs the course-set Jaccard is only 0.14-0.33, because the two rows were
    ingested with different subsets of the same catalogue. Overlap is therefore
    used as CORROBORATION for a pair already matched on name and place, never on
    its own, and low overlap is never taken as proof of distinctness.
    """
    be = backend or get_backend()
    sigs: dict[str, set[str]] = collections.defaultdict(set)
    for r in be.q("""SELECT college_id, title, full_name, program_level
                     FROM courses"""):
        label = _fold(r["full_name"] or r["title"])
        if label:
            sigs[r["college_id"]].add(f"{label}|{_fold(r['program_level'])}")
    return sigs


# ---------------------------------------------------------------------------
# Completeness — which row a merge should keep
# ---------------------------------------------------------------------------

def completeness(row: dict) -> int:
    """A keep-score for the merge PLAN. Higher wins.

    Weighted by what a student actually loses if the row is dropped, which is
    why n_courses dominates: a college row with no courses answers no query at
    all. Detail-enrichment is worth a lot because it cost an MME API call at
    ~0.5 req/s and cannot be cheaply recreated. Deliberately crude — it ranks
    two candidates for a human reviewer, it does not decide anything.
    """
    score = 0
    score += min(int(row.get("n_courses") or 0), 40) * 4
    if (row.get("source_level") or "") == "detail":
        score += 30
    if row.get("establish_year"):
        score += 8
    if host_of(row.get("website_url")):
        score += 8
    if row.get("district"):
        score += 4
    if row.get("logo_url"):
        score += 3
    score += min(len(row.get("details") or "") // 400, 6)
    return score


# ---------------------------------------------------------------------------
# Pairwise classification
# ---------------------------------------------------------------------------

VERDICTS = ("DISTINCT", "TRUE_DUPLICATE", "CONFLICTING", "SUBUNIT_OR_CAMPUS")


def _compare_key(field: str, value) -> str:
    """How a field is compared for disagreement, as opposed to displayed.

    One definition shared by the verdict logic and the merge plan, so the two
    can never disagree about whether a cluster has a conflict.
    """
    if value in (None, ""):
        return ""
    if field == "website_url":
        return host_of(value)
    if field == "type":
        return _OWNERSHIP.get(_fold(value), _fold(value))
    if field == "state":
        return _canon_state(value)
    if field in ("city", "district"):
        return place_key(value)
    return str(value).strip()


def _contradictions(a: dict, b: dict) -> list[str]:
    """Field-level disagreements between two rows held to be one institution."""
    out = []
    oa = _OWNERSHIP.get(_fold(a.get("type")))
    ob = _OWNERSHIP.get(_fold(b.get("type")))
    if oa and ob and oa != ob:
        out.append(f"type {a['type']} vs {b['type']}")

    ya, yb = a.get("establish_year"), b.get("establish_year")
    if ya and yb and ya != yb:
        out.append(f"establish_year {ya} vs {yb}")

    sa, sb = _canon_state(a.get("state")), _canon_state(b.get("state"))
    if sa and sb and sa != sb:
        out.append(f"state {a['state']} vs {b['state']}")

    ca, cb = a["_city"], b["_city"]
    if ca and cb and ca != cb:
        out.append(f"city {a['city']} vs {b['city']}")

    ha, hb = a["_host"], b["_host"]
    if ha and hb and ha != hb:
        out.append(f"website {ha} vs {hb}")
    return out


def _extra_tokens(a: dict, b: dict) -> str:
    """Tokens one name carries that the other's name fully lacks, when the
    shorter name is otherwise wholly contained in the longer."""
    ta, tb = _tokens(a["name"]), _tokens(b["name"])
    if ta == tb or not ta or not tb:
        return ""
    longer, shorter = (ta, tb) if len(ta) > len(tb) else (tb, ta)
    if not shorter <= longer:
        return ""
    return " ".join(sorted(longer - shorter))


def classify_pair(a: dict, b: dict, course_sigs: dict | None = None) -> tuple[str, list[str]]:
    """(verdict, evidence). Never raises.

    THE PRECISION RULE, and the most important decision in this module: only an
    IDENTICAL normalised name can produce TRUE_DUPLICATE or CONFLICTING, i.e.
    only an identical name can produce a merge proposal. Anything that merely
    looks similar tops out at SUBUNIT_OR_CAMPUS, which proposes nothing.

    That rule was not the first attempt. Allowing high name *similarity* to
    authorise a merge — containment >= 0.62 on rows sharing one domain — put 404
    clusters into CONFLICTING, and spot-checking them showed why that was wrong:
      "College of Engineering, Chandigarh Group of Colleges"  vs
      "Chandigarh Engineering College, Chandigarh Group of Colleges"
      "Himalayan Group of Professional Institutions, Sirmour" (Kala Amb) vs
      "Himalayan Group of Professional Institution, Joginder Nagar" (Mandi)
      "Arka Jain University" vs "Arka Jain Online"
    Every one is a real, separately-listed institution or arm sharing a trust
    and a domain, and every one had been handed a "merge into keep" row. Token
    containment hits 1.00 whenever the shorter name is generic
    ("Government College, Jhajjar" is contained in "Government Post Graduate
    College, Jhajjar"), so similarity cannot carry this decision. Name
    similarity still drives BLOCKING, where a false positive only costs a
    comparison.
    """
    ev: list[str] = []
    sim = name_similarity(a["name"], b["name"])
    ev.append(f"name_sim={sim:.2f}")
    same_name = a["_key"] == b["_key"] and bool(a["_key"])

    if not same_name:
        extra = _extra_tokens(a, b)
        if extra:
            kind = ("subunit/campus" if set(extra.split()) & _SUBUNIT_MARKERS
                    else "separate arm or renamed listing")
            ev.append(f"names differ by '{extra}' -> {kind}, not a copy")
            ev.extend(_contradictions(a, b))
            return "SUBUNIT_OR_CAMPUS", ev
        ev.append("names are similar but neither contains the other")
        return "DISTINCT", ev

    ca, cb = a["_city"], b["_city"]
    da, db = place_key(a.get("district")), place_key(b.get("district"))
    sa, sb = _canon_state(a.get("state")), _canon_state(b.get("state"))

    if ca and cb and ca != cb:
        # Identical name, different city. Overwhelmingly two real institutions
        # (all four "Presidency College" rows land here). The one exception is a
        # city/district swap — the corpus does record a college's district in
        # the city slot, e.g. "Government Degree College, Mangalore" carries
        # city=Mangalore, district=Haridwar — so a cross-match between one row's
        # city and the other's district is treated as the same place rather than
        # a different one.
        if (ca and ca == db) or (cb and cb == da):
            ev.append(f"city/district swap: {a['city']}|{a['district']} vs "
                      f"{b['city']}|{b['district']}")
        else:
            ev.append(f"identical name, different city: {a['city']}/{a['state']}"
                      f" vs {b['city']}/{b['state']}")
            return "DISTINCT", ev
    else:
        # Same city STRING is not the same place if the states differ: India has
        # many homonym cities and this is a real error the first version made —
        # "Government College, Bilaspur" exists in both Himachal Pradesh and
        # Chhattisgarh (Bilaspur is a real city in each), and the two rows were
        # clustered together with the HP row proposed as the survivor. A state
        # disagreement that genuinely IS an error is caught independently by
        # scan_city_state, which arbitrates against the address rather than
        # guessing which of two plausible rows to keep.
        if sa and sb and sa != sb:
            ev.append(f"identical name and city but different state -> homonym "
                      f"city ({a['city']}: {a['state']} vs {b['state']})")
            return "DISTINCT", ev
        ev.append(f"identical name, same city "
                  f"({a['city'] or b['city'] or 'city not recorded'})")

    if course_sigs is not None:
        siga = course_sigs.get(a["college_id"], set())
        sigb = course_sigs.get(b["college_id"], set())
        if siga and sigb:
            ev.append(f"course_overlap={len(siga & sigb)}/"
                      f"{min(len(siga), len(sigb))}")

    con = _contradictions(a, b)
    if con:
        ev.extend(con)
        return "CONFLICTING", ev
    return "TRUE_DUPLICATE", ev


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def _blocks(rows: list[dict]) -> list[list[dict]]:
    """Candidate sets to compare within. Two blocking passes, union'd.

    Name-key blocking alone cannot see "IIM Kashipur - Indian Institute of
    Management" next to its own Dehradun campus row, because the names differ by
    two tokens. Domain blocking catches those — but only on hosts that survive
    the umbrella filter, or every one of the 56 colleges on
    hte.rajasthan.gov.in would be compared against the other 55.
    """
    out: list[list[dict]] = []

    by_name: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        if r["_key"]:
            by_name[r["_key"]].append(r)
    out.extend(v for v in by_name.values() if len(v) > 1)

    by_host: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        h = r["_host"]
        if h and h not in NON_INSTITUTIONAL_HOSTS:
            by_host[h].append(r)
    for host, group in by_host.items():
        if len(group) < 2:
            continue
        if len({r["_key"] for r in group}) > MAX_NAMES_PER_HOST:
            continue  # umbrella: portal, affiliating university, franchise
        # Only pairs whose names are actually close; one trust legitimately owns
        # several unrelated colleges on one domain.
        keep = [r for r in group
                if any(other is not r
                       and name_similarity(r["name"], other["name"]) >= NAME_SIM_MIN
                       for other in group)]
        if len(keep) > 1:
            out.append(keep)
    return out


def find_clusters(rows: list[dict], course_sigs: dict | None = None) -> list[dict]:
    """Group rows into findings.

    Pairs are classified first and rows are then joined into connected
    components over the non-DISTINCT edges. A group is not assumed homogeneous:
    a three-row name group can hold one duplicate pair plus one unrelated
    college, and collapsing it to a single group-level verdict would either
    hide the duplicate or slander the third row.
    """
    edges: dict[tuple[str, str], tuple[str, list[str]]] = {}
    seen_pairs: set[tuple[str, str]] = set()
    distinct_groups = 0
    pair_counts: collections.Counter = collections.Counter()

    for block in _blocks(rows):
        for i in range(len(block)):
            for j in range(i + 1, len(block)):
                a, b = block[i], block[j]
                pid = tuple(sorted((a["college_id"], b["college_id"])))
                if pid in seen_pairs:
                    continue
                seen_pairs.add(pid)
                verdict, ev = classify_pair(a, b, course_sigs)
                pair_counts[verdict] += 1
                if verdict != "DISTINCT":
                    edges[pid] = (verdict, ev)
                else:
                    distinct_groups += 1

    by_id = {r["college_id"]: r for r in rows}
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for (x, y) in edges:
        union(x, y)

    comps: dict[str, list[str]] = collections.defaultdict(list)
    for cid in parent:
        comps[find(cid)].append(cid)

    clusters = []
    for n, (root, members) in enumerate(sorted(comps.items()), 1):
        if len(members) < 2:
            continue
        verdicts, evidence = [], []
        for (x, y), (v, ev) in edges.items():
            if x in members and y in members:
                verdicts.append(v)
                evidence.append(f"{x[-6:]}~{y[-6:]}: {v} [{'; '.join(ev)}]")
        # Worst-case wins: a cluster holding one contradiction is a conflict,
        # and a subunit edge downgrades a merge proposal to "review by hand".
        if "CONFLICTING" in verdicts:
            verdict = "CONFLICTING"
        elif "SUBUNIT_OR_CAMPUS" in verdicts:
            verdict = "SUBUNIT_OR_CAMPUS"
        else:
            verdict = "TRUE_DUPLICATE"
        rowset = sorted((by_id[m] for m in members),
                        key=lambda r: (-completeness(r), r["college_id"]))
        cluster = {
            "cluster_id": f"C{n:04d}",
            "verdict": verdict,
            "rows": rowset,
            "keep": rowset[0],
            "evidence": evidence,
        }
        # A cluster of three can hide a disagreement no pair can see: if A says
        # 2010, B says nothing and C says 2012, both edges A~B and B~C are clean
        # and only the whole cluster is contradictory. Promoting here keeps the
        # guarantee the client relies on when triaging the CSV — TRUE_DUPLICATE
        # means "no field disagreement anywhere in this cluster".
        if verdict == "TRUE_DUPLICATE" and merge_plan(cluster)["field_choices"]:
            cluster["verdict"] = "CONFLICTING"
            cluster["evidence"].append(
                "promoted: pairwise checks clean, but the cluster as a whole "
                "disagrees on a field")
        clusters.append(cluster)
    clusters.sort(key=lambda c: (VERDICTS.index(c["verdict"]), c["cluster_id"]))
    return clusters, {"pairs": dict(pair_counts), "distinct_pairs": distinct_groups}


def merge_plan(cluster: dict) -> dict:
    """What a merge WOULD do. Nothing executes this.

    Emitted even for CONFLICTING clusters, because the plan is where the
    disagreement becomes legible — "keep 1964, the other row says 1979" is the
    sentence a reviewer needs. For SUBUNIT_OR_CAMPUS no merge is proposed at
    all: those rows are different real places.
    """
    keep = cluster["keep"]
    drop = [r for r in cluster["rows"] if r["college_id"] != keep["college_id"]]
    plan = {
        "keep_id": keep["college_id"],
        "keep_name": keep["name"],
        "keep_score": completeness(keep),
        "drop_ids": [r["college_id"] for r in drop],
        "action": "REVIEW_ONLY - subunits are separate real places"
                  if cluster["verdict"] == "SUBUNIT_OR_CAMPUS" else
                  "MERGE_AFTER_REVIEW",
        "field_choices": [],
        "courses_to_reparent": sum(int(r.get("n_courses") or 0) for r in drop),
    }
    for field in ("city", "state", "district", "type", "establish_year",
                  "website_url"):
        # Grouped by COMPARISON key, displayed as the raw value. Without this,
        # 'www.x.in' and 'http://www.x.in/' read as a website conflict and a
        # reviewer is sent to arbitrate a trailing slash — which is also how a
        # cluster ended up labelled TRUE_DUPLICATE while carrying a
        # "conflicting_fields" note, since the verdict logic already compares
        # hosts and only this loop did not.
        groups: dict[str, list[str]] = {}
        for r in cluster["rows"]:
            v = r.get(field)
            if v in (None, ""):
                continue
            groups.setdefault(_compare_key(field, v), []).append(str(v))
        if len(groups) > 1:
            chosen = keep.get(field)
            ckey = _compare_key(field, chosen) if chosen not in (None, "") else None
            plan["field_choices"].append({
                "field": field,
                "keep_value": "" if chosen in (None, "") else str(chosen),
                "alternatives": sorted({vs[0] for k, vs in groups.items()
                                        if k != ckey}),
            })
    return plan


# ---------------------------------------------------------------------------
# Standalone conflict scans
# ---------------------------------------------------------------------------

def scan_city_state(rows: list[dict]) -> list[dict]:
    """State column vs the state named in the address tail."""
    out = []
    for r in rows:
        if r.get("is_abroad") or not r.get("state") or not r.get("address"):
            continue
        col = _canon_state(r["state"])
        if col not in _IN_STATES:
            continue
        found = state_in_address_tail(r["address"])
        if not found or col in found:
            continue
        addr_state = sorted(found)[0]
        adjacent = frozenset({col, addr_state}) in _ADJACENT_STATES
        out.append({
            "check": "city_state_mismatch",
            "severity": "low" if adjacent else "high",
            "college_id": r["college_id"],
            "name": r["name"],
            "detail": f"state column '{r['state']}' but address tail names "
                      f"'{addr_state}'"
                      + (" (adjacent conurbation - may be a branding choice)"
                         if adjacent else ""),
            "evidence": (r["address"] or "")[-90:],
        })
    out.sort(key=lambda d: (d["severity"] != "high", d["name"] or ""))
    return out


def scan_years(rows: list[dict], this_year: int) -> list[dict]:
    """Impossible establishment years, with the abroad rows excluded from the
    floor — that exclusion is the difference between 4 findings and 88."""
    out = []
    ceiling = this_year + YEAR_CEIL_SLACK
    for r in rows:
        y = r.get("establish_year")
        if not y:
            continue
        if y > ceiling:
            out.append({"check": "year_future", "severity": "high",
                        "college_id": r["college_id"], "name": r["name"],
                        "detail": f"establish_year {y} is after {ceiling}",
                        "evidence": ""})
        elif y < YEAR_FLOOR_IN and not r.get("is_abroad"):
            out.append({"check": "year_too_early", "severity": "high",
                        "college_id": r["college_id"], "name": r["name"],
                        "detail": f"establish_year {y} predates {YEAR_FLOOR_IN} "
                                  f"for an Indian institution",
                        "evidence": f"{r.get('city')}/{r.get('state')}"})
    return out


def scan_shared_domains(rows: list[dict]) -> list[dict]:
    """Hosts claimed by more than one distinct college name, in three tiers.

    Aggregated per host, not per row: 2,736 rows share a domain with someone,
    but they collapse to ~1,030 hosts, which is a list a human can actually
    filter. Measured tier sizes are 110 umbrella / 609 similar / 306 unrelated.

    The `unrelated` tier is the one with teeth, and it matters beyond tidiness:
    if two unrelated colleges carry the same website_url then at least one is
    wrong, and the harvest sweep will read that site and attach its fees,
    cutoffs and hostel charges to a college they do not belong to. That is a
    confidently wrong number in front of a student, which is the failure this
    whole pipeline is built to prevent — so it is reported even though it cannot
    be resolved without fetching, and it is ranked most-dissimilar first.
    """
    by_host: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        h = r["_host"]
        if h and h not in NON_INSTITUTIONAL_HOSTS:
            by_host[h].append(r)
    out = []
    for host, group in sorted(by_host.items()):
        names = {r["_key"] for r in group}
        if len(names) < 2:
            continue
        best = max(name_similarity(a["name"], b["name"])
                   for i, a in enumerate(group) for b in group[i + 1:])
        if len(names) > MAX_NAMES_PER_HOST:
            sev, why = "low", ("umbrella: state portal, affiliating university "
                               "or franchise chain - expected")
        elif best >= NAME_SIM_MIN:
            sev, why = "medium", ("names are near-identical - cross-check the "
                                  "cluster report for a duplicate")
        else:
            sev, why = "high", ("unrelated names on one domain - at least one "
                                "website_url is probably wrong, and the harvest "
                                "sweep would attach that site's facts to the "
                                "wrong college")
        out.append({
            "check": "shared_domain",
            "severity": sev,
            "college_id": ";".join(r["college_id"] for r in group[:6]),
            "name": host,
            "detail": f"{len(group)} rows / {len(names)} distinct names share "
                      f"{host} (max name_sim={best:.2f}) - {why}",
            "evidence": " | ".join((r["name"] or "")[:44] for r in group[:4]),
        })
    # Most-dissimilar first inside each tier: those are the likeliest wrong URLs.
    out.sort(key=lambda d: ({"high": 0, "medium": 1, "low": 2}[d["severity"]],
                            d["detail"]))
    return out


def scan_missing_place(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        missing = [f for f in ("city", "state") if not (r.get(f) or "").strip()]
        if missing:
            out.append({"check": "missing_place", "severity": "medium",
                        "college_id": r["college_id"], "name": r["name"],
                        "detail": f"no {', '.join(missing)}",
                        "evidence": (r.get("address") or "")[:80]})
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_clusters_csv(path: str, clusters: list[dict]) -> int:
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["cluster_id", "verdict", "recommendation", "college_id",
                    "name", "city", "district", "state", "type",
                    "establish_year", "website_url", "n_courses",
                    "source_level", "completeness", "courses_to_reparent",
                    "conflicting_fields", "evidence"])
        n = 0
        for c in clusters:
            plan = merge_plan(c)
            fields = "; ".join(
                f"{fc['field']}: keep '{fc['keep_value']}' vs {fc['alternatives']}"
                for fc in plan["field_choices"])
            for r in c["rows"]:
                is_keep = r["college_id"] == plan["keep_id"]
                if c["verdict"] == "SUBUNIT_OR_CAMPUS":
                    rec = "KEEP_BOTH_REVIEW"
                else:
                    rec = "KEEP (proposed)" if is_keep else "MERGE_INTO_KEEP (proposed)"
                w.writerow([
                    c["cluster_id"], c["verdict"], rec, r["college_id"],
                    r["name"], r.get("city"), r.get("district"), r.get("state"),
                    r.get("type"), r.get("establish_year"), r.get("website_url"),
                    r.get("n_courses"), r.get("source_level"), completeness(r),
                    plan["courses_to_reparent"] if is_keep else "",
                    fields if is_keep else "",
                    " || ".join(c["evidence"]) if is_keep else "",
                ])
                n += 1
    return n


def write_conflicts_csv(path: str, conflicts: list[dict]) -> int:
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["check", "severity", "college_id", "name", "detail",
                    "evidence"])
        for c in conflicts:
            w.writerow([c["check"], c["severity"], c["college_id"], c["name"],
                        c["detail"], c["evidence"]])
    return len(conflicts)


def _table(headers, rows, widths) -> str:
    line = "  ".join(h[:w].ljust(w) for h, w in zip(headers, widths))
    out = [line, "  ".join("-" * w for w in widths)]
    for r in rows:
        out.append("  ".join(str(c if c is not None else "")[:w].ljust(w)
                             for c, w in zip(r, widths)))
    return "\n".join(out)


def report(rows, clusters, stats, conflicts, out_dir) -> str:
    L = [
        "=" * 78,
        "MIVI catalogue - duplicate and conflict report (DETECTION ONLY)",
        "=" * 78,
        f"colleges scanned      : {len(rows):,}",
        f"candidate pairs judged: {sum(stats['pairs'].values()):,}",
        "",
        "PAIR VERDICTS",
        _table(["verdict", "pairs"],
               [(k, f"{v:,}") for k, v in sorted(stats["pairs"].items())],
               [22, 8]),
        "",
        "CLUSTERS (rows the same institution appears in more than once)",
    ]
    byv = collections.Counter(c["verdict"] for c in clusters)
    L.append(_table(
        ["verdict", "clusters", "rows"],
        [(v, byv[v], sum(len(c["rows"]) for c in clusters if c["verdict"] == v))
         for v in VERDICTS if byv[v]],
        [22, 10, 8]))

    for verdict in ("CONFLICTING", "TRUE_DUPLICATE", "SUBUNIT_OR_CAMPUS"):
        sel = [c for c in clusters if c["verdict"] == verdict][:6]
        if not sel:
            continue
        L += ["", f"--- {verdict}: first {len(sel)} of {byv[verdict]} ---"]
        for c in sel:
            plan = merge_plan(c)
            L.append(f"  [{c['cluster_id']}] {plan['action']}")
            for r in c["rows"]:
                tag = "KEEP " if r["college_id"] == plan["keep_id"] else "  -> "
                L.append(f"    {tag}{r['college_id']}  {(r['name'] or '')[:52]:<52} "
                         f"| {r.get('city')}/{r.get('state')} | {r.get('type')} "
                         f"| {r.get('establish_year')} | nc={r.get('n_courses')} "
                         f"| score={completeness(r)}")
            for fc in plan["field_choices"]:
                L.append(f"       conflict {fc['field']}: keep "
                         f"'{fc['keep_value']}' vs {fc['alternatives']}")
            if plan["courses_to_reparent"]:
                L.append(f"       {plan['courses_to_reparent']} course rows would "
                         f"need reparenting")

    L += ["", "OTHER CONFLICT CHECKS"]
    agg = collections.Counter((c["check"], c["severity"]) for c in conflicts)
    L.append(_table(["check", "severity", "findings"],
                    [(k[0], k[1], v) for k, v in sorted(agg.items())],
                    [26, 10, 9]))
    for check in ("city_state_mismatch", "year_too_early", "year_future",
                  "shared_domain", "missing_place"):
        sel = [c for c in conflicts if c["check"] == check][:5]
        if not sel:
            L.append(f"\n  {check}: none")
            continue
        total = sum(1 for c in conflicts if c["check"] == check)
        L.append(f"\n  {check} ({total} total), first {len(sel)}:")
        for c in sel:
            L.append(f"    [{c['severity']:<6}] {(c['name'] or '')[:44]:<44} "
                     f"{c['detail'][:80]}")
            if c["evidence"]:
                L.append(f"             evidence: {c['evidence'][:88]}")

    L += ["", f"CSV written to {out_dir}", "=" * 78]
    return "\n".join(L)


# ---------------------------------------------------------------------------

def run(out_dir: str = "reports", this_year: int | None = None,
        backend=None) -> dict:
    import datetime
    this_year = this_year or datetime.date.today().year
    os.makedirs(out_dir, exist_ok=True)

    rows = load_rows(backend)
    sigs = load_course_signatures(backend)
    clusters, stats = find_clusters(rows, sigs)

    conflicts = (scan_city_state(rows) + scan_years(rows, this_year)
                 + scan_shared_domains(rows) + scan_missing_place(rows))

    cpath = os.path.join(out_dir, "dedupe_clusters.csv")
    xpath = os.path.join(out_dir, "dedupe_conflicts.csv")
    write_clusters_csv(cpath, clusters)
    write_conflicts_csv(xpath, conflicts)

    return {"rows": rows, "clusters": clusters, "stats": stats,
            "conflicts": conflicts, "clusters_csv": cpath,
            "conflicts_csv": xpath}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Detect duplicate and conflicting college records. "
                    "Reports only - never writes to the catalogue.")
    p.add_argument("--out", default="reports", help="directory for the CSVs")
    p.add_argument("--year", type=int, default=None,
                   help="treat this as the current year for the future-year check")
    args = p.parse_args(argv)

    res = run(args.out, args.year)
    print(report(res["rows"], res["clusters"], res["stats"], res["conflicts"],
                 args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
