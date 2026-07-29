"""Coordinates and statutory approvals for the tier-1 colleges.

FOUR COLUMNS, TWO COMPLETELY DIFFERENT SOURCES
    latitude / longitude    a geocoder, gated on the returned address agreeing
                            with the college's own city and state
    approvals               AICTE / UGC / PCI / NMC / BCI / INC / NCTE, read
                            verbatim off the college's OWN site
    accreditations          NAAC (already in registry_fact — a join, not a
                            crawl) plus NBA off the same pages as approvals

WHY THE GEOCODER IS GATED SO HARD
A geocoder never says "I don't know". Ask it for a college it has never heard
of and it returns the centre of the town, with the same shape of response and
the same confident-looking coordinates as a real campus hit. Measured on the
first 25 tier-1 colleges, that is not a hypothetical: unfiltered, the service
answered every query.

So two gates, and a result has to pass BOTH:

  1. AGREEMENT. The returned address must be in the college's own state, and
     its city / district / suburb must match what the catalogue holds. A pin in
     the wrong state is not a near miss, it is a different college.

  2. PRECISION. The matched OSM object must be a PLACE — a campus, a building,
     a street address. If the geocoder came back with `place=city` or an
     administrative boundary, that IS the town-centre fallback described above,
     dressed as an answer, and it is refused however well the name agreed.

A wrong pin on a map is worse than no pin: no pin reads as "we don't know",
a wrong pin reads as "here it is" and sends somebody to the wrong town.

WHY APPROVALS ARE REGEX, NOT AN LLM
The whole vocabulary is seven statutory bodies and a handful of approval verbs.
A regex over the page text cannot invent an approval, because the evidence it
stores IS the substring it matched — verbatim by construction, which is exactly
what rule 1 demands. It is also free, and the LLM extraction quota is this
project's real bottleneck (see the provider budgets note: ~15 calls/minute).

The hard part is not finding "AICTE", it is refusing the four ways that string
appears on a page WITHOUT being an approval this college holds:

    "AICTE approval is awaited"            -> negation window
    "UGC NET coaching available"           -> UGC, no approval verb, exam context
    "Inc." in a footer company name        -> INC needs its full name or nursing
    "list of AICTE approved colleges"      -> generic, no first-person claim

A body is recorded only when an approval verb sits within 120 characters of the
body's own name, no negation or pending word sits in that same window, and —
for the acronyms that collide with ordinary English — either the full statutory
name is spelled out or a domain word vouches for it.

WHAT THIS DOES NOT DO
It does not touch pincode (the sibling harvester owns that column) and it does
not re-crawl NAAC: 2,504 grades are already in registry_fact, dated by the
NAAC's own "as on" file date, and re-fetching them could only make them worse.

Run:
    python -m rag_core.sources.harvest_geo --report
    python -m rag_core.sources.harvest_geo --naac                # a join, no network
    python -m rag_core.sources.harvest_geo --geocode --limit 25 --dry-run
    python -m rag_core.sources.harvest_geo --geocode
    python -m rag_core.sources.harvest_geo --approvals --limit 25 --dry-run
    python -m rag_core.sources.harvest_geo --approvals --dump data/raw/approvals_tier1.jsonl
    python -m rag_core.sources.harvest_geo --redate data/raw/approvals_tier1.jsonl

MEASURED ON TIER 1 (863 colleges, 29 Jul 2026)
    lat/long        369 stored, 494 refused. 489 of those 494 were refused
                    because OSM has no such place — not because a gate was
                    strict. 30 were offered only a town centroid, 43 pointed at
                    the wrong state, 31 came back under a name that does not
                    corroborate the college. 356 of the 369 are campus-precise.
    approvals       337 colleges, from 777 with a website. 122 sites were
                    unreachable over plain HTTP and 297 state no approval in
                    parseable form — so 419 of 777 yielded nothing, and that is
                    the honest number.
    accreditations  767 — 758 NAAC joined out of registry_fact at no cost, plus
                    128 NBA read off college sites.
    Only 4 of 585 approval statements state a year they are valid for. That is
    not a gap in the reader: approvals pages say a body approved the college,
    not for how long, and NULL is the true answer for the other 581.
"""
from __future__ import annotations

import argparse
import asyncio
import difflib
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx

from .. import config  # noqa: F401 - loads .env before the backend is imported
from ..store.backend import get_backend
from .registries import _norm, _sig, _place_key, _load_place_aliases
from .verify_facts import page_is_about, stated_year
from .verify_facts import verify as verify_fact
from .web_enrich import (
    HEADERS,
    UA,
    FirecrawlBudget,
    Politeness,
    fetch,
    normalise_site,
    parse_html,
)

ROOT = Path(__file__).resolve().parents[2]
GEO_CACHE = ROOT / "data" / "raw" / "geocode"


def _now():
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Geocoding
# ---------------------------------------------------------------------------

# Overridable so a self-hosted Nominatim can be dropped in without a code
# change — which is the recommended answer to everything below.
NOMINATIM_URL = os.getenv(
    "MIVI_NOMINATIM_URL", "https://nominatim.openstreetmap.org/search")

# THE RATE LIMIT IS THE WHOLE DEAL. OSMF gives Nominatim away and asks for at
# most one request per second from an identified client. That is not a
# throttle to tune, it is the price of admission, so it is enforced on a
# monotonic clock in ONE sequential worker — no concurrency, ever. Twelve
# workers each "respecting 1/s" is twelve requests per second.
NOMINATIM_MIN_INTERVAL = 1.05          # a hair over 1s, for clock jitter

# A ceiling on one run, so this can never quietly become the bulk geocoding job
# the usage policy asks people not to run. Tier 1 is 863 colleges at up to
# three queries each; the cache below means a run that stops at the cap and is
# restarted pays nothing for what it already fetched.
MAX_GEOCODE_REQUESTS = int(os.getenv("MIVI_GEOCODE_MAX", "2600"))

# ROBOTS.TXT vs THE API USAGE POLICY — read this before changing the default.
#
# nominatim.openstreetmap.org/robots.txt says `Disallow: /search` for every
# agent, while the OSMF Nominatim Usage Policy simultaneously grants automated
# API access at <=1 req/s to a client that identifies itself. Both documents
# are published by the same operator and they are not in conflict: robots.txt
# keeps SEARCH ENGINES from indexing generated result pages, the usage policy
# governs API clients. Every free geocoder looks like this — photon.komoot.io
# serves a blanket `Disallow: /`.
#
# This project's politeness rule is written for crawling third-party WEBSITES,
# and that is where it stays absolute (see the approvals sweep below, which
# checks robots on every host it touches). For the geocoding API the governing
# document is the usage policy, and this module complies with it literally:
# one identified request per second, capped per run, and cached to disk so a
# re-run never asks the same question twice.
#
# It is still somebody's call, not the crawler's. --respect-robots refuses any
# endpoint whose robots.txt disallows the path, which is also what makes the
# switch to a self-hosted instance a no-op.
GEOCODER_ATTRIBUTION = "OpenStreetMap contributors / Nominatim (ODbL)"

# State spellings that differ between the catalogue and OSM. Small, explicit
# and one-directional: a wrong equivalence here would let a pin in one state
# corroborate a college in another, which is the exact failure the gate exists
# to stop.
_STATE_FORMS = {
    "orissa": "odisha",
    "pondicherry": "puducherry",
    "uttaranchal": "uttarakhand",
    "nct of delhi": "delhi",
    "national capital territory of delhi": "delhi",
    "delhi nct": "delhi",
    "jammu and kashmir": "jammu kashmir",
    "jammu & kashmir": "jammu kashmir",
    "dadra and nagar haveli and daman and diu": "dadra nagar haveli daman diu",
    "andaman and nicobar islands": "andaman nicobar islands",
    "tamilnadu": "tamil nadu",
    "chattisgarh": "chhattisgarh",
    "banaras": "varanasi",
}

# The OSM address keys that name a settlement, most specific first. `city` is
# absent far more often than you would expect — Mumbai comes back as
# `state_district: Mumbai City District` — so all of them are searched.
_PLACE_KEYS = ("city", "town", "village", "municipality", "hamlet", "suburb",
               "city_district", "county", "state_district", "region", "borough")

# PRECISION. What kind of OSM object the geocoder actually matched.
# Anything not listed here is refused, which deliberately includes
# `place=city|town|village` and `boundary=administrative`: those ARE the
# town-centre fallback, and accepting them is how a map fills up with pins that
# all point at the bus stand.
_PRECISE = {
    "amenity": "poi",        # amenity=university / college / school
    "building": "poi",
    "office": "poi",
    "tourism": "poi",
    "leisure": "poi",
    "healthcare": "poi",
    "man_made": "poi",
    "historic": "poi",
    "shop": "poi",
    "highway": "street",     # the address query landing on the right road
}
# landuse is only precise for the campus polygon itself; landuse=residential is
# a neighbourhood, which is a centroid by another name.
_PRECISE_LANDUSE = {"education", "university", "college", "school", "campus"}
# place= is a centroid EXCEPT for these, which are real addressed points.
_PRECISE_PLACE = {"house", "house_number", "farm", "isolated_dwelling"}

_CONFIDENCE = {"poi": 0.9, "street": 0.72}

# NAME AGREEMENT. Asking the geocoder for a college BY NAME means the thing it
# returns has to be checked against that name — the query is not the evidence.
# "Presidency College" is four institutions in four cities and OSM holds at
# least two of them, so a name query alone would put a Bengaluru pin on a
# Chennai college half the time.
#
# 0.60 with the city agreeing, 0.90 without, and the asymmetry is the point.
# The pairing that has to die is a generic OSM name swallowing a specific
# catalogue one: "Government College" against "Government College of
# Engineering, Amravati" scores 0.59 and is refused, while "Birla Institute of
# Technology" against "Birla Institute of Technology, Mesra" scores 0.91 and is
# not. Same shape as registries.match_one's two floors, same reason.
NAME_RATIO_MIN = 0.60          # when city/district corroborates
NAME_RATIO_SOLO = 0.90         # when it does not
NAME_CONTAINMENT_MIN = 0.75    # of the OSM name's distinctive words


def _fold(s) -> str:
    """Strip diacritics before normalising. OSM writes Indian place names with
    macrons — 'Kāditoli' — and a byte comparison would call that a different
    town from the catalogue's 'Kaditoli'."""
    raw = unicodedata.normalize("NFKD", str(s or ""))
    return "".join(ch for ch in raw if not unicodedata.combining(ch))


def _state_key(value, aliases: dict[str, str]) -> str:
    v = _place_key(_fold(value), aliases)
    return _STATE_FORMS.get(v, v)


def _osm_name(hit: dict) -> str:
    """The matched object's OWN name, not its whole address hierarchy."""
    return _fold(hit.get("name") or (hit.get("display_name") or "").split(",")[0])


def name_agreement(osm_name: str, college_name: str) -> tuple[float, float]:
    """(similarity, containment of the OSM name's distinctive words)."""
    a, b = _norm(_fold(osm_name)), _norm(_fold(college_name))
    if not a or not b:
        return 0.0, 0.0
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    sa, sb = _sig(a), _sig(b)
    contain = len(sa & sb) / len(sa) if sa else 0.0
    return ratio, contain


class Geocoder:
    """One sequential, rate-limited, disk-cached client.

    Sequential is not an oversight. See NOMINATIM_MIN_INTERVAL: the limit is on
    the SERVICE, so it has to be enforced at the only place that sees every
    request.
    """

    def __init__(self, url: str = NOMINATIM_URL, respect_robots: bool = False,
                 max_requests: int = MAX_GEOCODE_REQUESTS):
        self.url = url
        self.respect_robots = respect_robots
        self.max_requests = max_requests
        self.requests = 0
        self.cache_hits = 0
        self._next_ok = 0.0
        self._robots_ok: bool | None = None
        GEO_CACHE.mkdir(parents=True, exist_ok=True)
        self._client = httpx.Client(
            headers={**HEADERS, "Accept": "application/json"},
            timeout=30.0, follow_redirects=True)

    # -- robots -------------------------------------------------------------

    def robots_allows(self) -> bool:
        if self._robots_ok is not None:
            return self._robots_ok
        import urllib.robotparser

        p = urlparse(self.url)
        rp = urllib.robotparser.RobotFileParser()
        try:
            r = self._client.get(f"{p.scheme}://{p.netloc}/robots.txt", timeout=15.0)
            if r.status_code == 200 and len(r.text) < 500_000:
                rp.parse(r.text.splitlines())
                self._robots_ok = bool(rp.can_fetch(UA, self.url))
            else:
                self._robots_ok = True      # no usable robots.txt -> allow
        except Exception:  # noqa: BLE001 - unreachable robots is not a disallow
            self._robots_ok = True
        return self._robots_ok

    # -- one query ----------------------------------------------------------

    def _cache_path(self, q: str, limit: int) -> Path:
        key = hashlib.sha1(f"{self.url}|{limit}|{q}".encode("utf-8")).hexdigest()[:24]
        return GEO_CACHE / f"{key}.json"

    def search(self, q: str, limit: int = 5) -> list[dict] | None:
        """Results for one query string, or None if the request was not made.

        A cached answer — including a cached EMPTY answer — costs nothing and
        does not touch the network, which is what makes re-running this safe.
        """
        path = self._cache_path(q, limit)
        if path.exists():
            try:
                self.cache_hits += 1
                return json.loads(path.read_text(encoding="utf-8"))["results"]
            except Exception:  # noqa: BLE001 - a corrupt cache entry is refetched
                pass
        if self.requests >= self.max_requests:
            return None
        if self.respect_robots and not self.robots_allows():
            return None

        now = time.monotonic()
        if now < self._next_ok:
            time.sleep(self._next_ok - now)
        self._next_ok = time.monotonic() + NOMINATIM_MIN_INTERVAL

        try:
            r = self._client.get(self.url, params={
                "q": q, "format": "jsonv2", "addressdetails": 1,
                "limit": limit, "countrycodes": "in",
            })
            self.requests += 1
            if r.status_code != 200:
                return None
            results = r.json()
            if not isinstance(results, list):
                return None
        except Exception:  # noqa: BLE001 - a dead query is not a fatal run
            return None

        try:
            path.write_text(json.dumps({"q": q, "url": self.url,
                                        "fetched_at": _now().isoformat(),
                                        "results": results}, ensure_ascii=False),
                            encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
        return results

    def close(self):
        self._client.close()


def precision_of(hit: dict) -> str | None:
    """'poi' | 'street' | None. None means a centroid, which is not an answer."""
    cat = (hit.get("category") or hit.get("class") or "").lower()
    typ = (hit.get("type") or "").lower()
    if cat == "landuse":
        return "poi" if typ in _PRECISE_LANDUSE else None
    if cat == "place":
        return "street" if typ in _PRECISE_PLACE else None
    if cat == "boundary":
        return None
    return _PRECISE.get(cat)


def _state_agrees(hit: dict, college: dict,
                  aliases: dict[str, str]) -> tuple[bool, str]:
    addr = hit.get("address") or {}
    if (addr.get("country_code") or "").lower() not in ("in", ""):
        return False, f"country={addr.get('country_code')}"
    want = _state_key(college.get("state"), aliases)
    if not want:
        return False, "catalogue has no state to check against"
    got = _state_key(addr.get("state") or addr.get("state_district"), aliases)
    if got:
        return (True, want) if got == want else (False, f"state {got!r} != {want!r}")
    # No state in the structured address: the display name always carries the
    # full hierarchy, so fall back to that rather than waiving the check.
    if re.search(rf"\b{re.escape(want)}\b", _norm(_fold(hit.get("display_name")))):
        return True, want
    return False, f"state {want!r} not in the returned address"


def _place_agrees(hit: dict, college: dict,
                  aliases: dict[str, str]) -> tuple[bool, str]:
    """Does the returned address name the college's own city or district?

    Either will do. A campus on the edge of a city is routinely filed by OSM
    under the surrounding district — Sharda University sits in 'Gautam Buddha
    Nagar', which is exactly what the catalogue's district column says — and
    neither spelling is wrong.
    """
    addr = hit.get("address") or {}
    wanted = {k for k in (_place_key(_fold(college.get("city")), aliases),
                          _place_key(_fold(college.get("district")), aliases)) if k}
    if not wanted:
        return False, "catalogue has no city or district"
    got = {_place_key(_fold(addr.get(k)), aliases) for k in _PLACE_KEYS}
    got.discard("")
    if wanted & got:
        return True, sorted(wanted & got)[0]
    # Last chance: named anywhere in the display name. Word boundaries matter —
    # "pune" must not match inside "punegaon".
    hay = _norm(_fold(hit.get("display_name")))
    for w in sorted(wanted):
        if re.search(rf"\b{re.escape(w)}\b", hay):
            return True, w
    return False, f"place {sorted(wanted)} not in {sorted(g for g in got)[:4]}"


def agrees(hit: dict, college: dict, aliases: dict[str, str],
           query_kind: str) -> tuple[bool, str]:
    """Does this hit corroborate THIS college? Two evidence paths, both of
    which require the state.

    State is never waived: it is the one check that cannot be fooled by a
    duplicate name, and India has plenty of those.

      name query  -> the matched object's own name has to look like the
                     college's, AND either the city agrees (floor 0.60) or the
                     name is near-identical (floor 0.90). NALSAR is the case
                     that needs the second path: OSM files it under
                     Medchal-Malkajgiri while the catalogue says Hyderabad, and
                     the names match exactly.
      address query -> nothing to name-check against (the hit is a road or a
                     house number), so the city MUST agree.
    """
    state_ok, state_note = _state_agrees(hit, college, aliases)
    if not state_ok:
        return False, state_note
    place_ok, place_note = _place_agrees(hit, college, aliases)

    if query_kind == "address":
        if place_ok:
            return True, f"place={place_note} state={state_note} (address)"
        return False, place_note

    ratio, contain = name_agreement(_osm_name(hit), college.get("name") or "")
    if contain < NAME_CONTAINMENT_MIN:
        return False, (f"osm name {_osm_name(hit)!r} shares only "
                       f"{contain:.0%} of its words with the college")
    if place_ok and ratio >= NAME_RATIO_MIN:
        return True, (f"name~{ratio:.2f} place={place_note} state={state_note}")
    if ratio >= NAME_RATIO_SOLO:
        return True, f"name~{ratio:.2f} state={state_note} (no city agreement)"
    if place_ok:
        return False, f"name~{ratio:.2f} below {NAME_RATIO_MIN} (city agreed)"
    return False, f"name~{ratio:.2f} below {NAME_RATIO_SOLO} and {place_note}"


def _name_queries(name: str) -> list[str]:
    """The forms of a college name worth asking for, most complete first.

    Nominatim's free-form parser is CONJUNCTIVE: every token has to resolve, so
    a longer query is a narrower one, and adding the city to a name makes the
    search FAIL rather than sharpen it. Measured on the live corpus:

        "Galgotias University, Greater Noida, Uttar Pradesh, India"   0 results
        "Galgotias University"                                        1, correct

    OSM files Galgotias under Dankaur / Gautam Buddha Nagar, so the catalogue's
    "Greater Noida" contradicted the index and killed the whole query. The
    location is therefore not asked for — it is CHECKED, afterwards, by
    agrees(). Same for the trailing place or acronym the catalogue tacks onto a
    name: "Birla Institute of Technology, Mesra" and "Integral University - IUL"
    both return nothing, while their heads return the right campus.
    """
    clean = re.sub(r"\s+", " ", name.replace("&", " and ")).strip(" ,-")
    out = [clean]
    head = re.split(r"\s*[,(]|\s+-\s+", clean)[0].strip()
    if head and head != clean and len(_sig(head)) >= 2:
        out.append(head)
    return out


def geocode_college(c: dict, geo: Geocoder,
                    aliases: dict[str, str]) -> dict | None:
    """Coordinates for one college, or None. Never guesses.

    Three queries at most, best evidence first: the college's name, the name
    with the catalogue's trailing place/acronym removed, then the street
    address as a fallback for a campus OSM has never heard of.

    A "city, state" query is deliberately absent. It would succeed almost every
    time and it would be the town centre every time.
    """
    name = (c.get("name") or "").strip()
    address = (c.get("address") or "").strip()

    queries: list[tuple[str, str]] = []
    if name and (c.get("state") or "").strip():
        queries += [("name", q) for q in _name_queries(name)]
    if address:
        q = address if re.search(r"india\s*$", address, re.I) else f"{address}, India"
        queries.append(("address", q))

    rejects: list[str] = []
    for kind, q in queries:
        results = geo.search(q, limit=5 if kind == "name" else 3)
        if results is None:
            rejects.append(f"{kind}: not requested (cap/robots)")
            continue
        if not results:
            rejects.append(f"{kind}: no result")
            continue
        for hit in results:
            prec = precision_of(hit)
            if prec is None:
                rejects.append(
                    f"{kind}: {hit.get('category')}={hit.get('type')} is a centroid")
                continue
            ok, why = agrees(hit, c, aliases, kind)
            if not ok:
                rejects.append(f"{kind}: {why}")
                continue
            try:
                lat, lon = float(hit["lat"]), float(hit["lon"])
            except (KeyError, TypeError, ValueError):
                rejects.append(f"{kind}: unparseable coordinates")
                continue
            # India's envelope, as a last backstop against a coordinate that
            # passed every name check and still landed in the sea.
            if not (6.0 <= lat <= 37.6 and 68.0 <= lon <= 97.5):
                rejects.append(f"{kind}: {lat},{lon} outside India")
                continue
            return {
                "college_id": c["college_id"], "name": name,
                "lat": lat, "lon": lon, "precision": prec, "why": why,
                "query": q, "query_kind": kind,
                "osm": f"{hit.get('category')}={hit.get('type')}",
                "display_name": hit.get("display_name"),
                "confidence": _CONFIDENCE[prec],
                "source_url": (f"https://www.openstreetmap.org/"
                               f"{hit.get('osm_type')}/{hit.get('osm_id')}"
                               if hit.get("osm_id") else geo.url),
            }
    return {"college_id": c["college_id"], "name": name, "lat": None,
            "rejects": rejects} if rejects else None


# ---------------------------------------------------------------------------
# Approvals and accreditations, off the college's own site
# ---------------------------------------------------------------------------

# code, kind, full statutory names, acronym pattern, context words the acronym
# needs when the full name is absent.
#
# `needs_context` is the difference between a fact and a false positive. INC is
# the worst offender — every "Foo Systems Inc." footer in the country matches a
# bare \bINC\b — so it is only believed next to the word nursing, or spelled
# out. NBA and PCI and NMC and BCI are the same problem, smaller.
BODIES: tuple[tuple[str, str, tuple[str, ...], str, tuple[str, ...]], ...] = (
    ("AICTE", "approval",
     ("all india council for technical education",), r"\bAICTE\b", ()),
    ("UGC", "approval",
     ("university grants commission",), r"\bUGC\b", ()),
    ("PCI", "approval",
     ("pharmacy council of india",), r"\bPCI\b", ("pharm",)),
    ("NMC", "approval",
     ("national medical commission", "medical council of india"),
     r"\bNMC\b", ("medic", "mbbs", "md ", "ms ")),
    ("BCI", "approval",
     ("bar council of india",), r"\bBCI\b", ("law", "llb", "legal")),
    ("INC", "approval",
     ("indian nursing council",), r"\bINC\b", ("nurs",)),
    ("NCTE", "approval",
     ("national council for teacher education",), r"\bNCTE\b", ()),
    ("NBA", "accreditation",
     ("national board of accreditation",), r"\bNBA\b",
     ("accredit", "programme", "program", "b.tech", "engineering")),
)

# The verb that turns a body's NAME into a claim about this college.
#
# "affiliated" is deliberately NOT here, and it used to be. Affiliation is a
# relationship with a PARENT UNIVERSITY, not an approval by a statutory body,
# and including it produced exactly the failure this window is supposed to
# prevent: on Scottish Church College's homepage, "Affiliated to University of
# Calcutta" sat 60 characters from the words "UGC Public Self Disclosure" and
# was recorded as a UGC approval the page never claimed.
_VERB = re.compile(
    r"\b(approv\w*|recogni[sz]\w*|permitted|permission|accredit\w*|sanction\w*)\b",
    re.I)

# A YEAR ONLY COUNTS IF THE PAGE TIES IT TO THE APPROVAL.
#
# verify_facts.stated_year() is calibrated for fee pages: it accepts a year
# vouched for by the word "admission" or "session", which is right there and
# wrong here. Run against an approval snippet it produced two dated facts that
# were not dated at all — an AICTE approval stamped 2026 because "Admissions
# 2026" was in the same nav bar, and a UGC one stamped 2026-27 off "PG
# Admission 2026-27". Both would have been read as "approval current for
# 2026", which is the undated-figure bug wearing a different hat.
#
# So the year has to be introduced by a validity phrase. Statutory approvals
# genuinely do carry one when they state it — AICTE issues an Extension of
# Approval per session — and when they do not, NULL is the true answer.
_VALIDITY_ALT = (
    r"valid(?:ity)?|upto|up\s*to|till|until|extension\s+of\s+approval"
    r"|e\.?\s*o\.?\s*a\.?|l\.?\s*o\.?\s*a\.?|granted\s+for"
    r"|approved\s+for\s+the|recogni[sz]ed\s+for\s+the")
_VALIDITY = re.compile(rf"\b(?:{_VALIDITY_ALT})\b", re.I)


# Anything that makes the approval NOT a present fact. "provisional" is not
# here on its own — a provisional approval is still an approval — but every
# form of "we do not have one yet" is.
_NEGATION = re.compile(
    r"\b(not\s+(?:yet\s+)?(?:approv|recogni|accredit)\w*"
    r"|non[-\s]?approv\w*|un[-\s]?approv\w*|without\s+(?:the\s+)?approval"
    r"|approval\s+(?:is\s+)?(?:awaited|pending|in\s+process|under\s+process)"
    r"|(?:awaiting|applied\s+for|apply\s+for|seeking|in\s+the\s+process\s+of)"
    r"\s+(?:the\s+)?(?:approval|recognition|accreditation)"
    r"|yet\s+to\s+be\s+(?:approv|recogni|accredit)\w*"
    r"|(?:de[-\s]?recogni|withdraw|cancell|revok|lapsed|suspend)\w*"
    r"|fake|bogus|not\s+affiliated)\b", re.I)

# UGC's exam and journal brands, which are not approvals of anything. A page
# that mentions one of these still counts if it ALSO carries a real recognition
# phrase — Section 2(f) / 12(B) of the UGC Act is the standard wording, and a
# university that states it usually coaches for UGC-NET as well.
_UGC_NOISE = re.compile(r"\bUGC[-\s]*(NET|CARE|JRF|CSIR|NTA)\b", re.I)
_UGC_REAL = re.compile(
    r"2\s*\(\s*f\s*\)|12\s*\(?\s*b\s*\)?"
    r"|\b(?:recogni[sz]ed|approved)\s+by\b|\bunder\s+section\b", re.I)

# A statement ABOUT the category, not about this college. A college describes
# itself in the singular — "an AICTE approved institute" — while "AICTE
# approved colleges in Maharashtra" is a directory blurb that says nothing
# about who is hosting it. Plural is the tell, and it is the only one that
# separates the two reliably.
_GENERIC_LISTING = re.compile(
    r"\b(approved|recognised|recognized|accredited)\s+"
    r"(?:\w+\s+){0,2}(colleges|institutes|institutions|universities|schools)\b"
    r"|\b(?:list|directory|top\s+\d+)\s+of\b", re.I)

# THE BODY APPROVED A DOCUMENT, NOT THIS COLLEGE.
#
# Found by hand-checking eight random written records: Shivalik College's
# notice board carries "PCI Approved Syllabus D.Pharm", and a syllabus approved
# by the Pharmacy Council says nothing about whether the college itself holds
# PCI approval. Same shape for a curriculum, a textbook or a form.
_APPROVED_ARTEFACT = re.compile(
    r"\b(?:approved|recognised|recognized|accredited|prescribed)\s+"
    r"(?:\w+\s+){0,1}"
    r"(syllabus|syllabi|curriculum|curricula|textbooks?|books?|format|formats"
    r"|template|templates|forms?|proforma|guidelines?|scheme|schemes)\b", re.I)

# The window either side of the body's name that the verb has to fall inside.
#
# 90, not 120, and the difference is measurable. The longest legitimate gap is
# a verb in front of a spelled-out statutory name — "approved by the All India
# Council for Technical Education" is 13 characters of verb-to-name — so 90 is
# already generous. Beyond that the window starts reaching into the NEXT
# sentence, and a verb from an unrelated claim then vouches for a body that is
# only mentioned in passing, which is how a nav bar full of acronyms turns into
# seven approvals this college never claimed.
APPROVAL_WINDOW = 90
EVIDENCE_WIDTH = 150

# Pages worth one request each, beyond the homepage. Approvals live on
# about//recognition/mandatory-disclosure pages far more often than on the
# front page.
APPROVAL_PAGE_KEYS = (
    "approval", "approvals", "recognition", "recognitions", "accreditation",
    "accreditations", "mandatory-disclosure", "mandatory disclosure",
    "statutory", "affiliation", "about-us", "about us", "aboutus", "/about",
    "naac", "nba", "aicte", "ugc", "iqac", "profile", "overview",
)
MAX_APPROVAL_PAGES = int(os.getenv("MIVI_APPROVAL_PAGES", "3"))


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


# Regulators this harvester does NOT collect, listed anyway.
#
# The point of this list is not what to extract — it is knowing which body a
# verb or a date BELONGS to, and an approvals page prints every regulator's
# letters in one block. Two measured failures come straight from a name being
# missing here: NAAC's "Re-Accredited" was read as a UGC approval, and Indus
# University's "COA Approval 2024 25" (Council of Architecture) dated an AICTE
# row whose own newest letter was 2020-21. Anything that shows up next to an
# approval verb on an Indian college site belongs in this list.
_VERB_OWNERS = BODIES + (
    ("NAAC", "accreditation",
     ("national assessment and accreditation council",), r"\bNAAC\b", ()),
    ("COA", "approval", ("council of architecture",), r"\bCOA\b", ()),
    ("DCI", "approval", ("dental council of india",), r"\bDCI\b", ()),
    ("VCI", "approval", ("veterinary council of india",), r"\bVCI\b", ()),
    ("RCI", "approval", ("rehabilitation council of india",), r"\bRCI\b", ()),
    ("NCISM", "approval",
     ("national commission for indian system of medicine",
      "central council of indian medicine"), r"\bNCISM|\bCCIM\b", ()),
    ("AIU", "approval", ("association of indian universities",), r"\bAIU\b", ()),
    ("ICAR", "approval",
     ("indian council of agricultural research",), r"\bICAR\b", ()),
    ("CCH", "approval", ("central council of homoeopathy",), r"\bCCH\b", ()),
)


def _body_spots(text: str) -> list[tuple[int, int, str]]:
    """Every statutory body named in this text: (start, end, code)."""
    spots: list[tuple[int, int, str]] = []
    low = text.lower()
    for code, _kind, full_names, acro_re, _ctx in _VERB_OWNERS:
        for full in full_names:
            spots += [(m.start(), m.end(), code) for m in re.finditer(re.escape(full), low)]
        spots += [(m.start(), m.end(), code) for m in re.finditer(acro_re, text)]
    return spots


# The academic year sitting NEXT TO a validity phrase, in either order —
# "EOA-2025-2026" and "2024 - 2025 Extension of Approval" are both how AICTE's
# Extension of Approval is listed, and both are real.
#
# This exists because verify_facts.stated_year() cannot see them, and that is
# not a bug in stated_year: it only believes a year that a fee/admission/
# session word vouches for, which is right for the fee pages it was built for
# and simply absent from an approvals page. Measured on the live run, it read
# M.S. Ramaiah's EOA year only because the words "Academic ... Audit" happened
# to be in the same blob — the right answer for the wrong reason.
#
# So the vouching word here is the validity phrase itself. This is not a second
# opinion on the same question; stated_year is still asked first, and the fact
# gate is still verify_facts.verify().
_AY = r"\b(?P<{0}s>20[0-4]\d)\s*[-/–]\s*(?P<{0}e>\d{{2}}|20[0-4]\d)\b"
_EOA_YEAR = re.compile(
    rf"\b(?:{_VALIDITY_ALT})\b[^.\n]{{0,40}}?{_AY.format('a')}"
    rf"|{_AY.format('b')}[^.\n]{{0,25}}?\b(?:{_VALIDITY_ALT})\b", re.I)


def approval_year(window: str) -> str | None:
    """The session an approval says it is valid for, or None.

    None is the ordinary answer and it is a true one: an approval page usually
    states that a body approved the college, not for how long.
    """
    if not _VALIDITY.search(window):
        return None
    got = stated_year(window)
    if got:
        return got

    cap = datetime.now(timezone.utc).year + 1
    best: tuple[int, str] | None = None
    for m in _EOA_YEAR.finditer(window):
        d = m.groupdict()
        raw_start = d.get("as") or d.get("bs")
        tail = d.get("ae") or d.get("be")
        if not raw_start or not tail:
            continue
        start = int(raw_start)
        end = int(tail) if len(tail) == 4 else int(str(start)[:2] + tail)
        # An academic session spans one or two calendar years, never more:
        # "2025-2028" is a three-year approval PERIOD, not the session it is
        # for, and reading it as one would date the approval to a year the
        # page never claims.
        if not 2000 <= start <= cap or end - start not in (0, 1):
            continue
        # The latest wins: these pages list every EOA ever granted, oldest
        # first, and the oldest is the one thing the approval is certainly NOT
        # current for.
        if best is None or start > best[0]:
            best = (start, f"{start}-{str(end)[-2:]}")
    return best[1] if best else None


def _gap(a0: int, a1: int, b0: int, b1: int) -> int:
    """Characters between two spans; 0 if they touch or overlap."""
    if a1 <= b0:
        return b0 - a1
    if b1 <= a0:
        return a0 - b1
    return 0


def _own_slice(window: str, body_start: int, body_end: int, code: str) -> str:
    """The part of the window that belongs to THIS body — cut at the nearest
    other body's name on each side.

    Dates need this even more than verbs do. An approvals page lists every
    regulator's letters together, and the years belong to whichever letter they
    are printed under. Measured on the live run, out of 575 statements only
    four carried a year and two of those four were wrong for exactly this
    reason: M.S. Ramaiah's UGC recognition ("UGC 2(f)-2016 UGC 2(f)-2021") was
    stamped 2025-26 off the AICTE EOA list beside it, and Millia Institute's
    UGC row was stamped 2002-03 off AICTE's oldest EOA report. Both would have
    been read as the year the UGC recognition is current for.
    """
    others = [(s, e) for s, e, c in _body_spots(window) if c != code]
    lo = max([e for s, e in others if e <= body_start], default=0)
    hi = min([s for s, e in others if s >= body_end], default=len(window))
    return window[lo:hi]


def _verb_belongs_to(window: str, body_start: int, body_end: int,
                     code: str) -> bool:
    """Is there an approval verb in this window that belongs to THIS body?

    A verb is owned by whichever body it sits closest to. Without this, one
    body's verb vouches for another's name a few words away, which is not a
    hypothetical: Scottish Church College's homepage carries "NAAC
    Re-Accredited Grade 'A' Institution (3rd Cycle) UGC Public Self
    Disclosure", and "Re-Accredited" — which belongs to NAAC, one character
    away — was being read as a UGC approval 34 characters later. The college
    claims no UGC approval anywhere on that page.
    """
    others = [(s, e) for s, e, c in _body_spots(window) if c != code]
    for m in _VERB.finditer(window):
        mine = _gap(m.start(), m.end(), body_start, body_end)
        if others and min(_gap(m.start(), m.end(), s, e) for s, e in others) < mine:
            continue
        return True
    return False


def find_approvals(page_text: str) -> list[dict]:
    """Every statutory body this page claims, with the sentence that says so.

    The returned `evidence` is a literal substring of `page_text` — that is the
    verbatim guarantee, and it is structural rather than checked afterwards.
    """
    if not page_text:
        return []
    out: dict[str, dict] = {}
    low = page_text.lower()

    for code, kind, full_names, acro_re, needs_context in BODIES:
        spots: list[tuple[int, int, bool]] = []   # start, end, spelled_out
        for full in full_names:
            for m in re.finditer(re.escape(full), low):
                spots.append((m.start(), m.end(), True))
        for m in re.finditer(acro_re, page_text):
            spots.append((m.start(), m.end(), False))
        if not spots:
            continue

        for start, end, spelled_out in spots:
            lo = max(0, start - APPROVAL_WINDOW)
            window = page_text[lo:end + APPROVAL_WINDOW]
            if not _verb_belongs_to(window, start - lo, end - lo, code):
                continue
            if _NEGATION.search(window):
                continue
            if _GENERIC_LISTING.search(window):
                continue
            if _APPROVED_ARTEFACT.search(window):
                continue
            if (code == "UGC" and _UGC_NOISE.search(window)
                    and not _UGC_REAL.search(window)):
                continue
            if not spelled_out and needs_context:
                if not any(w in window.lower() for w in needs_context):
                    continue

            ev_lo = max(0, start - EVIDENCE_WIDTH)
            evidence = _clean(page_text[ev_lo:end + EVIDENCE_WIDTH])
            year = approval_year(_own_slice(window, start - lo, end - lo, code))
            # A spelled-out statutory name is much harder to hit by accident
            # than three capital letters, so it is worth more.
            conf = 0.82 if spelled_out else 0.7
            prev = out.get(code)
            if prev is None or conf > prev["confidence"]:
                out[code] = {"body": code, "kind": kind, "confidence": conf,
                             "evidence": evidence, "spelled_out": spelled_out,
                             "year": year}
    return list(out.values())


def pick_approval_pages(base_url: str,
                        links: list[tuple[str, str]]) -> list[str]:
    """Subpages likely to state approvals, on this college's own domain only."""
    origin = urlparse(base_url).netloc
    chosen: list[str] = []
    seen: set[str] = set()
    for href, anchor in links:
        if not href or href.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue
        absolute = urljoin(base_url, href)
        p = urlparse(absolute)
        if p.scheme not in ("http", "https") or p.netloc != origin:
            continue
        if absolute in seen:
            continue
        hay = f"{p.path.lower()} {(anchor or '').lower()}"
        if any(k in hay for k in APPROVAL_PAGE_KEYS):
            seen.add(absolute)
            chosen.append(absolute)
        if len(chosen) >= MAX_APPROVAL_PAGES - 1:
            break
    return chosen


async def approvals_for_college(client: httpx.AsyncClient, pol: Politeness,
                                c: dict, fc: FirecrawlBudget | None) -> dict:
    """Fetch a college's own site and read its approvals off it.

    Everything found is handed to verify_facts.verify() before it counts. That
    gate owns the wrong-entity check, which is the one that matters here: a
    parked domain or a redirect to a portal would otherwise let one college's
    approvals be recorded against another's id.
    """
    site = normalise_site(c.get("website_url"))
    result = {"college_id": c["college_id"], "name": c.get("name"),
              "found": [], "pages": 0, "status": "no_site"}
    if not site:
        return result

    html = await fetch(client, pol, site, fc)
    if not html:
        result["status"] = "unreachable"
        return result
    text, links = parse_html(html)
    result["pages"] = 1
    pages = [(site, text)]

    for url in pick_approval_pages(site, links):
        sub = await fetch(client, pol, url, fc)
        if not sub:
            continue
        sub_text, _ = parse_html(sub)
        if sub_text:
            pages.append((url, sub_text))
            result["pages"] += 1

    best: dict[str, dict] = {}
    for url, page_text in pages:
        for hit in find_approvals(page_text):
            fact = {
                "summary": hit["evidence"],
                "facts": {hit["body"]: ("accredited" if hit["kind"] == "accreditation"
                                        else "approved")},
                "confidence": hit["confidence"],
            }
            v = verify_fact(fact, page_text, c.get("name") or "", c.get("city"))
            if not v.ok:
                continue
            year = hit["year"]
            rec = {"body": hit["body"], "kind": hit["kind"],
                   "statement": hit["evidence"], "source_url": url,
                   "stated_year": year, "confidence": round(v.confidence, 3)}
            prev = best.get(hit["body"])
            if prev is None or rec["confidence"] > prev["confidence"]:
                best[hit["body"]] = rec

    result["found"] = sorted(best.values(), key=lambda r: r["body"])
    result["status"] = "ok" if best else "nothing_stated"
    return result


# ---------------------------------------------------------------------------
# NAAC — a join, not a crawl
# ---------------------------------------------------------------------------

def naac_accreditations(college_ids: list[str]) -> dict[str, dict]:
    """The NAAC grade already in registry_fact, shaped for the column.

    Re-crawling this would be strictly worse: registry_fact carries the NAAC's
    own file, its "as on" date and the declaration date, which no college
    website states as reliably.
    """
    if not college_ids:
        return {}
    be = get_backend()
    out: dict[str, dict] = {}
    for i in range(0, len(college_ids), 500):
        chunk = college_ids[i:i + 500]
        marks = ",".join("?" * len(chunk))
        rows = be.q(
            f"SELECT college_id, payload, source_url, year, fetched_at "
            f"FROM registry_fact WHERE registry = 'naac' "
            f"AND college_id IN ({marks})", chunk)
        for r in rows:
            p = r["payload"] or {}
            if isinstance(p, str):
                try:
                    p = json.loads(p)
                except Exception:  # noqa: BLE001
                    p = {}
            if not p.get("grade"):
                continue
            out[r["college_id"]] = {
                "body": "NAAC", "kind": "accreditation",
                "grade": p.get("grade"), "cgpa": p.get("cgpa"),
                "cycle": p.get("cycle"), "declared": p.get("declared"),
                "as_on": p.get("as_on"), "source": "naac",
                "source_url": r["source_url"], "stated_year": r["year"],
                # The date the NAAC's own file was read, NOT now. Re-stamping it
                # with today would tell the nightly refresh this grade is fresh
                # when it is as old as the spreadsheet it came from.
                "fetched_at": r["fetched_at"],
            }
    return out


# ---------------------------------------------------------------------------
# Persist
# ---------------------------------------------------------------------------

CF_UPSERT = (
    "INSERT INTO college_field "
    "(college_id, field, value, value_num, source, source_url, stated_year, "
    " fetched_at, confidence) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
    "ON CONFLICT (college_id, field, source) DO UPDATE SET "
    "value = EXCLUDED.value, value_num = EXCLUDED.value_num, "
    "source_url = EXCLUDED.source_url, stated_year = EXCLUDED.stated_year, "
    "fetched_at = EXCLUDED.fetched_at, confidence = EXCLUDED.confidence"
)


def write_geo(hits: list[dict], dry_run: bool) -> int:
    """lat/long onto colleges, and one dated row per coordinate in college_field.

    Both, always. A value on `colleges` alone is undated and invisible to the
    nightly refresh, which is the whole reason college_field exists.
    """
    good = [h for h in hits if h.get("lat") is not None]
    if dry_run or not good:
        return len(good)
    be = get_backend()
    be.executemany(
        "UPDATE colleges SET latitude = ?, longitude = ? WHERE college_id = ?",
        [(h["lat"], h["lon"], h["college_id"]) for h in good])

    now = _now()
    rows = []
    for h in good:
        for field, val in (("latitude", h["lat"]), ("longitude", h["lon"])):
            rows.append((h["college_id"], field, f"{val:.7f}", float(val),
                         "nominatim", h["source_url"],
                         # A coordinate has no academic year, and inventing one
                         # would make an undated fact look dated.
                         None, now, h["confidence"]))
    be.executemany(CF_UPSERT, rows)
    return len(good)


def write_approvals(results: list[dict], naac: dict[str, dict],
                    dry_run: bool) -> tuple[int, int]:
    """approvals / accreditations onto colleges + college_field. Returns
    (colleges with an approval, colleges with an accreditation)."""
    col_rows, cf_rows = [], []
    now = _now()
    n_appr = n_accr = 0

    by_id = {r["college_id"]: r for r in results}
    ids = set(by_id) | set(naac)
    for cid in sorted(ids):
        res = by_id.get(cid)
        found = (res or {}).get("found") or []
        approvals = [f for f in found if f["kind"] == "approval"]
        accreds = [f for f in found if f["kind"] == "accreditation"]
        if cid in naac:
            accreds = [naac[cid]] + accreds

        if not approvals and not accreds:
            continue
        if approvals:
            n_appr += 1
        if accreds:
            n_accr += 1
        # `fetched_at` is provenance for college_field, not content for the
        # column, and it is a datetime — which json.dumps refuses. Dropping it
        # here keeps the column a plain document and keeps the date in the one
        # place the refresh job reads.
        col_accreds = [{k: v for k, v in a.items() if k != "fetched_at"}
                       for a in accreds]
        col_rows.append((
            json.dumps(approvals, ensure_ascii=False) if approvals else None,
            json.dumps(col_accreds, ensure_ascii=False) if col_accreds else None,
            cid))

        if approvals:
            # One row for the SET of approvals: they come from the same pass on
            # the same site and refresh together, so splitting them across rows
            # would only give the nightly job seven copies of one decision.
            best = max(approvals, key=lambda a: a["confidence"])
            # The YEAR is the newest any of them states — NOT the year of
            # whichever body scored highest, which is what this used to take.
            # M.S. Ramaiah's AICTE EOA is dated 2025-26 and its UGC row is
            # undated; because UGC was the more confident match the tracked
            # field said "no year", hiding a date the harvest had in hand.
            years = sorted(a["stated_year"] for a in approvals if a["stated_year"])
            cf_rows.append((cid, "approvals",
                            ", ".join(a["body"] for a in approvals), None,
                            "college_site", best["source_url"],
                            years[-1] if years else None, now,
                            round(min(a["confidence"] for a in approvals), 3)))
        site_accr = [a for a in accreds if a.get("source") != "naac"]
        if site_accr:
            best = max(site_accr, key=lambda a: a["confidence"])
            cf_rows.append((cid, "accreditations",
                            ", ".join(a["body"] for a in site_accr), None,
                            "college_site", best["source_url"],
                            best["stated_year"], now,
                            round(min(a["confidence"] for a in site_accr), 3)))
        if cid in naac:
            # The NAAC contribution to this column, tracked under the COLUMN's
            # own name. college_field already carries it as `naac_grade`, but a
            # refresh query asks "when was accreditations last checked" — and
            # against the old row it would find nothing and conclude the column
            # was never sourced. Same fact, addressed by the name the client's
            # spec uses. 0.95 because it is the NAAC's own published list.
            n = naac[cid]
            cf_rows.append((cid, "accreditations",
                            f"NAAC {n['grade']}", n.get("cgpa"), "naac",
                            n["source_url"], n["stated_year"],
                            n.get("fetched_at") or now, 0.95))

    if dry_run or not col_rows:
        return n_appr, n_accr
    be = get_backend()
    be.executemany(
        "UPDATE colleges SET approvals = COALESCE(?::jsonb, approvals), "
        "accreditations = COALESCE(?::jsonb, accreditations) "
        "WHERE college_id = ?", col_rows)
    if cf_rows:
        be.executemany(CF_UPSERT, cf_rows)
    return n_appr, n_accr


# ---------------------------------------------------------------------------
# Selection + drivers
# ---------------------------------------------------------------------------

def tier_colleges(tier: int, limit: int | None, missing: str | None) -> list[dict]:
    where = ["p.tier = ?"]
    params: list = [tier]
    if missing == "geo":
        where.append("c.latitude IS NULL")
    elif missing == "approvals":
        where.append("c.approvals IS NULL")
    sql = ("SELECT c.college_id, c.name, c.city, c.district, c.state, "
           "c.address, c.website_url FROM college_priority p "
           "JOIN colleges c USING(college_id) "
           f"WHERE {' AND '.join(where)} ORDER BY p.score DESC")
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    return get_backend().q(sql, params)


def run_geocode(tier: int, limit: int | None, dry_run: bool,
                respect_robots: bool, only_missing: bool) -> dict:
    aliases = _load_place_aliases()
    todo = tier_colleges(tier, limit, "geo" if only_missing else None)
    geo = Geocoder(respect_robots=respect_robots)
    if respect_robots and not geo.robots_allows():
        print(f"[geo] {urlparse(geo.url).netloc} robots.txt disallows "
              f"{urlparse(geo.url).path} — refusing to geocode. Point "
              f"MIVI_NOMINATIM_URL at a self-hosted instance, or drop "
              f"--respect-robots after reading the note in this module.",
              file=sys.stderr)
        geo.close()
        return {"colleges": len(todo), "stored": 0, "refused": 0,
                "blocked_by_robots": True}

    hits, refused, t0 = [], [], time.time()
    stored = 0
    pending: list[dict] = []      # written every CHECKPOINT_EVERY colleges

    # At one request per second a full tier-1 pass is three quarters of an
    # hour. Holding every row until the end would mean a Ctrl-C at minute 40
    # wrote nothing, so results land in batches; the queries themselves are
    # cached on disk, so the restart is fast either way.
    CHECKPOINT_EVERY = 100

    for i, c in enumerate(todo, 1):
        r = geocode_college(dict(c), geo, aliases)
        if r and r.get("lat") is not None:
            hits.append(r)
            pending.append(r)
        else:
            refused.append(r or {"college_id": c["college_id"],
                                 "name": c["name"], "rejects": ["no query built"]})
        if len(pending) >= CHECKPOINT_EVERY:
            stored += write_geo(pending, dry_run)
            pending = []
        if i % 25 == 0 or i == len(todo):
            print(f"[geo] {i}/{len(todo)} kept={len(hits)} refused={len(refused)} "
                  f"requests={geo.requests} cached={geo.cache_hits} "
                  f"{(time.time() - t0) / 60:.1f}m", file=sys.stderr)
        if geo.requests >= geo.max_requests:
            print(f"[geo] request cap {geo.max_requests} reached — stopping "
                  f"cleanly, re-run to continue (the cache makes it free)",
                  file=sys.stderr)
            break
    geo.close()

    stored += write_geo(pending, dry_run)
    return {"colleges": len(todo), "attempted": len(hits) + len(refused),
            "stored": stored, "refused": len(refused),
            "requests": geo.requests, "cache_hits": geo.cache_hits,
            "dry_run": dry_run, "samples": hits[:8],
            "reject_samples": [r for r in refused if r.get("rejects")][:8],
            "precision": {p: sum(1 for h in hits if h["precision"] == p)
                          for p in ("poi", "street")}}


async def _approvals_sweep(todo: list[dict], concurrency: int, delay: float,
                           fc: FirecrawlBudget | None) -> list[dict]:
    pol = Politeness(delay=delay)
    sem = asyncio.Semaphore(concurrency)
    out: list[dict] = []
    t0 = time.time()

    async with httpx.AsyncClient(headers=HEADERS, timeout=20.0,
                                 follow_redirects=True) as client:
        if fc is not None:
            await fc.prime(client)

        async def one(c: dict):
            async with sem:
                try:
                    r = await approvals_for_college(client, pol, dict(c), fc)
                except Exception as exc:  # noqa: BLE001 - one bad site != dead run
                    r = {"college_id": c["college_id"], "name": c.get("name"),
                         "found": [], "pages": 0, "status": f"error: {exc}"[:120]}
                out.append(r)
                if len(out) % 25 == 0:
                    got = sum(1 for x in out if x["found"])
                    print(f"[approvals] {len(out)}/{len(todo)} with-facts={got} "
                          f"{(time.time() - t0) / 60:.1f}m", file=sys.stderr)

        await asyncio.gather(*(one(c) for c in todo))
    return out


def run_approvals(tier: int, limit: int | None, dry_run: bool,
                  concurrency: int, delay: float, firecrawl_reserve: int | None,
                  only_missing: bool, dump: str | None = None) -> dict:
    universe = tier_colleges(tier, limit, "approvals" if only_missing else None)
    # The NAAC join covers the WHOLE tier, not just the crawlable part. A
    # college with no website still has its NAAC grade sitting in
    # registry_fact, and gating that on owning a website would throw away the
    # one accreditation we already hold for it.
    todo = [c for c in universe if normalise_site(c.get("website_url"))]

    fc = None
    if firecrawl_reserve is not None:
        fc = FirecrawlBudget(reserve=firecrawl_reserve)
    results = asyncio.run(_approvals_sweep(todo, concurrency, delay, fc))

    naac = naac_accreditations([c["college_id"] for c in universe])
    n_appr, n_accr = write_approvals(results, naac, dry_run)

    # EVERY fact, with the sentence it came from, so the run can actually be
    # hand-checked. A summary count cannot tell you that a nav bar was read as
    # an approval — reading the statements can, and that is how the two real
    # defects in this extractor were found.
    if dump:
        with open(dump, "w", encoding="utf-8") as fh:
            for r in results:
                for f in r["found"]:
                    fh.write(json.dumps({"college_id": r["college_id"],
                                         "name": r["name"], **f},
                                        ensure_ascii=False) + "\n")
        print(f"[approvals] {sum(len(r['found']) for r in results)} statements "
              f"written to {dump} — read them", file=sys.stderr)

    bodies: dict[str, int] = {}
    for r in results:
        for f in r["found"]:
            bodies[f["body"]] = bodies.get(f["body"], 0) + 1
    status: dict[str, int] = {}
    for r in results:
        key = r["status"].split(":")[0]
        status[key] = status.get(key, 0) + 1

    return {"colleges": len(universe), "crawled": len(todo),
            "no_website": len(universe) - len(todo),
            "with_approvals": n_appr, "with_accreditations": n_accr,
            "yielded_nothing": sum(1 for r in results if not r["found"]),
            "naac_joined": len(naac), "bodies": bodies, "status": status,
            "dry_run": dry_run,
            "samples": [r for r in results if r["found"]][:6]}


def redate_from_dump(path: str, dry_run: bool) -> dict:
    """Re-read a saved harvest's statements with the CURRENT dating rules.

    The dump holds every statement verbatim, so a fix to how a year is
    attributed can be applied to it without going back to 777 college websites.
    That matters here: the crawl budget is the scarce thing, and re-crawling to
    correct one date would cost more politeness than the correction is worth.

    It only ever moves `stated_year`. Nothing new is claimed, no body is added
    or removed, and a statement whose body can no longer be located in its own
    stored text loses its year rather than keeping an unverifiable one.
    """
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    changed: list[tuple] = []
    for r in rows:
        redone = {f["body"]: f.get("year") for f in find_approvals(r["statement"])}
        now_year = redone.get(r["body"])
        if now_year != r.get("stated_year"):
            changed.append((r["college_id"], r["name"], r["body"],
                            r.get("stated_year"), now_year))
    if dry_run:
        return {"statements": len(rows), "changed": len(changed),
                "dry_run": dry_run,
                "diffs": [{"college": n, "body": b, "was": w, "now": x}
                          for _, n, b, w, x in changed][:20]}

    be = get_backend()
    # The column first: each element of the approvals array carries its own
    # stated_year, so the array is rewritten element-wise rather than replaced.
    for cid, _name, body, _was, now_year in changed:
        be.execute(
            "UPDATE colleges SET approvals = ("
            "  SELECT jsonb_agg(CASE WHEN e->>'body' = ? "
            "                        THEN jsonb_set(e, '{stated_year}', "
            "                                       COALESCE(to_jsonb(?::text), 'null')) "
            "                        ELSE e END) "
            "  FROM jsonb_array_elements(approvals) e) "
            "WHERE college_id = ? AND approvals IS NOT NULL",
            (body, now_year, cid))
    # Then RECONCILE the tracked field against the column, for every college
    # with approvals rather than only the ones just changed. The two can drift
    # for a reason that has nothing to do with this dump — the tracked year
    # used to be copied off the highest-confidence body instead of the newest
    # dated one — and a reconcile that only visits changed rows would leave
    # that drift in place forever.
    synced = be.execute(
        "UPDATE college_field cf SET stated_year = sub.y FROM ("
        "  SELECT c.college_id, max(e->>'stated_year') AS y"
        "    FROM colleges c, jsonb_array_elements(c.approvals) e"
        "   WHERE c.approvals IS NOT NULL GROUP BY 1) sub "
        "WHERE cf.college_id = sub.college_id AND cf.field = 'approvals' "
        "  AND cf.source = 'college_site' "
        "  AND cf.stated_year IS DISTINCT FROM sub.y")
    return {"statements": len(rows), "changed": len(changed),
            "colleges_touched": len({c for c, *_ in changed}),
            "tracked_rows_synced": synced, "dry_run": dry_run,
            "diffs": [{"college": n, "body": b, "was": w, "now": x}
                      for _, n, b, w, x in changed][:20]}


def run_naac_only(tier: int, limit: int | None, dry_run: bool,
                  only_missing: bool) -> dict:
    """The NAAC half of `accreditations`, with no crawling at all.

    Worth its own entry point for two reasons. It is a JOIN — registry_fact
    already holds 2,504 grades — so it costs nothing and can be re-run after
    any change to how the column is shaped. And it is not limited to colleges
    with a website, which the approvals sweep necessarily is, so it reaches the
    tier-1 colleges that path can never touch.
    """
    universe = tier_colleges(tier, limit, "approvals" if only_missing else None)
    naac = naac_accreditations([c["college_id"] for c in universe])
    _, n_accr = write_approvals([], naac, dry_run)
    return {"colleges": len(universe), "naac_joined": len(naac),
            "with_accreditations": n_accr, "dry_run": dry_run,
            "grades": {g: sum(1 for v in naac.values() if v["grade"] == g)
                       for g in sorted({v["grade"] for v in naac.values()})}}


def report() -> None:
    be = get_backend()
    row = be.q("""
        SELECT COUNT(*) AS tier1,
               COUNT(c.latitude) AS geo,
               COUNT(c.approvals) AS approvals,
               COUNT(c.accreditations) AS accreditations
        FROM college_priority p JOIN colleges c USING(college_id)
        WHERE p.tier = 1""")[0]
    print(f"tier-1 colleges       {row['tier1']}")
    print(f"  with lat/long       {row['geo']}")
    print(f"  with approvals      {row['approvals']}")
    print(f"  with accreditations {row['accreditations']}")
    print("\ncollege_field rows written by this harvester:")
    for r in be.q("SELECT field, source, COUNT(*) n, MIN(confidence) lo, "
                  "MAX(confidence) hi, MAX(fetched_at) last FROM college_field "
                  "WHERE field IN ('latitude','longitude','approvals',"
                  "'accreditations') GROUP BY 1,2 ORDER BY 1,2"):
        print(f"  {r['field']:<16} {r['source']:<14} {r['n']:>6}  "
              f"conf {r['lo']:.2f}-{r['hi']:.2f}  last {r['last']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geocode", action="store_true")
    ap.add_argument("--approvals", action="store_true")
    ap.add_argument("--naac", action="store_true",
                    help="promote NAAC from registry_fact into accreditations. "
                         "A join, not a crawl — no network at all.")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--tier", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="do everything except write. Always run this first.")
    ap.add_argument("--all", action="store_true",
                    help="re-do colleges that already have a value")
    ap.add_argument("--respect-robots", action="store_true",
                    help="refuse the geocoder if its robots.txt disallows the "
                         "path (see the module note on robots vs usage policy)")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="approvals sweep, ACROSS domains — never within one")
    ap.add_argument("--delay", type=float, default=1.5,
                    help="seconds between requests to the same host")
    ap.add_argument("--firecrawl-reserve", type=int, default=None,
                    help="enable the Firecrawl rescue, holding back this many "
                         "credits per key. Off by default: approvals are not "
                         "worth the month's allowance.")
    ap.add_argument("--dump", default=None, metavar="FILE",
                    help="write every approval statement found, with its "
                         "source page, as JSONL — for hand-checking a run")
    ap.add_argument("--redate", default=None, metavar="FILE",
                    help="re-apply the current dating rules to a saved --dump "
                         "and correct stated_year in place. No crawling.")
    a = ap.parse_args()

    if a.report or not (a.geocode or a.approvals or a.naac or a.redate):
        report()
        return
    out: dict = {}
    if a.redate:
        out["redate"] = redate_from_dump(a.redate, a.dry_run)
    if a.naac:
        out["naac"] = run_naac_only(a.tier, a.limit, a.dry_run, not a.all)
    if a.geocode:
        out["geocode"] = run_geocode(a.tier, a.limit, a.dry_run,
                                     a.respect_robots, not a.all)
    if a.approvals:
        out["approvals"] = run_approvals(a.tier, a.limit, a.dry_run,
                                         a.concurrency, a.delay,
                                         a.firecrawl_reserve, not a.all,
                                         a.dump)
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
