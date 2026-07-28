"""Find a college's fees/scholarship/hostel/placement pages WITHOUT its homepage links.

WHY THIS EXISTS
The harvest sweep's biggest loss is not blocked sites, it is silence: 62% of the
colleges it crawled came back `empty`. `web_enrich.pick_pages()` can only follow
`<a href>` tags parsed out of the homepage, and on Indian college sites that
fails in three ways:

  * the nav is built in JavaScript, so a plain fetch sees a shell with no links
    (measured: www.iitjammu.ac.in serves the same 8,255-byte SPA body for every
    path on the host);
  * the fee page exists but is buried under Admissions with no matching word in
    the href or the anchor text;
  * the homepage is a splash/redirect page that carries no navigation at all.

IIT Roorkee is the case that shows the size of the loss. The sweep recorded it
`empty`. It publishes /Academics/Institute Fees.html, /Academics/
ScholarshipsAwards.html, /fao/Section/Fee.html and per-department placement
stats — 1,509 URLs, all listed in a sitemap the old path never asked for.

WHAT `empty` ACTUALLY MEANT (the measurement corrected the premise)
On 40 colleges recorded `empty`, re-running pick_pages against a fresh fetch
found topic pages for 16 of them. So `empty` is two different failures wearing
one label: 24/40 where no page was found (what this module fixes) and 16/40
where pages WERE found and extraction returned nothing. This module is therefore
a SUPPLEMENT to pick_pages, not a replacement — it wins where the homepage links
are silent and loses where they work, so the caller should union the two rather
than swap them. See `measure()` for the numbers.

THE ORDER OF OPERATIONS IS THE WHOLE DESIGN
1. SITEMAP FIRST. One request can replace dozens of blind probes, and it is
   authoritative: a URL in a sitemap exists, so it needs no verification fetch.
2. PATH PROBING SECOND, and only for the topics the sitemap did not answer.
   The brief expected probing to carry the corpus, because only 3 of 12 sites
   surveyed served a usable sitemap. It does not. Measured over 40 colleges,
   probing hit 4 times in 227 requests (1.8%) while the sitemap branch returned
   51 pages for ~76 requests. Probing earns its place — one of the eight
   recovered colleges came from it alone — but it is the minority partner, and
   MAX_PROBES is small on purpose.

WHERE THE PROBE LIST COMES FROM (not from my priors)
`college_web_facts.source_url` already holds 1,465 pages the running sweep
successfully extracted a fact from. Those are real, working Indian-college topic
paths, and the ranked stems in PROBE_STEMS below are their observed frequencies:
/scholarship (37) beats /scholarships (21); /fee-structure/ (17) beats /fees (5);
for hostel, /facilities (21) actually outranks /hostel (11) because that is where
these sites put hostel charges. Extension shape came from the same place — of
10,209 catalogue website URLs 10,017 are extensionless, but .php (76), .html (66)
and .aspx (39) are common enough that /hostel.php alone yielded 19 pages, so the
extension is chosen from a fingerprint of the server rather than tried blindly.

WHY IT IS BOUNDED AT ~12 REQUESTS
An unbounded path fan-out is how a crawler becomes an attack. The cap is per
college and the per-domain delay from `web_enrich.Politeness` is still enforced
on every single request, robots.txt included — sitemap URLs are checked against
robots too, because a sitemap listing a path is not the same as robots allowing
it (chitkara.edu.in lists /pdf/nirf/ in its sitemap and disallows it in robots).

SOFT-404 DETECTION IS A CORRECTNESS FEATURE, NOT A TIDY-UP
Path probing only works if "200 OK" means the page exists, and on college sites
it frequently does not. Measured shapes:
  * catch-all SPA: 200 + byte-identical body for every path (iitjammu.ac.in);
  * cross-domain bounce: /fees 200 but the final URL is another site's homepage
    (davpgcollege.in -> davpgcollegeddn.ac.in);
  * honest 404 (iitr.ac.in, homescience10.ac.in) — the easy case.
Without the control probe below, a catch-all host hands seven copies of its
homepage shell to seven different extractors, and a fee parser reading a
homepage is exactly the "confidently wrong number shown to a nervous teenager"
this codebase refuses to ship. Same reason off-site URLs are dropped hard:
gcechd.ac.in serves a 6 MB sitemap of 44,522 hijacked casino URLs.

API
    discover_pages(client, politeness, origin, robots) -> [(kind, url), ...]

Returns the same shape as `web_enrich.pick_pages()` so it drops straight in.
Pass `html_sink={}` to also receive `{url: html}` for pages this module already
fetched during probing, which saves the caller re-fetching them.

Intended wiring in enrich_one (union, cheapest first — pick_pages costs nothing
extra because the homepage is already in hand):

    targets = pick_pages(site, links)
    if len(targets) < MAX_PAGES_PER_COLLEGE:
        have = {k for k, _ in targets}
        rp = ...                       # the robots parser for this host
        for kind, url in await discover_pages(client, pol, site, rp,
                                              html_sink=sink):
            if kind not in have:
                targets.append((kind, url))

Measure it (nothing here writes to the database):
    python -m rag_core.sources.pagefind --url https://www.iitr.ac.in/
    python -m rag_core.sources.pagefind --measure 40 --sqlite data/mivi.db \
        --save pages.jsonl
    python -m rag_core.sources.pagefind --replay pages.jsonl --llm
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import os
import re
import sys
import time
from urllib.parse import urljoin, urlsplit

import httpx

from .web_enrich import (HEADERS, MAX_PAGES_PER_COLLEGE, PAGE_TARGETS,
                         Politeness, normalise_site, parse_html)

# The budget dial. 12 requests x 1.5 s of per-domain pacing is ~18 s of wall
# clock for one college, which is only acceptable because the sitemap branch
# usually finishes in 1-3. Lowering this mostly costs probe breadth.
MAX_REQUESTS = int(os.getenv("MIVI_DISCOVER_REQUESTS", "12"))

# Sitemap phase sub-budget. Three candidate locations covers the formats
# actually observed; a fourth has never paid for itself.
MAX_SITEMAP_TRIES = 3
MAX_SITEMAP_CHILDREN = 2

# A sitemap is a hint, not a payload. gcechd.ac.in's is 6 MB of injected spam,
# so the read is capped and aborted mid-stream rather than buffered whole.
MAX_SITEMAP_BYTES = 3_000_000
MAX_SITEMAP_URLS = 8_000
MAX_PROBE_BYTES = 400_000       # a probe body only has to be classifiable

# Probe sub-budget. Measured at 1.8% yield (4 hits in 227 probe requests) where
# the sitemap path returned 51 pages for ~76 requests, so this is deliberately
# the smaller half of the budget: a college with no sitemap costs at most
# 3 + 1 + 8 requests instead of spending the whole allowance proving absence.
MAX_PROBES = 8
PROBE_CHUNK = 2                 # forms tried per topic before moving on

# Order to SPEND PROBES in, which is not PAGE_TARGETS order. What comes back is
# still ordered and capped by PAGE_TARGETS — this only decides where a scarce
# request budget goes, and the evidence says PAGE_TARGETS is the wrong guide for
# that. Across the 1,465 pages that have already yielded a fact, cutoff appears
# 3 times; it sits 3rd in PAGE_TARGETS, so probing in that order spends two of
# eight requests on the one topic colleges essentially never publish at a
# guessable path (cutoffs show up as PDFs linked from a notice board, which no
# path probe can reach). Everything else keeps its relative priority.
PROBE_PRIORITY = ("scholarship", "fees", "admission", "placement", "hostel",
                  "events", "cutoff")

# Above this many first-party sitemap URLs — AND only if the sitemap already
# yielded a topic page, proving it indexes content and not just blog posts —
# treat it as a full index of the site and stop probing for what it omitted.
SITEMAP_LOOKS_COMPLETE = 25

# Extensions that are not HTML. Dropping them is not fussiness: web_enrich.fetch
# requires an HTML content-type, so returning a PDF spends a request to fetch
# something the extractors cannot read. Costs us the `/notice/*.pdf` cutoff
# sheets, which that pipeline could not use anyway.
_BAD_EXT = re.compile(
    r"\.(pdf|docx?|xlsx?|pptx?|zip|rar|csv|jpe?g|png|gif|svg|webp|mp4|mp3|"
    r"ico|css|js|json|xml|rss)$", re.I)

_PAGE_EXT = re.compile(r"\.(php|html?|aspx?|jsp|shtml|cgi)$", re.I)

# A page for ONE degree programme, not for the college. chitkara.edu.in has
# /ph.d/program-fees/ and /mba/financial-markets-practice/fee-structure/;
# answering "what are the fees" with the PhD page is a wrong answer dressed as a
# right one. Demoted rather than rejected, because on a single-programme
# institute it is the only fee page there is.
_PROGRAMME = re.compile(
    r"(?:^|[/_\-.])(b[\-_. ]?tech|m[\-_. ]?tech|b[\-_. ]?sc|m[\-_. ]?sc|"
    r"b[\-_. ]?com|m[\-_. ]?com|bba|mba|bca|mca|ph[\-_. ]?d|ll[\-_. ]?[bm]|"
    r"b[\-_. ]?ed|m[\-_. ]?ed|b[\-_. ]?pharm|m[\-_. ]?pharm|pharm[\-_. ]?d|"
    r"mbbs|bds|bams|bhms|b[\-_. ]?arch|bpt|mpt|bhm|bvsc)(?:[/_\-.]|$)", re.I)

# Kinds where a per-programme page is actively misleading vs merely narrow.
_PROGRAMME_PENALTY = {"cutoff": 35.0, "hostel": 35.0, "events": 35.0,
                      "placement": 25.0, "scholarship": 25.0,
                      "fees": 10.0, "admission": 10.0}

# A path segment that puts the page in a DIFFERENT scope from the college, so a
# keyword hit inside it is about something else. amrita.edu's best-scoring
# scholarship URL was /events/icce/scholarship/ — a conference's travel
# scholarship, scored highly because `scholarship` is the exact last segment.
# Only applied to kinds other than `events`, for which these words are the point.
_CROSS_SCOPE = frozenset({"events", "event", "conference", "workshop",
                          "seminar", "webinar", "blog", "fest", "news"})

# Floor for accepting a candidate from a sitemap. Without a floor the best of a
# bad field wins by default, and on a large university's sitemap the bad field is
# large: chitkara's only cutoff-ish URL was a course called "M.Sc Reproductive
# Health & Fertility COUNSELLING" (PAGE_TARGETS matches `counselling`), and
# amrita's best hostel URL was /unsdg/sdg3/shared-sport-facilities/ (matches
# `facilities`). Returning nothing for a topic is a correct answer; a missing
# fact is recoverable and a wrong one is not.
#
# Calibrated against every page in the 40-college measurement that actually
# produced a fact: the lowest-scoring true positive is a notice-board post at 88
# (/2020/03/13/notice-classes-off-till-31-march-2020/) and the highest-scoring
# false positive is 79, so the boundary is real rather than fitted to one case.
MIN_SCORE = 85.0

# Words that tokenise as a legitimate keyword hit but never carry the fact.
# Deliberately short — the tokeniser below already kills the classic false
# positives, so this only holds cases that survive it.
# `payment`/`pay` are here for a specific failure: ritroorkee.com's best-scoring
# fee candidate was /online-fee-payment/, a payment gateway that states no
# amounts. A -45 demotion does not blacklist it — if the college publishes
# nothing else about fees it is still returned — it just loses to a real fee
# page whenever one exists.
_NOISE = frozenset({
    "faculty", "staff", "gallery", "photo", "photos", "video", "videos",
    "alumni", "login", "signin", "privacy", "disclaimer", "sitemap", "feed",
    "rss", "tag", "author", "tender", "vacancy", "recruitment-notice",
    "grievance", "complaint", "wp", "payment", "pay", "cart", "checkout",
})

# Ranked probe stems, ordered by how often each ALREADY produced a fact in
# college_web_facts (counts in the module docstring). Order is the priority: the
# budget is spent top-down, so a wrong order costs coverage directly.
PROBE_STEMS: dict[str, tuple[str, ...]] = {
    "scholarship": ("scholarship", "scholarships", "scholarship-schemes",
                    "awards-scholarships", "financial-aid"),
    "fees": ("fee-structure", "fees", "fee", "fees-structure", "feestructure",
             "fee_structure", "course-fee", "tuition-fees"),
    "cutoff": ("cutoff", "cut-off", "merit-list", "counselling"),
    "admission": ("admission", "admissions", "admission-process",
                  "admissionprocedure", "how-to-apply"),
    "placement": ("placement", "placements", "placement-cell",
                  "training-placement", "placement-overview"),
    "hostel": ("hostel", "facilities", "hostels", "campus-life",
               "hostel-facilities", "accommodation"),
    "events": ("notice", "news", "notices", "events", "notice-board",
               "news-events", "announcements"),
}

# Sitemap locations to try, best-trust first. robots.txt `Sitemap:` directives
# outrank all of these and are prepended at runtime.
# Order is measured, not conventional, because only MAX_SITEMAP_TRIES of these
# get a request. /sitemap.txt is third-guess in most crawlers and second here:
# iitr.ac.in publishes 1,509 URLs there and nothing in XML, and with the usual
# ordering (/sitemap.xml, /sitemap_index.xml, /wp-sitemap.xml) the budget ran
# out one request before the file that existed. The two guesses displaced are
# the ones a CMS also announces in robots.txt, so they are rarely the only copy.
SITEMAP_CANDIDATES = (
    "/sitemap.xml",
    "/sitemap.txt",
    "/sitemap_index.xml",
    "/wp-sitemap.xml",      # WordPress core; gpdehradun.org.in served only this
    "/sitemap-index.xml",
    "/sitemap.xml.gz",
)

# Two-label public suffixes that matter for Indian institutions. Without these
# `iitr.ac.in` would reduce to `ac.in` and every .ac.in site would look like the
# same site — which is how a spam sitemap gets treated as first-party.
_TWO_LABEL_SUFFIXES = frozenset({
    "ac.in", "edu.in", "co.in", "org.in", "net.in", "gov.in", "nic.in",
    "res.in", "mil.in", "gen.in", "firm.in", "ind.in", "co.uk", "ac.uk",
    "org.uk", "com.au", "edu.au", "co.nz", "com.np", "edu.np", "edu.pk",
    "edu.bd", "ac.lk", "com.sg", "edu.sg",
})


# ---------------------------------------------------------------------------
# Tokenised matching
#
# `pick_pages` matches keywords as raw substrings, and that is measurably wrong:
# "fee" is a substring of "feedback", so the live sweep has already filed
# /feedback.html and /aicte-feedback/ as FEES pages, and "news" is a substring
# of "newsletter". Matching on word boundaries instead removes that whole class
# of error without needing a blacklist per topic.
# ---------------------------------------------------------------------------

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Za-z])(?=[0-9])")


def _tokens(s: str) -> list[str]:
    """Path words. Splits on separators AND on camel-case humps, because legacy
    college sites concatenate: IIT Roorkee's scholarship page is
    /Academics/ScholarshipsAwards.html, which has no separator to split on and
    is invisible to a separator-only tokeniser."""
    return [t for t in re.split(r"[^a-z0-9]+", _CAMEL.sub(" ", s).lower()) if t]


def _seg_tokens(seg: str) -> list[str]:
    return _tokens(_PAGE_EXT.sub("", seg))


def _tok_eq(a: str, b: str) -> bool:
    """Token equality that tolerates English plurals.

    PAGE_TARGETS spells some keywords both ways ("scholarship", "scholarships")
    and others only once, so without this /cutoffs, /facility and /fees-page
    silently miss on keyword lists that happen to hold the other number.
    """
    if a == b:
        return True
    if len(a) > 2 and len(b) > 2:
        if a == b + "s" or b == a + "s":
            return True
        if a.endswith("ies") and b.endswith("y") and a[:-3] == b[:-1]:
            return True
        if b.endswith("ies") and a.endswith("y") and b[:-3] == a[:-1]:
            return True
    return False


def _run_index(tokens: list[str], key: list[str]) -> int:
    """Where `key`'s words appear as a contiguous run in `tokens`, else -1."""
    n = len(key)
    if not n or n > len(tokens):
        return -1
    for i in range(len(tokens) - n + 1):
        if all(_tok_eq(tokens[i + j], key[j]) for j in range(n)):
            return i
    return -1


def clean_url(url: str) -> str:
    """Percent-encode the characters real sitemaps leave raw.

    IIT Roorkee's sitemap.txt lists paths like /Academics/Institute Fees.html
    with literal spaces. Left alone those either fail the request or get encoded
    inconsistently between the robots check and the fetch, so a page that exists
    looks blocked. Only the space/quote class is touched — re-encoding a whole
    URL would double-escape the %-sequences that are already there.
    """
    scheme, _, rest = url.partition("://")
    if not rest:
        return url.replace(" ", "%20")
    host, slash, path = rest.partition("/")
    for ch, sub in ((" ", "%20"), ('"', "%22"), ("<", "%3C"), (">", "%3E"),
                    ("|", "%7C"), ("\\", "%5C"), ("^", "%5E"), ("`", "%60")):
        path = path.replace(ch, sub)
    return f"{scheme}://{host}{slash}{path}"


def registrable(host: str) -> str:
    """The site-identity part of a hostname, for first-party checks."""
    h = (host or "").lower().split(":")[0].rstrip(".")
    labels = h.split(".")
    if len(labels) >= 3 and ".".join(labels[-2:]) in _TWO_LABEL_SUFFIXES:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:]) if len(labels) >= 2 else h


def same_site(a: str, b: str) -> bool:
    """First-party test that tolerates the host drift real sitemaps show.

    Stricter than "any .ac.in" and looser than exact netloc equality, because
    both extremes lose data: www.iitr.ac.in's sitemap lists everything under
    bare iitr.ac.in, and IIT Roorkee's management school lives on
    doms.iitr.ac.in. Same registrable domain is the line that keeps those and
    still rejects the casino domain injected into gcechd.ac.in's sitemap.
    """
    return bool(a) and bool(b) and registrable(a) == registrable(b)


def score_url(kind: str, keys: tuple[str, ...], url: str) -> float:
    """How well `url` looks like `kind`'s topic page. Negative = reject.

    Shape beats mere presence of the word. A top-level /fee-structure/ is the
    fee page; /mba/financial-markets-practice/fee-structure/ is one programme's,
    and answering "what are the fees" with one programme's page is worse than
    answering with the college-wide one. Hence the depth penalty.
    """
    parts = urlsplit(url)
    path = parts.path
    if _BAD_EXT.search(path):
        return -1.0
    segs = [s for s in path.split("/") if s]
    if not segs:
        return -1.0                      # the homepage is not a topic page
    path_toks = _tokens(path)
    last_toks = _seg_tokens(segs[-1])
    best = -1.0
    for rank, key in enumerate(keys):
        kt = _tokens(key)
        if _run_index(path_toks, kt) < 0:
            continue
        s = 100.0 - rank * 4.0           # earlier keyword = more on-topic
        if _run_index(last_toks, kt) == 0 and len(last_toks) == len(kt):
            s += 40.0                    # /fee-structure/ — the page itself
        elif _run_index(last_toks, kt) == 0:
            s += 25.0                    # /fee-structure-2025-26
        elif kt and any(_tok_eq(t, kt[0]) for t in last_toks):
            s += 12.0
        s -= 7.0 * (len(segs) - 1)       # prefer college-wide over per-programme
        s -= min(20.0, len(path) / 12.0)
        if parts.query:
            s -= 8.0
        best = max(best, s)
    if best < 0:
        return -1.0
    if any(t in _NOISE for t in path_toks):
        best -= 45.0
    if kind != "events" and any(t in _CROSS_SCOPE for t in path_toks[:-1]):
        best -= 45.0
    if _PROGRAMME.search(path):
        best -= _PROGRAMME_PENALTY.get(kind, 20.0)
    return best


def rank_candidates(origin: str, urls, limit: int = MAX_PAGES_PER_COLLEGE
                    ) -> list[tuple[str, str]]:
    """Best first-party URL per topic, in PAGE_TARGETS priority order.

    One URL is assigned to at most one kind (its best-scoring one), so a page
    called /admission-and-fee-structure cannot be submitted twice and burn two
    of the caller's page slots on the same fetch.
    """
    host = urlsplit(origin).netloc
    # When the catalogue URL is a deep path it is because that college is one
    # institute inside a larger university — chitkara.edu.in/hospitality,
    # chitkara.edu.in/cbs, ibsindia.org/dehradun. The sitemap is shared by the
    # whole university and its shallow URLs outscore everything on the depth
    # penalty, so without this bonus all four Chitkara colleges in the sample
    # resolve to the same university-wide /admissions/ page and MIVI would quote
    # one institute's fees for another's. A page under the college's own prefix
    # beats a university-wide one for that college.
    prefix = urlsplit(origin).path.rstrip("/").lower()
    if _PAGE_EXT.search(prefix):
        prefix = prefix.rsplit("/", 1)[0]
    best: dict[str, tuple[float, str]] = {}
    claim: dict[str, tuple[float, str]] = {}   # url -> (score, kind)
    for url in urls:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not same_site(parts.netloc, host):
            continue
        own = bool(prefix) and parts.path.lower().startswith(prefix + "/")
        for kind, keys in PAGE_TARGETS:
            s = score_url(kind, keys, url)
            if s < 0:
                continue
            if own:
                s += 45.0                # this college's own page, not its parent's
            if s < MIN_SCORE:
                continue
            prev = claim.get(url)
            if prev is None or s > prev[0]:
                claim[url] = (s, kind)
    for url, (s, kind) in claim.items():
        cur = best.get(kind)
        if cur is None or s > cur[0]:
            best[kind] = (s, url)
    out = [(k, clean_url(best[k][1])) for k, _ in PAGE_TARGETS if k in best]
    return out[:limit]


# ---------------------------------------------------------------------------
# Fetching, with the caller's politeness policy
# ---------------------------------------------------------------------------

class Resp:
    __slots__ = ("status", "url", "headers", "body", "truncated")

    def __init__(self, status, url, headers, body, truncated):
        self.status = status
        self.url = url
        self.headers = headers
        self.body = body
        self.truncated = truncated

    def text(self) -> str:
        raw = self.body
        if raw[:2] == b"\x1f\x8b":       # a .gz sitemap served as a file
            try:
                raw = gzip.decompress(raw)
            except Exception:            # noqa: BLE001 - truncated gzip: use as-is
                pass
        m = re.search(r"charset=([\w-]+)",
                      self.headers.get("content-type", ""), re.I)
        enc = (m.group(1) if m else "utf-8")
        try:
            return raw.decode(enc, "replace")
        except LookupError:
            return raw.decode("utf-8", "replace")

    @property
    def ctype(self) -> str:
        return self.headers.get("content-type", "").lower()


async def _get(client: httpx.AsyncClient, pol, url: str, cap: int) -> Resp | None:
    """One capped GET under the caller's robots + per-domain pacing.

    Streamed with an early break rather than buffered: a 6 MB spam sitemap
    should cost us the first 3 MB and a closed connection, not the whole file.
    """
    try:
        if not await pol.allowed(client, url):
            return None
    except Exception:                    # noqa: BLE001 - robots unavailable != deny
        pass
    await pol.wait(url)
    try:
        async with client.stream("GET", url) as r:
            buf = bytearray()
            truncated = False
            async for chunk in r.aiter_bytes():
                buf += chunk
                if len(buf) >= cap:
                    truncated = True
                    break
            return Resp(r.status_code, str(r.url), dict(r.headers), bytes(buf),
                        truncated)
    except Exception:                    # noqa: BLE001 - DNS/TLS/timeout: no page
        return None


# ---------------------------------------------------------------------------
# Soft-404 fingerprinting
# ---------------------------------------------------------------------------

_WS = re.compile(r"\s+")


def _body_hash(html: str) -> str:
    text, _ = parse_html(html)
    return hashlib.sha1(_WS.sub(" ", text).strip().encode("utf-8",
                                                          "replace")).hexdigest()


def _title(html: str) -> str:
    m = re.search(r"<title[^>]*>(.*?)</title>", html[:20_000], re.I | re.S)
    return _WS.sub(" ", m.group(1)).strip().lower() if m else ""


_ERROR_TITLE = re.compile(r"\b(404|403|not found|page not ?found|error|"
                          r"forbidden|access denied|under construction|"
                          r"coming soon)\b", re.I)


class HostFingerprint:
    """What this host does with a URL that cannot exist.

    One request, taken lazily and only if we are about to probe at all, then
    reused for every probe on the host — and across colleges, since 4 of the 40
    colleges in the measurement below share chitkara.edu.in.
    """

    # Statuses that mean "stop asking", not "this page is missing". Observed
    # live: gcechd.ac.in served /fee-structure.php happily, then began answering
    # 409 after a few dozen probes in one session. A 404 is information; these
    # are a host defending itself, and continuing to probe past one is how a
    # polite crawler turns into an impolite one despite honouring robots.txt and
    # its per-domain delay.
    REFUSAL = frozenset({401, 402, 403, 405, 406, 409, 421, 423, 425, 429,
                         444, 503, 509})

    __slots__ = ("host", "status", "length", "hash", "title", "stack", "probed",
                 "moved_to", "refused")

    def __init__(self, host: str):
        self.host = host
        self.status: int | None = None
        self.length = -1
        self.hash = ""
        self.title = ""
        self.stack = ""          # "php" | "aspx" | "jsp" | "wp" | ""
        self.probed = False
        self.moved_to: str | None = None
        self.refused = False

    async def prime(self, client, pol, origin: str) -> bool:
        """Fingerprint what this host does with a URL that cannot exist.

        Costs exactly one request and answers three questions at once: does a
        200 here mean anything, which URL extension does this stack use, and has
        the whole site moved to another domain.
        """
        self.probed = True
        control = urljoin(origin, f"/mivi-probe-{os.urandom(4).hex()}")
        r = await _get(client, pol, control, MAX_PROBE_BYTES)
        if r is None:
            return True                  # cannot fingerprint; fall back to rules
        self.status = r.status
        html = r.text()
        self.length = len(r.body)
        self.hash = _body_hash(html)
        self.title = _title(html)
        self.stack = self._detect_stack(r, html, origin)
        # A site-wide relocation, spotted for free. davpgcollege.in answers every
        # path with a redirect to davpgcollegeddn.ac.in's homepage. This is a
        # whole class of `empty` results rather than an oddity: the catalogue URL
        # still resolves, so the homepage fetch succeeds, but pick_pages then
        # filters every link on the page for the OLD netloc and discards all of
        # them. Probing the old host is pointless; the new one is where the
        # college lives.
        final = urlsplit(r.url)
        if (final.netloc and not same_site(final.netloc, urlsplit(origin).netloc)
                and not final.path.strip("/")):
            self.moved_to = f"{final.scheme}://{final.netloc}/"
        return True

    @staticmethod
    def _detect_stack(r: Resp, html: str, origin: str) -> str:
        """Pick the URL extension to probe from evidence, not from trying all.

        Trying /fees, /fees.php, /fees.html and /fees.aspx is four requests for
        one topic and blows the whole budget on it. The control response already
        names the stack for free, so this is the single move that makes a
        12-request budget cover seven topics.
        """
        h = " ".join(f"{k}: {v}" for k, v in r.headers.items()).lower()
        if "phpsessid" in h or "php" in h:
            return "php"
        if "asp.net" in h or "aspxauth" in h or "x-aspnet-version" in h:
            return "aspx"
        if "jsessionid" in h:
            return "jsp"
        low = html[:60_000].lower()
        if "wp-content" in low or "wp-includes" in low:
            return "wp"                  # WordPress: permalinks are extensionless
        ext = _PAGE_EXT.search(urlsplit(origin).path)
        if ext:
            e = ext.group(1).lower()
            return {"php": "php", "aspx": "aspx", "asp": "aspx",
                    "jsp": "jsp"}.get(e, "")
        return ""

    def variants(self) -> tuple[str, ...]:
        """Extension suffixes to try, in order. Extensionless first always: it
        is 10,017 of 10,209 catalogue URLs, and follow_redirects turns /fees
        into /fees/ for free where that is the canonical form."""
        if self.stack == "php":
            return ("", ".php", ".html")
        if self.stack == "aspx":
            return ("", ".aspx", ".html")
        if self.stack == "jsp":
            return ("", ".jsp", ".html")
        if self.stack == "wp":
            return ("", "/")
        return ("", ".php", ".html")

    def is_soft_404(self, requested: str, r: Resp) -> bool:
        """True when a 200 does not mean the page exists."""
        if r.status != 200:
            return True
        if "html" not in r.ctype and r.ctype:
            return True
        final = urlsplit(r.url)
        if not same_site(final.netloc, urlsplit(requested).netloc):
            return True                  # bounced off-site entirely
        if urlsplit(requested).path.strip("/") and not final.path.strip("/"):
            return True                  # deep path -> landed on a homepage
        html = r.text()
        title = _title(html)
        if _ERROR_TITLE.search(title):
            return True
        text, _ = parse_html(html)
        if len(text.strip()) < 300:
            return True                  # nothing an extractor could read
        if self.status == 200:
            if _body_hash(html) == self.hash:
                return True              # byte-for-byte the catch-all page
            if (self.length > 0 and title and title == self.title
                    and abs(len(r.body) - self.length) <= max(
                        256, self.length * 0.02)):
                return True              # same shell, trivially different bytes
        return False


# Per-process, per-host caches. Both are read-mostly and bounded by the number
# of distinct hosts in a sweep; the sitemap cache is what makes the four
# chitkara.edu.in colleges cost one sitemap fetch instead of four.
_FINGERPRINTS: dict[str, HostFingerprint] = {}
_SITEMAP_CACHE: dict[str, list[str]] = {}


# ---------------------------------------------------------------------------
# Sitemaps
# ---------------------------------------------------------------------------

_LOC = re.compile(r"<loc>\s*([^<\s][^<]*?)\s*</loc>", re.I)

# Which child of a sitemap index to spend a request on. Static pages hold the
# fee and hostel tables; news/tag/user children hold thousands of articles that
# score badly and crowd out the pages we want.
_CHILD_GOOD = ("page-sitemap", "posts-page", "pages", "page")
_CHILD_BAD = ("news", "tag", "category", "author", "user", "media",
              "attachment", "product", "elementor", "mega_menu", "image",
              "video")


def _child_rank(url: str) -> int:
    low = url.lower()
    if any(b in low for b in _CHILD_BAD):
        return 2
    if any(g in low for g in _CHILD_GOOD):
        return 0
    return 1


def _parse_sitemap(body_text: str) -> tuple[list[str], bool]:
    """(urls, is_index). Handles urlset, sitemapindex and plain-text sitemaps."""
    head = body_text[:4000].lower()
    if "<sitemapindex" in head:
        return _LOC.findall(body_text)[:MAX_SITEMAP_URLS], True
    if "<urlset" in head or "<loc>" in head:
        return _LOC.findall(body_text)[:MAX_SITEMAP_URLS], False
    # sitemap.txt: one absolute URL per line. Spaces are legal in these paths
    # (IIT Roorkee's are full of them), so lines are taken whole, not split.
    lines = [ln.strip() for ln in body_text.splitlines()]
    urls = [ln for ln in lines if ln[:4].lower() == "http" and " " not in ln[:8]]
    if len(urls) >= 3 and len(urls) >= len([l for l in lines if l]) * 0.6:
        return urls[:MAX_SITEMAP_URLS], False
    return [], False


def _declared_sitemaps(politeness, robots, origin: str) -> list[str]:
    """robots.txt `Sitemap:` lines — free, because the caller already fetched
    robots.txt to decide it was allowed to fetch the homepage at all."""
    rp = robots
    if rp is None:
        host = urlsplit(origin).netloc
        rp = (getattr(politeness, "_robots", {}) or {}).get(host)
    try:
        maps = rp.site_maps() if rp is not None else None
    except Exception:                    # noqa: BLE001
        maps = None
    return [m for m in (maps or []) if isinstance(m, str) and m[:4].lower() == "http"]


async def sitemap_urls(client, pol, origin: str, robots=None,
                       budget: int = MAX_SITEMAP_TRIES + MAX_SITEMAP_CHILDREN
                       ) -> tuple[list[str], int]:
    """Every first-party URL this host advertises. Returns (urls, requests_spent).

    Cached per registrable domain: sitemaps are site-wide, so colleges sharing a
    university's domain share the answer.
    """
    key = registrable(urlsplit(origin).netloc)
    if key in _SITEMAP_CACHE:
        return _SITEMAP_CACHE[key], 0

    spent = 0
    tried: set[str] = set()
    candidates = _declared_sitemaps(pol, robots, origin) + [
        urljoin(origin, c) for c in SITEMAP_CANDIDATES]

    urls: list[str] = []
    for cand in candidates:
        if spent >= min(budget, MAX_SITEMAP_TRIES) or urls:
            break
        if cand in tried:
            continue
        tried.add(cand)
        r = await _get(client, pol, cand, MAX_SITEMAP_BYTES)
        spent += 1
        if r is None or r.status != 200:
            continue
        found, is_index = _parse_sitemap(r.text())
        if not found:
            continue
        if is_index:
            children = sorted(found, key=_child_rank)
            for child in children[:MAX_SITEMAP_CHILDREN]:
                if spent >= budget:
                    break
                if not same_site(urlsplit(child).netloc, urlsplit(origin).netloc):
                    continue
                rc = await _get(client, pol, child, MAX_SITEMAP_BYTES)
                spent += 1
                if rc is None or rc.status != 200:
                    continue
                kids, kid_is_index = _parse_sitemap(rc.text())
                if not kid_is_index:
                    urls += kids
        else:
            urls += found

    host = urlsplit(origin).netloc
    urls = [u for u in urls if same_site(urlsplit(u).netloc, host)]
    _SITEMAP_CACHE[key] = urls
    return urls, spent


# ---------------------------------------------------------------------------
# Path probing
# ---------------------------------------------------------------------------

def probe_bases(origin: str) -> list[str]:
    """Where to hang a probe path off.

    The catalogue stores deep URLs for institutions inside a larger university
    (chitkara.edu.in/hospitality, ciiih.com/chandigarh, amrita.edu/campus/
    haridwar). For those the sub-path is the college, so it is probed FIRST —
    /hospitality/placement is that college's placement page and /placement is
    the parent university's.
    """
    p = urlsplit(origin)
    root = f"{p.scheme}://{p.netloc}/"
    path = p.path or "/"
    if _PAGE_EXT.search(path):
        path = path.rsplit("/", 1)[0] + "/"
    if not path.endswith("/"):
        path += "/"
    deep = f"{p.scheme}://{p.netloc}{path}"
    return [deep, root] if path.strip("/") else [root]


def probe_candidates(kind: str, bases: list[str],
                     variants: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for stem in PROBE_STEMS.get(kind, ()):
        for ext in variants:
            for base in bases:
                u = urljoin(base, stem + ext)
                if u not in seen:
                    seen.add(u)
                    out.append(u)
    return out


# ---------------------------------------------------------------------------
# The entry point
# ---------------------------------------------------------------------------

async def discover_pages(client: httpx.AsyncClient, politeness, origin: str,
                         robots=None, *, budget: int = MAX_REQUESTS,
                         limit: int = MAX_PAGES_PER_COLLEGE,
                         html_sink: dict | None = None,
                         stats: dict | None = None,
                         _hops: int = 0) -> list[tuple[str, str]]:
    """Topic pages for one college, found without reading its homepage links.

    Drop-in for `web_enrich.pick_pages()`: same `[(kind, url)]` return, same
    `kind` vocabulary, same MAX_PAGES_PER_COLLEGE cap.

    `html_sink` (optional): receives {url: html} for pages fetched here, so the
    caller can skip re-fetching a page probing already proved exists. Only
    complete bodies are handed over — a body cut off at MAX_PROBE_BYTES is left
    out so the caller refetches it whole rather than feeding an extractor a
    truncated fee table.
    """
    origin = normalise_site(origin) or origin
    st = stats if stats is not None else {}
    st.setdefault("requests", 0)
    st.setdefault("via", {})
    st.setdefault("probes", 0)          # probe requests spent
    st.setdefault("probe_hits", 0)      # probe requests that found a real page

    # --- 1. sitemap: one request that can answer every topic at once ---------
    urls, spent = await sitemap_urls(client, politeness, origin, robots,
                                     budget=min(budget,
                                                MAX_SITEMAP_TRIES
                                                + MAX_SITEMAP_CHILDREN))
    st["requests"] += spent
    st["sitemap_urls"] = len(urls)
    found: dict[str, str] = {}
    for kind, url in rank_candidates(origin, urls, limit=len(PAGE_TARGETS)):
        found[kind] = url
        st["via"][kind] = "sitemap"

    def ordered() -> list[tuple[str, str]]:
        return [(k, found[k]) for k, _ in PAGE_TARGETS if k in found][:limit]

    if len(found) >= limit:
        return ordered()

    # --- 2. probe only the topics the sitemap did not answer -----------------
    missing = [k for k, _ in PAGE_TARGETS if k not in found]
    left = budget - st["requests"]
    if not missing or left <= 1:
        return ordered()

    host = urlsplit(origin).netloc
    fp = _FINGERPRINTS.get(host)
    if fp is None:
        fp = _FINGERPRINTS[host] = HostFingerprint(host)
    if fp.refused:
        # This host already refused a probe earlier in the sweep. Several
        # colleges share one university domain, so without this the sweep would
        # rediscover the same refusal once per college.
        st["refused"] = "cached"
        return ordered()
    if not fp.probed:
        await fp.prime(client, politeness, origin)
        st["requests"] += 1
        left -= 1
    st["stack"] = fp.stack

    # The site moved. Spend what is left on the domain it moved TO — one hop
    # only, so a redirect loop between two domains cannot recurse.
    if fp.moved_to and _hops == 0 and left >= 3:
        st["moved_to"] = fp.moved_to
        sub: dict = {"requests": 0, "via": {}}
        more = await discover_pages(client, politeness, fp.moved_to, None,
                                    budget=left, limit=limit,
                                    html_sink=html_sink, stats=sub, _hops=1)
        st["requests"] += sub.get("requests", 0)
        for kind, url in more:
            found.setdefault(kind, url)
            st["via"].setdefault(kind, (sub.get("via") or {}).get(kind, "moved"))
        return ordered()

    # A sitemap that listed the whole site has already told us which topics
    # exist. Probing for the ones it did not list mostly buys 404s, so the
    # remaining budget is cut hard rather than spent proving a page is absent —
    # measured on gpdehradun.org.in, whose 43-URL sitemap IS the site and where
    # probing the five missing topics cost 9 requests for nothing.
    if found and st.get("sitemap_urls", 0) >= SITEMAP_LOOKS_COMPLETE:
        left = min(left, 3)

    bases = probe_bases(origin)
    left = min(left, MAX_PROBES)

    # Probe order is chunked per topic, not breadth-first across all seven.
    #
    # Measured on 40 `empty` colleges: 4 hits in 227 probe requests (1.8%),
    # against 62 pages from ~76 sitemap requests. At that yield the ordering has
    # to concentrate rather than spread. Breadth-first spends the whole budget on
    # the extensionless form of every topic and never reaches /fee-structure.php,
    # which is the form that actually existed on gcechd.ac.in — and .php is the
    # most common non-empty extension in the corpus, with /hostel.php alone
    # accounting for 19 pages that have already yielded facts. So each topic gets
    # its PROBE_CHUNK most likely URLs first: the primary stem at both bases when
    # the origin is a deep path, otherwise that stem's two likeliest extensions.
    #
    # Only the first `limit + 1` topics get an allocation, because
    # MAX_PAGES_PER_COLLEGE caps what the caller will keep anyway. That is a
    # floor, not a ceiling: a topic found early releases its unspent candidates
    # without costing a request, so the loop reaches further down this list the
    # better it is doing. Measured on ibsindia.org, where /admission hit on the
    # 7th request and /placement — nominally 9th, past the cap — was then
    # reached and hit as the 8th.
    order: list[tuple[str, str]] = []
    for kind in [k for k in PROBE_PRIORITY if k in missing][:limit + 1]:
        for cand in probe_candidates(kind, bases, fp.variants())[:PROBE_CHUNK]:
            order.append((kind, cand))

    tried: set[str] = set()
    for kind, url in order:
        if left <= 0 or len(found) >= limit:
            break
        if kind in found or url in tried:
            continue
        tried.add(url)
        r = await _get(client, politeness, url, MAX_PROBE_BYTES)
        st["requests"] += 1
        st["probes"] += 1
        left -= 1
        if r is not None and r.status in HostFingerprint.REFUSAL:
            fp.refused = True
            st["refused"] = r.status
            break                        # the host said stop; stop for good
        if r is not None and not fp.is_soft_404(url, r):
            # Score the URL we actually LANDED on: a probe that redirects to
            # /admissions/fee-structure/ found the real page, and its final URL
            # is what the caller should record as the source. The bar here is 0,
            # not MIN_SCORE: MIN_SCORE exists to stop the best of a bad field
            # winning when ranking thousands of sitemap URLs, whereas a probe
            # asked for this one topic by name and got a real page back — the
            # only thing left to check is that a redirect did not carry us
            # somewhere off-topic.
            final = r.url
            if score_url(kind, dict(PAGE_TARGETS)[kind], final) < 0:
                continue
            if final in found.values():
                continue
            found[kind] = final
            st["via"][kind] = "probe"
            st["probe_hits"] += 1
            if html_sink is not None and not r.truncated:
                html_sink[final] = r.text()

    return ordered()


# ---------------------------------------------------------------------------
# Measurement
#
# The deliverable is a before/after number on colleges the running sweep
# actually recorded as `empty`, so this harness lives with the code it measures
# and can be re-run. "Before" is not taken on trust: pick_pages() is re-run on
# the same freshly fetched homepage, so any difference is this module's, not a
# difference in what the site served on the day.
# ---------------------------------------------------------------------------

def _empty_colleges(n: int, sqlite_path: str | None) -> list[dict]:
    sql = ("SELECT s.college_id, s.website_url, c.name, c.city "
           "FROM web_crawl_state s JOIN colleges c USING(college_id) "
           "WHERE s.status = 'empty' AND s.website_url IS NOT NULL "
           "ORDER BY s.college_id")
    if sqlite_path:
        import sqlite3
        con = sqlite3.connect(sqlite_path)
        con.row_factory = sqlite3.Row
        rows = [dict(r) for r in con.execute(sql + " LIMIT ?", (n,))]
        con.close()
        return rows
    from ..store.backend import get_backend
    return [dict(r) for r in get_backend().q(sql + " LIMIT ?", [n])]


async def measure(n: int = 35, sqlite_path: str | None = None,
                  delay: float = 1.5, concurrency: int = 6,
                  use_llm: bool = False, save: str | None = None) -> dict:
    """Before/after on colleges the sweep recorded as `empty`.

    `save` writes every discovered page's extracted TEXT to JSONL so the
    extractor side can be re-measured (parsers vs LLM) without re-crawling
    anyone. Re-running a 370-request sweep against the same 40 small college
    servers just to change one extraction flag is not a cost worth paying twice.
    """
    import json

    from .web_enrich import extract_topic, fetch, pick_pages

    rows = _empty_colleges(n, sqlite_path)
    if not rows:
        print("[pagefind] no colleges with status='empty'", file=sys.stderr)
        return {}
    print(f"[pagefind] measuring {len(rows)} colleges the sweep called 'empty'",
          file=sys.stderr)

    pol = Politeness(delay=delay)
    sem = asyncio.Semaphore(concurrency)
    agg = {
        "colleges": len(rows), "home_ok": 0,
        "before_pages": 0, "before_found": 0,
        "after_pages": 0, "after_found": 0, "after_facts": 0,
        "requests": 0, "via_sitemap": 0, "via_probe": 0,
        "probes": 0, "probe_hits": 0,
        # The subset the brief is actually about: colleges where the homepage
        # links yield NOTHING, which is what `empty` was assumed to mean.
        "blind": 0, "blind_found": 0, "blind_facts": 0,
    }
    per_kind: dict[str, int] = {}
    lines: list[str] = []
    saved: list[dict] = []
    lock = asyncio.Lock()

    limits = httpx.Limits(max_connections=concurrency * 2,
                          max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(headers=HEADERS, timeout=20.0, limits=limits,
                                 follow_redirects=True, verify=False) as client:

        async def one(row: dict):
            site = normalise_site(row["website_url"])
            if not site:
                return
            async with sem:
                home = await fetch(client, pol, site)
                before: list[tuple[str, str]] = []
                if home:
                    _, links = parse_html(home)
                    before = pick_pages(site, links)

                st: dict = {}
                sink: dict[str, str] = {}
                rp = (getattr(pol, "_robots", {}) or {}).get(urlsplit(site).netloc)
                after = await discover_pages(client, pol, site, rp,
                                            html_sink=sink, stats=st)

                # A page only counts as recovered if a quota-free extractor can
                # actually read a fact off it. "Found a URL" is not the win.
                facts = 0
                kinds_hit = []
                pages: list[dict] = []
                for kind, url in after:
                    html = sink.get(url) or await fetch(client, pol, url)
                    if not html:
                        continue
                    text, _ = parse_html(html)
                    pages.append({"college_id": row["college_id"],
                                  "name": row["name"], "city": row.get("city"),
                                  "kind": kind, "url": url,
                                  "text": text[:20_000]})
                    got = extract_topic(row["name"], f", {row.get('city') or ''}",
                                        kind, url, text, use_llm=use_llm)
                    if got:
                        facts += 1
                        kinds_hit.append(kind)

                async with lock:
                    if home:
                        agg["home_ok"] += 1
                    agg["before_pages"] += len(before)
                    agg["after_pages"] += len(after)
                    agg["before_found"] += 1 if before else 0
                    agg["after_found"] += 1 if after else 0
                    agg["after_facts"] += 1 if facts else 0
                    agg["requests"] += st.get("requests", 0)
                    agg["probes"] += st.get("probes", 0)
                    agg["probe_hits"] += st.get("probe_hits", 0)
                    if not before:
                        agg["blind"] += 1
                        agg["blind_found"] += 1 if after else 0
                        agg["blind_facts"] += 1 if facts else 0
                    # Only count the pages actually RETURNED. `st["via"]` records
                    # every topic the sitemap matched, including the ones the
                    # MAX_PAGES_PER_COLLEGE slice then discards, and counting
                    # those would overstate the sitemap's contribution.
                    via = st.get("via") or {}
                    for k, _u in after:
                        agg["via_sitemap" if via.get(k) == "sitemap"
                            else "via_probe"] += 1
                    for k in kinds_hit:
                        per_kind[k] = per_kind.get(k, 0) + 1
                    saved.extend(pages)
                    lines.append(
                        f"  {site[:52]:<52} before={len(before)} "
                        f"after={len(after)} facts={facts} "
                        f"req={st.get('requests', 0)} "
                        f"stack={st.get('stack', '-') or '-'} "
                        f"sm={st.get('sitemap_urls', 0)} "
                        f"{','.join(kinds_hit)}")

        t0 = time.perf_counter()
        for i in range(0, len(rows), concurrency * 4):
            await asyncio.gather(*(one(r) for r in rows[i:i + concurrency * 4]))

    for ln in sorted(lines):
        print(ln)
    n_c = agg["colleges"]
    b = agg["blind"]
    print(f"\n[pagefind] {n_c} colleges the sweep recorded as 'empty', "
          f"{agg['home_ok']} homepages still reachable "
          f"({(time.perf_counter() - t0) / 60:.1f}m)")
    print(f"  BEFORE  pick_pages found a topic page for "
          f"{agg['before_found']}/{n_c}  ({agg['before_pages']} pages total)")
    print(f"  AFTER   discover_pages found one for "
          f"{agg['after_found']}/{n_c}  ({agg['after_pages']} pages total)")
    print(f"  USABLE  {'LLM+parsers' if use_llm else 'parsers only'} read a fact "
          f"for {agg['after_facts']}/{n_c}")
    print(f"\n  -- the subset the fix targets: homepage links yielded NOTHING --")
    print(f"  BLIND   {b}/{n_c} colleges where pick_pages found 0 pages")
    print(f"          discovery found a topic page for {agg['blind_found']}/{b}"
          f"  ({100 * agg['blind_found'] / max(1, b):.0f}%)")
    print(f"          a fact came out for {agg['blind_facts']}/{b}")
    print(f"\n  cost    {agg['requests']} discovery requests "
          f"({agg['requests'] / max(1, n_c):.1f}/college)")
    print(f"          {agg['via_sitemap']} pages via sitemap, "
          f"{agg['via_probe']} via probe")
    print(f"          probe yield: {agg['probe_hits']}/{agg['probes']} requests "
          f"hit ({100 * agg['probe_hits'] / max(1, agg['probes']):.1f}%)")
    print(f"  by kind {dict(sorted(per_kind.items(), key=lambda kv: -kv[1]))}")
    if save and saved:
        with open(save, "w", encoding="utf-8") as f:
            for rec in saved:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  saved   {len(saved)} page texts -> {save}")
    return agg


def replay(path: str, use_llm: bool = True) -> dict:
    """Re-measure extraction over saved page texts. No network, no re-crawl."""
    import json

    from .web_enrich import extract_topic

    # The same 0.5 the card renderer uses to decide a fact is worth showing.
    # An extractor returning confidence 0.0 has told us it found nothing
    # concrete, so counting it as a recovered college would be flattering the
    # result: several discovered pages are real topic pages whose content is a
    # link to a PDF, and they parse to nothing.
    SHOWABLE = 0.5

    per_college: dict[str, set[str]] = {}
    strong_college: dict[str, set[str]] = {}
    seen: set[str] = set()
    n_pages = 0
    per_kind: dict[str, int] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            n_pages += 1
            seen.add(rec["college_id"])
            got = extract_topic(rec["name"], f", {rec.get('city') or ''}",
                                rec["kind"], rec["url"], rec["text"],
                                use_llm=use_llm)
            if got:
                per_college.setdefault(rec["college_id"], set()).add(rec["kind"])
                conf = float(got.get("confidence") or 0.0)
                if conf >= SHOWABLE:
                    strong_college.setdefault(rec["college_id"], set()).add(
                        rec["kind"])
                    per_kind[rec["kind"]] = per_kind.get(rec["kind"], 0) + 1
                print(f"  {'FACT' if conf >= SHOWABLE else 'weak'} "
                      f"{rec['kind']:12} conf={conf:.2f} "
                      f"via={got.get('_via')} {rec['url'][:76]}")
    n_strong = sum(len(v) for v in strong_college.values())
    print(f"\n[replay] {n_pages} discovered pages from {len(seen)} colleges "
          f"({'LLM+parsers' if use_llm else 'parsers only'})")
    print(f"  any fact at all           "
          f"{len(per_college)}/{len(seen)} colleges")
    print(f"  fact at confidence >={SHOWABLE}  "
          f"{len(strong_college)}/{len(seen)} colleges, {n_strong} facts")
    print(f"  by kind {dict(sorted(per_kind.items(), key=lambda kv: -kv[1]))}")
    return {"pages": n_pages, "colleges": len(seen),
            "with_facts": len(per_college), "with_strong": len(strong_college),
            "strong_facts": n_strong}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--measure", type=int, metavar="N",
                    help="run the before/after harness on N status='empty' colleges")
    ap.add_argument("--sqlite", default=None,
                    help="read web_crawl_state from this SQLite file instead of "
                         "the configured backend (the earlier sweep's history)")
    ap.add_argument("--delay", type=float, default=1.5)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--llm", action="store_true",
                    help="also allow LLM extraction when scoring usability "
                         "(off by default: costs quota, and the point is to "
                         "measure discovery, not the extractor)")
    ap.add_argument("--url", default=None, help="discover pages for one origin")
    ap.add_argument("--save", default=None,
                    help="write discovered page texts to this JSONL")
    ap.add_argument("--replay", default=None,
                    help="re-measure extraction over a saved JSONL (no network)")
    a = ap.parse_args()

    if a.replay:
        replay(a.replay, use_llm=a.llm)
        return

    if a.url:
        async def run_one():
            pol = Politeness(delay=a.delay)
            async with httpx.AsyncClient(headers=HEADERS, timeout=20.0,
                                         follow_redirects=True,
                                         verify=False) as c:
                st: dict = {}
                got = await discover_pages(c, pol, a.url, stats=st)
                for k, u in got:
                    print(f"  {k:12} {u}")
                print(f"  [{st}]")
        asyncio.run(run_one())
        return
    if a.measure:
        asyncio.run(measure(a.measure, a.sqlite, delay=a.delay,
                            concurrency=a.concurrency, use_llm=a.llm,
                            save=a.save))
        return
    ap.print_help()


if __name__ == "__main__":
    main()
