"""Several sources, one defensible answer.

THE PROBLEM THIS EXISTS FOR
The catalogue is ~100% empty for NAAC, NIRF, placements, scholarships, hostel
and cutoffs, so every one of those fields is being filled from somewhere else:
the college's own website, government registries, and whatever the MME
catalogue does hold. Those sources overlap, and when they overlap they
disagree. `college_web_facts` cannot represent that — it is keyed
(college_id, kind) and its upsert overwrites in place, so the second source to
run silently erases the first and MIVI loses the ability to answer the only
question that matters when a number looks wrong: who told you?

So claims are stored per source (`college_source_fact`) and a winner is
computed (`college_fact_resolved`) with the losers kept verbatim beside it. A
disagreement between two sources about a fee is a fact about our data. It gets
recorded, not resolved away.

WHAT "DEFENSIBLE" MEANS HERE
1. AUTHORITY IS PER FIELD, NOT GLOBAL. The advertised tier order — official
   site, government registry, MME catalogue, other — is right for facts a
   college states about ITSELF (name, address, its own fee notification). It is
   wrong for facts a registry CERTIFIES about the college: a NAAC grade or NIRF
   rank on a college's own marketing page is a claim about someone else's
   judgement, and the body that made the judgement is the better source.
   KIND_AUTHORITY below inverts the top two tiers for exactly those fields, and
   nothing else.
2. FRESHNESS CAN OUTRANK AUTHORITY, BY ONE TIER, FOR FIELDS THAT CHANGE.
   Justified in full at STALE_AFTER_DAYS.
3. AGREEMENT IS EARNED ON NORMALISED NUMBERS, NOT ON STRINGS. "Rs 1,20,000",
   "120000" and "1.2 Lakh" are the same fee and must corroborate each other;
   "Rs 60,000 per semester" and "Rs 1,20,000 per year" are NOT, and must not.
   The normalisation is for COMPARISON ONLY — house rule 3 stands, every value
   stored and displayed is the source's own text with the source's own unit.
4. THE ARITHMETIC IS CONSERVATIVE IN ONE DIRECTION. Where the comparison
   cannot tell "extra detail" from "contradiction" it reports a conflict. The
   cost of that is a hedged answer; the cost of the opposite error is a
   confidently wrong fee shown to a nervous teenager.

The whole resolver is pure Python over a list of claims — no LLM call, no
network, microseconds per group. Only record/reconcile/read touch the database.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path

from ..store.backend import get_backend

# ---------------------------------------------------------------------------
# Tiers and authority
# ---------------------------------------------------------------------------

TIER_OFFICIAL = 1     # the college's own website
TIER_GOV = 2          # government registry (NIRF, NAAC, AISHE, AICTE, state boards)
TIER_CATALOGUE = 3    # the MME catalogue we already ship
TIER_OTHER = 4        # anything else admissible under the scraping rules

# Default tier from the source_key prefix. A claim may override it explicitly
# (record_claim(source_tier=...)) because tiering is a judgement — a state
# technical board might be promoted from "other" to "gov" later — and existing
# claims should keep the tier they were recorded under until something
# deliberately re-tiers them.
CLASS_TIER = {
    "official": TIER_OFFICIAL,
    "gov": TIER_GOV,
    "govt": TIER_GOV,
    "mme": TIER_CATALOGUE,
    "catalogue": TIER_CATALOGUE,
    "other": TIER_OTHER,
}

# Fields where a registry outranks the college's own site. See point 1 above:
# these are all facts about an external body's verdict, not about the college.
# Everything not listed here uses the plain tier order.
KIND_AUTHORITY = {
    "naac_grade":   {"gov": 1, "official": 2},
    "nirf_ranking": {"gov": 1, "official": 2},
    "approval":     {"gov": 1, "official": 2},
    "affiliation":  {"gov": 1, "official": 2},
    "intake":       {"gov": 1, "official": 2},
}

# Fields that change on the admission cycle. Only these are subject to the
# staleness demotion below; a stale NAME is still a name.
VOLATILE_KINDS = frozenset({
    "fees", "hostel", "scholarship", "cutoff", "placement", "admission",
    "events", "intake", "nirf_ranking", "seats",
})

# THE INTERESTING CASE, AND THE RULE FOR IT.
#
# A tier-1 fee page last fetched in 2023 versus a tier-2 government figure
# fetched last week. The naive rule (tier always wins) shows the 2023 number,
# and it is wrong — not because the college is a worse source, but because it
# is answering a different year's question. Indian fee notifications land
# roughly April-June and move 5-12% a year; a two-year-old official figure
# under-quotes a student's actual cost, and under-quoting is the failure mode
# that ends with someone unable to pay in September.
#
# So: for VOLATILE_KINDS only, a claim older than this is demoted by exactly
# ONE tier. One, not two, and that number carries the whole rule:
#   * demoted tier-1 lands on 2, TYING with a fresh registry claim, and the
#     existing newer-wins tie-break then picks the fresh one. Freshness decides
#     only when the two are otherwise equally credible.
#   * a stale tier-1 still beats tier-3 and tier-4. Old and official outranks
#     current and unaccountable.
#   * if the stale tier-1 is the ONLY claim it still wins, with its confidence
#     decayed (STALE_DECAY) so the card renderer's threshold can drop it.
# The demoted claim is never discarded: it lands in `rejected` with its age, so
# MIVI can still say "the college's own site said X when we last checked, in
# March 2023".
#
# 400 days, not 365: at exactly one year a March fee page goes stale in March,
# which is precisely when the new notification has not been published yet and
# the old official figure is still the best thing anyone has. 400 pushes the
# flip past the publication window.
STALE_AFTER_DAYS = int(os.getenv("MIVI_FACT_STALE_DAYS", "400"))

# ---------------------------------------------------------------------------
# Confidence arithmetic
#
# Priors are about EXTRACTION reliability, which is not the same axis as
# authority. A government registry is a dated, structured, transcribed record;
# a college website is a regex run over marketing HTML. So gov carries the
# highest prior even though `official` outranks it on authority for most kinds.
# Tier answers "whom do we believe about this subject"; the prior answers "how
# likely is this particular value to have been read off the page correctly".
# ---------------------------------------------------------------------------
TIER_PRIOR = {1: 0.80, 2: 0.88, 3: 0.72, 4: 0.50}

# Corroboration closes a fraction of the remaining gap to certainty, so it
# diminishes and never arrives: 0.80 -> 0.87 -> 0.9155. A second source that
# agrees is strong evidence; a fifth is nearly none.
CORROBORATION_LIFT = 0.35        # agreeing source at tier 1-2
WEAK_CORROBORATION_LIFT = 0.18   # agreeing source at tier 3-4

# Conflict is multiplicative, and heavier when the conflicting source sits at
# the SAME effective tier — that is the genuinely ambiguous case, where tier
# gives us no basis at all for preferring the winner.
CONFLICT_PENALTY = 0.30
SAME_TIER_CONFLICT_PENALTY = 0.45

# A contested value can never look confident, however good the winner was.
# render_card drops web facts below 0.5, so a hard ceiling here is what makes a
# conflict actually visible downstream instead of merely logged.
CONFLICT_CEILING = 0.60

STALE_DECAY = 0.85               # multiplier on a stale volatile winner
CONF_FLOOR = 0.10
CONF_CEILING = 0.97              # never 1.0: nothing a scraper does earns certainty

TEXT_AGREE_RATIO = 0.90          # normalised-similarity bar for non-numeric values
SHORT_CODE_LEN = 8               # "A++" vs "A+" must never blur; exact match only


# ---------------------------------------------------------------------------
# Number normalisation — FOR COMPARISON ONLY
#
# Nothing below is ever written back or shown. Its only job is to decide
# whether two verbatim strings are talking about the same money.
# ---------------------------------------------------------------------------

_CURRENCY = {
    "rs": "INR", "rs.": "INR", "inr": "INR", "₹": "INR", "rupee": "INR",
    "rupees": "INR", "usd": "USD", "us$": "USD", "$": "USD", "gbp": "GBP",
    "£": "GBP", "eur": "EUR", "€": "EUR", "aed": "AED",
    "cad": "CAD", "aud": "AUD", "sgd": "SGD",
}

_CURRENCY_RE = re.compile(
    r"₹|£|€|\$|\b(?:INR|Rs\.?|rupees?|USD|GBP|EUR|AED|CAD|AUD|SGD)\b",
    re.I)

# Indian grouping ("1,20,000") is 2-digit after the first group, western
# ("120,000") is 3 — both are just separators, so both are accepted and
# stripped. Getting this wrong is the classic bug: 1,20,000 read as 120.
_AMOUNT_RE = re.compile(
    r"(?P<pre>₹|£|€|\$|INR|Rs\.?|USD|GBP|EUR|AED|CAD|AUD|SGD)?"
    r"\s{0,3}"
    r"(?P<num>\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\s{0,3}"
    r"(?P<unit>lakhs?|lacs?|crores?|L|Cr|K)?(?![a-zA-Z])",
    re.I)

_UNIT_MULT = {"lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5, "l": 1e5,
              "crore": 1e7, "crores": 1e7, "cr": 1e7, "k": 1e3}

# A bare number with no currency mark and no unit has to clear this to count as
# money. Below it lives everything that is not a fee: counts of hostels, seat
# numbers, page furniture.
_BARE_MIN = 1_000.0
_MARKED_MIN = 1.0
# Above this it is not a fee at any Indian college; it is a phone number, a
# pincode run together, or an extraction accident.
_SANE_MAX = 1e9

_PERIODS = (
    # Checked in this order; "per semester" must not be read as "per year"
    # because a wrong period is a 2x error in a number a family budgets against.
    ("semester", re.compile(r"per\s+sem(?:ester)?\b|/\s?sem(?:ester)?\b|"
                            r"semester[-\s]?wise|each\s+semester|\bsemester\s+fee",
                            re.I)),
    ("month",    re.compile(r"per\s+month\b|/\s?month\b|monthly\b|\bp\s?\.\s?m\.?\b",
                            re.I)),
    ("year",     re.compile(r"per\s+(?:academic\s+)?(?:year|annum)\b|/\s?(?:year|yr|annum)\b|"
                            r"annual(?:ly)?\b|yearly\b|\bp\s?\.\s?a\.?\b", re.I)),
    ("course",   re.compile(r"(?:total|entire|whole|full|complete)\s+"
                            r"(?:course|program(?:me)?|duration|fee|fees)\b|"
                            r"for\s+the\s+(?:entire|full|complete)\b", re.I)),
)


def _looks_like_year(num: str) -> bool:
    s = num.replace(",", "")
    return len(s) == 4 and s.isdigit() and 1900 <= int(s) <= 2099


@dataclass(frozen=True)
class Sig:
    """A value reduced to what can be compared across sources."""
    amounts: frozenset = frozenset()
    currencies: frozenset = frozenset()
    periods: frozenset = frozenset()

    @property
    def numeric(self) -> bool:
        return bool(self.amounts)


def money_signature(text: str) -> Sig:
    """Every amount in `text`, normalised to a plain number of currency units.

    Public because it is the part worth testing and reusing; `_selftest`
    exercises it directly, and it is the single place the Indian formats are
    understood.
    """
    if not text:
        return Sig()
    amounts: set[float] = set()
    for m in _AMOUNT_RE.finditer(text):
        num, unit, pre = m.group("num"), m.group("unit"), m.group("pre")
        try:
            val = float(num.replace(",", ""))
        except ValueError:
            continue
        marked = bool(pre or unit)
        if unit:
            val *= _UNIT_MULT.get(unit.lower(), 1.0)
        if not marked:
            # "Session 2025-26" and "NIRF 2024" are not amounts. A digit run
            # long enough to be a phone number is not one either.
            if _looks_like_year(num) or len(num.replace(",", "")) >= 10:
                continue
        floor = _MARKED_MIN if marked else _BARE_MIN
        if floor <= val <= _SANE_MAX:
            # Rounded to the paisa: float multiplication by 1e5 leaves
            # 1.2 lakh at 120000.00000000001 on some inputs, and that must
            # still equal a literal 120000.
            amounts.add(round(val, 2))
    currencies = {_CURRENCY.get(t.lower().rstrip("."), t.upper())
                  for t in _CURRENCY_RE.findall(text)}
    periods = {name for name, rx in _PERIODS if rx.search(text)}
    return Sig(frozenset(amounts), frozenset(currencies), frozenset(periods))


def _compatible(a: Sig, b: Sig) -> bool:
    """Do two signatures describe the same money?

    Amounts must match exactly as a set. Currency and period are checked as
    NON-CONTRADICTION rather than equality: an unstated period is unknown, not
    "per year", and demanding equality would turn "1.2 Lakh" versus
    "Rs 1,20,000 per year" — the same fee — into a conflict. Two STATED and
    different periods do contradict, and that is the case worth catching: never
    annualise, so 60,000/semester and 1,20,000/year are different claims even
    though a human can see the arithmetic.
    """
    if a.amounts != b.amounts:
        return False
    if a.periods and b.periods and a.periods != b.periods:
        return False
    if a.currencies and b.currencies and a.currencies != b.currencies:
        return False
    return True


# ---------------------------------------------------------------------------
# Text and label normalisation
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset({"the", "of", "and", "a", "an", "for", "in", "at", "to",
                        "is", "are", "per", "as", "on", "with"})
_PUNCT_RE = re.compile(r"[^\w\s+]+", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def _norm_text(text: str) -> str:
    """Casefold, drop punctuation, collapse space. `+` survives because it is
    load-bearing in a NAAC grade ("A++")."""
    s = _PUNCT_RE.sub(" ", (text or "").lower())
    toks = _WS_RE.sub(" ", s).strip().split()
    if len(toks) > 4:
        toks = [t for t in toks if t not in _STOPWORDS] or toks
    return " ".join(toks)


# Period words are stripped from LABELS so "Tuition Fee (per semester)" and
# "Annual Tuition Fee" land on the same key and can be compared at all — the
# period they carry is not lost, it goes into that label's Sig instead.
#
# "total" is deliberately NOT in here. It is a scope word, not a period: a
# 4-year course total and an annual tuition are different facts, and folding
# "Total Fee" onto the "tuition" key would compare them as if they were the
# same one. Kept distinct, they fall through to the amount comparison and are
# correctly reported as a conflict.
_LABEL_NOISE_RE = re.compile(
    r"\b(?:per|each|semester|sem|year|yr|annum|annual|annually|monthly|month|"
    r"wise|approx|approximately|rs|inr)\b", re.I)

_LABEL_SYNONYM = {
    "tuition": "tuition", "tuitionfee": "tuition", "tuitionfees": "tuition",
    "coursefee": "tuition", "coursefees": "tuition", "academicfee": "tuition",
    "academicfees": "tuition", "programfee": "tuition", "programmefee": "tuition",
    # An unqualified "Fee" column in a fee table is tuition. Anything else in
    # such a table is labelled (hostel, mess, caution), so the bare case is safe.
    "fee": "tuition", "fees": "tuition",
    "hostel": "hostel", "hostelfee": "hostel", "hostelfees": "hostel",
    "accommodation": "hostel", "accommodationfee": "hostel", "boarding": "hostel",
    "mess": "mess", "messfee": "mess", "messcharges": "mess", "food": "mess",
    "admissionfee": "admission", "admissioncharges": "admission",
    "onetime": "onetime", "onetimefee": "onetime", "onetimecharges": "onetime",
    "caution": "caution", "cautiondeposit": "caution", "cautionmoney": "caution",
    "security": "caution", "securitydeposit": "caution",
    "exam": "exam", "examfee": "exam", "examinationfee": "exam",
    "development": "development", "developmentfee": "development",
    "totalfee": "total", "totalfees": "total", "grandtotal": "total",
}


def _norm_label(label: str) -> str:
    stripped = _LABEL_NOISE_RE.sub(" ", label or "")
    key = re.sub(r"[^a-z0-9]+", "", stripped.lower())
    return _LABEL_SYNONYM.get(key, key)


def _payload_text(payload) -> str:
    """Flatten a payload into one string for signature extraction.

    The extractors emit a flat {label: verbatim} dict, which is the shape this
    is tuned for, but an LLM-sourced payload or a hand-loaded claim can be a
    list or a nested dict — those are stringified rather than rejected, because
    losing a whole claim to an unexpected shape is worse than a slightly noisy
    comparison.
    """
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        return " ; ".join(f"{k}: {v}" for k, v in payload.items())
    if isinstance(payload, (list, tuple)):
        return " ; ".join(str(v) for v in payload)
    return str(payload)


def _labelled_sigs(payload) -> dict[str, Sig]:
    """Per-label signatures, keeping only labels that actually carry money."""
    if not isinstance(payload, dict):
        return {}
    out: dict[str, Sig] = {}
    for k, v in payload.items():
        if isinstance(v, (dict, list, tuple)) or v is None:
            continue
        # The label is fed in alongside the value so a period stated in the
        # column header ("Fee per semester") reaches the signature.
        sig = money_signature(f"{k} {v}")
        if sig.numeric:
            key = _norm_label(str(k))
            if key:
                out[key] = sig
    return out


# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------

# Two-label public suffixes, needed to collapse a host to its institution.
# Getting this wrong is not a cosmetic bug: naively keeping the last two labels
# turns EVERY Indian college into origin "ac.in", so two unrelated colleges
# would count as one voice and corroboration would become meaningless. Indian
# academic hosts are overwhelmingly under these.
_MULTI_SUFFIX = frozenset({
    "ac.in", "edu.in", "res.in", "org.in", "net.in", "co.in", "gen.in",
    "firm.in", "ind.in", "gov.in", "nic.in", "ernet.in", "mil.in",
    "ac.uk", "gov.uk", "org.uk", "co.uk", "sch.uk",
    "edu.au", "gov.au", "com.au", "org.au",
    "edu.pk", "edu.bd", "edu.np", "edu.lk", "ac.nz", "edu.sg", "com.sg",
    "edu.my", "ac.ae", "edu.ph", "ac.id", "ac.jp", "edu.cn", "ac.za",
})


def _origin_host(host: str) -> str:
    """Host reduced to the institution that publishes it."""
    labels = [p for p in host.split(".") if p]
    if len(labels) <= 2:
        return ".".join(labels)
    if ".".join(labels[-2:]) in _MULTI_SUFFIX:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


@dataclass
class Claim:
    college_id: str
    kind: str
    source_key: str
    source_tier: int = TIER_OTHER
    summary: str | None = None
    payload: object = None
    source_url: str | None = None
    fetched_at: datetime | None = None
    confidence: float | None = None

    @property
    def source_class(self) -> str:
        return self.source_key.split(":", 1)[0].strip().lower()

    @property
    def origin(self) -> str:
        """What makes two claims INDEPENDENT.

        `official:iitr.ac.in` and `official:admissions.iitr.ac.in` are one
        institution's word, twice. Counting them as two agreeing sources would
        manufacture confidence out of a site's own internal duplication, which
        is the easiest way for this whole layer to become dishonest. So the host
        collapses to the institution that publishes it, and the class is kept —
        a registry and a college agreeing is corroboration, a site agreeing with
        itself is not.
        """
        cls, _, rest = self.source_key.partition(":")
        host = rest.strip().lower().split("/")[0]
        return f"{cls.lower()}:{_origin_host(host)}"

    def prior(self) -> float:
        c = self.confidence
        if c is None:
            return TIER_PRIOR.get(self.source_tier, 0.5)
        return max(0.0, min(1.0, float(c)))

    def sig(self) -> Sig:
        return money_signature(
            " ; ".join(p for p in (self.summary or "", _payload_text(self.payload)) if p))

    def value_text(self) -> str:
        return (self.summary or "").strip() or _payload_text(self.payload).strip()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _age_days(claim: Claim, now: datetime) -> float | None:
    if claim.fetched_at is None:
        return None
    ts = claim.fetched_at
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (now - ts).total_seconds() / 86400.0


def _authority_tier(claim: Claim) -> int:
    """Tier after the per-kind authority override, before staleness."""
    override = KIND_AUTHORITY.get(claim.kind)
    if override:
        t = override.get(claim.source_class)
        if t is not None:
            return t
    return int(claim.source_tier)


def _is_stale(claim: Claim, now: datetime) -> bool:
    if claim.kind not in VOLATILE_KINDS:
        return False
    age = _age_days(claim, now)
    return age is not None and age > STALE_AFTER_DAYS


def _rank_key(claim: Claim, now: datetime, *, demote: bool = True) -> tuple:
    """Sort key; lowest wins.

    The trailing source_key is not cosmetic. Without a total order, two claims
    identical on tier, timestamp and confidence would resolve to whichever the
    query happened to return first, and reconcile_all() would stop being
    idempotent — the fingerprint would flip between runs and every re-run would
    rewrite the table (house rule 5).
    """
    tier = _authority_tier(claim)
    if demote and _is_stale(claim, now):
        tier = min(tier + 1, TIER_OTHER)
    ts = claim.fetched_at
    if ts is not None and ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    epoch = ts.timestamp() if ts is not None else 0.0
    return (tier, -epoch, -claim.prior(), claim.source_key)


def agrees(a: Claim, b: Claim) -> tuple[bool, str]:
    """Do two claims say the same thing? Returns (verdict, basis).

    Three strategies, most precise first:

      labelled — both payloads carry money under a shared normalised label.
        This is the accurate path and the extractors do emit label->verbatim
        dicts, so it is also the common one: compare tuition against tuition,
        never tuition against a caution deposit that happens to be the same
        number.

      amounts — no shared label. The full amount sets must be EQUAL, not
        overlapping. A subset rule would be kinder to a source that lists only
        tuition against one that lists tuition plus hostel, but it cannot
        distinguish that from a genuine contradiction, and the conservative
        error here is a hedged answer rather than a confident wrong one.

      text — neither side states an amount (a NAAC grade, a college name, a
        placement blurb). Normalised similarity, with an exact-match rule for
        short codes so "A++" can never blur into "A+".
    """
    la, lb = _labelled_sigs(a.payload), _labelled_sigs(b.payload)
    shared = set(la) & set(lb)
    if shared:
        for key in shared:
            if not _compatible(la[key], lb[key]):
                return False, "labelled"
        return True, "labelled"

    sa, sb = a.sig(), b.sig()
    if sa.numeric or sb.numeric:
        return _compatible(sa, sb), "amounts"

    ta, tb = _norm_text(a.value_text()), _norm_text(b.value_text())
    if not ta or not tb:
        return False, "text"
    if len(ta) <= SHORT_CODE_LEN or len(tb) <= SHORT_CODE_LEN:
        return ta == tb, "text"
    return SequenceMatcher(None, ta, tb).ratio() >= TEXT_AGREE_RATIO, "text"


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

@dataclass
class Resolution:
    college_id: str
    kind: str
    winner: Claim
    confidence: float
    agreement: str                       # single | corroborated | conflict
    n_sources: int
    n_agreeing: int
    rule: str
    rejected: list[dict] = field(default_factory=list)
    fingerprint: str = ""

    def as_fact(self) -> dict:
        """The shape the card renderer wants. Deliberately a superset of what
        render_card already reads off `college_web_facts`, so resolved facts can
        be passed straight into its `web_facts` argument."""
        conflicts = [r for r in self.rejected if r.get("verdict") == "conflicts"]
        out = {
            "summary": self.winner.summary,
            "payload": self.winner.payload,
            "confidence": round(self.confidence, 4),
            "source_key": self.winner.source_key,
            "source_tier": self.winner.source_tier,
            "source_url": self.winner.source_url,
            "fetched_at": self.winner.fetched_at,
            "agreement": self.agreement,
            "n_sources": self.n_sources,
            "n_agreeing": self.n_agreeing,
            "rule": self.rule,
            "conflicts": conflicts,
        }
        if conflicts:
            # Pre-built so the renderer never has to invent the wording, and
            # never states one figure as settled when two sources disagree.
            other = conflicts[0]
            out["caveat"] = (
                f"Sources disagree. {self.winner.source_key} states "
                f"\"{(self.winner.value_text() or '')[:120]}\"; "
                f"{other['source_key']} states \"{(other.get('value') or '')[:120]}\". "
                f"Confirm on the official site before relying on it."
            )
        return out


def _fingerprint(res: Resolution) -> str:
    """Digest of the DECISION, not of the inputs.

    Fingerprinting the outcome covers every input that could have moved it —
    including the staleness verdict, which is a function of today's date and not
    of the claims at all. A tier-1 fee crossing the 400-day line changes the
    answer with no new claim anywhere, and an input-only digest would miss it.
    """
    payload = json.dumps({
        "winner": res.winner.source_key,
        "tier": res.winner.source_tier,
        "summary": res.winner.summary,
        "value": res.winner.value_text(),
        "conf": round(res.confidence, 4),
        "agreement": res.agreement,
        "n": [res.n_sources, res.n_agreeing],
        "rule": res.rule,
        "rejected": [(r.get("source_key"), r.get("verdict"), r.get("reason"),
                      r.get("value")) for r in res.rejected],
    }, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def resolve(claims: list[Claim], now: datetime | None = None) -> Resolution | None:
    """Pure. One (college_id, kind) group in, one winner plus the audit out.

    Every other claim is compared AGAINST THE WINNER rather than clustered
    transitively. Transitive clustering has no well-defined answer for money
    (a can match b and b match c while a and c differ), and the question this
    number has to answer is narrower anyway: how much should MIVI trust the
    value it is about to show?
    """
    claims = [c for c in claims if c is not None]
    if not claims:
        return None
    now = now or _now()

    ranked = sorted(claims, key=lambda c: _rank_key(c, now))
    winner = ranked[0]

    # Did the staleness demotion actually change the outcome? Only worth saying
    # so in `rule` when it did — otherwise every volatile field would carry a
    # note about a rule that did nothing.
    naive = sorted(claims, key=lambda c: _rank_key(c, now, demote=False))[0]
    stale_flipped = naive.source_key != winner.source_key

    win_tier = _rank_key(winner, now)[0]
    rejected: list[dict] = []
    counted_origins = {winner.origin}
    n_agreeing = 1
    conf = winner.prior()

    if _is_stale(winner, now):
        # Winning while stale is still winning, but the number is older than
        # the cycle it describes and the confidence has to show that.
        conf *= STALE_DECAY

    for other in ranked[1:]:
        ok, basis = agrees(winner, other)
        other_tier = _rank_key(other, now)[0]
        age = _age_days(other, now)
        entry = {
            "source_key": other.source_key,
            "source_tier": other.source_tier,
            "effective_tier": other_tier,
            "verdict": "agrees" if ok else "conflicts",
            "basis": basis,
            "value": other.value_text()[:600],   # VERBATIM, as published
            "source_url": other.source_url,
            "fetched_at": other.fetched_at.isoformat() if other.fetched_at else None,
            "confidence": other.prior(),
            "stale": _is_stale(other, now),
        }
        if _is_stale(other, now) and age is not None:
            entry["age_days"] = int(age)

        independent = other.origin not in counted_origins
        if not independent:
            # Same institution twice. Recorded for provenance, but it neither
            # raises nor lowers confidence.
            entry["reason"] = "same-origin"
            rejected.append(entry)
            continue
        counted_origins.add(other.origin)

        if ok:
            n_agreeing += 1
            lift = CORROBORATION_LIFT if other_tier <= TIER_GOV else WEAK_CORROBORATION_LIFT
            conf += (1.0 - conf) * lift
            reason = "corroborates-winner"
        else:
            penalty = (SAME_TIER_CONFLICT_PENALTY if other_tier == win_tier
                       else CONFLICT_PENALTY)
            conf *= (1.0 - penalty)
            reason = ("lower-tier" if other_tier > win_tier
                      else "same-tier-older-or-less-confident")
        # Naming the demotion in the loser's own row, not only in the winner's
        # `rule`: someone auditing why MIVI stopped quoting a college's own fee
        # reads THIS row, and "same-tier" is only true after the demotion that
        # put it there.
        if entry["stale"]:
            reason += "+stale-demoted"
        entry["reason"] = reason
        rejected.append(entry)

    conflicts = sum(1 for r in rejected if r["verdict"] == "conflicts")
    if conflicts:
        agreement = "conflict"
        conf = min(conf, CONFLICT_CEILING)
    elif n_agreeing > 1:
        agreement = "corroborated"
    else:
        agreement = "single"
    conf = max(CONF_FLOOR, min(CONF_CEILING, conf))

    if len(claims) == 1:
        rule = "sole-source"
    else:
        runner = ranked[1]
        wk, rk = _rank_key(winner, now), _rank_key(runner, now)
        if wk[0] != rk[0]:
            rule = "tier"
        elif wk[1] != rk[1]:
            rule = "freshness"
        elif wk[2] != rk[2]:
            rule = "confidence"
        else:
            rule = "source-key-tiebreak"
        if stale_flipped:
            rule += ":stale-demotion"

    res = Resolution(
        college_id=winner.college_id, kind=winner.kind, winner=winner,
        confidence=conf, agreement=agreement, n_sources=len(claims),
        n_agreeing=n_agreeing, rule=rule, rejected=rejected,
    )
    res.fingerprint = _fingerprint(res)
    return res


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "store" / "multisource.sql"


def ensure_schema() -> None:
    """Create the two tables. Additive and idempotent — safe to run against the
    live database while the harvest sweep is writing to its own tables.

    Goes to the pool directly rather than through backend.execute(): the script
    is several statements, and psycopg only accepts a multi-statement query on
    the simple protocol, which it uses only when no parameters are bound.
    """
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    be = get_backend()
    pool = getattr(be, "_pool", None)
    if pool is None:
        raise RuntimeError(f"reconcile needs the Postgres pool; backend is {be.name!r}")
    with pool.connection() as conn:
        conn.execute(sql)


_CLAIM_UPSERT = """
INSERT INTO college_source_fact
    (college_id, kind, source_key, source_tier, summary, payload, source_url,
     fetched_at, confidence)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (college_id, kind, source_key) DO UPDATE SET
    source_tier = EXCLUDED.source_tier,
    summary     = EXCLUDED.summary,
    payload     = EXCLUDED.payload,
    source_url  = EXCLUDED.source_url,
    fetched_at  = EXCLUDED.fetched_at,
    confidence  = EXCLUDED.confidence
WHERE college_source_fact.fetched_at <= EXCLUDED.fetched_at"""


def _claim_row(college_id: str, kind: str, source_key: str, *, summary=None,
               payload=None, source_url=None, fetched_at=None, confidence=None,
               source_tier=None) -> tuple:
    cls = source_key.split(":", 1)[0].strip().lower()
    tier = int(source_tier) if source_tier is not None else CLASS_TIER.get(cls, TIER_OTHER)
    if not 1 <= tier <= 4:
        raise ValueError(f"source_tier must be 1-4, got {tier!r}")
    if ":" not in source_key:
        raise ValueError(
            f"source_key must be '<class>:<origin>' (e.g. official:iitr.ac.in); "
            f"got {source_key!r} — without the class prefix the tier cannot be "
            f"derived and two sources cannot be told apart by origin")
    return (college_id, kind, source_key, tier, summary,
            json.dumps(payload, ensure_ascii=False, default=str) if payload is not None else None,
            source_url, fetched_at or _now(),
            None if confidence is None else float(confidence))


def record_claim(college_id: str, kind: str, source_key: str, *, summary=None,
                 payload=None, source_url=None, fetched_at=None, confidence=None,
                 source_tier=None) -> int:
    """Record ONE source's claim. Does not resolve anything.

    The upsert refuses to overwrite a claim with an OLDER one
    (`fetched_at <= EXCLUDED.fetched_at`). That guard is what makes a re-run of
    a partially-completed sweep safe: replaying yesterday's batch must not undo
    a value fetched since (house rule 5).
    """
    return get_backend().execute(_CLAIM_UPSERT, _claim_row(
        college_id, kind, source_key, summary=summary, payload=payload,
        source_url=source_url, fetched_at=fetched_at, confidence=confidence,
        source_tier=source_tier))


def record_claims(rows) -> int:
    """Batch form of record_claim. One transaction: all or nothing, so a killed
    job never leaves a college half-claimed."""
    prepared = [_claim_row(r["college_id"], r["kind"], r["source_key"],
                           summary=r.get("summary"), payload=r.get("payload"),
                           source_url=r.get("source_url"),
                           fetched_at=r.get("fetched_at"),
                           confidence=r.get("confidence"),
                           source_tier=r.get("source_tier"))
                for r in rows]
    if not prepared:
        return 0
    return get_backend().executemany(_CLAIM_UPSERT, prepared)


_RESOLVED_UPSERT = """
INSERT INTO college_fact_resolved
    (college_id, kind, summary, payload, source_key, source_tier, source_url,
     fetched_at, confidence, agreement, n_sources, n_agreeing, rejected, rule,
     fingerprint, resolved_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (college_id, kind) DO UPDATE SET
    summary     = EXCLUDED.summary,
    payload     = EXCLUDED.payload,
    source_key  = EXCLUDED.source_key,
    source_tier = EXCLUDED.source_tier,
    source_url  = EXCLUDED.source_url,
    fetched_at  = EXCLUDED.fetched_at,
    confidence  = EXCLUDED.confidence,
    agreement   = EXCLUDED.agreement,
    n_sources   = EXCLUDED.n_sources,
    n_agreeing  = EXCLUDED.n_agreeing,
    rejected    = EXCLUDED.rejected,
    rule        = EXCLUDED.rule,
    fingerprint = EXCLUDED.fingerprint,
    resolved_at = EXCLUDED.resolved_at
WHERE college_fact_resolved.fingerprint <> EXCLUDED.fingerprint"""


def _resolved_row(res: Resolution, now: datetime) -> tuple:
    w = res.winner
    return (res.college_id, res.kind, w.summary,
            json.dumps(w.payload, ensure_ascii=False, default=str) if w.payload is not None else None,
            w.source_key, int(w.source_tier), w.source_url, w.fetched_at,
            round(res.confidence, 4), res.agreement, res.n_sources,
            res.n_agreeing,
            json.dumps(res.rejected, ensure_ascii=False, default=str),
            res.rule, res.fingerprint, now)


_CLAIM_SELECT = """
SELECT college_id, kind, source_key, source_tier, summary, payload, source_url,
       fetched_at, confidence
  FROM college_source_fact"""


def _claims_for(where: str, params: list) -> dict[tuple[str, str], list[Claim]]:
    groups: dict[tuple[str, str], list[Claim]] = {}
    for r in get_backend().q(f"{_CLAIM_SELECT} WHERE {where}", params):
        groups.setdefault((r["college_id"], r["kind"]), []).append(
            Claim(college_id=r["college_id"], kind=r["kind"],
                  source_key=r["source_key"], source_tier=r["source_tier"],
                  summary=r["summary"], payload=r["payload"],
                  source_url=r["source_url"], fetched_at=r["fetched_at"],
                  confidence=r["confidence"]))
    return groups


def reconcile_college(college_id: str, kinds=None) -> dict[str, Resolution]:
    """Resolve every kind for one college and persist the winners."""
    where, params = "college_id = ?", [college_id]
    if kinds:
        kinds = list(kinds)
        where += f" AND kind IN ({','.join('?' * len(kinds))})"
        params += kinds
    groups = _claims_for(where, params)
    now = _now()
    out: dict[str, Resolution] = {}
    rows = []
    for (_cid, kind), claims in groups.items():
        res = resolve(claims, now)
        if res is None:
            continue
        out[kind] = res
        rows.append(_resolved_row(res, now))
    if rows:
        get_backend().executemany(_RESOLVED_UPSERT, rows)
    return out


def reconcile_all(kinds=None, batch: int = 500, progress_every: int = 20) -> dict:
    """Resolve the whole claim table, in college_id order.

    KEYSET PAGINATION, not OFFSET: the pages have to stay stable while the
    harvest sweep appends claims underneath us, and OFFSET would silently skip
    or repeat colleges as rows land. Combined with the fingerprint guard on the
    upsert, that makes the job resumable by simply running it again — already
    settled groups cost one comparison and no write.
    """
    be = get_backend()
    kind_filter, kind_params = "", []
    if kinds:
        kinds = list(kinds)
        kind_filter = f" AND kind IN ({','.join('?' * len(kinds))})"
        kind_params = kinds

    stats = {"colleges": 0, "groups": 0, "written": 0, "unchanged": 0,
             "single": 0, "corroborated": 0, "conflict": 0}
    cursor = ""
    page = 0
    while True:
        ids = [r["college_id"] for r in be.q(
            f"SELECT DISTINCT college_id FROM college_source_fact "
            f"WHERE college_id > ?{kind_filter} ORDER BY college_id LIMIT ?",
            [cursor, *kind_params, batch])]
        if not ids:
            break
        cursor = ids[-1]
        page += 1

        where = f"college_id IN ({','.join('?' * len(ids))}){kind_filter}"
        groups = _claims_for(where, [*ids, *kind_params])
        now = _now()
        rows = []
        for claims in groups.values():
            res = resolve(claims, now)
            if res is None:
                continue
            stats["groups"] += 1
            stats[res.agreement] += 1
            rows.append(_resolved_row(res, now))
        stats["colleges"] += len(ids)
        if rows:
            written = be.executemany(_RESOLVED_UPSERT, rows)
            stats["written"] += written
            stats["unchanged"] += len(rows) - written
        if progress_every and page % progress_every == 0:
            print(f"[reconcile] {stats['colleges']:,} colleges, "
                  f"{stats['groups']:,} groups, {stats['written']:,} written",
                  file=sys.stderr)
    return stats


def resolved_facts(ids, kinds=None) -> dict[str, dict[str, dict]]:
    """{college_id: {kind: fact}} for the card renderer, in one round trip.

    The per-kind dict is a superset of what render_card already reads off
    `college_web_facts` (summary / confidence / fetched_at), plus the provenance
    it currently cannot show: which source won, whether anything corroborated
    it, and a ready-made caveat sentence when two sources disagree.
    """
    ids = [i for i in (ids or []) if i]
    if not ids:
        return {}
    where = f"college_id IN ({','.join('?' * len(ids))})"
    params = list(ids)
    if kinds:
        kinds = list(kinds)
        where += f" AND kind IN ({','.join('?' * len(kinds))})"
        params += kinds
    out: dict[str, dict[str, dict]] = {}
    for r in get_backend().q(
        f"""SELECT college_id, kind, summary, payload, source_key, source_tier,
                   source_url, fetched_at, confidence, agreement, n_sources,
                   n_agreeing, rejected, rule
              FROM college_fact_resolved WHERE {where}""", params):
        rejected = r["rejected"] or []
        conflicts = [x for x in rejected if x.get("verdict") == "conflicts"]
        fact = {
            "summary": r["summary"], "payload": r["payload"],
            "confidence": r["confidence"], "source_key": r["source_key"],
            "source_tier": r["source_tier"], "source_url": r["source_url"],
            "fetched_at": r["fetched_at"], "agreement": r["agreement"],
            "n_sources": r["n_sources"], "n_agreeing": r["n_agreeing"],
            "rule": r["rule"], "conflicts": conflicts,
        }
        if conflicts:
            other = conflicts[0]
            fact["caveat"] = (
                f"Sources disagree. {r['source_key']} states "
                f"\"{(r['summary'] or '')[:120]}\"; {other.get('source_key')} states "
                f"\"{(other.get('value') or '')[:120]}\". Confirm on the official "
                f"site before relying on it.")
        out.setdefault(r["college_id"], {})[r["kind"]] = fact
    return out


# ---------------------------------------------------------------------------
# Bringing the existing sources in as claims
#
# Neither of these edits the tables it reads. `college_web_facts` and the
# catalogue stay exactly as their owners write them; this layer only mirrors
# them in as competing claims, which is what house rule 4 asks for — a scraped
# value never overwrites a catalogue value, the two sit side by side with
# provenance and are reconciled.
# ---------------------------------------------------------------------------

def _host_of(url: str | None) -> str:
    if not url:
        return "unknown"
    host = re.sub(r"^[a-z]+://", "", url.strip().lower()).split("/")[0]
    return host.split(":")[0] or "unknown"


def ingest_web_facts(limit: int | None = None, batch: int = 2000) -> int:
    """Mirror harvested website facts in as tier-1 claims."""
    be = get_backend()
    sql = ("SELECT college_id, kind, summary, payload, source_url, fetched_at, "
           "confidence FROM college_web_facts ORDER BY college_id, kind")
    params: list = []
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    rows = be.q(sql, params)
    n = 0
    for i in range(0, len(rows), batch):
        n += record_claims([{
            "college_id": r["college_id"], "kind": r["kind"],
            "source_key": f"official:{_host_of(r['source_url'])}",
            "source_tier": TIER_OFFICIAL, "summary": r["summary"],
            "payload": r["payload"], "source_url": r["source_url"],
            "fetched_at": r["fetched_at"], "confidence": r["confidence"],
        } for r in rows[i:i + batch]])
    return n


# Catalogue columns worth claiming. Only fields the catalogue actually holds
# for a meaningful share of rows — NAAC and NIRF are ~100% empty there, so
# claiming them would just create 38,700 empty claims to reconcile.
_CATALOGUE_FIELDS = (
    ("name", "name"),
    ("affiliation", "affiliated_university"),
    ("type", "type"),
)


def ingest_catalogue(limit: int | None = None, batch: int = 2000) -> int:
    """Mirror catalogue values in as tier-3 claims.

    The catalogue is what MIVI has always answered from, so it has to be
    REPRESENTED in the claim table rather than assumed: without a tier-3 claim
    for a field, a single scraped tier-1 value looks uncontested when in fact it
    contradicts what the site is showing on the college's own MME page.
    """
    be = get_backend()
    cols = ", ".join(c for _, c in _CATALOGUE_FIELDS)
    sql = (f"SELECT college_id, source_updated_at, {cols} FROM colleges "
           f"ORDER BY college_id")
    params: list = []
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    rows = be.q(sql, params)
    claims = []
    for r in rows:
        when = r["source_updated_at"] or _now()
        for kind, col in _CATALOGUE_FIELDS:
            val = r[col]
            if val is None or not str(val).strip():
                continue
            claims.append({
                "college_id": r["college_id"], "kind": kind,
                "source_key": "mme:catalogue", "source_tier": TIER_CATALOGUE,
                "summary": str(val).strip(), "payload": {col: val},
                "source_url": f"https://makemyeducation.com/college/{r['college_id']}",
                "fetched_at": when, "confidence": 0.72,
            })
    n = 0
    for i in range(0, len(claims), batch):
        n += record_claims(claims[i:i + batch])
    return n


# ---------------------------------------------------------------------------
# Developer tools
# ---------------------------------------------------------------------------

def _selftest() -> int:  # pragma: no cover - developer tool
    """Pure-function checks. No database, no network."""
    fails = 0

    def check(label, got, want):
        nonlocal fails
        ok = got == want
        if not ok:
            fails += 1
        print(f"  {'ok  ' if ok else 'FAIL'} {label}: {got!r}")

    print("money_signature — the three formats that must be one number")
    for text in ("Rs 1,20,000", "120000", "1.2 Lakh", "INR 1,20,000/-", "Rs 1.2 L"):
        check(text, sorted(money_signature(text).amounts), [120000.0])
    check("12.5 lakhs", sorted(money_signature("12.5 lakhs").amounts), [1250000.0])
    check("Session 2025-26 (no amount)", sorted(money_signature("Session 2025-26").amounts), [])
    check("phone 9876543210", sorted(money_signature("call 9876543210").amounts), [])
    check("20 hostels (bare, too small)", sorted(money_signature("20 hostels").amounts), [])

    now = datetime(2026, 7, 27, tzinfo=timezone.utc)

    def c(key, tier, summary, days=1, conf=None, kind="fees", payload=None):
        return Claim("X", kind, key, tier, summary=summary, payload=payload,
                     fetched_at=now - timedelta(days=days), confidence=conf)

    print("agreement across formats")
    check("Rs 1,20,000 vs 1.2 Lakh",
          agrees(c("official:a.ac.in", 1, "Fee Rs 1,20,000"),
                 c("gov:nirf", 2, "Fee 1.2 Lakh"))[0], True)
    check("60,000/semester vs 1,20,000/year (never annualise)",
          agrees(c("official:a.ac.in", 1, "Rs 60,000 per semester"),
                 c("gov:nirf", 2, "Rs 1,20,000 per year"))[0], False)
    check("A++ vs A+ (short code, exact only)",
          agrees(c("gov:naac", 2, "A++", kind="naac_grade"),
                 c("official:a.ac.in", 1, "A+", kind="naac_grade"))[0], False)
    check("labelled: tuition matches, hostel extra",
          agrees(c("official:a.ac.in", 1, "fees", payload={"Tuition Fee": "Rs 1,20,000"}),
                 c("gov:aicte", 2, "fees", payload={"Tuition": "1.2 lakh",
                                                    "Hostel Fee": "Rs 60,000"}))[0], True)

    print("resolution")
    r = resolve([c("official:a.ac.in", 1, "Rs 1,50,000 per year"),
                 c("mme:catalogue", 3, "Rs 1,20,000 per year")], now)
    check("tier-1 wins", r.winner.source_key, "official:a.ac.in")
    check("conflict recorded", r.agreement, "conflict")

    r = resolve([c("official:a.ac.in", 1, "Rs 1,20,000 per year", conf=0.80),
                 c("gov:aishe", 2, "1.2 Lakh")], now)
    check("corroborated", r.agreement, "corroborated")
    check("confidence raised above 0.80", round(r.confidence, 3) > 0.80, True)

    r = resolve([c("official:a.ac.in", 1, "Rs 1,10,000 per year", days=700),
                 c("gov:aishe", 2, "Rs 1,45,000 per year", days=5)], now)
    check("stale tier-1 loses to fresh tier-2", r.winner.source_key, "gov:aishe")
    check("rule names the demotion", "stale-demotion" in r.rule, True)

    r = resolve([c("official:a.ac.in", 1, "Established 1847", kind="name",
                   days=3000)], now)
    check("stable field ignores age", r.winner.source_key, "official:a.ac.in")

    r = resolve([c("official:iitr.ac.in", 1, "Rs 1,20,000"),
                 c("official:admissions.iitr.ac.in", 1, "Rs 1,20,000")], now)
    check("same origin does not corroborate", r.n_agreeing, 1)
    check("...and stays 'single'", r.agreement, "single")

    a = resolve([c("official:a.ac.in", 1, "Rs 1,20,000"), c("gov:x.gov.in", 2, "1.2 lakh")], now)
    b = resolve([c("gov:x.gov.in", 2, "1.2 lakh"), c("official:a.ac.in", 1, "Rs 1,20,000")], now)
    check("order-independent fingerprint (idempotency)", a.fingerprint, b.fingerprint)

    print(f"\n{'all passed' if not fails else str(fails) + ' FAILED'}")
    return 1 if fails else 0


# Synthetic claims for the demo. Kept in one table so --demo-clean can delete
# exactly what --demo wrote and nothing else.
_DEMO_SOURCES = ("official:demo-fees.example.ac.in", "gov:demo-aishe",
                 "gov:demo-naac", "official:demo-naac.example.ac.in",
                 "mme:demo-catalogue", "official:demo-stale.example.ac.in",
                 "gov:demo-fresh-registry", "official:sub.demo-naac.example.ac.in")


def _demo_clean() -> int:  # pragma: no cover - developer tool
    be = get_backend()
    qs = ",".join("?" * len(_DEMO_SOURCES))
    n = be.execute(f"DELETE FROM college_source_fact WHERE source_key IN ({qs})",
                   list(_DEMO_SOURCES))
    be.execute(f"DELETE FROM college_fact_resolved WHERE source_key IN ({qs})",
               list(_DEMO_SOURCES))
    return n


def _demo() -> None:  # pragma: no cover - developer tool
    """Load competing claims on real colleges, reconcile, print the audit.

    Deliberately uses real college_ids (the FK requires it) and obviously fake
    `demo-` source hosts, so nothing here can be mistaken for harvested data
    and --demo-clean can remove it exactly.
    """
    be = get_backend()
    ids = [r["college_id"] for r in be.q(
        "SELECT college_id FROM colleges ORDER BY college_id LIMIT 3")]
    if len(ids) < 3:
        raise SystemExit("need at least 3 colleges in the corpus")
    a, b, cc = ids
    names = {r["college_id"]: r["name"] for r in be.q(
        f"SELECT college_id, name FROM colleges WHERE college_id IN (?,?,?)", ids)}

    _demo_clean()
    now = _now()
    claims = [
        # 1. TIER-1 WIN + CONFLICT. Fresh official fee page against the
        #    catalogue rollup; the amounts genuinely differ.
        {"college_id": a, "kind": "fees", "source_key": "official:demo-fees.example.ac.in",
         "summary": "B.Tech Tuition Fee: Rs 1,50,000 per year",
         "payload": {"Tuition Fee (per year)": "Rs 1,50,000"},
         "source_url": "https://demo-fees.example.ac.in/fee-structure",
         "fetched_at": now - timedelta(days=12), "confidence": 0.82},
        {"college_id": a, "kind": "fees", "source_key": "mme:demo-catalogue",
         "summary": "Tuition: Rs 1,20,000 per year",
         "payload": {"Tuition Fee (per year)": "120000"},
         "source_url": f"https://makemyeducation.com/college/{a}",
         "fetched_at": now - timedelta(days=40), "confidence": 0.72},
        {"college_id": a, "kind": "fees", "source_key": "gov:demo-aishe",
         "summary": "Fee (AISHE return): Rs 1.5 Lakh per year",
         "payload": {"Tuition": "1.5 Lakh"},
         "source_url": "https://demo-aishe.example.gov.in/institution",
         "fetched_at": now - timedelta(days=30), "confidence": 0.86},

        # 2. CORROBORATION. Registry and college agree on the NAAC grade in two
        #    different notations, and a second page on the SAME college host
        #    repeats it — which must NOT count as a third voice.
        {"college_id": b, "kind": "naac_grade", "source_key": "gov:demo-naac",
         "summary": "A++", "payload": {"Grade": "A++", "CGPA": "3.68"},
         "source_url": "https://demo-naac.example.gov.in/hei",
         "fetched_at": now - timedelta(days=60), "confidence": 0.90},
        {"college_id": b, "kind": "naac_grade",
         "source_key": "official:demo-naac.example.ac.in",
         "summary": "A++", "payload": {"NAAC Grade": "A++"},
         "source_url": "https://demo-naac.example.ac.in/accreditation",
         "fetched_at": now - timedelta(days=9), "confidence": 0.75},
        {"college_id": b, "kind": "naac_grade",
         "source_key": "official:sub.demo-naac.example.ac.in",
         "summary": "A++", "payload": {"NAAC Grade": "A++"},
         "source_url": "https://sub.demo-naac.example.ac.in/about",
         "fetched_at": now - timedelta(days=4), "confidence": 0.70},

        # 3. THE INTERESTING CASE. A tier-1 fee page nearly two years old
        #    against a government figure fetched last week.
        {"college_id": cc, "kind": "fees",
         "source_key": "official:demo-stale.example.ac.in",
         "summary": "Tuition Fee: Rs 1,10,000 per year (AY 2024-25)",
         "payload": {"Tuition Fee (per year)": "Rs 1,10,000"},
         "source_url": "https://demo-stale.example.ac.in/fees",
         "fetched_at": now - timedelta(days=700), "confidence": 0.84},
        {"college_id": cc, "kind": "fees", "source_key": "gov:demo-fresh-registry",
         "summary": "Approved tuition fee: Rs 1,45,000 per annum",
         "payload": {"Tuition Fee (per year)": "Rs 1,45,000"},
         "source_url": "https://demo-fresh-registry.example.gov.in/fee-order",
         "fetched_at": now - timedelta(days=6), "confidence": 0.88},
    ]
    print(f"loading {len(claims)} competing claims on 3 real colleges")
    record_claims(claims)

    for cid, label in ((a, "1. TIER-1 WIN with a recorded CONFLICT"),
                       (b, "2. CORROBORATION raising confidence"),
                       (cc, "3. STALE tier-1 vs FRESH tier-2")):
        print(f"\n{'=' * 78}\n{label}\n{cid}  {names.get(cid, '?')}\n{'=' * 78}")
        for kind, res in reconcile_college(cid).items():
            if res.winner.source_key not in _DEMO_SOURCES:
                continue
            print(f"kind        : {kind}")
            print(f"winner      : {res.winner.source_key} "
                  f"(tier {res.winner.source_tier})")
            print(f"value       : {res.winner.summary}")
            print(f"rule        : {res.rule}")
            print(f"agreement   : {res.agreement}  "
                  f"({res.n_agreeing}/{res.n_sources} sources agree)")
            print(f"confidence  : {res.confidence:.3f}  "
                  f"(winner alone: {res.winner.prior():.2f})")
            for r in res.rejected:
                extra = f" age {r['age_days']}d" if "age_days" in r else ""
                print(f"  {r['verdict']:9} {r['source_key']:38} "
                      f"tier {r['source_tier']}->{r['effective_tier']}"
                      f"{extra}  [{r['reason']}/{r['basis']}]")
                print(f"            value: {r['value'][:90]}")
    print(f"\n{'=' * 78}\nresolved_facts() as the card renderer sees it\n{'=' * 78}")
    facts = resolved_facts([a, cc])
    for cid in (a, cc):
        f = (facts.get(cid) or {}).get("fees")
        if not f:
            continue
        print(f"{cid} fees  conf {f['confidence']:.3f}  {f['agreement']}  "
              f"via {f['source_key']}")
        if f.get("caveat"):
            print(f"  caveat: {f['caveat']}")


def _status() -> None:  # pragma: no cover - developer tool
    be = get_backend()
    print("claims by source class / tier")
    for r in be.q("SELECT split_part(source_key, ':', 1) AS cls, source_tier, "
                  "COUNT(*) AS n FROM college_source_fact "
                  "GROUP BY 1, 2 ORDER BY n DESC"):
        print(f"  {r['cls']:12} tier {r['source_tier']}  {r['n']:>8,}")
    print("resolved by agreement")
    for r in be.q("SELECT agreement, COUNT(*) AS n, "
                  "ROUND(AVG(confidence)::numeric, 3) AS c "
                  "FROM college_fact_resolved GROUP BY 1 ORDER BY n DESC"):
        print(f"  {r['agreement']:14} {r['n']:>8,}  avg confidence {r['c']}")
    print("conflicts by kind")
    for r in be.q("SELECT kind, COUNT(*) AS n FROM college_fact_resolved "
                  "WHERE agreement = 'conflict' GROUP BY 1 ORDER BY n DESC LIMIT 15"):
        print(f"  {r['kind']:14} {r['n']:>8,}")


def main() -> None:  # pragma: no cover - developer tool
    ap = argparse.ArgumentParser(description="MIVI multi-source reconciliation")
    ap.add_argument("--init", action="store_true", help="create the two tables")
    ap.add_argument("--selftest", action="store_true", help="pure-function checks, no DB")
    ap.add_argument("--demo", action="store_true", help="load competing claims and show the audit")
    ap.add_argument("--demo-clean", action="store_true", help="delete the demo claims")
    ap.add_argument("--ingest-web", action="store_true", help="mirror college_web_facts as tier-1 claims")
    ap.add_argument("--ingest-catalogue", action="store_true", help="mirror catalogue values as tier-3 claims")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--college", help="reconcile one college and print it")
    ap.add_argument("--all", action="store_true", help="reconcile every claim group")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(_selftest())
    if args.init:
        ensure_schema()
        print("[reconcile] college_source_fact + college_fact_resolved ready")
    if args.demo_clean:
        print(f"[reconcile] removed {_demo_clean()} demo claims")
    if args.ingest_web:
        print(f"[reconcile] web facts claimed: {ingest_web_facts(args.limit):,}")
    if args.ingest_catalogue:
        print(f"[reconcile] catalogue values claimed: {ingest_catalogue(args.limit):,}")
    if args.demo:
        _demo()
    if args.college:
        for kind, res in sorted(reconcile_college(args.college).items()):
            print(f"{kind:14} {res.agreement:13} conf {res.confidence:.3f} "
                  f"via {res.winner.source_key}  [{res.rule}]")
    if args.all:
        print(json.dumps(reconcile_all(), indent=2))
    if args.status:
        _status()


if __name__ == "__main__":
    main()
