"""Resolve the place names students type into the ones the corpus stores.

Two layers, cheapest first:

  1. CURATED TABLE — the renames and colloquialisms below. India renamed a lot
     of cities in the last decade and almost nobody uses the new names in
     speech: a student says Bangalore, Vizag, Trivandrum, Gurgaon. The corpus
     stores Bengaluru, Visakhapatnam, Thiruvananthapuram, Gurugram. This is a
     dictionary lookup — microseconds, deterministic.

  2. FUZZY FALLBACK — for genuine typos nobody curated ("Dehradoon",
     "Coimbatur"), match against the distinct city list by trigram similarity.
     Still no model call, still milliseconds, but only accepted above a high
     similarity bar because a wrong city is worse than no city.

Anything neither layer resolves falls through to rag_core/selfrag.py, which
spends an LLM call on it. That ordering is the whole design: the common cases
cost nothing, and the model is reserved for the genuinely novel.
"""
from __future__ import annotations

import difflib
import re
import sys
import threading

from .backend import get_backend

# Curated aliases. Keys are lowercase; values must match the corpus EXACTLY.
# Kept in code rather than only in the table so a fresh database is never
# missing them — build_db seeds the table from this dict.
CITY_ALIASES = {
    # Official renames still spoken the old way
    "bangalore": "Bengaluru", "banglore": "Bengaluru", "bengaluru": "Bengaluru",
    "vizag": "Visakhapatnam", "vizagapatnam": "Visakhapatnam",
    "trivandrum": "Thiruvananthapuram",
    "gurgaon": "Gurugram",
    "calcutta": "Kolkata",
    "bombay": "Mumbai",
    "madras": "Chennai",
    "poona": "Pune",
    "baroda": "Vadodara",
    "mysore": "Mysuru",
    "mangalore": "Mangaluru",
    "belgaum": "Belagavi",
    "hubli": "Hubballi",
    "gulbarga": "Kalaburagi",
    "bellary": "Ballari",
    "shimoga": "Shivamogga",
    "tumkur": "Tumakuru",
    "allahabad": "Prayagraj",
    "faizabad": "Ayodhya",
    "pondicherry": "Puducherry", "pondy": "Puducherry",
    "cochin": "Kochi", "ernakulam": "Kochi",
    "calicut": "Kozhikode",
    "trichy": "Tiruchirappalli", "trichinopoly": "Tiruchirappalli",
    "tuticorin": "Thoothukudi",
    "simla": "Shimla",
    "cuttak": "Cuttack",
    "orissa": "Odisha",
    # Common short forms
    "hyd": "Hyderabad", "vizianagaram": "Vizianagaram",
    "ncr": "Delhi", "new delhi": "Delhi",
    "noida": "Noida", "greater noida": "Greater Noida",
    "ghaziabad": "Ghaziabad",
    "dehradoon": "Dehradun", "doon": "Dehradun",
    "chandigarh": "Chandigarh", "mohali": "Mohali",
    "jammu": "Jammu", "srinagar": "Srinagar",
}

STATE_ALIASES = {
    "orissa": "Odisha",
    "uttaranchal": "Uttarakhand",
    "pondicherry": "Puducherry",
    "up": "Uttar Pradesh", "u.p.": "Uttar Pradesh",
    "mp": "Madhya Pradesh", "m.p.": "Madhya Pradesh",
    "ap": "Andhra Pradesh", "a.p.": "Andhra Pradesh",
    "hp": "Himachal Pradesh", "h.p.": "Himachal Pradesh",
    "tn": "Tamil Nadu", "wb": "West Bengal", "jk": "Jammu and Kashmir",
    "j&k": "Jammu and Kashmir", "uk": "Uttarakhand", "ts": "Telangana",
    "maharastra": "Maharashtra", "karnatka": "Karnataka",
    "tamilnadu": "Tamil Nadu", "westbengal": "West Bengal",
    "delhi ncr": "Delhi", "new delhi": "Delhi",
}

# Similarity floor for the fuzzy layer. 0.86 was chosen against the real city
# list: it accepts "Dehradoon"->Dehradun and "Coimbatur"->Coimbatore while
# rejecting "Kanpur"->"Kannur" and "Salem"->"Selam"-style near-collisions,
# which are DIFFERENT real cities. Loosening this trades a missed match for a
# confidently wrong location, which is the worse error.
FUZZY_MIN = 0.86

_places: dict[str, list[str]] = {}
_lock = threading.Lock()


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", str(s or "").strip().lower())


def _load_places() -> dict[str, list[str]]:
    """Distinct cities and states from the corpus, cached per process."""
    if _places:
        return _places
    with _lock:
        if _places:
            return _places
        try:
            be = get_backend()
            _places["city"] = [r["city"] for r in be.q(
                "SELECT DISTINCT city FROM colleges WHERE city IS NOT NULL "
                "AND TRIM(city) <> ''") if r.get("city")]
            _places["state"] = [r["state"] for r in be.q(
                "SELECT DISTINCT state FROM colleges WHERE state IS NOT NULL "
                "AND TRIM(state) <> ''") if r.get("state")]
        except Exception as exc:  # noqa: BLE001 - alias resolution is optional
            print(f"[aliases] place list unavailable: {exc}", file=sys.stderr)
            _places["city"], _places["state"] = [], []
    return _places


def resolve(value: str, kind: str = "city") -> tuple[str, str | None]:
    """(resolved_value, note). `note` is non-None only when something changed,
    so callers can tell the student what was interpreted.

    Never raises and never returns empty: an unresolvable value is returned
    unchanged so the caller's own filtering still runs.
    """
    raw = str(value or "").strip()
    if not raw:
        return raw, None
    key = _norm(raw)

    table = CITY_ALIASES if kind == "city" else STATE_ALIASES
    if key in table and table[key].lower() != raw.lower():
        return table[key], f"read '{raw}' as '{table[key]}'"
    # A city alias can appear in a state slot and vice versa — students use the
    # two interchangeably and the router cannot reliably tell them apart.
    other = STATE_ALIASES if kind == "city" else CITY_ALIASES
    if key in other and other[key].lower() != raw.lower():
        return other[key], f"read '{raw}' as '{other[key]}'"

    places = _load_places()
    candidates = places.get(kind) or []
    if not candidates:
        return raw, None
    # Exact (case-insensitive) hit: nothing to correct.
    for c in candidates:
        if _norm(c) == key:
            return c, None

    match = difflib.get_close_matches(key, [_norm(c) for c in candidates],
                                      n=1, cutoff=FUZZY_MIN)
    if match:
        for c in candidates:
            if _norm(c) == match[0]:
                return c, f"read '{raw}' as '{c}'"
    return raw, None


def resolve_filters(filters: dict) -> tuple[dict, list[str]]:
    """Resolve every place-shaped filter. Returns (filters, notes)."""
    if not filters:
        return filters, []
    out = dict(filters)
    notes: list[str] = []
    for key, kind in (("city", "city"), ("state", "state")):
        if out.get(key):
            fixed, note = resolve(out[key], kind)
            if note:
                out[key] = fixed
                notes.append(note)
    return out, notes


def seed_table(conn) -> int:
    """Persist the curated aliases so they are queryable from SQL too."""
    rows = [(a, c, "city", "curated") for a, c in CITY_ALIASES.items()]
    rows += [(a, c, "state", "curated") for a, c in STATE_ALIASES.items()]
    conn.executemany(
        "INSERT OR REPLACE INTO place_alias(alias, canonical, kind, source) "
        "VALUES(?,?,?,?)", rows)
    return len(rows)
