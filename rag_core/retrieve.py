"""Hybrid retrieval over the production corpus: prefilter in SQL, then rank.

The shape of this module is dictated by scale. With 35k colleges / 211k courses
there is no "score everything and sort" step and no "put the corpus in the
prompt" step; both were viable against the old 15-row prototype and are simply
wrong here. Retrieval runs in three stages:

  1. HARD PREFILTER (SQL). Every structured constraint a student states —
     state, city, budget, course, level, hostel, entrance exam — becomes a
     WHERE clause over `colleges` + `college_agg`. This is the only place a
     constraint is enforced. The LLM never filters and never counts: a budget
     answer that is wrong is worse than no answer, and an LLM asked to respect
     "under 2 lakh" over a truncated context will eventually not.

  2. TWO RANKERS over the surviving candidate set. Dense cosine (semantic:
     "affordable engineering college near Chandigarh") and FTS5 BM25 (lexical:
     "SRM Sonipat", "GNM"). Neither alone is acceptable — embeddings lose exact
     institution names, BM25 loses paraphrase. Measured warm against the real
     38,700-college store: a filtered search returns in ~66 ms end to end
     (BM25 + fusion + card hydration), and index_build benchmarks the dense
     matvec it adds at ~2 ms. That is noise next to the LLM call it feeds.

  3. RRF FUSION into the final top-k, hydrated into cards by db.render_card.

Ordering matters and is not interchangeable. Prefilter-then-rank is correct
(the k returned are all admissible, so k is a relevance dial, not a survival
lottery); rank-then-filter is not, because a hard constraint applied after
truncation silently returns fewer than k — or zero — results whenever the
constraint is selective, which is precisely when the student cares most.
"Cheapest college in Ladakh" has ~10 admissible answers in 35k; ranking first
would never surface them. It is also cheaper: the SQL indexes reduce the
candidate set before any float math happens.
"""
from __future__ import annotations

import os
import re
import sys
import threading

import numpy as np

from .store.backend import get_backend
from .store.db import fetch_colleges, get_conn, get_meta, render_card

# Reciprocal Rank Fusion constant. 60 is the published default and behaves
# sanely here: 1/(60+rank) is nearly flat across the first few ranks, so a
# college that both rankers like beats one that either ranker loves.
RRF_K = 60

# How deep each ranker goes before fusion. RRF only needs a list long enough
# that a document appearing in *one* ranker still lands in the fused head;
# beyond ~8x top_k the extra ranks contribute < 1/(60+96) and cost latency.
_POOL_MULT = 8
_POOL_MIN = 96

# Ceiling on candidate ids pulled into Python. The whole corpus is 38.7k, so
# this is a blow-up guard for a future corpus, not a functional limit.
_MAX_CANDIDATES = 100_000

# Context budget. ~12k chars is roughly 3k tokens of grounding — enough for a
# dozen cards, small enough that the generator still attends to the question.
MAX_CONTEXT_CHARS = 12_000

# Floor below which an annual tuition is treated as a data-entry placeholder
# rather than a price, for SUPERLATIVE RANKING ONLY (see superlatives()).
# Rs 1,000/year is deliberately conservative: genuinely cheap government
# programmes in this corpus sit at Rs 2,000-3,000 (Kumaun University, Rs 2,160)
# and are correctly kept, while the placeholder cluster sits at Rs 10-500.
MIN_PLAUSIBLE_TUITION_INR = int(os.getenv("MIN_PLAUSIBLE_TUITION_INR", "1000"))

_CARD_SEP = "\n\n---\n\n"


# ---------------------------------------------------------------------------
# Filters -> SQL
# ---------------------------------------------------------------------------

_KNOWN_FILTERS = {
    "state", "city", "college_type", "max_tuition_inr", "min_tuition_inr",
    "course_terms", "program_level", "hostel_required", "entrance_exam",
    "include_abroad",
}
_warned_keys: set[str] = set()

# The router/LLM says "govt", the corpus says "Government". Normalising here
# keeps the mapping in one place instead of in every prompt.
_TYPE_SYNONYMS = {
    "govt": "government", "gov": "government", "sarkari": "government",
    "pvt": "private", "self-financed": "private", "self financed": "private",
    "deemed university": "deemed", "deemed-to-be-university": "deemed",
}

# program_level values are derived by sources/courses_csv.program_level, so the
# vocabulary is closed: Undergraduate / Postgraduate / Doctorate /
# Diploma/Certificate / Other. Everything a student might say maps into it.
_LEVEL_SYNONYMS = {
    "ug": "undergraduate", "under graduate": "undergraduate",
    "undergrad": "undergraduate", "bachelor": "undergraduate",
    "bachelors": "undergraduate", "graduation": "undergraduate",
    "pg": "postgraduate", "post graduate": "postgraduate",
    "postgrad": "postgraduate", "master": "postgraduate",
    "masters": "postgraduate",
    "phd": "doctorate", "doctoral": "doctorate", "research": "doctorate",
    "diploma": "diploma/certificate", "certificate": "diploma/certificate",
    "certification": "diploma/certificate",
}


def _as_int(value, key: str) -> int:
    """Ints stay ints. A budget that arrives as "2 lakh" is a bug upstream and
    must surface as one — coercing it to 2 would produce a confidently wrong
    answer, which is the single failure mode this system cannot have."""
    if isinstance(value, bool):
        raise ValueError(f"filter {key!r} must be a number, got a bool")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != int(value):
            raise ValueError(f"filter {key!r} must be a whole rupee amount, got {value!r}")
        return int(value)
    s = str(value).strip().replace(",", "").replace("_", "")
    if not re.fullmatch(r"\d+", s):
        raise ValueError(
            f"filter {key!r} must be an integer number of rupees, got {value!r} "
            f"— convert units (lakh/thousand) before calling retrieval"
        )
    return int(s)


def _normalise_filters(filters: dict | None) -> dict:
    f = dict(filters or {})
    for k in list(f):
        if k not in _KNOWN_FILTERS:
            # Unknown keys come from an over-eager extractor prompt, not from
            # corrupt data — drop them loudly once, never fail a student's query.
            if k not in _warned_keys:
                _warned_keys.add(k)
                print(f"[retrieve] ignoring unknown filter key {k!r}", file=sys.stderr)
            f.pop(k)
        elif f[k] is None or f[k] == "" or f[k] == []:
            f.pop(k)

    for k in ("max_tuition_inr", "min_tuition_inr"):
        if k in f:
            f[k] = _as_int(f[k], k)

    terms = f.get("course_terms")
    if terms is not None:
        if isinstance(terms, str):
            terms = [terms]
        terms = [str(t).strip() for t in terms if str(t).strip()]
        # More than a handful of ORed LIKEs on 211k rows stops being a filter
        # and starts being a scan; the extractor never legitimately emits more.
        f["course_terms"] = terms[:6] if terms else None
        if not f["course_terms"]:
            f.pop("course_terms")

    if "program_level" in f:
        lv = str(f["program_level"]).strip().lower()
        f["program_level"] = _LEVEL_SYNONYMS.get(lv, lv)
    if "college_type" in f:
        ct = str(f["college_type"]).strip().lower()
        f["college_type"] = _TYPE_SYNONYMS.get(ct, ct)

    # Place names LAST, and here rather than at a call site, so every entry
    # point gets it: search, count_matching and superlatives must agree on what
    # "Vizag" means or a count will contradict the list beneath it. Measured on
    # the real corpus: "Vizag" matched 0 colleges and "Visakhapatnam" 348, so
    # without this a student using the name everyone actually says is told
    # there is nothing there. Microseconds, cached, and deterministic — the
    # LLM critic in selfrag.py only handles what this cannot.
    try:
        from .store.aliases import resolve_filters

        f, _notes = resolve_filters(f)
    except Exception as exc:  # noqa: BLE001 - never fail a query over an alias
        print(f"[retrieve] alias resolution skipped: {exc}", file=sys.stderr)
    return f


def _like(value: str) -> str:
    """A contains-pattern with LIKE metacharacters neutralised. SQLite's LIKE
    is already case-insensitive for ASCII, which is why no LOWER() wrapper is
    needed (and the state/city indexes stay usable for prefix-ish patterns)."""
    s = str(value).strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{s}%"


_FROM = "FROM colleges c LEFT JOIN college_agg a ON a.college_id = c.college_id"


def _where(f: dict) -> tuple[str, list]:
    """Build the hard prefilter. Returns (sql, params) for use after `_FROM`.

    LEFT JOIN, not INNER: a college whose courses have not been rolled up yet
    (the enrichment crawl is still running) must still be findable by name and
    location. It only drops out when a filter genuinely needs an aggregate.
    """
    clauses: list[str] = []
    params: list = []

    # Abroad is excluded unless explicitly asked for. This is a correctness
    # gate, not a preference: 3,657 abroad rows carry fees in the institution's
    # own currency and tuition_*_inr is NULL for them by design, so letting one
    # into a rupee-budget result set produces either a missing fee or — far
    # worse — a number the reader assumes is rupees.
    if not f.get("include_abroad"):
        clauses.append("NOT c.is_abroad")

    # Students say "in Punjab" and "in Ludhiana" interchangeably, and the
    # extractor cannot reliably tell a city from a state (Chandigarh, Delhi and
    # Puducherry are both). Either key therefore matches either column;
    # district is included because the corpus often carries the town there.
    for key in ("state", "city"):
        if f.get(key):
            pat = _like(f[key])
            clauses.append(
                "(c.state LIKE ? ESCAPE '\\' OR c.city LIKE ? ESCAPE '\\' "
                "OR c.district LIKE ? ESCAPE '\\')"
            )
            params += [pat, pat, pat]

    if f.get("college_type"):
        clauses.append("c.type LIKE ? ESCAPE '\\'")
        params.append(_like(f["college_type"]))

    # Budget is tested against the college's CHEAPEST course: a college
    # qualifies if any single course fits. `IS NOT NULL` is deliberate — fee 0
    # was normalised to NULL ("not recorded", never free), and an unrecorded
    # fee must never be presented as fitting a budget.
    #
    # Note this is a college-level gate even when course_terms are set, i.e.
    # "MBA under 2 lakh" admits a college whose cheapest course is a 40k BA.
    # That is on purpose: tuition is unrecorded on a large share of course
    # rows, so pushing the gate into the EXISTS below would silently drop
    # colleges that genuinely offer the course. The per-college fee range then
    # appears in the card, so the answer layer can state the real number.
    if f.get("max_tuition_inr") is not None:
        clauses.append("a.min_tuition_inr IS NOT NULL AND a.min_tuition_inr <= ?")
        params.append(f["max_tuition_inr"])
    if f.get("min_tuition_inr") is not None:
        clauses.append("a.max_tuition_inr IS NOT NULL AND a.max_tuition_inr >= ?")
        params.append(f["min_tuition_inr"])

    # Course + level in ONE EXISTS so both must hold for the SAME course row.
    # Two separate subqueries would let a college pass "MBA" + "Postgraduate"
    # because it offers an MBA and, unrelatedly, an MSc.
    level, terms = f.get("program_level"), f.get("course_terms")
    if level or terms:
        sub = ["cr.college_id = c.college_id"]
        sub_params: list = []
        if level:
            sub.append("LOWER(cr.program_level) = ?")
            sub_params.append(level)
        if terms:
            ors = []
            for t in terms:
                ors.append("(cr.title LIKE ? ESCAPE '\\' OR cr.full_name LIKE ? ESCAPE '\\')")
                sub_params += [_like(t), _like(t)]
            sub.append("(" + " OR ".join(ors) + ")")
            # When a student names BOTH a course and a budget, the budget is
            # about THAT course. The college-level gate above only asks whether
            # anything at the college is affordable, which counted a college
            # whose cheapest course is a Rs 36,000 BA as an "MBA under 80k"
            # while its MBA actually costs Rs 115,999 — measured in Indore.
            #
            # Fees are unrecorded on ~42% of course rows though, so a strict
            # `fee <= budget` here would silently delete colleges that genuinely
            # offer the course. The rule is therefore "not provably over
            # budget": exclude a course only when its recorded fee exceeds the
            # limit, and keep unpriced ones (the card shows the real range, so
            # the answer layer can be honest about what is unknown).
            if f.get("max_tuition_inr") is not None:
                sub.append("(cr.tuition_min_inr IS NULL OR cr.tuition_min_inr <= ?)")
                sub_params.append(f["max_tuition_inr"])
        clauses.append(f"EXISTS (SELECT 1 FROM courses cr WHERE {' AND '.join(sub)})")
        params += sub_params

    if f.get("hostel_required"):
        clauses.append("a.has_hostel")

    if f.get("entrance_exam"):
        # college_agg.entrance_exams is a JSON array of strings; a LIKE over the
        # serialised array is an adequate containment test and needs no json1.
        clauses.append("a.entrance_exams LIKE ? ESCAPE '\\'")
        params.append(_like(f["entrance_exam"]))

    # TRUE, not 1. SQLite treated `WHERE 1` as truthy; Postgres rejects it with
    # "argument of WHERE must be type boolean". Every query with NO filters —
    # which is exactly the plain "tell me about <college>" lookup — was therefore
    # failing outright in production. The eval suite caught it the moment it was
    # pointed at the real corpus, and could not have caught it before, because it
    # was still reading the retired SQLite file.
    return (" AND ".join(clauses) if clauses else "TRUE"), params


# ---------------------------------------------------------------------------
# Lazy singletons. FastAPI serves from a threadpool, so first-request races are
# the normal case, not the exotic one: without the locks two threads load a
# 450 MB model and a 50 MB matrix twice.
# ---------------------------------------------------------------------------

_model = None
_index: dict | None = None
_model_lock = threading.Lock()
_index_lock = threading.Lock()
_db_lock = threading.Lock()
_db_checked = False
_index_failed: str | None = None   # remembered so we warn once, not per query


def _get_conn():
    """The store handle, with a one-time check that the ETL has actually run.

    Asks Postgres directly. This used to test `DB_FILE.exists()` and read
    `sqlite_master` — questions in SQLite's vocabulary answered by a
    compatibility shim, so a store that was DOWN produced "MIVI store not found
    at <path>. Build it first (python -m rag_core.store.build)": a wrong
    diagnosis pointing at a command that no longer exists.
    """
    global _db_checked
    if not _db_checked:
        with _db_lock:
            if not _db_checked:
                be = get_backend()
                try:
                    rows = be.q(
                        "SELECT table_name AS name FROM information_schema.tables "
                        "WHERE table_schema = current_schema()")
                except Exception as e:  # noqa: BLE001
                    raise RuntimeError(
                        f"MIVI cannot reach Postgres ({str(e)[:120]}). The store "
                        f"must be up before retrieval can serve anything."
                    ) from e
                tables = {r["name"] for r in rows}
                missing = {"colleges", "courses", "college_agg"} - tables
                if missing:
                    raise RuntimeError(
                        f"the store is missing table(s) {sorted(missing)} — the "
                        f"ETL has not finished. Retrieval cannot serve a partial store."
                    )
                _db_checked = True
    return get_conn()


def _embed_model_name() -> str:
    """The query encoder MUST be the model that produced the index — encoding a
    query with a different family than the passages yields plausible-looking
    garbage, not an error. Precedence therefore runs from what the build
    actually recorded down to config as a last resort."""
    idx = _index or {}
    if idx.get("model"):
        return str(idx["model"])
    try:
        recorded = get_meta(_get_conn(), "embed_model")
        if recorded:
            return str(recorded)
    except Exception:
        pass
    try:
        from .config import EMBED_MODEL
        return EMBED_MODEL
    except Exception:  # config is corpus-specific; retrieval must not die with it
        return os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer  # heavy: lazy

                name = _embed_model_name()
                # Same device probe the index build uses, so a query vector and
                # the indexed vectors are produced by the same configuration.
                # It falls back to CPU on any GPU trouble: a served query must
                # degrade in latency, never fail.
                from .index_build import pick_device
                device = pick_device()
                try:
                    # Cached model: skip the Hub online check, measured in
                    # seconds per process start.
                    _model = SentenceTransformer(name, local_files_only=True,
                                                 device=device)
                except Exception:
                    _model = SentenceTransformer(name, device=device)
    return _model


def _coerce_index(obj) -> dict:
    """Normalise whatever index_build.load_index() hands back into
    {ids, vectors, pos, model}. It is a sibling module built alongside this one,
    so the accepted shapes are broad — but an unrecognised one is a hard error,
    never a guess."""
    ids = vectors = model = None
    if isinstance(obj, dict) or hasattr(obj, "files"):        # dict / NpzFile
        if isinstance(obj, dict):
            def get(k):
                return obj.get(k)
        else:
            def get(k):
                return obj[k] if k in obj.files else None

        for key in ("ids", "college_ids", "keys"):
            if get(key) is not None:
                ids = get(key)
                break
        for key in ("vectors", "matrix", "embeddings", "vecs"):
            if get(key) is not None:
                vectors = get(key)
                break
        # Explicit None tests throughout: `a or b` on a numpy array raises
        # "truth value of an array is ambiguous", and every value here may be one.
        model = get("model")
        if model is None:
            model = get("embed_model")
        # index_build records the encoder inside meta; that is the authoritative
        # answer to "which model produced these vectors".
        meta = get("meta")
        if isinstance(meta, dict) and meta.get("embed_model"):
            model = meta["embed_model"]
    elif isinstance(obj, (tuple, list)) and len(obj) == 2:
        a, b = obj
        ids, vectors = (a, b) if np.asarray(b).ndim == 2 else (b, a)
    else:
        ids = getattr(obj, "ids", None)
        vectors = getattr(obj, "vectors", None)
        if vectors is None:
            vectors = getattr(obj, "matrix", None)
        model = getattr(obj, "model", None)

    if ids is None or vectors is None:
        raise RuntimeError(
            f"rag_core.index_build.load_index() returned {type(obj).__name__} with "
            f"no recognisable (ids, vectors) — retrieval needs both."
        )

    ids = [str(i) for i in np.asarray(ids).ravel().tolist()]
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(ids):
        raise RuntimeError(
            f"index is malformed: {len(ids)} ids vs vectors of shape {vectors.shape}"
        )

    # Cosine similarity is a dot product only on unit vectors. Check a sample
    # rather than every row (35k x 384 norms per process start is wasted work).
    sample = vectors[:: max(1, len(ids) // 64) or 1][:64]
    norms = np.linalg.norm(sample, axis=1)
    if norms.size and not np.allclose(norms, 1.0, atol=1e-3):
        print("[retrieve] index vectors are not L2-normalised — normalising in memory",
              file=sys.stderr)
        n = np.linalg.norm(vectors, axis=1, keepdims=True)
        n[n == 0] = 1.0
        vectors = vectors / n

    return {
        "ids": ids,
        "vectors": vectors,
        "pos": {cid: i for i, cid in enumerate(ids)},
        "model": str(model) if model is not None else None,
    }


def _get_index() -> dict:
    """The coerced index, rebuilt only when the underlying file changes.

    load_index() is called every time on purpose: it is internally cached on
    (path, mtime, size) and therefore returns the *same object* until someone
    rebuilds the index underneath a running server, at which point it returns a
    new one. Comparing identity is a stat() per query and buys live pickup of a
    rebuild; caching the coerced form here avoids re-deriving the 35k id->row
    map on that same query.
    """
    global _index
    try:
        from .index_build import load_index
    except ImportError as e:  # module absent — degrade, do not crash the server
        raise RuntimeError(
            f"rag_core.index_build is unavailable — the dense half of retrieval "
            f"cannot run ({e})."
        ) from e

    raw = load_index()
    cached = _index
    if cached is not None and cached.get("raw") is raw:
        return cached

    with _index_lock:
        if _index is not None and _index.get("raw") is raw:
            return _index
        idx = _coerce_index(raw)
        idx["raw"] = raw

        # A vector index built from a different corpus than the DB it is queried
        # against silently ranks the wrong colleges — no exception, just wrong
        # answers. Cheap guard: the ids must actually resolve in this store.
        sample = idx["ids"][:256]
        if sample:
            qs = ",".join("?" * len(sample))
            known = get_backend().one(
                f"SELECT COUNT(*) FROM colleges WHERE college_id IN ({qs})", sample
            ) or 0
            if known == 0:
                raise RuntimeError(
                    f"index at odds with {DB_FILE}: none of its first {len(sample)} "
                    f"ids exist in `colleges`. Rebuild the index against this store."
                )
            if known < len(sample) * 0.8:
                print(f"[retrieve] index/DB drift: only {known}/{len(sample)} sampled "
                      f"ids resolve — rebuild recommended", file=sys.stderr)
        _index = idx
    return _index


def encode_query(question: str):
    """Embed one query.

    e5 is asymmetric: index_build embedded every card as "passage: ...", so a
    query must carry "query: " or recall drops silently. The prefix is imported
    from the module that wrote the vectors rather than retyped here — one
    definition, so the two halves cannot drift.
    """
    model = _get_model()
    try:
        from .index_build import QUERY_PREFIX
    except ImportError:
        QUERY_PREFIX = "query: " if "e5" in _embed_model_name().lower() else ""
    return model.encode(
        [QUERY_PREFIX + str(question or "").strip()], normalize_embeddings=True
    )[0]


def warm() -> None:
    """Preload store, index and model at server start so the first student does
    not pay the cold-start seconds. Best effort: a failure here is logged, never
    fatal — the process must still come up and serve lexical-only."""
    try:
        _get_conn()
    except Exception as e:
        print(f"[retrieve] warm-up: store unavailable: {e}", file=sys.stderr)
        return
    for step, fn in (("index", _get_index), ("model", _get_model)):
        try:
            fn()
        except Exception as e:
            print(f"[retrieve] warm-up: {step} unavailable: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Rankers
# ---------------------------------------------------------------------------

# Tokens that match nearly every card carry no ranking signal and cost real
# time in a 35k-doc BM25 scan. BM25's IDF would discount them anyway; dropping
# them up front is purely a latency decision. Hinglish included because
# students type "punjab me sasta college" as often as English.
_STOP = {
    "a", "an", "the", "is", "are", "of", "in", "on", "at", "to", "for", "and",
    "or", "with", "which", "what", "who", "how", "can", "i", "me", "my", "you",
    "do", "does", "best", "good", "top", "please", "want", "need", "college",
    "colleges", "university", "universities", "list", "any", "there", "about",
    "kya", "konsa", "kaun", "kaunsa", "mein", "me", "hai", "ho", "ke", "ki",
    "ka", "se", "aur", "ya", "mujhe", "chahiye", "batao", "sabse", "acha",
    "accha", "achha", "kitna", "kitne",
}
_TOKEN_RE = re.compile(r"[0-9A-Za-zऀ-ॿ]+")


def _fts_query(question: str, terms: list[str] | None = None) -> str | None:
    """Build a *safe* FTS5 MATCH expression.

    FTS5 parses its query string: a stray quote, hyphen, `*`, `^`, `:` or a
    bare AND/OR/NOT/NEAR raises "fts5: syntax error near ..." and would 500 the
    request on a perfectly ordinary question. The defence is not escaping the
    user's punctuation but never forwarding it: tokenise to alphanumerics and
    re-emit each token as its own quoted phrase.

    Terms are ORed. FTS5's implicit operator is AND, which on a conversational
    question ("which colleges in Punjab offer MBA under 5 lakh") matches
    nothing at all; BM25 is what separates the good hits, not the connective.
    """
    seen: list[str] = []
    for raw in _TOKEN_RE.findall(str(question or "")):
        t = raw.lower()
        if len(t) < 2 or t in _STOP or t in seen:
            continue
        seen.append(t)
        if len(seen) >= 24:   # long pasted questions must not turn into a scan
            break

    parts = [f'"{t}"' for t in seen]
    for t in terms or []:
        # Multi-word course terms ("computer science") are worth a phrase match.
        toks = [x.lower() for x in _TOKEN_RE.findall(str(t))]
        if toks:
            phrase = '"' + " ".join(toks) + '"'
            if phrase not in parts:
                parts.append(phrase)
    return " OR ".join(parts) if parts else None


def _lexical(match: str, where: str, params: list, limit: int) -> list[str]:
    """BM25 over colleges_fts, restricted to the prefiltered candidates by
    joining the same WHERE — so the lexical ranker can never resurface a
    college the hard filters rejected.

    Column weights follow the fts5 column order (college_id, name, location,
    course_blob, about). Name is weighted hardest because an exact-institution
    query is the one case where lexical must beat dense outright; `about` is
    long marketing prose and is damped so it cannot drown a short precise card.
    """
    # The two engines disagree here more than anywhere else — FTS5 `MATCH` +
    # bm25() (ascending, negative) versus tsvector `@@` + ts_rank (descending) —
    # so the whole query lives behind the backend rather than being branched on
    # here. Both return ids already ordered best-first.
    try:
        return get_backend().lexical(match, where, params, limit)
    except Exception as e:
        # Losing the lexical half degrades ranking but never correctness, so a
        # missing FTS table or a tokenisation surprise still answers the query.
        print(f"[retrieve] lexical ranker skipped: {e}", file=sys.stderr)
        return []


def _dense(idx: dict, cand: list[str], qv, limit: int) -> list[str]:
    """Cosine against the index matrix, restricted to candidate ids."""
    pos = idx["pos"]
    rows = np.fromiter((pos[c] for c in cand if c in pos), dtype=np.int64)
    if rows.size == 0:
        return []

    mat = idx["vectors"]
    qv = np.asarray(qv, dtype=np.float32).ravel()
    if qv.shape[0] != mat.shape[1]:
        raise RuntimeError(
            f"query vector has dim {qv.shape[0]} but the index has {mat.shape[1]} "
            f"— the query encoder is not the model that built this index."
        )

    # Whole-matrix dot then select, unless the candidate set is small: fancy
    # indexing copies the submatrix, and at 35k x 384 that copy costs more than
    # the 13M-FLOP product it is trying to avoid. Below ~1/8 of the corpus the
    # copy wins.
    if rows.size * 8 < mat.shape[0]:
        sims = mat[rows] @ qv
    else:
        sims = (mat @ qv)[rows]

    k = min(limit, sims.shape[0])
    top = np.argpartition(-sims, k - 1)[:k] if k < sims.shape[0] else np.arange(sims.shape[0])
    top = top[np.argsort(-sims[top], kind="stable")]
    ids = idx["ids"]
    return [ids[rows[i]] for i in top]


def _rrf(*ranked_lists: list[str]) -> dict[str, float]:
    """Reciprocal Rank Fusion.

    Fusing by rank rather than by normalised score is the whole point. Cosine
    lives in [-1, 1] with a corpus-dependent floor (e5 similarities cluster
    around 0.75-0.9, so the *spread* that matters is tiny), while bm25() is an
    unbounded negative log-scale quantity whose magnitude depends on document
    length and term rarity in this particular corpus. Min-max normalising them
    per query makes the blend depend on the shape of each result list — one
    outlier rescales everything and the weighting silently changes from query
    to query. Ranks are the only quantity the two rankers agree on, and RRF's
    1/(k+rank) is monotone, bounded and parameter-light.
    """
    fused: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, cid in enumerate(lst, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)
    return fused


# ---------------------------------------------------------------------------
# Public contract
# ---------------------------------------------------------------------------

def count_matching(filters: dict | None = None) -> int:
    """Exact count of colleges satisfying the hard filters.

    "How many colleges offer MBA in Punjab" is answered by this COUNT and
    nothing else. The alternative — letting the model count cards in its
    context — cannot be right: the context holds top_k, not the population.

    Measured on the real store: 13-30 ms for indexed predicates (state, type,
    the courses EXISTS), ~200 ms for an unfiltered COUNT, and ~470 ms for
    entrance_exam, which is a LIKE over 38.7k serialised JSON arrays with no
    index to help it. If exam questions become common, that one filter is the
    thing to denormalise — not this function.
    """
    f = _normalise_filters(filters)
    where, params = _where(f)
    return int(get_backend().one(f"SELECT COUNT(*) {_FROM} WHERE {where}", params) or 0)


def superlatives(filters: dict | None = None, limit: int = 5) -> dict:
    """Cheapest / oldest / most courses within the filtered set, from SQL.

    Superlatives are a population property, so they are computed over every
    matching row rather than over the retrieved top_k — the cheapest college in
    Punjab is very unlikely to also be the most semantically similar one.
    """
    f = _normalise_filters(filters)
    where, params = _where(f)
    be = get_backend()
    limit = max(1, int(limit))
    cols = ("c.college_id, c.name, c.city, c.state, c.type, "
            "c.establish_year, a.n_courses, a.min_tuition_inr, a.max_tuition_inr")

    def run(extra: str, order: str, value_key: str) -> list[dict]:
        rows = be.q(
            f"SELECT {cols} {_FROM} WHERE {where} AND {extra} ORDER BY {order} LIMIT ?",
            [*params, limit],
        )
        out = []
        for r in rows:
            d = dict(r)
            d["value"] = d.get(value_key)
            out.append(d)
        return out

    return {
        # min_tuition_inr is already NULL-for-unrecorded (0 was never "free"),
        # but NOT NULL alone is not enough for a *ranking*: measured on this
        # corpus, 2,611 domestic course rows (2.2%) carry a placeholder tuition
        # under Rs 1,000 — a run of them sit at exactly Rs 10. Those are data
        # entry artifacts, and because "cheapest" sorts ascending they would
        # win every single time, making the one question students most want
        # answered the one MIVI answers worst.
        #
        # The floor applies ONLY here, to superlative ranking. Filtering and
        # search still see those colleges: if a student explicitly asks about
        # one, its recorded fee is shown as-is. Suppressing a real listing is a
        # worse failure than ranking it oddly; presenting Rs 10 as the cheapest
        # college in a state is worse than both.
        "cheapest": run(
            f"a.min_tuition_inr IS NOT NULL AND a.min_tuition_inr >= {MIN_PLAUSIBLE_TUITION_INR}",
            "a.min_tuition_inr ASC", "min_tuition_inr"),
        # Year sanity bounds: the corpus contains 0s and typos, and an
        # unguarded MIN() would crown a college founded in year 12 as India's
        # oldest. 1500 predates every real Indian institution.
        "oldest": run("c.establish_year IS NOT NULL AND c.establish_year BETWEEN 1500 AND 2100",
                      "c.establish_year ASC", "establish_year"),
        "most_courses": run("a.n_courses IS NOT NULL AND a.n_courses > 0",
                            "a.n_courses DESC", "n_courses"),
        # NIRF-ranked colleges, best first. This is the only real ranking
        # evidence in the system: the catalogue's own nirf_ranking column is
        # populated on 1 of 38,700 rows, so without this a "best college"
        # question has nothing to rank ON and the generator invents a
        # justification — observed live, an answer citing "NAAC grade and NIRF
        # ranking" for five colleges that have neither.
        "best_ranked": _nirf_ranked(where, params, limit, f),
    }


# NIRF publishes one table per discipline, so the same college holds several
# ranks. Which one is RELEVANT depends on what the student asked, and mixing
# them produces nonsense — "#1 Architecture" listed as an answer to a pharmacy
# question. Course words are the strongest available signal for that.
_NIRF_CATEGORY_HINTS = (
    ("Engineering", ("btech", "b.tech", "be", "mtech", "m.tech", "engineering",
                     "civil", "mechanical", "electrical", "electronics",
                     "computer science", "cse", "it")),
    ("Management", ("mba", "bba", "pgdm", "management", "business")),
    ("Pharmacy", ("bpharm", "b.pharm", "mpharm", "pharm", "pharmacy")),
    ("Medical", ("mbbs", "md", "ms", "medical", "medicine", "nursing", "bds")),
    ("Law", ("llb", "ll.b", "llm", "law", "legal")),
    ("Architecture", ("barch", "b.arch", "architecture", "planning")),
    ("Research", ("phd", "ph.d", "research", "doctorate")),
)


def _preferred_categories(f: dict) -> list[str]:
    terms = " ".join(str(t).lower() for t in (f.get("course_terms") or []))
    level = str(f.get("program_level") or "").lower()
    hay = f"{terms} {level}"
    hits = [cat for cat, keys in _NIRF_CATEGORY_HINTS if any(k in hay for k in keys)]
    # "Overall" and "University" are the discipline-agnostic tables, so they are
    # the right answer to a bare "best college" question and a sensible
    # fallback behind any discipline the student did name.
    return hits + ["Overall", "University", "College"]


def _nirf_ranked(where: str, params: list, limit: int,
                 f: dict | None = None) -> list[dict]:
    """Top NIRF-ranked colleges within the filtered set.

    Kept separate from run() because it needs a join the other superlatives do
    not, and because a missing registry_fact table (a store built before the
    registries pass) must degrade to "no ranking data" rather than break every
    superlative query alongside it.
    """
    try:
        rows = get_backend().q(
            f"""SELECT c.college_id, c.name, c.city, c.state, c.type,
                       c.establish_year, a.n_courses, a.min_tuition_inr,
                       a.max_tuition_inr,
                       (f.payload->>'rank')::int AS nirf_rank,
                       f.category AS nirf_category, f.year AS nirf_year
                FROM colleges c
                LEFT JOIN college_agg a ON a.college_id = c.college_id
                JOIN registry_fact f ON f.college_id = c.college_id
                WHERE {where} AND f.registry = 'nirf'
                  AND (f.payload->>'rank') ~ '^[0-9]+$'
                ORDER BY (f.payload->>'rank')::int ASC LIMIT ?""",
            # Over-fetch: the rows are then filtered to the relevant discipline
            # and deduped to one entry per college, both of which shrink the set.
            [*params, max(limit * 8, 40)])
    except Exception as exc:  # noqa: BLE001
        print(f"[retrieve] NIRF ranking unavailable: {exc}", file=sys.stderr)
        return []

    prefer = _preferred_categories(f or {})
    order = {cat: i for i, cat in enumerate(prefer)}

    # One entry per college, keeping the rank from the most relevant discipline.
    # Without this a single college occupies several slots — IIT Madras appeared
    # three times (Overall, Engineering, Research) in a five-item list.
    best: dict[str, dict] = {}
    for r in rows:
        d = dict(r)
        cid = d["college_id"]
        cat_rank = order.get(d.get("nirf_category"), len(prefer) + 1)
        d["_pref"] = cat_rank
        prev = best.get(cid)
        if prev is None or (cat_rank, d["nirf_rank"]) < (prev["_pref"], prev["nirf_rank"]):
            best[cid] = d

    out = sorted(best.values(), key=lambda d: (d["_pref"], d["nirf_rank"]))
    for d in out:
        d["value"] = d.get("nirf_rank")
        d.pop("_pref", None)
    return out[:limit]


def hydrate_ids(college_ids: list[str]) -> list[dict]:
    """Turn a bare id list into full hits, bypassing ranking entirely.

    Needed because superlatives and registry rankings identify colleges by SQL
    over the whole population, not by similarity to the question — so those
    colleges are frequently absent from the ranked top-k. Naming a college the
    generator cannot cite is worse than not naming it, so the caller retrieves
    them explicitly. Score is 0.0: these earned their place by being the answer,
    not by matching the query text.
    """
    if not college_ids:
        return []
    return _hydrate(list(college_ids), {})


def offers_program(college_ids: list[str], term: str) -> set[str]:
    """Which of `college_ids` actually offer `term`, per the courses table.

    This exists because the rendered card is NOT a sound source for that
    question: college_agg.course_titles holds deduped degree abbreviations
    ("MD; BSc; MS"), while the search prefilter matches `title OR full_name`.
    So "nursing" matches a college through "Bachelor of Science in Nursing"
    during retrieval, then appears absent when the card is re-read — measured
    at 200/200 sampled colleges for terms like engineering/nursing/law.

    Anything downstream that re-checks "does this college offer X?" has to ask
    the same question the same way retrieval did, or it will contradict itself
    and throw away correct results.
    """
    term = (term or "").strip()
    if not college_ids or not term:
        return set()
    be = get_backend()
    out: set[str] = set()
    like = f"%{term.lower()}%"
    # Chunked to stay clear of SQLite's variable limit on long id lists.
    for i in range(0, len(college_ids), 400):
        chunk = college_ids[i:i + 400]
        qs = ",".join("?" * len(chunk))
        rows = be.q(
            f"""SELECT DISTINCT college_id FROM courses
                WHERE college_id IN ({qs})
                  AND (LOWER(title) LIKE ? OR LOWER(full_name) LIKE ?)""",
            [*chunk, like, like],
        )
        out.update(r["college_id"] for r in rows)
    return out


def _hydrate(ids: list[str], scores: dict[str, float]) -> list[dict]:
    """ids -> the hit dicts the pipeline consumes. One round trip for the rows,
    one for the facts needed only when a card has to be rendered on the fly."""
    be = get_backend()
    if not ids:
        return []
    qs = ",".join("?" * len(ids))
    rows = {r["college_id"]: r for r in be.q(
        f"""SELECT c.*, a.n_courses, a.program_levels, a.course_titles,
                   a.departments, a.entrance_exams, a.min_tuition_inr,
                   a.max_tuition_inr, a.min_hostel_inr, a.has_hostel,
                   a.has_scholarship, a.card
            FROM colleges c LEFT JOIN college_agg a ON a.college_id = c.college_id
            WHERE c.college_id IN ({qs})""", ids)}
    need_render = [i for i in ids if i in rows and not rows[i].get("card")]
    facts: dict[str, dict] = {}
    if need_render:
        qs2 = ",".join("?" * len(need_render))
        for r in be.q(
            f"SELECT college_id, kind, payload FROM college_facts "
            f"WHERE college_id IN ({qs2})", need_render
        ):
            facts.setdefault(r["college_id"], {})[r["kind"]] = r["payload"]

    hits = []
    for cid in ids:
        row = rows.get(cid)
        if row is None:
            # Index knows an id the store does not (the enrichment crawl is
            # writing concurrently). Skip it rather than emit an uncitable hit.
            continue
        # The stored card is the ETL's rendering of render_card; re-rendering
        # here is the fallback so retrieval still works mid-rebuild. Both go
        # through the same function, so they cannot drift.
        card = row.get("card") or render_card(row, row, facts.get(cid))
        hits.append({
            "college_id": cid,
            "score": float(scores.get(cid, 0.0)),
            "card": card,
            "name": row.get("name"),
            "city": row.get("city"),
            "state": row.get("state"),
            "type": row.get("type"),
            "n_courses": row.get("n_courses") or 0,
            "min_tuition_inr": row.get("min_tuition_inr"),
            "max_tuition_inr": row.get("max_tuition_inr"),
        })
    return hits


def search(question: str, filters: dict | None = None, top_k: int = 12, query_vec=None) -> list[dict]:
    """Hybrid retrieval. Returns at most `top_k` hit dicts, best first.

    `query_vec`: a precomputed embedding from encode_query(), so the caller can
    overlap encoding with the router LLM call instead of paying for it twice.

    `score` is an RRF score, not a similarity. It is meaningful only for
    ordering within one result set and must never be shown to a student as a
    "match percentage".
    """
    global _index_failed
    f = _normalise_filters(filters)
    where, params = _where(f)
    be = get_backend()
    top_k = max(1, int(top_k))

    cand = [r["college_id"] for r in be.q(
        f"SELECT c.college_id {_FROM} WHERE {where} LIMIT ?", [*params, _MAX_CANDIDATES]
    )]
    if not cand:
        return []

    pool = max(top_k * _POOL_MULT, _POOL_MIN)

    lex_ids: list[str] = []
    match = _fts_query(question, f.get("course_terms"))
    if match:
        lex_ids = _lexical(match, where, params, pool)

    dense_ids: list[str] = []
    if str(question or "").strip() or query_vec is not None:
        try:
            idx = _get_index()
            qv = query_vec if query_vec is not None else encode_query(question)
            dense_ids = _dense(idx, cand, qv, pool)
        except Exception as e:
            # Serving lexical-only is degraded, not wrong: the hard filters and
            # every number in the card are unaffected. Warn once per process so
            # a broken deploy is visible without spamming a log per query.
            msg = str(e)
            if _index_failed != msg:
                _index_failed = msg
                print(f"[retrieve] dense ranker unavailable, BM25 only: {e}", file=sys.stderr)

    fused = _rrf(dense_ids, lex_ids)
    if fused:
        dense_rank = {cid: i for i, cid in enumerate(dense_ids)}
        # Ties are common at small pool sizes; break them on dense order and
        # then on id so the same query always returns the same list.
        ordered = sorted(fused, key=lambda c: (-fused[c], dense_rank.get(c, 10**6), c))[:top_k]
    else:
        # Filter-only questions ("government colleges in Goa with a hostel")
        # carry no rankable text. Falling back to breadth of offering is a
        # defensible, deterministic order — and far better than rowid. The
        # prefilter is re-run rather than fed back as an IN list: SQLite caps
        # host parameters at ~32k and an unfiltered candidate set exceeds it.
        ordered = [r["college_id"] for r in be.q(
            f"SELECT c.college_id {_FROM} WHERE {where} "
            f"ORDER BY COALESCE(a.n_courses, 0) DESC, c.name ASC LIMIT ?",
            [*params, top_k],
        )]

    return _hydrate(ordered, fused)


def to_context(hits: list[dict], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Concatenate cards into the grounding block, under a hard char budget.

    Truncation drops whole cards from the tail, never splits one. A card cut in
    half is the worst possible failure here: the fee line or the abroad-currency
    note can vanish while the college name survives, and the generator then
    answers confidently from a mutilated record.
    """
    out: list[str] = []
    used = 0
    for h in hits or []:
        card = (h.get("card") or "").strip()
        if not card:
            continue
        cost = len(card) + (len(_CARD_SEP) if out else 0)
        if used + cost > max_chars:
            if out:
                break
            # Degenerate case: even the top card exceeds the budget. Emitting
            # nothing would be worse, so cut at a line boundary and say so.
            keep = card[:max_chars].rsplit("\n", 1)[0]
            return keep + "\n[card truncated]"
        out.append(card)
        used += cost
    return _CARD_SEP.join(out)


if __name__ == "__main__":  # manual smoke check: python -m rag_core.retrieve "..."
    import json

    q = " ".join(sys.argv[1:]) or "affordable BTech college in Punjab with hostel"
    flt = {"state": "Punjab", "course_terms": ["BTech"], "hostel_required": True}
    print(f"matching colleges: {count_matching(flt)}", file=sys.stderr)
    for h in search(q, flt, top_k=5):
        print(f"{h['score']:.4f}  {h['name']} — {h['city']}, {h['state']}", file=sys.stderr)
    print(json.dumps(superlatives(flt, limit=3), indent=1, default=str))
