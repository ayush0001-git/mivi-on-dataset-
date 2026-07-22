"""Hybrid retrieval: hard structured filters + semantic similarity + a small lexical boost.

Hard filters (fee ceiling, score vs cutoff, type, hostel, city) are applied in
code, not left to the LLM — a wrong fee or cutoff must be impossible, not
merely unlikely. Semantic search then ranks whatever survives the filters.
"""
import hashlib
import sys
import threading

import numpy as np

from .config import DATA_CSV, EMBED_MODEL, INDEX_FILE, TOP_K
from .data import college_card, load_colleges

_model = None
_index = None
_model_lock = threading.Lock()   # concurrent first requests must not
_index_lock = threading.Lock()   # double-load a 450MB model / rebuild twice


def _fingerprint():
    """Identity of the data the index must match: CSV bytes + embed model."""
    h = hashlib.sha256()
    h.update(DATA_CSV.read_bytes())
    h.update(EMBED_MODEL.encode())
    return h.hexdigest()


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer  # lazy: heavy import

                try:
                    # Cached model: skip the Hugging Face Hub online check —
                    # measured to cost seconds per CLI run (it phones home on
                    # every load otherwise).
                    _model = SentenceTransformer(EMBED_MODEL, local_files_only=True)
                except Exception:
                    # First ever run on this machine: download normally.
                    _model = SentenceTransformer(EMBED_MODEL)
    return _model


def _get_index():
    global _index
    if _index is None:
        with _index_lock:
            if _index is None:
                from .ingest import build_index

                if not INDEX_FILE.exists():
                    build_index()
                data = np.load(INDEX_FILE, allow_pickle=True)
                index = {"vectors": data["vectors"], "ids": [str(i) for i in data["ids"]]}
                # Stale-index guard: an edited CSV (or changed embed model) with
                # an old index.npz would silently rank against wrong vectors —
                # or crash on unknown ids. Rebuild automatically when detected.
                stored_fp = str(data["fingerprint"]) if "fingerprint" in data.files else None
                stale = (stored_fp != _fingerprint()) if stored_fp else (
                    # legacy index without a fingerprint: weaker check on ids
                    set(index["ids"]) != {r["college_id"] for r in load_colleges()})
                if stale:
                    print("[retrieve] index out of date with the CSV — rebuilding",
                          file=sys.stderr)
                    build_index()
                    data = np.load(INDEX_FILE, allow_pickle=True)
                    index = {"vectors": data["vectors"], "ids": [str(i) for i in data["ids"]]}
                _index = index
    return _index


def encode_query(question):
    """Embed one query (e5 models expect the "query:" prefix)."""
    return _get_model().encode([f"query: {question}"], normalize_embeddings=True)[0]


def warm():
    """Preload index + embedding model (called in a background thread at
    server startup so the first student never pays the model-load seconds)."""
    try:
        _get_index()
        _get_model()
    except Exception as e:  # warming is best-effort, never fatal
        print(f"[retrieve] warm-up failed: {e}", file=sys.stderr)


def passes_filters(row, f):
    if not f:
        return True
    if f.get("max_annual_fee_inr") and row["annual_fees_inr"] > f["max_annual_fee_inr"]:
        return False
    if f.get("student_score_pct") is not None and row["last_year_cutoff_pct"] > f["student_score_pct"]:
        return False
    if f.get("college_type") and row["type"].lower() != str(f["college_type"]).lower():
        return False
    if f.get("hostel_required") and row["hostel_available"].strip().lower() != "yes":
        return False
    if f.get("city"):
        loc = str(f["city"]).lower()
        # users say "in Uttarakhand" as often as "in Dehradun" — match either
        if loc not in row["city"].lower() and loc not in row["state"].lower():
            return False
    return True


def retrieve(question, filters=None, course_terms=None, top_k=TOP_K, query_vec=None):
    """Return [(row, score), ...] for rows passing hard filters, ranked by score.

    `query_vec`: pass a precomputed embedding (from encode_query) to skip
    re-encoding — the pipeline computes it concurrently with the router call."""
    idx = _get_index()
    rows = {r["college_id"]: r for r in load_colleges()}
    qv = query_vec if query_vec is not None else encode_query(question)
    sims = idx["vectors"] @ qv

    scored = []
    for i, cid in enumerate(idx["ids"]):
        row = rows.get(cid)
        if row is None:  # id drift is repaired by the stale-index rebuild;
            continue     # this guard just makes it impossible to crash on it
        if not passes_filters(row, filters):
            continue
        score = float(sims[i])
        # Lexical boost: exact course mentions (e.g. "MBA") should not depend on
        # embedding luck. Cheap and transparent; capped so semantics still lead.
        if course_terms:
            hay = (row["courses_offered"] + " " + row["name"]).lower()
            hits = sum(1 for t in course_terms if t and t.lower() in hay)
            score += min(0.15 * hits, 0.30)
        scored.append((row, score))

    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]


def to_context(results):
    return "\n\n".join(college_card(r) for r, _ in results)
