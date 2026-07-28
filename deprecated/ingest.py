"""Embed all college cards once and save a tiny local index (numpy file).

For 15 records an in-process cosine-similarity index is the pragmatic choice:
zero infra to install, identical retrieval math to a vector DB. At real scale
this layer swaps for pgvector without touching the rest of the pipeline.

Run:  python -m rag_core.ingest
(answer.py also auto-builds the index on first use.)
"""
import sys
import time

import numpy as np

from .config import EMBED_MODEL, INDEX_FILE
from .data import college_card, load_colleges


def build_index(verbose=True):
    t0 = time.perf_counter()
    from sentence_transformers import SentenceTransformer  # lazy: heavy import

    from .retrieve import _fingerprint

    rows = load_colleges()
    cards = [college_card(r) for r in rows]
    try:  # cached model: skip the online hub check (seconds saved per load)
        model = SentenceTransformer(EMBED_MODEL, local_files_only=True)
    except Exception:  # first ever run: download normally
        model = SentenceTransformer(EMBED_MODEL)
    # e5 models expect "passage:" / "query:" prefixes.
    vecs = model.encode([f"passage: {c}" for c in cards], normalize_embeddings=True)
    np.savez(
        INDEX_FILE,
        vectors=vecs.astype(np.float32),
        ids=np.array([r["college_id"] for r in rows]),
        # data identity (CSV bytes + embed model) — lets retrieve.py detect a
        # stale index after the CSV is edited and rebuild automatically.
        fingerprint=np.array(_fingerprint()),
    )
    dt = time.perf_counter() - t0
    if verbose:
        print(
            f"Indexed {len(rows)} colleges with {EMBED_MODEL} in {dt:.1f}s -> {INDEX_FILE}",
            file=sys.stderr,
        )
    return dt


if __name__ == "__main__":
    build_index()
