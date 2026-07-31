"""Dense semantic index over the college cards: build, resume, load.

The unit indexed here is one college card (see `store.db.render_card`) — the
same text FTS5 indexes and the same text the generator is shown, so lexical
and dense retrieval can never disagree about what a college "is".

WHY A FLAT NUMPY MATRIX AND NOT AN ANN INDEX
--------------------------------------------
38,700 cards x 384 dims (multilingual-e5-small) in float32 is 59 MB. That fits
in RAM on the same box that serves the API, and one query is a single sgemv
over that matrix: 30 MFLOP, memory-bandwidth bound. Measured on this machine
(numpy, 35,316 x 384 matvec + top-12 partition, 30 warm runs): p50 2.1 ms,
p95 3.2 ms. That is ~1% of the LLM call it feeds.

hnswlib or faiss would shave maybe 2 ms off that and cost two things:

  * a second artefact that drifts. An ANN graph is built once and then has to
    be kept in sync with the DB by hand; a stale or partially-rebuilt graph
    degrades recall *silently* — there is no exception to catch, just answers
    that quietly stop containing the right college.
  * approximate recall. Top-k that changes between rebuilds is indistinguishable
    from a bug when a student says "you missed my college", and we cannot
    reproduce it.

Exact search has neither failure mode: the vectors ARE the index, so the only
consistency invariant is "ids align with rows", which is asserted on load.

The crossover is roughly 10^6 vectors — 1.5 GB resident and ~60 ms/query at
this dim (measured by extrapolating the same benchmark), which is where a p95
latency budget actually starts to hurt. That is ~30x the current corpus. Even
switching to per-course granularity (211k rows, ~13 ms/query) stays
comfortably exact. When it does cross, `build_index` /
`load_index` are the only two functions that change; nothing above this file
knows how neighbours are found.

COST MODEL
----------
Embedding is the expensive step, not search: 38,700 cards averaging 867 chars
(~220 tokens) on CPU. Measured 5.6 cards/s while two other jobs saturated this
box — a cold full build is an hour-plus here, minutes on an idle machine.

So this module is incremental by construction: every card's content hash is
stored alongside its vector, and a rebuild only re-embeds cards whose text
actually changed. A rebuild after an ETL run that touched 200 colleges costs
200 embeddings, not 38,700 — seconds instead of an hour, which is what makes
"re-crawl and re-index nightly" a routine operation rather than a project.
Long runs also checkpoint every few thousand vectors, so a killed process
resumes where it stopped instead of starting over.

Run:  python -m rag_core.index_build [--rebuild] [--limit N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import EMBED_MODEL, INDEX_FILE
from .store import db

# e5 is an asymmetric model: documents must carry "passage: " and queries
# "query: ". Getting this wrong costs ~10 points of recall silently, so both
# prefixes live here and retrieval imports QUERY_PREFIX rather than retyping it.
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "

# Sentence-transformers' own batch size. 128 keeps CPU batches full without a
# multi-GB activation spike; override when the ETL is not competing for cores.
DEFAULT_BATCH = int(os.getenv("MME_EMBED_BATCH", "128"))
# Bigger than the CPU default (which cannot keep a GPU fed) but deliberately not
# huge. Measured on a 4 GB RTX 3050: batch 512 filled dedicated VRAM to
# 3.9/4.0 GB and spilled into "shared GPU memory", which is system RAM -- host
# memory hit 97% and the CPU downclocked to 0.98 GHz while a crawler was still
# running. A batch that quietly borrows host RAM is worse than a smaller one:
# throughput gains flatten well before this point anyway.
GPU_BATCH = int(os.getenv("MME_EMBED_BATCH_GPU", "192"))

# Flush a partial index this often (in newly embedded cards). A 54 MB write
# costs ~0.2 s; losing 20 minutes of embedding to a killed terminal costs more.
CHECKPOINT_EVERY = int(os.getenv("MME_EMBED_CHECKPOINT", "4000"))

# The ETL owns build_meta's key names and has not landed yet; probe the
# plausible ones. The id is advisory (it lets the server refuse to serve an
# index built from a different corpus), so absence is not a build failure.
_BUILD_ID_KEYS = ("build_id", "corpus_build_id", "build_finished_at", "built_at")

_INDEX_FORMAT = 1


class IndexBuildError(RuntimeError):
    """Raised when the corpus the index is built from is missing or unusable."""


# ---------------------------------------------------------------------------
# corpus side
# ---------------------------------------------------------------------------


def _store():
    """The corpus store — Postgres.

    This module was left on sqlite3 when the rest of the store moved, and
    `db.DB_FILE` became a compatibility shim object rather than a path. So
    `Path(db.DB_FILE)` raised a TypeError and NO index could be built at all —
    silently, because nothing exercises this path except a rebuild.
    """
    from .store.backend import get_backend

    try:
        return get_backend()
    except Exception as e:  # noqa: BLE001
        raise IndexBuildError(
            f"cannot reach the corpus store ({e}). The dense index is built FROM "
            "the store, so Postgres must be up and the ETL must have populated "
            "college_agg.card before there is anything to embed."
        ) from e


def _db_build_meta(be) -> tuple[str, dict]:
    try:
        rows = be.q("SELECT key, value FROM build_meta")
    except Exception:  # noqa: BLE001 - advisory only; absence is not a failure
        return "", {}
    meta = {r["key"]: str(r["value"])[:200] for r in rows}
    build_id = next((meta[k] for k in _BUILD_ID_KEYS if meta.get(k)), "")
    return build_id, meta


def _read_cards(be, limit: int | None) -> tuple[list[str], list[str], dict]:
    """(ids, cards, corpus stats), ordered by id so the index is reproducible."""
    try:
        row = be.q(
            "SELECT COUNT(*) AS total, "
            "COUNT(*) FILTER (WHERE card IS NULL OR TRIM(card) = '') AS blank "
            "FROM college_agg")[0]
    except Exception as e:  # noqa: BLE001
        raise IndexBuildError(
            f"cannot read college_agg ({e}). Run the ETL that populates the "
            "corpus before indexing."
        ) from e
    total, blank = int(row["total"] or 0), int(row["blank"] or 0)
    if not total:
        raise IndexBuildError(
            "college_agg is empty — there are no cards to embed. The ETL has "
            "not populated the store yet (or is still running); re-run this "
            "once it reports a college count."
        )

    sql = ("SELECT college_id, card FROM college_agg "
           "WHERE card IS NOT NULL AND TRIM(card) <> '' ORDER BY college_id")
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    rows = be.q(sql)
    if not rows:
        raise IndexBuildError(
            f"college_agg has {total} rows but every card is NULL/empty. "
            "render_card produced nothing — fix the ETL rather than indexing "
            "blank text, which would embed 35k identical vectors."
        )
    # Not fatal: a mid-run ETL legitimately has cards missing. But it is a
    # silent recall hole (those colleges become unreachable), so it is counted
    # and reported rather than swallowed.
    if blank:
        print(f"[index] WARNING: {blank:,} of {total:,} college_agg rows have no "
              f"card and will NOT be retrievable", file=sys.stderr)

    ids = [r["college_id"] for r in rows]
    cards = [r["card"] for r in rows]
    if len(set(ids)) != len(ids):
        raise IndexBuildError("duplicate college_id in college_agg — the index "
                              "cannot map a vector back to one college")
    return ids, cards, {"agg_rows": int(total), "missing_card": blank}


def _card_hash(card: str) -> str:
    """Identity of the exact text handed to the encoder.

    Hashing the prefixed text (not the bare card) means a change to
    PASSAGE_PREFIX invalidates every vector automatically, which is correct:
    the stored vectors really would be wrong.
    """
    return hashlib.blake2b((PASSAGE_PREFIX + card).encode("utf-8"),
                           digest_size=16).hexdigest()


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


def pick_device() -> str:
    """"cuda" when a usable GPU is present, else "cpu".

    Embedding is the one genuinely GPU-shaped job here: 24,607 cards took 72
    minutes at 5.7/s on the CPU, and every harvest makes the index stale again,
    so this cost is paid repeatedly. It also shortens the query path, which is
    what a student actually waits on.

    NOT set to cuda blindly: a CPU-only torch build reports no CUDA, and a
    machine whose driver is older than the wheel's runtime raises on the first
    real kernel rather than at import. So the probe runs a tiny op and falls
    back rather than letting a 25,000-card run die at card 1.

    MIVI_EMBED_DEVICE overrides, for reproducing a CPU-built index exactly.
    """
    forced = os.environ.get("MIVI_EMBED_DEVICE", "").strip().lower()
    if forced:
        return forced
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu"
        torch.zeros(8, device="cuda").sum().item()   # prove a kernel runs
        return "cuda"
    except Exception as exc:  # noqa: BLE001 - any GPU failure means use the CPU
        print(f"[index] GPU unusable ({type(exc).__name__}: {str(exc)[:70]}); "
              f"embedding on CPU", file=sys.stderr)
        return "cpu"


def _load_model(name: str, device: str | None = None):
    from sentence_transformers import SentenceTransformer  # lazy: heavy import

    device = device or pick_device()
    try:
        # Cached model: skip the Hub online check, which costs seconds per run.
        return SentenceTransformer(name, local_files_only=True, device=device)
    except Exception:
        # first run on this machine: download
        return SentenceTransformer(name, device=device)


# ---------------------------------------------------------------------------
# npz io
# ---------------------------------------------------------------------------


def _save_npz(path: Path, vectors: np.ndarray, ids, hashes, meta: dict) -> None:
    """Atomic write. A half-written index.npz would be loaded as a valid one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")  # keep .npz: savez appends it otherwise
    np.savez(
        tmp,
        vectors=vectors,
        ids=np.asarray(ids),
        card_hashes=np.asarray(hashes),
        # JSON in a 0-d array keeps the file loadable with allow_pickle=False —
        # an index file is a deserialisation surface and must not run code.
        meta=np.asarray(json.dumps(meta)),
    )
    # savez_compressed is not used: normalised float32 embeddings are close to
    # incompressible (~5% saved) and it triples both write and load time.
    os.replace(tmp, path)


def _load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
        return {
            "vectors": z["vectors"],
            "ids": [str(i) for i in z["ids"]],
            "card_hashes": [str(h) for h in z["card_hashes"]] if "card_hashes" in z.files else [],
            "meta": meta,
        }


_cache_lock = threading.Lock()
_cache: tuple[tuple, dict] | None = None


def load_index(path: Path | None = None, *, use_cache: bool = True) -> dict:
    """{"vectors": (n,d) float32 C-contiguous, "ids": [str], "meta": {...}}.

    Cached per (path, mtime, size) so the FastAPI threadpool loads 54 MB once
    but a rebuild underneath a running server is still picked up.
    """
    global _cache
    path = Path(path or INDEX_FILE)
    if not path.exists():
        raise IndexBuildError(
            f"dense index not found: {path}\nBuild it with: python -m rag_core.index_build"
        )
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if not use_cache:
        return _read_index(path, key, store=False)

    with _cache_lock:
        if _cache and _cache[0] == key:
            return _cache[1]
        # Single-flight. The cache key includes mtime, and every write path
        # (_save_npz, and _checkpoint every 4,000 cards) ends in os.replace —
        # so a rebuild invalidates the cache for all in-flight requests at the
        # same instant. Without holding the lock across the LOAD, each thread
        # that misses reads and materialises its own ~59 MB copy: measured at
        # 16 concurrent loads for one invalidation, and FastAPI's threadpool is
        # wider than that. Holding the lock makes the other threads wait on one
        # load instead of stampeding memory.
        #
        # A plain lock (not a per-key future) is the right shape here: loads are
        # sub-second, the waiters have nothing useful to do without the index
        # anyway, and it keeps the invariant trivially checkable.
        return _read_index(path, key, store=True)


def _read_index(path: Path, key: tuple, *, store: bool) -> dict:
    """Load, validate and optionally publish the index. Callers that publish
    MUST already hold `_cache_lock` (see load_index)."""
    global _cache

    raw = _load_npz(path)
    vectors = np.ascontiguousarray(raw["vectors"], dtype=np.float32)
    ids = raw["ids"]
    if vectors.ndim != 2 or vectors.shape[0] != len(ids):
        raise IndexBuildError(
            f"corrupt index {path}: {vectors.shape} vectors vs {len(ids)} ids. "
            "Rebuild with --rebuild."
        )
    meta = raw["meta"]
    if meta.get("embed_model") and meta["embed_model"] != EMBED_MODEL:
        raise IndexBuildError(
            f"index was built with {meta['embed_model']} but EMBED_MODEL is now "
            f"{EMBED_MODEL}. Query and document vectors would live in different "
            "spaces and every score would be noise. Rebuild the index."
        )
    if not meta.get("complete", True):
        # A checkpoint or a --limit build. Loud, not fatal: smoke-testing a
        # partial index is legitimate, serving one unknowingly is not.
        print(f"[index] WARNING: {path.name} is PARTIAL "
              f"({len(ids):,} of {meta.get('n_cards_in_db', '?')} cards). "
              "Retrieval will miss colleges until a full build finishes.",
              file=sys.stderr)

    out = {"vectors": vectors, "ids": ids, "meta": meta}
    if store:
        _cache = (key, out)
    return out


# ---------------------------------------------------------------------------
# progress
# ---------------------------------------------------------------------------


class _Progress:
    """Throughput + ETA on stderr. A 35k-card embed is a coffee-break job; a
    silent one is indistinguishable from a hung one."""

    def __init__(self, total: int, every: float = 2.0):
        self.total = total
        self.t0 = time.perf_counter()
        self.last = 0.0
        self.last_done = -1
        self.tty = sys.stderr.isatty()
        # Non-tty (nohup, CI log): no \r rewriting, so throttle much harder.
        self.every = every if self.tty else max(every, 15.0)

    def update(self, done: int, *, force: bool = False) -> None:
        if done == self.last_done:
            return
        now = time.perf_counter()
        if not force and now - self.last < self.every:
            return
        self.last, self.last_done = now, done
        el = now - self.t0
        rate = done / el if el > 0 else 0.0
        eta = (self.total - done) / rate if rate > 0 else 0.0
        line = (f"[index] {done:,}/{self.total:,} cards "
                f"{100 * done / max(self.total, 1):5.1f}%  "
                f"{rate:6.1f} cards/s  elapsed {_hms(el)}  eta {_hms(eta)}")
        end = "\r" if (self.tty and done < self.total) else "\n"
        print(line.ljust(78), file=sys.stderr, end=end, flush=True)


def _hms(sec: float) -> str:
    sec = int(max(sec, 0))
    if sec >= 3600:
        return f"{sec // 3600}h{(sec % 3600) // 60:02d}m"
    if sec >= 60:
        return f"{sec // 60}m{sec % 60:02d}s"
    return f"{sec}s"


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def build_index(
    rebuild: bool = False,
    limit: int | None = None,
    *,
    db_path: Path | None = None,
    out_path: Path | None = None,
    batch_size: int = DEFAULT_BATCH,
    embed_model: str = EMBED_MODEL,
) -> dict:
    """Embed every college card, reusing vectors whose card text is unchanged.

    `rebuild=True` ignores the existing file and re-embeds everything (use it
    after an encoder change that the model name does not capture, e.g. a
    max_seq_length tweak). `limit` truncates for smoke tests and marks the
    result partial so it can never be mistaken for a servable index.
    """
    t0 = time.perf_counter()
    out = Path(out_path or INDEX_FILE)

    be = _store()
    ids, cards, corpus = _read_cards(be, limit)
    db_build_id, db_meta = _db_build_meta(be)

    n = len(ids)
    hashes = [_card_hash(c) for c in cards]

    # ---- reuse pass -------------------------------------------------------
    # Prefer a leftover .part: it is a killed run's progress and holds strictly
    # more finished work than the last complete index, so resuming from it is
    # what makes an interrupted rebuild cheap rather than merely restartable.
    prior: dict[str, tuple[str, np.ndarray]] = {}
    resume_from = out
    part_probe = out.with_suffix(out.suffix + ".part")
    if not rebuild and part_probe.exists():
        try:
            if len(_load_npz(part_probe)["ids"]) > (
                    len(_load_npz(out)["ids"]) if out.exists() else 0):
                resume_from = part_probe
                print(f"[index] resuming from {part_probe.name} "
                      f"(a previous run was interrupted)", file=sys.stderr)
        except Exception:  # noqa: BLE001 - a torn .part is simply ignored
            pass
    if not rebuild and resume_from.exists():
        try:
            old = _load_npz(resume_from)
            if old["meta"].get("embed_model") not in (None, embed_model):
                print(f"[index] embed model changed "
                      f"({old['meta'].get('embed_model')} -> {embed_model}); "
                      "re-embedding everything", file=sys.stderr)
            elif len(old["card_hashes"]) == len(old["ids"]):
                prior = {cid: (h, v) for cid, h, v
                         in zip(old["ids"], old["card_hashes"], old["vectors"])}
        except Exception as e:
            # A corrupt/legacy index must not block a rebuild — it is derived data.
            print(f"[index] could not reuse {out.name} ({e}); embedding from scratch",
                  file=sys.stderr)

    device = pick_device()
    model = _load_model(embed_model, device)
    # A GPU is idle-fast on tiny batches, so it wants a much bigger one than the
    # CPU default; the whole point is to keep the device fed.
    if device.startswith("cuda") and batch_size < GPU_BATCH:
        print(f"[index] cuda: batch {batch_size} -> {GPU_BATCH}", file=sys.stderr)
        batch_size = GPU_BATCH
    # sentence-transformers renamed this; support both so a library bump does
    # not break the build (or spam a FutureWarning on every run).
    _dim_fn = getattr(model, "get_embedding_dimension", None) or \
        model.get_sentence_embedding_dimension
    dim = int(_dim_fn())

    vectors = np.zeros((n, dim), dtype=np.float32)
    filled = np.zeros(n, dtype=bool)
    todo: list[int] = []
    for i, (cid, h) in enumerate(zip(ids, hashes)):
        hit = prior.get(cid)
        if hit is not None and hit[0] == h and hit[1].shape[0] == dim:
            vectors[i] = hit[1]
            filled[i] = True
        else:
            todo.append(i)
    reused = n - len(todo)
    dropped = len(set(prior) - set(ids))

    def _meta(complete: bool) -> dict:
        return {
            "format": _INDEX_FORMAT,
            "embed_model": embed_model,
            "dim": dim,
            "n": int(filled.sum()) if not complete else n,
            "n_cards_in_db": corpus["agg_rows"],
            "prefix": PASSAGE_PREFIX,
            "normalized": True,
            "complete": complete,
            "limit": limit,
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            # Which store this index was built from. A DSN label, not a file
            # path, since the SQLite port: the value is only for an operator
            # comparing an index against the corpus it claims to describe.
            "db_path": f"postgres:{getattr(be, 'name', 'unknown')}",
            "db_build_id": db_build_id,
            "db_build_meta": db_meta,
        }

    # Checkpoints go to a SIDECAR, never to the path retrieval reads.
    #
    # They used to write straight to `out`, which meant a rebuild made part of
    # the corpus unfindable for its whole duration: measured mid-run, the live
    # index held 26,381 vectors of 38,700, so 12,319 colleges were invisible to
    # students — caused BY the rebuild, not merely unfixed by it. The swap is
    # atomic, but atomically publishing a partial index is still publishing a
    # partial index.
    #
    # Now the old complete index keeps serving until the new one is finished.
    part = out.with_suffix(out.suffix + ".part")

    def _checkpoint() -> None:
        # Only the rows actually embedded so far: a zero row whose hash matched
        # would be reused as a real vector by the next run.
        keep = np.flatnonzero(filled)
        _save_npz(part, vectors[keep],
                  [ids[i] for i in keep], [hashes[i] for i in keep], _meta(False))

    print(f"[index] {n:,} cards | reuse {reused:,} | embed {len(todo):,} "
          f"| model {embed_model} (dim {dim})", file=sys.stderr)

    # ---- embed pass -------------------------------------------------------
    prog = _Progress(len(todo))
    since_ckpt = 0
    # Chunked so progress, ETA and checkpoints happen inside the job rather than
    # only at the end. Cards are not length-sorted globally: sentence-transformers
    # already sorts within each encode() call, and most cards saturate the
    # 512-token window anyway, so the padding win is noise.
    chunk = batch_size * 4
    for start in range(0, len(todo), chunk):
        block = todo[start:start + chunk]
        texts = [PASSAGE_PREFIX + cards[i] for i in block]
        vecs = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,   # cosine == dot product downstream
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32, copy=False)
        if vecs.shape != (len(block), dim):
            raise IndexBuildError(f"encoder returned {vecs.shape}, expected "
                                  f"{(len(block), dim)} — refusing to write a "
                                  "misaligned index")
        if not np.isfinite(vecs).all():
            raise IndexBuildError("encoder produced NaN/inf vectors; the ranking "
                                  "would be silently meaningless")
        vectors[block] = vecs
        filled[block] = True

        prog.update(start + len(block))
        since_ckpt += len(block)
        if since_ckpt >= CHECKPOINT_EVERY and start + len(block) < len(todo):
            _checkpoint()
            since_ckpt = 0
    if todo:
        prog.update(len(todo), force=True)

    if not filled.all():
        raise IndexBuildError("internal: not every card was embedded")
    # Cheap invariant: e5 + normalize_embeddings must give unit vectors, and a
    # non-unit row means dot-product scores are not cosine similarities.
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise IndexBuildError(f"vectors are not L2-normalised "
                              f"(min {norms.min():.4f}, max {norms.max():.4f})")

    complete = limit is None and corpus["missing_card"] == 0
    _save_npz(out, vectors, ids, hashes, _meta(complete))
    # The sidecar has served its purpose; leaving it would only invite a future
    # run to weigh a superseded checkpoint against the index it produced.
    part.unlink(missing_ok=True)

    dt = time.perf_counter() - t0
    stats = {
        "n": n,
        "dim": dim,
        "embedded": len(todo),
        "reused": reused,
        "dropped": dropped,
        "missing_card": corpus["missing_card"],
        "cards_in_db": corpus["agg_rows"],
        "complete": complete,
        "seconds": round(dt, 1),
        "cards_per_sec": round(len(todo) / dt, 1) if todo and dt else None,
        "megabytes": round(vectors.nbytes / 1e6, 1),
        "embed_model": embed_model,
        "db_build_id": db_build_id,
        "path": str(out),
    }
    print(f"[index] wrote {out} — {n:,} x {dim} float32 "
          f"({stats['megabytes']} MB) in {_hms(dt)} "
          f"({len(todo):,} embedded, {reused:,} reused)", file=sys.stderr)
    return stats


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m rag_core.index_build",
        description="Build the dense semantic index over college cards.")
    p.add_argument("--rebuild", action="store_true",
                   help="re-embed every card, ignoring the existing index")
    p.add_argument("--limit", type=int, default=None,
                   help="embed only the first N cards (smoke test; marks the "
                        "index partial)")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    p.add_argument("--model", type=str, default=EMBED_MODEL, help="embedding model name")
    p.add_argument("--db", type=Path, default=None, help="corpus sqlite path")
    p.add_argument("--out", type=Path, default=None, help="output .npz path")
    a = p.parse_args(argv)
    try:
        stats = build_index(rebuild=a.rebuild, limit=a.limit, db_path=a.db,
                            out_path=a.out, batch_size=a.batch_size, embed_model=a.model)
    except IndexBuildError as e:
        print(f"[index] {e}", file=sys.stderr)
        return 2
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
