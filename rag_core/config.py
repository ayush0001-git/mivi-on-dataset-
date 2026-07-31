"""Central config: paths, model names, env. Override anything via .env or environment."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# The corpus is the live MakeMyEducation production data, not a sample file:
# 211,511 course offerings across ~35,300 colleges, joined to the college
# catalogue pulled from admin.makemyeducation.com/api. The 15-row
# sample_colleges.csv the prototype used is retired — see data/DEPRECATED.md.
DB_FILE = ROOT / "data" / "mivi.db"          # built by rag_core.store.build_db
INDEX_FILE = ROOT / "data" / "index.npz"     # built by rag_core.index_build
RAW_DIR = ROOT / "data" / "raw"              # crawler output (JSONL)
METRICS_FILE = ROOT / "metrics" / "metrics.jsonl"

# The courses export lives outside the repo (169 MB). Override per machine.
COURSES_CSV = Path(
    os.getenv(
        "MME_COURSES_CSV",
        r"D:\make my education\colleg data cleaning\colleg data cleaning"
        r"\makeMyEducation.courses (2).csv",
    )
)

# Embeddings run locally (free, multilingual: English / Hindi / Hinglish).
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")

# Any OpenAI-compatible endpoint works. Default: Groq (free tier, fast).
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
GEN_MODEL = os.getenv("GEN_MODEL", "llama-3.3-70b-versatile")   # main answers
FAST_MODEL = os.getenv("FAST_MODEL", "llama-3.1-8b-instant")    # routing / filter extraction
# Provider-specific system-prompt prefix (e.g. "/no_think " disables reasoning
# tokens on NVIDIA Nemotron models). Empty by default.
GEN_SYSTEM_PREFIX = os.getenv("GEN_SYSTEM_PREFIX", "")

# Per-call HTTP timeout (seconds). Flash-class models answer in 2-3s; a hung
# call should fail into our retry/failover fast, not block a student. 12s is
# ~4x the healthy p50 — measured: a 25s timeout let one hung connection turn
# a 3s answer into an 89s answer.
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "12"))

# Optional AUTOMATIC FAILOVER provider: if the primary fails (rate limit /
# outage), the same call is retried on this one — demos never see the error.
FALLBACK_API_KEY = os.getenv("FALLBACK_API_KEY", "")
FALLBACK_BASE_URL = os.getenv("FALLBACK_BASE_URL", "")
FALLBACK_GEN_MODEL = os.getenv("FALLBACK_GEN_MODEL", "")
FALLBACK_FAST_MODEL = os.getenv("FALLBACK_FAST_MODEL", "")
FALLBACK_GEN_SYSTEM_PREFIX = os.getenv("FALLBACK_GEN_SYSTEM_PREFIX", "")

# In-memory answer cache — ON only in the demo server (app.py sets MME_CACHE=1).
# The graded CLI path stays uncached so measured latency/cost numbers and eval
# runs always reflect live pipeline behaviour.
CACHE_ENABLED = os.getenv("MME_CACHE", "0") == "1"

# Retrieval depth. The prototype could set k = "every record" because the
# corpus was 15 rows; at 35,300 colleges that is impossible, so k is now a real
# dial. 12 keeps the generator's context around 10-12k characters — enough to
# answer comparison and shortlist questions completely, while leaving the
# COUNTING of matches to SQL (retrieve.count_matching) rather than to whatever
# happened to fit in the window.
TOP_K = int(os.getenv("TOP_K", "12"))

# Adaptive top-k: override the default per question_kind. Lookups waste context
# on 11 irrelevant cards; enumerations truncate meaningful results. The router
# already classifies question_kind — use it.
ADAPTIVE_TOP_K = {
    "lookup":        int(os.getenv("TOP_K_LOOKUP", "5")),
    "profile_share": 0,  # no retrieval needed — student is sharing facts, not asking
    "enumerate":     int(os.getenv("TOP_K_ENUMERATE", "15")),
    "recommend":     int(os.getenv("TOP_K_RECOMMEND", "12")),
    "other":         TOP_K,
}

# Cited colleges per answer (citations[]). Capping here keeps the page a
# student reads from getting out of hand; the same cap also constrains the
# hydration in app.py so the two are aligned.
MAX_CITED_COLLEGES = int(os.getenv("MIVI_MAX_CITED_COLLEGES", "12"))

# Hard ceiling on the characters of college cards handed to the generator. A
# runaway context is both a cost and a quality problem: past ~12k the model
# starts losing the middle of the list.
#
# Derived from MAX_CITED_COLLEGES * ~1000 chars/card, with the override still
# honoured so an operator can decouple them when a card set is known to be
# unusually verbose (or terse).
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", str(MAX_CITED_COLLEGES * 1000)))

# Overseas institutions quote fees in their own currency (verified: domestic
# courses carry a rupee symbol, abroad carry dollars). Mixing them into a
# rupee budget is a factual error, so they are excluded unless a query is
# explicitly about studying abroad.
INCLUDE_ABROAD_DEFAULT = os.getenv("INCLUDE_ABROAD", "0") == "1"

# Public college page on the live site — a citation id IS the URL slug, which
# is what lets the chat UI link every claim back to a real page.
COLLEGE_URL = "https://makemyeducation.com/college/{college_id}"


EMBED_MODEL_OPTIONS = {
    "e5-small": "intfloat/multilingual-e5-small",       # current, 384 dim
    "e5-base": "intfloat/multilingual-e5-base",          # 768 dim, better quality
    "e5-large": "intfloat/multilingual-e5-large",        # 1024 dim, best e5
    "bge-m3": "BAAI/bge-m3",                            # 1024 dim, best multilingual
    "mpnet": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",  # 768 dim
    "cohere": "embed-multilingual-v3",                   # API, 1024 dim
}

EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))

RERANK_ENABLED = os.getenv("MIVI_RERANK", "1") == "1"
RERANK_CONFIDENCE_THRESHOLD = float(os.getenv("MIVI_RERANK_THRESHOLD", "0.01"))

