"""Optional demo UI — NOT part of the graded interface (no marks claimed).

    uvicorn app:app --port 8000      ->  http://localhost:8000

DB mode calls the exact same pipeline as answer.py. The "answer from web"
button calls the Firecrawl-backed web mode (rag_core/webmode.py), which is
clearly labelled and never used by answer.py / run_all.py / evals.
"""
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

os.environ.setdefault("MME_CACHE", "1")  # demo server: repeat questions answer instantly

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel


@asynccontextmanager
async def lifespan(_app):
    """Warm the embedding model + the answer cache at startup.

    The model warm-up means even the very first live question skips the
    multi-second model load. The cache is persisted to disk (data/
    answer_cache.json), so prewarm spends live LLM calls only on questions
    not already cached — after the first ever launch, startup is free.
    Disable question prewarm with MME_PREWARM=0."""
    import threading
    import time

    def warm_model():
        from rag_core.retrieve import warm

        warm()

    threading.Thread(target=warm_model, daemon=True).start()

    if os.environ.get("MME_PREWARM", "1") == "1":
        def run():
            from run_all import QUESTIONS
            from rag_core.pipeline import answer_question

            for q in QUESTIONS:
                try:
                    t0 = time.perf_counter()
                    answer_question(q)
                    if time.perf_counter() - t0 > 0.05:  # was a live call, not a cache hit:
                        time.sleep(2.0)                  # pace it — a burst of 14 calls
                except Exception as e:                   # tripped free-tier 429s at startup
                    print(f"[prewarm] failed for {q[:50]!r}: {e}", file=sys.stderr)

        threading.Thread(target=run, daemon=True).start()
    yield


app = FastAPI(title="MME RAG demo", lifespan=lifespan)
# Demo-friendliness: lets the page work even if someone opens the HTML file
# directly (file://) instead of http://localhost:8000.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])


class Ask(BaseModel):
    question: str
    mode: str = "db"  # "db" | "web"
    prefs: dict | None = None  # web mode: {course, budget, location, type}
    history: list | None = None  # recent [{role, content}] turns for memory
    known_profile: dict | None = None  # accumulated {marks_pct, field_interest,
                                        # budget, location} — survives history truncation


@app.post("/api/answer")
def api_answer(req: Ask):
    question = req.question.strip()
    if not question:
        return {"answer": "Please type a question.", "citations": [],
                "answered": False, "reason_if_unanswered": "Empty question."}
    if req.mode == "web":
        from rag_core.webmode import web_answer

        return web_answer(question, req.prefs)
    from rag_core.pipeline import answer_question

    return answer_question(question, history=req.history, known_profile=req.known_profile)


@app.get("/api/colleges")
def colleges():
    """id -> display name map, so the UI can show human-readable citations."""
    from rag_core.data import load_colleges

    return {r["college_id"]: f"{r['name']}, {r['city']}" for r in load_colleges()}


@app.get("/")
def home():
    return FileResponse(Path(__file__).resolve().parent / "static" / "index.html")
