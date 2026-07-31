import os
import re
import sys
import time
import requests
from typing import Any

import numpy as np

from . import config
from .llm import chat
from .pipeline import _format_answer, _log_metrics, _parse_json
from .retrieve import _get_model, encode_query
from .webmode import _compose_query, _search_tavily, _search_firecrawl, FIRECRAWL_SEARCH

DEEP_RESEARCH_SYSTEM = """You are an expert college-counselling researcher.
You are given a set of text chunks extracted directly from web pages (a temporary knowledge base).
Answer the user's question using ONLY the provided text chunks.

Rules:
1. Base your answer purely on the provided chunks. Do NOT hallucinate names, fees, dates, or cutoffs.
2. If the context does not contain enough information to fully answer the question, state what is missing.
3. Cite your sources inline using [N] where N is the index of the source URL.
4. User preferences (budget, course, location, college type) are HARD CONSTRAINTS. 
   - Never recommend a college that violates a constraint.
   - If no college matches all constraints, clearly state that.
5. Format your output as a clear, structured explanation or list.
6. Return ONLY JSON, exactly:
{"answer": "<string>", "citations": ["<source URL>", ...], "answered": true|false, "reason_if_unanswered": null | "<string>"}
In "citations", list the URLs of the sources you actually used from the provided chunks."""

def _chunk_text(text: str, url: str, max_words: int = 150, overlap: int = 30) -> list[dict]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_text = " ".join(words[i:i + max_words])
        chunks.append({"url": url, "text": chunk_text})
        if i + max_words >= len(words):
            break
        i += (max_words - overlap)
    return chunks

def _tavily_full_search(query: str, key: str) -> list[dict]:
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            headers={"Content-Type": "application/json"},
            json={"api_key": key, "query": query, "search_depth": "advanced", "include_raw_content": True, "max_results": 3},
            timeout=45,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        sources = []
        for item in results:
            content = item.get("raw_content") or item.get("content") or ""
            if item.get("url") and content:
                sources.append({"url": item["url"], "text": content})
        return sources
    except Exception as e:
        print(f"[deep_research] tavily error: {e}", file=sys.stderr)
        return []

def deep_research(question: str, prefs: dict | None = None) -> dict[str, Any]:
    """Deep Research Mode: Temporary RAG over live web pages."""
    t_start = time.perf_counter()
    fc_key = os.getenv("FIRECRAWL_API_KEY", "")
    tav_key = os.getenv("TAVILY_API_KEY", "")
    
    if not fc_key and not tav_key:
        return {"answer": "Deep Research mode requires web search API keys (FIRECRAWL_API_KEY or TAVILY_API_KEY).",
                "citations": [], "answered": False,
                "reason_if_unanswered": "API keys missing."}

    calls = []
    # 1. Compose targeted search query
    search_query = _compose_query(question, prefs, calls)
    print(f"[deep_research] search query: {search_query!r}", file=sys.stderr)

    # 2. Fetch full results (try Tavily advanced first)
    sources = []
    if tav_key:
        sources = _tavily_full_search(search_query, tav_key)
    if not sources and fc_key:
        # Fallback to firecrawl standard search (which can return markdown)
        sources = _search_firecrawl(search_query, fc_key) or []

    if not sources:
        return {"answer": "Web search returned no usable results for this question.",
                "citations": [], "answered": False,
                "reason_if_unanswered": "No web results."}

    # 3. Chunking
    chunks = []
    for s in sources:
        chunks.extend(_chunk_text(s["text"], s["url"]))
    
    if not chunks:
        return {"answer": "Could not extract readable text from the search results.",
                "citations": [], "answered": False,
                "reason_if_unanswered": "No extractable text."}

    print(f"[deep_research] extracted {len(chunks)} chunks from {len(sources)} pages", file=sys.stderr)

    # 4. Embedding
    qv = encode_query(question)
    model = _get_model()
    # add query instruction if embedding model expects it (e5 handles it via PASSAGE_PREFIX)
    chunk_texts = [c["text"] for c in chunks]
    chunk_vecs = model.encode(chunk_texts, normalize_embeddings=True)
    
    # 5. Retrieval
    scores = np.dot(chunk_vecs, qv)
    top_indices = np.argsort(scores)[::-1][:10]  # top 10 chunks
    
    top_chunks = [chunks[i] for i in top_indices]
    
    # 6. Generation
    # Build context mapping for LLM citations
    urls = list({c["url"] for c in top_chunks})
    url_map = {url: i + 1 for i, url in enumerate(urls)}
    
    context_blocks = []
    for c in top_chunks:
        ref_id = url_map[c["url"]]
        context_blocks.append(f"Source [{ref_id}] ({c['url']}):\n{c['text']}")
    
    context = "\n\n".join(context_blocks)
    
    prefs_line = ""
    stated = {k: str(v).strip() for k, v in (prefs or {}).items()
              if k in ("course", "budget", "location", "type")
              and isinstance(v, (str, int, float)) and str(v).strip()}
    if stated:
        prefs_line = f"USER PREFERENCES (hard constraints): {stated}\n"

    try:
        raw = chat(
            [{"role": "system", "content": config.GEN_SYSTEM_PREFIX + DEEP_RESEARCH_SYSTEM},
             {"role": "user",
              "content": f"TEMPORARY KNOWLEDGE BASE:\n{context}\n\n{prefs_line}QUESTION: {question}"}],
            model=config.GEN_MODEL, json_mode=True, temperature=0.2,
            max_tokens=1000, usage_log=calls,
        )
        result = _parse_json(raw)
        result = {
            "answer": _format_answer(str(result.get("answer", "")).replace("**", "")),
            "citations": [c for c in (result.get("citations") or []) if isinstance(c, str)],
            "answered": bool(result.get("answered")),
            "reason_if_unanswered": result.get("reason_if_unanswered"),
        }
    except Exception as e:
        print(f"[deep_research] generation error: {e}", file=sys.stderr)
        result = {"answer": "Could not generate a reliable deep research answer.",
                  "citations": [], "answered": False,
                  "reason_if_unanswered": "Deep research generation failed."}

    _log_metrics({"question": question, "route": "deep_research", "calls": calls,
                  "retrieved": urls, "verified": False,
                  "total_latency_s": round(time.perf_counter() - t_start, 3)})
    return result
