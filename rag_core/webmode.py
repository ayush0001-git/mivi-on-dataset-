"""OPTIONAL 'answer from the web' mode — used only by the demo UI (app.py).

The graded pipeline (answer.py, run_all.py, evals/) NEVER touches the web:
the assignment dataset is the whole world there. This mode exists in the demo
interface behind an explicit button, to show where the production system goes
(live web data via Firecrawl). Web answers cite source URLs and are labelled
unverified — they are never mixed with the college-database path.

Speed design: Firecrawl SEARCH ONLY (title + snippet, no page scraping) keeps
the round trip in seconds; a fast-model call first composes a targeted search
query from the question + the user's stated preferences (budget, course,
location, type) so results are relevant and the generator has little room to
hallucinate.
"""
import os
import re
import sys
import time

import requests

from . import config
from .llm import chat
from .pipeline import _format_answer, _log_metrics, _parse_json

FIRECRAWL_SEARCH = "https://api.firecrawl.dev/v2/search"

QUERY_COMPOSER = """Compose ONE short web search query (English, max 12 words) to
answer a student's question about colleges/courses in India. Fold in their
stated preferences. Output ONLY the query text, nothing else.

Question: {question}
Preferences: {prefs}"""

WEB_SYSTEM = """You are a college-counselling assistant answering from live web
SEARCH RESULTS (title + snippet + URL only — no full pages). Rules:
1. Use ONLY the provided results. Cite the source number like [1] right after
   each claim. Never invent names, fees, cutoffs or rankings.
2. The user's stated preferences are HARD CONSTRAINTS, not suggestions:
   - College type "Private" -> NEVER recommend government institutions. IITs,
     NITs, IIITs, AIIMS, and state/central universities are Government — they
     are excluded when the user asked for Private (and vice versa).
   - Budget, course and location preferences equally exclude non-matching
     options. If a result violates any constraint, leave it out entirely.
   - If a stated cost may exceed the budget, or its unit is unclear (total
     course cost vs per-year), either exclude the option or explicitly flag
     the mismatch — never present it as within budget.
3. FORMAT: one short lead line, then a numbered list — one option per line,
   separated by "\\n": "1. <College>, <City> — <what the result says> [n]".
   Best-fit first, then 2-3 alternatives. Under 120 words. Plain text.
4. Snippets are shallow: if a detail (exact fee, cutoff, seat count) is not in
   the results, say it should be verified on the official site — do not guess.
5. Reply in the SAME language as the user's question.
6. If no result satisfies ALL the constraints, set "answered": false and say
   plainly that nothing matching was found — never bend a constraint to
   produce an answer.
7. BROAD questions: if the question has no location/course scope (e.g. "list
   government colleges with hostels" — thousands exist across India), do NOT
   present a few arbitrary results as THE answer. Start by saying the
   question needs narrowing (which state/city? which course?), then you may
   list what the search returned, explicitly labelled: "a few examples from
   search results — not an exhaustive list".
Return ONLY JSON, exactly:
{"answer": "<string>", "citations": ["<source URL>", ...],
 "answered": true|false, "reason_if_unanswered": null | "<string>"}
In "citations", list the URLs of the sources you actually used."""


def _compose_query(question, prefs, calls):
    parts = []
    for key, label in (("course", "course"), ("budget", "budget"),
                       ("location", "location"), ("type", "college type")):
        v = str((prefs or {}).get(key) or "").strip()  # tolerate null/absent prefs
        if v:
            parts.append(f"{label}: {v}")
    prefs_text = "; ".join(parts) if parts else "(none given)"
    try:
        q = chat(
            [{"role": "user",
              "content": QUERY_COMPOSER.format(question=question, prefs=prefs_text)}],
            model=config.FAST_MODEL, temperature=0.0, max_tokens=40, usage_log=calls,
        )
        # A reasoning fallback model may wrap output in <think> — a search
        # query must be the bare text only.
        q = re.sub(r"<think>.*?</think>", "", q, flags=re.DOTALL).strip().strip('"')
        if q:
            return q
    except Exception as e:
        print(f"[webmode] query composer failed: {e}", file=sys.stderr)
    # fallback: raw question + preferences, India-biased
    return " ".join([question] + [p.split(": ", 1)[-1] for p in parts] + ["India college"])[:200]


def web_answer(question, prefs=None):
    t_start = time.perf_counter()
    key = os.getenv("FIRECRAWL_API_KEY", "")
    if not key:
        return {"answer": "Web mode is not configured (set FIRECRAWL_API_KEY in .env).",
                "citations": [], "answered": False,
                "reason_if_unanswered": "FIRECRAWL_API_KEY missing."}

    calls = []
    search_query = _compose_query(question, prefs, calls)
    print(f"[webmode] search query: {search_query!r}", file=sys.stderr)

    data = None
    for attempt in range(2):  # one transparent retry on transient failures
        try:
            resp = requests.post(
                FIRECRAWL_SEARCH,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"query": search_query, "limit": 4},  # snippets only — no scraping
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])
            break
        except Exception as e:
            print(f"[webmode] firecrawl error (attempt {attempt + 1}): {e}", file=sys.stderr)
            if attempt == 0:
                time.sleep(2)
    if data is None:
        return {"answer": "Web search failed — please try again in a moment.",
                "citations": [], "answered": False,
                "reason_if_unanswered": "Firecrawl request failed after retry."}

    # v2 nests results under data.web; v1 returned a flat list. Accept both.
    results = data.get("web", []) if isinstance(data, dict) else data
    sources = []
    for item in results or []:
        url = item.get("url")
        title = (item.get("title") or "").strip()
        snippet = (item.get("description") or item.get("markdown") or "").strip()
        if url and (title or snippet):
            sources.append({"url": url, "text": f"{title}\n{snippet[:600]}"})
        if len(sources) >= 4:
            break

    if not sources:
        return {"answer": "The web search returned no usable results for this question.",
                "citations": [], "answered": False,
                "reason_if_unanswered": "No web results."}

    context = "\n\n".join(f"[{i + 1}] {s['url']}\n{s['text']}" for i, s in enumerate(sources))
    prefs_line = ""
    stated = {k: str(v).strip() for k, v in (prefs or {}).items()
              if k in ("course", "budget", "location", "type")
              and isinstance(v, (str, int, float)) and str(v).strip()}
    if stated:
        prefs_line = f"USER PREFERENCES (hard constraints): {stated}\n"

    try:
        raw = chat(
            [{"role": "system", "content": config.GEN_SYSTEM_PREFIX + WEB_SYSTEM},
             {"role": "user",
              "content": f"SEARCH RESULTS:\n{context}\n\n{prefs_line}QUESTION: {question}"}],
            model=config.GEN_MODEL, json_mode=True, temperature=0.2,
            max_tokens=700, usage_log=calls,
        )
        result = _parse_json(raw)
        result = {
            "answer": _format_answer(str(result.get("answer", "")).replace("**", "")),
            "citations": [c for c in (result.get("citations") or []) if isinstance(c, str)],
            "answered": bool(result.get("answered")),
            "reason_if_unanswered": result.get("reason_if_unanswered"),
        }
    except Exception as e:
        print(f"[webmode] generation error: {e}", file=sys.stderr)
        result = {"answer": "Could not generate a reliable web answer.",
                  "citations": [], "answered": False,
                  "reason_if_unanswered": "Web generation failed."}

    _log_metrics({"question": question, "route": "web_mode", "calls": calls,
                  "retrieved": [s["url"] for s in sources], "verified": False,
                  "total_latency_s": round(time.perf_counter() - t_start, 3)})
    return result
