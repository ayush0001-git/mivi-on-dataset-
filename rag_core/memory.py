"""Memory-RAG: what MIVI remembers about a student, and how it stays useful.

Three layers, because they fail differently:

  1. RECENT TURNS — the last few messages, sent verbatim. Cheap, exact, and the
     only thing that resolves "the second one" or "what about its hostel?".
  2. ROLLING SUMMARY — everything older, compressed once and carried forward.
     The prompt window is finite; truncation silently DROPS turn 2, and turn 2
     is where the student said "I got 62% and my budget is 80 thousand". This
     layer is why the counsellor does not re-ask what it was already told.
  3. DURABLE PROFILE — the structured facts (marks, budget, stream, location)
     in the store, not in process memory. Survives a restart, and follows the
     student across app servers.

Why a profile AND a summary: the profile is what RETRIEVAL filters on, so it
has to be typed and exact; the summary is what the GENERATOR reads, so it has
to be prose. Deriving one from the other at request time would mean an LLM
parsing "80 thousand" into a filter on every single turn — which is both slower
and less reliable than storing the number once.

The cost discipline that matters: summarisation is an LLM call, so it does NOT
run per turn. It runs when the transcript outgrows the window (every ~6 turns),
and never on the request path of a question it would not change.
"""
from __future__ import annotations

import json
import sys
import threading
from datetime import datetime, timezone

from . import config
from .store.backend import get_backend

# Turns sent verbatim. Beyond this the tail is folded into the summary. Six is
# three exchanges — enough for "the second one" and follow-ups to resolve,
# small enough that the prompt stays mostly about the current question.
RECENT_TURNS = 6

# Summarise only once the transcript is meaningfully past the window; doing it
# at exactly RECENT_TURNS would re-summarise on almost every turn for no gain.
SUMMARISE_AFTER = RECENT_TURNS + 4

# Filter-relevant profile keys. Deliberately closed: an open dict would let the
# router invent keys that retrieval silently ignores, which looks like memory
# loss to the student and is invisible in the logs.
PROFILE_KEYS = (
    "marks_pct", "budget", "field_interest", "location", "state", "city",
    "college_type", "program_level", "category", "entrance_exam", "hostel_required",
)

_lock = threading.Lock()


def _now() -> datetime:
    # created_at/updated_at are TIMESTAMPTZ, so hand Postgres a real datetime
    # rather than a string it has to infer a type for. Storing them as real
    # timestamps is what makes "profiles idle for 30 days" a plain SQL
    # predicate instead of a parse-every-row loop.
    #
    # Measured, and it matters for the created_at/updated_at pair: on Python
    # 3.12/Windows this clock only advances every ~15.6 ms, so two writes in the
    # same tick get the SAME timestamp. Nothing here may treat equal timestamps
    # as "did not happen".
    return datetime.now(timezone.utc)


def _loads(v, default):
    if isinstance(v, (dict, list)):
        return v
    if not v:
        return default
    try:
        return json.loads(v)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def load(session_id: str) -> dict:
    """{'profile': {...}, 'summary': str, 'n_turns': int} — never raises.

    Memory is an enhancement, not a dependency: if the store is unreachable the
    counsellor must still answer the question in front of it, just without the
    history. Failing the request instead would trade a degraded answer for no
    answer at all.
    """
    if not session_id:
        return {"profile": {}, "summary": "", "n_turns": 0}
    try:
        rows = get_backend().q(
            "SELECT profile, summary, n_turns FROM student_profile WHERE session_id = ?",
            [session_id],
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[memory] load failed: {exc}", file=sys.stderr)
        return {"profile": {}, "summary": "", "n_turns": 0}
    if not rows:
        return {"profile": {}, "summary": "", "n_turns": 0}
    r = rows[0]
    return {
        "profile": _loads(r.get("profile"), {}),
        "summary": r.get("summary") or "",
        "n_turns": int(r.get("n_turns") or 0),
    }


def recent_turns(session_id: str, limit: int = RECENT_TURNS) -> list[dict]:
    """The last `limit` messages, oldest-first (the order a transcript reads)."""
    if not session_id:
        return []
    try:
        rows = get_backend().q(
            "SELECT role, content, turn_no FROM conversation_turn "
            "WHERE session_id = ? ORDER BY turn_no DESC, role DESC LIMIT ?",
            [session_id, limit],
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[memory] recent_turns failed: {exc}", file=sys.stderr)
        return []
    # Newest-first in SQL so LIMIT keeps the RECENT turns, then back into reading
    # order here. The ordering is restated instead of a plain reversed() because
    # the two keys do not mirror each other: 'assistant' < 'user' alphabetically,
    # so reversing the SQL order handed the generator the counsellor's reply
    # BEFORE the question it answered. The `role DESC` above is still right for
    # the other half of the job — when the limit splits a turn it keeps the
    # student's question, which is where the facts are, over the reply.
    rows.sort(key=lambda r: (r["turn_no"], 0 if r["role"] == "user" else 1))
    return [{"role": r["role"], "content": r["content"]} for r in rows]


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def merge_profile(old: dict, new: dict) -> dict:
    """Later statements win; a null NEVER erases a known fact.

    This asymmetry is the whole point. The router emits the full profile shape
    every turn and fills only what the CURRENT message mentions, so a plain
    dict.update() would wipe the student's marks the moment they asked an
    unrelated question — the exact failure that makes an assistant feel
    forgetful.
    """
    out = dict(old or {})
    for k, v in (new or {}).items():
        if k not in PROFILE_KEYS:
            continue
        if v is None or (isinstance(v, str) and not v.strip()):
            continue
        out[k] = v
    return out


def save(session_id: str, profile: dict | None = None, summary: str | None = None,
         bump_turns: int = 0) -> None:
    """Upsert the profile. Best-effort: memory must never fail a request.

    The read-merge-write is not a transaction, and deliberately so: the loser
    of a race writes a profile that is merged from a slightly older base, and
    because merge_profile only ever ADDS facts the worst case is one turn's
    fact arriving a turn late. Taking a row lock here would put a Postgres
    round trip on the critical path of every reply to protect against that.
    """
    if not session_id:
        return
    now = _now()
    try:
        with _lock:
            cur = load(session_id)
            merged = merge_profile(cur["profile"], profile or {})
            new_summary = summary if summary is not None else cur["summary"]
            n = cur["n_turns"] + max(0, bump_turns)
            # created_at is absent from the DO UPDATE list on purpose — it
            # records when this student was first seen, and re-stamping it on
            # every turn would erase that.
            _write("INSERT INTO student_profile"
                   "(session_id, profile, summary, n_turns, created_at, updated_at) "
                   "VALUES (?, ?, ?, ?, ?, ?) "
                   "ON CONFLICT (session_id) DO UPDATE SET "
                   "profile = EXCLUDED.profile, summary = EXCLUDED.summary, "
                   "n_turns = EXCLUDED.n_turns, updated_at = EXCLUDED.updated_at",
                   [session_id, json.dumps(merged, ensure_ascii=False),
                    new_summary, n, now, now])
    except Exception as exc:  # noqa: BLE001
        print(f"[memory] save failed: {exc}", file=sys.stderr)


def record_turn(session_id: str, turn_no: int, role: str, content: str,
                citations: list | None = None) -> None:
    """Append one message. Upsert rather than insert because a retried request
    replays the same (session, turn, role) and a duplicate transcript row would
    be summarised twice."""
    if not session_id or not content:
        return
    try:
        # created_at stays out of the DO UPDATE list for the same reason it does
        # in save(): it timestamps when the message was first recorded, and a
        # retry re-stamping it would reorder the transcript against itself.
        _write("INSERT INTO conversation_turn"
               "(session_id, turn_no, role, content, citations, created_at) "
               "VALUES (?, ?, ?, ?, ?, ?) "
               "ON CONFLICT (session_id, turn_no, role) DO UPDATE SET "
               "content = EXCLUDED.content, citations = EXCLUDED.citations",
               [session_id, int(turn_no), role, content[:8000],
                json.dumps(citations or [], ensure_ascii=False), _now()])
    except Exception as exc:  # noqa: BLE001
        print(f"[memory] record_turn failed: {exc}", file=sys.stderr)


def _write(sql: str, params: list) -> int:
    """Run one statement through the shared pool. Returns rows affected.

    The pool, not a psycopg.connect() per write: Postgres forks a backend
    process per connection, so opening one would cost more to set up than the
    UPSERT it carries — and a chat writes twice per turn (profile + turn), on
    the reply path, while the harvest sweep is writing too.

    backend.execute() rather than reaching for the pool directly: it commits on
    success and rolls back if the statement raises, which is the whole
    transaction a single-row upsert needs, and it keeps this file on ONE
    placeholder convention — the codebase's `?`, rewritten for psycopg inside
    the backend — for reads and writes alike.
    """
    return get_backend().execute(sql, params)


# ---------------------------------------------------------------------------
# Summarisation
# ---------------------------------------------------------------------------

SUMMARY_SYSTEM = """You maintain the running memory of a college-counselling chat.

You get the PREVIOUS MEMORY and the OLDER TURNS that are about to fall out of
the visible window. Fold them into one replacement memory.

Rules:
- Keep every concrete fact ABOUT THE STUDENT: marks, budget (with its unit),
  stream/course interest, preferred state or city, category, exams taken or
  planned, family constraints, and what they have already ruled out.
- Keep which colleges were already SUGGESTED and the student's reaction, so the
  counsellor never re-recommends something they rejected.
- Drop pleasantries, restated questions, and anything the counsellor said that
  carried no fact.
- Write compact third-person notes, not dialogue. Under 140 words.
- Never invent a fact the turns do not contain. If nothing is worth keeping,
  return the previous memory unchanged."""


def should_summarise(n_turns: int) -> bool:
    return n_turns >= SUMMARISE_AFTER and n_turns % RECENT_TURNS == 0


def summarise(session_id: str, older_turns: list[dict], previous: str = "") -> str:
    """Compress the turns falling out of the window. Returns the new memory,
    or `previous` unchanged on any failure — a lost summary must degrade the
    conversation, never break it."""
    if not older_turns:
        return previous
    from .llm import chat

    convo = "\n".join(
        f"{'Student' if t.get('role') == 'user' else 'Counsellor'}: "
        f"{str(t.get('content'))[:400]}"
        for t in older_turns if t.get("content")
    )
    try:
        out = chat(
            [{"role": "system", "content": SUMMARY_SYSTEM},
             {"role": "user",
              "content": f"PREVIOUS MEMORY:\n{previous or '(none)'}\n\n"
                         f"OLDER TURNS:\n{convo}"}],
            model=config.FAST_MODEL, temperature=0.0, max_tokens=300,
        )
        out = (out or "").strip()
        return out[:1200] if out else previous
    except Exception as exc:  # noqa: BLE001
        print(f"[memory] summarise failed: {exc}", file=sys.stderr)
        return previous


def as_prompt_block(mem: dict) -> str:
    """Render memory for the generator. Empty string when there is nothing to
    say — an empty 'WHAT I KNOW: (nothing)' block teaches the model that not
    knowing is normal, and it starts hedging answers it should give plainly."""
    profile = {k: v for k, v in (mem.get("profile") or {}).items() if v not in (None, "", [])}
    summary = (mem.get("summary") or "").strip()
    if not profile and not summary:
        return ""
    parts = []
    if profile:
        pretty = ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in profile.items())
        parts.append(f"WHAT YOU ALREADY KNOW ABOUT THIS STUDENT: {pretty}. "
                     f"Never ask them to repeat any of it.")
    if summary:
        parts.append(f"EARLIER IN THIS CONVERSATION: {summary}")
    return "\n".join(parts) + "\n\n"
