"""CLI entry point.

    python answer.py "Which colleges offer an MBA, and what do they cost?"

Prints exactly one JSON object to stdout:
    {"answer": ..., "citations": [...], "answered": true|false, "reason_if_unanswered": ...}
All diagnostics go to stderr.
"""
import json
import sys

if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to cp1252
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def main():
    if len(sys.argv) < 2 or not " ".join(sys.argv[1:]).strip():
        print('usage: python answer.py "your question here"', file=sys.stderr)
        sys.exit(2)
    question = " ".join(sys.argv[1:]).strip()

    from rag_core.pipeline import answer_question  # after stdout reconfigure

    try:
        result = answer_question(question)
    except Exception as e:
        # The stdout contract (exactly one JSON object) survives any failure
        print(f"[fatal] {type(e).__name__}: {e}", file=sys.stderr)
        result = {"answer": "Sorry — an internal error occurred while answering this question.",
                  "citations": [], "answered": False,
                  "reason_if_unanswered": f"Internal error: {type(e).__name__}"}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
