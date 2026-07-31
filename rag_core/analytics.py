import json
import time
from pathlib import Path
from collections import defaultdict
from . import config
from .llm import chat

def generate_daily_report() -> dict:
    """Analyze the previous day's metrics, reasoning evals, and feedback.
    Returns a structured summary dictionary."""
    metrics_file = config.DATA_DIR / "metrics" / "metrics.jsonl"
    evals_file = config.DATA_DIR / "metrics" / "reasoning_evals.jsonl"
    feedback_file = config.DATA_DIR / "metrics" / "feedback.jsonl"
    
    total_queries = 0
    total_conf = 0
    verification_failures = 0
    
    # Read metrics
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                total_queries += 1
                total_conf += data.get("confidence_score", 0)
                if not data.get("verified", True):
                    verification_failures += 1
                    
    # Read reasoning evals
    retrieval_failures = 0
    topics_failed = defaultdict(int)
    if evals_file.exists():
        with open(evals_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                eval_res = json.loads(line)
                if not eval_res.get("answered_question", True):
                    retrieval_failures += 1
                    topics_failed["General (Fallback)"] += 1

    # In a real system, we'd use the FAST_MODEL to categorize the top failed topics here.
    # For now, we simulate extraction from the reasoning failures.
    top_failed_topic = "Unknown"
    if topics_failed:
        top_failed_topic = max(topics_failed.items(), key=lambda x: x[1])[0]

    avg_conf = round(total_conf / total_queries) if total_queries else 0
    retrieval_err_pct = round((retrieval_failures / total_queries) * 100, 2) if total_queries else 0.0
    verif_err_pct = round((verification_failures / total_queries) * 100, 2) if total_queries else 0.0

    report = {
        "timestamp": time.time(),
        "total_queries": total_queries,
        "wrong_retrieval_pct": retrieval_err_pct,
        "wrong_verification_pct": verif_err_pct,
        "average_confidence": avg_conf,
        "top_failed_topic": top_failed_topic
    }
    
    # Optionally save this to a report file
    report_file = config.DATA_DIR / "metrics" / "daily_reports.jsonl"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(report, ensure_ascii=False) + "\n")
        
    return report

def categorize_failure(question: str, answer: str, comment: str):
    """Async task to categorize negative user feedback."""
    try:
        prompt = (f"User Question: {question}\nSystem Answer: {answer}\nUser Comment: {comment}\n"
                  f"Categorize this failure into one of: 'Hallucination', 'Missing Data', 'Bad Formatting', 'Irrelevant', 'Other'.\n"
                  f"Return JSON with 'category' and 'reason'.")
        raw = chat([{"role": "system", "content": "You are a QA analyst."},
                    {"role": "user", "content": prompt}], 
                   model=config.FAST_MODEL, json_mode=True, max_tokens=150)
        
        result = json.loads(raw)
        result["question"] = question
        result["timestamp"] = time.time()
        
        log_file = config.DATA_DIR / "metrics" / "categorized_failures.jsonl"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    except Exception as e:
        import sys
        print(f"[categorize_failure] failed: {e}", file=sys.stderr)

