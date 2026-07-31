import json
import time
from pathlib import Path
from . import config
from .llm import chat
from .store.backend import get_backend

def detect_and_update_stale_data(college_id: str, current_card: str, web_text: str):
    """
    Simulates the Data Freshness loop:
    1. Compares current DB card with freshly crawled web_text.
    2. Uses FAST_MODEL to extract diffs.
    3. Flags for human review if significant drift is found.
    """
    try:
        prompt = (f"Compare this Database record against the Live Web scrape.\n"
                  f"--- DB Record ---\n{current_card}\n"
                  f"--- Live Web ---\n{web_text[:3000]}\n\n"
                  f"Identify any critical facts (fees, rankings, placement stats) that are OUTDATED in the DB.\n"
                  f"Return JSON with 'is_stale': bool, 'outdated_fields': list, 'suggested_updates': dict")
        
        raw = chat([{"role": "system", "content": "You are a data freshness verification bot."},
                    {"role": "user", "content": prompt}], 
                   model=config.FAST_MODEL, json_mode=True, max_tokens=300)
                   
        result = json.loads(raw)
        
        if result.get("is_stale"):
            # Queue for human review rather than blindly overwriting the DB
            review_item = {
                "college_id": college_id,
                "timestamp": time.time(),
                "suggested_updates": result.get("suggested_updates"),
                "reason": result.get("outdated_fields"),
                "status": "pending_review"
            }
            
            queue_file = config.DATA_DIR / "admin" / "freshness_review_queue.jsonl"
            queue_file.parent.mkdir(parents=True, exist_ok=True)
            with open(queue_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(review_item, ensure_ascii=False) + "\n")
                
        return result
    except Exception as e:
        import sys
        print(f"[freshness_check] failed for {college_id}: {e}", file=sys.stderr)
        return {"is_stale": False}
