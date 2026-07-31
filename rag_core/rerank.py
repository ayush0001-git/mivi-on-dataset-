from sentence_transformers import CrossEncoder
import threading

_model = None
_lock = threading.Lock()

def get_reranker():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                _model = CrossEncoder(
                    "BAAI/bge-reranker-v2-m3",  # multilingual
                    max_length=512
                )
    return _model

def rerank(query: str, candidates: list[dict], top_k: int = 12, threshold: float = 0.5,
           complexity_score: int = 5, profile: dict = None) -> list[dict]:
    """Rerank candidates using cross-encoder and user profile. Returns top_k."""
    if not candidates or not query:
        return candidates[:top_k]
        
    # Adaptive execution: simple queries don't need heavy cross-encoder reranking
    if complexity_score <= 3:
        return candidates[:top_k]
    
    # Check confidence: if top score is very high and it's not super complex, skip reranking
    top_score = candidates[0].get("score", 0)
    if complexity_score <= 7 and top_score > threshold:
        return candidates[:top_k]
    
    try:
        reranker = get_reranker()
        pairs = [(query, c.get("card", "")[:1500]) for c in candidates]  # truncate cards
        scores = reranker.predict(pairs)
        
        boost_state = str(profile.get("state", "")).strip().lower() if profile else ""
        
        # Sort by cross-encoder score
        for c, s in zip(candidates, scores):
            ce_score = float(s)
            
            # Apply personalization boost
            if boost_state:
                card_text = c.get("card", "").lower()
                if boost_state in card_text:
                    ce_score += 0.5  # Significant boost for personalized location
                    
            c["ce_score"] = ce_score
            
        candidates.sort(key=lambda c: c.get("ce_score", 0), reverse=True)
        return candidates[:top_k]
    except Exception as e:
        import sys
        print(f"[rerank] failure: {e}. Falling back to original retrieval order.", file=sys.stderr)
        return candidates[:top_k]
