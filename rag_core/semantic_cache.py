import hashlib
import json
import time
import threading
from collections import OrderedDict

import numpy as np
from . import retrieve

class SemanticCache:
    """Cache keyed on (normalized question, profile) embedding similarity.
    
    Hits when cosine similarity > threshold.
    """
    def __init__(self, ttl_s: int = 900, max_entries: int = 2000, threshold: float = 0.92):
        self.ttl = ttl_s
        self.max = max_entries
        self.threshold = threshold
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, tuple[float, np.ndarray, dict]] = OrderedDict()
        self.hits = 0
        self.misses = 0
    
    def _embed(self, question: str, profile: dict) -> np.ndarray:
        """Embed question + filter-relevant profile."""
        from . import service
        # Use the same key derivation as service.answer_cache_key
        relevant = {k: v for k, v in (profile or {}).items()
                    if k in service.CACHE_KEY_PROFILE_FIELDS and v not in (None, "", [])}
        text = service.normalise_question(question) + "|" + json.dumps(relevant, sort_keys=True)
        return retrieve.encode_query(text)
    
    def get(self, question: str, profile: dict) -> dict | None:
        if not question:
            return None
        try:
            qv = self._embed(question, profile)
        except Exception:
            return None
        
        now = time.monotonic()
        with self._lock:
            best_sim = 0.0
            best_key = None
            best_value = None
            for key, (stored_at, stored_vec, value) in self._entries.items():
                if now - stored_at > self.ttl:
                    continue
                sim = float(np.dot(qv, stored_vec))
                if sim > best_sim:
                    best_sim = sim
                    best_key = key
                    best_value = value
            
            if best_sim >= self.threshold and best_value:
                self.hits += 1
                # Update LRU
                self._entries.move_to_end(best_key)
                return json.loads(json.dumps(best_value))
        
        self.misses += 1
        return None
    
    def put(self, question: str, profile: dict, value: dict) -> None:
        try:
            qv = self._embed(question, profile)
        except Exception:
            return
        
        with self._lock:
            # Use a stable key (the embedding itself) for storage
            key = qv.tobytes()[:32].hex()
            self._entries[key] = (time.monotonic(), qv, json.loads(json.dumps(value)))
            self._entries.move_to_end(key)
            while len(self._entries) > self.max:
                self._entries.popitem(last=False)
    
    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "entries": len(self._entries),
            "capacity": self.max,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
            "threshold": self.threshold
        }
