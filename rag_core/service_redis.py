import json
import time
from typing import Any
import redis

class RedisSessionStore:
    def __init__(self, redis_url: str, ttl_s: int = 1800, max_sessions: int = 20000):
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.ttl = ttl_s
        self.max = max_sessions
    
    def snapshot(self, session_id: str) -> dict:
        """Returns dict with session_id, history, profile, turns, is_new."""
        from .service import new_session_id, valid_session_id
        sid = session_id if valid_session_id(session_id) else new_session_id()
        key = f"session:{sid}"
        data = self.redis.get(key)
        if data:
            d = json.loads(data)
            self.redis.expire(key, self.ttl)  # slide TTL
            return {**d, "session_id": sid, "is_new": False}
        else:
            return {"session_id": sid, "history": [], "profile": {}, 
                    "turns": 0, "is_new": True}
    
    def record_turn(self, session_id, user_message, assistant_message, profile):
        snap = self.snapshot(session_id)
        snap["history"].append({"role": "user", "content": user_message})
        snap["history"].append({"role": "assistant", "content": assistant_message})
        snap["history"] = snap["history"][-16:]  # cap
        snap["profile"] = profile or {}
        snap["turns"] = snap.get("turns", 0) + 1
        self.redis.setex(f"session:{session_id}", self.ttl, json.dumps(snap))
    
    def drop(self, session_id: str) -> bool:
        return self.redis.delete(f"session:{session_id}") > 0
    
    def stats(self) -> dict:
        # Use SCAN to count keys (don't use KEYS in production)
        n = 0
        for _ in self.redis.scan_iter("session:*", count=1000):
            n += 1
        return {"live": n, "capacity": self.max, "backend": "redis"}
