# redis caching module for search system
# handles connection, serialization, and graceful fallback

import json
import hashlib
import os
from typing import Any, Optional

import numpy as np

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class RedisCache:
    
    def __init__(self):
        self.client = None
        self.enabled = False
        
        if not REDIS_AVAILABLE:
            print("CACHE: redis package not installed, caching disabled")
            return
        
        # support both REDIS_URL (Upstash/cloud) and REDIS_HOST/PORT (local)
        redis_url = os.getenv("REDIS_URL")
        
        try:
            if redis_url:
                # cloud Redis (Upstash, etc.) - URL format: redis://user:pass@host:port
                self.client = redis.from_url(
                    redis_url,
                    decode_responses=True,
                    socket_connect_timeout=5
                )
                self.client.ping()
                self.enabled = True
                print("CACHE: connected to Redis (cloud)")
            else:
                # local Redis
                host = os.getenv("REDIS_HOST", "localhost")
                port = int(os.getenv("REDIS_PORT", "6379"))
                
                self.client = redis.Redis(
                    host=host,
                    port=port,
                    decode_responses=True,
                    socket_connect_timeout=2
                )
                self.client.ping()
                self.enabled = True
                print(f"CACHE: connected to Redis at {host}:{port}")
        except Exception as e:
            print(f"CACHE: redis connection failed ({e}), caching disabled")
            self.client = None
    
    def _make_key(self, key: str) -> str:
        """Create SHA256 hash of key for consistent storage."""
        return hashlib.sha256(key.encode()).hexdigest()
    
    def _serialize(self, value: Any) -> str:
        """Convert value to JSON string, handling numpy arrays."""
        if isinstance(value, np.ndarray):
            value = value.tolist()
        return json.dumps(value)
    
    def _deserialize(self, data: str) -> Any:
        """Convert JSON string back to Python object."""
        return json.loads(data)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache. Returns None if not found or cache disabled."""
        if not self.enabled:
            return None
        
        try:
            hashed_key = self._make_key(key)
            data = self.client.get(hashed_key)
            if data is None:
                return None
            return self._deserialize(data)
        except Exception as e:
            print(f"CACHE: get error ({e})")
            return None
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache with TTL (seconds). Returns success status."""
        if not self.enabled:
            return False
        
        try:
            hashed_key = self._make_key(key)
            serialized = self._serialize(value)
            self.client.setex(hashed_key, ttl, serialized)
            return True
        except Exception as e:
            print(f"CACHE: set error ({e})")
            return False
