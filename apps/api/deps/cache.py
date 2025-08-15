"""
Cache dependencies for AI Sentiment Analysis API
"""

import logging
from typing import Dict, Any, Optional
import json

from redis import asyncio as aioredis  # Thay import
from redis.asyncio.client import Redis

from core.config.base import get_settings

logger = logging.getLogger(__name__)

# Global Redis connection
_redis_client: Optional[Redis] = None


async def get_redis_client() -> Redis:
    """Get Redis async client"""
    global _redis_client
    if _redis_client is None:
        settings = get_settings()
        _redis_client = aioredis.from_url(
            settings.REDIS_URL,
            password=settings.REDIS_PASSWORD,
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5,
            health_check_interval=30
        )
    return _redis_client


async def get_cache_health() -> Dict[str, Any]:
    """Check Redis cache health"""
    try:
        client = await get_redis_client()
        # Test basic operations
        await client.ping()

        # Test set/get
        test_key = "health_check_test"
        await client.set(test_key, "test_value", ex=10)
        test_value = await client.get(test_key)
        await client.delete(test_key)

        if test_value == "test_value":
            return {"redis": {"status": "healthy"}}
        else:
            return {"redis": {"status": "unhealthy", "error": "Set/get test failed"}}

    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return {"redis": {"status": "unhealthy", "error": str(e)}}


async def close_cache_connections():
    """Close Redis connections"""
    global _redis_client
    if _redis_client:
        await _redis_client.close()
        _redis_client = None


# FastAPI dependency
async def get_cache_client():
    """FastAPI dependency for Redis client"""
    settings = get_settings()
    if not settings.CACHE_ENABLED:
        return None

    try:
        return await get_redis_client()
    except Exception as e:
        logger.warning(f"Failed to get cache client: {e}")
        return None


class CacheManager:
    """High-level cache operations"""

    def __init__(self, redis_client: Redis):
        self.redis = redis_client

    async def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        """Get JSON data from cache"""
        try:
            data = await self.redis.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.warning(f"Cache get_json failed for key {key}: {e}")
            return None

    async def set_json(self, key: str, data: Dict[str, Any], ttl: int = 600) -> bool:
        """Set JSON data to cache"""
        try:
            await self.redis.set(key, json.dumps(data), ex=ttl)
            return True
        except Exception as e:
            logger.warning(f"Cache set_json failed for key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            await self.redis.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Cache delete failed for key {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        try:
            return bool(await self.redis.exists(key))
        except Exception as e:
            logger.warning(f"Cache exists check failed for key {key}: {e}")
            return False
