"""
Redis client for session management and caching.
"""

import json
import asyncio
from typing import Optional, Any, Dict
from datetime import timedelta
import redis.asyncio as redis
from loguru import logger


class RedisClient:
    """
    Async Redis client for session management and caching.
    """

    def __init__(self, url: str = "redis://localhost:6379"):
        """
        Initialize Redis client.

        Args:
            url: Redis connection URL
        """
        self.url = url
        self._client: Optional[redis.Redis] = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        """Establish connection to Redis."""
        if self._client is None:
            async with self._lock:
                if self._client is None:
                    try:
                        self._client = await redis.from_url(
                            self.url,
                            encoding="utf-8",
                            decode_responses=True,
                            socket_connect_timeout=5,
                            socket_keepalive=True,
                        )
                        await self._client.ping()
                        logger.info(f"Connected to Redis at {self.url}")
                    except Exception as e:
                        logger.error(f"Failed to connect to Redis: {e}")
                        raise

    async def disconnect(self) -> None:
        """Close connection to Redis."""
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("Disconnected from Redis")

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from Redis.

        Args:
            key: Redis key

        Returns:
            Deserialized value or None
        """
        if not self._client:
            await self.connect()

        try:
            value = await self._client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Redis GET error for key {key}: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        expire: Optional[int] = None
    ) -> bool:
        """
        Set value in Redis.

        Args:
            key: Redis key
            value: Value to store (will be JSON serialized)
            expire: Expiration time in seconds

        Returns:
            True if successful
        """
        if not self._client:
            await self.connect()

        try:
            serialized = json.dumps(value)
            if expire:
                await self._client.setex(key, expire, serialized)
            else:
                await self._client.set(key, serialized)
            return True
        except Exception as e:
            logger.error(f"Redis SET error for key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete key from Redis.

        Args:
            key: Redis key

        Returns:
            True if deleted
        """
        if not self._client:
            await self.connect()

        try:
            result = await self._client.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis DELETE error for key {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """
        Check if key exists.

        Args:
            key: Redis key

        Returns:
            True if exists
        """
        if not self._client:
            await self.connect()

        try:
            result = await self._client.exists(key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis EXISTS error for key {key}: {e}")
            return False

    async def expire(self, key: str, seconds: int) -> bool:
        """
        Set expiration time for key.

        Args:
            key: Redis key
            seconds: Expiration time in seconds

        Returns:
            True if successful
        """
        if not self._client:
            await self.connect()

        try:
            result = await self._client.expire(key, seconds)
            return result
        except Exception as e:
            logger.error(f"Redis EXPIRE error for key {key}: {e}")
            return False

    async def keys(self, pattern: str = "*") -> list[str]:
        """
        Get all keys matching pattern.

        Args:
            pattern: Redis key pattern

        Returns:
            List of matching keys
        """
        if not self._client:
            await self.connect()

        try:
            keys = await self._client.keys(pattern)
            return keys
        except Exception as e:
            logger.error(f"Redis KEYS error for pattern {pattern}: {e}")
            return []

    async def ping(self) -> bool:
        """
        Check Redis connection.

        Returns:
            True if connected
        """
        if not self._client:
            try:
                await self.connect()
            except:
                return False

        try:
            await self._client.ping()
            return True
        except:
            return False


# Global Redis client instance
_redis_client: Optional[RedisClient] = None


def get_redis_client(url: Optional[str] = None) -> RedisClient:
    """
    Get global Redis client instance (singleton).

    Args:
        url: Redis URL (only used for first call)

    Returns:
        RedisClient instance
    """
    global _redis_client
    if _redis_client is None:
        from config.settings import get_settings
        settings = get_settings()
        _redis_client = RedisClient(url or settings.redis_url)
    return _redis_client
