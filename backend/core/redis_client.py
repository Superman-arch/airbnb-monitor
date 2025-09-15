"""
Redis cache client and management
"""

import json
import pickle
from typing import Any, Optional, Union
from datetime import timedelta

import structlog
import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool

from backend.core.config import settings

logger = structlog.get_logger()

# Global Redis client
redis_client: redis.Redis = None
connection_pool: ConnectionPool = None


async def init_redis():
    """
    Initialize Redis connection
    """
    global redis_client, connection_pool
    
    try:
        # Parse Redis URL
        url_parts = settings.REDIS_URL.split('/')
        db_num = int(url_parts[-1]) if len(url_parts) > 3 else 0
        
        # Create connection pool
        connection_pool = ConnectionPool(
            host=settings.REDIS_URL.split('//')[1].split(':')[0] if '//' in settings.REDIS_URL else 'localhost',
            port=int(settings.REDIS_URL.split(':')[-1].split('/')[0]) if ':' in settings.REDIS_URL else 6379,
            db=db_num,
            password=settings.REDIS_PASSWORD,
            max_connections=settings.REDIS_MAX_CONNECTIONS,
            decode_responses=settings.REDIS_DECODE_RESPONSES,
            socket_keepalive=True,
            socket_keepalive_options={
                1: 1,  # TCP_KEEPIDLE
                2: 1,  # TCP_KEEPINTVL
                3: 5,  # TCP_KEEPCNT
            }
        )
        
        # Create Redis client
        redis_client = redis.Redis(
            connection_pool=connection_pool,
            decode_responses=False  # We'll handle encoding/decoding ourselves
        )
        
        # Test connection
        await redis_client.ping()
        
        logger.info("Redis initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize Redis", error=str(e))
        # Don't raise - allow system to work without cache
        redis_client = None


async def close_redis():
    """
    Close Redis connection
    """
    global redis_client, connection_pool
    
    if redis_client:
        await redis_client.close()
        redis_client = None
    
    if connection_pool:
        await connection_pool.disconnect()
        connection_pool = None
    
    logger.info("Redis connection closed")


class Cache:
    """
    Cache operations wrapper
    """
    
    @staticmethod
    async def get(key: str, default: Any = None) -> Any:
        """
        Get value from cache
        """
        if not redis_client:
            return default
        
        try:
            value = await redis_client.get(key)
            if value is None:
                return default
            
            # Try to deserialize
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                try:
                    return pickle.loads(value)
                except:
                    return value.decode('utf-8') if isinstance(value, bytes) else value
                    
        except Exception as e:
            logger.error(f"Cache get error", key=key, error=str(e))
            return default
    
    @staticmethod
    async def set(key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache
        """
        if not redis_client:
            return False
        
        try:
            # Serialize value
            if isinstance(value, (dict, list)):
                serialized = json.dumps(value)
            elif isinstance(value, (str, int, float, bool)):
                serialized = str(value)
            else:
                serialized = pickle.dumps(value)
            
            # Set with optional TTL
            if ttl:
                await redis_client.setex(key, ttl, serialized)
            else:
                await redis_client.set(key, serialized)
            
            return True
            
        except Exception as e:
            logger.error(f"Cache set error", key=key, error=str(e))
            return False
    
    @staticmethod
    async def delete(key: str) -> bool:
        """
        Delete value from cache
        """
        if not redis_client:
            return False
        
        try:
            await redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Cache delete error", key=key, error=str(e))
            return False
    
    @staticmethod
    async def exists(key: str) -> bool:
        """
        Check if key exists
        """
        if not redis_client:
            return False
        
        try:
            return await redis_client.exists(key) > 0
        except Exception as e:
            logger.error(f"Cache exists error", key=key, error=str(e))
            return False
    
    @staticmethod
    async def increment(key: str, amount: int = 1) -> Optional[int]:
        """
        Increment counter
        """
        if not redis_client:
            return None
        
        try:
            return await redis_client.incrby(key, amount)
        except Exception as e:
            logger.error(f"Cache increment error", key=key, error=str(e))
            return None
    
    @staticmethod
    async def expire(key: str, ttl: int) -> bool:
        """
        Set expiration on key
        """
        if not redis_client:
            return False
        
        try:
            await redis_client.expire(key, ttl)
            return True
        except Exception as e:
            logger.error(f"Cache expire error", key=key, error=str(e))
            return False
    
    @staticmethod
    async def get_pattern(pattern: str) -> dict:
        """
        Get all keys matching pattern
        """
        if not redis_client:
            return {}
        
        try:
            keys = await redis_client.keys(pattern)
            result = {}
            
            for key in keys:
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                value = await Cache.get(key_str)
                result[key_str] = value
            
            return result
            
        except Exception as e:
            logger.error(f"Cache get_pattern error", pattern=pattern, error=str(e))
            return {}
    
    @staticmethod
    async def clear_pattern(pattern: str) -> int:
        """
        Delete all keys matching pattern
        """
        if not redis_client:
            return 0
        
        try:
            keys = await redis_client.keys(pattern)
            if keys:
                return await redis_client.delete(*keys)
            return 0
            
        except Exception as e:
            logger.error(f"Cache clear_pattern error", pattern=pattern, error=str(e))
            return 0


# Cache key generators
class CacheKeys:
    """
    Standardized cache key generators
    """
    
    @staticmethod
    def door(door_id: str) -> str:
        return f"door:{door_id}"
    
    @staticmethod
    def person(person_id: str) -> str:
        return f"person:{person_id}"
    
    @staticmethod
    def zone(zone_id: str) -> str:
        return f"zone:{zone_id}"
    
    @staticmethod
    def stats() -> str:
        return "stats:current"
    
    @staticmethod
    def events(page: int = 1) -> str:
        return f"events:page:{page}"
    
    @staticmethod
    def frame(camera_id: str = "main") -> str:
        return f"frame:{camera_id}"
    
    @staticmethod
    def detection_result(frame_id: int) -> str:
        return f"detection:{frame_id}"
    
    @staticmethod
    def user_session(user_id: str) -> str:
        return f"session:{user_id}"
    
    @staticmethod
    def rate_limit(client_ip: str, endpoint: str) -> str:
        return f"rate_limit:{client_ip}:{endpoint}"


async def health_check() -> bool:
    """
    Check Redis health
    """
    if not redis_client:
        return False
    
    try:
        await redis_client.ping()
        return True
    except Exception as e:
        logger.error(f"Redis health check failed", error=str(e))
        return False