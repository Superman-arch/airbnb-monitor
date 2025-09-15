"""
Database connection and session management
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker
)
from sqlalchemy.orm import declarative_base

from backend.core.config import settings

logger = structlog.get_logger()

# Database engine
engine: AsyncEngine = None
AsyncSessionLocal: async_sessionmaker = None

# Base for models
Base = declarative_base()


async def init_db():
    """
    Initialize database connection
    """
    global engine, AsyncSessionLocal
    
    try:
        # Create async engine with optimized settings
        engine = create_async_engine(
            settings.DATABASE_URL,
            echo=settings.DATABASE_ECHO,
            pool_size=settings.DATABASE_POOL_SIZE,
            max_overflow=settings.DATABASE_MAX_OVERFLOW,
            pool_timeout=settings.DATABASE_POOL_TIMEOUT,
            pool_pre_ping=True,  # Verify connections before using
            poolclass=pool.NullPool if settings.ENVIRONMENT == "development" else pool.AsyncAdaptedQueuePool,
        )
        
        # Create session factory
        AsyncSessionLocal = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False
        )
        
        # Import all models to register them with Base
        from backend.models import door, person, zone  # noqa
        
        # Create tables if they don't exist
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize database", error=str(e))
        raise


async def close_db():
    """
    Close database connection
    """
    global engine
    
    if engine:
        await engine.dispose()
        logger.info("Database connection closed")


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get database session
    """
    if not AsyncSessionLocal:
        await init_db()
    
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for FastAPI to get database session
    """
    async with get_session() as session:
        yield session


async def execute_query(query, *args, **kwargs):
    """
    Execute a database query
    """
    async with get_session() as session:
        result = await session.execute(query, *args, **kwargs)
        return result


async def health_check() -> bool:
    """
    Check database health
    """
    try:
        async with get_session() as session:
            await session.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database health check failed", error=str(e))
        return False