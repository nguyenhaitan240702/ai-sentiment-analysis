"""
Database dependencies for AI Sentiment Analysis API
"""

import logging
from typing import Dict, Any, Optional
import asyncio
from contextlib import asynccontextmanager

import aiomysql
from motor.motor_asyncio import AsyncIOMotorClient
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from core.config.base import get_settings

logger = logging.getLogger(__name__)

# Global database connections
_mysql_engine = None
_mongo_client = None

async def get_mysql_engine():
    """Get MySQL async engine"""
    global _mysql_engine
    if _mysql_engine is None:
        settings = get_settings()
        # Use aiomysql for async operations
        mysql_url = f"mysql+aiomysql://{settings.MYSQL_USER}:{settings.MYSQL_PASSWORD}@{settings.MYSQL_HOST}:{settings.MYSQL_PORT}/{settings.MYSQL_DB}"
        _mysql_engine = create_async_engine(
            mysql_url,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=settings.DEBUG
        )
    return _mysql_engine

async def get_mongo_client():
    """Get MongoDB async client"""
    global _mongo_client
    if _mongo_client is None:
        settings = get_settings()
        _mongo_client = AsyncIOMotorClient(
            settings.MONGO_URI,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            socketTimeoutMS=5000
        )
    return _mongo_client

async def get_mongo_database():
    """Get MongoDB database"""
    settings = get_settings()
    client = await get_mongo_client()
    return client[settings.MONGO_DB]

async def get_database_health() -> Dict[str, Any]:
    """Check database health status"""
    health = {
        "mysql": {"status": "unknown"},
        "mongodb": {"status": "unknown"}
    }

    # Check MySQL
    try:
        engine = await get_mysql_engine()
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT 1"))
            if result.fetchone():
                health["mysql"] = {"status": "healthy"}
    except Exception as e:
        logger.error(f"MySQL health check failed: {e}")
        health["mysql"] = {"status": "unhealthy", "error": str(e)}

    # Check MongoDB
    try:
        client = await get_mongo_client()
        # Ping the server
        await client.admin.command('ping')
        health["mongodb"] = {"status": "healthy"}
    except Exception as e:
        logger.error(f"MongoDB health check failed: {e}")
        health["mongodb"] = {"status": "unhealthy", "error": str(e)}

    return health

async def close_database_connections():
    """Close all database connections"""
    global _mysql_engine, _mongo_client

    if _mysql_engine:
        await _mysql_engine.dispose()
        _mysql_engine = None

    if _mongo_client:
        _mongo_client.close()
        _mongo_client = None

# Dependency functions for FastAPI
async def get_mysql_session():
    """FastAPI dependency for MySQL session"""
    engine = await get_mysql_engine()
    async_session = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()

async def get_mongo_db():
    """FastAPI dependency for MongoDB database"""
    return await get_mongo_database()
