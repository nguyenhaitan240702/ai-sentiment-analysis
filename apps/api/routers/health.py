"""
Health check router for AI Sentiment Analysis API
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any
import logging
import asyncio

from apps.api.deps.db import get_database_health
from apps.api.deps.cache import get_cache_health
from core.config.base import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/healthz")
async def health_check() -> Dict[str, Any]:
    """
    Comprehensive health check endpoint
    Returns system status and component health
    """
    settings = get_settings()
    
    health_status = {
        "status": "healthy",
        "version": "1.0.0",
        "environment": settings.ENV,
        "timestamp": str(asyncio.get_event_loop().time()),
        "checks": {}
    }
    
    # Check database health
    try:
        db_health = await get_database_health()
        health_status["checks"]["database"] = db_health
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        health_status["checks"]["database"] = {
            "mysql": {"status": "unhealthy", "error": str(e)},
            "mongodb": {"status": "unhealthy", "error": str(e)}
        }
        health_status["status"] = "unhealthy"
    
    # Check cache health
    try:
        cache_health = await get_cache_health()
        health_status["checks"]["cache"] = cache_health
    except Exception as e:
        logger.error(f"Cache health check failed: {e}")
        health_status["checks"]["cache"] = {
            "redis": {"status": "unhealthy", "error": str(e)}
        }
        health_status["status"] = "unhealthy"
    
    # Check model health
    try:
        # This will be implemented when we create the model manager
        health_status["checks"]["models"] = {
            "backend": settings.MODEL_BACKEND,
            "status": "healthy"
        }
    except Exception as e:
        logger.error(f"Model health check failed: {e}")
        health_status["checks"]["models"] = {
            "status": "unhealthy", 
            "error": str(e)
        }
        health_status["status"] = "unhealthy"
    
    # Return appropriate status code
    if health_status["status"] == "unhealthy":
        raise HTTPException(status_code=503, detail=health_status)
    
    return health_status

@router.get("/ready")
async def readiness_check() -> Dict[str, str]:
    """
    Kubernetes readiness probe endpoint
    Simple check for service readiness
    """
    return {"status": "ready"}

@router.get("/live")
async def liveness_check() -> Dict[str, str]:
    """
    Kubernetes liveness probe endpoint
    Simple check that service is alive
    """
    return {"status": "alive"}
