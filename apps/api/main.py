"""
FastAPI main application for AI Sentiment Analysis
Production-grade sentiment analysis API with Vietnamese language focus
"""

from contextlib import asynccontextmanager
from typing import Dict, Any
import logging
import time

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from apps.api.routers import health, sentiment
from core.config.base import get_settings
from core.telemetry.logging import setup_logging
from core.models.base import ModelManager

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
SENTIMENT_REQUESTS = Counter('sentiment_requests_total', 'Total sentiment analysis requests', ['model', 'language'])

# Global settings
settings = get_settings()

# Global model manager
model_manager = ModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting AI Sentiment Analysis API...")

    try:
        # Initialize model manager
        await model_manager.initialize()
        logger.info(f"Model manager initialized with backend: {settings.MODEL_BACKEND}")

        # Initialize sentiment service
        from apps.api.services.sentiment_service import SentimentService
        from apps.api.deps.sentiment import set_sentiment_service

        sentiment_service = SentimentService()
        await sentiment_service.initialize(model_manager)
        set_sentiment_service(sentiment_service)
        logger.info("Sentiment service initialized")

        # Warm up models if enabled
        if settings.PRELOAD_MODELS:
            await model_manager.warmup()
            logger.info("Models warmed up successfully")

    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down AI Sentiment Analysis API...")
    await model_manager.cleanup()

# Create FastAPI app
app = FastAPI(
    title="AI Sentiment Analysis API",
    description="Production-grade sentiment analysis for Vietnamese and multi-language text",
    version="1.0.0",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.DEBUG else ["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*", "localhost", "127.0.0.1", "[::1]"]  # [::1] là IPv6 localhost
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Middleware to collect metrics"""
    start_time = time.time()

    response = await call_next(request)

    # Record metrics
    duration = time.time() - start_time
    REQUEST_DURATION.observe(duration)
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()

    return response

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Simple rate limiting middleware"""
    if not settings.RATE_LIMIT_ENABLED:
        return await call_next(request)

    # TODO: Implement Redis-based rate limiting
    # For now, just pass through
    return await call_next(request)

# Include routers
app.include_router(health.router, prefix="", tags=["Health"])
app.include_router(sentiment.router, prefix="/v1", tags=["Sentiment Analysis"])

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    if not settings.METRICS_ENABLED:
        raise HTTPException(status_code=404, detail="Metrics disabled")

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Global exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.DEBUG else "An unexpected error occurred"
        }
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Sentiment Analysis API",
        "version": "1.0.0",
        "docs": "/docs" if settings.DEBUG else "Contact admin for documentation",
        "health": "/healthz"
    }

# Make model manager available to routes
app.state.model_manager = model_manager

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
