"""
Sentiment analysis router for AI Sentiment Analysis API
"""

from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
from typing import List, Dict, Any, Optional
import logging
import hashlib
import time
import uuid

from apps.api.schemas.request import SentimentRequest, BatchSentimentRequest, ConversationSentimentRequest
from apps.api.schemas.response import SentimentResponse, BatchSentimentResponse, JobResponse, ConversationSentimentResponse
from apps.api.services.sentiment_service import SentimentService
from apps.api.deps.cache import get_cache_client
from apps.api.deps.sentiment import get_sentiment_service
from core.config.base import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(
    request: SentimentRequest,
    sentiment_service: SentimentService = Depends(get_sentiment_service),
    cache_client = Depends(get_cache_client)
) -> SentimentResponse:
    """
    Analyze sentiment of a single text
    Supports Vietnamese and English languages
    """
    settings = get_settings()
    start_time = time.time()
    
    # Generate cache key
    content_hash = hashlib.sha256(request.text.encode()).hexdigest()
    cache_key = f"sentiment:{request.lang or 'auto'}:{content_hash}"
    
    # Check cache first
    cached_result = None
    if settings.CACHE_ENABLED and cache_client:
        try:
            cached_result = await cache_client.get(cache_key)
            if cached_result:
                logger.debug(f"Cache hit for key: {cache_key}")
                result = SentimentResponse.parse_raw(cached_result)
                result.cached = True
                result.latency_ms = int((time.time() - start_time) * 1000)
                return result
        except Exception as e:
            logger.warning(f"Cache get failed: {e}")
    
    # Analyze sentiment
    try:
        result = await sentiment_service.analyze_text(
            text=request.text,
            language=request.lang
        )
        
        # Add metadata
        result.latency_ms = int((time.time() - start_time) * 1000)
        result.cached = False
        
        # Cache result
        if settings.CACHE_ENABLED and cache_client:
            try:
                await cache_client.setex(
                    cache_key,
                    settings.CACHE_TTL,
                    result.json()
                )
            except Exception as e:
                logger.warning(f"Cache set failed: {e}")
        
        return result
        
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Sentiment analysis failed: {str(e)}"
        )

@router.post("/sentiment/batch", response_model=BatchSentimentResponse)
async def analyze_sentiment_batch(
    request: BatchSentimentRequest,
    background_tasks: BackgroundTasks,
    sentiment_service: SentimentService = Depends(get_sentiment_service)
) -> BatchSentimentResponse:
    """
    Analyze sentiment for multiple texts
    Supports both sync and async processing
    """
    settings = get_settings()
    
    # Validate batch size
    if len(request.items) > settings.BATCH_MAX:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(request.items)} exceeds maximum {settings.BATCH_MAX}"
        )
    
    # Async processing
    if request.async_mode:
        job_id = str(uuid.uuid4())
        
        # Queue background task
        background_tasks.add_task(
            sentiment_service.process_batch_async,
            job_id,
            request.items
        )
        
        return BatchSentimentResponse(
            job_id=job_id,
            status="processing",
            results=None,
            total_items=len(request.items)
        )
    
    # Sync processing
    try:
        results = await sentiment_service.analyze_batch(request.items)
        
        return BatchSentimentResponse(
            job_id=None,
            status="completed",
            results=results,
            total_items=len(request.items)
        )
        
    except Exception as e:
        logger.error(f"Batch sentiment analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch processing failed: {str(e)}"
        )

@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(
    job_id: str,
    sentiment_service: SentimentService = Depends(get_sentiment_service)
) -> JobResponse:
    """
    Get status and results of async batch job
    """
    try:
        job_result = await sentiment_service.get_job_result(job_id)
        
        if not job_result:
            raise HTTPException(
                status_code=404,
                detail=f"Job {job_id} not found"
            )
        
        return job_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Job status check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get job status: {str(e)}"
        )

@router.post("/sentiment/conversation", response_model=ConversationSentimentResponse)
async def analyze_conversation_sentiment(
    request: ConversationSentimentRequest,
    sentiment_service: SentimentService = Depends(get_sentiment_service)
) -> ConversationSentimentResponse:
    """
    Analyze sentiment for entire conversation with advanced features:
    - Overall conversation sentiment and emotional flow
    - Per-participant sentiment analysis and communication patterns
    - Escalation/de-escalation tracking and conflict resolution
    - Content summarization and topic extraction
    - Conversation quality assessment and satisfaction scoring
    - Detailed message-by-message breakdown with context awareness

    Supports multiple domains: customer service, sales, support, general conversation
    """
    settings = get_settings()
    start_time = time.time()

    # Validate conversation
    if not request.conversation:
        raise HTTPException(
            status_code=400,
            detail="Conversation cannot be empty"
        )

    if len(request.conversation) > 200:
        raise HTTPException(
            status_code=400,
            detail="Conversation exceeds maximum length of 200 messages"
        )

    # Validate participants
    participants = set(msg.speaker for msg in request.conversation)
    if len(participants) < 2:
        logger.warning("Conversation has only one participant")

    # Set default analysis options
    if not request.analysis_options:
        from apps.api.schemas.request import ConversationAnalysisOptions
        request.analysis_options = ConversationAnalysisOptions()

    try:
        # Perform conversation analysis
        result = await sentiment_service.analyze_conversation(request)

        # Add processing metadata
        processing_time = int((time.time() - start_time) * 1000)
        result.analysis_metadata.update({
            "api_processing_time_ms": processing_time,
            "participants_detected": list(participants),
            "total_characters": sum(len(msg.text) for msg in request.conversation),
            "avg_message_length": sum(len(msg.text) for msg in request.conversation) / len(request.conversation),
            "conversation_duration_estimated": len(request.conversation) * 30  # Estimate 30 seconds per message
        })

        logger.info(
            f"Conversation analysis completed: "
            f"{len(request.conversation)} messages, "
            f"{len(participants)} participants, "
            f"overall sentiment: {result.overall_sentiment.label}, "
            f"processing time: {processing_time}ms"
        )

        return result

    except Exception as e:
        logger.error(f"Conversation sentiment analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Conversation analysis failed: {str(e)}"
        )
