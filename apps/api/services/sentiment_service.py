"""
Sentiment analysis service for AI Sentiment Analysis API
Orchestrates the sentiment analysis workflow
Enhanced with conversation analysis support
"""

import logging
import time
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio

from apps.api.schemas.request import BatchItem, ConversationSentimentRequest
from apps.api.schemas.response import SentimentResponse, SentimentResult, BatchItemResult, JobResponse, ConversationSentimentResponse
from apps.api.deps.db import get_mongo_db, get_mysql_session
from apps.api.deps.cache import CacheManager, get_cache_client
from core.models.base import ModelManager, SentimentPrediction
from core.nlp.language_detect import LanguageDetector
from core.config.base import get_settings

logger = logging.getLogger(__name__)

class SentimentService:
    """Service for sentiment analysis operations"""

    def __init__(self):
        self.settings = get_settings()
        self.model_manager: Optional[ModelManager] = None
        self.language_detector = LanguageDetector()
        self.job_cache: Dict[str, Dict[str, Any]] = {}
        self.conversation_service: Optional[Any] = None  # Will be set after import

    async def initialize(self, model_manager: ModelManager):
        """Initialize service with model manager"""
        self.model_manager = model_manager
        await self.language_detector.initialize()

        # Initialize conversation service
        from apps.api.services.conversation_service import ConversationSentimentService
        self.conversation_service = ConversationSentimentService()
        await self.conversation_service.initialize(self)
        logger.info("Conversation sentiment service initialized")

    async def analyze_text(self, text: str, language: Optional[str] = None) -> SentimentResponse:
        """Analyze sentiment for single text"""
        start_time = time.time()

        # Detect language if not provided
        if not language or language == "auto":
            detected_lang = await self.language_detector.detect(text)
            language = detected_lang if detected_lang in self.settings.supported_languages else self.settings.DEFAULT_LANG

        # Get prediction from model
        prediction = await self.model_manager.predict(text, language)

        # Convert to response format
        response = SentimentResponse(
            label=prediction.label,
            score=prediction.score,
            scores=prediction.scores,
            model=self.model_manager.default_model or "unknown",
            language_detected=language,
            latency_ms=int((time.time() - start_time) * 1000),
            cached=False,
            metadata=prediction.metadata
        )

        # Store in database (async)
        asyncio.create_task(self._store_inference(text, language, prediction, response))

        return response

    async def analyze_batch(self, items: List[BatchItem]) -> List[BatchItemResult]:
        """Analyze sentiment for batch of texts (sync)"""
        results = []

        for item in items:
            try:
                # Analyze individual item
                response = await self.analyze_text(item.text, item.lang)

                result = BatchItemResult(
                    id=item.id,
                    result=SentimentResult(
                        label=response.label,
                        score=response.score,
                        scores=response.scores
                    ),
                    error=None
                )
                results.append(result)

            except Exception as e:
                logger.error(f"Failed to analyze item {item.id}: {e}")
                result = BatchItemResult(
                    id=item.id,
                    result=None,
                    error=str(e)
                )
                results.append(result)

        return results

    async def process_batch_async(self, job_id: str, items: List[BatchItem]) -> None:
        """Process batch asynchronously"""
        start_time = datetime.utcnow()

        # Initialize job status
        self.job_cache[job_id] = {
            "status": "processing",
            "progress": 0.0,
            "total_items": len(items),
            "processed_items": 0,
            "failed_items": 0,
            "results": [],
            "created_at": start_time.isoformat(),
            "completed_at": None,
            "error": None
        }

        try:
            results = []

            for i, item in enumerate(items):
                try:
                    # Analyze item
                    response = await self.analyze_text(item.text, item.lang)

                    result = BatchItemResult(
                        id=item.id,
                        result=SentimentResult(
                            label=response.label,
                            score=response.score,
                            scores=response.scores
                        ),
                        error=None
                    )
                    results.append(result)
                    self.job_cache[job_id]["processed_items"] += 1

                except Exception as e:
                    logger.error(f"Failed to analyze item {item.id}: {e}")
                    result = BatchItemResult(
                        id=item.id,
                        result=None,
                        error=str(e)
                    )
                    results.append(result)
                    self.job_cache[job_id]["failed_items"] += 1

                # Update progress
                progress = ((i + 1) / len(items)) * 100
                self.job_cache[job_id]["progress"] = progress

            # Job completed successfully
            self.job_cache[job_id].update({
                "status": "completed",
                "progress": 100.0,
                "results": results,
                "completed_at": datetime.utcnow().isoformat()
            })

        except Exception as e:
            logger.error(f"Batch job {job_id} failed: {e}")
            self.job_cache[job_id].update({
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.utcnow().isoformat()
            })

    async def get_job_result(self, job_id: str) -> Optional[JobResponse]:
        """Get async job result"""
        if job_id not in self.job_cache:
            return None

        job_data = self.job_cache[job_id]

        return JobResponse(
            job_id=job_id,
            status=job_data["status"],
            progress=job_data.get("progress"),
            results=job_data.get("results"),
            total_items=job_data["total_items"],
            processed_items=job_data["processed_items"],
            failed_items=job_data["failed_items"],
            created_at=job_data["created_at"],
            completed_at=job_data.get("completed_at"),
            error=job_data.get("error")
        )

    async def analyze_conversation(
        self,
        request: ConversationSentimentRequest
    ) -> ConversationSentimentResponse:
        """Analyze sentiment for the entire conversation"""
        if not self.conversation_service:
            raise RuntimeError("Conversation service not initialized")

        try:
            result = await self.conversation_service.analyze_conversation(request)

            # Store conversation analysis in database (async)
            asyncio.create_task(self._store_conversation_analysis(request, result))

            return result

        except Exception as e:
            logger.error(f"Conversation analysis failed: {e}")
            raise

    @staticmethod
    async def _store_inference(text: str, language: str, prediction: SentimentPrediction, response: SentimentResponse) -> None:
        """Store inference result in database"""
        try:
            # Store in MongoDB
            mongo_db = await get_mongo_db()

            # Store text document
            text_doc = {
                "source": "api",
                "lang": language,
                "content": text,
                "content_hash": hashlib.sha256(text.encode()).hexdigest(),
                "created_at": datetime.utcnow()
            }
            text_result = await mongo_db.texts.insert_one(text_doc)

            # Store inference result
            inference_doc = {
                "text_id": text_result.inserted_id,
                "model_name": response.model,
                "version": "1.0",
                "sentiment": {
                    "label": prediction.label,
                    "score": prediction.score,
                    "scores": prediction.scores
                },
                "tokens_metadata": prediction.metadata,
                "latency_ms": response.latency_ms,
                "created_at": datetime.utcnow()
            }
            await mongo_db.inferences.insert_one(inference_doc)

            logger.debug(f"Stored inference for text: {text[:50]}...")

        except Exception as e:
            logger.warning(f"Failed to store inference: {e}")

    @staticmethod
    async def _store_conversation_analysis(
            request: ConversationSentimentRequest,
        result: ConversationSentimentResponse
    ) -> None:
        """Store conversation analysis result in database"""
        try:
            mongo_db = await get_mongo_db()

            # Create conversation document
            conversation_doc = {
                "source": "api",
                "conversation_type": result.conversation_type,
                "message_count": len(request.conversation),
                "participants": list(result.participants.keys()),
                "context": request.context.dict() if request.context else None,
                "messages": [
                    {
                        "message_id": msg.message_id,
                        "speaker": msg.speaker,
                        "text": msg.text,
                        "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                        "metadata": msg.metadata
                    }
                    for msg in request.conversation
                ],
                "created_at": datetime.utcnow()
            }
            conversation_result = await mongo_db.conversations.insert_one(conversation_doc)

            # Store conversation analysis
            analysis_doc = {
                "conversation_id": conversation_result.inserted_id,
                "overall_sentiment": {
                    "label": result.overall_sentiment.label,
                    "score": result.overall_sentiment.score,
                    "scores": result.overall_sentiment.scores
                },
                "emotional_flow": {
                    "trajectory": result.emotional_flow.trajectory,
                    "stability": result.emotional_flow.stability,
                    "dominant_emotion": result.emotional_flow.dominant_emotion,
                    "sentiment_progression": result.emotional_flow.sentiment_progression,
                    "turning_points": [tp.dict() for tp in result.emotional_flow.turning_points]
                },
                "participants_analysis": {
                    speaker: {
                        "overall_sentiment": analysis.overall_sentiment,
                        "overall_score": analysis.overall_score,
                        "dominant_emotions": analysis.dominant_emotions,
                        "attitude": analysis.attitude,
                        "communication_style": analysis.communication_style,
                        "engagement_level": analysis.engagement_level
                    }
                    for speaker, analysis in result.participants.items()
                },
                "escalation_analysis": {
                    "escalation_level": result.escalation_analysis.escalation_level,
                    "escalation_trend": result.escalation_analysis.escalation_trend,
                    "peak_tension_point": result.escalation_analysis.peak_tension_point,
                    "conflict_intensity": result.escalation_analysis.conflict_intensity,
                    "resolution_indicators": result.escalation_analysis.resolution_indicators
                },
                "conversation_summary": {
                    "main_topics": result.conversation_summary.main_topics,
                    "key_issues": result.conversation_summary.key_issues,
                    "resolution_status": result.conversation_summary.resolution_status,
                    "conversation_outcome": result.conversation_summary.conversation_outcome,
                    "satisfaction_indicators": result.conversation_summary.satisfaction_indicators
                },
                "quality_metrics": {
                    "communication_quality": result.conversation_quality.communication_quality,
                    "professionalism_level": result.conversation_quality.professionalism_level,
                    "empathy_score": result.conversation_quality.empathy_score,
                    "solution_effectiveness": result.conversation_quality.solution_effectiveness,
                    "customer_satisfaction_score": result.conversation_quality.customer_satisfaction_score
                },
                "analysis_metadata": result.analysis_metadata,
                "created_at": datetime.utcnow()
            }
            await mongo_db.conversation_analyses.insert_one(analysis_doc)

            logger.debug(f"Stored conversation analysis for {len(request.conversation)} messages")

        except Exception as e:
            logger.warning(f"Failed to store conversation analysis: {e}")

# FastAPI dependency
async def get_sentiment_service():
    """FastAPI dependency for sentiment service"""
    service = SentimentService()
    # Note: In real app, you'd inject the model manager here
    # For now, this will be handled in the main app
    return service
