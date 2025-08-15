"""
Celery worker tasks for AI Sentiment Analysis
Handles background processing and batch operations
"""

import logging
import time
from typing import List, Dict, Any
from datetime import datetime

from celery import Celery
from celery.signals import worker_ready, worker_shutdown

from core.config.base import get_settings
from core.models.base import ModelManager
from apps.api.schemas.request import BatchItem

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()

# Create Celery app
celery_app = Celery(
    "sentiment_worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["apps.worker.tasks"]
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=settings.CELERY_TASK_TIMEOUT,
    task_soft_time_limit=settings.CELERY_TASK_TIMEOUT - 30,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_max_tasks_per_child=1000,
    task_routes={
        "apps.worker.tasks.process_sentiment": {"queue": "sentiment.default"},
        "apps.worker.tasks.process_batch_sentiment": {"queue": "sentiment.bulk"},
        "apps.worker.tasks.aggregate_daily_stats": {"queue": "sentiment.analytics"}
    }
)

# Global model manager
model_manager: ModelManager = None

@worker_ready.connect
def worker_ready_handler(sender=None, **kwargs):
    """Initialize worker when ready"""
    global model_manager
    logger.info("Initializing Celery worker...")
    
    try:
        # Initialize model manager
        import asyncio
        model_manager = ModelManager()
        
        # Run async initialization in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(model_manager.initialize())
        
        logger.info("Celery worker initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize worker: {e}")
        raise

@worker_shutdown.connect
def worker_shutdown_handler(sender=None, **kwargs):
    """Cleanup when worker shuts down"""
    global model_manager
    logger.info("Shutting down Celery worker...")
    
    if model_manager:
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            loop.run_until_complete(model_manager.cleanup())
        except Exception as e:
            logger.warning(f"Error during worker cleanup: {e}")

@celery_app.task(bind=True, name="process_sentiment")
def process_sentiment(self, text: str, language: str = "vi") -> Dict[str, Any]:
    """Process single sentiment analysis task"""
    global model_manager
    
    if not model_manager:
        raise RuntimeError("Model manager not initialized")
    
    start_time = time.time()
    
    try:
        # Run prediction in async context
        import asyncio
        loop = asyncio.get_event_loop()
        
        prediction = loop.run_until_complete(
            model_manager.predict(text, language)
        )
        
        processing_time = int((time.time() - start_time) * 1000)
        
        result = {
            "label": prediction.label,
            "score": prediction.score,
            "scores": prediction.scores,
            "model": model_manager.default_model,
            "language": language,
            "processing_time_ms": processing_time,
            "metadata": prediction.metadata
        }
        
        logger.info(f"Processed sentiment for text: {text[:50]}... -> {prediction.label}")
        return result
        
    except Exception as e:
        logger.error(f"Sentiment processing failed: {e}")
        self.retry(countdown=60, max_retries=3)

@celery_app.task(bind=True, name="process_batch_sentiment")
def process_batch_sentiment(self, items_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Process batch sentiment analysis task"""
    global model_manager
    
    if not model_manager:
        raise RuntimeError("Model manager not initialized")
    
    start_time = time.time()
    results = []
    failed_count = 0
    
    try:
        import asyncio
        loop = asyncio.get_event_loop()
        
        for item_data in items_data:
            try:
                # Extract item data
                text = item_data["text"]
                language = item_data.get("lang", "vi")
                item_id = item_data["id"]
                
                # Process sentiment
                prediction = loop.run_until_complete(
                    model_manager.predict(text, language)
                )
                
                result = {
                    "id": item_id,
                    "result": {
                        "label": prediction.label,
                        "score": prediction.score,
                        "scores": prediction.scores
                    },
                    "error": None
                }
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process item {item_data.get('id', 'unknown')}: {e}")
                failed_count += 1
                
                result = {
                    "id": item_data.get("id", "unknown"),
                    "result": None,
                    "error": str(e)
                }
                results.append(result)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        batch_result = {
            "status": "completed",
            "results": results,
            "total_items": len(items_data),
            "processed_items": len(items_data) - failed_count,
            "failed_items": failed_count,
            "processing_time_ms": processing_time
        }
        
        logger.info(f"Batch processing completed: {len(items_data)} items, {failed_count} failed")
        return batch_result
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        self.retry(countdown=60, max_retries=3)

@celery_app.task(name="aggregate_daily_stats")
def aggregate_daily_stats(date_str: str) -> Dict[str, Any]:
    """Aggregate daily sentiment statistics"""
    try:
        # This would typically aggregate data from MongoDB to MySQL
        # For now, just a placeholder
        logger.info(f"Aggregating daily stats for {date_str}")
        
        # TODO: Implement actual aggregation logic
        # 1. Query MongoDB for inferences from the date
        # 2. Group by language, source, sentiment
        # 3. Calculate counts and averages
        # 4. Upsert into MySQL sentiments_daily table
        
        return {
            "status": "completed",
            "date": date_str,
            "processed_records": 0
        }
        
    except Exception as e:
        logger.error(f"Daily stats aggregation failed for {date_str}: {e}")
        raise

@celery_app.task(name="health_check")
def health_check() -> Dict[str, Any]:
    """Worker health check task"""
    global model_manager
    
    return {
        "status": "healthy",
        "worker_id": health_check.request.id,
        "model_loaded": model_manager is not None and model_manager.default_model is not None,
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    # Run worker
    celery_app.start()
