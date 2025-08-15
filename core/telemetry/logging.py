"""
Structured logging setup for AI Sentiment Analysis
JSON logging with correlation IDs and performance tracking
"""

import logging
import sys
from typing import Dict, Any, Optional
from datetime import datetime
import json
import structlog
from pathlib import Path

from core.config.base import get_settings

def setup_logging():
    """Setup structured logging for the application"""
    settings = get_settings()
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if settings.LOG_FORMAT == "json" else structlog.dev.ConsoleRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        context_class=dict,
        cache_logger_on_first_use=True,
    )
    
    # Configure Python logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.LOG_LEVEL.upper())
    )
    
    # Setup file logging if specified
    if settings.LOG_FILE:
        log_file = Path(settings.LOG_FILE)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add to root logger
        logging.getLogger().addHandler(file_handler)

class SentimentLogger:
    """Custom logger for sentiment analysis operations"""
    
    def __init__(self, name: str):
        self.logger = structlog.get_logger(name)
    
    def log_request(self, 
                   endpoint: str, 
                   method: str, 
                   text_length: int,
                   language: Optional[str] = None,
                   user_agent: Optional[str] = None,
                   ip_address: Optional[str] = None):
        """Log API request"""
        self.logger.info(
            "api_request",
            endpoint=endpoint,
            method=method,
            text_length=text_length,
            language=language,
            user_agent=user_agent,
            ip_address=ip_address
        )
    
    def log_prediction(self, 
                      text: str,
                      prediction: str,
                      score: float,
                      model: str,
                      latency_ms: int,
                      cached: bool = False):
        """Log sentiment prediction"""
        self.logger.info(
            "sentiment_prediction",
            text_preview=text[:100] + "..." if len(text) > 100 else text,
            prediction=prediction,
            score=score,
            model=model,
            latency_ms=latency_ms,
            cached=cached
        )
    
    def log_batch_job(self, 
                     job_id: str,
                     status: str,
                     total_items: int,
                     processed_items: int = 0,
                     failed_items: int = 0):
        """Log batch job status"""
        self.logger.info(
            "batch_job",
            job_id=job_id,
            status=status,
            total_items=total_items,
            processed_items=processed_items,
            failed_items=failed_items
        )
    
    def log_model_operation(self, 
                           operation: str,
                           model_name: str,
                           duration_ms: Optional[int] = None,
                           success: bool = True,
                           error: Optional[str] = None):
        """Log model operations (load, unload, predict)"""
        log_data = {
            "model_operation": operation,
            "model_name": model_name,
            "success": success
        }
        
        if duration_ms is not None:
            log_data["duration_ms"] = duration_ms
        
        if error:
            log_data["error"] = error
            self.logger.error("model_operation_failed", **log_data)
        else:
            self.logger.info("model_operation", **log_data)
    
    def log_database_operation(self, 
                              operation: str,
                              collection: str,
                              duration_ms: int,
                              success: bool = True,
                              error: Optional[str] = None):
        """Log database operations"""
        log_data = {
            "db_operation": operation,
            "collection": collection,
            "duration_ms": duration_ms,
            "success": success
        }
        
        if error:
            log_data["error"] = error
            self.logger.error("database_operation_failed", **log_data)
        else:
            self.logger.info("database_operation", **log_data)

# Global logger instances
api_logger = SentimentLogger("sentiment.api")
worker_logger = SentimentLogger("sentiment.worker")
model_logger = SentimentLogger("sentiment.model")
db_logger = SentimentLogger("sentiment.database")
