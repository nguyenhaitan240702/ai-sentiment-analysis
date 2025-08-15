"""
Sentiment service dependency for FastAPI
"""

from typing import Optional
from fastapi import HTTPException
from apps.api.services.sentiment_service import SentimentService

# Global sentiment service instance
_sentiment_service: Optional[SentimentService] = None

def set_sentiment_service(service: SentimentService):
    """Set the global sentiment service instance"""
    global _sentiment_service
    _sentiment_service = service

def get_sentiment_service() -> SentimentService:
    """Dependency to get the initialized sentiment service"""
    if _sentiment_service is None:
        raise HTTPException(
            status_code=503,
            detail="Sentiment service not initialized"
        )
    return _sentiment_service
