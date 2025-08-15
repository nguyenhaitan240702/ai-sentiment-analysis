"""
Response schemas for AI Sentiment Analysis API
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class SentimentLabel(str, Enum):
    """Sentiment classification labels"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class SentimentResult(BaseModel):
    """Sentiment analysis result"""
    
    label: SentimentLabel = Field(..., description="Sentiment classification")
    score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Confidence score (0.0 to 1.0)"
    )
    scores: Optional[Dict[str, float]] = Field(
        None,
        description="Detailed scores for each class"
    )


class SentimentResponse(BaseModel):
    """Single sentiment analysis response"""
    
    label: SentimentLabel = Field(..., description="Predicted sentiment")
    score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Confidence score"
    )
    scores: Optional[Dict[str, float]] = Field(
        None,
        description="Detailed class scores"
    )
    model: str = Field(..., description="Model used for prediction")
    language_detected: Optional[str] = Field(
        None,
        description="Detected language code"
    )
    latency_ms: int = Field(..., description="Processing time in milliseconds")
    cached: bool = Field(False, description="Whether result was cached")
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "label": "positive",
                "score": 0.85,
                "scores": {
                    "positive": 0.85,
                    "neutral": 0.12,
                    "negative": 0.03
                },
                "model": "rule_based_vi",
                "language_detected": "vi",
                "latency_ms": 15,
                "cached": False,
                "metadata": {
                    "tokens_count": 8,
                    "normalized_text": "tôi rất thích sản phẩm này"
                }
            }
        }


class BatchItemResult(BaseModel):
    """Result for single item in batch"""
    
    id: str = Field(..., description="Item identifier")
    result: Optional[SentimentResult] = Field(
        None,
        description="Sentiment result (null if error)"
    )
    error: Optional[str] = Field(
        None,
        description="Error message if processing failed"
    )


class BatchSentimentResponse(BaseModel):
    """Batch sentiment analysis response"""
    
    job_id: Optional[str] = Field(
        None,
        description="Job ID for async processing"
    )
    status: str = Field(
        ...,
        description="Processing status: processing, completed, failed"
    )
    results: Optional[List[BatchItemResult]] = Field(
        None,
        description="Results for each item (null if async)"
    )
    total_items: int = Field(..., description="Total number of items")
    processed_items: Optional[int] = Field(
        None,
        description="Number of processed items"
    )
    failed_items: Optional[int] = Field(
        None,
        description="Number of failed items"
    )
    processing_time_ms: Optional[int] = Field(
        None,
        description="Total processing time"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "job_id": None,
                "status": "completed",
                "results": [
                    {
                        "id": "1",
                        "result": {
                            "label": "positive",
                            "score": 0.90
                        },
                        "error": None
                    }
                ],
                "total_items": 1,
                "processed_items": 1,
                "failed_items": 0,
                "processing_time_ms": 45
            }
        }


class JobResponse(BaseModel):
    """Async job status response"""
    
    job_id: str = Field(..., description="Job identifier")
    status: str = Field(
        ...,
        description="Job status: processing, completed, failed"
    )
    progress: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Progress percentage"
    )
    results: Optional[List[BatchItemResult]] = Field(
        None,
        description="Results (available when completed)"
    )
    total_items: int = Field(..., description="Total items to process")
    processed_items: int = Field(0, description="Items processed so far")
    failed_items: int = Field(0, description="Items that failed")
    created_at: str = Field(..., description="Job creation timestamp")
    completed_at: Optional[str] = Field(
        None,
        description="Job completion timestamp"
    )
    error: Optional[str] = Field(
        None,
        description="Error message if job failed"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "job_id": "job_12345",
                "status": "completed",
                "progress": 100.0,
                "results": [],
                "total_items": 100,
                "processed_items": 100,
                "failed_items": 0,
                "created_at": "2025-08-13T14:30:00Z",
                "completed_at": "2025-08-13T14:31:15Z",
                "error": None
            }
        }


# === CONVERSATION ANALYSIS RESPONSES ===
class SentimentTurningPoint(BaseModel):
    """Sentiment turning point in conversation"""
    message_index: int = Field(..., description="Index of message where sentiment changed")
    from_sentiment: str = Field(..., description="Previous sentiment")
    to_sentiment: str = Field(..., description="New sentiment")
    trigger: str = Field(..., description="What triggered the change")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in turning point detection")

class EmotionalFlow(BaseModel):
    """Emotional flow analysis"""
    trajectory: str = Field(..., description="Overall emotional trajectory (e.g., 'negative -> neutral -> positive')")
    stability: str = Field(..., description="Emotional stability (stable, unstable, volatile)")
    dominant_emotion: str = Field(..., description="Most prominent emotion throughout conversation")
    turning_points: List[SentimentTurningPoint] = Field(default=[], description="Key sentiment changes")
    sentiment_progression: List[float] = Field(..., description="Sentiment scores over time")

class ParticipantSentiment(BaseModel):
    """Sentiment analysis for individual participant"""
    overall_sentiment: str = Field(..., description="Overall sentiment label")
    overall_score: float = Field(..., ge=-1, le=1, description="Overall sentiment score")
    dominant_emotions: List[str] = Field(default=[], description="Primary emotions displayed")
    sentiment_progression: List[float] = Field(..., description="Sentiment scores per message")
    attitude: str = Field(..., description="Overall attitude (cooperative, hostile, neutral, etc.)")
    communication_style: str = Field(..., description="Communication style (professional, casual, aggressive, etc.)")
    engagement_level: str = Field(..., description="Engagement level (high, medium, low)")

class EscalationAnalysis(BaseModel):
    """Escalation/de-escalation analysis"""
    escalation_level: str = Field(..., description="Current escalation level (none, low, medium, high, critical)")
    escalation_trend: str = Field(..., description="Escalation trend (escalating, de-escalating, stable)")
    peak_tension_point: Optional[int] = Field(None, description="Message index of highest tension")
    resolution_indicators: List[str] = Field(default=[], description="Signs of issue resolution")
    conflict_intensity: float = Field(..., ge=0, le=1, description="Overall conflict intensity")

class ConversationSummary(BaseModel):
    """Conversation content summary"""
    main_topics: List[str] = Field(default=[], description="Main topics discussed")
    key_issues: List[str] = Field(default=[], description="Key issues or problems raised")
    resolution_status: str = Field(..., description="Resolution status (resolved, partially_resolved, unresolved)")
    conversation_outcome: str = Field(..., description="Overall outcome (positive, negative, neutral)")
    satisfaction_indicators: List[str] = Field(default=[], description="Indicators of satisfaction/dissatisfaction")

class ConversationQuality(BaseModel):
    """Overall conversation quality metrics"""
    communication_quality: str = Field(..., description="Quality of communication (excellent, good, fair, poor)")
    professionalism_level: str = Field(..., description="Level of professionalism")
    empathy_score: float = Field(..., ge=0, le=1, description="Empathy demonstrated")
    solution_effectiveness: float = Field(..., ge=0, le=1, description="Effectiveness of solutions provided")
    customer_satisfaction_score: float = Field(..., ge=0, le=1, description="Estimated customer satisfaction")

class MessageAnalysis(BaseModel):
    """Detailed analysis of individual message"""
    message_index: int = Field(..., description="Message position in conversation")
    speaker: str = Field(..., description="Message speaker")
    text: str = Field(..., description="Message content")
    sentiment: SentimentResult = Field(..., description="Sentiment analysis result")
    emotions: List[str] = Field(default=[], description="Detected emotions")
    intent: Optional[str] = Field(None, description="Message intent")
    context_factors: List[str] = Field(default=[], description="Contextual factors affecting sentiment")
    response_to: Optional[int] = Field(None, description="Index of message this responds to")

class ConversationSentimentResponse(BaseModel):
    """Complete conversation sentiment analysis response"""

    # Overall Assessment
    overall_sentiment: SentimentResult = Field(..., description="Overall conversation sentiment")
    conversation_type: str = Field(..., description="Detected conversation type")

    # Emotional Flow Analysis
    emotional_flow: EmotionalFlow = Field(..., description="Emotional progression analysis")

    # Participant Analysis
    participants: Dict[str, ParticipantSentiment] = Field(..., description="Per-participant sentiment analysis")

    # Escalation Analysis
    escalation_analysis: EscalationAnalysis = Field(..., description="Escalation/conflict analysis")

    # Content Analysis
    conversation_summary: ConversationSummary = Field(..., description="Conversation content summary")

    # Quality Metrics
    conversation_quality: ConversationQuality = Field(..., description="Conversation quality assessment")

    # Detailed Breakdown
    message_analysis: List[MessageAnalysis] = Field(..., description="Detailed analysis of each message")

    # Metadata
    analysis_metadata: Dict[str, Any] = Field(default={}, description="Analysis metadata and metrics")

    class Config:
        schema_extra = {
            "example": {
                "overall_sentiment": {
                    "label": "positive",
                    "score": 0.65,
                    "scores": {"positive": 0.65, "negative": 0.15, "neutral": 0.20}
                },
                "conversation_type": "customer_service_resolution",
                "emotional_flow": {
                    "trajectory": "negative -> neutral -> positive",
                    "stability": "improving",
                    "dominant_emotion": "satisfaction",
                    "turning_points": [
                        {
                            "message_index": 3,
                            "from_sentiment": "negative",
                            "to_sentiment": "neutral",
                            "trigger": "acknowledgment_of_issue",
                            "confidence": 0.85
                        }
                    ],
                    "sentiment_progression": [-0.7, -0.3, 0.1, 0.6, 0.8]
                },
                "participants": {
                    "customer": {
                        "overall_sentiment": "satisfied",
                        "overall_score": 0.6,
                        "dominant_emotions": ["frustration", "relief", "satisfaction"],
                        "sentiment_progression": [-0.8, -0.2, 0.3, 0.7],
                        "attitude": "cooperative",
                        "communication_style": "direct",
                        "engagement_level": "high"
                    }
                },
                "escalation_analysis": {
                    "escalation_level": "resolved",
                    "escalation_trend": "de-escalating",
                    "peak_tension_point": 1,
                    "resolution_indicators": ["acknowledgment", "solution_provided", "satisfaction_expressed"],
                    "conflict_intensity": 0.2
                }
            }
        }
