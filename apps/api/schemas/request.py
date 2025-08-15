"""
Request and Response schemas for AI Sentiment Analysis API
Enhanced with conversation analysis support
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime
from pydantic.v1 import validator
import re



# === SINGLE TEXT ANALYSIS ===
class SentimentRequest(BaseModel):
    """Request for single text sentiment analysis"""

    text: str = Field(
        ..., 
        min_length=1, 
        max_length=5000,
        description="Text to analyze for sentiment"
    )
    lang: Optional[str] = Field(
        None,
        pattern="^(vi|en|auto)$",
        description="Language code (vi, en, auto for detection)"
    )
    
    @validator('text')
    def validate_text(cls, v):
        """Validate and clean text input"""
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        
        # Remove excessive whitespace
        v = re.sub(r'\s+', ' ', v.strip())
        
        # Basic security check - no script tags
        if re.search(r'<script.*?</script>', v, re.IGNORECASE):
            raise ValueError("Invalid text content")
        
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "text": "Tôi rất thích sản phẩm này! Chất lượng tuyệt vời.",
                "lang": "vi"
            }
        }


class BatchItem(BaseModel):
    """Single item in batch request"""
    
    id: str = Field(..., description="Unique identifier for this item")
    text: str = Field(
        ..., 
        min_length=1, 
        max_length=5000,
        description="Text to analyze"
    )
    lang: Optional[str] = Field(
        None,
        pattern="^(vi|en|auto)$",
        description="Language code"
    )
    
    @validator('text')
    def validate_text(cls, v):
        """Validate text input"""
        return SentimentRequest.validate_text(v)


class BatchSentimentRequest(BaseModel):
    """Request for batch sentiment analysis"""

    items: List[BatchItem] = Field(
        ...,
        min_items=1,
        max_items=1000,
        description="List of texts to analyze"
    )
    async_mode: bool = Field(
        False,
        description="Process asynchronously if True"
    )
    
    @validator('items')
    def validate_unique_ids(cls, v):
        """Ensure all IDs are unique"""
        ids = [item.id for item in v]
        if len(ids) != len(set(ids)):
            raise ValueError("All item IDs must be unique")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "items": [
                    {
                        "id": "1",
                        "text": "Sản phẩm rất tốt!",
                        "lang": "vi"
                    },
                    {
                        "id": "2", 
                        "text": "Không hài lòng với dịch vụ",
                        "lang": "vi"
                    }
                ],
                "async_mode": False
            }
        }

# === CONVERSATION ANALYSIS ===
class ConversationMessage(BaseModel):
    """Single message in conversation"""
    message_id: Optional[str] = Field(None, description="Unique message identifier")
    speaker: str = Field(..., description="Speaker identifier (user, agent, customer, etc.)")
    text: str = Field(..., min_length=1, max_length=5000, description="Message content")
    timestamp: Optional[datetime] = Field(None, description="Message timestamp")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional message metadata")

class ConversationParticipant(BaseModel):
    """Participant information"""
    role: Optional[str] = Field(None, description="Role (customer, agent, moderator, etc.)")
    segment: Optional[str] = Field(None, description="Customer segment (premium, regular, etc.)")
    experience_level: Optional[str] = Field(None, description="Experience level (junior, senior, etc.)")
    history: Optional[str] = Field(None, description="Historical context")

class ConversationContext(BaseModel):
    """Context information for conversation"""
    domain: Optional[str] = Field(None, description="Domain (customer_service, sales, support, etc.)")
    conversation_type: Optional[str] = Field(None, description="Type (support_ticket, sales_call, etc.)")
    language: Optional[str] = Field("vi", description="Primary language")
    channel: Optional[str] = Field(None, description="Communication channel (chat, email, phone)")
    participants: Optional[Dict[str, ConversationParticipant]] = Field(None, description="Participant details")

class ConversationAnalysisOptions(BaseModel):
    """Analysis options"""
    include_emotional_flow: Optional[bool] = Field(True, description="Include emotional flow analysis")
    include_participant_analysis: Optional[bool] = Field(True, description="Include per-participant analysis")
    include_escalation_tracking: Optional[bool] = Field(True, description="Include escalation analysis")
    include_summary: Optional[bool] = Field(True, description="Include conversation summary")
    granularity: Optional[str] = Field("detailed", description="Analysis granularity (basic, detailed, comprehensive)")

class ConversationSentimentRequest(BaseModel):
    """Request for conversation sentiment analysis"""
    conversation: List[ConversationMessage] = Field(..., min_items=1, max_items=200, description="Conversation messages")
    context: Optional[ConversationContext] = Field(None, description="Conversation context")
    analysis_options: Optional[ConversationAnalysisOptions] = Field(None, description="Analysis options")
