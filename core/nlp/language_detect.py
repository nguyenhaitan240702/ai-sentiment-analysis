"""
Language detection for AI Sentiment Analysis
Supports Vietnamese and English language detection
"""

import re
import logging
from typing import Optional, Dict
import asyncio

logger = logging.getLogger(__name__)

class LanguageDetector:
    """Simple language detector for Vietnamese and English"""
    
    def __init__(self):
        # Vietnamese specific characters
        self.vietnamese_chars = set('áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ')
        
        # Vietnamese common words
        self.vietnamese_words = {
            'và', 'của', 'có', 'được', 'một', 'này', 'đó', 'các', 'cho', 'từ',
            'với', 'họ', 'tôi', 'bạn', 'anh', 'chị', 'em', 'ông', 'bà', 'là',
            'sẽ', 'đã', 'đang', 'rất', 'nhiều', 'ít', 'lại', 'về', 'trong',
            'ngoài', 'trên', 'dưới', 'sau', 'trước', 'không', 'chưa', 'cũng',
            'như', 'thì', 'nếu', 'mà', 'hoặc', 'nhưng', 'vì', 'nên', 'để'
        }
        
        # English common words
        self.english_words = {
            'the', 'and', 'of', 'to', 'a', 'in', 'is', 'it', 'you', 'that',
            'he', 'was', 'for', 'on', 'are', 'as', 'with', 'his', 'they',
            'i', 'at', 'be', 'this', 'have', 'from', 'or', 'one', 'had',
            'by', 'word', 'but', 'not', 'what', 'all', 'were', 'we', 'when'
        }
        
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize language detector"""
        # In a real implementation, you might load fasttext models here
        self.is_initialized = True
        logger.info("Language detector initialized")
    
    async def detect(self, text: str) -> str:
        """Detect language of text"""
        if not text or not text.strip():
            return "vi"  # Default to Vietnamese
        
        text_lower = text.lower()
        
        # Check for Vietnamese characters
        vietnamese_char_count = sum(1 for char in text_lower if char in self.vietnamese_chars)
        
        # Check for Vietnamese words
        words = re.findall(r'\b\w+\b', text_lower)
        vietnamese_word_count = sum(1 for word in words if word in self.vietnamese_words)
        english_word_count = sum(1 for word in words if word in self.english_words)
        
        # Simple heuristic
        if vietnamese_char_count > 0 or vietnamese_word_count > english_word_count:
            return "vi"
        elif english_word_count > 0:
            return "en"
        else:
            return "vi"  # Default to Vietnamese
    
    async def detect_with_confidence(self, text: str) -> Dict[str, float]:
        """Detect language with confidence scores"""
        if not text or not text.strip():
            return {"vi": 1.0, "en": 0.0}
        
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        if not words:
            return {"vi": 1.0, "en": 0.0}
        
        # Count indicators
        vietnamese_chars = sum(1 for char in text_lower if char in self.vietnamese_chars)
        vietnamese_words = sum(1 for word in words if word in self.vietnamese_words)
        english_words = sum(1 for word in words if word in self.english_words)
        
        # Calculate scores
        total_chars = len(text_lower)
        total_words = len(words)
        
        vietnamese_score = 0.0
        english_score = 0.0
        
        # Character-based scoring
        if total_chars > 0:
            vietnamese_score += (vietnamese_chars / total_chars) * 0.5
        
        # Word-based scoring
        if total_words > 0:
            vietnamese_score += (vietnamese_words / total_words) * 0.5
            english_score += (english_words / total_words) * 0.5
        
        # Normalize scores
        total_score = vietnamese_score + english_score
        if total_score > 0:
            vietnamese_score = vietnamese_score / total_score
            english_score = english_score / total_score
        else:
            vietnamese_score = 1.0
            english_score = 0.0
        
        return {
            "vi": round(vietnamese_score, 3),
            "en": round(english_score, 3)
        }
