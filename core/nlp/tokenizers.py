"""
Vietnamese text tokenizers for AI Sentiment Analysis
Handles word segmentation and tokenization for Vietnamese text
"""

import re
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class VietnameseTokenizer:
    """Vietnamese text tokenizer with basic word segmentation"""
    
    def __init__(self):
        # Vietnamese syllable pattern
        self.syllable_pattern = re.compile(
            r'[aăâeêiouưyáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ]+'
            r'[bcdfghjklmnpqrstvwxz]*',
            re.IGNORECASE
        )
        
        # Punctuation to keep separate
        self.punctuation = set('.,!?;:()[]{}"\'-')
        
        # Stop words (Vietnamese)
        self.stop_words = {
            'và', 'của', 'có', 'được', 'một', 'này', 'đó', 'các', 'cho', 'từ',
            'với', 'họ', 'tôi', 'bạn', 'anh', 'chị', 'em', 'ông', 'bà',
            'là', 'sẽ', 'đã', 'đang', 'rất', 'nhiều', 'ít', 'lại', 'về',
            'trong', 'ngoài', 'trên', 'dưới', 'sau', 'trước', 'bên', 'giữa'
        }
    
    async def tokenize(self, text: str, remove_stop_words: bool = False) -> List[str]:
        """Tokenize Vietnamese text into words"""
        if not text:
            return []
        
        # Basic tokenization by splitting on whitespace and punctuation
        tokens = self._basic_tokenize(text)
        
        # Remove stop words if requested
        if remove_stop_words:
            tokens = [token for token in tokens if token.lower() not in self.stop_words]
        
        # Filter empty tokens
        tokens = [token for token in tokens if token.strip()]
        
        return tokens
    
    def _basic_tokenize(self, text: str) -> List[str]:
        """Basic tokenization splitting on whitespace and punctuation"""
        # Replace punctuation with spaces
        for punct in self.punctuation:
            text = text.replace(punct, f' {punct} ')
        
        # Split on whitespace
        tokens = text.split()
        
        # Remove punctuation tokens (keep only meaningful words)
        tokens = [token for token in tokens if token not in self.punctuation]
        
        return tokens
    
    async def extract_phrases(self, text: str, min_length: int = 2, max_length: int = 4) -> List[str]:
        """Extract meaningful phrases from text"""
        tokens = await self.tokenize(text, remove_stop_words=True)
        phrases = []
        
        # Extract n-grams
        for n in range(min_length, min(max_length + 1, len(tokens) + 1)):
            for i in range(len(tokens) - n + 1):
                phrase = ' '.join(tokens[i:i+n])
                phrases.append(phrase)
        
        return phrases
    
    def get_word_count(self, tokens: List[str]) -> int:
        """Get meaningful word count (excluding stop words)"""
        return len([token for token in tokens if token.lower() not in self.stop_words])


class SimpleVietnameseTokenizer:
    """Simplified Vietnamese tokenizer for basic use cases"""
    
    def __init__(self):
        self.word_pattern = re.compile(r'[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]+', re.IGNORECASE)
    
    async def tokenize(self, text: str) -> List[str]:
        """Simple tokenization using regex"""
        if not text:
            return []
        
        return self.word_pattern.findall(text.lower())
