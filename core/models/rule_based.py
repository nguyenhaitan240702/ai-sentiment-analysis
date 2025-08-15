"""
Rule-based sentiment analysis model for Vietnamese text
Fast, lightweight implementation using lexicon and heuristics
"""

import re
import logging
from typing import Dict, List, Optional, Set
import asyncio
from pathlib import Path

from core.models.base import SentimentModel, SentimentPrediction
from core.nlp.normalizers import VietnameseNormalizer
from core.nlp.tokenizers import VietnameseTokenizer

logger = logging.getLogger(__name__)

class RuleBasedModel(SentimentModel):
    """Rule-based sentiment model for Vietnamese"""
    
    def __init__(self):
        super().__init__("rule_based_vi", "1.0")
        self.normalizer = VietnameseNormalizer()
        self.tokenizer = VietnameseTokenizer()
        
        # Lexicons
        self.positive_words: Set[str] = set()
        self.negative_words: Set[str] = set()
        self.intensifiers: Set[str] = set()
        self.negators: Set[str] = set()
        
        # Weights and thresholds
        self.positive_threshold = 0.1
        self.negative_threshold = -0.1
        self.intensifier_boost = 1.5
        self.negator_flip = -1.0
    
    async def load(self) -> None:
        """Load lexicons and initialize model"""
        logger.info("Loading rule-based sentiment model...")
        
        try:
            # Load Vietnamese sentiment lexicons
            await self._load_lexicons()
            self.is_loaded = True
            logger.info("Rule-based model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load rule-based model: {e}")
            raise
    
    async def _load_lexicons(self) -> None:
        """Load sentiment lexicons"""
        # Default Vietnamese sentiment words
        self.positive_words = {
            "tốt", "hay", "đẹp", "tuyệt", "xuất sắc", "hoàn hảo", "tuyệt vời",
            "thích", "yêu", "hài lòng", "vui", "hạnh phúc", "tích cực",
            "ưng ý", "thú vị", "hữu ích", "chất lượng", "hiệu quả",
            "đáng tin cậy", "an toàn", "tiện lợi", "nhanh chóng", "dễ dàng",
            "rẻ", "hợp lý", "phù hợp", "đẹp mắt", "sang trọng", "tinh tế",
            "ổn", "ok", "được", "good", "nice", "great", "excellent", "amazing"
        }
        
        self.negative_words = {
            "xấu", "tệ", "kém", "dở", "khủng khiếp", "tồi tệ", "thất vọng",
            "không thích", "ghét", "không hài lòng", "buồn", "tức giận",
            "tiêu cực", "khó chịu", "phiền", "chậm", "lỗi", "hỏng",
            "không tốt", "không hay", "không đẹp", "không ổn", "không được",
            "đắt", "không hợp lý", "không phù hợp", "khó khăn", "phức tạp",
            "nguy hiểm", "không an toàn", "bad", "terrible", "awful", "poor"
        }
        
        self.intensifiers = {
            "rất", "cực", "cực kỳ", "vô cùng", "hết sức", "thật sự",
            "thực sự", "quá", "too", "very", "extremely", "really", "super"
        }
        
        self.negators = {
            "không", "chẳng", "chả", "chưa", "never", "not", "no"
        }
        
        # Try to load from files if they exist
        data_dir = Path("data/lexicons")
        if data_dir.exists():
            await self._load_lexicon_files(data_dir)
    
    async def _load_lexicon_files(self, data_dir: Path) -> None:
        """Load lexicon files if available"""
        try:
            positive_file = data_dir / "positive_vi.txt"
            if positive_file.exists():
                with open(positive_file, "r", encoding="utf-8") as f:
                    self.positive_words.update(line.strip() for line in f if line.strip())
            
            negative_file = data_dir / "negative_vi.txt"
            if negative_file.exists():
                with open(negative_file, "r", encoding="utf-8") as f:
                    self.negative_words.update(line.strip() for line in f if line.strip())
                    
        except Exception as e:
            logger.warning(f"Could not load lexicon files: {e}")
    
    async def predict(self, text: str, language: str = "vi") -> SentimentPrediction:
        """Predict sentiment for single text"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        # Normalize text
        normalized_text = await self.normalizer.normalize(text)
        
        # Tokenize
        tokens = await self.tokenizer.tokenize(normalized_text)
        
        # Calculate sentiment score
        score = self._calculate_sentiment_score(tokens)
        
        # Determine label
        if score > self.positive_threshold:
            label = "positive"
        elif score < self.negative_threshold:
            label = "negative"
        else:
            label = "neutral"
        
        # Convert score to 0-1 range
        confidence = min(abs(score), 1.0)
        
        # Calculate detailed scores
        scores = self._calculate_detailed_scores(score)
        
        return SentimentPrediction(
            label=label,
            score=confidence,
            scores=scores,
            metadata={
                "raw_score": score,
                "tokens_count": len(tokens),
                "normalized_text": normalized_text
            }
        )
    
    async def predict_batch(self, texts: List[str], language: str = "vi") -> List[SentimentPrediction]:
        """Predict sentiment for batch of texts"""
        tasks = [self.predict(text, language) for text in texts]
        return await asyncio.gather(*tasks)
    
    def _calculate_sentiment_score(self, tokens: List[str]) -> float:
        """Calculate sentiment score from tokens"""
        score = 0.0
        i = 0
        
        while i < len(tokens):
            token = tokens[i].lower()
            
            # Check for intensifier
            intensifier_multiplier = 1.0
            if i > 0 and tokens[i-1].lower() in self.intensifiers:
                intensifier_multiplier = self.intensifier_boost
            
            # Check for negator
            negator_multiplier = 1.0
            if i > 0 and tokens[i-1].lower() in self.negators:
                negator_multiplier = self.negator_flip
            elif i > 1 and tokens[i-2].lower() in self.negators:
                negator_multiplier = self.negator_flip
            
            # Calculate token score
            token_score = 0.0
            if token in self.positive_words:
                token_score = 1.0
            elif token in self.negative_words:
                token_score = -1.0
            
            # Apply modifiers
            final_score = token_score * intensifier_multiplier * negator_multiplier
            score += final_score
            
            i += 1
        
        # Normalize by text length
        if len(tokens) > 0:
            score = score / len(tokens)
        
        return score
    
    def _calculate_detailed_scores(self, raw_score: float) -> Dict[str, float]:
        """Calculate detailed class scores"""
        # Simple heuristic mapping
        if raw_score > self.positive_threshold:
            pos_score = min(0.5 + abs(raw_score), 1.0)
            neg_score = max(0.1, 1.0 - pos_score)
            neu_score = max(0.1, 1.0 - pos_score - neg_score)
        elif raw_score < self.negative_threshold:
            neg_score = min(0.5 + abs(raw_score), 1.0)
            pos_score = max(0.1, 1.0 - neg_score)
            neu_score = max(0.1, 1.0 - pos_score - neg_score)
        else:
            neu_score = 0.6
            pos_score = 0.2
            neg_score = 0.2
        
        # Normalize to sum to 1.0
        total = pos_score + neg_score + neu_score
        return {
            "positive": round(pos_score / total, 3),
            "negative": round(neg_score / total, 3),
            "neutral": round(neu_score / total, 3)
        }
    
    async def unload(self) -> None:
        """Unload the model"""
        self.positive_words.clear()
        self.negative_words.clear()
        self.intensifiers.clear()
        self.negators.clear()
        self.is_loaded = False
        logger.info("Rule-based model unloaded")
    
    @property
    def supported_languages(self) -> List[str]:
        """Get supported languages"""
        return ["vi", "en"]
