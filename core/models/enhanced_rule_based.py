"""
Enhanced Rule-based sentiment analysis model for Vietnamese text
Advanced implementation with weighted lexicons, pattern recognition, and context analysis
"""

import re
import logging
from typing import Dict, List, Optional, Set, Tuple
import asyncio
from pathlib import Path
import math

from core.models.base import SentimentModel, SentimentPrediction
from core.nlp.normalizers import VietnameseNormalizer
from core.nlp.tokenizers import VietnameseTokenizer

logger = logging.getLogger(__name__)

class EnhancedRuleBasedModel(SentimentModel):
    """Enhanced rule-based sentiment model with advanced features"""

    def __init__(self):
        super().__init__("enhanced_rule_based_vi", "2.0")
        self.normalizer = VietnameseNormalizer()
        self.tokenizer = VietnameseTokenizer()

        # Enhanced lexicons with weights
        self.weighted_lexicon: Dict[str, Dict[str, float]] = {}
        self.patterns: List[Dict] = []
        self.emoji_sentiment: Dict[str, Tuple[float, float]] = {}  # emoji -> (score, confidence)

        # Context analysis components
        self.negation_window = 3  # Look back/forward 3 words for negation
        self.intensifier_window = 2  # Look back 2 words for intensifiers

        # Advanced thresholds
        self.positive_threshold = 0.15
        self.negative_threshold = -0.15
        self.confidence_threshold = 0.5

        # Pattern weights
        self.pattern_weights = {
            'negation': 0.8,
            'intensifier': 1.0,
            'comparison': 0.9,
            'sarcasm': 0.7,
            'temporal': 0.6,
            'conditional': 0.4
        }

    async def load(self) -> None:
        """Load enhanced lexicons and patterns"""
        logger.info("Loading enhanced rule-based sentiment model...")

        try:
            await self._load_weighted_lexicon()
            await self._load_patterns()
            await self._load_emoji_sentiment()
            await self._load_sarcasm_patterns()
            await self._load_temporal_patterns()
            self.is_loaded = True
            logger.info("Enhanced rule-based model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load enhanced model: {e}")
            raise

    async def _load_weighted_lexicon(self) -> None:
        """Load weighted lexicon from enhanced file"""
        lexicon_file = Path("data/lexicons/enhanced_lexicon_vi.txt")
        if not lexicon_file.exists():
            logger.warning("Enhanced lexicon file not found, using default")
            return

        with open(lexicon_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        parts = line.split('|')
                        if len(parts) >= 4:
                            word = parts[0].strip()
                            score = float(parts[1])
                            category = parts[2].strip()
                            domain = parts[3].strip()

                            if word not in self.weighted_lexicon:
                                self.weighted_lexicon[word] = {}

                            self.weighted_lexicon[word] = {
                                'score': score,
                                'category': category,
                                'domain': domain,
                                'confidence': min(abs(score) / 3.0, 1.0)
                            }
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Invalid lexicon line: {line} - {e}")

        # Load comprehensive sentiment lexicon
        await self._load_comprehensive_lexicon()

        logger.info(f"Loaded {len(self.weighted_lexicon)} weighted terms")

    async def _load_comprehensive_lexicon(self) -> None:
        """Load comprehensive domain-specific lexicon"""
        comp_file = Path("data/lexicons/comprehensive_sentiment_vi.txt")
        if comp_file.exists():
            with open(comp_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '|' in line:
                        try:
                            parts = line.split('|')
                            if len(parts) >= 5:
                                word = parts[0].strip()
                                score = float(parts[1])
                                domain = parts[2].strip()
                                subcategory = parts[3].strip()
                                confidence = float(parts[4])

                                self.weighted_lexicon[word] = {
                                    'score': score,
                                    'category': 'domain_specific',
                                    'domain': domain,
                                    'subcategory': subcategory,
                                    'confidence': confidence
                                }
                        except (ValueError, IndexError) as e:
                            logger.warning(f"Invalid comprehensive lexicon line: {line} - {e}")

    async def _load_patterns(self) -> None:
        """Load grammar patterns and idioms"""
        patterns_file = Path("data/lexicons/patterns_vi.txt")
        if not patterns_file.exists():
            logger.warning("Patterns file not found")
            return

        with open(patterns_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        parts = line.split('|')
                        if len(parts) >= 4:
                            pattern = parts[0].strip()
                            score = parts[1].strip()
                            confidence = float(parts[2])
                            explanation = parts[3].strip()

                            # Convert pattern to regex if needed
                            regex_pattern = self._convert_pattern_to_regex(pattern)

                            self.patterns.append({
                                'pattern': pattern,
                                'regex': regex_pattern,
                                'score': score,
                                'confidence': confidence,
                                'explanation': explanation
                            })
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Invalid pattern line: {line} - {e}")

        logger.info(f"Loaded {len(self.patterns)} sentiment patterns")

    async def _load_emoji_sentiment(self) -> None:
        """Load emoji sentiment mappings"""
        emoji_file = Path("data/lexicons/emoji_sentiment.txt")
        if not emoji_file.exists():
            logger.warning("Emoji sentiment file not found")
            return

        with open(emoji_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        parts = line.split('|')
                        if len(parts) >= 4:
                            emoji = parts[0].strip()
                            score = float(parts[1])
                            confidence = float(parts[2])
                            category = parts[3].strip()

                            self.emoji_sentiment[emoji] = (score, confidence)
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Invalid emoji line: {line} - {e}")

        logger.info(f"Loaded {len(self.emoji_sentiment)} emoji sentiments")

    async def _load_sarcasm_patterns(self) -> None:
        """Load sarcasm patterns from file"""
        self.sarcasm_patterns = []
        self.sarcasm_context_words = {}

        sarcasm_file = Path("data/lexicons/sarcasm_patterns_vi.txt")
        if sarcasm_file.exists():
            with open(sarcasm_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        try:
                            parts = line.split('|')
                            if len(parts) >= 3:
                                phrase = parts[0].strip()
                                intensity = float(parts[1])
                                context_required = parts[2].strip().lower() == 'true'

                                self.sarcasm_patterns.append({
                                    'phrase': phrase,
                                    'intensity': intensity,
                                    'context_required': context_required
                                })
                            elif len(parts) == 2:  # Context words
                                word = parts[0].strip()
                                weight = float(parts[1])
                                self.sarcasm_context_words[word] = weight
                        except (ValueError, IndexError) as e:
                            logger.warning(f"Invalid sarcasm pattern line: {line} - {e}")

    async def _load_temporal_patterns(self) -> None:
        """Load temporal patterns from file"""
        self.temporal_patterns = []
        self.temporal_positive_words = []
        self.temporal_negative_words = []

        temporal_file = Path("data/lexicons/temporal_patterns_vi.txt")
        if temporal_file.exists():
            with open(temporal_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        try:
                            if line.startswith('positive_words='):
                                words = line.split('=')[1].split(',')
                                self.temporal_positive_words = [w.strip() for w in words]
                            elif line.startswith('negative_words='):
                                words = line.split('=')[1].split(',')
                                self.temporal_negative_words = [w.strip() for w in words]
                            elif '|' in line:
                                parts = line.split('|')
                                if len(parts) >= 4:
                                    start_phrase = parts[0].strip()
                                    connector = parts[1].strip()
                                    end_phrase = parts[2].strip()
                                    weight = float(parts[3])

                                    self.temporal_patterns.append({
                                        'start': start_phrase,
                                        'connector': connector,
                                        'end': end_phrase,
                                        'weight': weight
                                    })
                        except (ValueError, IndexError) as e:
                            logger.warning(f"Invalid temporal pattern line: {line} - {e}")

    def _convert_pattern_to_regex(self, pattern: str) -> str:
        """Convert pattern template to regex"""
        # Replace placeholders with regex groups
        regex = pattern
        regex = regex.replace('{adj}', r'(\w+)')
        regex = regex.replace('{noun}', r'(\w+)')
        regex = regex.replace('{sentiment}', r'(\w+)')
        regex = regex.replace('{sentiment1}', r'(\w+)')
        regex = regex.replace('{sentiment2}', r'(\w+)')
        regex = regex.replace('{condition}', r'([\w\s]+)')
        return regex

    async def predict(self, text: str, language: str = "vi") -> SentimentPrediction:
        """Enhanced sentiment prediction with context analysis"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        # Normalize and tokenize
        normalized_text = await self.normalizer.normalize(text)
        tokens = await self.tokenizer.tokenize(normalized_text)

        # Multi-level analysis
        lexicon_score, lexicon_confidence = self._analyze_lexicon(tokens, normalized_text)
        pattern_score, pattern_confidence = self._analyze_patterns(normalized_text)
        emoji_score, emoji_confidence = self._analyze_emojis(text)
        context_score, context_confidence = self._analyze_context(tokens, normalized_text)

        # Weighted combination
        total_score = (
            lexicon_score * 0.4 +
            pattern_score * 0.3 +
            emoji_score * 0.2 +
            context_score * 0.1
        )

        # Combined confidence
        total_confidence = (
            lexicon_confidence * 0.4 +
            pattern_confidence * 0.3 +
            emoji_confidence * 0.2 +
            context_confidence * 0.1
        )

        # Determine label
        if total_score > self.positive_threshold:
            label = "positive"
        elif total_score < self.negative_threshold:
            label = "negative"
        else:
            label = "neutral"

        # Calculate detailed scores
        scores = self._calculate_enhanced_scores(total_score, total_confidence)

        return SentimentPrediction(
            label=label,
            score=min(total_confidence, 1.0),
            scores=scores,
            metadata={
                'total_score': total_score,
                'lexicon_score': lexicon_score,
                'pattern_score': pattern_score,
                'emoji_score': emoji_score,
                'context_score': context_score,
                'confidence_breakdown': {
                    'lexicon': lexicon_confidence,
                    'pattern': pattern_confidence,
                    'emoji': emoji_confidence,
                    'context': context_confidence
                },
                'tokens_count': len(tokens),
                'normalized_text': normalized_text
            }
        )

    def _analyze_lexicon(self, tokens: List[str], text: str) -> Tuple[float, float]:
        """Enhanced lexicon analysis with context awareness"""
        total_score = 0.0
        total_confidence = 0.0
        matched_terms = 0

        # Check individual tokens and phrases
        for i, token in enumerate(tokens):
            token_lower = token.lower()

            # Check for exact matches
            if token_lower in self.weighted_lexicon:
                term_data = self.weighted_lexicon[token_lower]
                base_score = term_data['score']
                base_confidence = term_data['confidence']

                # Apply context modifiers
                modified_score, confidence_modifier = self._apply_context_modifiers(
                    base_score, i, tokens, text
                )

                total_score += modified_score
                total_confidence += base_confidence * confidence_modifier
                matched_terms += 1

        # Check for phrase matches
        phrase_score, phrase_confidence, phrase_matches = self._check_phrases(tokens, text)
        total_score += phrase_score
        total_confidence += phrase_confidence
        matched_terms += phrase_matches

        # Normalize by text length with minimum threshold
        if len(tokens) > 0:
            total_score = total_score / max(len(tokens), 3)

        if matched_terms > 0:
            total_confidence = total_confidence / matched_terms
        else:
            total_confidence = 0.1  # Low confidence for no matches

        return total_score, min(total_confidence, 1.0)

    def _apply_context_modifiers(self, base_score: float, position: int,
                               tokens: List[str], text: str) -> Tuple[float, float]:
        """Apply intensifiers, negators, and other context modifiers"""
        modified_score = base_score
        confidence_modifier = 1.0

        # Check for intensifiers (look back)
        for i in range(max(0, position - self.intensifier_window), position):
            token = tokens[i].lower()
            if token in self.weighted_lexicon:
                term_data = self.weighted_lexicon[token]
                if term_data.get('category') == 'intensifier':
                    intensifier_strength = abs(term_data['score'])
                    modified_score *= intensifier_strength
                    confidence_modifier *= 1.2
                    break  # Only apply one intensifier

        # Check for negators (look back and forward)
        negation_found = False
        for i in range(max(0, position - self.negation_window),
                      min(len(tokens), position + self.negation_window + 1)):
            if i != position:
                token = tokens[i].lower()
                if token in self.weighted_lexicon:
                    term_data = self.weighted_lexicon[token]
                    if term_data.get('category') == 'negator':
                        negation_strength = abs(term_data['score'])
                        modified_score *= -negation_strength
                        confidence_modifier *= 0.9  # Slightly reduce confidence for negation
                        negation_found = True
                        break

        # Special handling for partial negations and complex patterns
        if not negation_found:
            text_lower = text.lower()
            partial_negations = {
                'không hẳn': 0.3,
                'chưa thể': 0.4,
                'có gì đâu': 0.2,
                'đâu có': 0.2,
                'làm gì có': 0.2
            }

            for neg_phrase, multiplier in partial_negations.items():
                if neg_phrase in text_lower:
                    modified_score *= multiplier
                    confidence_modifier *= 0.7
                    break

        return modified_score, confidence_modifier

    def _check_phrases(self, tokens: List[str], text: str) -> Tuple[float, float, int]:
        """Check for multi-word phrases in lexicon"""
        total_score = 0.0
        total_confidence = 0.0
        matched_phrases = 0

        # Check 2-word phrases
        for i in range(len(tokens) - 1):
            phrase = f"{tokens[i]} {tokens[i+1]}".lower()
            if phrase in self.weighted_lexicon:
                term_data = self.weighted_lexicon[phrase]
                total_score += term_data['score']
                total_confidence += term_data['confidence']
                matched_phrases += 1

        # Check 3-word phrases
        for i in range(len(tokens) - 2):
            phrase = f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}".lower()
            if phrase in self.weighted_lexicon:
                term_data = self.weighted_lexicon[phrase]
                total_score += term_data['score'] * 1.2  # Boost for longer phrases
                total_confidence += term_data['confidence']
                matched_phrases += 1

        return total_score, total_confidence, matched_phrases

    def _analyze_patterns(self, text: str) -> Tuple[float, float]:
        """Analyze text for grammatical patterns and idioms"""
        total_score = 0.0
        total_confidence = 0.0
        matched_patterns = 0

        text_lower = text.lower()

        # Special sarcasm detection
        sarcasm_score, sarcasm_confidence = self._detect_sarcasm(text_lower)
        if sarcasm_score != 0:
            total_score += sarcasm_score
            total_confidence += sarcasm_confidence
            matched_patterns += 1

        # Temporal pattern analysis
        temporal_score, temporal_confidence = self._analyze_temporal_patterns(text_lower)
        if temporal_score != 0:
            total_score += temporal_score
            total_confidence += temporal_confidence
            matched_patterns += 1

        # Other patterns from file
        for pattern_data in self.patterns:
            pattern = pattern_data['pattern'].lower()

            if pattern in text_lower:
                try:
                    score = float(pattern_data['score']) if pattern_data['score'] != 'temporal' else 0
                    confidence = pattern_data['confidence']

                    total_score += score
                    total_confidence += confidence
                    matched_patterns += 1

                except ValueError:
                    continue

        if matched_patterns > 0:
            total_confidence = total_confidence / matched_patterns
        else:
            total_confidence = 0.0

        return total_score, min(total_confidence, 1.0)

    def _detect_sarcasm(self, text: str) -> Tuple[float, float]:
        """Advanced sarcasm detection using loaded patterns"""
        # Use loaded sarcasm patterns instead of hard-coded
        if hasattr(self, 'sarcasm_patterns') and self.sarcasm_patterns:
            for pattern_data in self.sarcasm_patterns:
                phrase = pattern_data['phrase']
                intensity = pattern_data['intensity']
                context_required = pattern_data['context_required']

                if phrase in text:
                    if context_required and hasattr(self, 'sarcasm_context_words'):
                        # Check for negative context words with weights
                        context_score = 0
                        for context_word, weight in self.sarcasm_context_words.items():
                            if context_word in text:
                                context_score += weight

                        if context_score > 0.5:  # Threshold for sarcasm detection
                            return intensity, 0.8
                    elif not context_required:
                        return intensity, 0.7

        # Fallback to basic detection if files not loaded
        basic_indicators = [
            ('hay quá đi', -1.5),
            ('tuyệt quá', -1.2),
            ('perfect', -1.0),
            ('tuyệt vời', -1.3)
        ]

        for indicator, score in basic_indicators:
            if indicator in text:
                negative_context = ['hỏng', 'lỗi', 'chờ', 'lâu', 'tệ', 'thất vọng']
                if any(neg_word in text for neg_word in negative_context):
                    return score, 0.8

        return 0.0, 0.0

    def _analyze_temporal_patterns(self, text: str) -> Tuple[float, float]:
        """Analyze temporal sentiment patterns using loaded data"""
        # Use loaded temporal patterns instead of hard-coded
        if hasattr(self, 'temporal_patterns') and self.temporal_patterns:
            for pattern_data in self.temporal_patterns:
                start = pattern_data['start']
                connector = pattern_data['connector']
                end = pattern_data['end']
                weight = pattern_data['weight']

                if start in text and connector in text and end in text:
                    # Extract sentiment after temporal transition
                    parts = text.split(connector)
                    if len(parts) == 2:
                        later_part = parts[1]

                        # Use loaded word lists for analysis
                        positive_words = getattr(self, 'temporal_positive_words', ['thích', 'tốt', 'hay', 'ổn', 'hài lòng'])
                        negative_words = getattr(self, 'temporal_negative_words', ['tệ', 'thất vọng', 'không thích', 'kém'])

                        pos_count = sum(1 for word in positive_words if word in later_part)
                        neg_count = sum(1 for word in negative_words if word in later_part)

                        if pos_count > neg_count:
                            return 1.0 * weight, 0.8
                        elif neg_count > pos_count:
                            return -1.0 * weight, 0.8

        # Fallback to basic patterns if files not loaded
        basic_patterns = [
            ('ban đầu', 'nhưng', 'sau', 0.7),
            ('lúc đầu', 'nhưng', 'giờ', 0.8),
            ('trước', 'nhưng', 'bây giờ', 0.8)
        ]

        for start, connector, end, weight in basic_patterns:
            if start in text and connector in text and end in text:
                parts = text.split(connector)
                if len(parts) == 2:
                    later_part = parts[1]
                    positive_words = ['thích', 'tốt', 'hay', 'ổn', 'hài lòng']
                    negative_words = ['tệ', 'thất vọng', 'không thích', 'kém']

                    pos_count = sum(1 for word in positive_words if word in later_part)
                    neg_count = sum(1 for word in negative_words if word in later_part)

                    if pos_count > neg_count:
                        return 1.0 * weight, 0.8
                    elif neg_count > pos_count:
                        return -1.0 * weight, 0.8

        return 0.0, 0.0

    def _analyze_emojis(self, text: str) -> Tuple[float, float]:
        """Analyze emoji sentiment"""
        total_score = 0.0
        total_confidence = 0.0
        emoji_count = 0

        for emoji, (score, confidence) in self.emoji_sentiment.items():
            emoji_occurrences = text.count(emoji)
            if emoji_occurrences > 0:
                # Multiple same emojis increase intensity
                intensity_multiplier = min(1 + (emoji_occurrences - 1) * 0.3, 2.0)
                total_score += score * intensity_multiplier
                total_confidence += confidence
                emoji_count += 1

        if emoji_count > 0:
            total_confidence = total_confidence / emoji_count
        else:
            total_confidence = 0.0

        return total_score, min(total_confidence, 1.0)

    def _analyze_context(self, tokens: List[str], text: str) -> Tuple[float, float]:
        """Advanced context analysis"""
        context_score = 0.0
        context_confidence = 0.0

        # Analyze punctuation patterns
        exclamation_count = text.count('!')
        question_count = text.count('?')
        dots_count = text.count('...')

        # Exclamation marks boost intensity
        if exclamation_count > 0:
            context_score += min(exclamation_count * 0.2, 0.6)
            context_confidence += 0.3

        # Multiple question marks suggest uncertainty
        if question_count > 1:
            context_score -= 0.2
            context_confidence += 0.2

        # Ellipsis suggests hesitation or negative sentiment
        if dots_count > 0:
            context_score -= dots_count * 0.1
            context_confidence += 0.2

        # Capitalization analysis
        caps_count = sum(1 for c in text if c.isupper())
        if caps_count > len(text) * 0.3:  # High ratio of caps
            context_score += 0.3  # Emphasis
            context_confidence += 0.2

        return context_score, min(context_confidence, 1.0)

    def _calculate_enhanced_scores(self, total_score: float, confidence: float) -> Dict[str, float]:
        """Calculate enhanced probability scores for each class"""
        # Use sigmoid function for more natural probability distribution
        def sigmoid(x):
            return 1 / (1 + math.exp(-x * 3))  # Scale factor for sensitivity

        if total_score > self.positive_threshold:
            pos_prob = sigmoid(total_score) * confidence
            neg_prob = (1 - sigmoid(total_score)) * confidence * 0.3
            neu_prob = max(0.1, 1 - pos_prob - neg_prob)
        elif total_score < self.negative_threshold:
            neg_prob = sigmoid(-total_score) * confidence
            pos_prob = (1 - sigmoid(-total_score)) * confidence * 0.3
            neu_prob = max(0.1, 1 - pos_prob - neg_prob)
        else:
            neu_prob = 0.6 + (1 - confidence) * 0.2
            pos_prob = neg_prob = (1 - neu_prob) / 2

        # Normalize to sum to 1.0
        total = pos_prob + neg_prob + neu_prob
        return {
            "positive": round(pos_prob / total, 4),
            "negative": round(neg_prob / total, 4),
            "neutral": round(neu_prob / total, 4)
        }

    async def predict_batch(self, texts: List[str], language: str = "vi") -> List[SentimentPrediction]:
        """Batch prediction with async processing"""
        tasks = [self.predict(text, language) for text in texts]
        return await asyncio.gather(*tasks)

    async def unload(self) -> None:
        """Unload the enhanced model"""
        self.weighted_lexicon.clear()
        self.patterns.clear()
        self.emoji_sentiment.clear()
        self.is_loaded = False
        logger.info("Enhanced rule-based model unloaded")

    @property
    def supported_languages(self) -> List[str]:
        """Get supported languages"""
        return ["vi", "en"]
