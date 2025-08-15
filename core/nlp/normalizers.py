"""
Vietnamese text normalizers for AI Sentiment Analysis
Handles Unicode normalization, TELEX conversion, emoji processing
"""

import re
import unicodedata
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class VietnameseNormalizer:
    """Vietnamese text normalizer"""

    def __init__(self):
        # TELEX to Unicode mapping
        self.telex_map = {
            'aa': 'â', 'aw': 'ă', 'ee': 'ê', 'oo': 'ô', 'ow': 'ơ', 'uw': 'ư',
            'dd': 'đ', 'Aa': 'Â', 'Aw': 'Ă', 'Ee': 'Ê', 'Oo': 'Ô', 'Ow': 'Ơ',
            'Uw': 'Ư', 'Dd': 'Đ', 'DD': 'Đ'
        }

        # Tone marks
        self.tone_map = {
            'as': 'á', 'af': 'à', 'ar': 'ả', 'ax': 'ã', 'aj': 'ạ',
            'es': 'é', 'ef': 'è', 'er': 'ẻ', 'ex': 'ẽ', 'ej': 'ẹ',
            'is': 'í', 'if': 'ì', 'ir': 'ỉ', 'ix': 'ĩ', 'ij': 'ị',
            'os': 'ó', 'of': 'ò', 'or': 'ỏ', 'ox': 'õ', 'oj': 'ọ',
            'us': 'ú', 'uf': 'ù', 'ur': 'ủ', 'ux': 'ũ', 'uj': 'ụ',
            'ys': 'ý', 'yf': 'ỳ', 'yr': 'ỷ', 'yx': 'ỹ', 'yj': 'ỵ'
        }

        # Common emoji patterns
        self.emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )

        # Vietnamese slang/abbreviation mappings
        self.slang_map = {
            'k': 'không',
            'ko': 'không',
            'dc': 'được',
            'đc': 'được',
            'vs': 'với',
            'w': 'với',
            'r': 'rồi',
            'ms': 'mới',
            'nx': 'nữa',
            'ntn': 'như thế nào',
            'sao': 'sao',
            'ok': 'được',
            'oke': 'được',
            'tks': 'cảm ơn',
            'thanks': 'cảm ơn',
            'ty': 'cảm ơn',
            'nó': 'nó',
            'ông': 'ông',
            'bà': 'bà',
            'thg': 'thằng',
            'đứa': 'đứa',
            'kg': 'không',
            'kh': 'không',
            'cx': 'cũng',
            'bh': 'bây giờ',
            'h': 'giờ',
            'lm': 'làm',
            'zậy': 'vậy',
            'z': 'gì',
            'j': 'gì',
            'wtf': 'gì thế',
            'omg': 'ôi'
        }

    async def normalize(self, text: str) -> str:
        """Normalize Vietnamese text"""
        if not text:
            return ""

        # Unicode normalization (NFC)
        text = unicodedata.normalize('NFC', text)

        # Convert TELEX to Unicode
        text = self._convert_telex(text)

        # Process emojis (convert to text or remove)
        text = self._process_emojis(text)

        # Expand Vietnamese slang
        text = self._expand_slang(text)

        # Clean whitespace
        text = re.sub(r'\s+', ' ', text.strip())

        # Optional: convert to lowercase (preserving Vietnamese characters)
        text = text.lower()

        return text

    def _convert_telex(self, text: str) -> str:
        """Convert TELEX input to Unicode Vietnamese"""
        # Sort by length (longer first) to avoid conflicts
        sorted_telex = sorted(self.telex_map.items(), key=lambda x: len(x[0]), reverse=True)

        for telex, unicode_char in sorted_telex:
            text = text.replace(telex, unicode_char)

        # Handle tone marks
        for tone, char in self.tone_map.items():
            text = text.replace(tone, char)

        return text

    def _process_emojis(self, text: str) -> str:
        """Process emojis - convert to sentiment indicators or remove"""
        # Simple emoji to sentiment mapping
        emoji_sentiment_map = {
            '😊': ' tích cực ',
            '😍': ' thích ',
            '😂': ' vui ',
            '😭': ' buồn ',
            '😡': ' tức giận ',
            '😔': ' buồn ',
            '👍': ' tốt ',
            '👎': ' không tốt ',
            '❤️': ' yêu ',
            '💔': ' buồn ',
            '🔥': ' tuyệt vời ',
            '💯': ' hoàn hảo '
        }

        # Replace known emojis with sentiment words
        for emoji, sentiment in emoji_sentiment_map.items():
            text = text.replace(emoji, sentiment)

        # Remove remaining emojis
        text = self.emoji_pattern.sub(' ', text)

        return text

    def _expand_slang(self, text: str) -> str:
        """Expand Vietnamese internet slang"""
        words = text.split()
        expanded_words = []

        for word in words:
            # Check if word is slang
            lower_word = word.lower()
            if lower_word in self.slang_map:
                expanded_words.append(self.slang_map[lower_word])
            else:
                expanded_words.append(word)

        return ' '.join(expanded_words)
