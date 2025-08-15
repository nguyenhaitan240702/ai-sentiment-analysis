"""
Test data for enhanced sentiment analysis model
Contains challenging cases to test accuracy improvements
"""

# Test cases with expected results for validation
TEST_CASES = [
    # === BASIC SENTIMENT ===
    {
        "text": "Tôi rất thích sản phẩm này",
        "expected": "positive",
        "difficulty": "easy"
    },
    {
        "text": "Sản phẩm này thật tệ",
        "expected": "negative",
        "difficulty": "easy"
    },
    {
        "text": "Bình thường thôi",
        "expected": "neutral",
        "difficulty": "easy"
    },

    # === NEGATION PATTERNS ===
    {
        "text": "Tôi không thích sản phẩm này",
        "expected": "negative",
        "difficulty": "medium"
    },
    {
        "text": "Không hẳn là tốt",
        "expected": "negative",
        "difficulty": "medium"
    },
    {
        "text": "Chưa thể nào tệ được",
        "expected": "positive",
        "difficulty": "hard"
    },
    {
        "text": "Có gì đâu mà tệ",
        "expected": "positive",
        "difficulty": "hard"
    },

    # === INTENSIFIERS ===
    {
        "text": "Cực kỳ tuyệt vời",
        "expected": "positive",
        "difficulty": "medium"
    },
    {
        "text": "Vô cùng thất vọng",
        "expected": "negative",
        "difficulty": "medium"
    },
    {
        "text": "Siêu tệ luôn",
        "expected": "negative",
        "difficulty": "medium"
    },

    # === COMPARISON PATTERNS ===
    {
        "text": "Tốt hơn kỳ vọng",
        "expected": "positive",
        "difficulty": "medium"
    },
    {
        "text": "Không tốt như tưởng",
        "expected": "negative",
        "difficulty": "medium"
    },
    {
        "text": "So với giá thì chất lượng ổn",
        "expected": "positive",
        "difficulty": "hard"
    },

    # === TEMPORAL PATTERNS ===
    {
        "text": "Ban đầu tốt nhưng sau này tệ",
        "expected": "negative",
        "difficulty": "hard"
    },
    {
        "text": "Lúc đầu thất vọng nhưng giờ thích rồi",
        "expected": "positive",
        "difficulty": "hard"
    },

    # === SARCASM/IRONY ===
    {
        "text": "Hay quá đi! Lại hỏng nữa rồi",
        "expected": "negative",
        "difficulty": "very_hard"
    },
    {
        "text": "Tuyệt quá! Chờ 2 tiếng mới được phục vụ",
        "expected": "negative",
        "difficulty": "very_hard"
    },

    # === IDIOMS ===
    {
        "text": "Như cá gặp nước",
        "expected": "positive",
        "difficulty": "medium"
    },
    {
        "text": "Như chuối ba gang",
        "expected": "negative",
        "difficulty": "medium"
    },

    # === EMOJIS ===
    {
        "text": "Sản phẩm ổn 😊",
        "expected": "positive",
        "difficulty": "easy"
    },
    {
        "text": "Thất vọng quá 😢😢😢",
        "expected": "negative",
        "difficulty": "easy"
    },
    {
        "text": "Tạm được :)",
        "expected": "positive",
        "difficulty": "easy"
    },

    # === CONDITIONAL STATEMENTS ===
    {
        "text": "Nếu rẻ hơn thì tốt",
        "expected": "neutral",
        "difficulty": "hard"
    },
    {
        "text": "Giá mà có thêm tính năng X thì perfect",
        "expected": "neutral",
        "difficulty": "hard"
    },

    # === ASPECT-BASED ===
    {
        "text": "Đồ ăn ngon nhưng dịch vụ tệ",
        "expected": "neutral",
        "difficulty": "very_hard"
    },
    {
        "text": "Giá đắt nhưng chất lượng tốt",
        "expected": "positive",
        "difficulty": "hard"
    },

    # === VIETNAMESE SLANG ===
    {
        "text": "Bá cháy luôn",
        "expected": "positive",
        "difficulty": "medium"
    },
    {
        "text": "Xịn sò phết",
        "expected": "positive",
        "difficulty": "medium"
    },
    {
        "text": "Chill phết",
        "expected": "positive",
        "difficulty": "medium"
    },

    # === COMPLEX CASES ===
    {
        "text": "Ban đầu tôi rất thất vọng về sản phẩm này, nhưng sau khi sử dụng một thời gian thì thấy cũng ổn, không tệ như tưởng",
        "expected": "positive",
        "difficulty": "very_hard"
    },
    {
        "text": "Tôi không nói là không thích, nhưng cũng chẳng có gì đặc biệt",
        "expected": "neutral",
        "difficulty": "very_hard"
    },
    {
        "text": "Với cái giá này mà có chất lượng như vậy thì quá tuyệt vời rồi 😍😍",
        "expected": "positive",
        "difficulty": "hard"
    }
]

# Edge cases for testing robustness
EDGE_CASES = [
    {
        "text": "",
        "expected": "neutral",
        "description": "empty string"
    },
    {
        "text": "😊😢😐",
        "expected": "neutral",
        "description": "mixed emojis"
    },
    {
        "text": "Tốt tốt tốt tốt tốt",
        "expected": "positive",
        "description": "repetitive positive"
    },
    {
        "text": "Tệ tệ tệ tệ tệ",
        "expected": "negative",
        "description": "repetitive negative"
    },
    {
        "text": "SIÊU TUYỆT VỜI!!!",
        "expected": "positive",
        "description": "all caps with exclamation"
    },
    {
        "text": "không không không không",
        "expected": "negative",
        "description": "multiple negations"
    }
]

# Performance benchmark texts
BENCHMARK_TEXTS = [
    "Tôi rất hài lòng với chất lượng dịch vụ",
    "Sản phẩm này không đáng tiền",
    "Bình thường, không có gì đặc biệt",
    "Cực kỳ thất vọng về trải nghiệm này",
    "Tạm được, chấp nhận được",
    "Quá tuyệt vời, vượt ngoài mong đợi",
    "Chưa bao giờ thấy tệ đến thế",
    "Ổn, đúng như mô tả",
    "Không thể tốt hơn được nữa",
    "Hoàn toàn không hài lòng"
]
