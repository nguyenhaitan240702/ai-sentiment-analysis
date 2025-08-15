"""
Unit tests for Vietnamese sentiment analysis rule-based model
"""

import pytest
import asyncio
from core.models.rule_based import RuleBasedModel
from core.models.base import SentimentPrediction

class TestRuleBasedModel:
    """Test suite for rule-based sentiment model"""

    @pytest.fixture
    async def model(self):
        """Create and load model for testing"""
        model = RuleBasedModel()
        await model.load()
        yield model
        await model.unload()

    @pytest.mark.asyncio
    async def test_positive_sentiment(self, model):
        """Test positive sentiment detection"""
        text = "Tôi rất thích sản phẩm này! Chất lượng tuyệt vời."
        result = await model.predict(text)

        assert isinstance(result, SentimentPrediction)
        assert result.label == "positive"
        assert result.score > 0.5
        assert "positive" in result.scores
        assert result.metadata is not None

    @pytest.mark.asyncio
    async def test_negative_sentiment(self, model):
        """Test negative sentiment detection"""
        text = "Sản phẩm này thật tệ. Tôi không hài lòng chút nào."
        result = await model.predict(text)

        assert result.label == "negative"
        assert result.score > 0.3

    @pytest.mark.asyncio
    async def test_neutral_sentiment(self, model):
        """Test neutral sentiment detection"""
        text = "Sản phẩm này bình thường. Không có gì đặc biệt."
        result = await model.predict(text)

        assert result.label in ["neutral", "negative"]  # Could be either

    @pytest.mark.asyncio
    async def test_negation_handling(self, model):
        """Test negation handling"""
        positive_text = "Tôi thích sản phẩm này"
        negative_text = "Tôi không thích sản phẩm này"

        pos_result = await model.predict(positive_text)
        neg_result = await model.predict(negative_text)

        assert pos_result.label == "positive"
        assert neg_result.label == "negative"

    @pytest.mark.asyncio
    async def test_intensifier_handling(self, model):
        """Test intensifier words"""
        normal_text = "Sản phẩm tốt"
        intensified_text = "Sản phẩm rất tốt"

        normal_result = await model.predict(normal_text)
        intensified_result = await model.predict(intensified_text)

        # Intensified should have higher score
        assert intensified_result.score >= normal_result.score

    @pytest.mark.asyncio
    async def test_batch_prediction(self, model):
        """Test batch prediction"""
        texts = [
            "Tôi thích điều này",
            "Không hài lòng",
            "Bình thường thôi"
        ]

        results = await model.predict_batch(texts)

        assert len(results) == 3
        assert all(isinstance(r, SentimentPrediction) for r in results)
        assert results[0].label == "positive"
        assert results[1].label == "negative"

    @pytest.mark.asyncio
    async def test_empty_text(self, model):
        """Test empty text handling"""
        result = await model.predict("")
        assert result.label == "neutral"
        assert result.score >= 0

    @pytest.mark.asyncio
    async def test_model_info(self, model):
        """Test model information"""
        info = model.get_info()

        assert info["name"] == "rule_based_vi"
        assert info["version"] == "1.0"
        assert info["is_loaded"] is True
        assert "vi" in info["supported_languages"]
