"""
Base model interface for AI Sentiment Analysis
Provides abstraction layer for different model backends
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import logging
import asyncio

from core.config.base import get_settings

logger = logging.getLogger(__name__)

@dataclass
class SentimentPrediction:
    """Sentiment prediction result"""
    label: str  # positive, negative, neutral
    score: float  # confidence score 0.0-1.0
    scores: Optional[Dict[str, float]] = None  # detailed class scores
    metadata: Optional[Dict[str, Any]] = None

class SentimentModel(ABC):
    """Abstract base class for sentiment models"""

    def __init__(self, model_name: str, version: str = "1.0"):
        self.model_name = model_name
        self.version = version
        self.is_loaded = False

    @abstractmethod
    async def load(self) -> None:
        """Load the model"""
        pass

    @abstractmethod
    async def predict(self, text: str, language: str = "vi") -> SentimentPrediction:
        """Predict sentiment for single text"""
        pass

    @abstractmethod
    async def predict_batch(self, texts: List[str], language: str = "vi") -> List[SentimentPrediction]:
        """Predict sentiment for batch of texts"""
        pass

    @abstractmethod
    async def unload(self) -> None:
        """Unload the model"""
        pass

    @property
    @abstractmethod
    def supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        pass

    def get_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": self.model_name,
            "version": self.version,
            "is_loaded": self.is_loaded,
            "supported_languages": self.supported_languages
        }

class ModelManager:
    """Manages multiple sentiment models"""

    def __init__(self):
        self.models: Dict[str, SentimentModel] = {}
        self.default_model: Optional[str] = None
        self.settings = get_settings()

    async def initialize(self) -> None:
        """Initialize model manager"""
        logger.info("Initializing model manager...")

        # Load models based on configuration
        if self.settings.MODEL_BACKEND == "rule":
            from core.models.enhanced_rule_based import EnhancedRuleBasedModel
            model = EnhancedRuleBasedModel()
            await self.register_model("enhanced_rule_based_vi", model)
            self.default_model = "enhanced_rule_based_vi"

        elif self.settings.MODEL_BACKEND == "transformer":
            from core.models.transformer import TransformerModel
            model = TransformerModel(
                model_name=self.settings.TRANSFORMER_MODEL,
                device=self.settings.TRANSFORMER_DEVICE
            )
            await self.register_model("transformer_vi", model)
            self.default_model = "transformer_vi"

        elif self.settings.MODEL_BACKEND == "llm":
            from core.models.llm import LLMModel
            model = LLMModel(
                api_url=self.settings.LLM_API_URL,
                model_name=self.settings.LLM_MODEL_NAME
            )
            await self.register_model("llm_vi", model)
            self.default_model = "llm_vi"

        else:
            raise ValueError(f"Unsupported model backend: {self.settings.MODEL_BACKEND}")

    async def register_model(self, name: str, model: SentimentModel) -> None:
        """Register and load a model"""
        logger.info(f"Registering model: {name}")
        await model.load()
        self.models[name] = model
        logger.info(f"Model {name} loaded successfully")

    async def get_model(self, name: Optional[str] = None) -> SentimentModel:
        """Get model by name or default"""
        model_name = name or self.default_model

        if not model_name or model_name not in self.models:
            raise ValueError(f"Model not found: {model_name}")

        return self.models[model_name]

    async def predict(self,
                     text: str,
                     language: str = "vi",
                     model_name: Optional[str] = None) -> SentimentPrediction:
        """Predict using specified or default model"""
        model = await self.get_model(model_name)
        return await model.predict(text, language)

    async def predict_batch(self,
                           texts: List[str],
                           language: str = "vi",
                           model_name: Optional[str] = None) -> List[SentimentPrediction]:
        """Batch predict using specified or default model"""
        model = await self.get_model(model_name)
        return await model.predict_batch(texts, language)

    async def warmup(self) -> None:
        """Warm up all models"""
        logger.info("Warming up models...")

        warmup_texts = [
            "Tôi rất hài lòng với sản phẩm này",
            "Dịch vụ khách hàng tuyệt vời",
            "Sản phẩm không đạt yêu cầu",
            "Bình thường, không có gì đặc biệt",
            "Tôi thích cái này"
        ]

        for model_name, model in self.models.items():
            try:
                logger.info(f"Warming up model: {model_name}")
                await model.predict_batch(warmup_texts[:self.settings.WARM_UP_REQUESTS])
                logger.info(f"Model {model_name} warmed up successfully")
            except Exception as e:
                logger.warning(f"Failed to warm up model {model_name}: {e}")

    async def cleanup(self) -> None:
        """Cleanup all models"""
        logger.info("Cleaning up models...")

        for model_name, model in self.models.items():
            try:
                await model.unload()
                logger.info(f"Model {model_name} unloaded")
            except Exception as e:
                logger.warning(f"Failed to unload model {model_name}: {e}")

        self.models.clear()
        self.default_model = None

    def get_models_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all models"""
        return {name: model.get_info() for name, model in self.models.items()}
