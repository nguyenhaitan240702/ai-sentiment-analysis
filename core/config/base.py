"""
Base configuration settings for AI Sentiment Analysis
"""

import os
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # ===== API Configuration =====
    API_HOST: str = Field(default="0.0.0.0", description="API host")
    API_PORT: int = Field(default=8080, description="API port")
    API_WORKERS: int = Field(default=1, description="Number of API workers")
    API_RELOAD: bool = Field(default=False, description="Enable auto-reload")
    
    # ===== Database Configuration =====
    # MySQL
    MYSQL_HOST: str = Field(default="localhost", description="MySQL host")
    MYSQL_PORT: int = Field(default=3306, description="MySQL port")
    MYSQL_USER: str = Field(default="root", description="MySQL user")
    MYSQL_PASSWORD: str = Field(default="secret", description="MySQL password")
    MYSQL_DB: str = Field(default="aisent", description="MySQL database")
    
    # MongoDB
    MONGO_URI: str = Field(default="mongodb://localhost:27017/aisent", description="MongoDB URI")
    MONGO_DB: str = Field(default="aisent", description="MongoDB database")
    
    # Redis
    REDIS_URL: str = Field(default="redis://localhost:6379/0", description="Redis URL")
    REDIS_PASSWORD: Optional[str] = Field(default=None, description="Redis password")
    
    # ===== Model Configuration =====
    MODEL_BACKEND: str = Field(default="rule", description="Model backend: rule, transformer, llm")
    MODEL_NAME: str = Field(default="rule_based_vi", description="Model name")
    MODEL_CACHE_TTL: int = Field(default=600, description="Model cache TTL in seconds")
    MODEL_MAX_LENGTH: int = Field(default=512, description="Maximum text length")
    
    # Transformer specific
    TRANSFORMER_MODEL: str = Field(default="vinai/phobert-base", description="Transformer model name")
    TRANSFORMER_DEVICE: str = Field(default="cpu", description="Device: cpu, cuda, auto")
    TRANSFORMER_BATCH_SIZE: int = Field(default=8, description="Transformer batch size")
    
    # LLM specific
    LLM_API_URL: str = Field(default="http://localhost:11434/api/generate", description="LLM API URL")
    LLM_MODEL_NAME: str = Field(default="llama2-vietnamese", description="LLM model name")
    LLM_TIMEOUT: int = Field(default=30, description="LLM timeout in seconds")
    LLM_MAX_RETRIES: int = Field(default=3, description="LLM max retries")
    
    # ===== Cache Configuration =====
    CACHE_TTL: int = Field(default=600, description="Cache TTL in seconds")
    CACHE_MAX_SIZE: int = Field(default=10000, description="Cache max size")
    CACHE_ENABLED: bool = Field(default=True, description="Enable cache")
    
    # ===== Worker Configuration =====
    CELERY_BROKER_URL: str = Field(default="redis://localhost:6379/0", description="Celery broker URL")
    CELERY_RESULT_BACKEND: str = Field(default="redis://localhost:6379/1", description="Celery result backend")
    CELERY_WORKER_CONCURRENCY: int = Field(default=2, description="Celery worker concurrency")
    CELERY_TASK_TIMEOUT: int = Field(default=300, description="Celery task timeout")
    
    # ===== Batch Processing =====
    BATCH_MAX: int = Field(default=128, description="Maximum batch size")
    BATCH_TIMEOUT: int = Field(default=300, description="Batch timeout in seconds")
    ASYNC_ENABLED: bool = Field(default=True, description="Enable async processing")
    
    # ===== Logging & Monitoring =====
    LOG_LEVEL: str = Field(default="INFO", description="Log level")
    LOG_FORMAT: str = Field(default="json", description="Log format: json, text")
    LOG_FILE: Optional[str] = Field(default=None, description="Log file path")
    
    # Metrics
    METRICS_ENABLED: bool = Field(default=True, description="Enable metrics")
    METRICS_PORT: int = Field(default=9090, description="Metrics port")
    
    # ===== Security =====
    RATE_LIMIT_ENABLED: bool = Field(default=True, description="Enable rate limiting")
    RATE_LIMIT_REQUESTS: int = Field(default=100, description="Rate limit requests per window")
    RATE_LIMIT_WINDOW: int = Field(default=60, description="Rate limit window in seconds")
    
    # ===== Language Detection =====
    LANG_DETECT_ENABLED: bool = Field(default=True, description="Enable language detection")
    DEFAULT_LANG: str = Field(default="vi", description="Default language")
    SUPPORTED_LANGS: str = Field(default="vi,en", description="Supported languages (comma-separated)")
    
    # ===== Performance =====
    PRELOAD_MODELS: bool = Field(default=True, description="Preload models on startup")
    WARM_UP_REQUESTS: int = Field(default=5, description="Number of warm-up requests")
    
    # ===== Development =====
    DEBUG: bool = Field(default=False, description="Debug mode")
    TESTING: bool = Field(default=False, description="Testing mode")
    ENV: str = Field(default="development", description="Environment: development, staging, production")
    
    @property
    def mysql_url(self) -> str:
        """Get MySQL connection URL"""
        return f"mysql+pymysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DB}"
    
    @property
    def supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        return [lang.strip() for lang in self.SUPPORTED_LANGS.split(",")]
    
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.ENV == "production"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get application settings singleton"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
