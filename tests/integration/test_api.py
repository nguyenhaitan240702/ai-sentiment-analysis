"""
Integration tests for AI Sentiment Analysis API
"""

import pytest
import asyncio
from httpx import AsyncClient
from fastapi.testclient import TestClient

from apps.api.main import app

class TestSentimentAPI:
    """Integration tests for sentiment analysis API"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)

    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/healthz")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "checks" in data

    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "AI Sentiment Analysis API"
        assert data["version"] == "1.0.0"

    def test_sentiment_analysis(self, client):
        """Test sentiment analysis endpoint"""
        payload = {
            "text": "Tôi rất thích sản phẩm này!",
            "lang": "vi"
        }

        response = client.post("/v1/sentiment", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "label" in data
        assert "score" in data
        assert "model" in data
        assert "latency_ms" in data
        assert data["label"] in ["positive", "negative", "neutral"]
        assert 0 <= data["score"] <= 1

    def test_sentiment_batch_sync(self, client):
        """Test batch sentiment analysis (sync)"""
        payload = {
            "items": [
                {"id": "1", "text": "Tôi thích điều này", "lang": "vi"},
                {"id": "2", "text": "Không hài lòng", "lang": "vi"}
            ],
            "async_mode": False
        }

        response = client.post("/v1/sentiment/batch", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "completed"
        assert data["total_items"] == 2
        assert len(data["results"]) == 2
        assert data["results"][0]["id"] == "1"
        assert data["results"][1]["id"] == "2"

    def test_sentiment_batch_async(self, client):
        """Test batch sentiment analysis (async)"""
        payload = {
            "items": [
                {"id": "1", "text": "Test text", "lang": "vi"}
            ],
            "async_mode": True
        }

        response = client.post("/v1/sentiment/batch", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "processing"
        assert "job_id" in data
        assert data["total_items"] == 1

    def test_invalid_text_empty(self, client):
        """Test validation for empty text"""
        payload = {"text": "", "lang": "vi"}

        response = client.post("/v1/sentiment", json=payload)
        assert response.status_code == 422  # Validation error

    def test_invalid_text_too_long(self, client):
        """Test validation for text too long"""
        payload = {
            "text": "x" * 6000,  # Exceeds max length
            "lang": "vi"
        }

        response = client.post("/v1/sentiment", json=payload)
        assert response.status_code == 422

    def test_invalid_language(self, client):
        """Test validation for invalid language"""
        payload = {
            "text": "Test text",
            "lang": "invalid"
        }

        response = client.post("/v1/sentiment", json=payload)
        assert response.status_code == 422

    def test_batch_size_limit(self, client):
        """Test batch size validation"""
        # Create payload exceeding batch limit
        items = [{"id": str(i), "text": f"Text {i}"} for i in range(200)]
        payload = {
            "items": items,
            "async_mode": False
        }

        response = client.post("/v1/sentiment/batch", json=payload)
        assert response.status_code == 400  # Bad request

    def test_duplicate_batch_ids(self, client):
        """Test duplicate ID validation in batch"""
        payload = {
            "items": [
                {"id": "1", "text": "Text 1"},
                {"id": "1", "text": "Text 2"}  # Duplicate ID
            ],
            "async_mode": False
        }

        response = client.post("/v1/sentiment", json=payload)
        assert response.status_code == 422
