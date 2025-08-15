"""
Development and production commands for AI Sentiment Analysis
Usage: make <command>
"""

.PHONY: help dev test lint format typecheck build clean run-prod

# Default target
help:
	@echo "AI Sentiment Analysis - Available Commands:"
	@echo ""
	@echo "Development:"
	@echo "  dev          - Start development environment"
	@echo "  test         - Run all tests"
	@echo "  test-unit    - Run unit tests"
	@echo "  test-int     - Run integration tests"
	@echo "  lint         - Run linting"
	@echo "  format       - Format code"
	@echo "  typecheck    - Run type checking"
	@echo ""
	@echo "Production:"
	@echo "  build        - Build Docker images"
	@echo "  run-prod     - Start production environment"
	@echo "  scale        - Scale workers (make scale WORKERS=3)"
	@echo ""
	@echo "Utilities:"
	@echo "  clean        - Clean up containers and volumes"
	@echo "  logs         - Show logs"
	@echo "  shell        - Open shell in API container"

# Development
dev:
	docker compose up -d --build
	@echo "Development environment started!"
	@echo "API: http://localhost:8080"
	@echo "Docs: http://localhost:8080/docs"
	@echo "Health: http://localhost:8080/healthz"

# Testing
test:
	pytest tests/ -v --cov=apps --cov=core --cov-report=html

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-e2e:
	pytest tests/e2e/ -v

# Code quality
lint:
	flake8 apps/ core/ tests/
	isort --check-only apps/ core/ tests/

format:
	black apps/ core/ tests/
	isort apps/ core/ tests/

typecheck:
	mypy apps/ core/

# Production
build:
	docker compose -f docker-compose.prod.yml build

run-prod:
	docker compose -f docker-compose.prod.yml up -d
	@echo "Production environment started!"

scale:
	docker compose up -d --scale worker=$(WORKERS)
	@echo "Scaled workers to $(WORKERS)"

# Utilities
clean:
	docker compose down -v
	docker system prune -f

logs:
	docker compose logs -f

logs-api:
	docker compose logs -f api

logs-worker:
	docker compose logs -f worker

shell:
	docker compose exec api bash

shell-worker:
	docker compose exec worker bash

# Database
db-migrate:
	@echo "Running database migrations..."
	docker compose exec mysql mysql -u root -psecret123 aisent < migrations/mysql/001_initial_schema.sql

db-shell:
	docker compose exec mysql mysql -u root -psecret123 aisent

mongo-shell:
	docker compose exec mongo mongosh aisent

# Monitoring
metrics:
	@echo "Opening Prometheus metrics..."
	@echo "http://localhost:9090"

dashboard:
	@echo "Opening Grafana dashboard..."
	@echo "http://localhost:3000 (admin/admin123)"

# Quick test
smoke-test:
	@echo "Running smoke test..."
	curl -f http://localhost:8080/healthz
	curl -X POST http://localhost:8080/v1/sentiment \
		-H "Content-Type: application/json" \
		-d '{"text": "Tôi rất thích sản phẩm này!"}'

# Install dependencies
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pre-commit install
