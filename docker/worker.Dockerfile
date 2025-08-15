# Multi-stage Dockerfile for Worker Service
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set work directory
WORKDIR /app

# ===== Development Stage =====
FROM base as development

# Install development dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code
COPY . .

# Create logs directory
RUN mkdir -p /app/logs && chown -R appuser:appuser /app

USER appuser

CMD ["python", "-m", "celery", "worker", "-A", "apps.worker.tasks", "--loglevel=info", "--concurrency=2"]

# ===== Production Stage =====
FROM base as production

# Install production dependencies only
COPY requirements.txt .
RUN pip install -r requirements.txt && \
    pip cache purge

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/tmp && \
    chown -R appuser:appuser /app && \
    chmod -R 755 /app

# Remove unnecessary files
RUN rm -rf tests/ scripts/dev_* .git/ .env*

# Security: Non-root user
USER appuser

# Health check for worker
HEALTHCHECK --interval=60s --timeout=30s --start-period=60s --retries=3 \
    CMD python -c "from apps.worker.tasks import celery_app; celery_app.control.inspect().active() or exit(1)"

# Use optimized worker settings for production
CMD ["python", "-m", "celery", "worker", "-A", "apps.worker.tasks", "--loglevel=warning", "--concurrency=4", "--max-tasks-per-child=1000", "--time-limit=300"]
