# AI Sentiment Analysis - Production Grade

Hệ thống phân tích cảm xúc (sentiment analysis) cho tiếng Việt với khả năng mở rộng đa ngôn ngữ. Kiến trúc microservices với FastAPI, Celery workers, Redis cache, MongoDB document store và MySQL analytics.

## 🏗️ Kiến trúc

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   FastAPI   │    │   Celery    │    │   Models    │
│     API     │───▶│   Worker    │───▶│ Rule/Trans/ │
│             │    │             │    │     LLM     │
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    Redis    │    │  MongoDB    │    │   MySQL     │
│   (Cache)   │    │ (Documents) │    │ (Analytics) │
└─────────────┘    └─────────────┘    └─────────────┘
```

## 🚀 Khởi chạy nhanh

### Development (Local)
```bash
# Clone và setup
git clone <repo-url>
cd ai-sentiment

# Tạo file môi trường
cp .env.example .env

# Khởi chạy tất cả services
docker compose up -d --build

# Kiểm tra health
curl http://localhost:8080/healthz

# Test sentiment analysis
curl -X POST http://localhost:8080/v1/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "Tôi rất thích sản phẩm này!"}'
```

### Production
```bash
# Chạy production stack
docker compose -f docker-compose.prod.yml up -d --build

# Scale workers
docker compose up -d --scale worker=3
```

## 📊 API Endpoints

### Health Check
```bash
GET /healthz
Response: {"status": "healthy", "version": "1.0.0"}
```

### Sentiment Analysis
```bash
POST /v1/sentiment
{
  "text": "Tôi rất hài lòng với dịch vụ này!",
  "lang": "vi"  // optional, auto-detect if not provided
}

Response:
{
  "label": "positive",
  "score": 0.85,
  "model": "rule_based_vi",
  "latency_ms": 15,
  "cached": false,
  "language_detected": "vi"
}
```

### Batch Processing
```bash
POST /v1/sentiment/batch
{
  "items": [
    {"id": "1", "text": "Tuyệt vời!", "lang": "vi"},
    {"id": "2", "text": "Không hài lòng", "lang": "vi"}
  ],
  "async": false
}

# Async batch
POST /v1/sentiment/batch
{
  "items": [...],
  "async": true
}
Response: {"job_id": "job_123", "status": "processing"}

# Check job status
GET /v1/jobs/job_123
```

## 🔧 Cấu hình Models

### Rule-based (Default - Nhanh, nhẹ)
```bash
MODEL_BACKEND=rule
```
- Sử dụng lexicon tiếng Việt + underthesea
- Latency: ~10-50ms
- RAM: ~100MB

### Transformer (Chính xác cao)
```bash
MODEL_BACKEND=transformer
MODEL_NAME=vinai/phobert-base
```
- Sử dụng PhoBERT hoặc mBERT
- Latency: ~100-500ms
- RAM: ~1-2GB
- Cần GPU cho tốc độ tối ưu

### LLM (Linh hoạt nhất)
```bash
MODEL_BACKEND=llm
LLM_API_URL=http://localhost:11434/api/generate
LLM_MODEL_NAME=llama2-vietnamese
```
- Kết nối LLM API (Ollama, OpenAI-compatible)
- Latency: ~1-5s
- Có fallback về rule-based khi lỗi

## 📈 Scaling & Production

### Horizontal Scaling
```bash
# Scale API instances
docker compose up -d --scale api=3

# Scale workers
docker compose up -d --scale worker=5

# Docker Swarm mode
docker stack deploy -c swarm-stack.yml ai-sentiment
```

### Monitoring
- Prometheus metrics: `http://localhost:9090`
- Grafana dashboard: `http://localhost:3000`
- Logs: JSON structured, centralized

### Performance Tuning
```bash
# Tăng worker concurrency
CELERY_WORKER_CONCURRENCY=4

# Tăng cache TTL
CACHE_TTL=3600

# Batch size optimization
BATCH_MAX=256
```

## 🗄️ Database Schema

### MongoDB Collections
```javascript
// texts
{
  "_id": ObjectId,
  "source": "api",
  "lang": "vi",
  "content": "Text content",
  "content_hash": "sha256_hash",
  "created_at": ISODate
}

// inferences
{
  "_id": ObjectId,
  "text_id": ObjectId,
  "model_name": "rule_based_vi",
  "version": "1.0",
  "sentiment": {"label": "positive", "score": 0.85},
  "tokens_metadata": {...},
  "latency_ms": 15,
  "created_at": ISODate
}
```

### MySQL Tables
```sql
-- Báo cáo hàng ngày
CREATE TABLE sentiments_daily (
    date DATE,
    lang VARCHAR(5),
    source VARCHAR(50),
    pos_cnt INT DEFAULT 0,
    neu_cnt INT DEFAULT 0,
    neg_cnt INT DEFAULT 0,
    avg_score DECIMAL(3,2),
    last_updated TIMESTAMP,
    PRIMARY KEY (date, lang, source)
);
```

## 🧪 Testing

```bash
# Chạy tất cả tests
make test

# Test coverage
make test-coverage

# Integration tests
make test-integration

# Load testing
make bench
```

## 🔒 Bảo mật

- Environment variables cho secrets
- Rate limiting (100 req/min mặc định)
- Input validation & sanitization
- Non-root containers
- Read-only filesystems

## 📝 Development

### Setup môi trường dev
```bash
# Python virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Pre-commit hooks
pre-commit install

# Run locally
make dev
```

### Code Quality
```bash
# Linting
make lint

# Format code
make format

# Type checking
make typecheck
```

## 🌐 Multi-language Support

Hệ thống được thiết kế để mở rộng:
- Vietnamese (primary) - underthesea, pyvi
- English - VADER, TextBlob
- Auto language detection - fasttext

Thêm ngôn ngữ mới:
1. Tạo model adapter trong `core/models/`
2. Thêm lexicon trong `data/lexicons/`
3. Cập nhật language detection

## ⚡ Performance Benchmarks

| Model Type | Latency (p95) | Throughput | Memory |
|------------|---------------|------------|---------|
| Rule-based | 50ms | 2000 req/s | 100MB |
| Transformer | 200ms | 500 req/s | 1.5GB |
| LLM | 2s | 50 req/s | 8GB |

## 🛟 Troubleshooting

### Common Issues

1. **Model loading failed**
   ```bash
   # Check model files
   docker exec -it ai-sentiment-api ls -la /app/models/
   
   # Verify model backend
   echo $MODEL_BACKEND
   ```

2. **Database connection errors**
   ```bash
   # Check service health
   docker compose ps
   
   # Check logs
   docker compose logs mongo
   docker compose logs mysql
   ```

3. **High memory usage**
   - Switch to rule-based model
   - Reduce batch size
   - Enable model quantization

### Logs & Debugging
```bash
# API logs
docker compose logs -f api

# Worker logs
docker compose logs -f worker

# All services
docker compose logs -f
```

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit pull request

---

**Built with ❤️ for Vietnamese NLP community**
