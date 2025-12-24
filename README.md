
# Vietnamese Legal AI Platform

Hệ thống AI hỗ trợ pháp luật Việt Nam với 3 chức năng chính:
- **Hỏi đáp pháp luật** (RAG-based Q&A)
- **Soạn thảo hợp đồng** (Contract Drafting)
- **Phân tích hợp đồng** (Contract Analysis)

## Tổng Quan Hệ Thống

### Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend API** | FastAPI + Celery |
| **Vector Search** | Qdrant (paraphrase-vietnamese-law) |
| **Lexical Search** | Elasticsearch BM25 |
| **LLM** | OpenAI GPT-4o-mini |
| **Database** | MongoDB |
| **Cache** | Redis |
| **Web Search** | Tavily API |

### Kiến Trúc

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (port 8002)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  Legal Q&A   │  │   Contract   │  │  Contract Analysis   │  │
│  │   /chat/*    │  │   Drafting   │  │    /contract/*       │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
         │                    │                     │
         ▼                    ▼                     ▼
┌─────────────┐      ┌─────────────┐       ┌─────────────┐
│   Qdrant    │      │  MongoDB    │       │   OpenAI    │
│   (6333)    │      │  (27017)    │       │   GPT-4o    │
└─────────────┘      └─────────────┘       └─────────────┘
         │
         ▼
┌─────────────┐      ┌─────────────┐       ┌─────────────┐
│Elasticsearch│      │    Redis    │       │   Celery    │
│   (9200)    │      │   (6379)    │       │   Worker    │
└─────────────┘      └─────────────┘       └─────────────┘
```

## Bắt Đầu

### Yêu Cầu

- Docker & Docker Compose
- NVIDIA GPU (khuyến nghị RTX 3060 12GB trở lên)
- OpenAI API Key
- Tavily API Key (cho web search fallback)

### Cài Đặt

```bash
# 1. Clone repository
git clone https://github.com/your-repo/ai-law.git
cd ai-law

# 2. Cấu hình environment
cp .env.example .env
# Chỉnh sửa .env với API keys của bạn

# 3. Khởi động services
docker-compose up -d

# 4. Kiểm tra logs
docker-compose logs -f backend-api
```

### Import Data (Nếu cần)

```bash
# Import 10% data (mặc định, ~45 phút)
python scripts/import_data.py --corpus data/corpus.csv

# Import 100% data (~7-8 giờ)
python scripts/import_data.py --corpus data/corpus.csv --percent 100

# Resume nếu bị gián đoạn
python scripts/import_data.py --corpus data/corpus.csv --resume
```

## API Endpoints

### Health Check
```
GET /
```

### Legal Q&A

| Endpoint | Method | Mô tả |
|----------|--------|-------|
| `/retrieval` | POST | Tìm kiếm văn bản pháp luật |
| `/chat/complete` | POST | Gửi câu hỏi (async/sync) |
| `/chat/complete_v2/{task_id}` | GET | Lấy kết quả câu trả lời |

### Contract Drafting

| Endpoint | Method | Mô tả |
|----------|--------|-------|
| `/contract/templates` | GET | Danh sách mẫu hợp đồng |
| `/contract/templates/{id}` | GET | Chi tiết mẫu hợp đồng |
| `/contract/draft` | POST | Tạo hợp đồng từ mẫu |
| `/contract/drafts/{user_id}` | GET | Lịch sử hợp đồng đã soạn |

### Contract Analysis

| Endpoint | Method | Mô tả |
|----------|--------|-------|
| `/contract/analyze` | POST | Upload và phân tích hợp đồng |
| `/contract/analysis/{id}` | GET | Lấy kết quả phân tích |
| `/contract/analyses/{user_id}` | GET | Lịch sử phân tích |

## Data & Collections

### Qdrant Collections

| Collection | Documents | Model | Mô tả |
|------------|-----------|-------|-------|
| `law_with_legal_emb` | 26,159 | paraphrase-vietnamese-law (768 dim) | **Đang sử dụng** |
| `law_with_bge_round1` | 26,159 | BGE-M3 (1024 dim) | Backup |
| `law_with_e5_emb_not_finetune` | 26,159 | E5-large (1024 dim) | Backup |

### Elasticsearch Index

| Index | Documents | Mô tả |
|-------|-----------|-------|
| `legal_data_part2` | 26,159 | BM25 lexical search |

### Phân Bố Chủ Đề Luật

```
Xây dựng          48,432 (18.5%) ██████████████████
Doanh nghiệp      46,620 (17.8%) █████████████████
Thuế              29,768 (11.4%) ███████████
Giao thông        27,129 (10.4%) ██████████
Lao động          24,406 ( 9.3%) █████████
Hành chính        21,697 ( 8.3%) ████████
Môi trường        20,796 ( 7.9%) ███████
Hình sự           20,697 ( 7.9%) ███████
Y tế              19,273 ( 7.4%) ███████
Giáo dục          17,604 ( 6.7%) ██████
Dân sự            10,124 ( 3.9%) ███
Đất đai            7,404 ( 2.8%) ██
```

## RAG Pipeline

### Luồng Xử Lý

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│  1. Intent Routing (GPT-4o)    │
│     - Legal question?          │
│     - Chitchat?                │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  2. Query Reflection           │
│     - Rewrite với chat history │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  3. Hybrid Search              │
│     - Qdrant (semantic)        │
│     - Elasticsearch (lexical)  │
│     - RRF Fusion               │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  4. LLM Generation             │
│     - GPT-4o-mini              │
│     - Fallback: Tavily Search  │
└─────────────────────────────────┘
    │
    ▼
Response
```

### Hybrid Search với RRF

Hệ thống kết hợp 2 phương pháp search:

| Method | Engine | Điểm mạnh |
|--------|--------|-----------|
| **Semantic Search** | Qdrant + paraphrase-vietnamese-law | Hiểu ngữ nghĩa, synonyms |
| **Lexical Search** | Elasticsearch BM25 | Chính xác với thuật ngữ pháp lý |

**Reciprocal Rank Fusion (RRF)**: Documents xuất hiện ở cả 2 nguồn sẽ được ưu tiên cao hơn.

## Service Ports

| Service | Port | Dashboard |
|---------|------|-----------|
| Backend API | 8002 | http://localhost:8002/docs |
| Qdrant | 6333 | http://localhost:6333/dashboard |
| Elasticsearch | 9200 | - |
| Redis | 6379 | - |
| MongoDB | 27017 | - |

## Environment Variables

```env
# Required
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...

# Optional (có defaults)
QDRANT_HOST=http://localhost:6333
ELASTICSEARCH_URL=http://localhost:9200
REDIS_HOST=localhost
MONGODB_URL=mongodb://localhost:27017/

# Collections
LEGAL_EMB_COLLECTION=law_with_legal_emb
ELASTIC_INDEX=legal_data_part2

# Timeouts
OPENAI_TIMEOUT=60.0
OPENAI_MAX_RETRIES=2
SEARCH_CACHE_TTL=3600
```

## Cấu Trúc Dự Án

```
ai-law/
├── backend/
│   ├── src/
│   │   ├── app.py                    # FastAPI entry point
│   │   ├── api_docs.py               # API documentation strings
│   │   ├── brain.py                  # OpenAI client & prompts
│   │   ├── tasks.py                  # Celery async tasks
│   │   ├── search_document/
│   │   │   ├── combine_search.py     # Hybrid search + RRF + caching
│   │   │   ├── search_elastic.py     # Elasticsearch BM25
│   │   │   └── search_with_legal_emb.py  # Qdrant semantic search
│   │   ├── contract_drafting/
│   │   │   ├── generator.py          # Contract generation
│   │   │   ├── models.py             # Template models
│   │   │   └── schemas.py            # Pydantic schemas
│   │   └── contract_analysis/
│   │       ├── comparator.py         # Contract analysis engine
│   │       ├── models.py             # MongoDB operations
│   │       ├── prompts.py            # LLM prompts
│   │       └── schemas.py            # Analysis schemas
│   ├── requirements.txt
│   └── entrypoint.sh
├── scripts/
│   └── import_data.py                # Data import script
├── data/
│   ├── corpus.csv                    # Legal documents corpus
│   ├── train.csv                     # Training data
│   └── public_test.csv               # Test data
├── retrieval/
│   ├── create_data_rerank.py         # Reranker training data
│   └── finetune.sh                   # Reranker finetuning
├── finetune_llm/
│   ├── finetune.py                   # LLM finetuning
│   └── merge_with_base.py            # Merge LoRA weights
├── docker-compose.yml
├── .env.example
└── README.md
```

## Đánh Giá Hiệu Năng

### Recall@k (Retrieval)

| Model | K=3 | K=5 | K=10 |
|-------|-----|-----|------|
| paraphrase-vietnamese-law | 58.25% | 66.12% | 74.83% |
| Elasticsearch BM25 | 42.54% | 49.61% | 56.85% |
| **Hybrid (RRF Fusion)** | **71.42%** | **77.38%** | **83.21%** |

> **Lưu ý:** Hệ thống hiện tại sử dụng Hybrid Search (semantic + lexical) với RRF Fusion, không sử dụng reranker để tiết kiệm VRAM.

### Correctness (Generation)

Điểm đánh giá trên thang 5: **4.27/5**

## Roadmap

- [ ] Multi-tenant support (data isolation per organization)
- [ ] Enhanced prompts với legal context từ RAG
- [ ] Query expansion với LLM
- [ ] PDF processing pipeline cho custom documents

## License

MIT License

## Contact

- Issues: https://github.com/your-repo/ai-law/issues
- Email: support@example.com
