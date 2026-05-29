# 🚀 Production RAG API

**LangChain + HuggingFace + ChromaDB + FastAPI**

A production-ready Retrieval-Augmented Generation (RAG) system that:
- Embeds documents locally using HuggingFace sentence-transformers (free, no API cost)
- Answers questions using **Mistral-7B** via HuggingFace Inference API
- Stores and retrieves vectors with **ChromaDB** (persistent on disk)
- Exposes a clean **FastAPI** REST API with Pydantic validation
- Structured JSON logging, health checks, request tracing, Docker support

---

## 📁 Project Structure

```
rag_project/
├── app/
│   ├── main.py                  # FastAPI app, middleware, lifespan
│   ├── core/
│   │   ├── config.py            # Pydantic settings (all from .env)
│   │   ├── logger.py            # Structured logging (structlog)
│   │   ├── vector_store.py      # ChromaDB + HuggingFace embeddings (singleton)
│   │   ├── ingestion.py         # Text splitter + ingest service
│   │   └── rag_chain.py         # LCEL RAG chain (retriever → LLM)
│   ├── api/routes/
│   │   ├── health.py            # GET  /health/
│   │   ├── ingest.py            # POST /api/v1/ingest/texts
│   │   └── query.py             # POST /api/v1/query/rag  &  /search
│   └── models/
│       └── schemas.py           # Pydantic request/response models
├── tests/
│   └── test_api.py              # Pytest unit + integration tests
├── scripts/
│   └── ingest_sample.py         # CLI to seed sample documents
├── .env.example                 # ← copy to .env and add your token
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## ⚡ Quick Start

### 1. Get a HuggingFace API Token (free)

1. Go to https://huggingface.co/settings/tokens
2. Click **New token** → choose **Read** role → copy the token

### 2. Clone & configure

```bash
git clone <your-repo>
cd rag_project

cp .env.example .env
# Edit .env and set:  HUGGINGFACEHUB_API_TOKEN=hf_your_token_here
```

### 3. Install dependencies

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

---

## 🐳 Docker

```bash
# Build and start
docker-compose up --build

# Run in background
docker-compose up -d

# Stop
docker-compose down

# View logs
docker-compose logs -f rag-api

# Rebuild after code changes
docker-compose up --build --force-recreate
```

---

## 📥 Ingest Documents

### Option A — CLI script (no server needed)

```bash
# Ingest built-in sample docs directly into ChromaDB
python scripts/ingest_sample.py

# Ingest your own text file
python scripts/ingest_sample.py --file /path/to/your/document.txt

# Ingest via the running HTTP API
python scripts/ingest_sample.py --api --url http://localhost:8000
```

### Option B — HTTP API (curl)

```bash
curl -X POST http://localhost:8000/api/v1/ingest/texts \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "LangChain is a framework for developing applications powered by language models.",
      "ChromaDB is an open-source embedding database for AI applications."
    ],
    "metadatas": [
      {"source": "langchain_docs", "topic": "framework"},
      {"source": "chroma_docs",   "topic": "database"}
    ]
  }'
```

**Response:**
```json
{
  "success": true,
  "raw_document_count": 2,
  "chunk_count": 2,
  "stored_ids": ["a1b2c3d4", "e5f6g7h8"],
  "message": "Successfully ingested 2 document(s) into 2 chunk(s)."
}
```

---

## 🔍 Query

### Full RAG (retrieval + LLM answer)

```bash
curl -X POST http://localhost:8000/api/v1/query/rag \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is LangChain and how does it relate to RAG?",
    "k": 4,
    "include_sources": true
  }'
```

### Filter by metadata

```bash
curl -X POST http://localhost:8000/api/v1/query/rag \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What databases are available?",
    "k": 3,
    "filter": {"topic": "database"}
  }'
```

### Pure vector search (no LLM, debug retrieval)

```bash
curl -X POST http://localhost:8000/api/v1/query/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "embedding databases",
    "k": 3,
    "with_scores": true
  }'
```

---

## 🏥 Health Check

```bash
curl http://localhost:8000/health/
```

```json
{
  "status": "ok",
  "version": "1.0.0",
  "vector_store_doc_count": 9,
  "collection_name": "rag_documents"
}
```

---

## 🧪 Run Tests

```bash
# Install test deps (already in requirements.txt)
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=app --cov-report=term-missing

# Run a specific test
pytest tests/test_api.py::test_rag_query -v
```

---

## ⚙️ Configuration Reference

All settings live in `.env` (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `HUGGINGFACEHUB_API_TOKEN` | **required** | Your HF token |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `LLM_MODEL_ID` | `mistralai/Mistral-7B-Instruct-v0.3` | HF Inference API LLM |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB storage path |
| `RETRIEVAL_SEARCH_TYPE` | `mmr` | `mmr` or `similarity` |
| `RETRIEVAL_K` | `4` | Chunks returned per query |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `DEBUG` | `false` | Pretty-print logs if true |
| `LOG_LEVEL` | `INFO` | `DEBUG/INFO/WARNING/ERROR` |

---

## 🏗️ Production Checklist

- [x] Environment-based config (no hardcoded secrets)
- [x] Pydantic input validation
- [x] Structured JSON logging with request IDs
- [x] Global exception handler
- [x] Health check endpoint
- [x] Persistent vector store (survives restarts)
- [x] MMR retrieval (diverse, non-redundant results)
- [x] Metadata filtering
- [x] Deterministic chunk IDs (idempotent re-ingestion)
- [x] Dockerfile + docker-compose
- [x] Unit + integration tests
- [ ] Auth (add API key middleware for real deployments)
- [ ] Rate limiting (add `slowapi` for production)
- [ ] Swap Chroma → Pinecone/Weaviate for horizontal scale
- [ ] Add Redis caching for repeated queries

---

## 📡 API Endpoints Summary

| Method | Path | Description |
|---|---|---|
| GET | `/` | Root info |
| GET | `/health/` | Health + vector store stats |
| GET | `/docs` | Swagger UI |
| POST | `/api/v1/ingest/texts` | Ingest text documents |
| POST | `/api/v1/query/rag` | Ask a question (RAG) |
| POST | `/api/v1/query/search` | Vector similarity search |
