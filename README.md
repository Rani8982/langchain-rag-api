# RAG API

**LangChain + HuggingFace + ChromaDB + FastAPI**

A production-ready Retrieval-Augmented Generation (RAG) system that:
- Embeds documents locally using HuggingFace sentence-transformers (free, no API cost)
- Answers questions using **Mistral-7B** via HuggingFace Inference API
- Stores and retrieves vectors with **ChromaDB** (persistent on disk)
- Exposes a clean **FastAPI** REST API with Pydantic validation
- Structured JSON logging, health checks, request tracing, Docker support


## 📁 Project Structure

```
rag_project/
├── app/
│   ├── main.py                  # FastAPI app, middleware, lifespan
│   ├── core/
│   │   ├── config.py            # Pydantic settings
│   │   ├── logger.py            # Structured logging (structlog)
│   │   ├── vector_store.py      # ChromaDB + HuggingFace embeddings
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

##  Quick Start

### 1. Get a HuggingFace API Token 

1. Go to https://huggingface.co/settings/tokens
2. Click **New token** → choose **Read** role → copy the token

### 2. Clone & configure

```bash
git clone <your-repo>
cd rag_project

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

```

---

## ⚙️ Configuration Reference

All settings live in `.env`:

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
## 🔄 How It Works — Data Flow
 
### 📥 Ingestion Pipeline (Upload a document)
 
```
User uploads PDF / DOCX / TXT
           │
           ▼
 ┌─────────────────────┐
 │  FastAPI             │  POST /api/v1/ingest/file
 │  receives file       │  or  /api/v1/ingest/texts
 └────────┬────────────┘
          │
          ▼
 ┌─────────────────────┐
 │  pypdf / python-docx │  Extracts raw text from file
 │  extracts raw text   │
 └────────┬────────────┘
          │
          ▼
 ┌─────────────────────┐
 │  LangChain           │  RecursiveCharacterTextSplitter
 │  splits into chunks  │  chunk_size=512, overlap=64
 └────────┬────────────┘
          │
          ▼
 ┌──────────────────────────┐
 │  HuggingFace (LOCAL)      │  sentence-transformers/all-MiniLM-L6-v2
 │  converts chunk → vector  │  chunk text → [0.12, 0.87, -0.34, ...]
 │  (free, runs on your CPU) │  384-dimensional embedding
 └────────┬─────────────────┘
          │
          ▼
 ┌─────────────────────┐
 │  ChromaDB            │  Stores chunk text + vector + metadata
 │  persists to disk    │  saved in ./chroma_db folder
 └─────────────────────┘
```
 
---
 
### 💬 Query Pipeline (Ask a question)
 
```
User asks: "What does the document say about X?"
           │
           ▼
 ┌─────────────────────┐
 │  FastAPI             │  POST /api/v1/query/rag
 │  receives question   │
 └────────┬────────────┘
          │
          ▼
 ┌──────────────────────────┐
 │  HuggingFace (LOCAL)      │  Converts question → vector
 │  embeds the question      │  same model as ingestion
 └────────┬─────────────────┘
          │
          ▼
 ┌─────────────────────┐
 │  ChromaDB            │  Finds top-k most similar chunks
 │  MMR similarity      │  MMR = diverse, non-redundant results
 │  search              │  returns chunk1, chunk2, chunk3, chunk4
 └────────┬────────────┘
          │
          ▼
 ┌──────────────────────────────────────────────┐
 │  LangChain builds prompt                      │
 │                                               │
 │  System: "Answer using ONLY this context:     │
 │           [chunk1] [chunk2] [chunk3] [chunk4]"│
 │  Human:  "What does the document say about X?"│
 └────────┬─────────────────────────────────────┘
          │
          ▼
 ┌──────────────────────────┐
 │  HuggingFace Inference    │  Sends prompt to Zephyr-7B / Mistral
 │  API (LLM)                │  via https://router.huggingface.co/v1
 │                           │  uses your HUGGINGFACEHUB_API_TOKEN
 └────────┬─────────────────┘
          │
          ▼
 ┌─────────────────────┐
 │  LLM returns         │  Grounded answer based only on
 │  grounded answer     │  the retrieved chunks — no hallucination
 └────────┬────────────┘
          │
          ▼
 ┌─────────────────────┐
 │  FastAPI returns     │  { "answer": "...",
 │  JSON response       │    "sources": [chunk1, chunk2...],
 │  to UI               │    "model_used": "zephyr-7b-beta" }
 └─────────────────────┘
```
 
---
 
## 🛠️ Tools Used
 
| Tool | Role | Why |
|---|---|---|
| **FastAPI** | REST API layer | Auto Swagger UI, Pydantic validation, async |
| **LangChain** | Orchestration | Connects splitter → embeddings → retriever → LLM |
| **HuggingFace** | Embeddings + LLM | Free, open-source models, no credit card needed |
| **ChromaDB** | Vector database | Local, persistent, metadata filtering, MMR search |
| **pypdf** | PDF parsing | Extracts text from PDF files page by page |
| **python-docx** | DOCX parsing | Extracts paragraphs from Word documents |
| **structlog** | Logging | Structured JSON logs with request IDs |
| **Pydantic** | Validation | Type-safe request/response models |
| **Docker** | Containerization | Reproducible deployment anywhere |
 