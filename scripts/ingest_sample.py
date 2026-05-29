#!/usr/bin/env python3
"""
scripts/ingest_sample.py
------------------------
Quick CLI to ingest sample documents into the vector store.
Run AFTER the server is up, or standalone if you just want to pre-populate.

Usage:
    python scripts/ingest_sample.py                     # ingest built-in samples
    python scripts/ingest_sample.py --file myfile.txt   # ingest a text file
    python scripts/ingest_sample.py --url http://...    # ingest via API call
"""

import argparse
import json
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

SAMPLE_DOCS = [
    {
        "text": "LangChain is a framework for developing applications powered by language models. "
                "It provides tools for chaining LLM calls, managing prompts, and building agents.",
        "metadata": {"source": "langchain_docs", "topic": "framework", "lang": "en"},
    },
    {
        "text": "LangGraph is a library for building stateful, multi-actor applications with LLMs. "
                "It extends LangChain with graph-based orchestration for complex agent workflows.",
        "metadata": {"source": "langgraph_docs", "topic": "framework", "lang": "en"},
    },
    {
        "text": "Vector stores are databases optimized for storing and searching high-dimensional embeddings. "
                "They enable semantic similarity search, which is the foundation of RAG systems.",
        "metadata": {"source": "vector_guide", "topic": "database", "lang": "en"},
    },
    {
        "text": "RAG (Retrieval-Augmented Generation) combines information retrieval with text generation. "
                "It grounds LLM responses in external knowledge, reducing hallucinations.",
        "metadata": {"source": "rag_guide", "topic": "architecture", "lang": "en"},
    },
    {
        "text": "Embeddings convert text into dense numerical vectors capturing semantic meaning. "
                "Similar texts produce similar vectors, enabling semantic search without keyword matching.",
        "metadata": {"source": "embeddings_guide", "topic": "fundamentals", "lang": "en"},
    },
    {
        "text": "Chroma is an open-source embedding database designed for AI applications. "
                "It supports persistent storage, metadata filtering, and integrates natively with LangChain.",
        "metadata": {"source": "chroma_docs", "topic": "database", "lang": "en"},
    },
    {
        "text": "HuggingFace provides thousands of pre-trained models for NLP, vision, and audio. "
                "Sentence-transformers from HuggingFace are ideal for generating text embeddings locally.",
        "metadata": {"source": "huggingface_docs", "topic": "platform", "lang": "en"},
    },
    {
        "text": "Mistral-7B is a 7-billion parameter language model by Mistral AI. "
                "It achieves strong performance across benchmarks and is available via HuggingFace Inference API.",
        "metadata": {"source": "mistral_docs", "topic": "llm", "lang": "en"},
    },
    {
        "text": "Production RAG systems require error handling, structured logging, API rate limiting, "
                "monitoring, and a scalable vector database. Local Chroma is fine for prototyping; "
                "Pinecone or Weaviate are better for large-scale deployments.",
        "metadata": {"source": "production_guide", "topic": "architecture", "lang": "en"},
    },
]


def ingest_via_api(base_url: str = "http://localhost:8000") -> None:
    """POST sample docs to the running API server."""
    import urllib.request

    url = f"{base_url}/api/v1/ingest/texts"
    payload = {
        "texts": [d["text"] for d in SAMPLE_DOCS],
        "metadatas": [d["metadata"] for d in SAMPLE_DOCS],
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")

    print(f"Ingesting {len(SAMPLE_DOCS)} documents via {url} ...")
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
    print(json.dumps(result, indent=2))


def ingest_directly() -> None:
    """Ingest directly using the Python modules (no server needed)."""
    from app.core.vector_store import VectorStoreManager
    from app.core.ingestion import IngestionService

    print("Initializing vector store...")
    vs = VectorStoreManager()
    vs.initialize()

    print(f"Ingesting {len(SAMPLE_DOCS)} sample documents directly...")
    service = IngestionService()
    result = service.ingest_texts(
        texts=[d["text"] for d in SAMPLE_DOCS],
        metadatas=[d["metadata"] for d in SAMPLE_DOCS],
    )
    print(f"✓ Ingested {result['raw_document_count']} docs → {result['chunk_count']} chunks")
    print(f"  Total docs in store: {vs.count()}")


def ingest_file_directly(file_path: str) -> None:
    from app.core.vector_store import VectorStoreManager
    from app.core.ingestion import IngestionService

    vs = VectorStoreManager()
    vs.initialize()

    service = IngestionService()
    result = service.ingest_file(file_path)
    print(f"✓ File ingested: {result['chunk_count']} chunks stored.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG vector store.")
    parser.add_argument("--api", action="store_true", help="Send via HTTP API (server must be running).")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the API server.")
    parser.add_argument("--file", type=str, default=None, help="Path to a .txt file to ingest.")
    args = parser.parse_args()

    if args.file:
        ingest_file_directly(args.file)
    elif args.api:
        ingest_via_api(args.url)
    else:
        ingest_directly()
