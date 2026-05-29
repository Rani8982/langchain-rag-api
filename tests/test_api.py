"""
Tests for the RAG API.

Run with:  pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """
    Create a test client with the vector store mocked out
    so tests don't need a real HuggingFace token or ChromaDB on disk.
    """
    mock_vs = MagicMock()
    mock_vs.count.return_value = 42
    mock_vs.collection_name.return_value = "rag_documents"
    mock_vs.add_documents.return_value = ["id1", "id2"]
    mock_vs.similarity_search_with_scores.return_value = []

    with patch("app.core.vector_store.VectorStoreManager.initialize"):
        with patch("app.core.vector_store.VectorStoreManager._vectorstore", mock_vs):
            from app.main import app
            app.state.vs_manager = mock_vs
            yield TestClient(app)


# ── Health ────────────────────────────────────────────────────────────────────

def test_health_check(client):
    resp = client.get("/health/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "vector_store_doc_count" in data


# ── Root ──────────────────────────────────────────────────────────────────────

def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "version" in resp.json()


# ── Ingest ────────────────────────────────────────────────────────────────────

def test_ingest_texts_success(client):
    with patch("app.api.routes.ingest.IngestionService") as MockService:
        mock_svc = MockService.return_value
        mock_svc.ingest_texts.return_value = {
            "raw_document_count": 2,
            "chunk_count": 4,
            "stored_ids": ["a1", "a2", "a3", "a4"],
        }
        resp = client.post(
            "/api/v1/ingest/texts",
            json={
                "texts": ["LangChain is a framework.", "ChromaDB stores embeddings."],
                "metadatas": [{"source": "doc1"}, {"source": "doc2"}],
            },
        )
    assert resp.status_code == 201
    data = resp.json()
    assert data["success"] is True
    assert data["chunk_count"] == 4


def test_ingest_empty_text_rejected(client):
    resp = client.post(
        "/api/v1/ingest/texts",
        json={"texts": ["  "]},
    )
    assert resp.status_code == 422


def test_ingest_empty_list_rejected(client):
    resp = client.post(
        "/api/v1/ingest/texts",
        json={"texts": []},
    )
    assert resp.status_code == 422


# ── Query / Search ────────────────────────────────────────────────────────────

def test_vector_search(client):
    with patch("app.api.routes.query.VectorStoreManager") as MockVS:
        mock_vs = MockVS.return_value
        from langchain_core.documents import Document
        mock_vs.similarity_search_with_scores.return_value = [
            (Document(page_content="LangChain is a framework.", metadata={"source": "doc1"}), 0.92),
        ]
        resp = client.post(
            "/api/v1/query/search",
            json={"query": "What is LangChain?", "k": 2},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["query"] == "What is LangChain?"
    assert len(data["results"]) == 1
    assert data["results"][0]["score"] == 0.92


def test_rag_query(client):
    with patch("app.api.routes.query.build_rag_chain") as mock_chain_fn:
        mock_chain = MagicMock()
        from langchain_core.documents import Document
        mock_chain.invoke.return_value = {
            "answer": "LangChain is a framework for LLM applications.",
            "context": [Document(page_content="LangChain is a framework.", metadata={"source": "docs"})],
        }
        mock_chain_fn.return_value = mock_chain

        resp = client.post(
            "/api/v1/query/rag",
            json={"question": "What is LangChain?", "k": 3},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "LangChain" in data["answer"]
    assert data["sources"] is not None


# ── Ingestion Service Unit Tests ──────────────────────────────────────────────

def test_ingestion_service_splits_text():
    with patch("app.core.ingestion.VectorStoreManager") as MockVS:
        mock_vs = MockVS.return_value
        mock_vs.add_documents.return_value = ["id1"]

        from app.core.ingestion import IngestionService
        service = IngestionService()
        result = service.ingest_texts(
            texts=["This is a test document. " * 50],
            metadatas=[{"source": "test"}],
        )
        assert result["raw_document_count"] == 1
        assert result["chunk_count"] >= 1


def test_ingestion_metadata_length_mismatch():
    with patch("app.core.ingestion.VectorStoreManager"):
        from app.core.ingestion import IngestionService
        service = IngestionService()
        with pytest.raises(ValueError, match="same length"):
            service.ingest_texts(
                texts=["text one", "text two"],
                metadatas=[{"source": "only_one"}],
            )
