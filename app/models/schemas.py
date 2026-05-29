"""Request and response schemas — all validated by Pydantic v2."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ── Ingest ────────────────────────────────────────────────────────────────────

class IngestTextRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, description="List of raw text strings to ingest.")
    metadatas: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional metadata dicts (one per text). Keys like 'source', 'topic', etc.",
    )

    @field_validator("texts")
    @classmethod
    def texts_not_empty(cls, v: List[str]) -> List[str]:
        for i, t in enumerate(v):
            if not t.strip():
                raise ValueError(f"texts[{i}] is empty or whitespace.")
        return v


class IngestResponse(BaseModel):
    success: bool
    raw_document_count: int
    chunk_count: int
    stored_ids: List[str]
    message: str


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="The user's question.")
    k: int = Field(default=4, ge=1, le=20, description="Number of docs to retrieve.")
    filter: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata filter, e.g. {'topic': 'database'}.",
    )
    include_sources: bool = Field(default=True, description="Include source documents in response.")


class SourceDocument(BaseModel):
    content: str
    metadata: Dict[str, Any]
    score: Optional[float] = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: Optional[List[SourceDocument]] = None
    retrieval_k: int
    model_used: str


# ── Search (vector only, no LLM) ──────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2)
    k: int = Field(default=4, ge=1, le=20)
    filter: Optional[Dict[str, Any]] = None
    with_scores: bool = Field(default=True)


class SearchResponse(BaseModel):
    query: str
    results: List[SourceDocument]


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    vector_store_doc_count: int
    collection_name: str
