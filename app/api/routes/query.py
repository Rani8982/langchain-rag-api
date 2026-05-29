from fastapi import APIRouter, HTTPException, status

from app.core.config import settings
from app.core.logger import get_logger
from app.core.rag_chain import build_rag_chain
from app.core.vector_store import VectorStoreManager
from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    SearchRequest,
    SearchResponse,
    SourceDocument,
)

router = APIRouter()
logger = get_logger(__name__)


@router.post("/rag", response_model=QueryResponse)
async def rag_query(payload: QueryRequest) -> QueryResponse:
    """
    Full RAG pipeline:
    1. Retrieve relevant chunks from ChromaDB.
    2. Pass them + the question to Mistral-7B via HuggingFace Inference API.
    3. Return grounded answer + source documents.
    """
    try:
        chain = build_rag_chain(filter=payload.filter)
        result = chain.invoke({"input": payload.question})

        sources = None
        if payload.include_sources and result.get("context"):
            sources = [
                SourceDocument(
                    content=doc.page_content,
                    metadata=doc.metadata,
                )
                for doc in result["context"]
            ]

        return QueryResponse(
            question=payload.question,
            answer=result["answer"],
            sources=sources,
            retrieval_k=payload.k,
            model_used=settings.LLM_MODEL_ID,
        )

    except Exception as e:
        logger.error("rag_query_failed", question=payload.question, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG query failed: {str(e)}",
        )


@router.post("/search", response_model=SearchResponse)
async def vector_search(payload: SearchRequest) -> SearchResponse:
    """
    Pure vector similarity search — no LLM, just retrieve matching chunks.
    Useful for debugging retrieval quality.
    """
    try:
        vs_manager = VectorStoreManager()
        results_with_scores = vs_manager.similarity_search_with_scores(
            query=payload.query,
            k=payload.k,
            filter=payload.filter,
        )
        return SearchResponse(
            query=payload.query,
            results=[
                SourceDocument(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    score=score if payload.with_scores else None,
                )
                for doc, score in results_with_scores
            ],
        )
    except Exception as e:
        logger.error("vector_search_failed", query=payload.query, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Vector search failed.",
        )
