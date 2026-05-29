from fastapi import APIRouter, HTTPException, status

from app.core.ingestion import IngestionService
from app.core.logger import get_logger
from app.models.schemas import IngestResponse, IngestTextRequest

router = APIRouter()
logger = get_logger(__name__)


@router.post("/texts", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_texts(payload: IngestTextRequest) -> IngestResponse:
    """
    Ingest raw text strings into the vector store.

    - Splits texts into chunks automatically.
    - Embeds using HuggingFace sentence-transformers (local, no API cost).
    - Stores in ChromaDB with optional metadata.
    """
    try:
        service = IngestionService()
        result = service.ingest_texts(
            texts=payload.texts,
            metadatas=payload.metadatas,
        )
        return IngestResponse(
            success=True,
            message=f"Successfully ingested {result['raw_document_count']} document(s) into {result['chunk_count']} chunk(s).",
            **result,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.error("ingest_failed", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Ingestion failed.")
