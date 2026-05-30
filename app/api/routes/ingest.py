import os
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.ingestion import IngestionService
from app.core.logger import get_logger
from app.models.schemas import IngestResponse, IngestTextRequest

router = APIRouter()
logger = get_logger(__name__)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".md"}


@router.post("/texts", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_texts(payload: IngestTextRequest) -> IngestResponse:
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


@router.post("/file", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_file(file: UploadFile = File(...)) -> IngestResponse:
    filename = file.filename or "upload"
    ext = os.path.splitext(filename)[1].lower()
    tmp_path = None

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )
    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        service = IngestionService()
        result = service.ingest_file(
            file_path=tmp_path,
            original_filename=filename,
            ext=ext,
        )
        return IngestResponse(
            success=True,
            message=f"File '{filename}' ingested into {result['chunk_count']} chunk(s).",
            **result,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("file_ingest_failed", filename=filename, error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to ingest file: {str(e)}")
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass