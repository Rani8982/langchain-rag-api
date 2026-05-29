from fastapi import APIRouter, Request

from app.core.config import settings
from app.models.schemas import HealthResponse

router = APIRouter()


@router.get("/", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    vs_manager = request.app.state.vs_manager
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        vector_store_doc_count=vs_manager.count(),
        collection_name=vs_manager.collection_name(),
    )
