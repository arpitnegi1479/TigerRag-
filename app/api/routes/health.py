from fastapi import APIRouter

from app.schemas.api import HealthResponse
from app.core.config import settings
from app.repositories.graph_repository import GraphRepository
from app.repositories.postgres_repository import PostgresDocumentRepository
from app.repositories.vector_repository import VectorRepository

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    services = {
        "postgres": "live" if PostgresDocumentRepository.get_default()._conn is not None else "fallback",
        "qdrant": "live" if VectorRepository.get_default().is_live else "fallback",
        "neo4j": "live" if GraphRepository.get_default().is_live else "fallback",
        "gemini": "configured" if settings.gemini_api_key else "not_configured",
    }
    status = "ok" if all(value in {"live", "configured"} for value in services.values()) else "degraded"
    return HealthResponse(status=status, app_name=settings.app_name, environment=settings.environment, services=services)
