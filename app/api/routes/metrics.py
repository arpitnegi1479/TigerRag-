from fastapi import APIRouter

from app.services.metrics_service import MetricsService

router = APIRouter(prefix="/api", tags=["metrics"])

metrics_service = MetricsService()


@router.get("/metrics")
def get_metrics() -> dict:
    return metrics_service.aggregate()
