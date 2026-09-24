from fastapi import APIRouter

from app.services.evaluation_service import EvaluationService

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])

evaluation_service = EvaluationService()


@router.get("/benchmark")
def benchmark() -> dict:
    return evaluation_service.run_benchmark()


@router.get("/runs")
def list_runs() -> list[dict]:
    return []
