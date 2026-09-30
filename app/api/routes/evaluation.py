from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from app.repositories.postgres_repository import PostgresDocumentRepository
from app.schemas.evaluation import EvaluationRunRequest
from app.services.evaluation_runner import EvaluationRunner, load_questions
from app.services.evaluation_service import build_failure_analysis
from app.services.verification_service import DEFAULT_FALLBACK_VERSION

router = APIRouter(tags=["evaluation"])
RUN_DIRECTORY = Path(__file__).resolve().parents[3] / "data" / "evaluation_runs"


@router.post("/api/evaluate")
def evaluate(payload: EvaluationRunRequest) -> dict[str, Any]:
    repository = PostgresDocumentRepository.get_default()
    dataset_version, questions = load_questions()

    if payload.resume_run_id:
        stored_run = repository.get_evaluation_run(payload.resume_run_id)
        if stored_run is None:
            raise HTTPException(status_code=404, detail="Evaluation run not found")
        checkpoint = Path(str(stored_run.get("checkpoint_path", ""))).resolve()
        try:
            checkpoint.relative_to(RUN_DIRECTORY.resolve())
        except ValueError as exc:
            raise HTTPException(status_code=409, detail="Evaluation checkpoint is unavailable") from exc
        if not checkpoint.is_file():
            raise HTTPException(status_code=409, detail="Evaluation checkpoint is unavailable")
        configuration = stored_run["configuration"]
        selection = stored_run["selection"]
        runner = EvaluationRunner(
            top_k=configuration["top_k"],
            max_depth=configuration["max_graph_depth"],
            pacing_seconds=configuration["minimum_gemini_request_interval_seconds"],
            fallback_version=configuration.get("faithfulness_fallback_version", DEFAULT_FALLBACK_VERSION),
        )
        question_ids = set(selection["question_ids"]) if selection.get("question_ids") else None
        category = selection.get("category")
        limit = selection.get("limit")
        resume = True
    else:
        runner = EvaluationRunner(
            top_k=payload.top_k,
            max_depth=payload.max_graph_depth,
            pacing_seconds=payload.pacing_seconds,
        )
        question_ids = set(payload.question_ids) if payload.question_ids else None
        category = payload.category
        limit = payload.limit
        resume = False
        RUN_DIRECTORY.mkdir(parents=True, exist_ok=True)
        checkpoint = RUN_DIRECTORY / f"api-{uuid4()}.json"

    try:
        state = runner.run(
            questions=questions,
            dataset_version=dataset_version,
            output_path=checkpoint,
            question_ids=question_ids,
            category=category,
            limit=limit,
            resume=resume,
        )
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "run_id": state["run_id"],
        "dataset_version": state["dataset_version"],
        "status": state["status"],
        "summary": state.get("summary", {}),
    }


@router.get("/api/evaluation/runs")
def list_runs(limit: int = 50) -> list[dict[str, Any]]:
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 200")
    return PostgresDocumentRepository.get_default().list_evaluation_runs(limit)


@router.get("/api/evaluation/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    repository = PostgresDocumentRepository.get_default()
    run = repository.get_evaluation_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    public_run = {key: value for key, value in run.items() if key != "checkpoint_path"}
    return {**public_run, "results": repository.list_evaluation_results(run_id)}


@router.get("/api/evaluation/runs/{run_id}/failures")
def get_run_failures(run_id: str) -> dict[str, Any]:
    repository = PostgresDocumentRepository.get_default()
    run = repository.get_evaluation_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    return {
        "run_id": run_id,
        "failures": build_failure_analysis(repository.list_evaluation_results(run_id)),
    }
