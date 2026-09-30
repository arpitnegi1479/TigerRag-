from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EvaluationRunRequest(BaseModel):
    question_ids: list[str] | None = Field(default=None, min_length=1)
    category: str | None = Field(default=None, min_length=1)
    limit: int | None = Field(default=None, ge=1, le=50)
    top_k: int = Field(default=5, ge=1, le=100)
    max_graph_depth: int = Field(default=3, ge=1, le=10)
    pacing_seconds: float = Field(default=0.5, ge=0, le=60)
    resume_run_id: str | None = None


class EvaluationResult(BaseModel):
    result_id: str
    run_id: str
    dataset_version: str
    question_id: str
    category: str
    mode: str
    query: str
    answer: str
    evidence_status: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    raw_result: dict[str, Any] = Field(default_factory=dict)
    scores: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    llm_telemetry: dict[str, Any] = Field(default_factory=dict)
    configuration: dict[str, Any] = Field(default_factory=dict)
    execution_quality: str = "clean"
    cache_status: str = "fresh"