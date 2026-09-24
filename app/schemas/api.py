from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class QueryMode(str, Enum):
    RAG = "rag"
    GRAPHRAG = "graphrag"
    AGENTIC = "agentic"


class EvidenceStatus(str, Enum):
    VERIFIED = "VERIFIED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
    NOT_FOUND = "NOT_FOUND"
    UNVERIFIED = "UNVERIFIED"


class Citation(BaseModel):
    document_id: str | None = None
    chunk_id: str | None = None
    page: int | None = None
    section: str | None = None
    snippet: str | None = None
    score: float | None = None


class RetrievalStats(BaseModel):
    top_k: int = 0
    nodes_searched: int = 0
    edges_searched: int = 0
    path_length: int | None = None
    latency_ms: float | None = None
    token_count: int | None = None
    cost_usd: float | None = None


class AgentStep(BaseModel):
    step: int
    action: str
    tool: str | None = None
    input_summary: str | None = None
    result_summary: str | None = None
    status: str = "completed"
    latency_ms: float | None = None


class QueryResult(BaseModel):
    query_id: str | None = None
    mode: QueryMode
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float | None = None
    evidence_status: EvidenceStatus = EvidenceStatus.UNVERIFIED
    retrieval_stats: RetrievalStats = Field(default_factory=RetrievalStats)
    graph_paths: list[dict[str, Any]] = Field(default_factory=list)
    agent_steps: list[AgentStep] = Field(default_factory=list)
    reasoning_summary: str | None = None
    latency_ms: float | None = None


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    max_depth: int = 3


class VerifyClaimRequest(BaseModel):
    claim: str
    evidence: str


class CompareQueryRequest(BaseModel):
    query: str
    top_k: int = 5
    max_depth: int = 3


class EvaluationRunResult(BaseModel):
    run_id: str
    dataset_version: str
    status: str = "pending"
    created_at: str | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    app_name: str
    environment: str
    services: dict[str, str] = Field(default_factory=dict)
