from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AgentAction(str, Enum):
    SEARCH_DOCUMENTS = "SEARCH_DOCUMENTS"
    SEARCH_ENTITY = "SEARCH_ENTITY"
    TRAVERSE_GRAPH = "TRAVERSE_GRAPH"
    FIND_RELATIONSHIP = "FIND_RELATIONSHIP"
    GET_SOURCE = "GET_SOURCE"
    VERIFY_CLAIM = "VERIFY_CLAIM"
    REWRITE_QUERY = "REWRITE_QUERY"
    FINALIZE = "FINALIZE"


class AgentDecision(BaseModel):
    action: AgentAction
    reason: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class AgentToolCall(BaseModel):
    action: AgentAction
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result_count: int = 0
    result_summary: str = ""
    latency_ms: float = 0.0
    status: str = "completed"


class AgentState(BaseModel):
    query: str
    query_type: str = "UNKNOWN"
    entities: list[str] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list)
    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list)
    graph_paths: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: list[AgentToolCall] = Field(default_factory=list)
    steps: list[dict[str, Any]] = Field(default_factory=list)
    verification_results: list[dict[str, Any]] = Field(default_factory=list)