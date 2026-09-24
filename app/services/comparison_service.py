from __future__ import annotations

import time

from app.services.agentic_graphrag_service import AgenticGraphRagService
from app.services.graphrag_service import GraphRagService
from app.services.rag_service import RagService


class ComparisonService:
    """Runs all three retrieval modes against the same query for honest comparison."""

    def __init__(self):
        self.rag_service = RagService()
        self.graphrag_service = GraphRagService()
        self.agentic_service = AgenticGraphRagService()

    def compare(self, query: str, top_k: int = 5, max_depth: int = 3) -> dict:
        started = time.perf_counter()
        rag_result = self.rag_service.answer(query, top_k=top_k)
        rag_elapsed = round((time.perf_counter() - started) * 1000, 2)
        started = time.perf_counter()
        graph_result = self.graphrag_service.answer(query, max_depth=max_depth)
        graph_elapsed = round((time.perf_counter() - started) * 1000, 2)
        started = time.perf_counter()
        agentic_result = self.agentic_service.answer(query)
        agentic_elapsed = round((time.perf_counter() - started) * 1000, 2)

        results = {
            "rag": self._with_measured_latency(rag_result.model_dump(), rag_elapsed),
            "graphrag": self._with_measured_latency(graph_result.model_dump(), graph_elapsed),
            "agentic": self._with_measured_latency(agentic_result.model_dump(), agentic_elapsed),
        }

        return {
            "query": query,
            **results,
            "comparison": {
                "rag": {
                    "retrieval_scope": "semantic document chunks from the vector store",
                    "graph_paths": False,
                    "adaptive_tool_selection": False,
                },
                "graphrag": {
                    "retrieval_scope": "entity and relationship subgraph with provenance",
                    "graph_paths": True,
                    "adaptive_tool_selection": False,
                },
                "agentic": {
                    "retrieval_scope": "adaptive combination of document, entity, graph, source, and verification tools",
                    "graph_paths": True,
                    "adaptive_tool_selection": True,
                },
            },
            "explanation": {
                "rag": "RAG answers from retrieved chunks, so its scope is semantic pass-through rather than graph traversal.",
                "graphrag": "GraphRAG answers from entity and relationship traversals across the subgraph and attached provenance.",
                "agentic": "Agentic GraphRAG chooses tools adaptively, resolves missing or conflicting evidence, and verifies before stopping.",
            },
        }

    @staticmethod
    def _with_measured_latency(result: dict, elapsed_ms: float) -> dict:
        result["latency_ms"] = elapsed_ms
        retrieval_stats = result.get("retrieval_stats") or {}
        retrieval_stats["latency_ms"] = elapsed_ms
        result["retrieval_stats"] = retrieval_stats
        return result
