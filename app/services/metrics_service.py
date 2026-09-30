from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.repositories.postgres_repository import PostgresDocumentRepository


class MetricsService:
    """Aggregates benchmark metrics and execution summaries."""

    def aggregate(self) -> dict:
        repository = PostgresDocumentRepository.get_default()
        runs = repository.list_evaluation_runs(limit=200)
        results = [
            result
            for run in runs
            for result in repository.list_evaluation_results(run["run_id"])
        ]
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for result in results:
            grouped[(result["mode"], result.get("execution_quality", "clean"))].append(result)

        modes: dict[str, Any] = {}
        for mode in ("rag", "graphrag", "agentic"):
            modes[mode] = {}
            for quality in ("clean", "degraded"):
                records = grouped[(mode, quality)]
                modes[mode][quality] = {
                    "executions": len(records),
                    "mean_answer_correctness": _average(records, ("scores", "answer_correctness", "score")),
                    "mean_precision_at_k": _average(records, ("scores", "retrieval", "precision_at_k")),
                    "mean_recall_at_k": _average(records, ("scores", "retrieval", "recall_at_k")),
                    "mean_faithfulness": _average(records, ("scores", "faithfulness", "score")),
                    "mean_citation_accuracy": _average(records, ("scores", "citation_accuracy", "score")),
                    "mean_end_to_end_latency_ms": _average(records, ("metrics", "performance", "end_to_end_latency_ms")),
                }
        return {
            "total_runs": len(runs),
            "total_executions": len(results),
            "latest_run": runs[0] if runs else None,
            "modes": modes,
        }


def _average(records: list[dict[str, Any]], path: tuple[str, ...]) -> float | None:
    values = []
    for record in records:
        value: Any = record
        for key in path:
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return round(sum(values) / len(values), 4) if values else None
