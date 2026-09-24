from __future__ import annotations


class MetricsService:
    """Aggregates benchmark metrics and execution summaries."""

    def aggregate(self) -> dict:
        return {
            "total_runs": 0,
            "latest_run": None,
            "modes": {"rag": {}, "graphrag": {}, "agentic": {}},
            "notes": "Metrics are computed from real execution data once evaluation runs are collected.",
        }
