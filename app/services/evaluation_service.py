from __future__ import annotations

from datetime import datetime, timezone


class EvaluationService:
    """Benchmark and evaluation service skeleton for fixed benchmark execution."""

    def __init__(self):
        self.dataset_version = "v1.0.0"

    def build_benchmark(self) -> list[dict]:
        return [
            {
                "id": "q1",
                "category": "relationship",
                "question": "What is the relationship between Entity A and Entity D?",
                "expected_answer": "Entity A relates to Entity D through the known graph path.",
                "required_entities": ["Entity A", "Entity D"],
                "required_relationships": ["Entity A -> Entity B -> Entity D"],
                "gold_sources": ["doc-3", "doc-5"],
            },
            {
                "id": "q2",
                "category": "multi-hop",
                "question": "What evidence supports the link between Entity C and Entity D?",
                "expected_answer": "The support is grounded in the graph path and source chunks.",
                "required_entities": ["Entity C", "Entity D"],
                "required_relationships": ["Entity C -> Entity D"],
                "gold_sources": ["doc-5"],
            },
        ]

    def run_benchmark(self) -> dict:
        run_id = f"eval-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        questions = self.build_benchmark()

        return {
            "run_id": run_id,
            "dataset_version": self.dataset_version,
            "status": "completed",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "questions": questions,
            "summary": {
                "total_questions": len(questions),
                "question_types": sorted({q["category"] for q in questions}),
            },
        }
