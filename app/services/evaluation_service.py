from __future__ import annotations

from __future__ import annotations

from typing import Any


def build_failure_analysis(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures = []
    for result in results:
        scores = result.get("scores", {})
        correctness = scores.get("answer_correctness", {}).get("score")
        retrieval = scores.get("retrieval", {})
        faithfulness = scores.get("faithfulness", {}).get("score")
        citation_accuracy = scores.get("citation_accuracy", {}).get("score")
        reasons = []
        if correctness is not None and correctness < 1:
            reasons.append({"metric": "answer_correctness", "observed": correctness})
        if retrieval.get("recall_at_k") is not None and retrieval["recall_at_k"] < 1:
            reasons.append({"metric": "retrieval_recall_at_k", "observed": retrieval["recall_at_k"]})
        if faithfulness is not None and faithfulness < 1:
            reasons.append({"metric": "faithfulness", "observed": faithfulness})
        if citation_accuracy is not None and citation_accuracy < 1:
            reasons.append({"metric": "citation_accuracy", "observed": citation_accuracy})
        if result.get("execution_quality") == "degraded":
            reasons.append({"metric": "execution_quality", "observed": "degraded"})
        if reasons:
            failures.append(
                {
                    "question_id": result.get("question_id"),
                    "category": result.get("category"),
                    "mode": result.get("mode"),
                    "answer": result.get("answer"),
                    "evidence_status": result.get("evidence_status"),
                    "reasons": reasons,
                }
            )
    return failures
