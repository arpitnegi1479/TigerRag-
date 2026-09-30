from fastapi.testclient import TestClient

from app.main import app
from app.repositories.postgres_repository import PostgresDocumentRepository
from app.services.metrics_service import MetricsService


client = TestClient(app)


def test_explicit_empty_dsn_uses_memory_even_when_live_dsn_is_configured(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "postgres_dsn", "postgresql://configured/live")
    repository = PostgresDocumentRepository(dsn="")

    assert repository.dsn == ""
    assert repository._conn is None


def test_evaluation_run_and_results_round_trip_in_memory():
    repository = PostgresDocumentRepository(dsn="")
    run = {
        "run_id": "run-test",
        "dataset_version": "benchmark-v1.0",
        "status": "running",
        "created_at": "2026-09-30T00:00:00+00:00",
        "results": [{"result_id": "nested-result"}],
    }
    result = {
        "result_id": "result-test",
        "run_id": "run-test",
        "question_id": "f01",
        "category": "factual",
        "mode": "rag",
        "answer": "Helios leads Aster.",
        "evidence_status": "VERIFIED",
        "execution_quality": "clean",
        "scores": {
            "answer_correctness": {"score": 1.0},
            "retrieval": {"precision_at_k": 0.2, "recall_at_k": 1.0},
            "faithfulness": {"score": 1.0},
            "citation_accuracy": {"score": 1.0},
        },
        "metrics": {"performance": {"end_to_end_latency_ms": 30.0}},
    }

    repository.save_evaluation_run(run)
    repository.save_evaluation_result(result)

    assert repository.get_evaluation_run("run-test")["status"] == "running"
    assert "results" not in repository.get_evaluation_run("run-test")
    assert repository.list_evaluation_results("run-test") == [result]


def test_metrics_aggregate_real_saved_results_and_failures():
    repository = PostgresDocumentRepository.get_default()
    run = {
        "run_id": "metrics-test",
        "dataset_version": "benchmark-v1.0",
        "status": "completed",
        "created_at": "2026-09-30T00:00:00+00:00",
        "checkpoint_path": "C:/internal/evaluation.json",
    }
    result = {
        "result_id": "metrics-result",
        "run_id": "metrics-test",
        "question_id": "m01",
        "category": "multi_hop",
        "mode": "agentic",
        "answer": "Partial answer",
        "evidence_status": "INSUFFICIENT_EVIDENCE",
        "execution_quality": "degraded",
        "scores": {
            "answer_correctness": {"score": 0.5},
            "retrieval": {"precision_at_k": 0.0, "recall_at_k": 0.0},
            "faithfulness": {"score": 0.5},
            "citation_accuracy": {"score": 0.0},
        },
        "metrics": {"performance": {"end_to_end_latency_ms": 80.0}},
    }
    repository.save_evaluation_run(run)
    repository.save_evaluation_result(result)

    metrics = MetricsService().aggregate()
    metrics_response = client.get("/api/metrics")
    run_response = client.get("/api/evaluation/runs/metrics-test")
    failures_response = client.get("/api/evaluation/runs/metrics-test/failures")

    assert metrics["total_runs"] == 1
    assert metrics["total_executions"] == 1
    assert metrics["modes"]["agentic"]["degraded"]["mean_answer_correctness"] == 0.5
    assert metrics_response.status_code == 200
    assert metrics_response.json()["total_executions"] == 1
    assert run_response.status_code == 200
    assert run_response.json()["results"] == [result]
    assert "checkpoint_path" not in run_response.json()
    assert "checkpoint_path" not in metrics_response.json()["latest_run"]
    failures = failures_response.json()["failures"]
    assert failures_response.status_code == 200
    assert {item["metric"] for item in failures[0]["reasons"]} == {
        "answer_correctness",
        "retrieval_recall_at_k",
        "faithfulness",
        "citation_accuracy",
        "execution_quality",
    }


def test_evaluate_request_validation_rejects_invalid_limit():
    response = client.post("/api/evaluate", json={"limit": 51})

    assert response.status_code == 422


def test_evaluate_request_rejects_empty_question_subset():
    response = client.post("/api/evaluate", json={"question_ids": []})

    assert response.status_code == 422


def test_evaluate_request_rejects_empty_category():
    response = client.post("/api/evaluate", json={"category": ""})

    assert response.status_code == 422


def test_evaluate_route_starts_selected_benchmark_run(monkeypatch, tmp_path):
    import app.api.routes.evaluation as evaluation_route

    seen = {}

    def fake_run(self, **kwargs):
        seen.update(kwargs)
        return {
            "run_id": "run-api-test",
            "dataset_version": "benchmark-v1.0",
            "status": "completed",
            "summary": {"executions": 3},
        }

    monkeypatch.setattr(evaluation_route, "RUN_DIRECTORY", tmp_path)
    monkeypatch.setattr(evaluation_route.EvaluationRunner, "run", fake_run)
    response = client.post("/api/evaluate", json={"question_ids": ["f01"]})

    assert response.status_code == 200
    assert response.json()["run_id"] == "run-api-test"
    assert seen["question_ids"] == {"f01"}
    assert seen["resume"] is False
