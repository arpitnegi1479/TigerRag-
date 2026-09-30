import json
import sys

import pytest

from app.llm.gemini import GeminiProvider
from app.schemas.api import EvidenceStatus, QueryMode, QueryResult
from app.services.evaluation_runner import EvaluationRunner, capture_configuration
from app.services.evaluation_telemetry import GeminiCallTrace, capture_gemini_calls, evaluation_step
from app.services.verification_service import VerificationService


class DisabledProvider:
    configured = False


def test_selection_supports_ids_category_and_limit():
    questions = [
        {"id": "a", "category": "factual"},
        {"id": "b", "category": "relationship"},
        {"id": "c", "category": "factual"},
    ]

    assert [q["id"] for q in EvaluationRunner.select_questions(questions, category="factual", limit=1)] == ["a"]
    assert [q["id"] for q in EvaluationRunner.select_questions(questions, question_ids={"b"})] == ["b"]


def test_selection_rejects_explicit_empty_id_set():
    with pytest.raises(ValueError, match="selection cannot be empty"):
        EvaluationRunner.select_questions([], question_ids=set())


def test_selection_rejects_explicit_empty_category():
    with pytest.raises(ValueError, match="category selection cannot be empty"):
        EvaluationRunner.select_questions([], category="")


def test_configuration_captures_fallback_verifier_version():
    configuration = capture_configuration(top_k=5, max_depth=3, pacing_seconds=0)
    legacy_configuration = capture_configuration(
        top_k=5,
        max_depth=3,
        pacing_seconds=0,
        fallback_version="claim-local-negation-v2",
    )

    assert configuration["faithfulness_fallback_version"] == "sentence-scoped-negation-v3"
    assert legacy_configuration["faithfulness_fallback_version"] == "claim-local-negation-v2"


def test_cli_resume_restores_checkpoint_verifier_version(tmp_path, monkeypatch, capsys):
    import scripts.run_benchmark as benchmark_cli

    checkpoint = tmp_path / "run.json"
    checkpoint.write_text(
        json.dumps(
            {
                "configuration": {"faithfulness_fallback_version": "claim-local-negation-v2"}
            }
        ),
        encoding="utf-8",
    )
    captured = {}

    class CapturingRunner:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self, **kwargs):
            return {
                "run_id": "run-test",
                "status": "completed",
                "dataset_version": "test-v1",
                "summary": {},
                "configuration": {},
            }

    monkeypatch.setattr(benchmark_cli, "EvaluationRunner", CapturingRunner)
    monkeypatch.setattr(benchmark_cli, "load_questions", lambda: ("test-v1", []))
    monkeypatch.setattr(sys, "argv", ["run_benchmark.py", "--resume", "--output", str(checkpoint)])

    assert benchmark_cli.main() == 0
    assert captured["fallback_version"] == "claim-local-negation-v2"
    assert "run-test" in capsys.readouterr().out


def test_checkpoint_resumes_after_last_completed_mode(tmp_path, monkeypatch):
    import app.services.evaluation_runner as runner_module

    monkeypatch.setattr(runner_module, "verify_live_benchmark_stores", lambda: {"annotated_edges": 23})
    runner = EvaluationRunner(pacing_seconds=0)
    calls = []

    def execute_mode(mode, question):
        calls.append((question["id"], mode.value))
        result = QueryResult(
            mode=mode,
            answer=question["expected_answer"],
            citations=[],
            evidence_status=EvidenceStatus.VERIFIED,
        )
        return result, VerificationService(DisabledProvider())

    monkeypatch.setattr(runner, "_execute_mode", execute_mode)
    question = {
        "id": "q1",
        "category": "factual",
        "question": "What is the answer?",
        "expected_answer": "The answer is forty two.",
        "acceptable_answer_variants": [],
        "gold_sources": [],
        "required_entities": [],
        "required_relationships": [],
    }
    output_path = tmp_path / "checkpoint.json"

    first = runner.run(
        questions=[question],
        dataset_version="test-v1",
        output_path=output_path,
        question_ids={"q1"},
    )
    assert len(first["results"]) == 3
    assert len(calls) == 3

    resumed = runner.run(
        questions=[question],
        dataset_version="test-v1",
        output_path=output_path,
        question_ids={"q1"},
        resume=True,
    )

    assert len(resumed["results"]) == 3
    assert len(calls) == 3


def test_retry_telemetry_counts_attempts_and_logical_call():
    trace = GeminiCallTrace(minimum_interval_seconds=0)

    with capture_gemini_calls(trace), evaluation_step("answer_generation"):
        trace.record_request(
            request_id="logical-1",
            provider="generation",
            attempt=1,
            status="failed",
            status_code=429,
            latency_ms=5,
            retry_wait_ms=0,
        )
        trace.record_request(
            request_id="logical-1",
            provider="generation",
            attempt=2,
            status="succeeded",
            status_code=200,
            latency_ms=7,
            usage={"promptTokenCount": 12, "candidatesTokenCount": 5, "totalTokenCount": 17},
        )

    summary = trace.summary()
    assert summary["gemini_http_attempts"] == 2
    assert summary["gemini_logical_requests"] == 1
    assert summary["gemini_logical_requests_failed"] == 0
    assert summary["token_usage"]["total"] == 17
    assert summary["degraded"] is False


def test_summary_reports_per_question_calls_and_full_run_projection():
    def result(question_id, mode, attempts, logical_requests):
        return {
            "question_id": question_id,
            "mode": mode,
            "execution_quality": "clean",
            "scores": {
                "answer_correctness": {"score": 1.0},
                "retrieval": {"precision_at_k": 1.0, "recall_at_k": 1.0},
                "faithfulness": {"score": 1.0},
                "citation_accuracy": {"score": 1.0},
            },
            "llm_telemetry": {
                "gemini_http_attempts": attempts,
                "gemini_logical_requests": logical_requests,
            },
        }

    summary = EvaluationRunner._summary(
        [result("q1", "rag", 4, 2), result("q2", "graphrag", 2, 1)],
        total_question_count=50,
    )

    assert summary["by_question"]["q1"]["gemini_http_attempts"] == 4
    assert summary["by_question"]["q1"]["by_mode"]["rag"]["gemini_logical_requests"] == 2
    assert summary["full_run_projection"]["execution_count"] == 150
    assert summary["full_run_projection"]["estimated_gemini_http_attempts"] == 450
    assert summary["full_run_projection"]["estimated_gemini_logical_requests"] == 225