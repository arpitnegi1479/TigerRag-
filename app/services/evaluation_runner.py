from __future__ import annotations

import json
import os
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.llm.gemini import GeminiProvider
from app.repositories.graph_repository import GraphRepository
from app.repositories.postgres_repository import PostgresDocumentRepository
from app.repositories.vector_repository import VectorRepository
from app.schemas.api import QueryMode, QueryResult
from app.schemas.evaluation import EvaluationResult
from app.services.agent_tools import AgentToolRegistry
from app.services.agentic_graphrag_service import AgenticGraphRagService
from app.services.evaluation_scoring import (
    answer_tokens,
    score_answer_correctness,
    score_citation_support,
    score_retrieval_at_k,
    split_claims,
)
from app.services.evaluation_telemetry import GeminiCallTrace, capture_gemini_calls
from app.services.graphrag_service import GraphRagService
from app.services.rag_service import RagService
from app.services.verification_service import DEFAULT_FALLBACK_VERSION, VerificationService

MODES = (QueryMode.RAG, QueryMode.GRAPHRAG, QueryMode.AGENTIC)
BENCHMARK_PATH = Path(__file__).resolve().parents[2] / "data" / "benchmark_questions.json"


def load_questions() -> tuple[str, list[dict[str, Any]]]:
    payload = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    return payload["dataset_version"], payload["questions"]


def verify_live_benchmark_stores() -> dict[str, int]:
    dataset = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    document_ids = dataset["corpus_documents"]
    graph = GraphRepository.get_default()
    vectors = VectorRepository.get_default()
    postgres = PostgresDocumentRepository.get_default()
    if settings.graph_backend.lower() != "neo4j" or not graph.is_live:
        raise RuntimeError("Evaluation requires the live Neo4j benchmark graph.")
    if not vectors.is_live:
        raise RuntimeError("Evaluation requires the live Qdrant benchmark collection.")
    if postgres._conn is None:
        raise RuntimeError("Evaluation requires the live Postgres benchmark documents.")
    counts = {
        "annotated_edges": len(graph.list_annotated_relationships()),
        "qdrant_benchmark_points": vectors.count_documents(document_ids),
        "postgres_benchmark_documents": postgres.count_documents(document_ids),
    }
    if counts["annotated_edges"] == 0:
        raise RuntimeError("Evaluation refused: no benchmark-annotated graph edges are present.")
    if counts["qdrant_benchmark_points"] == 0:
        raise RuntimeError("Evaluation refused: no benchmark vectors are present in Qdrant.")
    if counts["postgres_benchmark_documents"] != len(document_ids):
        raise RuntimeError("Evaluation refused: Postgres does not contain all benchmark documents.")
    return counts


def capture_configuration(
    top_k: int,
    max_depth: int,
    pacing_seconds: float,
    fallback_version: str = DEFAULT_FALLBACK_VERSION,
) -> dict[str, Any]:
    return {
        "model": settings.gemini_model,
        "embedding_model": settings.embedding_model if settings.gemini_api_key else "deterministic-384",
        "embedding_dimension": settings.embedding_dimension if settings.gemini_api_key else 384,
        "top_k": top_k,
        "max_graph_depth": max_depth,
        "max_agent_steps": settings.max_agent_steps,
        "max_tool_calls": settings.max_tool_calls,
        "temperature": 0,
        "gemini_configured": bool(settings.gemini_api_key),
        "retry_max_attempts": 2,
        "minimum_gemini_request_interval_seconds": pacing_seconds,
        "cache_enabled": False,
        "cache_status": "fresh",
        "faithfulness_fallback_version": fallback_version,
    }


def _execute_service(
    mode: QueryMode,
    question: dict[str, Any],
    top_k: int,
    max_depth: int,
) -> tuple[QueryResult, VerificationService]:
    provider = GeminiProvider()
    verifier = VerificationService(provider)
    if mode == QueryMode.RAG:
        return RagService(provider=provider, verifier=verifier).answer(question["question"], top_k=top_k), verifier
    if mode == QueryMode.GRAPHRAG:
        return GraphRagService(provider=provider, verifier=verifier).answer(question["question"], max_depth=max_depth), verifier
    tools = AgentToolRegistry(verifier=verifier)
    return AgenticGraphRagService(provider=provider, tools=tools).answer(question["question"]), verifier


def _citation_evidence(citations: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    documents = PostgresDocumentRepository.get_default()
    evidence_citations = []
    document_ids = []
    seen_chunks = set()
    for citation in citations:
        document_id = citation.get("document_id")
        chunk_id = citation.get("chunk_id")
        if document_id:
            document_ids.append(str(document_id))
        identity = (document_id, chunk_id)
        if identity in seen_chunks:
            continue
        seen_chunks.add(identity)
        chunk = documents.get_chunk(str(chunk_id)) if chunk_id else None
        evidence_citations.append(
            {
                **citation,
                "content": chunk.get("content") if chunk else citation.get("snippet"),
            }
        )
    return evidence_citations, list(dict.fromkeys(document_ids))


def _faithfulness(
    answer: str,
    evidence_citations: list[dict[str, Any]],
    verifier: VerificationService,
) -> dict[str, Any]:
    claims = split_claims(answer)
    cited_texts = [str(item["content"]) for item in evidence_citations if item.get("content")]
    if not claims or not cited_texts:
        return {"score": None, "method": None, "claims": []}

    results = []
    verdict_scores = {
        "SUPPORTS": 1.0,
        "PARTIALLY_SUPPORTS": 0.5,
        "CONTRADICTS": 0.0,
        "DOES_NOT_SUPPORT": 0.0,
    }
    cited_evidence = "\n\n".join(cited_texts)
    for claim in claims:
        verification = verifier.verify_claim(
            claim,
            cited_evidence,
            telemetry_step="faithfulness_verification",
        )
        results.append(
            {
                "claim": claim,
                "verdict": verification.verdict.value,
                "explanation": verification.explanation,
                "score": verdict_scores[verification.verdict.value],
                "verification_method": (
                    "lexical_fallback"
                    if verification.explanation.startswith("Fallback lexical verification")
                    else "gemini"
                ),
            }
        )
    return {
        "score": round(sum(item["score"] for item in results) / len(results), 4),
        "method": "existing_verify_claim_on_cited_chunks",
        "claims": results,
    }


def rescore_fallback_faithfulness(state: dict[str, Any]) -> int:
    class FallbackOnlyProvider:
        configured = False

    verifier = VerificationService(FallbackOnlyProvider())
    updated_claims = 0
    verdict_scores = {
        "SUPPORTS": 1.0,
        "PARTIALLY_SUPPORTS": 0.5,
        "CONTRADICTS": 0.0,
        "DOES_NOT_SUPPORT": 0.0,
    }
    for result in state.get("results", []):
        evidence_citations, _ = _citation_evidence(result.get("citations", []))
        cited_evidence = "\n\n".join(
            str(item["content"]) for item in evidence_citations if item.get("content")
        )
        faithfulness = result.get("scores", {}).get("faithfulness", {})
        claims = faithfulness.get("claims", [])
        revised = 0
        for claim_result in claims:
            if claim_result.get("verification_method") != "lexical_fallback":
                continue
            verification = verifier.verify_claim(
                claim_result["claim"],
                cited_evidence,
                telemetry_step="faithfulness_rescore",
            )
            claim_result["verdict"] = verification.verdict.value
            claim_result["explanation"] = verification.explanation
            claim_result["score"] = verdict_scores[verification.verdict.value]
            revised += 1
        if revised:
            previous_score = faithfulness.get("score")
            faithfulness["score"] = round(
                sum(claim.get("score", 0.0) for claim in claims) / len(claims), 4
            ) if claims else None
            faithfulness["method"] = "existing_verify_claim_on_cited_chunks_with_claim_local_lexical_fallback"
            result.setdefault("scoring_revisions", []).append(
                {
                    "reason": "Corrected fallback negation scope to the most claim-relevant evidence sentence.",
                    "recomputed_fallback_claims": revised,
                    "previous_faithfulness_score": previous_score,
                    "corrected_faithfulness_score": faithfulness["score"],
                }
            )
            updated_claims += revised
    state["summary"] = EvaluationRunner._summary(state.get("results", []))
    state["scoring_revision"] = "claim-local-negation-fallback-v2"
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    return updated_claims


def _graph_metrics(question: dict[str, Any], result: QueryResult, mode: QueryMode) -> dict[str, Any] | None:
    if mode == QueryMode.RAG:
        return None

    repository = GraphRepository.get_default()
    entities = {item["id"]: item.get("canonical_name", item["id"]) for item in repository.list_entities()}
    relationship_lookup = repository.list_relationships()
    observed_paths = []
    observed_edges = {}
    for graph_path in result.graph_paths:
        path_ids = graph_path.get("path") or []
        path_names = [entities.get(entity_id, entity_id) for entity_id in path_ids]
        path_relationships = graph_path.get("relationships") or []
        if path_ids and not graph_path.get("relationships"):
            for left, right in zip(path_ids, path_ids[1:]):
                path_relationships.extend(
                    relationship
                    for relationship in relationship_lookup
                    if {relationship.get("source"), relationship.get("target")} == {left, right}
                )
        for relationship in path_relationships:
            source = relationship.get("source")
            target = relationship.get("target")
            source_name = entities.get(source, source) if source else None
            target_name = entities.get(target, target) if target else None
            if source_name:
                path_names.append(source_name)
            if target_name:
                path_names.append(target_name)
            key = (
                source,
                target,
                relationship.get("type"),
                relationship.get("source_document_id"),
                relationship.get("source_chunk_id"),
            )
            observed_edges[key] = relationship
        observed_paths.append(list(dict.fromkeys(path_names)))

    required_entities = question.get("required_entities", [])
    expected_path_found = None
    if required_entities and question.get("required_relationships"):
        required = {" ".join(sorted(answer_tokens(name))) for name in required_entities}
        expected_path_found = any(
            required.issubset({" ".join(sorted(answer_tokens(name))) for name in path})
            for path in observed_paths
        )
    path_depths = [len(path) - 1 for path in observed_paths if path]
    return {
        "nodes_explored": len({name for path in observed_paths for name in path}),
        "nodes_on_returned_paths": len({name for path in observed_paths for name in path}),
        "nodes_explored_definition": "unique nodes exposed by returned graph paths/tool results; backend-internal search expansions are not exposed",
        "edges_traversed": len(observed_edges),
        "depth": max(path_depths) if path_depths else result.retrieval_stats.path_length,
        "expected_path_found": expected_path_found,
    }


def _agent_metrics(result: QueryResult, mode: QueryMode) -> dict[str, Any] | None:
    if mode != QueryMode.AGENTIC:
        return None
    steps = [item.model_dump(mode="json") for item in result.agent_steps]
    tools = [item for item in steps if item.get("tool")]
    return {
        "steps": len(steps),
        "tool_calls": len(tools),
        "rewrites": sum(item.get("tool") == "rewrite_query" for item in tools),
        "verification_attempts": sum(item.get("tool") == "verify_claim" for item in tools),
        "successful_calls": sum(item.get("status") == "completed" for item in tools),
        "failed_calls": sum(item.get("status") == "failed" for item in tools),
    }


def evaluate_execution(
    question: dict[str, Any],
    result: QueryResult,
    mode: QueryMode,
    verifier: VerificationService,
    trace: GeminiCallTrace,
    top_k: int,
) -> dict[str, Any]:
    citations = [item.model_dump(mode="json") for item in result.citations]
    evidence_citations, retrieved_document_ids = _citation_evidence(citations)
    correctness = score_answer_correctness(question, result.answer, result.evidence_status.value)
    retrieval = score_retrieval_at_k(retrieved_document_ids, question.get("gold_sources", []), top_k)
    claims = split_claims(result.answer)
    citation_accuracy = score_citation_support(claims, evidence_citations)
    faithfulness = _faithfulness(result.answer, evidence_citations, verifier)
    graph_metrics = _graph_metrics(question, result, mode)
    agent_metrics = _agent_metrics(result, mode)
    return {
        "citations": citations,
        "retrieval_document_ids": retrieved_document_ids,
        "scores": {
            "answer_correctness": correctness,
            "retrieval": retrieval,
            "faithfulness": faithfulness,
            "citation_accuracy": citation_accuracy,
        },
        "metrics": {
            "graph": graph_metrics,
            "agent": agent_metrics,
        },
        "telemetry": trace.summary(),
    }


class EvaluationRunner:
    def __init__(
        self,
        *,
        top_k: int = 5,
        max_depth: int | None = None,
        pacing_seconds: float = 0.5,
        fallback_version: str = DEFAULT_FALLBACK_VERSION,
    ):
        self.top_k = top_k
        self.max_depth = max_depth if max_depth is not None else settings.max_graph_depth
        self.pacing_seconds = pacing_seconds
        self.fallback_version = fallback_version

    def _execute_mode(self, mode: QueryMode, question: dict[str, Any]) -> tuple[QueryResult, VerificationService]:
        provider = GeminiProvider()
        verifier = VerificationService(provider, fallback_version=self.fallback_version)
        if mode == QueryMode.RAG:
            result = RagService(provider=provider, verifier=verifier).answer(question["question"], top_k=self.top_k)
        elif mode == QueryMode.GRAPHRAG:
            result = GraphRagService(provider=provider, verifier=verifier).answer(question["question"], max_depth=self.max_depth)
        else:
            tools = AgentToolRegistry(verifier=verifier)
            result = AgenticGraphRagService(provider=provider, tools=tools).answer(question["question"])
        return result, verifier

    @staticmethod
    def select_questions(
        questions: list[dict[str, Any]],
        *,
        question_ids: set[str] | None = None,
        category: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        selected = questions
        if question_ids is not None:
            if not question_ids:
                raise ValueError("Question ID selection cannot be empty.")
            selected = [question for question in selected if question["id"] in question_ids]
            missing = question_ids - {question["id"] for question in selected}
            if missing:
                raise ValueError(f"Unknown benchmark question IDs: {sorted(missing)}")
        if category is not None:
            if not category:
                raise ValueError("Question category selection cannot be empty.")
            selected = [question for question in selected if question["category"] == category]
        if limit is not None:
            selected = selected[:limit]
        return selected

    @staticmethod
    def _checkpoint(path: Path, state: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary_path, path)

    @staticmethod
    def _persist_checkpoint(path: Path, state: dict[str, Any]) -> None:
        EvaluationRunner._checkpoint(path, state)
        PostgresDocumentRepository.get_default().save_evaluation_run(state)

    @staticmethod
    def _summary(
        results: list[dict[str, Any]],
        total_question_count: int | None = None,
    ) -> dict[str, Any]:
        def average(items: list[dict[str, Any]], getter) -> float | None:
            values = [value for item in items if (value := getter(item)) is not None]
            return round(sum(values) / len(values), 4) if values else None

        grouped = defaultdict(list)
        question_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in results:
            grouped[(item["mode"], item["execution_quality"])].append(item)
            question_groups[item["question_id"]].append(item)
        by_mode = {}
        for mode in (item.value for item in MODES):
            by_mode[mode] = {}
            for quality in ("clean", "degraded"):
                items = grouped[(mode, quality)]
                by_mode[mode][quality] = {
                    "executions": len(items),
                    "mean_answer_correctness": average(items, lambda item: item["scores"]["answer_correctness"]["score"]),
                    "mean_precision_at_k": average(items, lambda item: item["scores"]["retrieval"]["precision_at_k"]),
                    "mean_recall_at_k": average(items, lambda item: item["scores"]["retrieval"]["recall_at_k"]),
                    "mean_faithfulness": average(items, lambda item: item["scores"]["faithfulness"]["score"]),
                    "mean_citation_accuracy": average(items, lambda item: item["scores"]["citation_accuracy"]["score"]),
                    "gemini_http_attempts": sum(item["llm_telemetry"]["gemini_http_attempts"] for item in items),
                    "gemini_logical_requests": sum(item["llm_telemetry"]["gemini_logical_requests"] for item in items),
                }
        by_question = {}
        for question_id, items in question_groups.items():
            by_mode_for_question = {}
            for item in items:
                telemetry = item["llm_telemetry"]
                by_mode_for_question[item["mode"]] = {
                    "gemini_http_attempts": telemetry["gemini_http_attempts"],
                    "gemini_logical_requests": telemetry["gemini_logical_requests"],
                    "execution_quality": item["execution_quality"],
                }
            by_question[question_id] = {
                "executions": len(items),
                "gemini_http_attempts": sum(item["llm_telemetry"]["gemini_http_attempts"] for item in items),
                "gemini_logical_requests": sum(item["llm_telemetry"]["gemini_logical_requests"] for item in items),
                "clean_executions": sum(item["execution_quality"] == "clean" for item in items),
                "degraded_executions": sum(item["execution_quality"] == "degraded" for item in items),
                "by_mode": by_mode_for_question,
            }

        summary = {
            "executions": len(results),
            "clean_executions": sum(item["execution_quality"] == "clean" for item in results),
            "degraded_executions": sum(item["execution_quality"] == "degraded" for item in results),
            "gemini_http_attempts": sum(item["llm_telemetry"]["gemini_http_attempts"] for item in results),
            "gemini_logical_requests": sum(item["llm_telemetry"]["gemini_logical_requests"] for item in results),
            "by_question": by_question,
            "by_mode": by_mode,
        }
        if total_question_count and results:
            full_run_executions = total_question_count * len(MODES)
            summary["full_run_projection"] = {
                "question_count": total_question_count,
                "execution_count": full_run_executions,
                "estimated_gemini_http_attempts": round(
                    summary["gemini_http_attempts"] / len(results) * full_run_executions
                ),
                "estimated_gemini_logical_requests": round(
                    summary["gemini_logical_requests"] / len(results) * full_run_executions
                ),
                "method": "linear extrapolation from observed executions; quota use may differ by category and degradation",
            }
        return summary

    def run(
        self,
        *,
        questions: list[dict[str, Any]],
        dataset_version: str,
        output_path: Path,
        question_ids: set[str] | None = None,
        category: str | None = None,
        limit: int | None = None,
        resume: bool = False,
    ) -> dict[str, Any]:
        selected = self.select_questions(questions, question_ids=question_ids, category=category, limit=limit)
        if not selected:
            raise ValueError("Question selection is empty.")
        verify_live_benchmark_stores()
        configuration = capture_configuration(
            self.top_k,
            self.max_depth,
            self.pacing_seconds,
            fallback_version=self.fallback_version,
        )
        selection = {
            "question_ids": sorted(question_ids) if question_ids is not None else None,
            "category": category,
            "limit": limit,
        }
        if resume:
            if not output_path.is_file():
                raise FileNotFoundError(f"Cannot resume; checkpoint does not exist: {output_path}")
            state = json.loads(output_path.read_text(encoding="utf-8"))
            if state.get("dataset_version") != dataset_version or state.get("configuration") != configuration:
                raise ValueError("Cannot resume with a different dataset version or execution configuration.")
            if state.get("selection") != selection:
                raise ValueError("Cannot resume with a different question selection.")
        else:
            if output_path.exists():
                raise FileExistsError(f"Output already exists; pass resume=True to continue: {output_path}")
            state = {
                "run_id": str(uuid.uuid4()),
                "dataset_version": dataset_version,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "running",
                "checkpoint_path": str(output_path.resolve()),
                "configuration": configuration,
                "selection": selection,
                "results": [],
            }
            self._persist_checkpoint(output_path, state)

        repository = PostgresDocumentRepository.get_default()
        repository.save_evaluation_run(state)
        if resume:
            for result in state.get("results", []):
                repository.save_evaluation_result(result)

        completed = {(item["question_id"], item["mode"]) for item in state["results"]}
        for question in selected:
            for mode in MODES:
                if (question["id"], mode.value) in completed:
                    continue
                trace = GeminiCallTrace(self.pacing_seconds)
                started = time.perf_counter()
                with capture_gemini_calls(trace):
                    query_result, verifier = self._execute_mode(mode, question)
                    service_latency_ms = (time.perf_counter() - started) * 1000
                    evaluated = evaluate_execution(question, query_result, mode, verifier, trace, self.top_k)
                end_to_end_latency_ms = (time.perf_counter() - started) * 1000
                telemetry = evaluated["telemetry"]
                result_record = EvaluationResult(
                    result_id=str(uuid.uuid4()),
                    run_id=state["run_id"],
                    dataset_version=dataset_version,
                    question_id=question["id"],
                    category=question["category"],
                    mode=mode.value,
                    query=question["question"],
                    answer=query_result.answer,
                    evidence_status=query_result.evidence_status.value,
                    citations=evaluated["citations"],
                    raw_result=query_result.model_dump(mode="json"),
                    scores=evaluated["scores"],
                    metrics={
                        **evaluated["metrics"],
                        "performance": {
                            "service_latency_ms": round(service_latency_ms, 3),
                            "end_to_end_latency_ms": round(end_to_end_latency_ms, 3),
                            "llm_latency_ms": telemetry["llm_latency_ms"],
                            "token_usage": telemetry["token_usage"],
                            "estimated_cost_usd": telemetry["estimated_cost_usd"],
                        },
                    },
                    llm_telemetry=telemetry,
                    configuration=configuration,
                    execution_quality="degraded" if telemetry["degraded"] else "clean",
                    cache_status="fresh",
                ).model_dump(mode="json")
                state["results"].append(result_record)
                completed.add((question["id"], mode.value))
                state["updated_at"] = datetime.now(timezone.utc).isoformat()
                state["status"] = "running"
                state["summary"] = self._summary(state["results"], total_question_count=len(questions))
                repository.save_evaluation_result(result_record)
                self._persist_checkpoint(output_path, state)

        state["status"] = "completed"
        state["completed_at"] = datetime.now(timezone.utc).isoformat()
        state["summary"] = self._summary(state["results"], total_question_count=len(questions))
        self._persist_checkpoint(output_path, state)
        return state