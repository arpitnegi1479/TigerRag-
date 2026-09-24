from __future__ import annotations

import json
import time
import uuid
from typing import Any

from pydantic import ValidationError

from app.core.config import settings
from app.llm.gemini import GeminiProvider
from app.llm.prompts import AGENT_DECISION_PROMPT
from app.schemas.agent import AgentAction, AgentDecision, AgentState, AgentToolCall
from app.schemas.api import AgentStep, Citation, EvidenceStatus, QueryMode, QueryResult, RetrievalStats
from app.services.agent_tools import AgentToolRegistry
from app.repositories.postgres_repository import PostgresDocumentRepository


class AgenticGraphRagService:
    """Adaptive, bounded agent loop with explicit state and sanitized traces."""

    def __init__(self, provider: GeminiProvider | None = None, tools: AgentToolRegistry | None = None):
        self.provider = provider or GeminiProvider()
        self.tools = tools or AgentToolRegistry()

    def _classify(self, query: str) -> tuple[str, list[str]]:
        lowered = query.lower()
        relationship_terms = ("relationship", "connected", "relate", "between", "path", "link", "how does", "license", "claim")
        query_type = "RELATIONSHIP" if any(term in lowered for term in relationship_terms) else "FACTUAL"
        tokens = query.replace("?", "").split()
        stopwords = {"What", "Which", "Who", "How", "Is", "Are", "The", "Between", "And", "Does", "Did"}
        entities: list[str] = []
        current: list[str] = []
        for token in tokens:
            clean = token.strip(",.!?:;")
            if clean[:1].isupper() and clean not in stopwords:
                current.append(clean)
            elif current:
                entities.append(" ".join(current))
                current = []
        if current:
            entities.append(" ".join(current))
        return query_type, list(dict.fromkeys(entities))

    def _decision_schema(self) -> dict[str, Any]:
        return {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "enum": [item.value for item in AgentAction]},
                "reason": {"type": "STRING"},
                "arguments": {"type": "OBJECT"},
            },
            "required": ["action", "reason", "arguments"],
        }

    def _model_decision(self, state: AgentState) -> AgentDecision | None:
        if not self.provider.configured:
            return None
        try:
            raw = self.provider.generate_json(
                AGENT_DECISION_PROMPT.format(state=state.model_dump_json()),
                self._decision_schema(),
            )
            return AgentDecision.model_validate_json(raw)
        except (ValidationError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            state.steps.append({"event": "invalid_decision_rejected"})
            return None
        except Exception:
            state.steps.append({"event": "decision_provider_failed"})
            return None

    def _fallback_decision(self, state: AgentState) -> AgentDecision:
        called = {call.action for call in state.tool_calls}
        if state.conflicts:
            return AgentDecision(action=AgentAction.FINALIZE, reason="Conflicting claims require an explicit conflict result.")
        document_calls = [call for call in state.tool_calls if call.action == AgentAction.SEARCH_DOCUMENTS]
        if document_calls and document_calls[-1].result_count == 0 and AgentAction.REWRITE_QUERY not in called:
            return AgentDecision(
                action=AgentAction.REWRITE_QUERY,
                reason="Initial retrieval returned no chunks; rewrite the query with a more specific evidence target.",
                arguments={"original_query": state.query, "missing_information": "the specific entities, event, or relationship needed to answer"},
            )
        if len(document_calls) == 1 and document_calls[-1].result_count == 0 and AgentAction.REWRITE_QUERY in called:
            return AgentDecision(action=AgentAction.SEARCH_DOCUMENTS, reason="Retry retrieval using the rewritten query.", arguments={"query": state.query, "top_k": 5})
        if state.query_type == "FACTUAL":
            if AgentAction.SEARCH_DOCUMENTS not in called:
                return AgentDecision(action=AgentAction.SEARCH_DOCUMENTS, reason="A factual query needs document evidence first.", arguments={"query": state.query, "top_k": 5})
            return AgentDecision(action=AgentAction.FINALIZE, reason="Retrieved document evidence is sufficient for this factual query.")
        if AgentAction.SEARCH_DOCUMENTS not in called:
            return AgentDecision(action=AgentAction.SEARCH_DOCUMENTS, reason="Start with broad evidence for a relationship query.", arguments={"query": state.query, "top_k": 5})
        searched = {call.arguments.get("name") for call in state.tool_calls if call.action == AgentAction.SEARCH_ENTITY}
        endpoint_entities = state.entities if len(state.entities) <= 2 else [state.entities[0], state.entities[-1]]
        for entity in endpoint_entities:
            if entity not in searched:
                return AgentDecision(action=AgentAction.SEARCH_ENTITY, reason="Resolve a named graph entity before traversal.", arguments={"name": entity})
        if any(item.startswith("entity:") for item in state.missing_information):
            return AgentDecision(action=AgentAction.FINALIZE, reason="A named endpoint was not found, so traversal would not be credible.")
        relationship_checks = [call for call in state.tool_calls if call.action == AgentAction.FIND_RELATIONSHIP]
        if not relationship_checks and len(state.entities) >= 2:
            return AgentDecision(action=AgentAction.FIND_RELATIONSHIP, reason="Check for direct claims before expanding to multi-hop traversal.", arguments={"entity_a": state.entities[0], "entity_b": state.entities[-1]})
        if not state.graph_paths:
            if relationship_checks and relationship_checks[-1].result_count == 0:
                traversal_checks = [call for call in state.tool_calls if call.action == AgentAction.TRAVERSE_GRAPH]
                if traversal_checks and traversal_checks[-1].result_count == 0 and state.retrieved_chunks and not any(item.get("content") for item in state.evidence):
                    best_chunk = state.retrieved_chunks[0]
                    return AgentDecision(
                        action=AgentAction.GET_SOURCE,
                        reason="Graph traversal found no path; inspect the best retrieved document evidence before stopping.",
                        arguments={"document_id": best_chunk.get("document_id"), "chunk_id": best_chunk.get("chunk_id")},
                    )
                if any(item.get("content") for item in state.evidence) and not state.verification_results:
                    return AgentDecision(action=AgentAction.VERIFY_CLAIM, reason="Verify the grounded answer candidate from the retrieved source.", arguments={"claim": self._answer_from_state(state), "evidence": self._evidence_text(state)})
                if state.verification_results:
                    return AgentDecision(action=AgentAction.FINALIZE, reason="The fallback document evidence has been verified.")
                return AgentDecision(action=AgentAction.TRAVERSE_GRAPH, reason="No direct claim was found; expand to a bounded multi-hop traversal.", arguments={"entity": state.entities[0], "depth": 3})
            if len(state.entities) >= 2:
                return AgentDecision(action=AgentAction.TRAVERSE_GRAPH, reason="Traverse a bounded graph path between the query entities.", arguments={"entity": state.entities[0], "depth": 3})
            return AgentDecision(action=AgentAction.FINALIZE, reason="The query did not resolve two graph entities.")
        if not any(item.get("content") for item in state.evidence):
            evidence_ref = state.graph_paths[0].get("evidence", [{}])[0]
            return AgentDecision(action=AgentAction.GET_SOURCE, reason="Fetch source text for the traversed graph claim.", arguments={"document_id": evidence_ref.get("source_document_id"), "chunk_id": evidence_ref.get("source_chunk_id")})
        if not state.verification_results:
            return AgentDecision(action=AgentAction.VERIFY_CLAIM, reason="Verify the grounded answer candidate against source evidence.", arguments={"claim": self._answer_from_state(state), "evidence": self._evidence_text(state)})
        return AgentDecision(action=AgentAction.FINALIZE, reason="The path has source evidence and a verification result.")

    def _choose_decision(self, state: AgentState) -> AgentDecision:
        return self._model_decision(state) or self._fallback_decision(state)

    def _evidence_text(self, state: AgentState) -> str:
        source_evidence = [item.get("content", "") for item in state.evidence if item.get("content")]
        if source_evidence:
            return "\n\n".join(source_evidence)
        return "\n\n".join(item.get("content", "") for item in state.retrieved_chunks if item.get("content"))

    def _detect_conflicts(self, relationships: list[dict[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
        for relationship in relationships:
            key = (relationship.get("source"), relationship.get("target"))
            grouped.setdefault(key, []).append(relationship)
        conflicts = []
        for claims in grouped.values():
            types = {item.get("type") for item in claims if item.get("type")}
            documents = {item.get("source_document_id") for item in claims if item.get("source_document_id")}
            if len(types) > 1 and len(documents) > 1:
                conflicts.append({"claims": claims, "reason": "Multiple relationship claims were found for the same entity pair."})
        return conflicts

    def _apply_result(self, state: AgentState, decision: AgentDecision, result: Any) -> tuple[int, str]:
        action = decision.action
        if action == AgentAction.SEARCH_DOCUMENTS:
            state.retrieved_chunks = list(result or [])
            if not state.retrieved_chunks:
                state.missing_information.append("supporting document text")
            return len(state.retrieved_chunks), f"Retrieved {len(state.retrieved_chunks)} document chunks."
        if action == AgentAction.SEARCH_ENTITY:
            if result:
                state.evidence.append({"entity": result})
                return 1, f"Resolved entity {result.get('canonical_name', decision.arguments.get('name'))}."
            state.missing_information.append(f"entity:{decision.arguments.get('name')}")
            return 0, "Entity was not found."
        if action == AgentAction.TRAVERSE_GRAPH:
            relationships = list(result or [])
            if relationships:
                state.graph_paths.append({"entity": decision.arguments.get("entity"), "relationships": relationships, "evidence": relationships})
            return len(relationships), f"Found {len(relationships)} graph relationships."
        if action == AgentAction.FIND_RELATIONSHIP:
            relationships = list(result or [])
            state.evidence.extend({"relationship": item} for item in relationships)
            if relationships:
                state.graph_paths.append({"source": decision.arguments.get("entity_a"), "target": decision.arguments.get("entity_b"), "relationships": relationships, "evidence": relationships})
            state.conflicts.extend(self._detect_conflicts(relationships))
            return len(relationships), f"Found {len(relationships)} relationship claims."
        if action == AgentAction.GET_SOURCE:
            if result:
                state.evidence.append(result)
                return 1, "Loaded the cited source chunk."
            return 0, "The cited source chunk was unavailable."
        if action == AgentAction.VERIFY_CLAIM:
            verification = result.model_dump(mode="json") if hasattr(result, "model_dump") else result
            state.verification_results.append(verification)
            if verification.get("verdict") == "CONTRADICTS":
                state.conflicts.append({"verification": verification})
            return 1, f"Verification verdict: {verification.get('verdict')}."
        if action == AgentAction.REWRITE_QUERY:
            state.query = str(result)
            return 1, f"Rewrote query to: {state.query}"
        return 0, "Finalization selected."

    def _answer_from_state(self, state: AgentState) -> str:
        if state.query_type == "RELATIONSHIP" and state.graph_paths:
            relationships = state.graph_paths[0].get("relationships", [])
            return "Graph evidence: " + "; ".join(f"{item.get('source')} -[{item.get('type')}]-> {item.get('target')}" for item in relationships)
        if state.retrieved_chunks:
            return str(state.retrieved_chunks[0].get("content", ""))
        return "Evidence was found, but it was insufficient to form a grounded answer."

    def _final_result(self, state: AgentState, started: float) -> QueryResult:
        citations: list[Citation] = []
        for item in state.retrieved_chunks:
            citations.append(Citation(document_id=item.get("document_id"), chunk_id=item.get("chunk_id"), snippet=str(item.get("content", ""))[:400], score=item.get("score")))
        for path in state.graph_paths:
            for evidence in path.get("evidence", []):
                if evidence.get("source_document_id"):
                    citations.append(Citation(document_id=evidence.get("source_document_id"), chunk_id=evidence.get("source_chunk_id"), score=evidence.get("confidence", 0.0)))
        for item in state.evidence:
            relationship = item.get("relationship", {})
            if relationship.get("source_document_id"):
                citations.append(Citation(document_id=relationship.get("source_document_id"), chunk_id=relationship.get("source_chunk_id"), score=relationship.get("confidence", 0.0)))
        unique_citations = {(item.document_id, item.chunk_id): item for item in citations}
        if state.conflicts:
            status = EvidenceStatus.CONFLICTING_EVIDENCE
            answer = "Conflicting evidence was found for the requested claim; the agent did not select one side."
            confidence = 0.0
        elif state.query_type == "RELATIONSHIP" and not state.graph_paths:
            verification_verdicts = {item.get("verdict") for item in state.verification_results}
            if state.retrieved_chunks and "SUPPORTS" in verification_verdicts:
                status = EvidenceStatus.VERIFIED
                answer = self._answer_from_state(state)
                confidence = min(1.0, round(0.4 + 0.1 * len(state.verification_results), 3))
            else:
                status = EvidenceStatus.INSUFFICIENT_EVIDENCE
                answer = "I could not find enough evidence in the current corpus to answer reliably."
                confidence = 0.0
        elif not state.retrieved_chunks and not state.graph_paths:
            status = EvidenceStatus.INSUFFICIENT_EVIDENCE
            answer = "I could not find enough evidence in the current corpus to answer reliably."
            confidence = 0.0
        else:
            status = EvidenceStatus.VERIFIED
            answer = self._answer_from_state(state)
            confidence = min(1.0, round(0.45 + 0.1 * len(state.verification_results) + 0.1 * bool(state.graph_paths), 3))
        returned_citations = [] if status == EvidenceStatus.INSUFFICIENT_EVIDENCE else list(unique_citations.values())
        elapsed = round((time.perf_counter() - started) * 1000, 2)
        trace = [AgentStep(step=index + 1, action=item["action"], tool=item["tool"], input_summary=item["input_summary"], result_summary=item["result_summary"], status=item["status"], latency_ms=item["latency_ms"]) for index, item in enumerate(state.steps) if item.get("action")]
        query_id = str(uuid.uuid4())
        result = QueryResult(
            query_id=query_id,
            mode=QueryMode.AGENTIC,
            answer=answer,
            citations=returned_citations,
            confidence=confidence,
            evidence_status=status,
            retrieval_stats=RetrievalStats(top_k=len(state.retrieved_chunks), nodes_searched=len(state.entities), edges_searched=sum(len(path.get("relationships", [])) for path in state.graph_paths), latency_ms=elapsed),
            graph_paths=state.graph_paths,
            agent_steps=trace,
            reasoning_summary=f"Adaptive loop completed with {len(state.tool_calls)} tool calls and {len(trace)} trace steps.",
            latency_ms=elapsed,
        )
        PostgresDocumentRepository.get_default().save_agent_trace(query_id, state.query, result.model_dump(mode="json"))
        return result

    def answer(self, query: str, max_steps: int | None = None, max_tool_calls: int | None = None) -> QueryResult:
        started = time.perf_counter()
        query_type, entities = self._classify(query)
        state = AgentState(query=query, query_type=query_type, entities=entities, hypotheses=[f"Initial query classified as {query_type}."])
        step_limit = max_steps or settings.max_agent_steps
        tool_limit = max_tool_calls or settings.max_tool_calls
        for _ in range(step_limit):
            decision = self._choose_decision(state)
            if decision.action == AgentAction.FINALIZE:
                break
            if len(state.tool_calls) >= tool_limit:
                state.missing_information.append("tool-call limit reached")
                break
            tool_started = time.perf_counter()
            try:
                result = self.tools.execute(decision.action.value, decision.arguments)
                count, summary = self._apply_result(state, decision, result)
                status = "completed"
            except Exception as exc:
                count, summary, status = 0, f"Tool failed: {type(exc).__name__}", "failed"
            latency = round((time.perf_counter() - tool_started) * 1000, 2)
            state.tool_calls.append(AgentToolCall(action=decision.action, tool=decision.action.value.lower(), arguments=decision.arguments, result_count=count, result_summary=summary, latency_ms=latency, status=status))
            state.steps.append({"action": decision.action.value, "tool": decision.action.value.lower(), "input_summary": decision.reason, "result_summary": summary, "latency_ms": latency, "status": status})
        return self._final_result(state, started)
