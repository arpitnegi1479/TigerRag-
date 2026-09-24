from app.schemas.agent import AgentAction, AgentDecision, AgentState
from app.schemas.api import EvidenceStatus
from app.schemas.verification import ClaimVerification, VerificationVerdict
from app.services.agentic_graphrag_service import AgenticGraphRagService


class DisabledProvider:
    configured = False


class InvalidDecisionProvider:
    configured = True

    def generate_json(self, prompt, schema):
        return '{"action":"INVENTED_ACTION","reason":"invalid","arguments":{}}'


class FakeTools:
    def __init__(self, conflict=False, missing=False):
        self.conflict = conflict
        self.missing = missing

    def execute(self, action, arguments):
        if action == "SEARCH_DOCUMENTS":
            return [{"document_id": "doc-1", "chunk_id": "chunk-0", "content": "Northstar Labs deployed Orion Analytics.", "score": 0.9}]
        if action == "SEARCH_ENTITY":
            if arguments["name"] == "Omega":
                return None
            return {"id": f"entity:{arguments['name'].lower()}", "canonical_name": arguments["name"]}
        if action == "FIND_RELATIONSHIP":
            if self.conflict:
                return [
                    {"source": "entity:alpha", "target": "entity:beta", "type": "USES", "confidence": 0.9, "source_document_id": "doc-a", "source_chunk_id": "doc-a-0"},
                    {"source": "entity:alpha", "target": "entity:beta", "type": "PART_OF", "confidence": 0.9, "source_document_id": "doc-b", "source_chunk_id": "doc-b-0"},
                ]
            return []
        if action == "TRAVERSE_GRAPH":
            if self.missing:
                return []
            return [{"source": "entity:alpha", "target": "entity:gamma", "type": "RELATED_TO", "confidence": 0.9, "source_document_id": "doc-1", "source_chunk_id": "chunk-0"}]
        if action == "GET_SOURCE":
            return {"document_id": "doc-1", "chunk_id": "chunk-0", "content": "Alpha is related to Gamma."}
        if action == "VERIFY_CLAIM":
            return ClaimVerification(verdict=VerificationVerdict.SUPPORTS, explanation="The source supports the claim.")
        raise AssertionError(f"Unexpected action {action}")


class DocumentFallbackTools(FakeTools):
    def execute(self, action, arguments):
        if action == "SEARCH_DOCUMENTS":
            return [{"document_id": "doc-fallback", "chunk_id": "doc-fallback-0", "content": "Alpha runs on Gamma Cloud.", "score": 0.95}]
        if action == "GET_SOURCE":
            return {"document_id": "doc-fallback", "chunk_id": "doc-fallback-0", "content": "Alpha runs on Gamma Cloud."}
        return super().execute(action, arguments)


class RewriteTools:
    def __init__(self):
        self.searches = 0

    def execute(self, action, arguments):
        if action == "SEARCH_DOCUMENTS":
            self.searches += 1
            if self.searches == 1:
                return []
            return [{"document_id": "rewrite-doc", "chunk_id": "rewrite-doc-0", "content": "The specific answer is in this rewritten retrieval.", "score": 0.9}]
        if action == "REWRITE_QUERY":
            return f"{arguments['original_query']} {arguments['missing_information']}"
        raise AssertionError(f"Unexpected action {action}")


def test_simple_factual_query_takes_short_adaptive_trace():
    result = AgenticGraphRagService(provider=DisabledProvider(), tools=FakeTools()).answer("What did Northstar Labs deploy?")

    assert result.evidence_status == EvidenceStatus.VERIFIED
    assert [step.tool for step in result.agent_steps] == ["search_documents"]
    assert len(result.citations) == 1


def test_relationship_query_expands_to_graph_and_source_tools():
    result = AgenticGraphRagService(provider=DisabledProvider(), tools=FakeTools()).answer(
        "What is the relationship between Alpha and Gamma?"
    )

    tools = [step.tool for step in result.agent_steps]
    assert tools == ["search_documents", "search_entity", "search_entity", "find_relationship", "traverse_graph", "get_source", "verify_claim"]
    assert result.graph_paths
    assert result.evidence_status == EvidenceStatus.VERIFIED


def test_missing_relationship_returns_insufficient_evidence():
    result = AgenticGraphRagService(provider=DisabledProvider(), tools=FakeTools(missing=True)).answer(
        "What is the relationship between Alpha and Omega?"
    )

    assert result.evidence_status == EvidenceStatus.INSUFFICIENT_EVIDENCE
    assert result.citations == []


def test_conflicting_relationship_claims_are_reported_with_both_citations():
    result = AgenticGraphRagService(provider=DisabledProvider(), tools=FakeTools(conflict=True)).answer(
        "What is the relationship between Alpha and Beta?"
    )

    assert result.evidence_status == EvidenceStatus.CONFLICTING_EVIDENCE
    citation_keys = {(item.document_id, item.chunk_id) for item in result.citations}
    assert {("doc-a", "doc-a-0"), ("doc-b", "doc-b-0")} <= citation_keys


def test_failed_graph_lookup_falls_back_to_verified_document_evidence():
    result = AgenticGraphRagService(provider=DisabledProvider(), tools=DocumentFallbackTools(missing=True)).answer(
        "What is the relationship between Alpha and Gamma?"
    )

    assert result.evidence_status == EvidenceStatus.VERIFIED
    assert [step.tool for step in result.agent_steps] == [
        "search_documents", "search_entity", "search_entity", "find_relationship",
        "traverse_graph", "get_source", "verify_claim",
    ]
    assert result.citations[0].document_id == "doc-fallback"


def test_invalid_model_action_is_rejected_and_fallback_decides():
    service = AgenticGraphRagService(provider=InvalidDecisionProvider(), tools=FakeTools())
    state = AgentState(query="What did Northstar Labs deploy?", query_type="FACTUAL")

    decision = service._choose_decision(state)

    assert decision.action == AgentAction.SEARCH_DOCUMENTS
    assert state.steps == [{"event": "invalid_decision_rejected"}]


def test_empty_retrieval_forces_query_rewrite_and_retry():
    result = AgenticGraphRagService(provider=DisabledProvider(), tools=RewriteTools()).answer("What is this?")

    assert result.evidence_status == EvidenceStatus.VERIFIED
    assert [step.tool for step in result.agent_steps] == ["search_documents", "rewrite_query", "search_documents"]
    assert "rewritten retrieval" in result.answer
