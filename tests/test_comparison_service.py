from app.schemas.api import EvidenceStatus, QueryMode, QueryResult, RetrievalStats
from app.services.comparison_service import ComparisonService


def result(mode, status=EvidenceStatus.VERIFIED, latency=1.0):
    return QueryResult(
        mode=mode,
        answer=f"{mode.value} answer",
        evidence_status=status,
        confidence=0.5,
        retrieval_stats=RetrievalStats(latency_ms=latency),
    )


class FakeRag:
    def answer(self, query, top_k):
        return result(QueryMode.RAG)


class FakeGraph:
    def answer(self, query, max_depth):
        return result(QueryMode.GRAPHRAG, EvidenceStatus.INSUFFICIENT_EVIDENCE)


class FakeAgent:
    def answer(self, query):
        return result(QueryMode.AGENTIC, EvidenceStatus.CONFLICTING_EVIDENCE)


def test_compare_preserves_mode_results_and_measures_each_mode():
    service = ComparisonService()
    service.rag_service = FakeRag()
    service.graphrag_service = FakeGraph()
    service.agentic_service = FakeAgent()

    comparison = service.compare("same query", top_k=3, max_depth=2)

    assert comparison["rag"]["evidence_status"] == "VERIFIED"
    assert comparison["graphrag"]["evidence_status"] == "INSUFFICIENT_EVIDENCE"
    assert comparison["agentic"]["evidence_status"] == "CONFLICTING_EVIDENCE"
    assert comparison["rag"]["latency_ms"] >= 0
    assert comparison["graphrag"]["retrieval_stats"]["latency_ms"] >= 0
    assert comparison["comparison"]["agentic"]["adaptive_tool_selection"] is True
