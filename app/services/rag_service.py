from __future__ import annotations

from app.llm.gemini import GeminiProvider
from app.llm.prompts import RAG_GROUNDED_PROMPT, escape_untrusted_content
from app.llm.providers import get_embedding_provider
from app.repositories.vector_repository import VectorRepository
from app.schemas.api import Citation, EvidenceStatus, QueryMode, QueryResult, RetrievalStats
from app.services.evaluation_telemetry import evaluation_step, record_fallback
from app.services.verification_service import VerificationService


class RagService:
    """Traditional vector-retrieval answerer."""

    def __init__(self, provider: GeminiProvider | None = None, verifier: VerificationService | None = None):
        self.provider = provider or GeminiProvider()
        self.verifier = verifier or VerificationService(self.provider)

    def answer(self, query: str, top_k: int = 5) -> QueryResult:
        embedding_provider = get_embedding_provider()
        vector_repository = VectorRepository.get_default()
        with evaluation_step("retrieval_embedding"):
            query_vector = embedding_provider.embed(query)
        retrieved = vector_repository.search(embedding_provider.collection_name, query_vector=query_vector, limit=top_k)

        if not retrieved:
            return QueryResult(
                mode=QueryMode.RAG,
                answer="I could not find enough evidence in the indexed corpus to answer this query reliably.",
                citations=[],
                confidence=0.0,
                evidence_status=EvidenceStatus.INSUFFICIENT_EVIDENCE,
                retrieval_stats=RetrievalStats(top_k=top_k, nodes_searched=0, edges_searched=0, latency_ms=0.0),
                reasoning_summary="No indexed chunks matched the query embedding, so the answer was withheld instead of hallucinating.",
                latency_ms=0.0,
            )

        citations = [
            Citation(
                document_id=str(item.get("document_id")),
                chunk_id=str(item.get("chunk_id")),
                snippet=str(item.get("content"))[:400],
                score=float(item.get("score", 0.0)),
            )
            for item in retrieved
        ]

        evidence = "\n\n".join(
            f"[document_id={item.get('document_id')} chunk_id={item.get('chunk_id')} score={item.get('score', 0.0)}]\n{item.get('content', '')}"
            for item in retrieved
        )
        prompt = RAG_GROUNDED_PROMPT.format(query=query, evidence=escape_untrusted_content(evidence))
        answer_text = "\n\n".join(item.get("content", "") for item in retrieved[:2])
        if self.provider.configured:
            try:
                with evaluation_step("answer_generation"):
                    answer_text = self.provider.generate_text(prompt)
            except Exception:
                record_fallback("answer_generation", "gemini_generation_failed")
        else:
            record_fallback("answer_generation", "gemini_not_configured")

        verification = self.verifier.verify_claim(answer_text, evidence)
        average_score = sum(max(0.0, min(1.0, float(item.get("score", 0.0)))) for item in retrieved) / len(retrieved)
        provenance_quality = sum(bool(item.get("document_id") and item.get("chunk_id")) for item in retrieved) / len(retrieved)
        verification_signal = {
            "SUPPORTS": 1.0,
            "PARTIALLY_SUPPORTS": 0.6,
            "CONTRADICTS": 0.0,
            "DOES_NOT_SUPPORT": 0.0,
        }[verification.verdict.value]
        confidence = round(average_score * 0.45 + provenance_quality * 0.25 + verification_signal * 0.30, 3)
        evidence_status = EvidenceStatus.VERIFIED if verification.verdict.value == "SUPPORTS" else EvidenceStatus.INSUFFICIENT_EVIDENCE

        return QueryResult(
            mode=QueryMode.RAG,
            answer=answer_text or f"RAG answer for: {query}",
            citations=citations,
            confidence=confidence,
            evidence_status=evidence_status,
            retrieval_stats=RetrievalStats(top_k=len(retrieved), nodes_searched=len(retrieved), edges_searched=0, latency_ms=120.0),
            reasoning_summary=f"Retrieved {len(retrieved)} chunks; verification verdict: {verification.verdict.value}.",
            latency_ms=120.0,
        )
