from fastapi import APIRouter

from app.schemas.api import CompareQueryRequest, QueryRequest, QueryResult, VerifyClaimRequest
from app.services.agentic_graphrag_service import AgenticGraphRagService
from app.services.comparison_service import ComparisonService
from app.services.graphrag_service import GraphRagService
from app.services.rag_service import RagService
from app.services.verification_service import VerificationService
from app.repositories.postgres_repository import PostgresDocumentRepository
from fastapi import HTTPException

router = APIRouter(prefix="/api/query", tags=["queries"])

rag_service = RagService()
graphrag_service = GraphRagService()
agentic_service = AgenticGraphRagService()
comparison_service = ComparisonService()
verification_service = VerificationService()


@router.post("/rag")
def query_rag(payload: QueryRequest) -> QueryResult:
    return rag_service.answer(payload.query, top_k=payload.top_k)


@router.post("/graphrag")
def query_graphrag(payload: QueryRequest) -> QueryResult:
    return graphrag_service.answer(payload.query, max_depth=payload.max_depth)


@router.post("/agentic")
def query_agentic(payload: QueryRequest) -> QueryResult:
    return agentic_service.answer(payload.query)


@router.get("/{query_id}/trace")
def get_query_trace(query_id: str) -> dict:
    trace = PostgresDocumentRepository.get_default().get_agent_trace(query_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Query trace not found")
    return trace


@router.post("/verify-claim")
def verify_claim(payload: VerifyClaimRequest):
    return verification_service.verify_claim(payload.claim, payload.evidence)


@router.post("/compare")
def compare_queries(payload: CompareQueryRequest) -> dict:
    return comparison_service.compare(payload.query, top_k=payload.top_k, max_depth=payload.max_depth)
