from __future__ import annotations

from typing import Any

from app.llm.providers import get_embedding_provider
from app.repositories.graph_repository import GraphRepository
from app.repositories.postgres_repository import PostgresDocumentRepository
from app.repositories.vector_repository import VectorRepository
from app.services.verification_service import VerificationService


class AgentToolRegistry:
    """Standalone tools used by the adaptive agent loop."""

    def __init__(self, verifier: VerificationService | None = None):
        self.graph = GraphRepository.get_default()
        self.vector = VectorRepository.get_default()
        self.documents = PostgresDocumentRepository.get_default()
        self.verifier = verifier or VerificationService()

    def search_documents(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        embedding_provider = get_embedding_provider()
        vector = embedding_provider.embed(query)
        results = self.vector.search(embedding_provider.collection_name, vector, limit=top_k)
        terms = {term.lower() for term in query.split() if len(term) >= 3}
        if terms:
            for item in results:
                content_terms = set(str(item.get("content", "")).lower().split())
                lexical_overlap = len(terms & content_terms) / len(terms)
                item["score"] = max(float(item.get("score", 0.0)), lexical_overlap)
            results.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return results[:top_k]

    def search_entity(self, name: str) -> dict[str, Any] | None:
        return self.graph.get_entity_by_name(name)

    def traverse_graph(self, entity: str, depth: int = 2, relationship_type: str | None = None) -> list[dict[str, Any]]:
        resolved = self.graph.get_entity_by_name(entity) or self.graph._entity_by_id(entity)
        if resolved is None:
            return []
        relationships = self.graph.list_relationships()
        results = []
        seen_relationships: set[tuple[str | None, str | None, str | None]] = set()
        frontier = {resolved["id"]}
        visited = set(frontier)
        for current_depth in range(depth):
            next_frontier: set[str] = set()
            for relationship in relationships:
                if relationship_type and relationship.get("type") != relationship_type:
                    continue
                if relationship.get("source") == resolved["id"] and relationship.get("target") in visited:
                    continue
                if relationship.get("source") in frontier or relationship.get("target") in frontier:
                    identity = (
                        relationship.get("source"),
                        relationship.get("target"),
                        relationship.get("identity_key"),
                    )
                    if identity in seen_relationships:
                        continue
                    seen_relationships.add(identity)
                    results.append(relationship)
                    neighbor = relationship.get("target") if relationship.get("source") in frontier else relationship.get("source")
                    if neighbor and neighbor not in visited:
                        next_frontier.add(neighbor)
            visited.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break
        return results

    def find_relationship(self, entity_a: str, entity_b: str) -> list[dict[str, Any]]:
        return self.graph.find_relationships(entity_a, entity_b)

    def get_source(self, document_id: str, chunk_id: str) -> dict[str, Any] | None:
        return self.documents.get_chunk(chunk_id)

    def verify_claim(self, claim: str, evidence: str):
        return self.verifier.verify_claim(claim, evidence)

    def rewrite_query(self, original_query: str, missing_information: str) -> str:
        return f"{original_query} Focus on {missing_information}."

    def execute(self, action: str, arguments: dict[str, Any]) -> Any:
        tools = {
            "SEARCH_DOCUMENTS": self.search_documents,
            "SEARCH_ENTITY": self.search_entity,
            "TRAVERSE_GRAPH": self.traverse_graph,
            "FIND_RELATIONSHIP": self.find_relationship,
            "GET_SOURCE": self.get_source,
            "VERIFY_CLAIM": self.verify_claim,
            "REWRITE_QUERY": self.rewrite_query,
        }
        if action not in tools:
            raise ValueError(f"Unsupported agent action: {action}")
        return tools[action](**arguments)