from fastapi import APIRouter, HTTPException, Query

from app.repositories.graph_repository import GraphRepository
from app.repositories.postgres_repository import PostgresDocumentRepository

router = APIRouter(prefix="/api/graph", tags=["graph"])


@router.get("/")
def list_graph() -> dict:
    repository = GraphRepository.get_default()
    return {"nodes": len(repository.list_entities()), "relationships": len(repository.list_relationships()), "status": "live" if repository.is_live else "fallback"}


@router.get("/health")
def graph_health() -> dict:
    repository = GraphRepository.get_default()
    return {"status": "ok" if repository.is_live else "degraded", "backend": "neo4j" if repository.is_live else "memory"}


@router.get("/entity/{entity_id}")
def get_entity(entity_id: str) -> dict:
    repository = GraphRepository.get_default()
    entity = repository.get_entity_by_id(entity_id)
    if entity is None:
        raise HTTPException(status_code=404, detail="Entity not found")
    return {"entity": entity, "relationships": repository.get_entity_relationships(entity_id)}


@router.get("/path")
def get_path(source: str = Query(...), target: str = Query(...), max_depth: int = Query(3, ge=1, le=20)) -> dict:
    repository = GraphRepository.get_default()
    source_entity = repository.get_entity_by_name(source) or repository.get_entity_by_id(source)
    target_entity = repository.get_entity_by_name(target) or repository.get_entity_by_id(target)
    if source_entity is None or target_entity is None:
        raise HTTPException(status_code=404, detail="Source or target entity not found")
    path = repository.find_path(source_entity["id"], target_entity["id"], max_depth=max_depth)
    if path is None:
        raise HTTPException(status_code=404, detail="No graph path found")
    relationships = [item for item in repository.list_relationships() if item.get("source") in path and item.get("target") in path]
    sources = [{"document_id": item.get("source_document_id"), "chunk_id": item.get("source_chunk_id")} for item in relationships]
    evidence = [PostgresDocumentRepository.get_default().get_chunk(item.get("source_chunk_id")) for item in relationships]
    return {"path": path, "relationships": relationships, "sources": sources, "evidence": [item for item in evidence if item]}
