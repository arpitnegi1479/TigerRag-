import pytest

from app.repositories.graph_repository import GraphRepository
from app.schemas.api import EvidenceStatus
from app.services.graphrag_service import GraphRagService
from app.services.ingestion_service import IngestionService


def reset_graph_state() -> GraphRepository:
    GraphRepository._default_instance = None
    repo = GraphRepository()
    GraphRepository._default_instance = repo
    return repo


def test_ingestion_extracts_entities_and_relationships():
    repo = reset_graph_state()
    service = IngestionService()
    text = "Alpha and Beta are related. Beta and Gamma are connected."
    chunks = [
        "Alpha and Beta are related.",
        "Beta and Gamma are connected.",
    ]

    relations = service.extract_entities_and_relationships("doc-graph-test", text, chunks)

    assert len(relations) == 2
    entity_names = {entity["canonical_name"] for entity in repo.list_entities()}
    assert {"Alpha", "Beta", "Gamma"}.issubset(entity_names)
    assert all(rel["source_document_id"] == "doc-graph-test" for rel in relations)
    assert all(rel["source_chunk_id"].startswith("doc-graph-test-chunk-") for rel in relations)


def test_graph_repository_upsert_and_bounded_path_search():
    repo = reset_graph_state()
    repo.upsert_entity({"id": "entity:alpha", "canonical_name": "Alpha", "type": "PERSON"})
    repo.upsert_entity({"id": "entity:beta", "canonical_name": "Beta", "type": "PERSON"})
    repo.upsert_entity({"id": "entity:gamma", "canonical_name": "Gamma", "type": "PERSON"})

    repo.add_relationship("entity:alpha", "entity:beta", "RELATED_TO")
    repo.add_relationship("entity:beta", "entity:gamma", "RELATED_TO")
    repo.add_relationship("entity:gamma", "entity:beta", "RELATED_TO")

    path = repo.find_path("entity:alpha", "entity:gamma", max_depth=2)
    assert path == ["entity:alpha", "entity:beta", "entity:gamma"]
    assert repo.find_path("entity:alpha", "entity:omega", max_depth=2) is None


def test_graphrag_service_found_and_not_found_branches():
    repo = reset_graph_state()
    repo.upsert_entity({"id": "entity:alpha", "canonical_name": "Alpha", "type": "PERSON"})
    repo.upsert_entity({"id": "entity:beta", "canonical_name": "Beta", "type": "PERSON"})
    repo.upsert_entity({"id": "entity:gamma", "canonical_name": "Gamma", "type": "PERSON"})
    repo.add_relationship("entity:alpha", "entity:beta", "RELATED_TO", source_document_id="doc-1", source_chunk_id="doc-1-chunk-0")
    repo.add_relationship("entity:beta", "entity:gamma", "RELATED_TO", source_document_id="doc-1", source_chunk_id="doc-1-chunk-0")

    service = GraphRagService()
    found = service.answer("What is the relationship between Alpha and Gamma?", max_depth=3)
    assert found.evidence_status == EvidenceStatus.VERIFIED
    assert found.graph_paths
    assert found.graph_paths[0]["path"] == ["entity:alpha", "entity:beta", "entity:gamma"]

    missing = service.answer("What is the relationship between Alpha and Omega?", max_depth=3)
    assert missing.evidence_status == EvidenceStatus.NOT_FOUND
    assert missing.citations == []


def test_graph_repository_clear_resets_in_memory_state():
    repo = reset_graph_state()
    repo.upsert_entity({"id": "entity:alpha", "canonical_name": "Alpha", "type": "PERSON"})
    repo.clear()

    assert repo.list_entities() == []
    assert repo._memory_relationships == []
    assert repo.get_entity_by_name("Alpha") is None


def test_graph_repository_clear_refuses_live_graph(monkeypatch):
    repo = GraphRepository(backend="memory")
    repo.upsert_entity({"id": "entity:protected", "canonical_name": "Protected", "type": "PERSON"})
    monkeypatch.setattr(GraphRepository, "is_live", property(lambda self: True))

    with pytest.raises(RuntimeError, match="Refusing to clear a live Neo4j graph"):
        repo.clear()

    assert "entity:protected" in repo._memory_nodes


def test_delete_relationships_for_documents_keeps_other_sources():
    repo = GraphRepository(backend="memory")
    for entity_id, name in (("entity:alpha", "Alpha"), ("entity:beta", "Beta"), ("entity:gamma", "Gamma")):
        repo.upsert_entity({"id": entity_id, "canonical_name": name, "type": "PERSON"})
    repo.add_relationship("entity:alpha", "entity:beta", "RELATED_TO", source_document_id="benchmark.txt")
    repo.add_relationship("entity:beta", "entity:gamma", "RELATED_TO", source_document_id="other.txt")

    deleted_count = repo.delete_relationships_for_documents(["benchmark.txt"])

    assert deleted_count == 1
    assert [item["source_document_id"] for item in repo.list_relationships()] == ["other.txt"]


def test_list_annotated_relationships_returns_only_marked_edges():
    repo = GraphRepository(backend="memory")
    for entity_id, name in (("entity:alpha", "Alpha"), ("entity:beta", "Beta"), ("entity:gamma", "Gamma")):
        repo.upsert_entity({"id": entity_id, "canonical_name": name, "type": "PERSON"})
    repo.add_relationship("entity:alpha", "entity:beta", "RELATED_TO", source_document_id="gold.txt", metadata={"benchmark_annotation": True})
    repo.add_relationship("entity:beta", "entity:gamma", "RELATED_TO", source_document_id="noise.txt")

    annotations = repo.list_annotated_relationships()

    assert len(annotations) == 1
    assert annotations[0]["source_document_id"] == "gold.txt"
    assert annotations[0]["metadata"]["benchmark_annotation"] is True


def test_delete_orphan_entities_only_removes_unconnected_source_entities():
    repo = GraphRepository(backend="memory")
    for entity_id, name, document_id in (
        ("entity:orphan", "Orphan", "benchmark.txt"),
        ("entity:connected", "Connected", "benchmark.txt"),
        ("entity:outside", "Outside", "other.txt"),
    ):
        repo.upsert_entity(
            {
                "id": entity_id,
                "canonical_name": name,
                "type": "PERSON",
                "metadata": {"source_document_id": document_id},
            }
        )
    repo.add_relationship("entity:connected", "entity:outside", "RELATED_TO", source_document_id="other.txt")

    deleted_count = repo.delete_orphan_entities_for_documents(["benchmark.txt"])

    assert deleted_count == 1
    assert "entity:orphan" not in repo._memory_nodes
    assert "entity:connected" in repo._memory_nodes


def test_relationship_upsert_preserves_type_and_deduplicates_endpoint_pair():
    repo = GraphRepository(backend="memory")
    repo.upsert_entity({"id": "entity:orion", "canonical_name": "Orion Analytics", "type": "TECHNOLOGY"})
    repo.upsert_entity({"id": "entity:helios", "canonical_name": "Helios Cloud", "type": "TECHNOLOGY"})

    repo.add_relationship("entity:orion", "entity:helios", "PART_OF", source_document_id="doc-1", source_chunk_id="doc-1-chunk-0")
    repo.add_relationship("entity:orion", "entity:helios", "LOCATED_IN", source_document_id="doc-1", source_chunk_id="doc-1-chunk-0")
    repo.add_relationship("entity:orion", "entity:helios", "RELATED_TO", source_document_id="doc-2", source_chunk_id="doc-2-chunk-0")

    relationships = repo.list_relationships()
    assert len(relationships) == 2
    assert {item["type"] for item in relationships} == {"LOCATED_IN", "RELATED_TO"}
    assert len(repo.find_relationships("Orion Analytics", "Helios Cloud")) == 2
