from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import settings
from app.repositories.graph_repository import GraphRepository
from app.repositories.postgres_repository import PostgresDocumentRepository
from app.repositories.vector_repository import VectorRepository
from app.services.ingestion_service import IngestionService

CORPUS_DIRECTORY = ROOT / "data" / "benchmark_corpus"
QUESTIONS_PATH = ROOT / "data" / "benchmark_questions.json"
CORPUS_DOCUMENTS = (
    "01_program_brief.txt",
    "02_meridian_atlas.txt",
    "03_zephyr_nereid.txt",
    "04_celeste_analysis.txt",
    "05_license_record.txt",
    "06_counter_record.txt",
)
BENCHMARK_VERSION = "benchmark-v1.0"


@dataclass(frozen=True)
class GoldEdge:
    source: str
    relationship_type: str
    target: str
    document_id: str
    evidence: str
    question_ids: tuple[str, ...]


def edge(
    source: str,
    relationship_type: str,
    target: str,
    document_id: str,
    evidence: str,
    *question_ids: str,
) -> GoldEdge:
    return GoldEdge(source, relationship_type, target, document_id, evidence, question_ids)


GOLD_EDGES = (
    edge("Helios Research Institute", "LEADS", "Aster Program", "01_program_brief.txt", "Helios Research Institute leads the Aster Program", "f01", "e06", "m04", "m07", "c05"),
    edge("Aster Program", "STUDIES", "Nereid Basin", "01_program_brief.txt", "The Aster Program studies the Nereid Basin", "f02"),
    edge("Aster Program", "USES", "Atlas Sensor", "01_program_brief.txt", "using the Atlas Sensor", "m04"),
    edge("Meridian Labs", "SUPPLIES", "Atlas Sensor", "01_program_brief.txt", "Meridian Labs supplies the Atlas Sensor", "e01", "r01", "m02", "m04", "c05"),
    edge("Aster Program", "APPROVED_IN", "2022 program approval", "01_program_brief.txt", "In 2022, Helios Research Institute approved the Aster Program", "d01", "t01"),
    edge("Aster Program", "EXPANDED_OBSERVATIONS_IN", "2024 observation expansion", "01_program_brief.txt", "In 2024, the Aster Program expanded its Nereid Basin observations", "t02", "c04"),
    edge("Meridian Labs", "DEVELOPED", "Atlas Sensor", "02_meridian_atlas.txt", "Meridian Labs developed the Atlas Sensor", "f03", "e02", "r01", "m04", "c01", "c04", "c05"),
    edge("Atlas Sensor", "SENDS_MEASUREMENTS_TO", "Zephyr Station", "02_meridian_atlas.txt", "The Atlas Sensor sends measurements to Zephyr Station", "r02", "m01", "m02", "m03", "m06", "m08", "c01", "c04", "c05", "cause02"),
    edge("Meridian Labs", "DELIVERED_THIRD_SENSOR_TO", "Zephyr Station", "02_meridian_atlas.txt", "In March 2024, Meridian Labs delivered the third Atlas Sensor to Zephyr Station", "d02", "t03", "c04"),
    edge("Third Atlas Sensor", "REPLACED_2023_PROTOTYPE", "Prototype Sensor", "02_meridian_atlas.txt", "The delivery replaced the prototype sensor used in 2023", "t04", "cause03"),
    edge("Zephyr Station", "MONITORS", "Nereid Basin", "03_zephyr_nereid.txt", "Zephyr Station monitors the Nereid Basin", "f04", "d03", "e03", "r03", "m01", "m02", "m03", "m06", "m08", "c02", "c05", "cause02"),
    edge("Nereid Basin", "CONTAINS", "Lumen Shelf", "03_zephyr_nereid.txt", "The Nereid Basin contains the Lumen Shelf", "r04", "m01", "m02", "m03", "m06", "m08"),
    edge("Zephyr Station", "RECORDS_METHANE_AND_WATER_VAPOR_AT", "Lumen Shelf", "03_zephyr_nereid.txt", "Zephyr Station recorded methane and water-vapor signatures at the Lumen Shelf", "f05", "t05", "c02"),
    edge("Helios Research Institute", "MAPS", "Lumen Shelf", "03_zephyr_nereid.txt", "Atlas Sensor observations from Zephyr Station helped Helios Research Institute map the Lumen Shelf", "m05", "m08", "c02"),
    edge("Celeste Analytics", "IDENTIFIES_MOST_METHANE_RICH", "Lumen Shelf", "04_celeste_analysis.txt", "Celeste Analytics concluded that the Lumen Shelf is the most methane-rich area", "f05", "d04", "e04", "r05", "m05", "c02", "cause01"),
    edge("Atlas Sensor", "ENABLES_ZEPHYR_PROCESSING", "Zephyr Station", "04_celeste_analysis.txt", "Atlas Sensor measurements enabled Zephyr Station processing", "m01", "m02", "m03", "m08", "cause02"),
    edge("Zephyr Station", "ENABLES_LUMEN_MAP", "Lumen Shelf", "04_celeste_analysis.txt", "Zephyr Station processing enabled the Lumen Shelf map", "m01", "m02", "m03", "m05", "m08", "cause02"),
    edge("Lumen Shelf map", "CAUSES_SURVEY_PRIORITY", "Aster Program survey", "04_celeste_analysis.txt", "the map caused the next survey to prioritize the Lumen Shelf", "m03", "m05", "m08", "cause01"),
    edge("Celeste Analytics", "CONCLUSION_USED_BY", "Helios Research Institute", "04_celeste_analysis.txt", "Helios Research Institute used the conclusion", "m07"),
    edge("Meridian Labs", "LICENSES_ATLAS_DESIGN_TO", "Helios Research Institute", "05_license_record.txt", "Meridian Labs licensed the Atlas Sensor design to Helios Research Institute", "r06", "d05", "c03", "conf01", "conf02", "conf03"),
    edge("Meridian Labs", "LICENSE_COVERS", "three field units", "05_license_record.txt", "The license covered three field units", "f06"),
    edge("Meridian Labs", "NOT_LICENSED_TO", "Helios Research Institute", "06_counter_record.txt", "Meridian Labs did not license the Atlas Sensor design to Helios Research Institute", "r08", "c03", "conf01", "conf03"),
    edge("Meridian Labs", "LICENSES_ATLAS_DESIGN_TO", "Nova Dynamics", "06_counter_record.txt", "Meridian Labs licensed the design to Nova Dynamics", "r07", "e05", "d06", "c03", "conf02", "conf03"),
)

ENTITY_TYPES = {
    "Helios Research Institute": "ORGANIZATION",
    "Meridian Labs": "ORGANIZATION",
    "Celeste Analytics": "ORGANIZATION",
    "Nova Dynamics": "ORGANIZATION",
    "Aster Program": "EVENT",
    "Aster Program survey": "EVENT",
    "Atlas Sensor": "PRODUCT",
    "Third Atlas Sensor": "PRODUCT",
    "Prototype Sensor": "PRODUCT",
    "Zephyr Station": "LOCATION",
    "Nereid Basin": "LOCATION",
    "Lumen Shelf": "LOCATION",
    "Lumen Shelf map": "CONCEPT",
}

COMPARISON_SOURCES = {
    "c01": {"01_program_brief.txt", "02_meridian_atlas.txt"},
    "c02": {"03_zephyr_nereid.txt", "04_celeste_analysis.txt"},
    "c03": {"05_license_record.txt", "06_counter_record.txt"},
    "c04": {"01_program_brief.txt", "02_meridian_atlas.txt"},
    "c05": {"01_program_brief.txt", "02_meridian_atlas.txt", "03_zephyr_nereid.txt", "04_celeste_analysis.txt"},
}
CONFLICT_SOURCES = {"05_license_record.txt", "06_counter_record.txt"}


def validate_question_catalog(questions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    question_by_id = {question["id"]: question for question in questions}
    corpus_names = set(CORPUS_DOCUMENTS)
    errors = []
    if len(questions) != 50:
        errors.append(f"Expected 50 questions, found {len(questions)}")
    if len({question["category"] for question in questions}) != 9:
        errors.append("Expected exactly 9 question categories")
    for question in questions:
        missing_sources = set(question["gold_sources"]) - corpus_names
        if missing_sources:
            errors.append(f"{question['id']} references unknown corpus files: {sorted(missing_sources)}")
        if question["category"] == "conflict" and not CONFLICT_SOURCES.issubset(question["gold_sources"]):
            errors.append(f"{question['id']} must cite both contradictory license records")
    for question_id, required_sources in COMPARISON_SOURCES.items():
        question = question_by_id.get(question_id)
        if question is None or not required_sources.issubset(question["gold_sources"]):
            errors.append(f"{question_id} is missing a compared document: {sorted(required_sources)}")
    covered_questions = {question_id for gold_edge in GOLD_EDGES for question_id in gold_edge.question_ids}
    missing_dependencies = set(question_by_id) - covered_questions
    unknown_dependencies = covered_questions - set(question_by_id)
    if missing_dependencies:
        errors.append(f"Questions lack an explicit graph dependency mapping: {sorted(missing_dependencies)}")
    if unknown_dependencies:
        errors.append(f"Graph dependencies reference unknown questions: {sorted(unknown_dependencies)}")
    if errors:
        raise ValueError("Question catalog validation failed:\n- " + "\n- ".join(errors))
    return question_by_id


def resolve_source_chunk(repository: PostgresDocumentRepository, gold_edge: GoldEdge) -> str:
    for chunk_index in range(100):
        chunk_id = f"{gold_edge.document_id}-chunk-{chunk_index}"
        chunk = repository.get_chunk(chunk_id)
        if chunk is None:
            break
        if gold_edge.evidence.casefold() in chunk["content"].casefold():
            return chunk_id
    raise ValueError(
        f"No persisted chunk contains the gold evidence for {gold_edge.document_id}: {gold_edge.evidence}"
    )


def normalized_signature(graph: GraphRepository, relationship: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        graph.normalize_entity_name(relationship["source"]),
        relationship["type"],
        graph.normalize_entity_name(relationship["target"]),
        relationship["source_document_id"],
        relationship["source_chunk_id"],
    )


def audit_annotations(
    graph: GraphRepository,
    postgres: PostgresDocumentRepository,
    questions: dict[str, dict[str, Any]],
) -> tuple[list[str], int]:
    expected_signatures = {}
    source_question_checks = 0
    errors = []
    for gold_edge in GOLD_EDGES:
        chunk_id = resolve_source_chunk(postgres, gold_edge)
        relationship = {
            "source": gold_edge.source,
            "type": gold_edge.relationship_type,
            "target": gold_edge.target,
            "source_document_id": gold_edge.document_id,
            "source_chunk_id": chunk_id,
        }
        signature = normalized_signature(graph, relationship)
        if signature in expected_signatures:
            errors.append(f"Duplicate gold edge definition: {signature}")
        expected_signatures[signature] = gold_edge
        for question_id in gold_edge.question_ids:
            source_question_checks += 1
            if gold_edge.document_id not in questions[question_id]["gold_sources"]:
                errors.append(
                    f"{question_id} depends on {gold_edge.source} {gold_edge.relationship_type} "
                    f"{gold_edge.target}, but omits {gold_edge.document_id} from gold_sources"
                )

    actual_relationships = graph.list_annotated_relationships()
    actual_by_signature = {}
    for relationship in actual_relationships:
        signature = normalized_signature(graph, relationship)
        actual_by_signature[signature] = relationship
        gold_edge = expected_signatures.get(signature)
        if gold_edge is None:
            errors.append(f"Unexpected annotated edge: {signature}")

    for signature, gold_edge in expected_signatures.items():
        if signature not in actual_by_signature:
            errors.append(f"Missing annotated gold edge: {signature}")

    return errors, source_question_checks


def main() -> int:
    dataset = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    questions = validate_question_catalog(dataset["questions"])
    graph = GraphRepository.get_default()
    vectors = VectorRepository.get_default()
    postgres = PostgresDocumentRepository.get_default()
    if settings.graph_backend.lower() != "neo4j" or not graph.is_live:
        raise RuntimeError("Benchmark seeding requires a live Neo4j backend.")
    if not vectors.is_live:
        raise RuntimeError("Benchmark seeding requires a live Qdrant service.")
    if postgres._conn is None:
        raise RuntimeError("Benchmark seeding requires a live Postgres connection.")

    ingestion = IngestionService()
    vectors.delete_documents(list(CORPUS_DOCUMENTS))
    ingestion_results = []
    for document_id in CORPUS_DOCUMENTS:
        document_path = CORPUS_DIRECTORY / document_id
        if not document_path.is_file():
            raise FileNotFoundError(document_path)
        ingestion_results.append(
            ingestion.process_upload(
                filename=document_id,
                file_bytes=document_path.read_bytes(),
                content_type="text/plain",
            )
        )

    graph.delete_relationships_for_documents(list(CORPUS_DOCUMENTS))
    graph.delete_orphan_entities_for_documents(list(CORPUS_DOCUMENTS))
    entity_names = {
        name
        for gold_edge in GOLD_EDGES
        for name in (gold_edge.source, gold_edge.target)
    }
    for name in sorted(entity_names):
        graph.upsert_entity(
            {
                "id": f"entity:{graph.normalize_entity_name(name)}",
                "canonical_name": name,
                "type": ENTITY_TYPES.get(name, "CONCEPT"),
                "confidence": 1.0,
                "metadata": {"benchmark_annotation": True, "benchmark_version": BENCHMARK_VERSION},
                "mentions": [name],
            }
        )

    for gold_edge in GOLD_EDGES:
        chunk_id = resolve_source_chunk(postgres, gold_edge)
        graph.add_relationship(
            source_entity_id=f"entity:{graph.normalize_entity_name(gold_edge.source)}",
            target_entity_id=f"entity:{graph.normalize_entity_name(gold_edge.target)}",
            relationship_type=gold_edge.relationship_type,
            confidence=1.0,
            source_document_id=gold_edge.document_id,
            source_chunk_id=chunk_id,
            metadata={
                "benchmark_annotation": True,
                "benchmark_version": BENCHMARK_VERSION,
                "evidence": gold_edge.evidence,
            },
        )

    mismatches, source_question_checks = audit_annotations(graph, postgres, questions)
    entities = graph.list_entities()
    relationships = graph.list_relationships()
    annotations = graph.list_annotated_relationships()
    benchmark_endpoint_ids = {
        f"entity:{graph.normalize_entity_name(name)}"
        for gold_edge in GOLD_EDGES
        for name in (gold_edge.source, gold_edge.target)
    }
    benchmark_entity_count = sum(entity.get("id") in benchmark_endpoint_ids for entity in entities)
    summary = {
        "ingested_documents": len(ingestion_results),
        "document_chunk_counts": {result["filename"]: result["chunks"] for result in ingestion_results},
        "graph_entity_count": len(entities),
        "benchmark_entity_count": benchmark_entity_count,
        "graph_edge_count": len(relationships),
        "annotated_edge_count": len(annotations),
        "qdrant_total_point_count": vectors.count_points(),
        "qdrant_benchmark_point_count": vectors.count_documents(list(CORPUS_DOCUMENTS)),
        "postgres_total_document_count": postgres.count_documents(),
        "postgres_benchmark_document_count": postgres.count_documents(list(CORPUS_DOCUMENTS)),
        "audit": {
            "expected_annotated_edges": len(GOLD_EDGES),
            "actual_annotated_edges": len(annotations),
            "edge_to_question_source_checks": source_question_checks,
            "mismatches": mismatches,
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if mismatches else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Benchmark seed failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error