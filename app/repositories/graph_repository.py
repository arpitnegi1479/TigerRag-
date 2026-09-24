from __future__ import annotations

import json
import re
from typing import Any

from neo4j import GraphDatabase

from app.core.config import settings


class GraphRepository:
    """Neo4j-backed graph repository with provenance-aware edges and canonical entity normalization."""

    _default_instance: "GraphRepository | None" = None

    def __init__(self, uri: str | None = None, user: str | None = None, password: str | None = None, backend: str | None = None):
        self.uri = uri or settings.neo4j_uri
        self.user = user or settings.neo4j_user
        self.password = password or settings.neo4j_password
        self.backend = (backend or settings.graph_backend).lower()
        self._driver = None
        self._memory_nodes: dict[str, dict[str, Any]] = {}
        self._memory_relationships: list[dict[str, Any]] = []
        self._connect()

    @classmethod
    def get_default(cls) -> "GraphRepository":
        if cls._default_instance is None:
            cls._default_instance = cls()
        return cls._default_instance

    def _connect(self) -> None:
        if self.backend != "neo4j":
            return
        try:
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            with self._driver.session() as session:
                session.run("RETURN 1")
        except Exception:
            self._driver = None

    @property
    def is_live(self) -> bool:
        return self.backend == "neo4j" and self._driver is not None

    def normalize_entity_name(self, name: str) -> str:
        normalized = re.sub(r"[^\w\s]", " ", name, flags=re.UNICODE)
        normalized = re.sub(r"\s+", " ", normalized).strip().lower()
        normalized = re.sub(r"\s+(inc|incorporated|corp|corporation|ltd|limited|llc|co)$", "", normalized)
        return normalized

    def upsert_entity(self, entity: dict[str, Any]) -> dict[str, Any]:
        canonical_name = entity.get("canonical_name") or entity.get("name")
        key = self.normalize_entity_name(canonical_name)
        record = {
            "id": entity.get("id") or f"entity:{key}",
            "canonical_name": canonical_name,
            "type": entity.get("type", "UNKNOWN"),
            "description": entity.get("description"),
            "confidence": float(entity.get("confidence", 0.0)),
            "metadata": entity.get("metadata") or {},
            "mentions": entity.get("mentions", [canonical_name]),
        }
        self._memory_nodes[record["id"]] = record
        if self._driver is not None:
            with self._driver.session() as session:
                session.run(
                    """
                    MERGE (e:Entity {id: $id})
                    SET e.canonical_name = $canonical_name,
                        e.type = $type,
                        e.description = $description,
                        e.confidence = $confidence,
                        e.metadata = $metadata,
                        e.mentions = $mentions
                    """,
                    {
                        "id": record["id"],
                        "canonical_name": record["canonical_name"],
                        "type": record["type"],
                        "description": record.get("description"),
                        "confidence": record["confidence"],
                        "metadata": json.dumps(record["metadata"], sort_keys=True),
                        "mentions": record["mentions"],
                    },
                )
        return record

    def add_relationship(
        self,
        source_entity_id: str,
        target_entity_id: str,
        relationship_type: str,
        confidence: float = 0.9,
        source_document_id: str | None = None,
        source_chunk_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        rel = {
            "source": source_entity_id,
            "target": target_entity_id,
            "type": relationship_type,
            "confidence": confidence,
            "source_document_id": source_document_id,
            "source_chunk_id": source_chunk_id,
            "metadata": metadata or {},
        }
        identity_key = f"{source_document_id or ''}:{source_chunk_id or ''}"
        rel["identity_key"] = identity_key
        existing = next(
            (
                item for item in self._memory_relationships
                if item.get("source") == source_entity_id
                and item.get("target") == target_entity_id
                and item.get("identity_key") == identity_key
            ),
            None,
        )
        if existing is None:
            self._memory_relationships.append(rel)
        else:
            existing.update(rel)
            rel = existing
        if self._driver is not None:
            with self._driver.session() as session:
                session.run(
                    """
                    MATCH (a:Entity {id: $source_entity_id}), (b:Entity {id: $target_entity_id})
                    MERGE (a)-[r:RELATES {identity_key: $identity_key}]->(b)
                    SET r.confidence = $confidence,
                        r.type = $relationship_type,
                        r.source_document_id = $source_document_id,
                        r.source_chunk_id = $source_chunk_id,
                        r.metadata = $metadata
                    """,
                    {
                        "source_entity_id": source_entity_id,
                        "target_entity_id": target_entity_id,
                        "relationship_type": relationship_type,
                        "identity_key": identity_key,
                        "confidence": confidence,
                        "source_document_id": source_document_id,
                        "source_chunk_id": source_chunk_id,
                        "metadata": json.dumps(metadata or {}, sort_keys=True),
                    },
                )
        return rel

    def find_relationships(self, entity_a: str, entity_b: str) -> list[dict[str, Any]]:
        first = self.get_entity_by_name(entity_a) or self._entity_by_id(entity_a)
        second = self.get_entity_by_name(entity_b) or self._entity_by_id(entity_b)
        if first is None or second is None:
            return []
        first_id = first["id"]
        second_id = second["id"]
        if self.is_live:
            with self._driver.session() as session:
                return [
                    {
                        "source": row["source"],
                        "target": row["target"],
                        "type": row["relationship"]["type"],
                        "confidence": row["relationship"].get("confidence", 0.0),
                        "source_document_id": row["relationship"].get("source_document_id"),
                        "source_chunk_id": row["relationship"].get("source_chunk_id"),
                        "identity_key": row["relationship"].get("identity_key"),
                    }
                    for row in session.run(
                        """
                        MATCH (a:Entity)-[r:RELATES]->(b:Entity)
                        WHERE (a.id = $first_id AND b.id = $second_id)
                           OR (a.id = $second_id AND b.id = $first_id)
                        RETURN a.id AS source, b.id AS target, r AS relationship
                        ORDER BY relationship.identity_key
                        """,
                        first_id=first_id,
                        second_id=second_id,
                    )
                ]
        return [
            item for item in self._memory_relationships
            if {item.get("source"), item.get("target")} == {first_id, second_id}
        ]

    def _entity_by_id(self, entity_id: str) -> dict[str, Any] | None:
        if self.is_live:
            with self._driver.session() as session:
                record = session.run("MATCH (e:Entity {id: $entity_id}) RETURN e", entity_id=entity_id).single()
                return dict(record["e"]) if record else None
        return self._memory_nodes.get(entity_id)

    def get_related_nodes(self, node_id: str, depth: int = 1) -> list[dict[str, Any]]:
        return [rel for rel in self._memory_relationships if rel.get("source") == node_id or rel.get("target") == node_id][:depth]

    def find_path(self, start_entity: str, end_entity: str, max_depth: int = 3) -> list[str] | None:
        if start_entity == end_entity:
            return [start_entity]

        if self.is_live:
            bounded_depth = max(1, min(int(max_depth), 20))
            with self._driver.session() as session:
                record = session.run(
                    f"""
                    MATCH p = (start:Entity)-[*1..{bounded_depth}]-(end:Entity)
                    WHERE start.id = $start_id AND end.id = $end_id
                    RETURN [node IN nodes(p) | node.id] AS path
                    ORDER BY length(p)
                    LIMIT 1
                    """,
                    start_id=start_entity,
                    end_id=end_entity,
                ).single()
                return list(record["path"]) if record else None

        adjacency: dict[str, list[str]] = {}
        for rel in self._memory_relationships:
            source = rel.get("source")
            target = rel.get("target")
            if source:
                adjacency.setdefault(source, []).append(target)
            if target:
                adjacency.setdefault(target, []).append(source)

        queue = [(start_entity, [start_entity])]
        seen: set[str] = {start_entity}
        while queue:
            current, path = queue.pop(0)
            if len(path) - 1 >= max_depth:
                continue
            for neighbor in adjacency.get(current, []):
                if neighbor in seen:
                    continue
                if neighbor == end_entity:
                    return path + [neighbor]
                seen.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
        return None

    def get_entity_by_name(self, name: str) -> dict[str, Any] | None:
        normalized = self.normalize_entity_name(name)
        if self.is_live:
            with self._driver.session() as session:
                record = session.run(
                    "MATCH (e:Entity) RETURN e ORDER BY e.id",
                )
                for row in record:
                    entity = dict(row["e"])
                    if self.normalize_entity_name(entity.get("canonical_name", "")) == normalized:
                        return entity
            return None
        for entity in self._memory_nodes.values():
            if self.normalize_entity_name(entity.get("canonical_name", "")) == normalized:
                return entity
        return None

    def get_entity_by_id(self, entity_id: str) -> dict[str, Any] | None:
        return self._entity_by_id(entity_id)

    def get_entity_relationships(self, entity_id: str) -> list[dict[str, Any]]:
        if self.is_live:
            with self._driver.session() as session:
                return [
                    {
                        "source": row["source"],
                        "target": row["target"],
                        "type": row["relationship"]["type"],
                        "confidence": row["relationship"].get("confidence", 0.0),
                        "source_document_id": row["relationship"].get("source_document_id"),
                        "source_chunk_id": row["relationship"].get("source_chunk_id"),
                    }
                    for row in session.run(
                        "MATCH (a:Entity)-[r:RELATES]->(b:Entity) WHERE a.id = $entity_id OR b.id = $entity_id RETURN a.id AS source, b.id AS target, r AS relationship ORDER BY source, target",
                        entity_id=entity_id,
                    )
                ]
        return [item for item in self._memory_relationships if item.get("source") == entity_id or item.get("target") == entity_id]

    def clear(self) -> None:
        self._memory_nodes.clear()
        self._memory_relationships.clear()
        if self.is_live:
            with self._driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
            return
        if self._driver is not None:
            try:
                self._driver.close()
            except Exception:
                pass
            self._driver = None
        self._connect()

    def list_entities(self) -> list[dict[str, Any]]:
        if self.is_live:
            with self._driver.session() as session:
                return [dict(row["e"]) for row in session.run("MATCH (e:Entity) RETURN e ORDER BY e.id")]
        return list(self._memory_nodes.values())

    def list_relationships(self) -> list[dict[str, Any]]:
        if self.is_live:
            with self._driver.session() as session:
                return [
                    {
                        "source": row["source"],
                        "target": row["target"],
                        "type": row["relationship"]["type"],
                        "confidence": row["relationship"].get("confidence", 0.0),
                        "source_document_id": row["relationship"].get("source_document_id"),
                        "source_chunk_id": row["relationship"].get("source_chunk_id"),
                        "identity_key": row["relationship"].get("identity_key"),
                    }
                    for row in session.run(
                        """
                        MATCH (a:Entity)-[r:RELATES]->(b:Entity)
                        RETURN a.id AS source, b.id AS target, r AS relationship
                        ORDER BY source, target
                        """
                    )
                ]
        return list(self._memory_relationships)
