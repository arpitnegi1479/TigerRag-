from __future__ import annotations

import uuid
from typing import Any

from app.core.config import settings

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
except Exception:  # pragma: no cover - fallback for limited environments
    QdrantClient = None
    models = None


class VectorRepository:
    """Vector storage adapter with a real Qdrant path and an in-memory fallback for local tests."""

    _default_instance: "VectorRepository | None" = None

    def __init__(self, url: str | None = None, api_key: str | None = None):
        self.collections: dict[str, list[dict[str, Any]]] = {}
        self._client = None
        if QdrantClient is not None:
            try:
                self._client = QdrantClient(url=url or settings.qdrant_url, api_key=api_key or settings.qdrant_api_key, timeout=3)
                self._client.get_collections()
            except Exception:
                self._client = None

    @classmethod
    def get_default(cls) -> "VectorRepository":
        if cls._default_instance is None:
            cls._default_instance = cls()
        return cls._default_instance

    def _ensure_collection(self, collection: str, vector_size: int = 384) -> None:
        if self._client is None:
            self.collections.setdefault(collection, [])
            return
        try:
            existing = self._client.get_collection(collection_name=collection)
            existing_size = existing.config.params.vectors.size
            if existing_size != vector_size:
                raise ValueError(
                    f"Qdrant collection '{collection}' has dimension {existing_size}, expected {vector_size}. "
                    "Recreate the collection before ingesting with a different embedding model."
                )
        except Exception:
            if self._client.collection_exists(collection):
                raise
            self._client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            )

    def upsert(self, collection: str, item: dict[str, Any]) -> dict[str, Any]:
        self._ensure_collection(collection, vector_size=len(item.get("embedding", [])) or 384)
        if self._client is not None:
            payload = {k: v for k, v in item.items() if k != "embedding"}
            raw_point_id = item.get("id") or payload.get("chunk_id") or str(len(self.collections.get(collection, [])))
            try:
                point_id: int | str = int(raw_point_id)
            except (TypeError, ValueError):
                point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{collection}:{raw_point_id}"))
            self._client.upsert(
                collection_name=collection,
                points=[
                    {
                        "id": point_id,
                        "vector": item.get("embedding", []),
                        "payload": payload,
                    }
                ],
            )
            return item

        self.collections.setdefault(collection, [])
        self.collections[collection].append(item)
        return item

    def search(self, collection: str, query_vector: list[float], limit: int = 5) -> list[dict[str, Any]]:
        if self._client is not None:
            try:
                response = self._client.query_points(
                    collection_name=collection,
                    query=query_vector,
                    limit=limit,
                    with_payload=True,
                )
                return [
                    {
                        "id": hit.id,
                        "score": hit.score,
                        **(hit.payload or {}),
                    }
                    for hit in response.points
                ]
            except Exception:
                pass

        items = self.collections.get(collection, [])
        if not items:
            return []
        scored = []
        for item in items:
            embedding = item.get("embedding", [])
            if not embedding:
                continue
            similarity = sum(a * b for a, b in zip(query_vector[: len(embedding)], embedding)) / max(1.0, len(embedding))
            scored.append({**item, "score": similarity})
        return sorted(scored, key=lambda x: x["score"], reverse=True)[:limit]

    @property
    def is_live(self) -> bool:
        return self._client is not None

    def delete_documents(self, document_ids: list[str]) -> int:
        scoped_ids = list(dict.fromkeys(document_ids))
        if not scoped_ids:
            return 0
        if self._client is None:
            deleted_count = 0
            for collection, items in self.collections.items():
                retained = [item for item in items if item.get("document_id") not in scoped_ids]
                deleted_count += len(items) - len(retained)
                self.collections[collection] = retained
            return deleted_count

        point_filter = models.Filter(
            must=[models.FieldCondition(key="document_id", match=models.MatchAny(any=scoped_ids))]
        )
        deleted_count = 0
        for collection in self._client.get_collections().collections:
            count = self._client.count(
                collection_name=collection.name,
                count_filter=point_filter,
                exact=True,
            ).count
            if count:
                self._client.delete(
                    collection_name=collection.name,
                    points_selector=models.FilterSelector(filter=point_filter),
                )
                deleted_count += count
        return deleted_count

    def count_documents(self, document_ids: list[str]) -> int:
        scoped_ids = list(dict.fromkeys(document_ids))
        if not scoped_ids:
            return 0
        if self._client is None:
            return sum(
                1
                for items in self.collections.values()
                for item in items
                if item.get("document_id") in scoped_ids
            )

        point_filter = models.Filter(
            must=[models.FieldCondition(key="document_id", match=models.MatchAny(any=scoped_ids))]
        )
        return sum(
            self._client.count(
                collection_name=collection.name,
                count_filter=point_filter,
                exact=True,
            ).count
            for collection in self._client.get_collections().collections
        )

    def count_points(self) -> int:
        if self._client is None:
            return sum(len(items) for items in self.collections.values())
        return sum(
            self._client.count(collection_name=collection.name, exact=True).count
            for collection in self._client.get_collections().collections
        )

    def add_chunk_embedding(self, document_id: str, chunk_id: str, content: str, embedding: list[float], metadata: dict[str, Any] | None = None, collection: str = "document_chunks") -> dict[str, Any]:
        payload = {
            "document_id": document_id,
            "chunk_id": chunk_id,
            "content": content,
            "metadata": metadata or {},
            "score": 1.0,
            "embedding": embedding,
        }
        return self.upsert(collection, payload)
