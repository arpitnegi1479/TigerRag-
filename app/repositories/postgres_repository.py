from __future__ import annotations

from typing import Any
import json

import psycopg
from psycopg.types.json import Jsonb

from app.core.config import settings


class PostgresDocumentRepository:
    """Postgres-backed document metadata store with in-memory fallback for local development/tests."""

    _default_instance: "PostgresDocumentRepository | None" = None

    def __init__(self, dsn: str | None = None):
        self.dsn = (dsn or settings.postgres_dsn).replace("postgresql+psycopg://", "postgresql://")
        self._conn = None
        self.documents: dict[str, dict[str, Any]] = {}
        self.chunks: dict[str, dict[str, Any]] = {}
        self.agent_traces: dict[str, dict[str, Any]] = {}
        self._connect()

    @classmethod
    def get_default(cls) -> "PostgresDocumentRepository":
        if cls._default_instance is None:
            cls._default_instance = cls()
        return cls._default_instance

    def _connect(self) -> None:
        if not self.dsn:
            self._conn = None
            return
        try:
            self._conn = psycopg.connect(self.dsn, connect_timeout=3)
        except Exception:
            self._conn = None

    def create_document(self, document_id: str, title: str, content: str, file_type: str, source: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {
            "id": document_id,
            "title": title,
            "content": content,
            "file_type": file_type,
            "source": source,
            "metadata": metadata or {},
        }
        self.documents[document_id] = payload
        if self._conn is not None:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        id TEXT PRIMARY KEY,
                        title TEXT,
                        content TEXT,
                        file_type TEXT,
                        source TEXT,
                        metadata JSONB
                    )
                    """
                )
                cur.execute(
                    "INSERT INTO documents (id, title, content, file_type, source, metadata) VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO UPDATE SET title = EXCLUDED.title, content = EXCLUDED.content, file_type = EXCLUDED.file_type, source = EXCLUDED.source, metadata = EXCLUDED.metadata",
                    (document_id, title, content, file_type, source, Jsonb(payload["metadata"])),
                )
            self._conn.commit()
        return payload

    def create_chunk(self, chunk_id: str, document_id: str, content: str, chunk_index: int, page: int | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {
            "id": chunk_id,
            "document_id": document_id,
            "content": content,
            "chunk_index": chunk_index,
            "page": page,
            "metadata": metadata or {},
        }
        self.chunks[chunk_id] = payload
        if self._conn is not None:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chunks (
                        id TEXT PRIMARY KEY,
                        document_id TEXT,
                        content TEXT,
                        chunk_index INTEGER,
                        page INTEGER,
                        metadata JSONB
                    )
                    """
                )
                cur.execute(
                    "INSERT INTO chunks (id, document_id, content, chunk_index, page, metadata) VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO UPDATE SET document_id = EXCLUDED.document_id, content = EXCLUDED.content, chunk_index = EXCLUDED.chunk_index, page = EXCLUDED.page, metadata = EXCLUDED.metadata",
                    (chunk_id, document_id, content, chunk_index, page, Jsonb(payload["metadata"])),
                )
            self._conn.commit()
        return payload

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        return self.documents.get(document_id)

    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        if self._conn is not None:
            with self._conn.cursor() as cur:
                cur.execute("SELECT id, document_id, content, chunk_index, page, metadata FROM chunks WHERE id = %s", (chunk_id,))
                row = cur.fetchone()
            if row:
                return {
                    "id": row[0],
                    "document_id": row[1],
                    "content": row[2],
                    "chunk_index": row[3],
                    "page": row[4],
                    "metadata": row[5] or {},
                }
        return self.chunks.get(chunk_id)

    def list_documents(self) -> list[dict[str, Any]]:
        if self._conn is not None:
            with self._conn.cursor() as cur:
                cur.execute("SELECT id, title, content, file_type, source, metadata FROM documents ORDER BY id")
                return [
                    {"id": row[0], "title": row[1], "content": row[2], "file_type": row[3], "source": row[4], "metadata": row[5] or {}, "status": "COMPLETED"}
                    for row in cur.fetchall()
                ]
        return [{**document, "status": "COMPLETED"} for document in self.documents.values()]

    def save_agent_trace(self, query_id: str, query: str, result: dict[str, Any]) -> None:
        payload = {"query_id": query_id, "query": query, "result": result, "trace": result.get("agent_steps", [])}
        self.agent_traces[query_id] = payload
        if self._conn is not None:
            with self._conn.cursor() as cur:
                cur.execute("CREATE TABLE IF NOT EXISTS agent_traces (query_id TEXT PRIMARY KEY, query TEXT, result JSONB, trace JSONB)")
                cur.execute(
                    "INSERT INTO agent_traces (query_id, query, result, trace) VALUES (%s, %s, %s, %s) ON CONFLICT (query_id) DO UPDATE SET query=EXCLUDED.query, result=EXCLUDED.result, trace=EXCLUDED.trace",
                    (query_id, query, Jsonb(result), Jsonb(result.get("agent_steps", []))),
                )
            self._conn.commit()

    def get_agent_trace(self, query_id: str) -> dict[str, Any] | None:
        if self._conn is not None:
            with self._conn.cursor() as cur:
                cur.execute("SELECT query_id, query, result, trace FROM agent_traces WHERE query_id = %s", (query_id,))
                row = cur.fetchone()
            if row:
                return {"query_id": row[0], "query": row[1], "result": row[2], "trace": row[3]}
        return self.agent_traces.get(query_id)
