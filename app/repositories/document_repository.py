from __future__ import annotations

from collections.abc import Iterable


class DocumentRepository:
    """A thin placeholder repository for document metadata and chunk records."""

    def __init__(self):
        self.documents: dict[str, dict] = {}
        self.chunks: dict[str, dict] = {}

    def add_document(self, document: dict) -> dict:
        self.documents[document["id"]] = document
        return document

    def add_chunk(self, chunk: dict) -> dict:
        self.chunks[chunk["id"]] = chunk
        return chunk

    def list_documents(self) -> Iterable[dict]:
        return list(self.documents.values())

    def get_document(self, document_id: str) -> dict | None:
        return self.documents.get(document_id)

    def get_chunk(self, chunk_id: str) -> dict | None:
        return self.chunks.get(chunk_id)
