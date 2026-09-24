from __future__ import annotations

import io
import logging
from typing import Any

from docx import Document as DocxDocument
from pypdf import PdfReader

from app.core.config import settings
from app.core.exceptions import IngestionError
from app.llm.providers import get_embedding_provider
from app.repositories.graph_repository import GraphRepository
from app.repositories.postgres_repository import PostgresDocumentRepository
from app.repositories.vector_repository import VectorRepository
from app.services.extraction_service import StructuredExtractionService

logger = logging.getLogger(__name__)


class IngestionService:
    """Service responsible for validating documents and preparing them for indexing."""

    def validate_file(self, filename: str, content_type: str | None = None, size_bytes: int | None = None) -> dict[str, Any]:
        extension = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
        allowed = {item.strip().lower() for item in settings.allowed_file_types.split(",") if item.strip()}
        clean_ext = f".{extension}"

        if clean_ext not in allowed:
            raise IngestionError(f"Unsupported file type: {filename}")

        if size_bytes is None:
            size_bytes = 0

        max_bytes = settings.max_file_size_mb * 1024 * 1024
        if size_bytes > max_bytes:
            raise IngestionError(f"File exceeds maximum size of {settings.max_file_size_mb}MB")

        return {
            "filename": filename,
            "extension": clean_ext,
            "content_type": content_type,
            "size_bytes": size_bytes,
        }

    def extract_text(self, file_bytes: bytes, extension: str) -> str:
        if extension in {".txt", ".md", ".html"}:
            return file_bytes.decode("utf-8", errors="replace")
        if extension == ".pdf":
            reader = PdfReader(io.BytesIO(file_bytes))
            pages: list[str] = []
            for page in reader.pages:
                text = page.extract_text() or ""
                pages.append(text)
            return "\n\n".join(pages)
        if extension == ".docx":
            document = DocxDocument(io.BytesIO(file_bytes))
            paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
            return "\n\n".join(paragraphs)
        raise IngestionError(f"No extraction routine implemented for {extension}")

    def chunk_text(self, text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
        chunk_size = chunk_size or settings.default_chunk_size
        overlap = overlap or settings.default_chunk_overlap
        if overlap >= chunk_size:
            raise IngestionError("Chunk overlap must be smaller than the chunk size.")

        normalized = text.strip()
        if not normalized:
            return []

        paragraphs = [part.strip() for part in normalized.splitlines() if part.strip()]
        if not paragraphs:
            return [normalized]

        chunks: list[str] = []
        current: list[str] = []
        current_length = 0

        for paragraph in paragraphs:
            paragraph_length = len(paragraph)
            if current and current_length + paragraph_length > chunk_size:
                chunks.append("\n\n".join(current))
                current = [paragraph]
                current_length = paragraph_length
            else:
                current.append(paragraph)
                current_length += paragraph_length

        if current:
            chunks.append("\n\n".join(current))

        return chunks

    def extract_entities_and_relationships(self, document_id: str, text: str, chunks: list[str]) -> list[dict[str, Any]]:
        graph_repository = GraphRepository.get_default()
        postgres_repository = PostgresDocumentRepository.get_default()
        extraction_service = StructuredExtractionService()
        entities: dict[str, dict[str, Any]] = {}
        relations: list[dict[str, Any]] = []

        for chunk_index, chunk in enumerate(chunks):
            chunk_id = f"{document_id}-chunk-{chunk_index}"
            postgres_repository.create_chunk(
                chunk_id=chunk_id,
                document_id=document_id,
                content=chunk,
                chunk_index=chunk_index,
                metadata={"source": document_id, "page": 1},
            )

            extraction = extraction_service.extract(document_id, chunk_id, chunk)
            for extracted_entity in extraction.entities:
                name = extracted_entity.canonical_name
                key = graph_repository.normalize_entity_name(name)
                entity = entities.setdefault(
                    key,
                    {
                        "id": f"entity:{key}",
                        "canonical_name": name,
                        "type": extracted_entity.type.value,
                        "confidence": extracted_entity.confidence,
                        "metadata": {"source_document_id": document_id, "source_chunk_id": chunk_id},
                        "mentions": [name],
                    },
                )
                entity["mentions"] = list(dict.fromkeys(entity["mentions"] + [name]))
                graph_repository.upsert_entity(entity)

            for extracted_relationship in extraction.relationships:
                left = extracted_relationship.source
                right = extracted_relationship.target
                left_id = entities.get(graph_repository.normalize_entity_name(left), {}).get("id")
                right_id = entities.get(graph_repository.normalize_entity_name(right), {}).get("id")
                if not left_id or not right_id:
                    logger.warning("Skipping relationship with unknown entity: %s -> %s", left, right)
                    continue
                rel = {
                    "source": left_id,
                    "target": right_id,
                    "type": extracted_relationship.type.value,
                    "confidence": extracted_relationship.confidence,
                    "source_document_id": extracted_relationship.source_document_id,
                    "source_chunk_id": extracted_relationship.source_chunk_id,
                    "metadata": {"source": document_id},
                }
                graph_repository.add_relationship(
                    source_entity_id=left_id,
                    target_entity_id=right_id,
                    relationship_type=extracted_relationship.type.value,
                    confidence=extracted_relationship.confidence,
                    source_document_id=extracted_relationship.source_document_id,
                    source_chunk_id=extracted_relationship.source_chunk_id,
                    metadata={"source": document_id},
                )
                relations.append(rel)

        return relations

    def process_upload(self, filename: str, file_bytes: bytes, content_type: str | None = None) -> dict[str, Any]:
        validated = self.validate_file(filename, content_type=content_type, size_bytes=len(file_bytes))
        text = self.extract_text(file_bytes, validated["extension"])
        chunks = self.chunk_text(text)

        vector_repository = VectorRepository.get_default()
        embedding_provider = get_embedding_provider()
        postgres_repository = PostgresDocumentRepository.get_default()
        postgres_repository.create_document(
            document_id=filename,
            title=filename,
            content=text,
            file_type=validated["extension"],
            source=filename,
            metadata={"content_type": content_type or "unknown"},
        )

        for index, chunk in enumerate(chunks):
            embedding = embedding_provider.embed(chunk)
            vector_repository.add_chunk_embedding(
                document_id=filename,
                chunk_id=f"{filename}-chunk-{index}",
                content=chunk,
                embedding=embedding,
                collection=embedding_provider.collection_name,
                metadata={
                    "document_id": filename,
                    "chunk_index": index,
                    "source": filename,
                    "page": 1,
                },
            )

        self.extract_entities_and_relationships(filename, text, chunks)

        return {
            "filename": validated["filename"],
            "extension": validated["extension"],
            "content_type": validated["content_type"],
            "size_bytes": validated["size_bytes"],
            "text_length": len(text),
            "chunks": len(chunks),
            "chunks_preview": chunks[:3],
            "status": "COMPLETED",
        }
