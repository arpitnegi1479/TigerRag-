from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class Document(BaseModel):
    id: str
    title: str
    source: str
    file_type: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    status: DocumentStatus = DocumentStatus.PENDING
    created_at: datetime | None = None
    updated_at: datetime | None = None


class Chunk(BaseModel):
    id: str
    document_id: str
    content: str
    page: int | None = None
    section: str | None = None
    chunk_index: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding_ref: str | None = None


class Entity(BaseModel):
    id: str
    canonical_name: str
    type: str
    description: str | None = None
    confidence: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class Relationship(BaseModel):
    id: str
    source_entity_id: str
    target_entity_id: str
    type: str
    confidence: float = 0.0
    source_document_id: str | None = None
    source_chunk_id: str | None = None
