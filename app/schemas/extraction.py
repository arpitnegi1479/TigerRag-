from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class EntityType(str, Enum):
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    PRODUCT = "PRODUCT"
    EVENT = "EVENT"
    TECHNOLOGY = "TECHNOLOGY"
    CONCEPT = "CONCEPT"


class RelationshipType(str, Enum):
    RELATED_TO = "RELATED_TO"
    WORKS_FOR = "WORKS_FOR"
    PART_OF = "PART_OF"
    LOCATED_IN = "LOCATED_IN"
    USES = "USES"
    CREATED = "CREATED"
    ACQUIRED = "ACQUIRED"
    CAUSED_BY = "CAUSED_BY"


class ExtractedEntity(BaseModel):
    canonical_name: str = Field(min_length=1)
    type: EntityType
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("canonical_name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        return " ".join(value.split()).strip()


class ExtractedRelationship(BaseModel):
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    type: RelationshipType
    confidence: float = Field(ge=0.0, le=1.0)
    source_document_id: str
    source_chunk_id: str


class StructuredExtraction(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)