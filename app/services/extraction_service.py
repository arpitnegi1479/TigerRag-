from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol

from pydantic import ValidationError

from app.core.config import settings
from app.llm.gemini import GeminiProvider
from app.schemas.extraction import (
    EntityType,
    ExtractedEntity,
    ExtractedRelationship,
    RelationshipType,
    StructuredExtraction,
)

logger = logging.getLogger(__name__)


class ExtractionProvider(Protocol):
    def generate_json(self, prompt: str, schema: dict[str, Any]) -> str:
        ...


class GeminiExtractionProvider(GeminiProvider):
    """Gemini structured JSON provider, loaded only when a Gemini key is configured."""

    def __init__(self, api_key: str, model: str):
        super().__init__(api_key=api_key, model=model)


class StructuredExtractionService:
    """Extract typed graph facts, retry invalid provider output once, then use local fallback."""

    def __init__(self, provider: ExtractionProvider | None = None):
        self.provider = provider

    def _provider_or_none(self) -> ExtractionProvider | None:
        if self.provider is not None:
            return self.provider
        if settings.gemini_api_key:
            return GeminiExtractionProvider(settings.gemini_api_key, settings.gemini_model)
        return None

    def _prompt(self, document_id: str, chunk_id: str, text: str) -> str:
        return f"""Extract only facts explicitly stated in this document chunk.
Use only the allowed entity and relationship enum values in the JSON schema.
Canonicalize aliases to one name where the text makes the alias clear. Do not infer facts.
Every relationship must include the supplied provenance IDs.
document_id={document_id}
chunk_id={chunk_id}

CHUNK:
{text}
"""

    def _provider_schema(self) -> dict[str, Any]:
        return {
            "type": "OBJECT",
            "properties": {
                "entities": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "canonical_name": {"type": "STRING"},
                            "type": {"type": "STRING", "enum": [item.value for item in EntityType]},
                            "confidence": {"type": "NUMBER"},
                        },
                        "required": ["canonical_name", "type", "confidence"],
                    },
                },
                "relationships": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "source": {"type": "STRING"},
                            "target": {"type": "STRING"},
                            "type": {"type": "STRING", "enum": [item.value for item in RelationshipType]},
                            "confidence": {"type": "NUMBER"},
                            "source_document_id": {"type": "STRING"},
                            "source_chunk_id": {"type": "STRING"},
                        },
                        "required": ["source", "target", "type", "confidence", "source_document_id", "source_chunk_id"],
                    },
                },
            },
            "required": ["entities", "relationships"],
        }

    def extract(self, document_id: str, chunk_id: str, text: str) -> StructuredExtraction:
        provider = self._provider_or_none()
        if provider is not None:
            prompt = self._prompt(document_id, chunk_id, text)
            schema = self._provider_schema()
            for attempt in range(2):
                try:
                    raw = provider.generate_json(prompt, schema)
                    parsed = StructuredExtraction.model_validate_json(raw)
                    return self._ensure_provenance(parsed, document_id, chunk_id)
                except (ValidationError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                    logger.warning("Structured graph extraction validation failed (attempt %s/2): %s", attempt + 1, exc)
                except Exception as exc:
                    status_code = getattr(getattr(exc, "response", None), "status_code", None)
                    logger.warning(
                        "Structured graph extraction provider failed (%s, status=%s)",
                        type(exc).__name__,
                        status_code,
                    )
                    break
        return self._heuristic_fallback(document_id, chunk_id, text)

    def _ensure_provenance(self, result: StructuredExtraction, document_id: str, chunk_id: str) -> StructuredExtraction:
        relationships = [rel.model_copy(update={"source_document_id": document_id, "source_chunk_id": chunk_id}) for rel in result.relationships]
        return StructuredExtraction(entities=result.entities, relationships=relationships)

    def _heuristic_fallback(self, document_id: str, chunk_id: str, text: str) -> StructuredExtraction:
        names = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
        unique_names = list(dict.fromkeys(name.strip() for name in names if name.strip()))
        entities = [ExtractedEntity(canonical_name=name, type=EntityType.ORGANIZATION, confidence=0.35) for name in unique_names]
        relationships = [
            ExtractedRelationship(
                source=left,
                target=right,
                type=RelationshipType.RELATED_TO,
                confidence=0.25,
                source_document_id=document_id,
                source_chunk_id=chunk_id,
            )
            for left, right in zip(unique_names, unique_names[1:])
        ]
        return StructuredExtraction(entities=entities, relationships=relationships)