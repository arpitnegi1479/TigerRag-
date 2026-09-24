from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from typing import Any

import httpx

from app.core.config import settings


class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError


class NullLLMProvider(BaseLLMProvider):
    """Safe default provider for local scaffold/testing."""

    def generate(self, prompt: str, **kwargs) -> str:
        return "This is a placeholder model response until a real provider is wired in."


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    @property
    def collection_name(self) -> str:
        return "document_chunks"


class NullEmbeddingProvider(EmbeddingProvider):
    def embed(self, text: str) -> list[float]:
        return [0.0] * 1536


class DeterministicEmbeddingProvider(EmbeddingProvider):
    """Local deterministic fallback that keeps the system usable without external model downloads."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    @property
    def collection_name(self) -> str:
        return "document_chunks_deterministic_384"

    def embed(self, text: str) -> list[float]:
        seed = hashlib.sha256(text.encode("utf-8")).hexdigest()
        values: list[float] = []
        for idx in range(self.dimension):
            token = f"{seed}:{idx}"
            raw = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
            values.append(math.sin(raw / 1_000_000) if raw else 0.0)
        return values


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Gemini embedContent provider with an explicit 768-dimensional output."""

    def __init__(self, api_key: str | None = None, model_name: str | None = None, dimension: int | None = None):
        self.api_key = api_key or settings.gemini_api_key
        self.model_name = model_name or settings.embedding_model
        self.dimension = dimension or settings.embedding_dimension

    @property
    def collection_name(self) -> str:
        return f"document_chunks_{self.model_name.replace('-', '_')}"

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def embed(self, text: str) -> list[float]:
        if not self.api_key:
            raise RuntimeError("Gemini embedding API key is not configured")
        response = httpx.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:embedContent",
            params={"key": self.api_key},
            json={
                "content": {"parts": [{"text": text}]},
                "outputDimensionality": self.dimension,
            },
            timeout=60,
        )
        response.raise_for_status()
        values = response.json()["embedding"]["values"]
        if len(values) != self.dimension:
            raise ValueError(f"Gemini returned {len(values)} dimensions, expected {self.dimension}")
        return [float(value) for value in values]


class ResilientEmbeddingProvider(EmbeddingProvider):
    def __init__(self, primary: EmbeddingProvider, fallback: EmbeddingProvider):
        self.primary = primary
        self.fallback = fallback
        self.used_fallback = False

    @property
    def collection_name(self) -> str:
        return self.fallback.collection_name if self.used_fallback else self.primary.collection_name

    def embed(self, text: str) -> list[float]:
        try:
            self.used_fallback = False
            return self.primary.embed(text)
        except Exception:
            self.used_fallback = True
            return self.fallback.embed(text)


def get_embedding_provider() -> EmbeddingProvider:
    fallback = DeterministicEmbeddingProvider()
    if settings.embedding_provider.lower() == "deterministic":
        return fallback
    if settings.gemini_api_key:
        return ResilientEmbeddingProvider(GeminiEmbeddingProvider(), fallback)
    return fallback


try:
    from sentence_transformers import SentenceTransformer

    class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
        """Real embedding provider using a local sentence-transformers model when available."""

        def __init__(self, model_name: str | None = None, dimension: int = 384):
            self.model_name = model_name or settings.embedding_model
            self.dimension = dimension
            self._model = None

        def _get_model(self):
            if self._model is None:
                self._model = SentenceTransformer(self.model_name)
            return self._model

        def embed(self, text: str) -> list[float]:
            model = self._get_model()
            embedding = model.encode(text, convert_to_numpy=True)
            return [float(value) for value in embedding]

except Exception:  # pragma: no cover - fallback when model dependencies are absent
    class SentenceTransformerEmbeddingProvider(DeterministicEmbeddingProvider):
        pass
