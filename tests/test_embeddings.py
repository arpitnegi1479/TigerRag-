import httpx

from app.core.config import settings
from app.llm.providers import DeterministicEmbeddingProvider, GeminiEmbeddingProvider, ResilientEmbeddingProvider, get_embedding_provider


def test_gemini_embedding_provider_uses_native_endpoint_and_dimension(monkeypatch):
    response = httpx.Response(
        200,
        json={"embedding": {"values": [0.1, 0.2, 0.3]}},
        request=httpx.Request("POST", "https://example.test"),
    )
    calls = []

    def fake_post(*args, **kwargs):
        calls.append((args, kwargs))
        return response

    monkeypatch.setattr(httpx, "post", fake_post)
    provider = GeminiEmbeddingProvider(api_key="test-key", model_name="gemini-embedding-test", dimension=3)

    assert provider.embed("semantic text") == [0.1, 0.2, 0.3]
    assert calls[0][0][0].endswith("models/gemini-embedding-test:embedContent")
    assert calls[0][1]["json"]["outputDimensionality"] == 3
    assert provider.collection_name == "document_chunks_gemini_embedding_test"


def test_embedding_selection_uses_deterministic_fallback_without_key(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", None)
    monkeypatch.setattr(settings, "embedding_provider", "auto")

    provider = get_embedding_provider()

    assert isinstance(provider, DeterministicEmbeddingProvider)
    assert len(provider.embed("fallback")) == 384
    assert provider.collection_name == "document_chunks_deterministic_384"


def test_resilient_embedding_provider_falls_back_after_provider_failure():
    class BrokenProvider:
        collection_name = "document_chunks_broken"

        def embed(self, text):
            raise RuntimeError("provider unavailable")

    provider = ResilientEmbeddingProvider(BrokenProvider(), DeterministicEmbeddingProvider())

    vector = provider.embed("fallback")

    assert len(vector) == 384
    assert provider.used_fallback is True
    assert provider.collection_name == "document_chunks_deterministic_384"
