import httpx

from app.schemas.extraction import EntityType, RelationshipType
from app.services.extraction_service import StructuredExtractionService


class ValidProvider:
    def __init__(self):
        self.calls = 0

    def generate_json(self, prompt, schema):
        self.calls += 1
        return '{"entities":[{"canonical_name":"Microsoft","type":"ORGANIZATION","confidence":0.98},{"canonical_name":"Azure","type":"PRODUCT","confidence":0.93}],"relationships":[{"source":"Microsoft","target":"Azure","type":"USES","confidence":0.91,"source_document_id":"wrong","source_chunk_id":"wrong"}]}'


class InvalidThenValidProvider:
    def __init__(self):
        self.calls = 0

    def generate_json(self, prompt, schema):
        self.calls += 1
        if self.calls == 1:
            return '{"entities":[{"canonical_name":"Microsoft","type":"NOT_ALLOWED","confidence":2.0}],"relationships":[]}'
        return '{"entities":[{"canonical_name":"Microsoft","type":"ORGANIZATION","confidence":0.8}],"relationships":[]}'


class AlwaysInvalidProvider:
    def generate_json(self, prompt, schema):
        return '{"entities":[{"canonical_name":"Microsoft","type":"NOT_ALLOWED","confidence":2.0}],"relationships":[]}'


def test_structured_extraction_returns_typed_entities_and_provenance():
    provider = ValidProvider()
    result = StructuredExtractionService(provider).extract("doc-1", "doc-1-chunk-0", "Microsoft uses Azure.")

    assert result.entities[0].type == EntityType.ORGANIZATION
    assert result.entities[1].type == EntityType.PRODUCT
    assert result.relationships[0].type == RelationshipType.USES
    assert result.relationships[0].source_document_id == "doc-1"
    assert result.relationships[0].source_chunk_id == "doc-1-chunk-0"
    assert provider.calls == 1


def test_schema_failure_retries_once_then_accepts_valid_output():
    provider = InvalidThenValidProvider()
    result = StructuredExtractionService(provider).extract("doc-2", "doc-2-chunk-1", "Microsoft is an organization.")

    assert provider.calls == 2
    assert result.entities[0].type == EntityType.ORGANIZATION


def test_schema_failure_falls_back_to_heuristic_extraction():
    result = StructuredExtractionService(AlwaysInvalidProvider()).extract(
        "doc-3", "doc-3-chunk-0", "Microsoft works with Azure."
    )

    assert {entity.canonical_name for entity in result.entities} == {"Microsoft", "Azure"}
    assert result.relationships[0].type == RelationshipType.RELATED_TO
    assert result.relationships[0].confidence < 0.5
    assert result.relationships[0].source_document_id == "doc-3"


def test_provider_failure_log_does_not_expose_request_url_credentials(caplog):
    request = httpx.Request("POST", "https://example.test/generate?key=do-not-log")
    response = httpx.Response(429, request=request)
    failure = httpx.HTTPStatusError("quota exceeded", request=request, response=response)

    class FailingProvider:
        def generate_json(self, prompt, schema):
            raise failure

    with caplog.at_level("WARNING", logger="app.services.extraction_service"):
        result = StructuredExtractionService(FailingProvider()).extract(
            "doc-4", "doc-4-chunk-0", "Microsoft works with Azure."
        )

    assert result.relationships
    assert "HTTPStatusError" in caplog.text
    assert "status=429" in caplog.text
    assert "do-not-log" not in caplog.text
