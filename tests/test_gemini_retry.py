import httpx

from app.llm.gemini import GeminiProvider
from app.services.evaluation_telemetry import GeminiCallTrace, capture_gemini_calls, evaluation_step


def test_gemini_retries_once_after_rate_limit(monkeypatch):
    responses = [
        httpx.Response(429, request=httpx.Request("POST", "https://example.test")),
        httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": "{\"ok\":true}"}]}}],
                "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 3, "totalTokenCount": 11},
            },
            request=httpx.Request("POST", "https://example.test"),
        ),
    ]
    calls = []

    def fake_post(*args, **kwargs):
        calls.append((args, kwargs))
        return responses.pop(0)

    monkeypatch.setattr(httpx, "post", fake_post)
    provider = GeminiProvider(api_key="test-key", model="gemini-test", retry_delay_seconds=0)

    trace = GeminiCallTrace(minimum_interval_seconds=0)
    with capture_gemini_calls(trace), evaluation_step("agent_decision"):
        assert provider.generate_json("prompt", {"type": "OBJECT"}) == '{"ok":true}'
    assert len(calls) == 2
    summary = trace.summary()
    assert summary["gemini_http_attempts"] == 2
    assert summary["gemini_logical_requests"] == 1
    assert summary["token_usage"]["total"] == 11
    assert summary["degraded"] is False