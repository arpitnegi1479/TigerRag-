import httpx

from app.llm.gemini import GeminiProvider


def test_gemini_retries_once_after_rate_limit(monkeypatch):
    responses = [
        httpx.Response(429, request=httpx.Request("POST", "https://example.test")),
        httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": "{\"ok\":true}"}]}}]},
            request=httpx.Request("POST", "https://example.test"),
        ),
    ]
    calls = []

    def fake_post(*args, **kwargs):
        calls.append((args, kwargs))
        return responses.pop(0)

    monkeypatch.setattr(httpx, "post", fake_post)
    provider = GeminiProvider(api_key="test-key", model="gemini-test", retry_delay_seconds=0)

    assert provider.generate_json("prompt", {"type": "OBJECT"}) == '{"ok":true}'
    assert len(calls) == 2