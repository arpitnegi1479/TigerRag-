from __future__ import annotations

import time
from typing import Any

import httpx

from app.core.config import settings
from app.services.evaluation_telemetry import new_request_id, pace_gemini_request, record_gemini_request


class GeminiProvider:
    """Small Gemini REST adapter for text and native structured JSON generation."""

    def __init__(self, api_key: str | None = None, model: str | None = None, retry_delay_seconds: float = 1.0):
        self.api_key = api_key or settings.gemini_api_key
        self.model = model or settings.gemini_model
        self.retry_delay_seconds = retry_delay_seconds

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _generate(self, prompt: str, response_schema: dict[str, Any] | None = None) -> str:
        if not self.api_key:
            raise RuntimeError("Gemini API key is not configured")

        generation_config: dict[str, Any] = {"temperature": 0}
        if response_schema is not None:
            generation_config.update({"responseMimeType": "application/json", "responseSchema": response_schema})

        request = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": generation_config,
        }
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        request_id = new_request_id()
        max_attempts = 2
        for attempt in range(1, max_attempts + 1):
            pace_gemini_request()
            started = time.monotonic()
            response = None
            try:
                response = httpx.post(url, params={"key": self.api_key}, json=request, timeout=60)
                status_code = response.status_code
                if status_code == 429 or status_code >= 500:
                    if attempt < max_attempts:
                        retry_after = response.headers.get("Retry-After")
                        try:
                            delay = float(retry_after) if retry_after else self.retry_delay_seconds * attempt
                        except ValueError:
                            delay = self.retry_delay_seconds * attempt
                        delay = max(0.0, min(delay, 10.0))
                        elapsed = (time.monotonic() - started) * 1000
                        record_gemini_request(
                            request_id=request_id,
                            provider="generation",
                            attempt=attempt,
                            status="failed",
                            status_code=status_code,
                            latency_ms=elapsed,
                            retry_wait_ms=delay * 1000,
                        )
                        time.sleep(delay)
                        continue

                response.raise_for_status()
                payload = response.json()
                usage_metadata = payload.get("usageMetadata")
                usage = usage_metadata if isinstance(usage_metadata, dict) else None
                record_gemini_request(
                    request_id=request_id,
                    provider="generation",
                    attempt=attempt,
                    status="succeeded",
                    status_code=status_code,
                    latency_ms=(time.monotonic() - started) * 1000,
                    usage=usage,
                )
                return payload["candidates"][0]["content"]["parts"][0]["text"]
            except httpx.HTTPStatusError as error:
                record_gemini_request(
                    request_id=request_id,
                    provider="generation",
                    attempt=attempt,
                    status="failed",
                    status_code=error.response.status_code,
                    latency_ms=(time.monotonic() - started) * 1000,
                )
                raise
            except Exception:
                if response is None:
                    record_gemini_request(
                        request_id=request_id,
                        provider="generation",
                        attempt=attempt,
                        status="failed",
                        status_code=None,
                        latency_ms=(time.monotonic() - started) * 1000,
                    )
                raise
        raise RuntimeError("Gemini generation failed after bounded retry")

    def generate_text(self, prompt: str) -> str:
        return self._generate(prompt)

    def generate_json(self, prompt: str, response_schema: dict[str, Any]) -> str:
        return self._generate(prompt, response_schema=response_schema)