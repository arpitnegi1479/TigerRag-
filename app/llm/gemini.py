from __future__ import annotations

import time
from typing import Any

import httpx

from app.core.config import settings


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
        for attempt in range(2):
            response = httpx.post(url, params={"key": self.api_key}, json=request, timeout=60)
            if response.status_code == 429 and attempt == 0:
                time.sleep(self.retry_delay_seconds)
                continue
            response.raise_for_status()
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        raise RuntimeError("Gemini generation failed after retry")

    def generate_text(self, prompt: str) -> str:
        return self._generate(prompt)

    def generate_json(self, prompt: str, response_schema: dict[str, Any]) -> str:
        return self._generate(prompt, response_schema=response_schema)