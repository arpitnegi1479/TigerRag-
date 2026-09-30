from __future__ import annotations

import time
import threading
import uuid
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any, Iterator


_active_trace: ContextVar["GeminiCallTrace | None"] = ContextVar("evaluation_gemini_trace", default=None)
_active_step: ContextVar[str] = ContextVar("evaluation_gemini_step", default="unspecified")


class GeminiCallTrace:
    _global_request_lock = threading.Lock()
    _global_last_request_at = 0.0

    def __init__(self, minimum_interval_seconds: float = 0.5):
        self.minimum_interval_seconds = max(0.0, minimum_interval_seconds)
        self.events: list[dict[str, Any]] = []

    def pace(self) -> None:
        with self._global_request_lock:
            now = time.monotonic()
            remaining = self.minimum_interval_seconds - (now - self._global_last_request_at)
            if remaining > 0:
                time.sleep(remaining)
            type(self)._global_last_request_at = time.monotonic()

    def record_request(
        self,
        *,
        request_id: str,
        provider: str,
        attempt: int,
        status: str,
        status_code: int | None,
        latency_ms: float,
        usage: dict[str, Any] | None = None,
        retry_wait_ms: float = 0.0,
    ) -> None:
        self.events.append(
            {
                "kind": "gemini_request",
                "step": _active_step.get(),
                "request_id": request_id,
                "provider": provider,
                "attempt": attempt,
                "status": status,
                "status_code": status_code,
                "latency_ms": round(latency_ms, 3),
                "retry_wait_ms": round(retry_wait_ms, 3),
                "usage": usage,
            }
        )

    def record_fallback(self, step: str, reason: str) -> None:
        self.events.append({"kind": "fallback", "step": step, "reason": reason})

    def summary(self) -> dict[str, Any]:
        requests = [event for event in self.events if event["kind"] == "gemini_request"]
        fallback_steps = [event for event in self.events if event["kind"] == "fallback"]
        successful_requests = {
            event["request_id"] for event in requests if event["status"] == "succeeded"
        }
        request_ids = {event["request_id"] for event in requests}
        failed_logical_requests = request_ids - successful_requests
        usage_events = [event["usage"] for event in requests if event.get("usage")]

        def sum_usage(key: str) -> int | None:
            values = [value[key] for value in usage_events if isinstance(value.get(key), int)]
            return sum(values) if values else None

        steps: dict[str, dict[str, Any]] = {}
        for event in requests:
            step = steps.setdefault(event["step"], {"gemini_attempts": 0, "succeeded": 0, "failed": 0, "fallbacks": 0})
            step["gemini_attempts"] += 1
            step["succeeded"] += event["status"] == "succeeded"
            step["failed"] += event["status"] == "failed"
        for event in fallback_steps:
            step = steps.setdefault(event["step"], {"gemini_attempts": 0, "succeeded": 0, "failed": 0, "fallbacks": 0})
            step["fallbacks"] += 1

        return {
            "gemini_http_attempts": len(requests),
            "gemini_logical_requests": len(request_ids),
            "gemini_logical_requests_failed": len(failed_logical_requests),
            "llm_latency_ms": round(
                sum(event["latency_ms"] + event["retry_wait_ms"] for event in requests), 3
            ),
            "token_usage": {
                "prompt": sum_usage("promptTokenCount"),
                "response": sum_usage("candidatesTokenCount"),
                "total": sum_usage("totalTokenCount"),
            },
            "estimated_cost_usd": None,
            "steps": steps,
            "fallback_events": fallback_steps,
            "degraded": bool(fallback_steps or failed_logical_requests),
        }


@contextmanager
def capture_gemini_calls(trace: GeminiCallTrace) -> Iterator[None]:
    token = _active_trace.set(trace)
    try:
        yield
    finally:
        _active_trace.reset(token)


@contextmanager
def evaluation_step(name: str) -> Iterator[None]:
    token = _active_step.set(name)
    try:
        yield
    finally:
        _active_step.reset(token)


def pace_gemini_request() -> None:
    trace = _active_trace.get()
    if trace is not None:
        trace.pace()


def record_gemini_request(**kwargs: Any) -> None:
    trace = _active_trace.get()
    if trace is not None:
        trace.record_request(**kwargs)


def record_fallback(step: str, reason: str) -> None:
    trace = _active_trace.get()
    if trace is not None:
        trace.record_fallback(step, reason)


def new_request_id() -> str:
    return str(uuid.uuid4())