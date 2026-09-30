from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app: Any,
        *,
        max_requests: int = 120,
        window_seconds: float = 60.0,
    ):
        super().__init__(app)
        self.max_requests = max(1, max_requests)
        self.window_seconds = max(1.0, window_seconds)
        self._requests: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    async def dispatch(self, request: Request, call_next):
        client_key = request.client.host if request.client else "unknown"
        now = time.monotonic()
        with self._lock:
            window_start = now - self.window_seconds
            for key in list(self._requests):
                requests = self._requests[key]
                while requests and requests[0] <= window_start:
                    requests.popleft()
                if not requests:
                    del self._requests[key]
            requests = self._requests.setdefault(client_key, deque())
            if len(requests) >= self.max_requests:
                retry_after = max(1, math.ceil(self.window_seconds - (now - requests[0])))
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded"},
                    headers={"Retry-After": str(retry_after)},
                )
            requests.append(now)
        return await call_next(request)
