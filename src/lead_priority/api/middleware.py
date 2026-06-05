"""HTTP middleware for the FastAPI service.

Holds the cross-cutting Starlette middleware that doesn't fit in any single
endpoint module. Generic JSON logging setup lives in
``lead_priority.utils.logging``; this module is the API-only complement that
imports from that shared setup.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

REQUEST_ID_HEADER = "X-Request-Id"


class RequestIdMiddleware(BaseHTTPMiddleware):  # type: ignore[misc, unused-ignore]
    """Attach a request id, measure latency, and log one record per request.

    The middleware reads ``X-Request-Id`` from the incoming request (a caller
    or upstream proxy may set it for trace correlation); otherwise it generates
    a fresh ``uuid4().hex``. Both inbound and outbound expose the value on
    ``request.state.request_id`` and the response header.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        incoming = request.headers.get(REQUEST_ID_HEADER)
        request_id = incoming if incoming else uuid.uuid4().hex
        request.state.request_id = request_id
        start = time.perf_counter()
        logger = logging.getLogger("lead_priority.api.request")
        try:
            response = await call_next(request)
        except Exception:
            latency_ms = (time.perf_counter() - start) * 1000.0
            logger.exception(
                "request_failed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "latency_ms": round(latency_ms, 2),
                },
            )
            raise
        latency_ms = (time.perf_counter() - start) * 1000.0
        response.headers[REQUEST_ID_HEADER] = request_id
        logger.info(
            "request_completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": round(latency_ms, 2),
            },
        )
        return response


__all__ = ["REQUEST_ID_HEADER", "RequestIdMiddleware"]
