"""Structured JSON logging for the FastAPI service.

The case study lists logging as a bonus, and a JSON formatter makes the logs
greppable in any aggregator (Loki, CloudWatch, BigQuery) without an extra
parser stage. PII guard: ``lead`` payload contents and ``interaction_text``
NEVER appear in log records — only feature counts and string lengths.
"""

from __future__ import annotations

import json
import logging
import logging.config
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

REQUEST_ID_HEADER = "X-Request-Id"

_RESERVED_LOG_RECORD_ATTRS = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
        "asctime",
        "taskName",
    }
)


class JsonFormatter(logging.Formatter):
    """One JSON object per log record on stdout.

    ``extra={...}`` keys passed at the call site are merged into the top-level
    payload so callers can attach ``request_id``, ``latency_ms``, ``priority``,
    etc. without nesting.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _RESERVED_LOG_RECORD_ATTRS or key.startswith("_"):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(level: str) -> None:
    """Install :class:`JsonFormatter` on root + uvicorn loggers.

    Idempotent: calling twice is safe because :func:`logging.config.dictConfig`
    replaces handlers wholesale.
    """
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {"()": "lead_priority.api.logging.JsonFormatter"},
            },
            "handlers": {
                "stdout": {
                    "class": "logging.StreamHandler",
                    "formatter": "json",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {"level": level.upper(), "handlers": ["stdout"]},
            "loggers": {
                "uvicorn": {"level": level.upper(), "handlers": ["stdout"], "propagate": False},
                "uvicorn.access": {
                    "level": level.upper(),
                    "handlers": ["stdout"],
                    "propagate": False,
                },
                "uvicorn.error": {
                    "level": level.upper(),
                    "handlers": ["stdout"],
                    "propagate": False,
                },
            },
        }
    )


class RequestIdMiddleware(BaseHTTPMiddleware):  # type: ignore[misc]
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


__all__ = [
    "REQUEST_ID_HEADER",
    "JsonFormatter",
    "RequestIdMiddleware",
    "configure_logging",
]
