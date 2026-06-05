"""Structured JSON logging for the whole `lead_priority` package.

A JSON formatter keeps records greppable in any aggregator (Loki, CloudWatch,
BigQuery) without an extra parser stage. Two handlers are installed by
:func:`configure_logging`:

1. stdout — the 12-factor default; Docker / k8s / Cloud Run capture it.
2. rotating file at ``settings.LOG_FILE`` — local-dev inspection after the
   process exits. ``RotatingFileHandler`` caps each file at 10 MiB with five
   backups so the log volume can't run away on a developer laptop.

PII guard: ``lead`` payload contents and ``interaction_text`` NEVER appear
in log records — only feature counts and string lengths. The formatter
itself enforces nothing; the contract is upheld at every call site.

The module is intentionally transport-agnostic — HTTP-specific middleware
lives in :mod:`lead_priority.api.middleware`.
"""

from __future__ import annotations

import json
import logging
import logging.config
import time
from typing import Any

from lead_priority.settings import get_settings

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
    """Install :class:`JsonFormatter` on root + uvicorn loggers, plus a file handler.

    Idempotent: calling twice is safe because :func:`logging.config.dictConfig`
    replaces handlers wholesale. Creates the ``LOG_FILE`` parent directory if
    it doesn't exist — local dev runs may not have ``logs/`` yet.
    """
    settings = get_settings()
    log_file = settings.log_file
    log_file.parent.mkdir(parents=True, exist_ok=True)

    handlers_config: dict[str, dict[str, Any]] = {
        "stdout": {
            "class": "logging.StreamHandler",
            "formatter": "json",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "json",
            "filename": str(log_file),
            "maxBytes": 10 * 1024 * 1024,
            "backupCount": 5,
            "encoding": "utf-8",
        },
    }
    handler_names = list(handlers_config.keys())

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {"()": "lead_priority.utils.logging.JsonFormatter"},
            },
            "handlers": handlers_config,
            "root": {"level": level.upper(), "handlers": handler_names},
            "loggers": {
                "uvicorn": {"level": level.upper(), "handlers": handler_names, "propagate": False},
                "uvicorn.access": {
                    "level": level.upper(),
                    "handlers": handler_names,
                    "propagate": False,
                },
                "uvicorn.error": {
                    "level": level.upper(),
                    "handlers": handler_names,
                    "propagate": False,
                },
            },
        }
    )


__all__ = ["JsonFormatter", "configure_logging"]
