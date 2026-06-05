"""JSON log line shape + request-id middleware."""

from __future__ import annotations

import io
import json
import logging
import sys
import uuid

from lead_priority.api.logging import JsonFormatter


def test_json_formatter_emits_parseable_json() -> None:
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="lead_priority.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="score_completed",
        args=(),
        exc_info=None,
    )
    record.request_id = "req-abc"
    record.priority = 0.85
    line = formatter.format(record)
    payload = json.loads(line)
    assert payload["level"] == "INFO"
    assert payload["logger"] == "lead_priority.test"
    assert payload["msg"] == "score_completed"
    assert payload["request_id"] == "req-abc"
    assert payload["priority"] == 0.85
    assert "ts" in payload


def test_json_formatter_handles_exception() -> None:
    formatter = JsonFormatter()
    try:
        raise ValueError("boom")
    except ValueError:
        record = logging.LogRecord(
            name="lead_priority.test",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="kaput",
            args=(),
            exc_info=True,
        )
        record.exc_info = sys.exc_info()
        line = formatter.format(record)
    payload = json.loads(line)
    assert payload["level"] == "ERROR"
    assert "exc" in payload
    assert "ValueError: boom" in payload["exc"]


def test_json_formatter_writes_to_stream_handler() -> None:
    """End-to-end: handler + formatter → buffer → parseable JSON line."""
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger(f"lead_priority.test.{uuid.uuid4().hex}")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.info("hello", extra={"latency_ms": 12.3})
    line = buffer.getvalue().strip()
    payload = json.loads(line)
    assert payload["msg"] == "hello"
    assert payload["latency_ms"] == 12.3
