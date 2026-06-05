"""JSON log line shape + rotating-file handler installation."""

from __future__ import annotations

import io
import json
import logging
import sys
import uuid
from pathlib import Path

import pytest

from lead_priority.utils.logging import JsonFormatter, configure_logging


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


def test_configure_logging_writes_to_log_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """configure_logging installs a rotating file handler at settings.log_file.

    Guards against silent regression where the dictConfig drops the file
    handler — the user's "logs in the project directory" requirement.
    """
    log_path = tmp_path / "subdir" / "app.log"
    monkeypatch.setenv("LOG_FILE", str(log_path))
    configure_logging("INFO")
    logger = logging.getLogger(f"lead_priority.test.{uuid.uuid4().hex}")
    logger.info("file_handler_check", extra={"k": "v"})
    for handler in logging.getLogger().handlers:
        handler.flush()
    assert log_path.exists()
    written = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert written, "log file should contain at least one record"
    payload = json.loads(written[-1])
    assert payload["msg"] == "file_handler_check"
    assert payload["k"] == "v"
