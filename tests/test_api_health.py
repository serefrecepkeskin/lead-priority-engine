"""Liveness + readiness probes."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from lead_priority.api import deps
from lead_priority.api.main import create_app


def test_healthz_returns_ok(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "app_env" in body


def test_readyz_returns_ok_when_configured(client: TestClient) -> None:
    response = client.get("/readyz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["checks"]["feature_pipeline"] is True
    assert body["checks"]["lead_scoring"] is True
    assert body["checks"]["top_leads_cache"] is True
    assert body["checks"]["openrouter_key"] is True
    assert body["model_versions"]["lead_scoring_kind"] == "lightgbm"


def test_readyz_returns_degraded_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty OPEN_ROUTER_API_KEY → /readyz returns 503 + degraded status.

    Empty (rather than ``delenv``) because the project ``.env`` is committed
    on developer machines and pydantic-settings reads it after the process
    env. Setting an explicit empty string makes the override deterministic.
    """
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "")
    deps.reset_caches()
    app = create_app()
    with TestClient(app) as test_client:
        response = test_client.get("/readyz")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["checks"]["openrouter_key"] is False


def test_request_id_header_echoed(client: TestClient) -> None:
    """RequestIdMiddleware echoes incoming X-Request-Id on the response."""
    response = client.get("/healthz", headers={"X-Request-Id": "trace-abc-123"})
    assert response.status_code == 200
    assert response.headers.get("X-Request-Id") == "trace-abc-123"


def test_request_id_header_generated(client: TestClient) -> None:
    """Without an incoming X-Request-Id the middleware generates one."""
    response = client.get("/healthz")
    request_id = response.headers.get("X-Request-Id")
    assert request_id is not None
    assert len(request_id) >= 16
