"""Test bootstrap.

The data-generation code lives under ``scripts/`` (not in the runtime
package), so we prepend ``scripts/`` to ``sys.path`` here. CLI scripts
already see it automatically when run directly; tests and the notebook
need this nudge.

The API fixtures construct a fresh ``FastAPI`` instance per test with
``lru_cache``s cleared so monkeypatched env vars take effect — see
``test_api_*.py`` for usage.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lead_priority.api import deps
from lead_priority.api.main import create_app

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


@pytest.fixture
def api_app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    """A fresh FastAPI app with a fake OpenRouter key and cleared caches."""
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "sk-or-test-key")
    deps.reset_caches()
    return create_app()


@pytest.fixture
def client(api_app: FastAPI) -> Iterator[TestClient]:
    """Sync ``TestClient`` over the app — context manager triggers lifespan."""
    with TestClient(api_app) as test_client:
        yield test_client
