"""FastAPI application factory and uvicorn entry point.

The ``app`` instance at module bottom is what ``uvicorn lead_priority.api.main:app``
imports — this matches the ``make run`` target in the project Makefile.
``create_app()`` exists so tests can build fresh instances with a clean
``lru_cache`` state between cases.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version

from fastapi import FastAPI

from lead_priority.api import deps
from lead_priority.api.endpoints import health, score, top_leads
from lead_priority.api.errors import register_handlers
from lead_priority.api.middleware import RequestIdMiddleware
from lead_priority.settings import get_settings
from lead_priority.utils.logging import configure_logging

logger = logging.getLogger("lead_priority.api")


def _package_version() -> str:
    try:
        return version("lead-priority-engine")
    except PackageNotFoundError:
        return "0.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # noqa: ARG001
    """Warm every loader at boot so the first user request is fast.

    Sentiment is best-effort — see :func:`lead_priority.api.deps.warm_models`.
    Failures are logged but do not crash the process; ``/readyz`` is the
    canonical place to surface a partial deploy to an orchestrator.
    """
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info(
        "lifespan_startup",
        extra={"app_env": settings.app_env, "version": _package_version()},
    )
    status = deps.warm_models()
    logger.info("warm_models_completed", extra={"status": status})
    yield
    logger.info("lifespan_shutdown")


def create_app() -> FastAPI:
    """Build a fresh :class:`FastAPI` instance with all routes and middleware."""
    app = FastAPI(
        title="Lead Priority Engine",
        version=_package_version(),
        description=(
            "Tabular lead scoring + interaction sentiment combined into a "
            "single priority score. See "
            "docs/5_fastapi_serving_and_deployment.docx for setup, service "
            "design, and operational notes."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(RequestIdMiddleware)
    register_handlers(app)
    app.include_router(health.router)
    app.include_router(score.router)
    app.include_router(top_leads.router)
    return app


app = create_app()

__all__ = ["app", "create_app", "lifespan"]
