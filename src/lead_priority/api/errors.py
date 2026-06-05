"""Centralised exception handlers for the FastAPI service.

The case study asks for ``/score`` to keep producing a useful priority even
when the LLM is unavailable, so transient OpenRouter failures are absorbed
inside the endpoint and ONLY non-transient errors propagate here. The handlers
below cover the remaining classes: permanent upstream errors (502), config
errors (500), and unexpected exceptions (500 with a redacted detail).
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from lead_priority.models import OpenRouterError, OpenRouterPermanentError

logger = logging.getLogger("lead_priority.api.errors")


class ConfigurationError(RuntimeError):
    """Raised on startup or per-request when the runtime is misconfigured.

    Distinct from a plain ``RuntimeError`` so callers can decide whether to
    fail loudly (boot) or degrade gracefully (per-request).
    """


def _request_id(request: Request) -> str | None:
    state = getattr(request, "state", None)
    return getattr(state, "request_id", None) if state is not None else None


def _problem(
    *,
    request: Request,
    status_code: int,
    detail: str,
    message: str,
    extra: dict[str, Any] | None = None,
) -> JSONResponse:
    body: dict[str, Any] = {
        "detail": detail,
        "message": message,
        "request_id": _request_id(request),
    }
    if extra:
        body.update(extra)
    return JSONResponse(status_code=status_code, content=body)


async def validation_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    return _problem(
        request=request,
        status_code=422,
        detail="validation_error",
        message="Request body did not match the expected schema.",
        extra={"errors": exc.errors()},
    )


async def permanent_openrouter_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, OpenRouterPermanentError)
    logger.warning(
        "openrouter_permanent_error",
        extra={"request_id": _request_id(request), "error": str(exc)},
    )
    return _problem(
        request=request,
        status_code=502,
        detail="openrouter_permanent_error",
        message="Sentiment provider rejected the request (non-transient).",
    )


async def openrouter_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, OpenRouterError)
    logger.warning(
        "openrouter_error",
        extra={"request_id": _request_id(request), "error": str(exc)},
    )
    return _problem(
        request=request,
        status_code=502,
        detail="openrouter_error",
        message="Sentiment provider returned malformed or unexpected output.",
    )


async def configuration_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ConfigurationError)
    logger.error(
        "configuration_error",
        extra={"request_id": _request_id(request), "error": str(exc)},
    )
    return _problem(
        request=request,
        status_code=500,
        detail="service_misconfigured",
        message="The service is missing required configuration. Check .env.",
    )


async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(
        "unhandled_exception",
        extra={"request_id": _request_id(request), "error_type": type(exc).__name__},
    )
    return _problem(
        request=request,
        status_code=500,
        detail="internal_error",
        message="An unexpected error occurred while handling the request.",
    )


def register_handlers(app: FastAPI) -> None:
    """Wire every handler in this module onto ``app``.

    Order matters: more-specific exceptions register before their parents.
    ``OpenRouterPermanentError`` is a subclass of :class:`OpenRouterError`, so
    its handler must register first.
    """
    app.add_exception_handler(RequestValidationError, validation_handler)
    app.add_exception_handler(OpenRouterPermanentError, permanent_openrouter_handler)
    app.add_exception_handler(OpenRouterError, openrouter_error_handler)
    app.add_exception_handler(ConfigurationError, configuration_handler)
    app.add_exception_handler(Exception, unhandled_handler)


__all__ = [
    "ConfigurationError",
    "configuration_handler",
    "openrouter_error_handler",
    "permanent_openrouter_handler",
    "register_handlers",
    "unhandled_handler",
    "validation_handler",
]
