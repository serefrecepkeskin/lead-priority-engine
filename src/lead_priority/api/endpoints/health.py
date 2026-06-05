"""Liveness and readiness probes.

``/healthz`` is a process-liveness check — it never inspects loaders so even
a misconfigured container returns 200 until the kernel kills the process.
``/readyz`` actively touches every loader plus the OpenRouter API key so a
load balancer can route traffic away from a partially-broken instance.
"""

from __future__ import annotations

from fastapi import APIRouter, Response

from lead_priority.api.deps import (
    get_feature_transformer,
    get_lead_scoring_model,
    get_metrics_summary,
    get_top_leads_cache,
    model_versions_payload,
)
from lead_priority.api.schemas import HealthResponse, ModelVersions
from lead_priority.settings import get_settings

router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    """Process liveness. Always 200 unless the process is dying."""
    settings = get_settings()
    return HealthResponse(status="ok", app_env=settings.app_env)


@router.get("/readyz", response_model=HealthResponse)
def readyz(response: Response) -> HealthResponse:
    """Readiness: every loader resolved + API key present.

    Returns 503 if any dependency is missing so a Kubernetes-style probe
    pulls the pod out of rotation. Probe never raises — all failures are
    surfaced as ``checks`` flags.
    """
    settings = get_settings()
    checks: dict[str, bool] = {}
    try:
        get_feature_transformer()
        checks["feature_pipeline"] = True
    except Exception:
        checks["feature_pipeline"] = False
    try:
        get_lead_scoring_model()
        checks["lead_scoring"] = True
    except Exception:
        checks["lead_scoring"] = False
    try:
        get_top_leads_cache()
        checks["top_leads_cache"] = True
    except Exception:
        checks["top_leads_cache"] = False
    checks["openrouter_key"] = bool(settings.open_router_api_key)

    all_ok = all(checks.values())
    if not all_ok:
        response.status_code = 503

    versions: ModelVersions | None
    if checks["feature_pipeline"] and checks["lead_scoring"]:
        versions = ModelVersions(**model_versions_payload())
    else:
        versions = None

    metrics = get_metrics_summary() if all_ok else None

    return HealthResponse(
        status="ok" if all_ok else "degraded",
        app_env=settings.app_env,
        model_versions=versions,
        metrics_summary=metrics,
        checks=checks,
    )


__all__ = ["router"]
