"""LRU-cached model + cache loaders for the FastAPI service.

Each loader returns a fully-warmed object that the request handlers can use
without paying load latency on the request path. ``warm_models()`` is called
from the FastAPI ``lifespan`` so cold-start cost lives in the container boot
sequence, not in the first user-visible request.

Sentiment is intentionally loaded best-effort: a missing ``OPEN_ROUTER_API_KEY``
must NOT crash the service, because /healthz and /leads/top do not need it.
Instead :meth:`get_sentiment_classifier` raises and the score endpoint catches
the exception to enter graceful-degradation mode.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from lead_priority.core.features import FeatureTransformer
from lead_priority.core.inference.lead_scoring import LeadScoringModel
from lead_priority.infra.openrouter.sentiment import OpenRouterSentiment
from lead_priority.settings import get_settings

logger = logging.getLogger("lead_priority.api.deps")

FEATURE_PIPELINE_FILENAME = "feature_pipeline.joblib"
LEAD_SCORING_FALLBACK_FILENAME = "lead_scoring_lgbm.joblib"
SENTIMENT_PREDICTIONS_FILENAME = "sentiment_predictions/glm-4-5-air_test.parquet"
RAW_DATA_FILENAME = "Lead Scoring.csv"

METRICS_FILES: dict[str, str] = {
    "lead_scoring": "lead_scoring_metrics.json",
    "sentiment": "sentiment_metrics.json",
    "priority": "priority_metrics.json",
}


@lru_cache(maxsize=1)
def get_feature_transformer() -> FeatureTransformer:
    """Load the fitted feature pipeline. Cached for the process lifetime."""
    settings = get_settings()
    path = settings.artifacts_dir / FEATURE_PIPELINE_FILENAME
    logger.info("loading_feature_pipeline", extra={"path": str(path)})
    return FeatureTransformer.load(path)


@lru_cache(maxsize=1)
def get_lead_scoring_model() -> LeadScoringModel:
    """Load the trained lead scoring classifier. Cached for the process lifetime."""
    settings = get_settings()
    path = settings.artifacts_dir / settings.lead_scoring_model
    logger.info("loading_lead_scoring_model", extra={"path": str(path)})
    return LeadScoringModel.load(path)


@lru_cache(maxsize=1)
def get_sentiment_classifier() -> OpenRouterSentiment:
    """Construct the OpenRouter sentiment wrapper from Settings.

    Raises :class:`RuntimeError` (from :meth:`OpenRouterSentiment.from_settings`)
    when ``OPEN_ROUTER_API_KEY`` is unset. Callers in the score endpoint catch
    this and fall back to neutral sentiment; the readiness probe surfaces it
    as a 503.
    """
    settings = get_settings()
    logger.info(
        "loading_sentiment_classifier",
        extra={"model_name": settings.sentiment_model_name},
    )
    return OpenRouterSentiment.from_settings(settings.sentiment_model_name)


@lru_cache(maxsize=1)
def get_top_leads_cache() -> Any:
    """Build the in-memory top-leads list at startup.

    Lazy-imports :class:`TopLeadsCache` to break the otherwise-circular dep
    between this module and ``endpoints/top_leads.py`` (where the class lives
    so it can sit next to the route that consumes it).
    """
    from lead_priority.api.endpoints.top_leads import TopLeadsCache  # noqa: PLC0415

    settings = get_settings()
    raw_csv = settings.data_dir / RAW_DATA_FILENAME
    predictions_parquet = settings.artifacts_dir / SENTIMENT_PREDICTIONS_FILENAME
    transformer = get_feature_transformer()
    scorer = get_lead_scoring_model()
    logger.info(
        "building_top_leads_cache",
        extra={"raw_csv": str(raw_csv), "predictions_parquet": str(predictions_parquet)},
    )
    return TopLeadsCache.build(
        raw_csv=raw_csv,
        predictions_parquet=predictions_parquet,
        transformer=transformer,
        scorer=scorer,
    )


@lru_cache(maxsize=1)
def get_metrics_summary() -> dict[str, Any]:
    """Return a small subset of each tracked metrics JSON for /readyz.

    Reads the full JSONs once and keeps only the headline numbers — the
    full bundles can be ~50 KB and we do not want /readyz to be heavy.
    Missing files are tolerated (logged + omitted) so a partial deploy can
    still pass readiness if the relevant model is present.
    """
    settings = get_settings()
    summary: dict[str, Any] = {}
    for key, filename in METRICS_FILES.items():
        path = settings.artifacts_dir / filename
        if not path.exists():
            logger.warning("metrics_file_missing", extra={"path": str(path)})
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.exception("metrics_file_unreadable", extra={"path": str(path)})
            continue
        summary[key] = _trim_metrics(key, data)
    return summary


def _trim_metrics(key: str, data: dict[str, Any]) -> dict[str, Any]:
    """Pull the headline numbers out of a full metrics bundle.

    Heuristic — the JSONs do not share a fixed schema, so we walk a small
    set of well-known keys and copy whatever exists. Anything we miss is
    not load-bearing for /readyz.
    """
    headline_keys = (
        "test_roc_auc",
        "test_pr_auc",
        "test_f1_macro",
        "test_accuracy",
        "auc",
        "f1_macro",
        "accuracy",
        "n_test",
        "n_samples",
        "weight_conversion",
        "weight_sentiment",
        "model_kind",
        "model_name",
    )
    trimmed: dict[str, Any] = {}
    for k in headline_keys:
        if k in data:
            trimmed[k] = data[k]
    # The lead-scoring file nests under "lightgbm" / "logistic_regression" —
    # surface the LightGBM headline metrics if present.
    if key == "lead_scoring" and "lightgbm" in data and isinstance(data["lightgbm"], dict):
        lgbm = data["lightgbm"]
        for k in headline_keys:
            if k in lgbm:
                trimmed[f"lightgbm_{k}"] = lgbm[k]
    return trimmed


def warm_models() -> dict[str, bool]:
    """Eagerly load every cached loader.

    Sentiment is best-effort: a missing API key is logged but does not crash
    the boot. The returned dict shows which loaders succeeded so the caller
    (lifespan) can decide how loudly to complain.
    """
    status: dict[str, bool] = {}
    try:
        get_feature_transformer()
        status["feature_pipeline"] = True
    except Exception:
        logger.exception("feature_pipeline_load_failed")
        status["feature_pipeline"] = False
    try:
        get_lead_scoring_model()
        status["lead_scoring"] = True
    except Exception:
        logger.exception("lead_scoring_load_failed")
        status["lead_scoring"] = False
    try:
        get_top_leads_cache()
        status["top_leads_cache"] = True
    except Exception:
        logger.exception("top_leads_cache_build_failed")
        status["top_leads_cache"] = False
    try:
        get_sentiment_classifier()
        status["sentiment"] = True
    except RuntimeError as exc:
        logger.warning("sentiment_classifier_unavailable", extra={"reason": str(exc)})
        status["sentiment"] = False
    except Exception:
        logger.exception("sentiment_classifier_load_failed")
        status["sentiment"] = False
    return status


def reset_caches() -> None:
    """Clear every ``lru_cache`` in this module.

    Tests call this between cases so monkeypatched env vars take effect on
    the next loader call.
    """
    get_feature_transformer.cache_clear()
    get_lead_scoring_model.cache_clear()
    get_sentiment_classifier.cache_clear()
    get_top_leads_cache.cache_clear()
    get_metrics_summary.cache_clear()


def model_versions_payload() -> dict[str, Any]:
    """Build the ``model_versions`` block shared by /score, /leads/top, /readyz."""
    transformer = get_feature_transformer()
    scorer = get_lead_scoring_model()
    settings = get_settings()
    return {
        "feature_pipeline_schema": transformer.schema_version,
        "lead_scoring_kind": scorer.model_kind,
        "lead_scoring_schema": scorer.schema_version,
        "sentiment_model_name": settings.sentiment_model_name,
    }


def resolve_artifact_path(filename: str) -> Path:
    """Return ``settings.artifacts_dir / filename`` as an absolute Path."""
    return (get_settings().artifacts_dir / filename).resolve()


__all__ = [
    "FEATURE_PIPELINE_FILENAME",
    "LEAD_SCORING_FALLBACK_FILENAME",
    "METRICS_FILES",
    "RAW_DATA_FILENAME",
    "SENTIMENT_PREDICTIONS_FILENAME",
    "get_feature_transformer",
    "get_lead_scoring_model",
    "get_metrics_summary",
    "get_sentiment_classifier",
    "get_top_leads_cache",
    "model_versions_payload",
    "reset_caches",
    "resolve_artifact_path",
    "warm_models",
]
