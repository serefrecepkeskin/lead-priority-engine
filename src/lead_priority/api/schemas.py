"""Pydantic v2 request / response models for the FastAPI service.

The lead payload is intentionally typed as ``dict[str, Any]`` because the
feature pipeline already validates column presence inside ``derive_features``
and the case study leaves the schema open ("ham CSV column → value sözlüğü").
Pulling each of the 30+ raw columns into a Pydantic model would mostly add
maintenance burden without catching real bugs — pipeline-level validation is
the source of truth.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from lead_priority.models import SENTIMENT_CLASSES, SentimentClass


class ScoreRequest(BaseModel):
    """Request body for ``POST /score``."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "lead": {
                    "Lead Origin": "API",
                    "Lead Source": "Olark Chat",
                    "TotalVisits": 3,
                    "Total Time Spent on Website": 540,
                },
                "interaction_text": "Fiyatlandırmayı yüksek buldu ve geri "
                "dönüş süresini sorguladı.",
            }
        }
    )

    lead: dict[str, Any] = Field(
        ...,
        description="Raw lead record (CSV-shaped). Unknown keys are dropped by "
        "the feature pipeline; required columns must be present.",
    )
    interaction_text: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="The most recent sales-rep interaction note. TR / EN / Mix are all supported.",
    )


class SentimentBlock(BaseModel):
    """Sentiment sub-block of :class:`ScoreResponse`."""

    predicted_attitude: SentimentClass = Field(
        ...,
        description=f"One of {list(SENTIMENT_CLASSES)}.",
    )
    sentiment_score: float = Field(..., ge=0.0, le=1.0)
    sentiment_unavailable: bool = Field(
        ...,
        description="True when the upstream LLM call failed and the response "
        "used the neutral fallback to compute priority.",
    )
    fallback_reason: str | None = Field(
        None,
        description="Populated when sentiment_unavailable is true. One of: "
        "openrouter_rate_limit, openrouter_unavailable, config_error.",
    )
    latency_ms: float | None = Field(
        None,
        description="OpenRouter call latency in milliseconds. None when "
        "sentiment_unavailable is true.",
    )
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class ModelVersions(BaseModel):
    """Versioning block returned by /score, /leads/top, /readyz."""

    feature_pipeline_schema: int
    lead_scoring_kind: str
    lead_scoring_schema: int
    sentiment_model_name: str


class ScoreResponse(BaseModel):
    """Response body for ``POST /score``."""

    p_conversion: float = Field(..., ge=0.0, le=1.0)
    sentiment: SentimentBlock
    priority: float = Field(..., ge=0.0, le=1.0)
    weights: dict[str, float]
    model_versions: ModelVersions
    request_id: str | None = None


class TopLeadEntry(BaseModel):
    """Single row of :class:`TopLeadsResponse.leads`."""

    lead_id: str
    p_conversion: float = Field(..., ge=0.0, le=1.0)
    predicted_attitude: SentimentClass
    sentiment_score: float = Field(..., ge=0.0, le=1.0)
    priority: float = Field(..., ge=0.0, le=1.0)
    language: str


class TopLeadsResponse(BaseModel):
    """Response body for ``GET /leads/top``."""

    count: int = Field(..., ge=0)
    total_available: int = Field(..., ge=0)
    leads: list[TopLeadEntry]
    model_versions: ModelVersions
    request_id: str | None = None


class HealthResponse(BaseModel):
    """Response body for /healthz and /readyz.

    ``/healthz`` returns the trivial subset (status + app_env); ``/readyz``
    additionally populates ``model_versions``, ``metrics_summary``, ``checks``.
    """

    status: Literal["ok", "degraded"]
    app_env: str
    model_versions: ModelVersions | None = None
    metrics_summary: dict[str, Any] | None = None
    checks: dict[str, bool] | None = None


__all__ = [
    "HealthResponse",
    "ModelVersions",
    "ScoreRequest",
    "ScoreResponse",
    "SentimentBlock",
    "TopLeadEntry",
    "TopLeadsResponse",
]
