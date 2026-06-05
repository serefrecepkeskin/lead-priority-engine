"""Pydantic v2 request / response models for the FastAPI service.

The lead payload is intentionally typed as ``dict[str, Any]`` because the
feature pipeline already validates column presence inside ``derive_features``
and the project intentionally leaves the wire schema open ("raw CSV column →
value dict"). Pulling each of the 30+ raw columns into a Pydantic model would
mostly add maintenance burden without catching real bugs — pipeline-level
validation is the source of truth.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from lead_priority.core.scoring.sentiment_classes import SENTIMENT_CLASSES, SentimentClass

_SCORE_REQUEST_EXAMPLE: dict[str, Any] = {
    "lead": {
        "Prospect ID": "7927b2df-8bba-4d29-b9a2-b6e0beafe620",
        "Lead Number": 660737,
        "Lead Origin": "API",
        "Lead Source": "Olark Chat",
        "Do Not Email": "No",
        "Do Not Call": "No",
        "TotalVisits": 0,
        "Total Time Spent on Website": 0,
        "Page Views Per Visit": 0.0,
        "Last Activity": "Page Visited on Website",
        "Country": "India",
        "Specialization": "Select",
        "How did you hear about X Education": "Select",
        "What is your current occupation": "Unemployed",
        "What matters most to you in choosing a course": "Better Career Prospects",
        "Search": "No",
        "Magazine": "No",
        "Newspaper Article": "No",
        "X Education Forums": "No",
        "Newspaper": "No",
        "Digital Advertisement": "No",
        "Through Recommendations": "No",
        "Receive More Updates About Our Courses": "No",
        "Tags": "Interested in other courses",
        "Lead Quality": "Low in Relevance",
        "Update me on Supply Chain Content": "No",
        "Get updates on DM Content": "No",
        "Lead Profile": "Select",
        "City": "Select",
        "Asymmetrique Activity Index": "02.Medium",
        "Asymmetrique Profile Index": "02.Medium",
        "Asymmetrique Activity Score": 15,
        "Asymmetrique Profile Score": 15,
        "I agree to pay the amount through cheque": "No",
        "A free copy of Mastering The Interview": "No",
        "Last Notable Activity": "Modified",
    },
    "interaction_text": (
        "Fiyatlandırmayı yüksek buldu ve geri dönüş süresini sorguladı; "
        "mevcut çözümlerine kıyasla teklif edilen onboarding planının "
        "kapsamını sorguladı."
    ),
}
"""Mirrors ``examples/score_request.json`` so the Swagger UI 'Try it out'
button hits a complete row. Inlined (rather than loaded from disk) because
``examples/`` is not copied into the runtime Docker image."""


class ScoreRequest(BaseModel):
    """Request body for ``POST /score``."""

    model_config = ConfigDict(json_schema_extra={"example": _SCORE_REQUEST_EXAMPLE})

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
