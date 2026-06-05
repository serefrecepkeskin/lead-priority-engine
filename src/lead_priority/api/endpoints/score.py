"""``POST /score`` — combined priority for one lead + interaction note.

The endpoint runs three operations in sequence:

1. **Feature transform + conversion probability** via the cached
   :class:`FeatureTransformer` and :class:`LeadScoringModel`. A bad lead
   payload (missing required column) bubbles up as a 422 through the
   pipeline's own ``KeyError`` → translated below.
2. **Sentiment** via the cached :class:`OpenRouterSentiment`. The call runs
   in a worker thread because the wrapper opens a fresh ``httpx.Client`` per
   call. Rate-limit / transport failures are absorbed: the response stays
   200 with ``sentiment_unavailable=true`` so the rep still gets a useful
   (neutral-fallback) priority. ``OpenRouterPermanentError`` propagates to
   the 502 handler in ``errors.py`` because a 4xx from the provider is a
   real bug and silently degrading would mask it.
3. **Priority** via :func:`compute_priority` with the resolved attitude
   (real or neutral fallback).

Logging avoids the raw ``lead`` payload and ``interaction_text``; only
shape/size and the model outputs reach the log line.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Request, status
from starlette.concurrency import run_in_threadpool

from lead_priority.api.deps import (
    get_feature_transformer,
    get_lead_scoring_model,
    get_sentiment_classifier,
    model_versions_payload,
)
from lead_priority.api.schemas import (
    ModelVersions,
    ScoreRequest,
    ScoreResponse,
    SentimentBlock,
)
from lead_priority.core.scoring.priority import compute_priority
from lead_priority.core.scoring.sentiment_classes import (
    SENTIMENT_SCORE_MAP,
    SentimentClass,
)
from lead_priority.infra.openrouter.sentiment import (
    OpenRouterError,
    OpenRouterMalformedError,
    OpenRouterPermanentError,
    OpenRouterRateLimitError,
    OpenRouterSentiment,
)
from lead_priority.settings import get_settings

logger = logging.getLogger("lead_priority.api.score")

FALLBACK_ATTITUDE: SentimentClass = "neutral"
"""Used when sentiment is unavailable. ``neutral`` is the mid-low score
(0.40 in :data:`SENTIMENT_SCORE_MAP`) — biases priority toward the
conversion-probability signal alone."""

router = APIRouter(tags=["score"])


def _score_lead(lead_payload: dict[str, Any]) -> float:
    """Run the feature pipeline + classifier on a single lead record."""
    transformer = get_feature_transformer()
    scorer = get_lead_scoring_model()
    raw_df = pd.DataFrame.from_records([lead_payload])
    try:
        features = transformer.transform(raw_df)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"detail": "missing_lead_columns", "missing": str(exc)},
        ) from exc
    proba = scorer.predict_proba(features)
    return float(proba[0])


def _sentiment_or_fallback(text: str) -> tuple[SentimentBlock, SentimentClass]:
    """Resolve sentiment with graceful degradation on transient failures.

    Returns the sentiment block and the attitude label to feed into
    :func:`compute_priority` (either the real class or the neutral fallback).
    """
    classifier: OpenRouterSentiment
    try:
        classifier = get_sentiment_classifier()
    except RuntimeError:
        logger.warning("sentiment_config_error_using_fallback")
        return (
            SentimentBlock(
                predicted_attitude=FALLBACK_ATTITUDE,
                sentiment_score=SENTIMENT_SCORE_MAP[FALLBACK_ATTITUDE],
                sentiment_unavailable=True,
                fallback_reason="config_error",
                latency_ms=None,
                prompt_tokens=None,
                completion_tokens=None,
            ),
            FALLBACK_ATTITUDE,
        )

    try:
        attitude = classifier.predict(text)
    except (OpenRouterPermanentError, OpenRouterMalformedError):
        # Real bug — let the 502 handler in errors.py surface it.
        raise
    except OpenRouterRateLimitError:
        logger.warning("openrouter_rate_limit_using_fallback")
        return (
            SentimentBlock(
                predicted_attitude=FALLBACK_ATTITUDE,
                sentiment_score=SENTIMENT_SCORE_MAP[FALLBACK_ATTITUDE],
                sentiment_unavailable=True,
                fallback_reason="openrouter_rate_limit",
                latency_ms=None,
                prompt_tokens=None,
                completion_tokens=None,
            ),
            FALLBACK_ATTITUDE,
        )
    except OpenRouterError:
        logger.warning("openrouter_transient_failure_using_fallback")
        return (
            SentimentBlock(
                predicted_attitude=FALLBACK_ATTITUDE,
                sentiment_score=SENTIMENT_SCORE_MAP[FALLBACK_ATTITUDE],
                sentiment_unavailable=True,
                fallback_reason="openrouter_unavailable",
                latency_ms=None,
                prompt_tokens=None,
                completion_tokens=None,
            ),
            FALLBACK_ATTITUDE,
        )

    usage = classifier.last_usage()
    return (
        SentimentBlock(
            predicted_attitude=attitude,
            sentiment_score=SENTIMENT_SCORE_MAP[attitude],
            sentiment_unavailable=False,
            fallback_reason=None,
            latency_ms=round(classifier.last_latency_ms(), 2),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        ),
        attitude,
    )


@router.post("/score", response_model=ScoreResponse)
async def score(req: ScoreRequest, request: Request) -> ScoreResponse:
    """Score a single lead and classify the interaction note."""
    request_id = getattr(request.state, "request_id", None)
    start = time.perf_counter()

    p_conversion = _score_lead(req.lead)

    sentiment_block, attitude = await run_in_threadpool(
        _sentiment_or_fallback, req.interaction_text
    )

    priority = compute_priority(p_conversion, attitude)
    settings = get_settings()

    response = ScoreResponse(
        p_conversion=round(p_conversion, 6),
        sentiment=sentiment_block,
        priority=round(priority, 6),
        weights={
            "weight_conversion": settings.priority_weight_conversion,
            "weight_sentiment": settings.priority_weight_sentiment,
        },
        model_versions=ModelVersions(**model_versions_payload()),
        request_id=request_id,
    )

    logger.info(
        "score_completed",
        extra={
            "request_id": request_id,
            "feature_count": len(req.lead),
            "interaction_text_len": len(req.interaction_text),
            "p_conversion": response.p_conversion,
            "predicted_attitude": sentiment_block.predicted_attitude,
            "sentiment_unavailable": sentiment_block.sentiment_unavailable,
            "fallback_reason": sentiment_block.fallback_reason,
            "sentiment_latency_ms": sentiment_block.latency_ms,
            "sentiment_tokens_total": (
                (sentiment_block.prompt_tokens or 0) + (sentiment_block.completion_tokens or 0)
                if sentiment_block.prompt_tokens is not None
                else None
            ),
            "priority": response.priority,
            "total_latency_ms": round((time.perf_counter() - start) * 1000.0, 2),
        },
    )

    return response


__all__ = ["FALLBACK_ATTITUDE", "router"]
