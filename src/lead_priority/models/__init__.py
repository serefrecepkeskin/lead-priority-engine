"""Serving-side model wrappers loaded by the FastAPI runtime.

Only predict-time code lives here. See ``scripts/train_lead_scoring.py`` for
the training-time CLI that produces the joblib bundles consumed by these
wrappers, and ``scripts/evaluate_openrouter_sentiment.py`` for the LLM
evaluation CLI that exercises the sentiment wrapper end-to-end.
"""

from __future__ import annotations

from lead_priority.models.lead_scoring import (
    ALLOWED_MODEL_KINDS,
    SCHEMA_VERSION,
    LeadScoringModel,
)
from lead_priority.models.sentiment import (
    MODEL_ALIASES,
    SENTIMENT_CLASSES,
    SENTIMENT_SCORE_MAP,
    OpenRouterError,
    OpenRouterPermanentError,
    OpenRouterRateLimitError,
    OpenRouterSentiment,
    SentimentClass,
)

__all__ = [
    "ALLOWED_MODEL_KINDS",
    "MODEL_ALIASES",
    "SCHEMA_VERSION",
    "SENTIMENT_CLASSES",
    "SENTIMENT_SCORE_MAP",
    "LeadScoringModel",
    "OpenRouterError",
    "OpenRouterPermanentError",
    "OpenRouterRateLimitError",
    "OpenRouterSentiment",
    "SentimentClass",
]
