"""Shared sentiment label constants used by Phase 3 (classifier) and Phase 4 (priority).

Lives under ``core/scoring`` rather than ``infra/openrouter`` so the priority
aggregator can consume it without pulling in the HTTP client — the layered
architecture forbids ``core`` from importing ``infra``.
"""

from __future__ import annotations

from typing import Literal

SentimentClass = Literal[
    "positive_engagement",
    "objection",
    "neutral",
    "disengaged",
]

SENTIMENT_CLASSES: tuple[SentimentClass, ...] = (
    "positive_engagement",
    "objection",
    "neutral",
    "disengaged",
)
"""The four interaction-note attitudes synthesised in Phase 0."""

SENTIMENT_SCORE_MAP: dict[str, float] = {
    "positive_engagement": 1.0,
    "objection": 0.65,
    "neutral": 0.40,
    "disengaged": 0.10,
}
"""Sentiment → priority-score mapping consumed by Phase 4 ``compute_priority``.

The 0.65 weight on ``objection`` is the load-bearing decision: the Phase 0
synthetic crosstab shows ~53% of objection leads convert, so collapsing
objections to a low score would discard real buying signal. The numbers
come from ``notes/next_steps.md`` §4 and are restated in
``docs/3_sentiment_classifier.docx`` §12.
"""

__all__ = [
    "SENTIMENT_CLASSES",
    "SENTIMENT_SCORE_MAP",
    "SentimentClass",
]
