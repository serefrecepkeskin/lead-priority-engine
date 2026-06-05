"""Serving-side priority score: weighted average of P(conversion) and sentiment_score.

This is the Phase 4 wiring module. It combines the Phase 2 lead-scoring
probability with the Phase 3 sentiment score into a single number a sales rep
can sort on. The formula is the simplest defensible mix — a fixed weighted
average — because the design goal here is product intuition over a perfectly
tuned formula, and because either component being uncalibrated (LGBM
probabilities or LLM-label confidence) makes a learned meta-model brittle on
the kind of synthetic data this project ships with.

Serving flow::

    p_conv = lead_scoring_model.predict_proba(transformer.transform(raw_df))
    attitude = sentiment.predict(interaction_text)
    priority = compute_priority(p_conv, attitude)
    # -> 0.6 * p_conv + 0.4 * SENTIMENT_SCORE_MAP[attitude]

Default weights live in :class:`lead_priority.settings.Settings`
(``priority_weight_conversion=0.6``, ``priority_weight_sentiment=0.4``) so the
operator can re-balance via ``.env`` without touching code. Both weights and
the sentiment → score map are injectable so notebooks can run ablations
without going through ``Settings`` reloads.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from lead_priority.core.scoring.sentiment_classes import SENTIMENT_CLASSES, SENTIMENT_SCORE_MAP
from lead_priority.settings import get_settings

WEIGHT_SUM_TOLERANCE = 1e-6
"""Slack on ``w_conv + w_sent == 1`` to absorb float round-off without
allowing the operator to silently de-normalise the formula."""


def _resolve_weights(
    weight_conversion: float | None,
    weight_sentiment: float | None,
) -> tuple[float, float]:
    """Fill in ``None`` weights from settings and validate the pair.

    Both values must be provided together for an override to take effect;
    if either is ``None``, both are read from :class:`Settings` so the
    operator's ``.env`` stays the single source of truth.
    """
    if weight_conversion is None or weight_sentiment is None:
        settings = get_settings()
        w_conv = settings.priority_weight_conversion
        w_sent = settings.priority_weight_sentiment
    else:
        w_conv = weight_conversion
        w_sent = weight_sentiment
    if abs((w_conv + w_sent) - 1.0) > WEIGHT_SUM_TOLERANCE:
        raise ValueError(
            f"priority weights must sum to 1.0 (±{WEIGHT_SUM_TOLERANCE}); "
            f"got w_conv={w_conv}, w_sent={w_sent}, sum={w_conv + w_sent}"
        )
    if w_conv < 0.0 or w_sent < 0.0:
        raise ValueError(
            f"priority weights must be non-negative; got w_conv={w_conv}, w_sent={w_sent}"
        )
    return w_conv, w_sent


def _resolve_score_map(
    sentiment_score_map: Mapping[str, float] | None,
) -> Mapping[str, float]:
    """Return the caller-provided map or the package default."""
    return sentiment_score_map if sentiment_score_map is not None else SENTIMENT_SCORE_MAP


def compute_priority(
    p_conversion: float,
    attitude: str,
    *,
    weight_conversion: float | None = None,
    weight_sentiment: float | None = None,
    sentiment_score_map: Mapping[str, float] | None = None,
) -> float:
    """Combine ``P(conversion)`` and a sentiment label into one priority score.

    Args:
        p_conversion: Output of :meth:`LeadScoringModel.predict_proba` for a
            single lead; must be in ``[0, 1]``.
        attitude: One of :data:`SENTIMENT_CLASSES` — typically the output of
            :meth:`OpenRouterSentiment.predict`.
        weight_conversion: Override for ``w_conv``. ``None`` falls back to
            ``Settings.priority_weight_conversion``.
        weight_sentiment: Override for ``w_sent``. ``None`` falls back to
            ``Settings.priority_weight_sentiment``.
        sentiment_score_map: Override for the label → float map. ``None``
            falls back to :data:`SENTIMENT_SCORE_MAP`. Injected by notebooks
            running mapping ablations.

    Returns:
        ``w_conv * p_conversion + w_sent * score_map[attitude]`` — a float in
        ``[0, 1]`` when both weights and inputs respect their declared ranges.

    Raises:
        ValueError: ``p_conversion`` is outside ``[0, 1]``, ``attitude`` is not
            in the score map, or the resolved weights do not sum to ``1.0``.
    """
    if not 0.0 <= p_conversion <= 1.0:
        raise ValueError(f"p_conversion must be in [0, 1]; got {p_conversion}")
    score_map = _resolve_score_map(sentiment_score_map)
    if attitude not in score_map:
        raise ValueError(
            f"attitude {attitude!r} not in score map; expected one of {sorted(score_map.keys())}"
        )
    w_conv, w_sent = _resolve_weights(weight_conversion, weight_sentiment)
    return w_conv * p_conversion + w_sent * score_map[attitude]


def batch_compute_priority(
    p_conversion: np.ndarray,
    attitudes: Sequence[str],
    *,
    weight_conversion: float | None = None,
    weight_sentiment: float | None = None,
    sentiment_score_map: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Vectorised :func:`compute_priority` for a batch of leads.

    Used by the Phase 4 demo notebook to score all 924 sentiment-test leads in
    a single call, and reusable from any future batch scorer (e.g. a nightly
    re-ranking job). The vectorised path skips the per-row Python loop but
    keeps the same range / membership validation, so a bad row in either
    array surfaces as a single ``ValueError`` rather than corrupt output.

    Args:
        p_conversion: 1D array of probabilities in ``[0, 1]``.
        attitudes: Sequence of attitude labels, same length as
            ``p_conversion``. Each label must be in the resolved score map.
        weight_conversion: See :func:`compute_priority`.
        weight_sentiment: See :func:`compute_priority`.
        sentiment_score_map: See :func:`compute_priority`.

    Returns:
        1D ndarray of priorities, same length as ``p_conversion``.

    Raises:
        ValueError: Shape mismatch, any probability outside ``[0, 1]``, any
            unknown attitude, or weight sum off from ``1.0``.
    """
    p_arr = np.asarray(p_conversion, dtype=float)
    if p_arr.ndim != 1:
        raise ValueError(f"p_conversion must be 1D; got shape {p_arr.shape}")
    if len(p_arr) != len(attitudes):
        raise ValueError(
            f"p_conversion and attitudes must have the same length; "
            f"got {len(p_arr)} and {len(attitudes)}"
        )
    if p_arr.size and (p_arr.min() < 0.0 or p_arr.max() > 1.0):
        raise ValueError(
            f"p_conversion must be in [0, 1]; got min={p_arr.min()}, max={p_arr.max()}"
        )
    score_map = _resolve_score_map(sentiment_score_map)
    unknown = {label for label in attitudes if label not in score_map}
    if unknown:
        raise ValueError(
            f"unknown attitudes {sorted(unknown)}; expected one of {sorted(score_map.keys())}"
        )
    w_conv, w_sent = _resolve_weights(weight_conversion, weight_sentiment)
    sentiment_arr = np.fromiter(
        (score_map[label] for label in attitudes),
        dtype=float,
        count=len(attitudes),
    )
    return w_conv * p_arr + w_sent * sentiment_arr


__all__ = [
    "SENTIMENT_CLASSES",
    "SENTIMENT_SCORE_MAP",
    "WEIGHT_SUM_TOLERANCE",
    "batch_compute_priority",
    "compute_priority",
]
