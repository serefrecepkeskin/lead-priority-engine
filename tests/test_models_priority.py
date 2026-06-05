"""Unit tests for the Phase 4 priority-score wiring.

The module under test (`lead_priority.core.scoring.priority`) is a pure-Python
serving function: no HTTP, no model load, no disk I/O. Tests therefore avoid
fixtures from `tests/test_models_lead_scoring.py` and stay self-contained.

`Settings` is monkeypatched per-test rather than via env vars because the
`.env` at repo root would otherwise override the test expectations.
"""

from __future__ import annotations

import numpy as np
import pytest

import lead_priority.core.scoring.priority as priority_module
import lead_priority.settings as settings_module
from lead_priority.core.scoring.priority import batch_compute_priority, compute_priority
from lead_priority.core.scoring.sentiment_classes import SENTIMENT_CLASSES, SENTIMENT_SCORE_MAP


@pytest.fixture(autouse=True)
def _force_default_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``get_settings()`` return the documented Phase 4 defaults.

    The repo `.env` may override `priority_weight_*` for an operator's local
    experiments; tests need a deterministic 0.6 / 0.4 baseline so the manual
    arithmetic in the boundary cases stays valid.
    """

    class _DefaultSettings(settings_module.Settings):  # type: ignore[misc, valid-type]
        priority_weight_conversion: float = 0.6
        priority_weight_sentiment: float = 0.4
        model_config = settings_module.SettingsConfigDict(
            env_file=None,
            extra="ignore",
            case_sensitive=False,
        )

    monkeypatch.setattr(priority_module, "get_settings", _DefaultSettings)


@pytest.mark.parametrize("attitude", list(SENTIMENT_CLASSES))
@pytest.mark.parametrize("p_conversion", [0.0, 1.0])
def test_compute_priority_boundary_cases(attitude: str, p_conversion: float) -> None:
    """Hand-computed expected values at the input boundaries.

    Covers all four attitudes × both probability extremes — 8 cases total.
    Catches any sign-flip / weight-swap regression in the inner expression.
    """
    expected = 0.6 * p_conversion + 0.4 * SENTIMENT_SCORE_MAP[attitude]

    result = compute_priority(p_conversion, attitude)

    assert result == pytest.approx(expected)


def test_compute_priority_uses_settings_defaults() -> None:
    """When weights are omitted, the function reads them from settings."""
    # 0.5 P(conv) + objection (0.65) under 0.6/0.4 defaults
    expected = 0.6 * 0.5 + 0.4 * 0.65

    result = compute_priority(0.5, "objection")

    assert result == pytest.approx(expected)


def test_compute_priority_accepts_override_weights() -> None:
    """Explicit overrides take precedence over Settings."""
    # Heavy on sentiment: 0.3 conv + 0.7 sentiment
    expected = 0.3 * 0.8 + 0.7 * 1.0  # positive_engagement

    result = compute_priority(
        0.8,
        "positive_engagement",
        weight_conversion=0.3,
        weight_sentiment=0.7,
    )

    assert result == pytest.approx(expected)


def test_compute_priority_rejects_weights_not_summing_to_one() -> None:
    with pytest.raises(ValueError, match=r"must sum to 1\.0"):
        compute_priority(
            0.5,
            "neutral",
            weight_conversion=0.5,
            weight_sentiment=0.4,
        )


def test_compute_priority_rejects_negative_weight() -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        compute_priority(
            0.5,
            "neutral",
            weight_conversion=1.2,
            weight_sentiment=-0.2,
        )


@pytest.mark.parametrize("bad_p", [-0.01, 1.01, float("nan"), float("inf")])
def test_compute_priority_rejects_p_out_of_range(bad_p: float) -> None:
    with pytest.raises(ValueError, match=r"p_conversion must be in \[0, 1\]"):
        compute_priority(bad_p, "neutral")


def test_compute_priority_rejects_unknown_attitude() -> None:
    with pytest.raises(ValueError, match="not in score map"):
        compute_priority(0.5, "very_excited")


def test_compute_priority_accepts_custom_score_map() -> None:
    """Notebooks running mapping ablations can inject a custom dict."""
    custom_map = {
        "positive_engagement": 0.9,
        "objection": 0.4,
        "neutral": 0.3,
        "disengaged": 0.0,
    }
    expected = 0.6 * 0.5 + 0.4 * 0.4  # objection -> 0.4 under the custom map

    result = compute_priority(0.5, "objection", sentiment_score_map=custom_map)

    assert result == pytest.approx(expected)


def test_batch_compute_priority_matches_scalar() -> None:
    """Vectorised path must produce the same values as per-row calls."""
    p_arr = np.array([0.1, 0.5, 0.9, 0.25])
    attitudes = ["positive_engagement", "objection", "neutral", "disengaged"]

    batch_result = batch_compute_priority(p_arr, attitudes)
    scalar_result = np.array(
        [compute_priority(p, a) for p, a in zip(p_arr, attitudes, strict=True)]
    )

    assert batch_result.shape == (4,)
    np.testing.assert_allclose(batch_result, scalar_result)


def test_batch_compute_priority_validates_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        batch_compute_priority(np.array([0.5, 0.5]), ["objection"])


def test_batch_compute_priority_validates_attitude_membership() -> None:
    with pytest.raises(ValueError, match="unknown attitudes"):
        batch_compute_priority(
            np.array([0.5, 0.5]),
            ["objection", "very_excited"],
        )


def test_batch_compute_priority_validates_probability_range() -> None:
    with pytest.raises(ValueError, match=r"p_conversion must be in \[0, 1\]"):
        batch_compute_priority(np.array([0.5, 1.5]), ["objection", "neutral"])
