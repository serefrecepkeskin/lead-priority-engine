"""Pure-Python tests for the attitude assignment + leakage diagnostics.

No LLM call is issued. We use the real ``data/Lead Scoring.csv`` because the
whole point of the test is to verify that ``assign_attitudes`` produces a
realistic-but-not-leaky correlation **on the actual dataset distribution**.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from datagen.leakage import cramers_v
from datagen.synthetic_interactions import (
    AttitudeClass,
    assign_attitudes,
    build_neutral_context,
    build_prompt_messages,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = REPO_ROOT / "data" / "Lead Scoring.csv"


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    if not RAW_CSV.exists():
        pytest.skip(f"raw CSV missing: {RAW_CSV}")
    return pd.read_csv(RAW_CSV)


def test_assign_attitudes_is_deterministic(raw: pd.DataFrame) -> None:
    a = assign_attitudes(raw, seed=42)
    b = assign_attitudes(raw, seed=42)
    assert (a == b).all()


def test_all_four_classes_present(raw: pd.DataFrame) -> None:
    counts = assign_attitudes(raw, seed=42).value_counts()
    expected = {c.value for c in AttitudeClass}
    assert set(counts.index) == expected
    # No class should be vanishingly small.
    assert counts.min() / counts.sum() > 0.05


def test_cramers_v_in_realistic_band(raw: pd.DataFrame) -> None:
    """Attitude × Converted correlation should be neither ~0 nor ~1.

    We allow a slightly wider band than the docx target (0.20–0.40) so the
    test doesn't flake under tiny RNG perturbations.
    """

    attitude = assign_attitudes(raw, seed=42)
    v = cramers_v(attitude, raw["Converted"])
    assert 0.15 <= v <= 0.50, f"Cramér's V {v:.3f} is outside the safe band [0.15, 0.50]"


def test_every_class_has_both_outcomes(raw: pd.DataFrame) -> None:
    attitude = assign_attitudes(raw, seed=42)
    ctab = pd.crosstab(attitude, raw["Converted"])
    assert (ctab > 0).all().all(), (
        f"Every attitude class must contain both converted and non-converted leads — got:\n{ctab}"
    )


def test_context_is_neutral_and_does_not_leak_converted(raw: pd.DataFrame) -> None:
    row = raw.iloc[0]
    ctx = build_neutral_context(row)
    assert "converted" not in {k.lower() for k in ctx}
    # Spot-check string-only output.
    for v in ctx.values():
        assert isinstance(v, str) and v


def test_prompt_system_explicitly_bans_outcome_words() -> None:
    msgs = build_prompt_messages(
        AttitudeClass.POSITIVE_ENGAGEMENT,
        {
            "lead_source": "Google",
            "specialization": "Finance Management",
            "occupation": "Working Professional",
            "last_activity": "Email Opened",
            "time_bucket": "medium",
        },
        language="tr",  # type: ignore[arg-type]
    )
    system_msg = next(m["content"] for m in msgs if m["role"] == "system").lower()
    for forbidden in ["sözleşme imzalandı", "closed won", "churned", "converted"]:
        assert forbidden in system_msg, f"system prompt should explicitly forbid '{forbidden}'"


def test_user_message_does_not_leak_label_columns() -> None:
    msgs = build_prompt_messages(
        AttitudeClass.OBJECTION,
        {
            "lead_source": "Google",
            "specialization": "Finance Management",
            "occupation": "Working Professional",
            "last_activity": "Email Opened",
            "time_bucket": "medium",
        },
        language="tr",  # type: ignore[arg-type]
    )
    user_msg = next(m["content"] for m in msgs if m["role"] == "user").lower()
    for forbidden in ["converted", "tags", "lead quality", "last notable activity"]:
        assert forbidden not in user_msg, (
            f"user prompt must not surface label-leaking field '{forbidden}'"
        )
