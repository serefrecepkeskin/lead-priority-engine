"""Unit tests for the pure ``derive_features`` step."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lead_priority.core.features.constants import (
    REQUIRED_DERIVED_COLUMNS,
    REQUIRED_RAW_COLUMNS,
)
from lead_priority.core.features.derive import derive_features


def _make_raw_row(**overrides: object) -> dict[str, object]:
    """Build a single-row raw payload covering every required column."""
    base: dict[str, object] = dict.fromkeys(REQUIRED_RAW_COLUMNS, np.nan)
    base.update(
        {
            "TotalVisits": 5.0,
            "Total Time Spent on Website": 1234,
            "Page Views Per Visit": 2.5,
            "Asymmetrique Activity Score": 14.0,
            "Asymmetrique Profile Score": 16.0,
            "Asymmetrique Activity Index": "02.Medium",
            "Asymmetrique Profile Index": "02.Medium",
            "Lead Origin": "API",
            "Lead Source": "Google",
            "Last Activity": "Email Opened",
            "Specialization": "Finance Management",
            "What is your current occupation": "Working Professional",
            "What matters most to you in choosing a course": "Better Career Prospects",
            "How did you hear about X Education": "Online Search",
            "Lead Profile": "Potential Lead",
            "City": "Mumbai",
            "Country": "India",
            "Do Not Email": "No",
            "Do Not Call": "No",
            "A free copy of Mastering The Interview": "No",
            "Through Recommendations": "No",
            "Newspaper Article": "No",
            "X Education Forums": "No",
            "Newspaper": "No",
            "Digital Advertisement": "No",
            "Search": "No",
        }
    )
    base.update(overrides)
    return base


def test_derive_returns_required_columns_in_order() -> None:
    df = pd.DataFrame([_make_raw_row()])
    out = derive_features(df)
    assert list(out.columns) == list(REQUIRED_DERIVED_COLUMNS)


def test_derive_missing_required_column_raises() -> None:
    df = pd.DataFrame([_make_raw_row()]).drop(columns=["Lead Source"])
    with pytest.raises(KeyError, match="Lead Source"):
        derive_features(df)


def test_select_placeholder_becomes_nan() -> None:
    df = pd.DataFrame([_make_raw_row(Specialization="Select")])
    out = derive_features(df)
    assert pd.isna(out.loc[0, "Specialization"])


def test_country_is_india_binary() -> None:
    rows = [
        _make_raw_row(Country="India"),
        _make_raw_row(Country="United States"),
        _make_raw_row(Country=np.nan),
    ]
    out = derive_features(pd.DataFrame(rows))
    assert out["country_is_india"].tolist() == [1, 0, 0]


def test_high_intent_activity_buckets() -> None:
    rows = [
        _make_raw_row(**{"Last Activity": "SMS Sent"}),
        _make_raw_row(**{"Last Activity": "Had a Phone Conversation"}),
        _make_raw_row(**{"Last Activity": "Email Opened"}),
    ]
    out = derive_features(pd.DataFrame(rows))
    assert out["is_high_intent_activity"].tolist() == [1, 1, 0]


def test_negative_activity_buckets() -> None:
    rows = [
        _make_raw_row(**{"Last Activity": "Email Bounced"}),
        _make_raw_row(**{"Last Activity": "Unsubscribed"}),
        _make_raw_row(**{"Last Activity": "Page Visited on Website"}),
    ]
    out = derive_features(pd.DataFrame(rows))
    assert out["is_negative_activity"].tolist() == [1, 1, 0]


def test_total_time_per_visit_nan_safe() -> None:
    rows = [
        _make_raw_row(TotalVisits=np.nan, **{"Total Time Spent on Website": 500}),
        _make_raw_row(TotalVisits=0, **{"Total Time Spent on Website": 500}),
        _make_raw_row(TotalVisits=10, **{"Total Time Spent on Website": 500}),
    ]
    out = derive_features(pd.DataFrame(rows))
    ratios = out["total_time_per_visit"].tolist()
    assert not any(np.isinf(r) for r in ratios), f"Inf produced: {ratios}"
    # TotalVisits=NaN or 0 → denom clamped to 1 → ratio = 500
    assert ratios[0] == pytest.approx(500.0)
    assert ratios[1] == pytest.approx(500.0)
    # TotalVisits=10 → 500/10 = 50
    assert ratios[2] == pytest.approx(50.0)


def test_channel_diversity_count_sums_near_zero_booleans() -> None:
    row = _make_raw_row(
        **{
            "Newspaper Article": "Yes",
            "X Education Forums": "No",
            "Newspaper": "Yes",
            "Digital Advertisement": "No",
            "Search": "Yes",
        }
    )
    out = derive_features(pd.DataFrame([row]))
    assert int(out.loc[0, "channel_diversity_count"]) == 3


def test_dead_booleans_and_leakage_columns_are_dropped() -> None:
    raw = _make_raw_row()
    raw["Magazine"] = "No"
    raw["Tags"] = "Closed by Horizzon"
    raw["Lead Quality"] = "High in Relevance"
    raw["Last Notable Activity"] = "SMS Sent"
    raw["Prospect ID"] = "fake-id"
    df = pd.DataFrame([raw])
    out = derive_features(df)
    forbidden = {
        "Magazine",
        "Tags",
        "Lead Quality",
        "Last Notable Activity",
        "Prospect ID",
        "Newspaper Article",
        "X Education Forums",
        "Newspaper",
        "Digital Advertisement",
        "Search",
        "Country",
    }
    assert forbidden.isdisjoint(set(out.columns))


def test_yes_no_binaries_become_int_0_or_1() -> None:
    rows = [
        _make_raw_row(**{"Do Not Email": "Yes"}),
        _make_raw_row(**{"Do Not Email": "no"}),  # case-insensitive
        _make_raw_row(**{"Do Not Email": np.nan}),
    ]
    out = derive_features(pd.DataFrame(rows))
    assert out["Do Not Email"].tolist() == [1, 0, 0]
