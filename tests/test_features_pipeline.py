"""Integration tests for the shared feature pipeline + persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lead_priority.core.features import FeatureTransformer, build_pipeline, derive_features
from lead_priority.core.features.constants import REQUIRED_RAW_COLUMNS
from lead_priority.settings import REPO_ROOT

RAW_CSV = REPO_ROOT / "data" / "Lead Scoring.csv"


@pytest.fixture(scope="module")
def raw_sample() -> pd.DataFrame:
    """200-row sample from the real dataset (fast, deterministic)."""
    df = pd.read_csv(RAW_CSV)
    return df.sample(n=200, random_state=42).reset_index(drop=True)


def test_pipeline_fit_transform_no_nan_no_inf(raw_sample: pd.DataFrame) -> None:
    derived = derive_features(raw_sample)
    pipeline = build_pipeline()
    x = pipeline.fit_transform(derived)
    assert x.shape[0] == 200
    assert not np.isnan(x).any(), "fit_transform produced NaN"
    assert not np.isinf(x).any(), "fit_transform produced Inf"


def test_pipeline_feature_count_stable_across_row_orderings(raw_sample: pd.DataFrame) -> None:
    derived_a = derive_features(raw_sample)
    derived_b = derive_features(raw_sample.sample(frac=1, random_state=7).reset_index(drop=True))
    pipe_a = build_pipeline().fit(derived_a)
    pipe_b = build_pipeline().fit(derived_b)
    names_a = list(pipe_a.named_steps["features"].get_feature_names_out())
    names_b = list(pipe_b.named_steps["features"].get_feature_names_out())
    assert names_a == names_b


def test_save_load_roundtrip_numerically_equal(raw_sample: pd.DataFrame, tmp_path: Path) -> None:
    derived = derive_features(raw_sample)
    pipeline = build_pipeline()
    pipeline.fit(derived)
    feature_names = list(pipeline.named_steps["features"].get_feature_names_out())

    transformer = FeatureTransformer(pipeline=pipeline, feature_names=feature_names)
    bundle_path = tmp_path / "feature_pipeline.joblib"
    transformer.save(bundle_path)

    reloaded = FeatureTransformer.load(bundle_path)
    assert reloaded.feature_names == feature_names

    # Numerical equality on the same 200-row sample.
    x_orig = pipeline.transform(derived)
    x_reload = reloaded.transform(raw_sample)
    np.testing.assert_allclose(np.asarray(x_orig), x_reload)


def test_serving_transform_handles_single_row(raw_sample: pd.DataFrame, tmp_path: Path) -> None:
    derived = derive_features(raw_sample)
    pipeline = build_pipeline()
    pipeline.fit(derived)
    feature_names = list(pipeline.named_steps["features"].get_feature_names_out())
    transformer = FeatureTransformer(pipeline=pipeline, feature_names=feature_names)

    single_row_raw = raw_sample.head(1).copy()
    x = transformer.transform(single_row_raw)
    assert x.shape == (1, len(feature_names))
    assert not np.isnan(x).any()


def test_serving_transform_tolerates_oov_categorical(
    raw_sample: pd.DataFrame, tmp_path: Path
) -> None:
    derived = derive_features(raw_sample)
    pipeline = build_pipeline()
    pipeline.fit(derived)
    feature_names = list(pipeline.named_steps["features"].get_feature_names_out())
    transformer = FeatureTransformer(pipeline=pipeline, feature_names=feature_names)

    one = raw_sample.head(1).copy()
    one["Lead Source"] = "BrandNewChannelNeverSeen"
    one["Specialization"] = "Astrology Engineering"  # not in fit data
    x = transformer.transform(one)
    assert x.shape == (1, len(feature_names))
    assert not np.isnan(x).any()


def test_required_raw_columns_present_in_real_csv() -> None:
    df = pd.read_csv(RAW_CSV, nrows=1)
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    assert not missing, f"raw CSV is missing required columns: {missing}"
