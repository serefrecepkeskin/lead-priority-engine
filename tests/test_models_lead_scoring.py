"""Unit tests for the serving-side ``LeadScoringModel`` wrapper.

The training-time CLI (``scripts/train_lead_scoring.py``) writes joblib bundles
that the FastAPI runtime later loads through this wrapper. The tests below
exercise the wrapper directly with a tiny in-process fit so they stay fast and
do not depend on the full GridSearchCV run.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from lead_priority.core.features import FeatureTransformer, build_pipeline, derive_features
from lead_priority.core.inference.lead_scoring import SCHEMA_VERSION, LeadScoringModel
from lead_priority.settings import REPO_ROOT

RAW_CSV = REPO_ROOT / "data" / "Lead Scoring.csv"


@pytest.fixture(scope="module")
def raw_sample() -> pd.DataFrame:
    """Stratified-ish 400-row sample so both classes are present for a tiny fit."""
    df = pd.read_csv(RAW_CSV)
    return df.sample(n=400, random_state=42).reset_index(drop=True)


@pytest.fixture(scope="module")
def fitted_transformer(raw_sample: pd.DataFrame) -> FeatureTransformer:
    derived = derive_features(raw_sample)
    pipeline = build_pipeline().fit(derived)
    feature_names = list(pipeline.named_steps["features"].get_feature_names_out())
    return FeatureTransformer(pipeline=pipeline, feature_names=feature_names)


@pytest.fixture(scope="module")
def fitted_model(
    raw_sample: pd.DataFrame, fitted_transformer: FeatureTransformer
) -> LeadScoringModel:
    x_array = fitted_transformer.transform(raw_sample)
    x = pd.DataFrame(x_array, columns=fitted_transformer.feature_names)
    y = raw_sample["Converted"].to_numpy().astype(int)
    classifier = LogisticRegression(max_iter=1000, random_state=42).fit(x, y)
    return LeadScoringModel(
        classifier=classifier,
        model_kind="logistic_regression",
        feature_names=fitted_transformer.feature_names,
    )


def test_save_load_roundtrip_preserves_metadata(
    fitted_model: LeadScoringModel, tmp_path: Path
) -> None:
    bundle_path = tmp_path / "lead_scoring_lr.joblib"
    fitted_model.save(bundle_path)

    reloaded = LeadScoringModel.load(bundle_path)
    assert reloaded.model_kind == fitted_model.model_kind
    assert reloaded.feature_names == fitted_model.feature_names
    assert reloaded.schema_version == SCHEMA_VERSION


def test_predict_proba_shape_and_range(
    fitted_model: LeadScoringModel,
    fitted_transformer: FeatureTransformer,
    raw_sample: pd.DataFrame,
) -> None:
    x = fitted_transformer.transform(raw_sample)
    scores = fitted_model.predict_proba(x)
    assert scores.shape == (len(raw_sample),)
    assert not np.isnan(scores).any()
    assert not np.isinf(scores).any()
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0


def test_serving_single_row_smoke(
    fitted_model: LeadScoringModel,
    fitted_transformer: FeatureTransformer,
    raw_sample: pd.DataFrame,
) -> None:
    """End-to-end serving path: one raw row → one calibrated score."""
    one_raw = raw_sample.head(1).copy()
    x = fitted_transformer.transform(one_raw)
    scores = fitted_model.predict_proba(x)
    assert scores.shape == (1,)
    assert 0.0 <= float(scores[0]) <= 1.0


def test_load_rejects_future_schema_version(fitted_model: LeadScoringModel, tmp_path: Path) -> None:
    bundle_path = tmp_path / "future.joblib"
    joblib.dump(
        {
            "classifier": fitted_model.classifier,
            "model_kind": fitted_model.model_kind,
            "feature_names": list(fitted_model.feature_names),
            "schema_version": SCHEMA_VERSION + 1,
        },
        bundle_path,
    )
    with pytest.raises(ValueError, match="schema_version"):
        LeadScoringModel.load(bundle_path)


def test_load_rejects_unknown_model_kind(fitted_model: LeadScoringModel, tmp_path: Path) -> None:
    bundle_path = tmp_path / "bogus.joblib"
    joblib.dump(
        {
            "classifier": fitted_model.classifier,
            "model_kind": "random_forest",
            "feature_names": list(fitted_model.feature_names),
            "schema_version": SCHEMA_VERSION,
        },
        bundle_path,
    )
    with pytest.raises(ValueError, match="model_kind"):
        LeadScoringModel.load(bundle_path)
