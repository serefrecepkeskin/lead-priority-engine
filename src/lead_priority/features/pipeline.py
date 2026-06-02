"""Single shared sklearn pipeline + serving-time wrapper.

The same fitted pipeline serves both Logistic Regression and LightGBM in
Phase 2:

* **LR** needs scaled imputed numerics + one-hot categoricals →
  ``num_clip_pipe`` and ``cat_pipe`` cover this.
* **LGBM** can handle NaN natively, but here imputation runs first, so
  the missingness signal is recovered via
  ``SimpleImputer(add_indicator=True)`` flags. Particularly load-bearing
  for the Asymmetrique 45.65% MAR cohort.

The pipeline is fitted ONCE on the training split and the same fitted
estimator is used across every CV evaluation in Phase 2 — never re-fit
per fold.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from lead_priority.features.constants import (
    BASE_NUMERIC,
    CATEGORICAL_ONE_HOT,
    DERIVED_NUMERIC,
    PASSTHROUGH_BINARIES,
)
from lead_priority.features.derive import derive_features
from lead_priority.features.transformers import PercentileClipper, SelectToNaN

SCHEMA_VERSION = 1
"""Bumped whenever the pipeline structure or the serialized bundle layout
changes. Loaders compare against this and refuse to load older bundles."""


def build_pipeline() -> Pipeline:
    """Construct the unfitted shared feature pipeline.

    Returns:
        Unfitted :class:`sklearn.pipeline.Pipeline` ready to be
        ``.fit(derive_features(raw_df))``-ed.
    """
    num_clip_pipe = Pipeline(
        [
            ("clip", PercentileClipper(q=0.95)),
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    num_ratio_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=False)),
            ("scale", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        [
            ("select_to_nan", SelectToNaN()),
            ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
            (
                "ohe",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    min_frequency=20,
                ),
            ),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("num_clipped", num_clip_pipe, list(BASE_NUMERIC)),
            ("num_ratio", num_ratio_pipe, list(DERIVED_NUMERIC)),
            ("cat", cat_pipe, list(CATEGORICAL_ONE_HOT)),
            ("country", "passthrough", ["country_is_india"]),
            ("binaries", "passthrough", list(PASSTHROUGH_BINARIES)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Pipeline([("features", column_transformer)])


@dataclass
class FeatureTransformer:
    """Serving-time wrapper around a fitted pipeline + feature-name list.

    Created by ``scripts/fit_feature_pipeline.py`` and reloaded inside
    the FastAPI service via :meth:`load`. The wrapper exists (rather than
    exposing the bare sklearn pipeline) so that:

    * ``derive_features`` is always called before the sklearn graph,
    * the feature-name list and schema version travel together with the
      pipeline weights in a single bundle,
    * the load path validates the schema version up front and fails loudly
      on bundle drift.
    """

    pipeline: Pipeline
    feature_names: list[str]
    schema_version: int = SCHEMA_VERSION

    def transform(self, raw_df: pd.DataFrame) -> np.ndarray:
        """Run ``derive_features`` then the sklearn pipeline."""
        derived = derive_features(raw_df)
        result = self.pipeline.transform(derived)
        return np.asarray(result)

    @classmethod
    def load(cls, path: Path | str) -> FeatureTransformer:
        """Load a persisted ``FeatureTransformer`` bundle.

        Args:
            path: Path to the ``.joblib`` file produced by
                :meth:`save` / ``scripts/fit_feature_pipeline.py``.

        Returns:
            Ready-to-call ``FeatureTransformer``.

        Raises:
            ValueError: If the bundle schema version is newer than this
                code understands.
        """
        bundle: dict[str, Any] = joblib.load(Path(path))
        version = int(bundle.get("schema_version", 0))
        if version > SCHEMA_VERSION:
            raise ValueError(
                f"feature_pipeline.joblib schema_version {version} is newer "
                f"than this code (SCHEMA_VERSION={SCHEMA_VERSION}). "
                "Upgrade lead_priority before loading."
            )
        return cls(
            pipeline=bundle["pipeline"],
            feature_names=list(bundle["feature_names"]),
            schema_version=version,
        )

    def save(self, path: Path | str) -> None:
        """Persist the bundle to ``path`` via joblib."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "feature_names": list(self.feature_names),
                "schema_version": self.schema_version,
            },
            out_path,
        )


__all__ = ["SCHEMA_VERSION", "FeatureTransformer", "build_pipeline"]
