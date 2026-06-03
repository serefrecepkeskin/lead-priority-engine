"""Serving-side lead scoring model: predict-only wrapper around a trained classifier.

This module ONLY contains the code that the FastAPI service runs at prediction
time: load a joblib bundle, validate its schema, and call ``predict_proba`` on
the already-fitted classifier. Training, hyperparameter search, evaluation,
plotting, and notebook exploration live under ``scripts/`` and ``notebooks/``
and must never be imported from here — otherwise the production image would
have to carry matplotlib / sklearn.model_selection / etc.

The wrapper mirrors :class:`lead_priority.features.pipeline.FeatureTransformer`:
a dataclass-shaped bundle holding the fitted estimator, the feature-name list
used at training time (for input-shape validation), and a schema version so
loaders fail loudly on bundle drift.

Serving flow::

    transformer = FeatureTransformer.load(...)
    scorer = LeadScoringModel.load(...)
    x = transformer.transform(raw_df)  # (n, 104) ndarray
    scores = scorer.predict_proba(x)  # (n,) ndarray in [0, 1]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

SCHEMA_VERSION = 1
"""Bumped whenever the bundle layout changes. Loaders refuse newer bundles."""

ALLOWED_MODEL_KINDS = frozenset({"logistic_regression", "lightgbm"})
"""Whitelist of model_kind values written by ``scripts/train_lead_scoring.py``.
Catches accidental loads of unrelated joblib files at the boundary."""


@dataclass
class LeadScoringModel:
    """Serving-time wrapper around a fitted lead-scoring classifier.

    Both the Logistic Regression baseline and the LightGBM model are persisted
    under this wrapper; the production runtime treats them identically (call
    ``predict_proba`` and read column 1). The ``model_kind`` tag exists for
    observability and to let downstream consumers branch on the underlying
    estimator if they ever need to.
    """

    classifier: Any
    model_kind: str
    feature_names: list[str]
    schema_version: int = SCHEMA_VERSION

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return the positive-class probability for each row of ``x``.

        The input is wrapped into a :class:`pandas.DataFrame` with the
        training-time feature names before being handed to the underlying
        classifier. This keeps train/serve symmetric when the classifier
        was fit on a DataFrame (which is required to silence the LightGBM
        feature-name warning and to make ``feature_importances_`` keyed by
        readable names).

        Args:
            x: Feature matrix shaped ``(n_samples, n_features)``, produced by
                :meth:`lead_priority.features.FeatureTransformer.transform`.

        Returns:
            One-dimensional ndarray of length ``n_samples`` with values in
            ``[0, 1]`` representing ``P(Converted=1 | x)``.
        """
        frame = pd.DataFrame(np.asarray(x), columns=self.feature_names)
        proba = self.classifier.predict_proba(frame)
        return np.asarray(proba)[:, 1]

    def save(self, path: Path | str) -> None:
        """Persist the bundle to ``path`` via joblib."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "classifier": self.classifier,
                "model_kind": self.model_kind,
                "feature_names": list(self.feature_names),
                "schema_version": self.schema_version,
            },
            out_path,
        )

    @classmethod
    def load(cls, path: Path | str) -> LeadScoringModel:
        """Load a persisted ``LeadScoringModel`` bundle.

        Args:
            path: Path to the ``.joblib`` file produced by :meth:`save`
                (or ``scripts/train_lead_scoring.py``).

        Returns:
            Ready-to-call ``LeadScoringModel``.

        Raises:
            ValueError: If the bundle ``schema_version`` is newer than this
                code understands, or the ``model_kind`` tag is not one of
                :data:`ALLOWED_MODEL_KINDS`.
        """
        bundle: dict[str, Any] = joblib.load(Path(path))
        version = int(bundle.get("schema_version", 0))
        if version > SCHEMA_VERSION:
            raise ValueError(
                f"lead_scoring model joblib schema_version {version} is newer "
                f"than this code (SCHEMA_VERSION={SCHEMA_VERSION}). "
                "Upgrade lead_priority before loading."
            )
        kind = str(bundle["model_kind"])
        if kind not in ALLOWED_MODEL_KINDS:
            raise ValueError(
                f"unexpected model_kind {kind!r}; allowed kinds: {sorted(ALLOWED_MODEL_KINDS)}"
            )
        return cls(
            classifier=bundle["classifier"],
            model_kind=kind,
            feature_names=list(bundle["feature_names"]),
            schema_version=version,
        )


__all__ = ["ALLOWED_MODEL_KINDS", "SCHEMA_VERSION", "LeadScoringModel"]
