"""Serving-side feature engineering: derive + transform + persist API.

This package contains ONLY code needed at prediction time. EDA, training-time
fitting, and notebook exploration live under ``scripts/`` and ``notebooks/``.
The production Docker image carries ``src/`` only, so anything imported from
here must work without pandas-heavy EDA dependencies (no matplotlib, etc.).

Public API:

* :func:`derive_features` — pure ``raw_df → engineered_df`` step. Same call
  is used by the training script and by the serving transformer to guarantee
  train/serve symmetry.
* :class:`FeatureTransformer` — wraps the joblib-persisted sklearn pipeline
  and exposes ``load(path) / transform(raw_df) → np.ndarray``.
"""

from __future__ import annotations

from lead_priority.features.derive import derive_features
from lead_priority.features.pipeline import FeatureTransformer, build_pipeline

__all__ = ["FeatureTransformer", "build_pipeline", "derive_features"]
