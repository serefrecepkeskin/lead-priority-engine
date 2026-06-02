"""Custom sklearn transformers used by the feature pipeline.

These live under ``src/`` (not ``scripts/``) because joblib pickles
estimator instances by reference: when ``FeatureTransformer.load`` is
called at serving time, Python resolves the class via its fully qualified
module path. Moving or renaming these classes would invalidate every
previously persisted ``feature_pipeline.joblib`` bundle, so the location
is part of the bundle's contract.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PercentileClipper(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Clip each numeric column at the q-th percentile observed during fit.

    Only the upper tail is clipped — the lower bound is left untouched
    because behavioral engagement signals (visits, time on site, scores)
    are one-sidedly heavy-tailed in this dataset; a single outlier user
    with 5h on the site would dominate the StandardScaler otherwise.

    NaN values pass through unchanged so the downstream
    :class:`sklearn.impute.SimpleImputer` can fill them.
    """

    def __init__(self, q: float = 0.95) -> None:
        self.q = q

    def fit(self, X: Any, y: Any = None) -> PercentileClipper:  # noqa: ARG002, N803
        arr = _as_2d_array(X)
        # nanpercentile so NaN rows do not poison the threshold.
        self.clip_values_: np.ndarray = np.nanpercentile(arr, self.q * 100.0, axis=0)
        self.n_features_in_: int = arr.shape[1]
        return self

    def transform(self, X: Any) -> np.ndarray:  # noqa: N803
        arr = _as_2d_array(X).astype(float, copy=True)
        return np.asarray(np.minimum(arr, self.clip_values_))

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if input_features is None:
            return np.asarray([f"x{i}" for i in range(self.n_features_in_)])
        return np.asarray(list(input_features))


class SelectToNaN(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Replace the literal placeholder string ``'Select'`` with NaN.

    Stateless — there is nothing to learn. Sits before the categorical
    :class:`sklearn.impute.SimpleImputer` so that source-CSV rows where
    the user did not pick a value ('Select') are treated identically to
    true missing values and bucketed into ``'Unknown'``.
    """

    def __init__(self, placeholder: str = "Select") -> None:
        self.placeholder = placeholder

    def fit(self, X: Any, y: Any = None) -> SelectToNaN:  # noqa: ARG002, N803
        arr = _as_2d_array(X)
        self.n_features_in_ = arr.shape[1]
        return self

    def transform(self, X: Any) -> np.ndarray:  # noqa: N803
        arr = _as_2d_array(X).astype(object, copy=True)
        # ``SimpleImputer`` (default ``missing_values=np.nan``) recognizes
        # ``np.nan`` placeholders inside object arrays, but does NOT treat
        # ``None`` as missing — so we explicitly set ``np.nan`` here.
        mask = pd.isna(arr) | (arr == self.placeholder)
        arr[mask] = np.nan
        return arr

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if input_features is None:
            return np.asarray([f"x{i}" for i in range(self.n_features_in_)])
        return np.asarray(list(input_features))


def _as_2d_array(X: Any) -> np.ndarray:  # noqa: N803
    """Coerce DataFrame / 1D array / 2D array to a 2D numpy array."""
    if isinstance(X, pd.DataFrame):
        return X.to_numpy()
    arr = np.asarray(X)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


__all__ = ["PercentileClipper", "SelectToNaN"]
