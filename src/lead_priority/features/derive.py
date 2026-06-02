"""Stateless feature derivation shared by training and serving.

``derive_features`` is the SAME entry point used by:

* ``scripts/fit_feature_pipeline.py`` (training): runs once over the full
  CSV before fitting the sklearn pipeline.
* :class:`lead_priority.features.pipeline.FeatureTransformer.transform`
  (serving): runs over a single-row DataFrame coming from FastAPI's JSON
  payload before the sklearn pipeline scores it.

Symmetry of these two call sites is the reason this lives under ``src/``:
divergence would cause silent train/serve drift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lead_priority.features.constants import (
    DEAD_BOOLEANS,
    HIGH_INTENT_ACTIVITIES,
    ID_PII_DROP,
    LEAKAGE_DROP,
    NEAR_ZERO_BOOLEANS,
    NEGATIVE_ACTIVITIES,
    RAW_YES_NO_BINARIES,
    REQUIRED_DERIVED_COLUMNS,
    REQUIRED_RAW_COLUMNS,
    SELECT_COLUMNS,
)


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a raw lead DataFrame to the engineered-feature DataFrame.

    Pure function: no fitting, no learned state, no I/O. Same call shape
    at training time (full 9,240 rows) and serving time (single-row
    payload from FastAPI).

    Steps:

    1. Validate schema against ``REQUIRED_RAW_COLUMNS``.
    2. Drop leakage, PII, and dead-boolean columns.
    3. Coerce Yes/No binaries to 0/1.
    4. Compute engineered numerics
       (``total_time_per_visit``, ``channel_diversity_count``).
    5. Compute engineered binaries
       (``country_is_india``, ``is_high_intent_activity``,
       ``is_negative_activity``).
    6. Drop merged-in raw marketing channels and the original ``Country``.
    7. Replace ``'Select'`` with NaN on the documented categorical
       columns (so the sklearn imputer can bucket them into ``'Unknown'``).
    8. Reorder columns deterministically.

    Args:
        df: Raw lead DataFrame (columns named per the source CSV).

    Returns:
        DataFrame with exactly ``REQUIRED_DERIVED_COLUMNS`` columns, in
        that order.

    Raises:
        KeyError: If any required raw column is missing.
    """
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"derive_features: raw DataFrame missing required columns: {missing}")

    out = df.copy()

    # -- 2. drop leakage / PII / dead booleans --------------------------------
    out = out.drop(columns=[c for c in ID_PII_DROP if c in out.columns], errors="ignore")
    out = out.drop(columns=[c for c in LEAKAGE_DROP if c in out.columns], errors="ignore")
    out = out.drop(columns=[c for c in DEAD_BOOLEANS if c in out.columns], errors="ignore")

    # -- 3. Yes/No binaries → 0/1 (case-insensitive, NaN → 0) -----------------
    for col in RAW_YES_NO_BINARIES:
        out[col] = _yes_no_to_int(out[col])
    for col in NEAR_ZERO_BOOLEANS:
        out[col] = _yes_no_to_int(out[col])

    # -- 4. engineered numerics ----------------------------------------------
    total_time = out["Total Time Spent on Website"].astype(float)
    visits = out["TotalVisits"].astype(float)
    # NaN-safe denominator: where TotalVisits is NaN or 0, fall back to
    # Total Time itself (treated as a 1-visit equivalent). Downstream
    # SimpleImputer handles any residual NaN in Total Time.
    safe_denom = visits.where(visits.notna() & (visits > 0), 1.0)
    out["total_time_per_visit"] = total_time / safe_denom

    out["channel_diversity_count"] = out[list(NEAR_ZERO_BOOLEANS)].sum(axis=1).astype("int8")

    # -- 5. engineered binaries ----------------------------------------------
    out["country_is_india"] = (
        out["Country"].astype("string").str.strip().eq("India").fillna(False).astype("int8")
    )
    last_activity = out["Last Activity"].astype("string")
    out["is_high_intent_activity"] = (
        last_activity.isin(HIGH_INTENT_ACTIVITIES).fillna(False).astype("int8")
    )
    out["is_negative_activity"] = (
        last_activity.isin(NEGATIVE_ACTIVITIES).fillna(False).astype("int8")
    )

    # -- 6. drop merged-in raw marketing channels and original Country -------
    out = out.drop(columns=[*NEAR_ZERO_BOOLEANS, "Country"], errors="ignore")

    # -- 7. 'Select' → NaN on documented columns -----------------------------
    # ``.where(cond, NaN)`` keeps the value when the predicate is True and
    # writes NaN otherwise — same behavior as ``replace`` but free of the
    # pandas downcasting FutureWarning triggered by ``replace`` on object
    # columns.
    for col in SELECT_COLUMNS:
        if col in out.columns:
            out[col] = out[col].where(out[col].ne("Select"), other=np.nan)

    # -- 8. deterministic column order ---------------------------------------
    return out[list(REQUIRED_DERIVED_COLUMNS)]


def _yes_no_to_int(series: pd.Series) -> pd.Series:
    """Map Yes/No (case-insensitive) to 1/0, NaN/anything-else to 0."""
    s = series.astype("string").str.strip().str.lower()
    return s.eq("yes").fillna(False).astype("int8")


__all__ = ["derive_features"]
