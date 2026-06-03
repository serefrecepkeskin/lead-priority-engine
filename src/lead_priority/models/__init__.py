"""Serving-side model wrappers loaded by the FastAPI runtime.

Only predict-time code lives here. See ``scripts/train_lead_scoring.py`` for
the training-time CLI that produces the joblib bundles consumed by these
wrappers.
"""

from __future__ import annotations

from lead_priority.models.lead_scoring import (
    ALLOWED_MODEL_KINDS,
    SCHEMA_VERSION,
    LeadScoringModel,
)

__all__ = ["ALLOWED_MODEL_KINDS", "SCHEMA_VERSION", "LeadScoringModel"]
