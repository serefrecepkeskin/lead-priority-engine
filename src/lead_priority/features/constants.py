"""Column lists, drop-sets, activity buckets — single source of truth.

All lists below are validated in
``notebooks/1_eda_and_feature_engineering.ipynb`` (variance checks, missing
audits, leakage report cross-references). Keeping them in one module means
the fit script, the serving transformer, the tests, and the notebook all
agree on what a "feature" is.
"""

from __future__ import annotations

from typing import Final

SEED: Final = 42

# -- Raw column groups --------------------------------------------------------

ID_PII_DROP: Final[tuple[str, ...]] = ("Prospect ID", "Lead Number")
"""IDs / PII. Never enter the model."""

LEAKAGE_DROP: Final[tuple[str, ...]] = ("Tags", "Lead Quality", "Last Notable Activity")
"""Flagged by Phase 0 (artifacts/leakage_report.json) with single-column
AUC 0.92 / 0.80 / 0.70 respectively — populated after the lead outcome is
known in the source CRM. Must not leak into the scoring model."""

DEAD_BOOLEANS: Final[tuple[str, ...]] = (
    "Magazine",
    "Receive More Updates About Our Courses",
    "Update me on Supply Chain Content",
    "Get updates on DM Content",
    "I agree to pay the amount through cheque",
)
"""Variance-0 boolean columns (all 9,240 rows are 'No'). Carry no
information for either LR or LGBM and only add noise/regularization
surface — dropped."""

NEAR_ZERO_BOOLEANS: Final[tuple[str, ...]] = (
    "Newspaper Article",
    "X Education Forums",
    "Newspaper",
    "Digital Advertisement",
    "Search",
)
"""Marketing-channel binaries with 1–14 'Yes' total each. Individually
they noise the LR side; summed they form ``channel_diversity_count``
(see derive.py) which keeps the case-study "how many channels did they
hear from" signal."""

# -- Numeric columns ----------------------------------------------------------

BASE_NUMERIC: Final[tuple[str, ...]] = (
    "TotalVisits",
    "Total Time Spent on Website",
    "Page Views Per Visit",
    "Asymmetrique Activity Score",
    "Asymmetrique Profile Score",
)
"""Raw numeric inputs. Go through clip (95th percentile) → median impute
(``add_indicator=True``) → StandardScaler."""

DERIVED_NUMERIC: Final[tuple[str, ...]] = (
    "total_time_per_visit",
    "channel_diversity_count",
)
"""Numeric features computed in :func:`derive_features`. No clip needed
(``total_time_per_visit`` denom-clamped, ``channel_diversity_count`` is
bounded {0..5}). Go through median impute → StandardScaler."""

# -- Categorical columns ------------------------------------------------------

SELECT_COLUMNS: Final[tuple[str, ...]] = (
    "Specialization",
    "How did you hear about X Education",
    "Lead Profile",
    "City",
)
"""Columns where the literal string 'Select' is a placeholder for "user
did not pick". Replaced with NaN by :class:`SelectToNaN`, then bucketed
into 'Unknown' by the imputer."""

CATEGORICAL_ONE_HOT: Final[tuple[str, ...]] = (
    "Lead Origin",
    "Lead Source",
    "Last Activity",
    "Specialization",
    "What is your current occupation",
    "What matters most to you in choosing a course",
    "How did you hear about X Education",
    "Lead Profile",
    "City",
    "Asymmetrique Activity Index",
    "Asymmetrique Profile Index",
)
"""Categorical columns one-hot encoded with ``handle_unknown='ignore'``
and ``min_frequency=20`` (Specialization long tail collapsed into
``infrequent_sklearn``)."""

# -- Binary pass-through columns ---------------------------------------------

RAW_YES_NO_BINARIES: Final[tuple[str, ...]] = (
    "Do Not Email",
    "Do Not Call",
    "A free copy of Mastering The Interview",
    "Through Recommendations",
)
"""Binary Yes/No columns kept as-is (mapped to 0/1 by derive_features so
the pipeline sees a numeric pass-through column)."""

DERIVED_BINARIES: Final[tuple[str, ...]] = (
    "country_is_india",
    "is_high_intent_activity",
    "is_negative_activity",
)
"""Binaries created in :func:`derive_features`."""

PASSTHROUGH_BINARIES: Final[tuple[str, ...]] = (
    *RAW_YES_NO_BINARIES,
    "is_high_intent_activity",
    "is_negative_activity",
)
"""All binary 0/1 columns the ColumnTransformer should pass through
unchanged. ``country_is_india`` is in its own ColumnTransformer slot for
clarity."""

# -- Activity bucket sets -----------------------------------------------------

HIGH_INTENT_ACTIVITIES: Final[frozenset[str]] = frozenset(
    {
        "SMS Sent",
        "Had a Phone Conversation",
        "Approached upfront",
        "Form Submitted on Website",
    }
)
"""Values of ``Last Activity`` that signal active sales engagement.
``is_high_intent_activity = 1`` when Last Activity falls in this set."""

NEGATIVE_ACTIVITIES: Final[frozenset[str]] = frozenset(
    {
        "Email Bounced",
        "Unsubscribed",
        "Unreachable",
    }
)
"""Values of ``Last Activity`` that signal a dead or hostile channel.
``is_negative_activity = 1`` when Last Activity falls in this set."""

# -- Schema contracts ---------------------------------------------------------

REQUIRED_RAW_COLUMNS: Final[tuple[str, ...]] = (
    "TotalVisits",
    "Total Time Spent on Website",
    "Page Views Per Visit",
    "Asymmetrique Activity Score",
    "Asymmetrique Profile Score",
    "Asymmetrique Activity Index",
    "Asymmetrique Profile Index",
    "Lead Origin",
    "Lead Source",
    "Last Activity",
    "Specialization",
    "What is your current occupation",
    "What matters most to you in choosing a course",
    "How did you hear about X Education",
    "Lead Profile",
    "City",
    "Country",
    "Do Not Email",
    "Do Not Call",
    "A free copy of Mastering The Interview",
    "Through Recommendations",
    *NEAR_ZERO_BOOLEANS,
)
"""Columns the raw DataFrame MUST contain when ``derive_features`` is
called. Validated at the top of :func:`derive_features` so a serving-time
schema drift fails loudly rather than silently filling NaNs."""

REQUIRED_DERIVED_COLUMNS: Final[tuple[str, ...]] = (
    *BASE_NUMERIC,
    *DERIVED_NUMERIC,
    *CATEGORICAL_ONE_HOT,
    "country_is_india",
    *PASSTHROUGH_BINARIES,
)
"""Stable column order of the DataFrame returned by ``derive_features``.
Used by the fit script to feed the ColumnTransformer with deterministic
positional input."""


__all__ = [
    "BASE_NUMERIC",
    "CATEGORICAL_ONE_HOT",
    "DEAD_BOOLEANS",
    "DERIVED_BINARIES",
    "DERIVED_NUMERIC",
    "HIGH_INTENT_ACTIVITIES",
    "ID_PII_DROP",
    "LEAKAGE_DROP",
    "NEAR_ZERO_BOOLEANS",
    "NEGATIVE_ACTIVITIES",
    "PASSTHROUGH_BINARIES",
    "RAW_YES_NO_BINARIES",
    "REQUIRED_DERIVED_COLUMNS",
    "REQUIRED_RAW_COLUMNS",
    "SEED",
    "SELECT_COLUMNS",
]
