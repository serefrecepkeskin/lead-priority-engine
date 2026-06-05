#!/usr/bin/env python3
"""CLI: fit the shared feature pipeline + persist artifacts.

Reads the raw Lead Scoring CSV, runs ``derive_features`` over the full
table, fits the sklearn pipeline (``build_pipeline``), and writes two
artifacts:

* ``artifacts/feature_pipeline.joblib`` — gitignored, contains the fitted
  pipeline + the feature-name list + a schema version.
* ``artifacts/feature_summary.json`` — tracked (lightweight), enumerates
  every output feature plus its source column / kind, and the drop-set
  rationale (leakage, PII, dead booleans, marketing-channel merge).

Phase 2 loads the joblib bundle once and feeds the same fitted pipeline
to both Logistic Regression and LightGBM.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from lead_priority.core.features import build_pipeline, derive_features
from lead_priority.core.features.constants import (
    BASE_NUMERIC,
    DEAD_BOOLEANS,
    DERIVED_NUMERIC,
    ID_PII_DROP,
    LEAKAGE_DROP,
    NEAR_ZERO_BOOLEANS,
    PASSTHROUGH_BINARIES,
    SEED,
)
from lead_priority.core.features.pipeline import SCHEMA_VERSION, FeatureTransformer
from lead_priority.settings import REPO_ROOT

logger = logging.getLogger("fit_feature_pipeline")

DEFAULT_INPUT = REPO_ROOT / "data" / "Lead Scoring.csv"
DEFAULT_PIPELINE = REPO_ROOT / "artifacts" / "feature_pipeline.joblib"
DEFAULT_SUMMARY = REPO_ROOT / "artifacts" / "feature_summary.json"


_BASE_SET = set(BASE_NUMERIC)
_RATIO_SET = set(DERIVED_NUMERIC)
_PASS_SET = set(PASSTHROUGH_BINARIES)
_LAST_ACTIVITY_DERIVED = {"is_high_intent_activity", "is_negative_activity"}


def _classify_feature(name: str) -> tuple[str, str]:  # noqa: PLR0911
    """Return (kind, source_column) tags for a pipeline output column."""
    if name in _BASE_SET:
        return "numeric_clipped", name
    if name in _RATIO_SET:
        return "numeric_ratio", "derived"
    if name == "country_is_india":
        return "binary_derived", "Country"
    if name in _LAST_ACTIVITY_DERIVED:
        return "binary_derived", "Last Activity"
    if name in _PASS_SET:
        return "binary_passthrough", name
    if name.startswith("missingindicator_"):
        return "missing_indicator", name.removeprefix("missingindicator_")
    # One-hot column: longest matching categorical prefix wins.
    for sep_idx in range(1, len(name)):
        head = name[:sep_idx]
        if head in _CATEGORICAL_NAMES_LOOKUP:
            tail = name[sep_idx + 1 :]
            kind = "onehot_rare" if tail == "infrequent_sklearn" else "onehot"
            return kind, head
    return "onehot", re.sub(r"_[^_]+$", "", name)


# Pre-computed set so _classify_feature is O(k) instead of O(n*k).
from lead_priority.core.features.constants import CATEGORICAL_ONE_HOT  # noqa: E402

_CATEGORICAL_NAMES_LOOKUP: set[str] = set(CATEGORICAL_ONE_HOT)


def _build_summary(feature_names: list[str], n_rows: int) -> dict[str, Any]:
    features: list[dict[str, str]] = []
    for name in feature_names:
        kind, source = _classify_feature(name)
        features.append(
            {
                "name": name,
                "source_column": source,
                "kind": kind,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "seed": SEED,
        "n_rows_fit": n_rows,
        "n_features": len(feature_names),
        "features": features,
        "dropped_columns": {
            "leakage": list(LEAKAGE_DROP),
            "pii": list(ID_PII_DROP),
            "dead_bool": list(DEAD_BOOLEANS),
            "merged_into_channel_diversity": list(NEAR_ZERO_BOOLEANS),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_PIPELINE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    if not args.input.exists():
        raise SystemExit(f"input CSV not found: {args.input}")

    logger.info("reading %s", args.input)
    raw = pd.read_csv(args.input)
    logger.info("raw shape: %s", raw.shape)

    derived = derive_features(raw)
    logger.info("derived shape: %s", derived.shape)

    pipeline = build_pipeline()
    pipeline.fit(derived)

    column_transformer = pipeline.named_steps["features"]
    feature_names = list(column_transformer.get_feature_names_out())
    logger.info("fitted pipeline output features: %d", len(feature_names))

    transformer = FeatureTransformer(
        pipeline=pipeline,
        feature_names=feature_names,
    )
    transformer.save(args.out)
    logger.info("wrote pipeline → %s", args.out)

    summary = _build_summary(feature_names, n_rows=len(derived))
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    logger.info("wrote summary → %s", args.summary)

    return 0


if __name__ == "__main__":
    sys.exit(main())
