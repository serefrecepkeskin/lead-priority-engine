#!/usr/bin/env python3
"""CLI: train the lead-scoring LR baseline and LGBM model + persist artifacts.

Pipeline:

1. Load the raw Lead Scoring CSV.
2. Load the fitted feature pipeline from Phase 1 (``feature_pipeline.joblib``);
   the pipeline is NEVER re-fitted here so every CV fold sees the same
   pre-computed clip thresholds / OHE categories / scaler statistics.
3. Stratified 60/20/20 split on ``Converted`` with a fixed seed.
4. Tune Logistic Regression and LightGBM with ``GridSearchCV(cv=5,
   scoring="roc_auc")`` on the train portion only.
5. Report train / val / test ROC-AUC, PR-AUC, a confusion matrix at 0.5, plus
   a 10-bucket calibration curve and a top-{10, 20, 30}% cumulative-gain
   table on the test split.
6. Persist both models via :class:`LeadScoringModel` joblib bundles + a
   tracked ``lead_scoring_metrics.json`` summary.

The script writes the artifacts listed under ``--out-lr``, ``--out-lgbm``,
and ``--metrics`` and exits with code ``0`` on success.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split

from lead_priority.features import FeatureTransformer
from lead_priority.features.constants import SEED
from lead_priority.models import LeadScoringModel
from lead_priority.settings import REPO_ROOT

logger = logging.getLogger("train_lead_scoring")

DEFAULT_RAW_CSV = REPO_ROOT / "data" / "Lead Scoring.csv"
DEFAULT_PIPELINE = REPO_ROOT / "artifacts" / "feature_pipeline.joblib"
DEFAULT_OUT_LR = REPO_ROOT / "artifacts" / "lead_scoring_lr.joblib"
DEFAULT_OUT_LGBM = REPO_ROOT / "artifacts" / "lead_scoring_lgbm.joblib"
DEFAULT_METRICS = REPO_ROOT / "artifacts" / "lead_scoring_metrics.json"

METRICS_SCHEMA_VERSION = 2

BOOTSTRAP_ITERS = 1000
"""Number of stratified bootstrap resamples for the 95% CI on test metrics."""

THRESHOLD_SWEEP = (0.3, 0.4, 0.5, 0.6, 0.7)
"""Concrete decision thresholds reported for the LGBM / LR test predictions."""

# LightGBM refuses feature names that contain JSON-special characters.
# The shared pipeline emits at least one such name from the OneHotEncoder
# (``Specialization_Banking, Investment And Insurance``), so the training
# script replaces these characters with underscores before fitting. The
# sanitized names become the canonical feature_names stored on the
# LeadScoringModel bundle, and the serving wrapper applies the same names
# when wrapping incoming ndarrays.
_LGBM_UNSAFE = set('[]{}":,')


def sanitize_feature_names(names: list[str]) -> list[str]:
    return ["".join("_" if ch in _LGBM_UNSAFE else ch for ch in name) for name in names]


def cumulative_gain(y_true: np.ndarray, y_proba: np.ndarray, fraction: float) -> float:
    """Fraction of positives captured in the top ``fraction`` of ranked scores.

    Standard sales-lift definition: sort by predicted probability descending,
    take the top ``k = ceil(fraction * n)`` rows, divide the positives within
    that slice by the total positives.
    """
    order = np.argsort(-y_proba)
    k = int(np.ceil(fraction * len(y_true)))
    if k == 0 or y_true.sum() == 0:
        return 0.0
    return float(y_true[order[:k]].sum()) / float(y_true.sum())


def per_split_metrics(
    classifier: Any,
    x: pd.DataFrame,
    y: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    """Return ROC-AUC, PR-AUC, and the positive-class probability vector."""
    proba = classifier.predict_proba(x)[:, 1]
    roc = float(roc_auc_score(y, proba))
    pr = float(average_precision_score(y, proba))
    return roc, pr, proba


def calibration_records(
    y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10
) -> list[dict[str, float]]:
    """Compute a 10-bucket reliability diagram suitable for JSON storage."""
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="quantile")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    counts, _ = np.histogram(y_proba, bins=bins)
    records: list[dict[str, float]] = []
    for mp, fp, count in zip(mean_pred, frac_pos, counts, strict=False):
        records.append(
            {
                "bin_mean_pred": float(mp),
                "bin_frac_pos": float(fp),
                "count": int(count),
            }
        )
    return records


def confusion_at_threshold(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5
) -> dict[str, int]:
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def lift_table(y_true: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    return {
        "top_10": cumulative_gain(y_true, y_proba, 0.10),
        "top_20": cumulative_gain(y_true, y_proba, 0.20),
        "top_30": cumulative_gain(y_true, y_proba, 0.30),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_iters: int = BOOTSTRAP_ITERS,
    alpha: float = 0.05,
) -> dict[str, dict[str, float]]:
    """Stratified bootstrap 95% CI for test ROC-AUC and PR-AUC.

    Reports test-set sampling uncertainty without retraining: at each iteration
    a bootstrap sample of the same size is drawn with replacement, the metric
    is recomputed, and the empirical 2.5 / 97.5 percentiles bound the CI.
    """
    rng = np.random.default_rng(SEED)
    n = len(y_true)
    roc_scores: list[float] = []
    pr_scores: list[float] = []
    for _ in range(n_iters):
        idx = rng.integers(0, n, n)
        y_sample = y_true[idx]
        if len(np.unique(y_sample)) < 2:
            # Degenerate resample (all one class) — skip; both AUCs undefined.
            continue
        roc_scores.append(float(roc_auc_score(y_sample, y_proba[idx])))
        pr_scores.append(float(average_precision_score(y_sample, y_proba[idx])))
    lower_q = 100 * (alpha / 2)
    upper_q = 100 * (1 - alpha / 2)
    return {
        "roc_auc": {
            "lower": float(np.percentile(roc_scores, lower_q)),
            "upper": float(np.percentile(roc_scores, upper_q)),
            "n_iters": len(roc_scores),
        },
        "pr_auc": {
            "lower": float(np.percentile(pr_scores, lower_q)),
            "upper": float(np.percentile(pr_scores, upper_q)),
            "n_iters": len(pr_scores),
        },
    }


def paired_bootstrap_ci(
    y_true: np.ndarray,
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    n_iters: int = BOOTSTRAP_ITERS,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Paired bootstrap 95% CI for the test ROC-AUC difference (B − A).

    Even when individual model CIs overlap, the per-sample paired test can
    show that B strictly outperforms A on each resample. The CI is computed
    on the difference between the two models' ROC-AUC on the same bootstrap
    sample. If 0 is outside the CI, B's improvement over A is statistically
    significant at the (1 - alpha) confidence level.
    """
    rng = np.random.default_rng(SEED)
    n = len(y_true)
    diffs: list[float] = []
    for _ in range(n_iters):
        idx = rng.integers(0, n, n)
        y_sample = y_true[idx]
        if len(np.unique(y_sample)) < 2:
            continue
        auc_a = float(roc_auc_score(y_sample, proba_a[idx]))
        auc_b = float(roc_auc_score(y_sample, proba_b[idx]))
        diffs.append(auc_b - auc_a)
    lower_q = 100 * (alpha / 2)
    upper_q = 100 * (1 - alpha / 2)
    lower = float(np.percentile(diffs, lower_q))
    upper = float(np.percentile(diffs, upper_q))
    return {
        "point_estimate": float(np.mean(diffs)),
        "lower": lower,
        "upper": upper,
        "n_iters": len(diffs),
        "significant_at_95": bool(lower > 0 or upper < 0),
    }


def threshold_sweep(
    y_true: np.ndarray, y_proba: np.ndarray, thresholds: tuple[float, ...] = THRESHOLD_SWEEP
) -> list[dict[str, float]]:
    """Per-threshold operating point: precision, recall, predicted-positive count.

    Lets the docx and notebook quote concrete numbers like "at threshold 0.6
    the model flags 28% of leads with 78% precision", which makes the
    threshold-tuning section in §10 actionable instead of abstract.
    """
    n = len(y_true)
    records: list[dict[str, float]] = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        predicted_pos = int(y_pred.sum())
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        records.append(
            {
                "threshold": float(t),
                "predicted_positive_n": predicted_pos,
                "predicted_positive_rate": float(predicted_pos / n),
                "precision": precision,
                "recall": recall,
            }
        )
    return records


def summarize_model(
    classifier: Any,
    best_params: dict[str, Any],
    splits: dict[str, tuple[pd.DataFrame, np.ndarray]],
) -> dict[str, Any]:
    roc_auc: dict[str, float] = {}
    pr_auc: dict[str, float] = {}
    proba_by_split: dict[str, np.ndarray] = {}
    for name, (x_split, y_split) in splits.items():
        roc, pr, proba = per_split_metrics(classifier, x_split, y_split)
        roc_auc[name] = roc
        pr_auc[name] = pr
        proba_by_split[name] = proba

    y_test, proba_test = splits["test"][1], proba_by_split["test"]
    return {
        "best_params": best_params,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "test_ci_95": bootstrap_ci(y_test, proba_test),
        "confusion_matrix_at_0_5": confusion_at_threshold(y_test, proba_test),
        "threshold_sweep": threshold_sweep(y_test, proba_test),
        "lift": lift_table(y_test, proba_test),
        "calibration": calibration_records(y_test, proba_test),
    }


def train_logistic_regression(
    x_train: pd.DataFrame, y_train: np.ndarray
) -> tuple[LogisticRegression, dict[str, Any]]:
    logger.info("tuning Logistic Regression baseline (4-cell C grid × 5 folds)")
    base = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=SEED,
    )
    grid = GridSearchCV(
        base,
        param_grid={"C": [0.01, 0.1, 1.0, 10.0]},
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        refit=True,
    )
    grid.fit(x_train, y_train)
    logger.info("LR best params: %s (cv roc_auc=%.4f)", grid.best_params_, grid.best_score_)
    return grid.best_estimator_, dict(grid.best_params_)


def train_lightgbm(
    x_train: pd.DataFrame, y_train: np.ndarray
) -> tuple[LGBMClassifier, dict[str, Any]]:
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / pos if pos else 1.0
    logger.info("tuning LightGBM (12 cells × 5 folds, scale_pos_weight=%.3f)", scale_pos_weight)
    base = LGBMClassifier(
        objective="binary",
        scale_pos_weight=scale_pos_weight,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    grid = GridSearchCV(
        base,
        param_grid={
            "num_leaves": [15, 31, 63],
            "learning_rate": [0.05, 0.1],
            "n_estimators": [100, 300],
        },
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        refit=True,
    )
    grid.fit(x_train, y_train)
    logger.info("LGBM best params: %s (cv roc_auc=%.4f)", grid.best_params_, grid.best_score_)
    return grid.best_estimator_, dict(grid.best_params_)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-csv", type=Path, default=DEFAULT_RAW_CSV)
    parser.add_argument("--pipeline", type=Path, default=DEFAULT_PIPELINE)
    parser.add_argument("--out-lr", type=Path, default=DEFAULT_OUT_LR)
    parser.add_argument("--out-lgbm", type=Path, default=DEFAULT_OUT_LGBM)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    if not args.raw_csv.exists():
        raise SystemExit(f"raw CSV not found: {args.raw_csv}")
    if not args.pipeline.exists():
        raise SystemExit(
            f"feature pipeline joblib not found: {args.pipeline}. "
            "Run scripts/fit_feature_pipeline.py first."
        )

    logger.info("reading %s", args.raw_csv)
    raw = pd.read_csv(args.raw_csv)
    logger.info("raw shape: %s", raw.shape)

    transformer = FeatureTransformer.load(args.pipeline)
    logger.info("loaded feature pipeline (%d features)", len(transformer.feature_names))

    x_array = transformer.transform(raw)
    feature_names = sanitize_feature_names(transformer.feature_names)
    x = pd.DataFrame(x_array, columns=feature_names)
    y = raw["Converted"].to_numpy().astype(int)
    logger.info("feature matrix: %s, positive rate: %.4f", x.shape, float(y.mean()))

    # Stratified 60/20/20: first peel off test (20%), then val (25% of the
    # remaining 80% → 20% of the original).
    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x, y, test_size=0.20, stratify=y, random_state=SEED
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=SEED
    )
    logger.info("split sizes: train=%d, val=%d, test=%d", len(y_train), len(y_val), len(y_test))

    splits = {
        "train": (x_train, y_train),
        "val": (x_val, y_val),
        "test": (x_test, y_test),
    }

    lr_estimator, lr_best = train_logistic_regression(x_train, y_train)
    lgbm_estimator, lgbm_best = train_lightgbm(x_train, y_train)

    lr_summary = summarize_model(lr_estimator, lr_best, splits)
    lgbm_summary = summarize_model(lgbm_estimator, lgbm_best, splits)

    # Paired bootstrap: per-resample (LGBM ROC-AUC − LR ROC-AUC) on the same
    # test sample. Even if individual model CIs overlap, the paired test can
    # surface a consistent improvement that the marginal CIs hide.
    proba_lr_test = lr_estimator.predict_proba(x_test)[:, 1]
    proba_lgbm_test = lgbm_estimator.predict_proba(x_test)[:, 1]
    paired = paired_bootstrap_ci(y_test, proba_lr_test, proba_lgbm_test)
    logger.info(
        "paired LGBM-LR ROC-AUC diff: %.4f [%.4f, %.4f] significant=%s",
        paired["point_estimate"],
        paired["lower"],
        paired["upper"],
        paired["significant_at_95"],
    )

    LeadScoringModel(
        classifier=lr_estimator,
        model_kind="logistic_regression",
        feature_names=feature_names,
    ).save(args.out_lr)
    logger.info("wrote LR model → %s", args.out_lr)

    LeadScoringModel(
        classifier=lgbm_estimator,
        model_kind="lightgbm",
        feature_names=feature_names,
    ).save(args.out_lgbm)
    logger.info("wrote LGBM model → %s", args.out_lgbm)

    metrics = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "seed": SEED,
        "split": {
            "train_n": len(y_train),
            "val_n": len(y_val),
            "test_n": len(y_test),
            "positive_rate": float(y.mean()),
        },
        "feature_pipeline": {
            "n_features": len(feature_names),
            "schema_version": int(transformer.schema_version),
        },
        "models": {
            "logistic_regression": lr_summary,
            "lightgbm": lgbm_summary,
        },
        "model_comparison": {
            "paired_roc_auc_diff_lgbm_minus_lr": paired,
        },
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    logger.info("wrote metrics → %s", args.metrics)

    return 0


if __name__ == "__main__":
    sys.exit(main())
