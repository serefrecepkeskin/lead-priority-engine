#!/usr/bin/env python3
"""CLI: evaluate an OpenRouter ``:free`` LLM on the Phase 3 sentiment test set.

Reads ``data/synthetic/interactions.parquet``, deterministically carves a
test split (80/10/10 stratified on the joint ``attitude`` × ``language``
key, seed=42 — same protocol as Phase 2), and calls the supplied OpenRouter
model on every test note. Predictions are appended to
``artifacts/sentiment_predictions/{model_slug}_test.{parquet,csv}`` in batches
of 25 so a quota stall mid-run never loses progress: re-running the same
command resumes from the last cached ``lead_id``.

The CLI is intentionally separate from ``src/lead_priority/``: only the
``OpenRouterSentiment`` wrapper imports cleanly into the serving image, and
the eval loop carries pandas / pyarrow / tqdm that the API never needs.

Typical usage::

    python scripts/evaluate_openrouter_sentiment.py --model llama --limit 5
    python scripts/evaluate_openrouter_sentiment.py --model llama
    python scripts/evaluate_openrouter_sentiment.py --model qwen
    python scripts/evaluate_openrouter_sentiment.py --model deepseek

``--force-refresh`` deletes the existing cache file before running so a
fresh evaluation can be produced when the prompt or model alias changes.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from lead_priority.core.features.constants import SEED
from lead_priority.infra.openrouter.sentiment import (
    MODEL_ALIASES,
    OpenRouterError,
    OpenRouterRateLimitError,
    OpenRouterSentiment,
)
from lead_priority.settings import REPO_ROOT

logger = logging.getLogger("evaluate_openrouter_sentiment")

DEFAULT_INTERACTIONS = REPO_ROOT / "data" / "synthetic" / "interactions.parquet"
PREDICTIONS_DIR = REPO_ROOT / "artifacts" / "sentiment_predictions"

CHECKPOINT_EVERY = 25
"""Flush the predictions parquet + csv after this many new completions."""


def _model_slug(model_name: str) -> str:
    """Turn ``meta-llama/llama-3.3-70b-instruct:free`` into a safe filename stem."""
    base = model_name.split("/", maxsplit=1)[-1]
    base = base.split(":")[0]
    base = re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-")
    return base


def stratified_test_split(
    df: pd.DataFrame,
    seed: int = SEED,
) -> pd.DataFrame:
    """Two consecutive stratified splits → 80% train / 10% val / 10% test.

    Stratification key is ``attitude + "_" + language`` so the test fold
    preserves the joint distribution (4 classes × 3 languages = 12 strata).
    The function is deterministic for fixed ``seed`` so the notebook +
    builder reproduce the same 924 test rows the CLI scored.
    """
    if df.empty:
        raise ValueError("interactions parquet is empty")
    strata = df["attitude"].astype(str) + "_" + df["language"].astype(str)
    train_temp, test_part = train_test_split(
        df,
        test_size=0.10,
        random_state=seed,
        stratify=strata,
        shuffle=True,
    )
    # Second split: carve the val fold off train_temp so test stays untouched.
    train_strata = train_temp["attitude"].astype(str) + "_" + train_temp["language"].astype(str)
    _train, _val = train_test_split(
        train_temp,
        test_size=10.0 / 90.0,  # 10% of the full df = 1/9 of the 90% remainder
        random_state=seed,
        stratify=train_strata,
        shuffle=True,
    )
    return test_part.reset_index(drop=True)


def _output_paths(model_slug: str, output_override: Path | None) -> tuple[Path, Path]:
    """Return the (parquet, csv) destinations for the given model slug."""
    if output_override is not None:
        parquet_path = output_override
    else:
        parquet_path = PREDICTIONS_DIR / f"{model_slug}_test.parquet"
    csv_path = parquet_path.with_suffix(".csv")
    return parquet_path, csv_path


PREDICTION_COLUMNS = [
    "lead_id",
    "language",
    "true_attitude",
    "predicted_attitude",
    "model_name",
    "prompt_tokens",
    "completion_tokens",
    "latency_ms",
    "timestamp_utc",
]


def _load_existing_predictions(parquet_path: Path) -> pd.DataFrame:
    if not parquet_path.exists():
        return pd.DataFrame(columns=PREDICTION_COLUMNS)
    existing = pd.read_parquet(parquet_path)
    missing = [c for c in PREDICTION_COLUMNS if c not in existing.columns]
    if missing:
        raise ValueError(
            f"existing predictions at {parquet_path} missing columns {missing}; "
            "delete the file or run with --force-refresh"
        )
    return existing[PREDICTION_COLUMNS].copy()


def _merge_predictions(
    existing: pd.DataFrame, new_records: list[dict[str, object]]
) -> pd.DataFrame:
    """Append new records to ``existing`` without warning on empty bases.

    ``pd.concat`` of an empty typed frame with a populated frame raises a
    FutureWarning under modern pandas; sidestep it by returning the new frame
    directly when ``existing`` is empty.
    """
    new_df = pd.DataFrame(new_records, columns=PREDICTION_COLUMNS)
    if existing.empty:
        return new_df
    return pd.concat([existing, new_df], ignore_index=True)


def _flush(records: pd.DataFrame, parquet_path: Path, csv_path: Path) -> None:
    """Atomically write the predictions parquet + csv."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_parquet = parquet_path.with_suffix(parquet_path.suffix + ".tmp")
    tmp_csv = csv_path.with_suffix(csv_path.suffix + ".tmp")
    records.to_parquet(tmp_parquet, index=False)
    records.to_csv(tmp_csv, index=False)
    tmp_parquet.replace(parquet_path)
    tmp_csv.replace(csv_path)


def _now_utc_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def evaluate(
    *,
    model_name: str,
    interactions_path: Path,
    output_override: Path | None,
    limit: int | None,
    force_refresh: bool,
    seed: int,
) -> Path:
    """Run the evaluation loop and return the parquet output path.

    Errors propagate to ``main`` so the process exits non-zero on irrecoverable
    failures (e.g. quota stall after the full retry budget). Re-run the same
    command to resume from the cache.
    """
    df = pd.read_parquet(interactions_path)
    test_df = stratified_test_split(df, seed=seed)
    if limit is not None:
        test_df = test_df.head(limit).copy()
    logger.info(
        "loaded_test_split rows=%d strata=%d",
        len(test_df),
        (test_df["attitude"].astype(str) + "_" + test_df["language"].astype(str)).nunique(),
    )

    model_slug = _model_slug(model_name)
    parquet_path, csv_path = _output_paths(model_slug, output_override)

    if force_refresh:
        for path in (parquet_path, csv_path):
            if path.exists():
                logger.info("force_refresh removing existing %s", path)
                path.unlink()

    existing = _load_existing_predictions(parquet_path)
    completed_ids = set(existing["lead_id"].astype(str).tolist())
    pending = test_df[~test_df["lead_id"].astype(str).isin(completed_ids)].copy()
    logger.info(
        "resume cached=%d pending=%d total=%d",
        len(existing),
        len(pending),
        len(test_df),
    )
    if pending.empty:
        logger.info("nothing to do — all test rows already cached at %s", parquet_path)
        return parquet_path

    client = OpenRouterSentiment.from_settings(model_name=model_name)

    new_records: list[dict[str, object]] = []
    progress = tqdm(
        pending.itertuples(index=False),
        total=len(pending),
        desc=model_slug,
        unit="note",
    )
    try:
        for row in progress:
            lead_id = str(row.lead_id)
            text = str(row.text)
            try:
                label = client.predict(text)
            except OpenRouterRateLimitError:
                # Persist what we have, then surface the rate-limit so the
                # operator can resume after the daily cap resets.
                logger.warning(
                    "rate_limit_exhausted lead_id=%s — flushing %d new rows and exiting",
                    lead_id,
                    len(new_records),
                )
                if new_records:
                    combined = pd.concat(
                        [existing, pd.DataFrame(new_records, columns=PREDICTION_COLUMNS)],
                        ignore_index=True,
                    )
                    _flush(combined, parquet_path, csv_path)
                raise
            except OpenRouterError as exc:
                logger.error("skipping lead_id=%s due to OpenRouter error: %s", lead_id, exc)
                continue
            usage = client.last_usage()
            new_records.append(
                {
                    "lead_id": lead_id,
                    "language": str(row.language),
                    "true_attitude": str(row.attitude),
                    "predicted_attitude": label,
                    "model_name": model_name,
                    "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                    "completion_tokens": int(usage.get("completion_tokens", 0)),
                    "latency_ms": float(client.last_latency_ms()),
                    "timestamp_utc": _now_utc_iso(),
                }
            )
            if len(new_records) >= CHECKPOINT_EVERY:
                combined = _merge_predictions(existing, new_records)
                _flush(combined, parquet_path, csv_path)
                existing = combined
                new_records = []
    finally:
        progress.close()

    if new_records:
        combined = _merge_predictions(existing, new_records)
        _flush(combined, parquet_path, csv_path)

    logger.info("done parquet=%s csv=%s", parquet_path, csv_path)
    return parquet_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Either an alias ({}) or a fully-qualified OpenRouter model id "
            "such as 'meta-llama/llama-3.3-70b-instruct:free'."
        ).format(", ".join(sorted(MODEL_ALIASES.keys()))),
    )
    parser.add_argument(
        "--interactions",
        type=Path,
        default=DEFAULT_INTERACTIONS,
        help=f"Path to interactions parquet (default: {DEFAULT_INTERACTIONS}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Override the output parquet path. The csv mirror is written next "
            "to it (same stem, .csv extension)."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Truncate the test set to the first N rows (for smoke tests).",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Delete the cached predictions before running and start from scratch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed for the stratified split (default: {SEED}).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    model_name = MODEL_ALIASES.get(args.model, args.model)
    logger.info("starting evaluate_openrouter_sentiment model=%s", model_name)
    start = time.perf_counter()
    try:
        evaluate(
            model_name=model_name,
            interactions_path=args.interactions,
            output_override=args.output,
            limit=args.limit,
            force_refresh=args.force_refresh,
            seed=args.seed,
        )
    except OpenRouterRateLimitError:
        elapsed = time.perf_counter() - start
        logger.error(
            "exiting due to OpenRouter rate-limit after %.1fs — resume the same "
            "command once the quota resets",
            elapsed,
        )
        return 2
    elapsed = time.perf_counter() - start
    logger.info("completed in %.1fs", elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
