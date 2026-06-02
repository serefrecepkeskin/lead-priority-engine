#!/usr/bin/env python3
"""CLI: generate synthetic CRM interaction notes for the lead-scoring dataset.

Reads ``data/Lead Scoring.csv``, assigns a *latent* attitude class to each
lead from observable behavioral features (NEVER from ``Converted``), then
calls Azure OpenAI ``gpt-4o-mini`` to write a 1–3 sentence note per lead.

Output (default): ``data/synthetic/interactions.{parquet,csv}``.

The Azure deployment, endpoint and key are read from ``.env`` via
``lead_priority.settings``. The ``reasoning_effort`` field is *intentionally*
not passed to the model: it is only valid for o1/o3/gpt-5 reasoning models and
will be rejected by ``gpt-4o-mini``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr

from datagen.synthetic_interactions import (
    DEFAULT_LANG_MIX,
    LanguageMode,
    run_generation,
)
from lead_priority.settings import REPO_ROOT, get_settings

logger = logging.getLogger("generate_interactions")

DEFAULT_INPUT = REPO_ROOT / "data" / "Lead Scoring.csv"
DEFAULT_OUT = REPO_ROOT / "data" / "synthetic" / "interactions"
SMOKE_OUT = REPO_ROOT / "data" / "synthetic" / "interactions_smoke"


def build_llm() -> AzureChatOpenAI:
    """Construct the Azure chat client from settings.

    ``reasoning_effort`` is omitted on purpose: it is only valid for reasoning
    models (o1/o3/gpt-5 family). The deployment we use is ``gpt-4o-mini``,
    which rejects it.
    """

    s = get_settings()
    missing = [
        name
        for name, value in {
            "AZURE_OPENAI_API_KEY": s.azure_openai_api_key,
            "AZURE_OPENAI_ENDPOINT": s.azure_openai_endpoint,
            "AZURE_OPENAI_DEPLOYMENT": s.azure_openai_deployment,
            "AZURE_OPENAI_API_VERSION": s.azure_openai_api_version,
        }.items()
        if not value
    ]
    if missing:
        raise SystemExit("Azure OpenAI settings missing in .env: " + ", ".join(missing))

    assert s.azure_openai_api_key is not None  # narrowed by check above
    kwargs: dict[str, Any] = {
        "api_key": SecretStr(s.azure_openai_api_key),
        "azure_endpoint": s.azure_openai_endpoint,
        "azure_deployment": s.azure_openai_deployment,
        "api_version": s.azure_openai_api_version,
        "temperature": 0.9,
    }
    if s.azure_openai_max_tokens is not None:
        kwargs["max_tokens"] = s.azure_openai_max_tokens
    if s.azure_openai_timeout is not None:
        kwargs["timeout"] = s.azure_openai_timeout

    return AzureChatOpenAI(**kwargs)


def parse_lang_mix(raw: str) -> dict[LanguageMode, float]:
    """Parse ``tr=0.5,en=0.2,mix=0.3`` form into a normalized dict."""

    parts = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    out: dict[LanguageMode, float] = {}
    for part in parts:
        if "=" not in part:
            raise argparse.ArgumentTypeError(f"Invalid lang-mix entry {part!r}; expected key=value")
        key, val = part.split("=", 1)
        try:
            mode = LanguageMode(key.strip().lower())
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Unknown language {key!r}; one of tr/en/mix") from exc
        out[mode] = float(val)
    total = sum(out.values())
    if total <= 0:
        raise argparse.ArgumentTypeError("lang-mix weights must sum > 0")
    return {k: v / total for k, v in out.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Raw CSV path (default: data/Lead Scoring.csv).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output path WITHOUT extension; .parquet + .csv are written.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=0,
        help="0 (default) ⇒ generate for every lead. Otherwise stratified sample.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="50-lead smoke run; overrides --n-samples and --out.",
    )
    parser.add_argument(
        "--lang-mix",
        type=parse_lang_mix,
        default=None,
        help="Comma-separated weights, e.g. tr=0.5,en=0.2,mix=0.3.",
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Parallel LLM calls (default 8). 1 disables threading.",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    n_samples = 50 if args.smoke else args.n_samples
    out_path = SMOKE_OUT if args.smoke else args.out
    lang_mix = args.lang_mix if args.lang_mix is not None else dict(DEFAULT_LANG_MIX)

    logger.info("loading %s", args.input)
    df = pd.read_csv(args.input)
    logger.info("loaded %d leads, %d columns", len(df), df.shape[1])

    llm = build_llm()
    logger.info("Azure deployment ready; starting generation (n=%s)", n_samples or len(df))

    out_df = run_generation(
        df,
        llm,
        out_path=out_path,
        seed=args.seed,
        n_samples=n_samples,
        lang_mix=lang_mix,
        progress=not args.no_progress,
        concurrency=args.concurrency,
    )

    by_attitude = out_df["attitude"].value_counts().to_dict()
    by_lang = out_df["language"].value_counts().to_dict()
    n_empty = int((out_df["text"].str.len() == 0).sum())
    logger.info("wrote %d rows", len(out_df))
    logger.info("attitude breakdown: %s", by_attitude)
    logger.info("language breakdown: %s", by_lang)
    logger.info("empty notes (LLM failures): %d", n_empty)
    return 0


if __name__ == "__main__":
    sys.exit(main())
