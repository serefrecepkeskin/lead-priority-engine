#!/usr/bin/env python3
"""CLI: leakage diagnostics for the synthetic attitude → Converted relationship.

Joins ``data/synthetic/interactions.parquet`` back to ``data/Lead Scoring.csv``
on ``Prospect ID`` and runs the diagnostics in
:mod:`datagen.leakage`. Writes ``artifacts/leakage_report.json``
(picked up by the methodology docx builder) and prints a stdout summary.

The notebook ``notebooks/leakage_analysis.ipynb`` uses the same module and so
produces identical numbers.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from datagen.leakage import compute_leakage_report
from lead_priority.settings import REPO_ROOT

logger = logging.getLogger("leakage_analysis")

DEFAULT_INTERACTIONS = REPO_ROOT / "data" / "synthetic" / "interactions.parquet"
DEFAULT_RAW = REPO_ROOT / "data" / "Lead Scoring.csv"
DEFAULT_REPORT = REPO_ROOT / "artifacts" / "leakage_report.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interactions", type=Path, default=DEFAULT_INTERACTIONS)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    if not args.interactions.exists():
        raise SystemExit(
            f"interactions parquet not found: {args.interactions}\n"
            "Run scripts/generate_interactions.py first."
        )

    logger.info("reading %s", args.interactions)
    interactions = pd.read_parquet(args.interactions)
    # Drop empty LLM failures from leakage analysis (they have no text downstream).
    interactions = interactions[interactions["text"].str.len() > 0].reset_index(drop=True)
    logger.info("reading %s", args.raw)
    raw = pd.read_csv(args.raw)

    report = compute_leakage_report(interactions, raw, seed=args.seed)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, ensure_ascii=False, indent=2)
    logger.info("wrote %s", args.report)

    print()
    print("=" * 72)
    print(f"Lead sayısı (analiz edilen)     : {report.n_leads}")
    print(f"Converted oranı                 : {report.converted_rate:.3f}")
    print(f"attitude → Converted AUC (5-CV) : {report.attitude_to_converted_auc:.3f}")
    print(f"Cramér's V (attitude, Converted): {report.cramers_v:.3f}")
    print(f"Mutual information (nats)       : {report.mutual_information_nats:.4f}")
    print("-" * 72)
    print("Attitude dağılımı:")
    for k, v in report.attitude_distribution.items():
        print(f"  {k:<22} {v}")
    print("-" * 72)
    print("Crosstab (attitude × Converted):")
    print(pd.DataFrame(report.crosstab).to_string(index=False))
    print("-" * 72)
    print("Tabular leakage adayı kolonlar (in-sample AUC):")
    print(pd.DataFrame(report.tabular_leakage_suspects).to_string(index=False))
    print("-" * 72)
    print("Yorum:")
    for note in report.interpretation:
        print(f"  • {note}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
