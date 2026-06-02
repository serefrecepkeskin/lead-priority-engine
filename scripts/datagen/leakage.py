"""Leakage diagnostics for the synthetic-attitude → `Converted` relationship.

The whole point of these checks is to *catch ourselves* if the synthetic
generator accidentally turned the sentiment label into a near-copy of the
conversion target. We compute:

* the AUC of a tiny logistic-regression on attitude alone (under 5-fold CV);
  > 0.85 is a strong leakage signal,
* Cramér's V and mutual information between attitude and ``Converted``; the
  target band is roughly 0.20–0.40 — realistic, neither independent nor a
  clone of the label,
* a row-normalized crosstab so each attitude class clearly contains both
  converted and non-converted leads,
* per-column AUC for the *tabular* columns that are known leakage suspects
  (``Tags``, ``Lead Quality``, ``Last Notable Activity``). These are filled
  in *after* an outcome is decided in the source CRM, so the scoring model
  must drop them — we flag them here so the report can quote numbers.

All functions are pure (input → numbers/dataframes), so both the CLI script
and the notebook can call them and get identical results.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mutual_info_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Bias-corrected Cramér's V for two categorical series."""

    confusion = pd.crosstab(x, y)
    chi2 = _chi2_no_yates(confusion.to_numpy())
    n = confusion.to_numpy().sum()
    if n == 0:
        return 0.0
    phi2 = chi2 / n
    r, k = confusion.shape
    phi2_corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2_corr / denom))


def _chi2_no_yates(table: np.ndarray) -> float:
    row_totals = table.sum(axis=1, keepdims=True)
    col_totals = table.sum(axis=0, keepdims=True)
    grand = table.sum()
    if grand == 0:
        return 0.0
    expected = row_totals @ col_totals / grand
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2_terms = np.where(expected > 0, (table - expected) ** 2 / expected, 0.0)
    return float(chi2_terms.sum())


def attitude_to_converted_auc(
    attitude: pd.Series,
    converted: pd.Series,
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> float:
    """Cross-validated AUC of a logistic regression on one-hot attitudes.

    Returns 0.5 (and skips folds) gracefully if a fold has only one class.
    """

    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    x = enc.fit_transform(attitude.astype(str).to_numpy().reshape(-1, 1))
    y = converted.astype(int).to_numpy()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs: list[float] = []
    for train_idx, test_idx in skf.split(x, y):
        if len(np.unique(y[test_idx])) < 2:
            continue
        model = LogisticRegression(max_iter=1000)
        model.fit(x[train_idx], y[train_idx])
        proba = model.predict_proba(x[test_idx])[:, 1]
        aucs.append(roc_auc_score(y[test_idx], proba))
    if not aucs:
        return float("nan")
    return float(np.mean(aucs))


def attitude_converted_mutual_info(attitude: pd.Series, converted: pd.Series) -> float:
    """Mutual information (in nats) between attitude and the binary outcome."""

    return float(mutual_info_score(attitude.astype(str), converted.astype(int)))


def crosstab_with_pct(attitude: pd.Series, converted: pd.Series) -> pd.DataFrame:
    """Counts plus row-normalized percentages of converted within each attitude."""

    counts = pd.crosstab(attitude, converted)
    counts.columns = [f"converted_{c}" for c in counts.columns]
    row_totals = counts.sum(axis=1).replace(0, np.nan)
    pct = counts.div(row_totals, axis=0).mul(100.0).round(2)
    pct.columns = [f"{c}_pct" for c in pct.columns]
    return pd.concat([counts, pct], axis=1)


def tabular_column_aucs(
    df: pd.DataFrame,
    *,
    columns: Iterable[str],
    target: str = "Converted",
) -> pd.DataFrame:
    """Per-column AUC for known leakage-suspect tabular columns.

    For categorical columns we one-hot encode (after marking NaNs as "Missing"),
    fit logistic regression with no CV (we're characterizing the column, not
    benchmarking a model), and report the in-sample AUC. Anything close to or
    above 0.90 is a strong signal the column was filled in after the outcome.
    """

    rows: list[dict[str, float | str | int]] = []
    y = df[target].astype(int).to_numpy()
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col].astype(str).fillna("Missing").replace({"nan": "Missing"})
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        x = enc.fit_transform(series.to_numpy().reshape(-1, 1))
        model = LogisticRegression(max_iter=1000)
        model.fit(x, y)
        proba = model.predict_proba(x)[:, 1]
        auc = float(roc_auc_score(y, proba))
        rows.append(
            {
                "column": col,
                "auc": round(auc, 4),
                "n_unique": int(series.nunique()),
                "n_missing": int((series == "Missing").sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)


@dataclass
class LeakageReport:
    """Full leakage diagnostic bundle, JSON-serializable via :meth:`to_dict`."""

    n_leads: int
    attitude_distribution: dict[str, int]
    converted_rate: float
    attitude_to_converted_auc: float
    cramers_v: float
    mutual_information_nats: float
    crosstab: list[dict[str, Any]] = field(default_factory=list)
    tabular_leakage_suspects: list[dict[str, Any]] = field(default_factory=list)
    interpretation: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _interpret(
    auc: float, cramers: float, crosstab: pd.DataFrame, suspects: pd.DataFrame
) -> list[str]:
    notes: list[str] = []
    if np.isnan(auc):
        notes.append("AUC hesaplanamadı (sınıf dengesi yetersiz).")
    elif auc > 0.85:
        notes.append(f"⚠️ attitude→Converted AUC çok yüksek ({auc:.3f} > 0.85): leakage işareti.")
    elif auc < 0.55:
        notes.append(f"attitude→Converted AUC zayıf ({auc:.3f}): sentiment çok az sinyal taşıyor.")
    else:
        notes.append(f"attitude→Converted AUC sağlıklı bantta ({auc:.3f}).")

    if 0.20 <= cramers <= 0.40:
        notes.append(f"Cramér's V hedef bantta ({cramers:.3f}∈[0.20, 0.40]) — gerçekçi korelasyon.")
    elif cramers > 0.40:
        notes.append(
            f"⚠️ Cramér's V hedef üstü ({cramers:.3f}): attitude `Converted`'a fazla yakın."
        )
    else:
        notes.append(f"Cramér's V düşük ({cramers:.3f}): attitude `Converted` ile zayıf bağlı.")

    for _, row in crosstab.iterrows():
        c0 = row.get("converted_0", 0)
        c1 = row.get("converted_1", 0)
        if c0 == 0 or c1 == 0:
            notes.append(
                f"⚠️ '{row.name}' sınıfında hem converted hem non-converted lead YOK"
                f" (counts: 0→{c0}, 1→{c1}) — leakage habercisi."
            )

    if not suspects.empty:
        worst = suspects.iloc[0]
        if worst["auc"] > 0.90:
            notes.append(
                f"⚠️ Tabular sızıntı adayı '{worst['column']}' tek-kolon AUC={worst['auc']}: "
                "scoring modelinde DROP edilmeli."
            )
    return notes


def compute_leakage_report(
    interactions: pd.DataFrame,
    raw: pd.DataFrame,
    *,
    leakage_suspect_columns: Iterable[str] = ("Tags", "Lead Quality", "Last Notable Activity"),
    seed: int = 42,
) -> LeakageReport:
    """End-to-end leakage diagnostic. Joins notes back to raw on Prospect ID."""

    merged = interactions.merge(
        raw[["Prospect ID", "Converted", *leakage_suspect_columns]],
        left_on="lead_id",
        right_on="Prospect ID",
        how="inner",
    )
    attitude = merged["attitude"]
    converted = merged["Converted"]

    auc = attitude_to_converted_auc(attitude, converted, seed=seed)
    cramers = cramers_v(attitude, converted)
    mi = attitude_converted_mutual_info(attitude, converted)
    crosstab = crosstab_with_pct(attitude, converted)
    suspects = tabular_column_aucs(raw, columns=leakage_suspect_columns)

    crosstab_records: list[dict[str, Any]] = []
    for idx, row in crosstab.iterrows():
        record: dict[str, Any] = {"attitude": str(idx)}
        for k, v in row.items():
            record[str(k)] = int(v) if isinstance(v, np.integer) else float(v)
        crosstab_records.append(record)
    suspects_records: list[dict[str, Any]] = [
        {str(k): v for k, v in row.items()} for row in suspects.to_dict(orient="records")
    ]

    report = LeakageReport(
        n_leads=len(merged),
        attitude_distribution={
            str(k): int(v) for k, v in attitude.value_counts().to_dict().items()
        },
        converted_rate=float(converted.mean()),
        attitude_to_converted_auc=float(auc),
        cramers_v=float(cramers),
        mutual_information_nats=float(mi),
        crosstab=crosstab_records,
        tabular_leakage_suspects=suspects_records,
        interpretation=_interpret(auc, cramers, crosstab, suspects),
    )
    return report


__all__ = [
    "LeakageReport",
    "attitude_converted_mutual_info",
    "attitude_to_converted_auc",
    "compute_leakage_report",
    "cramers_v",
    "crosstab_with_pct",
    "tabular_column_aucs",
]
