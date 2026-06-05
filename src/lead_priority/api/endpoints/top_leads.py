"""``GET /leads/top`` — pre-computed top-N leads sorted by priority.

The 924 leads in ``artifacts/sentiment_predictions/glm-4-5-air_test.parquet``
already have a cached sentiment label, so /leads/top never calls OpenRouter.
At startup the cache joins the parquet against ``data/Lead Scoring.csv``,
runs the full feature pipeline + classifier in a single batch, computes the
priority, and stores the sorted list in memory. Per-request work is then a
filter + slice on an in-memory list — sub-millisecond at this size.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pandas as pd
from fastapi import APIRouter, Query, Request

from lead_priority.api.deps import get_top_leads_cache, model_versions_payload
from lead_priority.api.schemas import ModelVersions, TopLeadEntry, TopLeadsResponse
from lead_priority.features import FeatureTransformer
from lead_priority.models import (
    SENTIMENT_SCORE_MAP,
    LeadScoringModel,
    SentimentClass,
    batch_compute_priority,
)

logger = logging.getLogger("lead_priority.api.top_leads")

MAX_TOP_N = 924
"""Upper bound on ``n`` — matches the test split size in the cached parquet.
The Pydantic validator enforces this so a caller asking for n=10_000 gets a
422 instead of an opaque empty list."""

JOIN_OVERLAP_FLOOR = 0.95
"""Minimum fraction of parquet rows that must match the raw CSV. Below this
the join is corrupt (likely a stale parquet checked into git after a CSV
rebuild) and the service should fail-fast at boot rather than serve garbage."""

router = APIRouter(tags=["top_leads"])


@dataclass
class TopLeadsCache:
    """Sorted, in-memory list of leads ranked by combined priority.

    Built once at startup from the raw CSV + the cached sentiment parquet.
    Lookup is a filter-then-slice; no model calls happen on the request path.
    """

    sorted_leads: list[TopLeadEntry] = field(default_factory=list)

    @classmethod
    def build(
        cls,
        *,
        raw_csv: Path,
        predictions_parquet: Path,
        transformer: FeatureTransformer,
        scorer: LeadScoringModel,
    ) -> TopLeadsCache:
        """Construct from the file pair on disk.

        Raises:
            FileNotFoundError: Either input is missing.
            RuntimeError: The join recovers less than :data:`JOIN_OVERLAP_FLOOR`
                of the parquet rows.
        """
        if not raw_csv.exists():
            raise FileNotFoundError(f"raw CSV not found: {raw_csv}")
        if not predictions_parquet.exists():
            raise FileNotFoundError(f"sentiment parquet not found: {predictions_parquet}")

        leads_df = pd.read_csv(raw_csv)
        sentiment_df = pd.read_parquet(predictions_parquet)

        joined = leads_df.merge(
            sentiment_df[["lead_id", "language", "predicted_attitude"]],
            left_on="Prospect ID",
            right_on="lead_id",
            how="inner",
        )
        overlap = len(joined) / max(len(sentiment_df), 1)
        if overlap < JOIN_OVERLAP_FLOOR:
            raise RuntimeError(
                f"top-leads join overlap {overlap:.2%} below floor "
                f"{JOIN_OVERLAP_FLOOR:.0%}; parquet has {len(sentiment_df)} rows, "
                f"join produced {len(joined)} rows"
            )
        logger.info(
            "top_leads_join",
            extra={
                "parquet_rows": len(sentiment_df),
                "csv_rows": len(leads_df),
                "joined_rows": len(joined),
                "overlap": round(overlap, 4),
            },
        )

        features = transformer.transform(joined)
        p_conversion = scorer.predict_proba(features)
        attitudes = joined["predicted_attitude"].astype(str).tolist()
        priority = batch_compute_priority(p_conversion, attitudes)

        entries: list[TopLeadEntry] = []
        for idx, row in enumerate(joined.itertuples(index=False)):
            attitude = cast(SentimentClass, attitudes[idx])
            entries.append(
                TopLeadEntry(
                    lead_id=str(row.lead_id),
                    p_conversion=round(float(p_conversion[idx]), 6),
                    predicted_attitude=attitude,
                    sentiment_score=SENTIMENT_SCORE_MAP[attitude],
                    priority=round(float(priority[idx]), 6),
                    language=str(row.language),
                )
            )
        entries.sort(key=lambda e: e.priority, reverse=True)
        logger.info("top_leads_cache_built", extra={"n": len(entries)})
        return cls(sorted_leads=entries)

    def query(self, n: int, min_priority: float) -> list[TopLeadEntry]:
        """Filter by ``priority >= min_priority`` then return the first ``n``."""
        if min_priority <= 0.0:
            return list(self.sorted_leads[:n])
        # sorted descending; the first entry below the floor terminates the slice.
        filtered: list[TopLeadEntry] = []
        for entry in self.sorted_leads:
            if entry.priority < min_priority:
                break
            filtered.append(entry)
            if len(filtered) >= n:
                break
        return filtered

    @property
    def total_available(self) -> int:
        return len(self.sorted_leads)


@router.get("/leads/top", response_model=TopLeadsResponse)
def top_leads(
    request: Request,
    n: int = Query(10, ge=0, le=MAX_TOP_N, description="Number of leads to return."),
    min_priority: float = Query(
        0.0,
        ge=0.0,
        le=1.0,
        description="Filter floor on priority score before truncating to n.",
    ),
) -> TopLeadsResponse:
    """Return the top-N leads (priority desc) from the in-memory cache."""
    cache = get_top_leads_cache()
    leads = cache.query(n=n, min_priority=min_priority)
    logger.info(
        "top_leads_served",
        extra={
            "request_id": getattr(request.state, "request_id", None),
            "n_requested": n,
            "n_returned": len(leads),
            "min_priority": min_priority,
        },
    )
    return TopLeadsResponse(
        count=len(leads),
        total_available=cache.total_available,
        leads=leads,
        model_versions=ModelVersions(**model_versions_payload()),
        request_id=getattr(request.state, "request_id", None),
    )


__all__ = ["JOIN_OVERLAP_FLOOR", "MAX_TOP_N", "TopLeadsCache", "router"]
