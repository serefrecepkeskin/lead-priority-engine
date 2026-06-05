"""``GET /leads/top`` — precomputed cache, sort order, filters."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


def test_top_default_returns_sorted_desc(client: TestClient) -> None:
    response = client.get("/leads/top")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["count"] == 10
    assert body["total_available"] >= 900  # 924 in the tracked parquet
    priorities = [entry["priority"] for entry in body["leads"]]
    assert priorities == sorted(priorities, reverse=True)
    first = body["leads"][0]
    assert 0.0 <= first["p_conversion"] <= 1.0
    assert 0.0 <= first["sentiment_score"] <= 1.0
    assert 0.0 <= first["priority"] <= 1.0
    assert first["predicted_attitude"] in {
        "positive_engagement",
        "objection",
        "neutral",
        "disengaged",
    }


def test_top_with_n_zero_returns_empty(client: TestClient) -> None:
    response = client.get("/leads/top", params={"n": 0})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 0
    assert body["leads"] == []
    assert body["total_available"] >= 900


def test_top_with_n_above_max_returns_422(client: TestClient) -> None:
    response = client.get("/leads/top", params={"n": 10_000})
    assert response.status_code == 422


def test_top_with_min_priority_filters(client: TestClient) -> None:
    response = client.get("/leads/top", params={"n": 50, "min_priority": 0.9})
    assert response.status_code == 200
    body = response.json()
    assert all(entry["priority"] >= 0.9 for entry in body["leads"])
    assert body["count"] == len(body["leads"])


def test_top_with_n_larger_than_filtered_returns_partial(client: TestClient) -> None:
    """min_priority floor higher than any entry → empty leads, count=0."""
    response = client.get("/leads/top", params={"n": 10, "min_priority": 0.999999})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 0
    assert body["leads"] == []


def test_top_response_includes_model_versions(client: TestClient) -> None:
    response = client.get("/leads/top", params={"n": 3})
    assert response.status_code == 200
    body = response.json()
    versions = body["model_versions"]
    assert versions["lead_scoring_kind"] == "lightgbm"
    assert versions["sentiment_model_name"] == "z-ai/glm-4.5-air:free"
    assert versions["feature_pipeline_schema"] >= 1
    assert versions["lead_scoring_schema"] >= 1
