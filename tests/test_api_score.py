"""``POST /score`` — happy path + validation + graceful fallback."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lead_priority.models import OpenRouterSentiment

EXAMPLE_PAYLOAD_PATH = Path(__file__).resolve().parents[1] / "examples" / "score_request.json"
CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"


@pytest.fixture
def example_payload() -> dict:
    return json.loads(EXAMPLE_PAYLOAD_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def fast_sentiment(monkeypatch: pytest.MonkeyPatch, api_app: FastAPI) -> None:
    """Replace the cached sentiment classifier with a zero-backoff instance.

    The default max_retries (5) + exponential backoff (1→30s) makes the
    429-fallback test wait ~30 seconds. Test path needs deterministic timing,
    not realistic retry semantics — those live in the sentiment unit tests.
    """
    fast = OpenRouterSentiment(
        model_name="z-ai/glm-4.5-air:free",
        api_key="sk-or-test-key",
        max_retries=1,
        backoff_base_seconds=0.0,
        backoff_cap_seconds=0.0,
    )
    monkeypatch.setattr(
        "lead_priority.api.endpoints.score.get_sentiment_classifier",
        lambda: fast,
    )


def _openrouter_ok_response(attitude: str = "positive_engagement") -> dict:
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps({"attitude": attitude}),
                    "role": "assistant",
                }
            }
        ],
        "usage": {"prompt_tokens": 245, "completion_tokens": 8, "total_tokens": 253},
    }


@respx.mock
def test_score_happy_path(
    client: TestClient,
    example_payload: dict,
    fast_sentiment: None,
) -> None:
    respx.post(CHAT_COMPLETIONS_URL).mock(
        return_value=httpx.Response(200, json=_openrouter_ok_response("objection"))
    )
    response = client.post("/score", json=example_payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert 0.0 <= body["p_conversion"] <= 1.0
    assert body["sentiment"]["predicted_attitude"] == "objection"
    assert body["sentiment"]["sentiment_unavailable"] is False
    assert body["sentiment"]["fallback_reason"] is None
    assert 0.0 <= body["priority"] <= 1.0
    weight_sum = body["weights"]["weight_conversion"] + body["weights"]["weight_sentiment"]
    assert weight_sum == pytest.approx(1.0)
    assert body["model_versions"]["lead_scoring_kind"] == "lightgbm"
    assert body["model_versions"]["sentiment_model_name"] == "z-ai/glm-4.5-air:free"
    assert body["request_id"] is not None


def test_score_missing_interaction_text_returns_422(
    client: TestClient,
    example_payload: dict,
) -> None:
    payload = {"lead": example_payload["lead"]}
    response = client.post("/score", json=payload)
    assert response.status_code == 422


def test_score_empty_interaction_text_returns_422(
    client: TestClient,
    example_payload: dict,
) -> None:
    payload = {**example_payload, "interaction_text": ""}
    response = client.post("/score", json=payload)
    assert response.status_code == 422


@respx.mock
def test_score_falls_back_on_rate_limit(
    client: TestClient,
    example_payload: dict,
    fast_sentiment: None,
) -> None:
    respx.post(CHAT_COMPLETIONS_URL).mock(return_value=httpx.Response(429))
    response = client.post("/score", json=example_payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["sentiment"]["sentiment_unavailable"] is True
    assert body["sentiment"]["fallback_reason"] == "openrouter_rate_limit"
    assert body["sentiment"]["predicted_attitude"] == "neutral"
    assert body["sentiment"]["sentiment_score"] == pytest.approx(0.40)
    # Priority still computed via neutral fallback weight.
    assert 0.0 <= body["priority"] <= 1.0


@respx.mock
def test_score_returns_502_on_permanent_error(
    client: TestClient,
    example_payload: dict,
    fast_sentiment: None,
) -> None:
    respx.post(CHAT_COMPLETIONS_URL).mock(return_value=httpx.Response(400, text="bad model id"))
    response = client.post("/score", json=example_payload)
    assert response.status_code == 502
    body = response.json()
    assert body["detail"] == "openrouter_permanent_error"


@respx.mock
def test_score_returns_502_on_malformed_attitude(
    client: TestClient,
    example_payload: dict,
    fast_sentiment: None,
) -> None:
    bad_body = {
        "choices": [{"message": {"content": "I would say it's positive overall."}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    respx.post(CHAT_COMPLETIONS_URL).mock(return_value=httpx.Response(200, json=bad_body))
    response = client.post("/score", json=example_payload)
    assert response.status_code == 502
    body = response.json()
    assert body["detail"] == "openrouter_error"


@respx.mock
def test_score_logs_no_pii(
    client: TestClient,
    example_payload: dict,
    fast_sentiment: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    respx.post(CHAT_COMPLETIONS_URL).mock(
        return_value=httpx.Response(200, json=_openrouter_ok_response("neutral"))
    )
    secret_phrase = example_payload["interaction_text"][:30]
    with caplog.at_level(logging.INFO, logger="lead_priority.api.score"):
        response = client.post("/score", json=example_payload)
    assert response.status_code == 200
    combined = "\n".join(record.getMessage() + str(record.__dict__) for record in caplog.records)
    assert secret_phrase not in combined, "interaction_text leaked into logs"
