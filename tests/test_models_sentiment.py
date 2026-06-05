"""Unit tests for the OpenRouter-backed sentiment wrapper.

Every test mocks the HTTP transport with ``respx`` so the suite never reaches
OpenRouter — CI runs without an API key and contributors do not consume the
free-tier quota when running ``pytest`` locally. The retry logic is exercised
with the real ``time.sleep`` patched to a no-op so tests stay fast.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

import lead_priority.infra.openrouter.sentiment as sentiment_module
import lead_priority.settings as settings_module
from lead_priority.core.scoring.sentiment_classes import SENTIMENT_CLASSES, SENTIMENT_SCORE_MAP
from lead_priority.infra.openrouter.sentiment import (
    OpenRouterError,
    OpenRouterRateLimitError,
    OpenRouterSentiment,
)

API_URL = "https://openrouter.ai/api/v1/chat/completions"


def _ok_response(label: str, prompt_tokens: int = 50, completion_tokens: int = 5) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "model": "test/model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"attitude": label}),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@pytest.fixture
def sentiment_client() -> OpenRouterSentiment:
    """Wrapper with small retry budget so the retry tests stay snappy."""
    return OpenRouterSentiment(
        model_name="test/model",
        api_key="sk-fake-test-key",
        max_retries=2,
        backoff_base_seconds=0.0,
        backoff_cap_seconds=0.0,
    )


@pytest.fixture(autouse=True)
def _patch_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip backoff sleeps inside the retry loop so tests don't wait."""
    monkeypatch.setattr(sentiment_module.time, "sleep", lambda _seconds: None)


@respx.mock
def test_predict_returns_valid_class(sentiment_client: OpenRouterSentiment) -> None:
    respx.post(API_URL).mock(return_value=httpx.Response(200, json=_ok_response("objection")))

    label = sentiment_client.predict("Fiyat çok yüksek, bütçe yetmez.")

    assert label == "objection"
    assert label in SENTIMENT_CLASSES
    assert sentiment_client.last_usage()["total_tokens"] == 55
    assert sentiment_client.last_latency_ms() > 0


@respx.mock
@pytest.mark.parametrize(
    ("label", "expected_score"),
    [
        ("positive_engagement", 1.0),
        ("objection", 0.65),
        ("neutral", 0.40),
        ("disengaged", 0.10),
    ],
)
def test_predict_score_maps_correctly(
    sentiment_client: OpenRouterSentiment, label: str, expected_score: float
) -> None:
    respx.post(API_URL).mock(return_value=httpx.Response(200, json=_ok_response(label)))

    score = sentiment_client.predict_score("synthetic interaction note text")

    assert score == expected_score
    assert SENTIMENT_SCORE_MAP[label] == expected_score


@respx.mock
def test_predict_retries_on_429(sentiment_client: OpenRouterSentiment) -> None:
    route = respx.post(API_URL).mock(
        side_effect=[
            httpx.Response(429, json={"error": "rate_limited"}),
            httpx.Response(429, json={"error": "rate_limited"}),
            httpx.Response(200, json=_ok_response("neutral")),
        ]
    )

    label = sentiment_client.predict("note")

    assert label == "neutral"
    assert route.call_count == 3


@respx.mock
def test_predict_raises_after_max_retries(sentiment_client: OpenRouterSentiment) -> None:
    respx.post(API_URL).mock(return_value=httpx.Response(429, json={"error": "rate_limited"}))

    with pytest.raises(OpenRouterRateLimitError):
        sentiment_client.predict("note")


@respx.mock
def test_predict_handles_markdown_code_fences(sentiment_client: OpenRouterSentiment) -> None:
    """Some free models wrap JSON in ```json fences — the parser strips them."""
    fenced = "```json\n" + json.dumps({"attitude": "positive_engagement"}) + "\n```"
    response = _ok_response("positive_engagement")
    response["choices"][0]["message"]["content"] = fenced
    respx.post(API_URL).mock(return_value=httpx.Response(200, json=response))

    label = sentiment_client.predict("note")

    assert label == "positive_engagement"


@respx.mock
def test_predict_uses_reasoning_when_content_null(
    sentiment_client: OpenRouterSentiment,
) -> None:
    """z-ai/glm-4.5-air puts the JSON answer in `reasoning` and nulls `content`."""
    response = _ok_response("objection")
    response["choices"][0]["message"]["content"] = None
    response["choices"][0]["message"]["reasoning"] = (
        'Let me analyze. The note shows the lead pushed back on price.\n\n{"attitude": "objection"}'
    )
    respx.post(API_URL).mock(return_value=httpx.Response(200, json=response))

    label = sentiment_client.predict("note")

    assert label == "objection"


@respx.mock
def test_predict_handles_double_json_blob(sentiment_client: OpenRouterSentiment) -> None:
    """Some free models emit back-to-back JSON; the parser keeps the last one."""
    response = _ok_response("disengaged")
    response["choices"][0]["message"]["content"] = (
        '{"attitude": " "disengaged"}"}{"attitude": "disengaged"}'
    )
    respx.post(API_URL).mock(return_value=httpx.Response(200, json=response))

    label = sentiment_client.predict("note")

    assert label == "disengaged"


@respx.mock
def test_predict_retries_on_invalid_json_then_raises(
    sentiment_client: OpenRouterSentiment,
) -> None:
    """Non-JSON content surfaces as OpenRouterError after the retry budget."""
    garbage = _ok_response("positive_engagement")
    garbage["choices"][0]["message"]["content"] = "I think it's positive."
    respx.post(API_URL).mock(return_value=httpx.Response(200, json=garbage))

    with pytest.raises(OpenRouterError):
        sentiment_client.predict("note")


@respx.mock
def test_predict_rejects_unknown_attitude_label(
    sentiment_client: OpenRouterSentiment,
) -> None:
    response = _ok_response("very_positive")  # not in SENTIMENT_CLASSES
    respx.post(API_URL).mock(return_value=httpx.Response(200, json=response))

    with pytest.raises(OpenRouterError, match="unknown attitude"):
        sentiment_client.predict("note")


def test_from_settings_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPEN_ROUTER_API_KEY", "sk-or-test-from-env")

    client = OpenRouterSentiment.from_settings("meta-llama/llama-3.3-70b-instruct:free")

    assert client.api_key == "sk-or-test-from-env"
    assert client.model_name == "meta-llama/llama-3.3-70b-instruct:free"
    assert client.base_url.endswith("openrouter.ai/api/v1")


def test_from_settings_raises_when_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPEN_ROUTER_API_KEY", raising=False)

    # The .env file at repo root may still define the key; force Settings
    # to ignore it by pointing env_file at a nonexistent path.
    class _IsolatedSettings(settings_module.Settings):  # type: ignore[misc, valid-type]
        model_config = settings_module.SettingsConfigDict(
            env_file=None,
            extra="ignore",
            case_sensitive=False,
        )

    monkeypatch.setattr(settings_module, "Settings", _IsolatedSettings)

    with pytest.raises(RuntimeError, match="OPEN_ROUTER_API_KEY"):
        OpenRouterSentiment.from_settings("test/model")
