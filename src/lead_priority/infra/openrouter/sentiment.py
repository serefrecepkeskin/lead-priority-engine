"""Serving-side sentiment / intent classifier: OpenRouter LLM wrapper.

Phase 3 takes the "open-source LLM, zero/few-shot" path; the alternative
considered was fine-tuning an encoder (XLM-R / DistilBERT). The reasoning for
that choice is documented in ``docs/3_sentiment_classifier.docx`` §1; in short
the LLM route lets the production runtime add a class in minutes without any
training step, and the modelling depth budget was spent on Phase 2.

This module is import-safe with no API key — every call site asserts the key
exists before issuing a request, mirroring the Azure OpenAI pattern in
``settings.py``. The wrapper hand-rolls retry and JSON parsing rather than
pulling in ``langchain-openai`` or ``openai`` so the production image stays
lean (``httpx`` is the only new runtime dep).

Serving flow::

    sentiment = OpenRouterSentiment.from_settings("meta-llama/llama-3.3-70b-instruct:free")
    label = sentiment.predict("Demo talep ettiler, pricing'i high buldu")
    # -> "objection"
    score = sentiment.predict_score("Demo talep ettiler, pricing'i high buldu")
    # -> 0.65
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, cast, get_args

import httpx

from lead_priority.core.scoring.sentiment_classes import (
    SENTIMENT_CLASSES,
    SENTIMENT_SCORE_MAP,
    SentimentClass,
)
from lead_priority.settings import get_settings

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

# Model aliases used by scripts/evaluate_openrouter_sentiment.py so the CLI
# accepts short --model flags. Kept here next to the wrapper so the canonical
# model strings live in one place. The :free tier turns over frequently
# (deepseek-chat and qwen-2.5-72b were dropped between the plan and the
# implementation), so check `GET /api/v1/models` before adding new entries.
MODEL_ALIASES: dict[str, str] = {
    "llama": "meta-llama/llama-3.3-70b-instruct:free",
    "glm": "z-ai/glm-4.5-air:free",
    "nemotron-nano": "nvidia/nemotron-nano-9b-v2:free",
}

_SYSTEM_PROMPT = (
    "You are a sales-operations assistant that classifies the attitude "
    "expressed in a sales-rep interaction note. The note can be in Turkish, "
    "English, or a mix of the two (code-switching is common). Pick exactly "
    "one of these four labels:\n"
    '- "positive_engagement": the lead is engaged, asks follow-up questions, '
    "or expresses concrete interest.\n"
    '- "objection": the lead raises a concern or doubt about price, '
    "duration, scope, or content.\n"
    '- "neutral": the note is informational with no strong sentiment in '
    "either direction.\n"
    '- "disengaged": the lead is silent, distant, or unresponsive after '
    "outreach.\n"
    "Respond with strict JSON only, no prose, no markdown fences. Schema: "
    '{"attitude": "<one of the four labels>"}. '
    "Choose the single best label even if the note is ambiguous."
)

# 8 few-shot examples: 4 classes × 2 languages (TR + EN). Mix code-switching
# is left to the model's generalisation so per-language F1 can probe it.
_FEW_SHOT_EXAMPLES: tuple[tuple[str, SentimentClass], ...] = (
    (
        "Lead webinar sonrası iletişime geçti, demo talep etti ve takım "
        "üyeleriyle birlikte deneyebilmek için ek erişim sordu.",
        "positive_engagement",
    ),
    (
        "After the intro call the lead asked for a tailored pricing sheet "
        "and offered two slots next week to bring in their data team.",
        "positive_engagement",
    ),
    (
        "Fiyatlandırmayı yüksek buldu ve mevcut çözümlerine kıyasla "
        "geri dönüş süresinin nasıl olacağını sorguladı.",
        "objection",
    ),
    (
        "The lead pushed back on the contract length and was not "
        "comfortable with the proposed onboarding scope.",
        "objection",
    ),
    (
        "Görüşmede ürün özelliklerine dair bilgilendirme yapıldı, lead "
        "notları aldı fakat sonraki adım belirtmedi.",
        "neutral",
    ),
    (
        "Shared the standard product overview deck; the lead noted the "
        "information and did not signal a clear next step either way.",
        "neutral",
    ),
    (
        "Üç farklı kanaldan ulaşma denemesine rağmen geri dönüş alınamadı, "
        "son etkileşim üzerinden iki haftadan fazla geçti.",
        "disengaged",
    ),
    (
        "The lead has been silent across the last three follow-up attempts "
        "and has not opened the most recent emails.",
        "disengaged",
    ),
)


def _build_messages(text: str) -> list[dict[str, str]]:
    """Assemble the chat-completion messages: system + 8 few-shot + user."""
    messages: list[dict[str, str]] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    for example_text, label in _FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": example_text})
        messages.append({"role": "assistant", "content": json.dumps({"attitude": label})})
    messages.append({"role": "user", "content": text})
    return messages


_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
_ATTITUDE_JSON_RE = re.compile(
    r'\{\s*"attitude"\s*:\s*"([a-z_ ]+)"\s*\}',
    re.IGNORECASE,
)


def _strip_code_fences(raw: str) -> str:
    """Remove markdown ``` fences some free models wrap their JSON in."""
    stripped = _CODE_FENCE_RE.sub("", raw.strip())
    return stripped.strip()


def _extract_attitude_label(raw: str) -> str | None:
    """Find the last well-formed ``{"attitude": "..."}`` JSON object in ``raw``.

    Some free models prepend reasoning prose, emit two back-to-back JSON
    objects, or wrap the answer in extra whitespace. The regex tolerates all
    three by scanning for the canonical shape and returning the trailing
    match — the model's final decision when prose precedes the answer.
    """
    matches = _ATTITUDE_JSON_RE.findall(raw)
    if not matches:
        return None
    return str(matches[-1]).strip().lower()


class OpenRouterError(RuntimeError):
    """Raised when OpenRouter is unreachable or returns malformed output.

    Distinct from :class:`OpenRouterRateLimitError` so callers can decide
    whether to checkpoint-and-exit (rate limit) or to log-and-skip (genuine
    parse / transport error).
    """


class OpenRouterRateLimitError(OpenRouterError):
    """Raised after exhausting retries against 429 / quota responses."""


class OpenRouterPermanentError(OpenRouterError):
    """Raised for non-transient 4xx responses (bad model id, malformed request).

    These propagate out of the retry loop on the first occurrence — retrying
    a 404 just wastes the daily quota.
    """


class OpenRouterMalformedError(OpenRouterError):
    """Raised when the model returned an unparseable or out-of-schema response.

    Separated from generic transport failures so the FastAPI service can
    surface it as a 502 (upstream-bug) rather than silently degrading to
    a neutral fallback — a model returning gibberish is a real signal that
    operators should see, not absorb.
    """


@dataclass
class OpenRouterSentiment:
    """Predict-only sentiment / intent classifier backed by an OpenRouter model.

    The wrapper is stateless beyond config: every ``predict`` opens a fresh
    HTTP request through a long-lived ``httpx.Client`` (so connection pooling
    happens) and parses the JSON response. Use :meth:`from_settings` to inject
    the API key from the ``.env``-backed ``Settings`` instance.
    """

    model_name: str
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    timeout: float = 30.0
    max_retries: int = 5
    temperature: float = 0.0
    """``0.0`` so re-runs of the eval CLI produce identical predictions."""

    backoff_base_seconds: float = 1.0
    backoff_cap_seconds: float = 30.0
    """Exponential backoff schedule: ``min(base * 2**attempt, cap)`` with a
    small random jitter so concurrent workers do not retry in lock-step."""

    _last_usage: dict[str, int] = field(default_factory=dict, repr=False)
    """Token + latency counters from the most recent successful call, exposed
    via :meth:`last_usage` for the CLI to log."""

    _last_latency_ms: float = 0.0

    @classmethod
    def from_settings(cls, model_name: str, **overrides: Any) -> OpenRouterSentiment:
        """Construct from the project-wide ``Settings``.

        Raises ``RuntimeError`` if ``OPEN_ROUTER_API_KEY`` is unset — failing
        loudly at construction is better than at the first prediction call.
        """
        settings = get_settings()
        if not settings.open_router_api_key:
            raise RuntimeError(
                "OPEN_ROUTER_API_KEY is not set; add it to .env before using OpenRouterSentiment."
            )
        return cls(
            model_name=model_name,
            api_key=settings.open_router_api_key,
            base_url=settings.open_router_base_url,
            **overrides,
        )

    def predict(self, text: str) -> SentimentClass:
        """Classify a single interaction note into one of the four attitudes.

        Args:
            text: The interaction note. TR / EN / Mix are all supported by
                the prompt; nothing is done at this layer to detect language.

        Returns:
            One of the four labels in :data:`SENTIMENT_CLASSES`.

        Raises:
            OpenRouterRateLimitError: After exhausting :attr:`max_retries`
                against 429 / quota responses.
            OpenRouterError: For other transport or parse failures that
                survive the retry loop.
        """
        raw = self._call_openrouter_with_retry(text)
        label = self._parse_attitude(raw)
        return label

    def predict_score(self, text: str) -> float:
        """Classify ``text`` and map the label to the Phase 4 priority weight.

        Returns:
            Float in ``[0.0, 1.0]`` from :data:`SENTIMENT_SCORE_MAP`.
        """
        return SENTIMENT_SCORE_MAP[self.predict(text)]

    def last_usage(self) -> dict[str, int]:
        """Token counts from the most recent successful call (``prompt_tokens``,
        ``completion_tokens``, ``total_tokens``). Empty before the first call."""
        return dict(self._last_usage)

    def last_latency_ms(self) -> float:
        """Wall-clock latency of the most recent successful call in milliseconds."""
        return self._last_latency_ms

    # -- internal: HTTP transport + parsing -----------------------------------

    def _call_openrouter_with_retry(self, text: str) -> str:
        """Issue the chat-completion POST, retrying on transient failures.

        Returns the raw assistant ``content`` string. Parsing into a label is
        the caller's responsibility so retry semantics stay separate from
        validation semantics.
        """
        attempt = 0
        last_error: Exception | None = None
        with httpx.Client(timeout=self.timeout) as client:
            while attempt <= self.max_retries:
                try:
                    return self._issue_request(client, text)
                except OpenRouterPermanentError:
                    # Non-transient 4xx — fail fast, do not burn quota retrying.
                    raise
                except OpenRouterRateLimitError as exc:
                    last_error = exc
                    if attempt == self.max_retries:
                        raise
                    self._sleep_with_backoff(attempt)
                except (httpx.HTTPError, OpenRouterError) as exc:
                    last_error = exc
                    if attempt == self.max_retries:
                        raise OpenRouterError(
                            f"OpenRouter call failed after {self.max_retries + 1} attempts: {exc!r}"
                        ) from exc
                    self._sleep_with_backoff(attempt)
                attempt += 1
        # Unreachable — the loop either returns or raises — but mypy needs it.
        raise OpenRouterError(f"OpenRouter call failed unexpectedly: {last_error!r}")

    def _issue_request(self, client: httpx.Client, text: str) -> str:
        """Single attempt: POST and translate HTTP errors into typed exceptions."""
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": _build_messages(text),
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/SerefRecepKeskin/lead-priority-engine",
            "X-Title": "Lead Priority Engine Sentiment Classifier",
        }
        start = time.perf_counter()
        response = client.post(url, json=payload, headers=headers)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if response.status_code == 429:
            raise OpenRouterRateLimitError(f"OpenRouter returned 429 for model {self.model_name!r}")
        if response.status_code >= 500:
            # Server-side hiccups are transient — let the retry loop have it.
            raise OpenRouterError(
                f"OpenRouter returned {response.status_code} for model "
                f"{self.model_name!r}: {response.text[:200]!r}"
            )
        if response.status_code >= 400:
            # 4xx other than 429 are permanent (bad model id, bad request,
            # missing auth). Retrying just burns rate-limit budget.
            raise OpenRouterPermanentError(
                f"OpenRouter returned {response.status_code} for model "
                f"{self.model_name!r}: {response.text[:200]!r}"
            )
        body = response.json()
        choices = body.get("choices") or []
        if not choices:
            raise OpenRouterError(f"OpenRouter response missing 'choices': {body!r}")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            # Some reasoning models (e.g. z-ai/glm-4.5-air) put the JSON answer
            # in `reasoning` while leaving `content` null. Fall back to the
            # reasoning trace; the JSON-extraction regex in _parse_attitude
            # picks the last `{"attitude": ...}` block out of the prose.
            reasoning = message.get("reasoning")
            if isinstance(reasoning, str) and reasoning.strip():
                content = reasoning
            else:
                raise OpenRouterError(f"OpenRouter response missing message content: {body!r}")
        usage = body.get("usage") or {}
        self._last_usage = {
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", 0)),
        }
        self._last_latency_ms = elapsed_ms
        return content

    def _parse_attitude(self, raw: str) -> SentimentClass:
        """Extract the ``attitude`` field from the model's JSON response.

        Two layers of tolerance for free-tier model quirks:

        1. Strict ``json.loads`` after stripping markdown fences — the happy
           path for well-behaved models.
        2. Regex fallback that finds the last canonical
           ``{"attitude": "..."}`` block in the raw string. Catches reasoning
           prose with a final JSON line and back-to-back duplicate JSON.
        """
        cleaned = _strip_code_fences(raw)
        normalised: str | None = None
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            label = data.get("attitude")
            if isinstance(label, str):
                normalised = label.strip().lower()
        if normalised is None:
            normalised = _extract_attitude_label(raw)
        if normalised is None:
            raise OpenRouterMalformedError(
                f"OpenRouter response has no parseable 'attitude' field: {raw[:200]!r}"
            )
        if normalised not in get_args(SentimentClass):
            raise OpenRouterMalformedError(
                f"OpenRouter returned unknown attitude label {normalised!r}; "
                f"allowed: {SENTIMENT_CLASSES}"
            )
        return cast(SentimentClass, normalised)

    def _sleep_with_backoff(self, attempt: int) -> None:
        delay = min(
            self.backoff_base_seconds * (2**attempt),
            self.backoff_cap_seconds,
        )
        delay += random.uniform(0.0, delay * 0.25)
        logger.info(
            "openrouter_backoff",
            extra={"model": self.model_name, "attempt": attempt, "delay_s": delay},
        )
        time.sleep(delay)


__all__ = [
    "MODEL_ALIASES",
    "OpenRouterError",
    "OpenRouterMalformedError",
    "OpenRouterPermanentError",
    "OpenRouterRateLimitError",
    "OpenRouterSentiment",
]
