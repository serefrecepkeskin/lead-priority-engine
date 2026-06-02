"""Runtime configuration loaded from environment variables and the local `.env` file."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT: Path = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Project settings.

    Values come from (in priority order): process env vars, then `.env` at the
    repo root, then the defaults declared here. `.env.example` is the template
    a new contributor should copy to `.env`.
    """

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: str = "development"
    log_level: str = "INFO"

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    data_dir: Path = REPO_ROOT / "data"
    artifacts_dir: Path = REPO_ROOT / "artifacts"

    lead_scoring_model: str = "lead_scoring_lgbm.joblib"
    sentiment_model: str = "sentiment_model"

    # Azure OpenAI. All optional at load time so the package imports cleanly
    # without a configured LLM; the caller is responsible for asserting the
    # required values before issuing a request.
    azure_openai_api_key: str | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_deployment: str | None = None
    azure_openai_api_version: str | None = None
    azure_openai_max_tokens: int | None = None
    azure_openai_timeout: float | None = None
    azure_openai_reasoning_effort: str | None = None

    priority_weight_conversion: float = Field(default=0.6, ge=0.0, le=1.0)
    priority_weight_sentiment: float = Field(default=0.4, ge=0.0, le=1.0)


def get_settings() -> Settings:
    """Return a freshly-loaded :class:`Settings` instance.

    Kept as a function (rather than a module-level singleton) so tests can
    monkeypatch environment variables before construction.
    """
    return Settings()
