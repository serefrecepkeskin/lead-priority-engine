# Multi-stage build for the Lead Priority Engine FastAPI service.
#
# Builder stage:  installs pip dependencies into a throwaway prefix.
# Runtime stage:  copies that prefix plus the source tree + the small
#                 tracked artefacts the service needs at runtime.
#
# Image size budget: ≤800 MB. LightGBM, scikit-learn, and pyarrow account
# for ~250 MB on their own; the rest is python:3.12-slim's base layers.
# Heavy training-only deps (matplotlib, shap, jupyter, langchain-openai)
# live in requirements-dev.txt and are intentionally NOT installed here.

FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


FROM python:3.12-slim AS runtime

# libgomp1 is LightGBM's only runtime shared library; everything else
# ships as wheels with bundled binaries.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app

COPY --from=builder /install /usr/local

WORKDIR /app

# Source + project metadata (for `pip install -e .` so `lead_priority`
# is importable). pyproject.toml + README.md are referenced from setup.
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Tracked serving artefacts. The full artifacts/ tree is intentionally NOT
# copied — the dev-only LR model, figures, and CSV mirrors stay out of
# the image. See .dockerignore for the negative list.
COPY artifacts/feature_pipeline.joblib ./artifacts/
COPY artifacts/lead_scoring_lgbm.joblib ./artifacts/
COPY artifacts/lead_scoring_metrics.json ./artifacts/
COPY artifacts/sentiment_metrics.json ./artifacts/
COPY artifacts/priority_metrics.json ./artifacts/
COPY artifacts/sentiment_predictions/glm-4-5-air_test.parquet ./artifacts/sentiment_predictions/

# Raw data file — needed by the TopLeadsCache startup join to recover
# features that are not stored in the predictions parquet. ~1 MB.
COPY ["data/Lead Scoring.csv", "./data/"]

RUN pip install --no-cache-dir --no-deps -e . \
    && chown -R app:app /app

USER app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_ENV=production \
    LOG_LEVEL=INFO

EXPOSE 8000

# A 30 s start-period covers the lifespan warm-up (feature pipeline load +
# top-leads cache build on 924 rows ≈ 100 ms, but JIT-warm models can be
# slower on cold containers).
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/healthz >/dev/null || exit 1

CMD ["uvicorn", "lead_priority.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
