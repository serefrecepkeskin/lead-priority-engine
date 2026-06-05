# Lead Priority Engine

> Türkçe için: [`README.tr.md`](README.tr.md)

A two-part system for ranking sales leads: a tabular model that predicts conversion, and a sentiment / intent classifier on interaction text. The two signals combine into a single priority score that a sales rep can act on.

Status: phases 0 through 5 are complete — synthetic sentiment training data + leakage diagnostics, EDA + shared feature pipeline, LR baseline + LightGBM lead scoring, OpenRouter LLM sentiment classifier, combined priority score, and the FastAPI service + Docker deployment. Ready for review.

## What is the project

The runtime project is **`src/lead_priority/`** — that is the package that gets installed, containerized, and served. Everything else in the tree is supporting material: `notebooks/` and `docs/` explain how we got here, `scripts/train_*` and `scripts/fit_*` produced the fitted artifacts once (offline, not run on each request), `data/` holds the inputs, and `artifacts/` holds the fitted models the runtime loads at boot.

## Quick start

One-shot interactive installer (stdlib-only, no prerequisites beyond Python 3.12):

    python3 deploy/setup.py

It copies `.env` from the template, prompts for `OPEN_ROUTER_API_KEY` if missing, builds the Docker image (or a venv), and smoke-tests `/healthz` + `/score`. If something fails it points at [`docs/5_fastapi_serving_and_deployment.docx`](docs/5_fastapi_serving_and_deployment.docx) §9 for the manual recovery table.

### Getting an OpenRouter API key (free — no credit card needed)

The runtime uses the **`z-ai/glm-4.5-air:free`** model on OpenRouter, which is on the **free tier** — you do **not** need to add a payment method or top up any credit balance to run this project. The free tier has a daily request cap that is more than enough for evaluation.

1. Go to **<https://openrouter.ai/>** and sign up (Google / GitHub / email — all free).
2. Open the keys page: **<https://openrouter.ai/keys>**.
3. Click **Create Key**, give it any name (e.g. `lead-priority-engine`), and copy the value (starts with `sk-or-...`).
4. Paste it into `.env` next to `OPEN_ROUTER_API_KEY=` (or let `python3 deploy/setup.py` prompt you for it).

That's it — no billing setup, no credits. If the daily quota is exhausted, `/score` automatically falls back to neutral sentiment so the service stays usable; you can simply retry the next day.

## Setup

Python 3.12 is required.

    make install-dev
    cp .env.example .env   # then edit as needed

`install-dev` creates `.venv`, installs deps, installs the package in editable mode, and registers the pre-commit hooks.

## Daily commands

    make lint        # ruff check
    make format      # ruff format + autofix
    make typecheck   # mypy strict on src/
    make test        # pytest
    make pre-commit  # run all hooks against the whole tree
    make run         # FastAPI dev server (once the API exists)

## CI

GitHub Actions runs ruff + mypy + pytest on every push and pull request. The workflow is at `.github/workflows/` and live runs are visible at [github.com/SerefRecepKeskin/lead-priority-engine/actions](https://github.com/SerefRecepKeskin/lead-priority-engine/actions) — the easiest place to confirm the pipeline is green on the current commit before pulling the code.

## Layout

    src/lead_priority/    runtime package (api, core, infra, utils, settings)
    scripts/              CLI entry-points
    scripts/datagen/      offline data-generation modules (kept out of runtime)
    tests/                pytest suite
    notebooks/            EDA and experiments
    data/                 raw inputs + data/synthetic/ (LLM-generated notes)
    artifacts/            trained models + leakage report (mix of tracked summaries and joblibs)
    docs/                 written deliverables
    .github/workflows/    CI (ruff + mypy + pytest)

## What gets loaded from `artifacts/` at serving time

The FastAPI app does **no training on boot** — it only loads. Loaders live in [`src/lead_priority/api/deps.py`](src/lead_priority/api/deps.py) and are LRU-cached for the process lifetime.

- `feature_pipeline.joblib` — fitted feature transformer (loaded by `get_feature_transformer`)
- `lead_scoring_lgbm.joblib` — trained LightGBM lead-scoring model (filename overridable via the `LEAD_SCORING_MODEL` env var)
- `sentiment_predictions/glm-4-5-air_test.parquet` — precomputed sentiment labels used to build the `/leads/top` cache, so `/leads/top` does not call the LLM on every request
- `lead_scoring_metrics.json`, `sentiment_metrics.json`, `priority_metrics.json` — surfaced by `/readyz` (headline numbers only; missing files are tolerated)

Retraining lives in the notebooks and the offline `scripts/train_*`, `scripts/fit_*` helpers — never on the request path.

## Configuration

Runtime config lives in `.env` (gitignored) and is loaded by `src/lead_priority/settings.py` via `pydantic-settings`. `.env.example` is the committed template.

### LLM API keys — what is for what

The repo references two LLM providers. They serve completely different roles and a reviewer running the service needs to know which one is actually required:

- **OpenRouter (`OPEN_ROUTER_API_KEY`)** — the **only LLM used at serving time**. Powers Phase 3 sentiment classification through the `/score` endpoint. The configured model `z-ai/glm-4.5-air:free` is on OpenRouter's **free tier** — no billing setup needed. See the [Getting an OpenRouter API key](#getting-an-openrouter-api-key-free--no-credit-card-needed) section above for how to obtain one. If unset, `/score` falls back to neutral sentiment so the service stays usable; `/readyz` surfaces the missing key as 503.
- **Azure OpenAI (`AZURE_OPENAI_*`)** — used **only for offline data generation** (Phase 0 synthetic interaction notes, produced by `scripts/datagen/` and `scripts/generate_interactions.py`). The runtime package never imports it. The generated notes are already committed under `data/synthetic/`, so a reviewer running the service does **not** need an Azure key.

## Documentation

Each phase ships its own write-up under `docs/` and (where applicable) an
exploratory notebook under `notebooks/`. Files are numbered so the chronological
order is visible at a glance. The README intentionally stays short — click into
the relevant doc for depth.

### Phase 0 — Synthetic data + leakage

Synthetic interaction notes (TR / EN / Mix code-switching), labelling strategy, train→serve leakage diagnostics on the synthetic ↔ raw join.

📄 [`docs/0_synthetic_data_and_leakage.docx`](docs/0_synthetic_data_and_leakage.docx) · 📓 [`notebooks/0_leakage_analysis.ipynb`](notebooks/0_leakage_analysis.ipynb)

### Phase 1 — EDA + feature engineering

Conversion-rate distribution, class imbalance, missing-data patterns, source-level conversion gaps; derived features (`channel_diversity_count`, `total_time_per_visit`, `days_since_last_activity`, …) with the rationale for each.

📄 [`docs/1_eda_and_feature_engineering.docx`](docs/1_eda_and_feature_engineering.docx) · 📓 [`notebooks/1_eda_and_feature_engineering.ipynb`](notebooks/1_eda_and_feature_engineering.ipynb)

### Phase 2 — Lead scoring model

LR baseline (interpretability) vs LightGBM (modern, hyperparameter-tuned); ROC / PR / accuracy + calibration plot + threshold sweep + top-20% gain & lift chart; bootstrap-CI paired test; SHAP feature importance.

📄 [`docs/2_lead_scoring.docx`](docs/2_lead_scoring.docx) · 📓 [`notebooks/2_lead_scoring.ipynb`](notebooks/2_lead_scoring.ipynb)

### Phase 3 — Sentiment / intent classifier

Four attitudes (`positive_engagement` / `objection` / `neutral` / `disengaged`); OpenRouter open-source LLM, zero/few-shot prompt (XLM-R / DistilBERT fine-tune alternative discussed); TR + EN + Mix handling; per-class + per-language confusion matrix + macro-F1 + bootstrap CI; fairness & ethics analysis.

📄 [`docs/3_sentiment_classifier.docx`](docs/3_sentiment_classifier.docx) · 📓 [`notebooks/3_sentiment_classifier.ipynb`](notebooks/3_sentiment_classifier.ipynb)

### Phase 4 — Combined priority score

Weighted-average mix of `P(conversion)` and sentiment ordinal; weight rationale, sensitivity sweep, meta-model alternative consciously declined.

📄 [`docs/4_priority_score.docx`](docs/4_priority_score.docx) · 📓 [`notebooks/4_priority_demo.ipynb`](notebooks/4_priority_demo.ipynb)

### Phase 5 — FastAPI service + Docker deployment

`POST /score` + `GET /leads/top` endpoint contracts, structured JSON logging + request-ID middleware, Dockerfile multi-stage build, integration tests, manual recovery table; production notes — feature-drift monitoring, retraining cadence, sales-rep feedback loop, false-positive cost framing, 3-day-budget next steps.

📄 [`docs/5_fastapi_serving_and_deployment.docx`](docs/5_fastapi_serving_and_deployment.docx) · (no notebook)

**Doc format contract** (every numbered docx follows the same shape):

- Clickable table of contents at the top
- Third-person / passive Turkish — written for the reviewer to read, not the author
- Technical concepts explained in plain language first, formal notation second
- Section numbering `1. → 1.1 → 1.2 → 2.` etc.
- Numerical results inside tables, not buried in prose

**Notebook contract:**

- Numbered prefix matches the corresponding docx
- Markdown cells between code cells explain *what* the next block does and *why* — a reviewer should be able to read top-to-bottom without running it
- First markdown cell links back to the phase docx (`docs/N_*.docx`)

## API examples

> Interactive Swagger UI: **http://127.0.0.1:8000/docs** — the “Try it out” button on `POST /score` is prefilled with a complete sample payload, so you can fire a real request without crafting JSON by hand. Redoc is also available at `/redoc`.

`POST /score` — combined priority for a single lead. The request payload is `examples/score_request.json`:

```bash
$ curl -X POST http://localhost:8000/score \
       -H "Content-Type: application/json" \
       -d @examples/score_request.json | jq .
{
  "p_conversion": 0.7234,
  "sentiment": {
    "predicted_attitude": "objection",
    "sentiment_score": 0.65,
    "sentiment_unavailable": false,
    "latency_ms": 412.5
  },
  "priority": 0.6940,
  "weights": { "weight_conversion": 0.6, "weight_sentiment": 0.4 },
  "model_versions": { "feature_pipeline_schema": 2, "lead_scoring_kind": "lightgbm", "sentiment_model_name": "z-ai/glm-4.5-air:free" },
  "request_id": "5f2e1c3a..."
}
```

`GET /leads/top?n=N` — top-N leads sorted by combined priority, served from the in-memory cache built at startup (no LLM call per request):

```bash
$ curl 'http://localhost:8000/leads/top?n=3' | jq .
{
  "count": 3,
  "total_available": 924,
  "leads": [
    { "lead_id": "74878c4b-...", "p_conversion": 0.994828, "predicted_attitude": "positive_engagement", "sentiment_score": 1.0, "priority": 0.996897, "language": "tr" },
    { "lead_id": "2caa32d0-...", "p_conversion": 0.994314, "predicted_attitude": "positive_engagement", "sentiment_score": 1.0, "priority": 0.996588, "language": "tr" },
    { "lead_id": "bd5ca024-...", "p_conversion": 0.993130, "predicted_attitude": "positive_engagement", "sentiment_score": 1.0, "priority": 0.995878, "language": "en" }
  ],
  "model_versions": { "feature_pipeline_schema": 2, "lead_scoring_kind": "lightgbm", "sentiment_model_name": "z-ai/glm-4.5-air:free" },
  "request_id": "d3baf801..."
}
```

`min_priority` and `n` (up to 924) are the supported query params; a smaller `n` is a slice off the front of the same sorted list.

## Deployment

For setup, service design, the failure-mode troubleshooting table, and manual recovery steps, see **[`docs/5_fastapi_serving_and_deployment.docx`](docs/5_fastapi_serving_and_deployment.docx)**. The one-shot `deploy/setup.py` covers the happy path; the docx covers everything else (Docker vs venv modes, endpoint specs, recovery procedures referenced by the installer when a step fails).

## Tests covering the serving surface

The FastAPI service is the surface that runs in production. These four tests prove it works end-to-end; the remaining tests under `tests/` are supporting tests for the model and feature layers.

| File | What it proves |
|---|---|
| [`tests/test_api_health.py`](tests/test_api_health.py) | `/healthz` liveness + `/readyz` model-loaded checks, request-ID middleware |
| [`tests/test_api_logging.py`](tests/test_api_logging.py) | Structured JSON logs are emitted with request-id correlation and exception stacks |
| [`tests/test_api_score.py`](tests/test_api_score.py) | `POST /score` happy path, request validation, graceful sentiment fallback when OpenRouter is unavailable / rate-limited |
| [`tests/test_api_top_leads.py`](tests/test_api_top_leads.py) | `GET /leads/top` sort, pagination (`n`), `min_priority` filter, served from the precomputed startup cache |

Run with `make test`. The other tests under `tests/` cover the model layer (lead scoring, sentiment, priority) and the feature pipeline — supporting, not the public serving surface.

## Project tree

```
lead-priority-engine/
├── src/lead_priority/          ← the actual project (installed, containerized, served)
│   ├── api/                    FastAPI app (transport layer)
│   │   ├── main.py             app factory + lifespan (warms models on boot)
│   │   ├── deps.py             LRU-cached loaders for everything read from artifacts/
│   │   ├── schemas.py          Pydantic request / response models
│   │   ├── errors.py           exception handlers (OpenRouter, config, validation)
│   │   ├── middleware.py       request-id + JSON access-log middleware
│   │   └── endpoints/
│   │       ├── health.py       /healthz, /readyz
│   │       ├── score.py        POST /score (combined priority for a single lead)
│   │       └── top_leads.py    GET /leads/top (precomputed cache)
│   ├── core/                   domain logic (no transport, no external IO)
│   │   ├── features/           feature pipeline (derive + transformers + persistence)
│   │   ├── inference/lead_scoring.py   LR / LightGBM wrapper
│   │   └── scoring/
│   │       ├── priority.py     weighted-average priority formula
│   │       └── sentiment_classes.py  SentimentClass + label-to-score map
│   ├── infra/                  adapters for external services
│   │   └── openrouter/sentiment.py   OpenRouter LLM sentiment classifier
│   ├── utils/
│   │   └── logging.py          JSON formatter + rotating file handler installer
│   └── settings.py             pydantic-settings loader (.env)
├── tests/                      pytest suite (see Tests section above for the API tests)
├── artifacts/                  fitted models + metrics read at serving time
│   ├── feature_pipeline.joblib
│   ├── lead_scoring_lgbm.joblib
│   ├── lead_scoring_lr.joblib                              (LR baseline; not loaded by default)
│   ├── sentiment_predictions/glm-4-5-air_test.parquet      (/leads/top cache source)
│   ├── lead_scoring_metrics.json / sentiment_metrics.json / priority_metrics.json
│   ├── feature_summary.json / leakage_report.json          (Phase 0–1 diagnostics)
│   └── figures/                plots embedded in the docx write-ups
├── data/
│   ├── Lead Scoring.csv        raw input
│   ├── Leads Data Dictionary.xlsx
│   └── synthetic/              committed LLM-generated interaction notes (Phase 0)
├── docs/                       numbered phase write-ups (0–5) — see Documentation table
├── notebooks/                  EDA + experiments matching the doc numbers
├── scripts/
│   ├── datagen/                offline Phase-0 synthetic-data tooling (NOT runtime)
│   ├── generate_interactions.py / evaluate_openrouter_sentiment.py
│   └── fit_feature_pipeline.py / train_lead_scoring.py    (offline training)
├── deploy/
│   └── setup.py                one-shot interactive installer (stdlib-only)
├── logs/                       rotating JSON service logs (gitignored)
├── examples/score_request.json example POST /score payload
├── Dockerfile                  multi-stage build (runtime only)
├── Makefile                    install-dev / lint / format / typecheck / test / run
├── .env.example                config template (loaded by src/lead_priority/settings.py)
└── pyproject.toml
```
