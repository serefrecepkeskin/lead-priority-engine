# Lead Priority Engine

> Türkçe için: [`README.tr.md`](README.tr.md)

A two-part system for ranking sales leads: a tabular model that predicts conversion, and a sentiment / intent classifier on interaction text. The two signals combine into a single priority score that a sales rep can act on.

Status: phases 0 through 5 are complete — synthetic sentiment training data + leakage diagnostics, EDA + shared feature pipeline, LR baseline + LightGBM lead scoring, OpenRouter LLM sentiment classifier, combined priority score, and the FastAPI service + Docker deployment. Ready for review.

## What is the project

The runtime project is **`src/lead_priority/`** — that is the package that gets installed, containerized, and served. Everything else in the tree is supporting material: `notebooks/` and `docs/` explain how we got here, `scripts/datagen/` and the `scripts/build_*`, `scripts/train_*`, `scripts/fit_*` helpers produced the synthetic notes / fitted artifacts / docx write-ups once (offline, not run on each request), `data/` holds the inputs, and `artifacts/` holds the fitted models the runtime loads at boot.

## Quick start

One-shot interactive installer (stdlib-only, no prerequisites beyond Python 3.12):

    python3 scripts/setup.py

It copies `.env` from the template, prompts for `OPEN_ROUTER_API_KEY` if missing, builds the Docker image (or a venv), and smoke-tests `/healthz` + `/score`. If something fails it points at [`docs/6_deployment.docx`](docs/6_deployment.docx) §9 for the manual recovery table.

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

    src/lead_priority/    runtime package (api, features, models, utils, settings)
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

- **OpenRouter (`OPEN_ROUTER_API_KEY`)** — the **only LLM used at serving time**. Powers Phase 3 sentiment classification through the `/score` endpoint. If unset, `/score` falls back to neutral sentiment so the service stays usable; `/readyz` surfaces the missing key as 503.
- **Azure OpenAI (`AZURE_OPENAI_*`)** — used **only for offline data generation** (Phase 0 synthetic interaction notes, produced by `scripts/datagen/` and `scripts/generate_interactions.py`). The runtime package never imports it. The generated notes are already committed under `data/synthetic/`, so a reviewer running the service does **not** need an Azure key.

## Documentation

Each phase ships its own write-up under `docs/` and (where applicable) an
exploratory notebook under `notebooks/`. Files are numbered so the chronological
order is visible at a glance. The README intentionally stays short — click into
the relevant doc for depth.

| # | Phase | Write-up | Notebook |
|---|---|---|---|
| 0 | Synthetic interaction data + leakage diagnostics | [`docs/0_synthetic_data_and_leakage.docx`](docs/0_synthetic_data_and_leakage.docx) | [`notebooks/0_leakage_analysis.ipynb`](notebooks/0_leakage_analysis.ipynb) |
| 1 | EDA + feature engineering | [`docs/1_eda_and_feature_engineering.docx`](docs/1_eda_and_feature_engineering.docx) | [`notebooks/1_eda_and_feature_engineering.ipynb`](notebooks/1_eda_and_feature_engineering.ipynb) |
| 2 | Lead scoring model (LR baseline + LGBM) | [`docs/2_lead_scoring.docx`](docs/2_lead_scoring.docx) | [`notebooks/2_lead_scoring.ipynb`](notebooks/2_lead_scoring.ipynb) |
| 3 | Sentiment / intent classifier (OpenRouter LLM zero/few-shot) | [`docs/3_sentiment_classifier.docx`](docs/3_sentiment_classifier.docx) | [`notebooks/3_sentiment_classifier.ipynb`](notebooks/3_sentiment_classifier.ipynb) |
| 4 | Combined priority score (weighted average) | [`docs/4_priority_score.docx`](docs/4_priority_score.docx) | [`notebooks/4_priority_demo.ipynb`](notebooks/4_priority_demo.ipynb) |
| 6 | FastAPI servisi + Docker deployment (service design + setup guide) | [`docs/6_deployment.docx`](docs/6_deployment.docx) | — |

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

## Deployment

For setup, service design, the failure-mode troubleshooting table, and manual recovery steps, see **[`docs/6_deployment.docx`](docs/6_deployment.docx)**. The one-shot `scripts/setup.py` covers the happy path; the docx covers everything else (Docker vs venv modes, endpoint specs, recovery procedures referenced by the installer when a step fails).

## Tests covering the case-study serving surface

The case study evaluates the FastAPI service. These four tests prove that surface works; the remaining tests under `tests/` are supporting tests for the model and feature layers.

| File | What it proves |
|---|---|
| [`tests/test_api_health.py`](tests/test_api_health.py) | `/healthz` liveness + `/readyz` model-loaded checks, request-ID middleware |
| [`tests/test_api_logging.py`](tests/test_api_logging.py) | Structured JSON logs are emitted with request-id correlation and exception stacks |
| [`tests/test_api_score.py`](tests/test_api_score.py) | `POST /score` happy path, request validation, graceful sentiment fallback when OpenRouter is unavailable / rate-limited |
| [`tests/test_api_top_leads.py`](tests/test_api_top_leads.py) | `GET /leads/top` sort, pagination (`n`), `min_priority` filter, served from the precomputed startup cache |

Run with `make test`. The other tests under `tests/` cover the model layer (lead scoring, sentiment, priority) and the feature pipeline — supporting, not the case-study serving surface.

## Project tree

```
lead-priority-engine/
├── src/lead_priority/          ← the actual project (installed, containerized, served)
│   ├── api/                    FastAPI app
│   │   ├── main.py             app factory + lifespan (warms models on boot)
│   │   ├── deps.py             LRU-cached loaders for everything read from artifacts/
│   │   ├── schemas.py          Pydantic request / response models
│   │   ├── errors.py           exception handlers (OpenRouter, config, validation)
│   │   ├── logging.py          JSON formatter + request-id middleware
│   │   └── endpoints/
│   │       ├── health.py       /healthz, /readyz
│   │       ├── score.py        POST /score (combined priority for a single lead)
│   │       └── top_leads.py    GET /leads/top (precomputed cache)
│   ├── features/               feature pipeline (derive + transformers + persistence)
│   ├── models/
│   │   ├── lead_scoring.py     LR / LightGBM wrapper
│   │   ├── sentiment.py        OpenRouter LLM sentiment classifier
│   │   └── priority.py         weighted-average priority formula
│   ├── utils/                  small shared helpers
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
├── docs/                       numbered phase write-ups (0–4, 6) — see Documentation table
├── notebooks/                  EDA + experiments matching the doc numbers
├── scripts/
│   ├── setup.py                one-shot interactive installer (stdlib-only)
│   ├── datagen/                offline Phase-0 synthetic-data tooling (NOT runtime)
│   ├── generate_interactions.py / evaluate_openrouter_sentiment.py
│   ├── fit_feature_pipeline.py / train_lead_scoring.py    (offline training)
│   └── build_*_docx.py / build_3_sentiment_notebook.py    (docx + notebook builders)
├── examples/score_request.json example POST /score payload
├── Dockerfile                  multi-stage build (runtime only)
├── Makefile                    install-dev / lint / format / typecheck / test / run
├── .env.example                config template (loaded by src/lead_priority/settings.py)
└── pyproject.toml
```
