# Lead Priority Engine

A two-part system for ranking sales leads: a tabular model that predicts conversion, and a sentiment / intent classifier on interaction text. The two signals combine into a single priority score that a sales rep can act on.

Status: phases 0 (synthetic sentiment training data + leakage diagnostics), 1 (EDA + shared feature pipeline), and 2 (LR baseline + LightGBM lead scoring model) are complete; sentiment classifier and the FastAPI service are next.

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

## Configuration

Runtime config lives in `.env` (gitignored) and is loaded by `src/lead_priority/settings.py` via `pydantic-settings`. `.env.example` is the committed template.

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
| 3 | Sentiment / intent classifier (XLM-R + LLM baseline) | _coming: `docs/3_sentiment_classifier.docx`_ | _coming: `notebooks/3_sentiment_classifier.ipynb`_ |
| 4 | Combined priority score | _coming: `docs/4_priority_score.docx`_ | _(no notebook — small wiring step)_ |

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
