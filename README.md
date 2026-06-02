# Lead Priority Engine

A two-part system for ranking sales leads: a tabular model that predicts conversion, and a sentiment / intent classifier on interaction text. The two signals combine into a single priority score that a sales rep can act on.

Status: phase 1 (synthetic sentiment training data + leakage diagnostics) is complete; tabular lead scoring, sentiment classifier, and the FastAPI service are next.

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
    artifacts/            trained models + leakage report (gitignored except .gitkeep)
    docs/                 written deliverables
    .github/workflows/    CI (ruff + mypy + pytest)

## Configuration

Runtime config lives in `.env` (gitignored) and is loaded by `src/lead_priority/settings.py` via `pydantic-settings`. `.env.example` is the committed template.

## Documentation

Detailed write-ups live under `docs/`. The README intentionally stays short —
read the relevant doc for depth.

- **`docs/synthetic_data_and_leakage.docx`** — how the sentiment training notes were synthesized from the X Education tabular set, why this design avoids both label and temporal leakage (engagement-score derivation, prompt sandboxing, banned-phrase list), and the numerical validation: 5-fold AUC, Cramér's V, mutual information, crosstab, and a per-column AUC sweep that flags `Tags` / `Lead Quality` / `Last Notable Activity` as outcome-leaking columns to drop from lead scoring.

More docs will land here as each phase ships (scoring model card, sentiment classifier notes, API contract).
