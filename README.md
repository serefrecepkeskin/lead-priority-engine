# Lead Priority Engine

A two-part system for ranking sales leads: a tabular model that predicts conversion, and a sentiment / intent classifier on interaction text. The two signals combine into a single priority score that a sales rep can act on.

This commit is scaffolding only. More to come.

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

    src/lead_priority/    package code (api, data, features, models, utils)
    tests/                pytest suite
    notebooks/            EDA and experiments
    scripts/              CLI tools
    data/                 raw inputs (processed/ subdirs gitignored)
    artifacts/            trained model files (contents gitignored)
    docs/                 design notes, decisions, results
    .github/workflows/    CI (ruff + mypy + pytest)

## Configuration

Runtime config lives in `.env` (gitignored) and is loaded by `src/lead_priority/settings.py` via `pydantic-settings`. `.env.example` is the committed template.
