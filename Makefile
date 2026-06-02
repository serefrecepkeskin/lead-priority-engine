.PHONY: help venv install install-dev lint format typecheck test test-cov run docker-build docker-run pre-commit clean

PYTHON := python3.12
VENV   := .venv
BIN    := $(VENV)/bin

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  %-16s %s\n", $$1, $$2}'

venv:  ## Create Python 3.12 virtualenv at .venv
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install -U pip wheel

install: venv  ## Install runtime deps + package (editable)
	$(BIN)/pip install -r requirements.txt
	$(BIN)/pip install -e .

install-dev: venv  ## Install dev deps + package + register pre-commit hooks
	$(BIN)/pip install -r requirements-dev.txt
	$(BIN)/pip install -e .
	$(BIN)/pre-commit install

lint:  ## Ruff lint check
	$(BIN)/ruff check src tests scripts

format:  ## Ruff format + autofix
	$(BIN)/ruff format src tests scripts
	$(BIN)/ruff check --fix src tests scripts

typecheck:  ## mypy strict on src + scripts/datagen
	$(BIN)/mypy

test:  ## Run tests
	$(BIN)/pytest

test-cov:  ## Run tests with coverage report
	$(BIN)/pytest --cov=lead_priority --cov-report=html --cov-report=term-missing

run:  ## Start FastAPI dev server (will exist once api/main.py is written)
	$(BIN)/uvicorn lead_priority.api.main:app --reload --host 0.0.0.0 --port 8000

docker-build:  ## Build Docker image (Dockerfile lands with the API)
	docker build -t lead-priority-engine:latest -f Dockerfile .

docker-run:  ## Run Docker container
	docker run --rm -p 8000:8000 lead-priority-engine:latest

pre-commit:  ## Run all pre-commit hooks against all files
	$(BIN)/pre-commit run --all-files

clean:  ## Remove caches and build artifacts
	rm -rf .ruff_cache .mypy_cache .pytest_cache htmlcov .coverage build dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name "*.egg-info" -prune -exec rm -rf {} +
