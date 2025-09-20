# Seizure Detection Pipeline - Modern ML Makefile
.PHONY: help install test lint format train clean setup hooks

# Colors for beautiful output
CYAN := \033[0;36m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

# UV defaults for WSL/cross-filesystem performance + permission stability
export UV_LINK_MODE ?= copy
export UV_CACHE_DIR ?= .uv_cache

help: ## Show this help message
	@echo '${CYAN}Seizure Detection Pipeline${NC}'
	@echo ''
	@echo 'Usage:'
	@echo '  ${GREEN}make${NC} ${YELLOW}<target>${NC}'
	@echo ''
	@echo 'Targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  ${GREEN}%-15s${NC} %s\n", $$1, $$2}'

install: ## Install all dependencies with uv
	@echo "${CYAN}Installing dependencies...${NC}"
	uv sync
	@echo "${GREEN}✓ Dependencies installed${NC}"

dev: ## Install dev dependencies and pre-commit hooks
	@echo "${CYAN}Setting up development environment...${NC}"
	uv sync
	uv run pre-commit install
	@echo "${GREEN}✓ Development environment ready${NC}"

# Detect available test runner (professional pattern from Google/DeepMind)
PYTEST := $(if $(wildcard .venv/bin/pytest),.venv/bin/pytest,uv run pytest)

test: ## Run tests with coverage
	@echo "${CYAN}Running non-serial tests (xdist)...${NC}"
	$(PYTEST) -n auto -m "not serial" --cov=src --cov-append --cov-report=term-missing:skip-covered
	@echo "${CYAN}Running serial tests...${NC}"
	$(PYTEST) -n 0 -m serial --cov=src --cov-append --cov-report=term-missing:skip-covered --cov-report=html

test-fast: ## Run tests without coverage (faster)
	@echo "${CYAN}Running fast tests...${NC}"
	$(PYTEST) -n 4 --dist=loadfile -q

test-cov: ## Run tests with full coverage report
	@echo "${CYAN}Running tests with full coverage...${NC}"
	$(PYTEST) -n auto --cov=src --cov-report=term-missing --cov-report=html

test-gpu: ## Run tests optimized for GPU (serial)
	@echo "${CYAN}Running GPU tests (serial)...${NC}"
	$(PYTEST) -n 1 -v -k "mamba or cuda"

test-cpu: ## Run CPU tests in parallel
	@echo "${CYAN}Running CPU tests (parallel)...${NC}"
	$(PYTEST) -n 4 --dist=loadfile -k "not (mamba or cuda)" -q

# Detect available tools (professional pattern)
RUFF := $(if $(wildcard .venv/bin/ruff),.venv/bin/ruff,uv run ruff)
MYPY := $(if $(wildcard .venv/bin/mypy),.venv/bin/mypy,uv run mypy)

lint: ## Run ruff linter
	@echo "${CYAN}Linting code...${NC}"
	$(RUFF) check src/ tests/ --fix

lint-fix: ## Fix all lint issues and format code
	@echo "${CYAN}Fixing lint issues and formatting...${NC}"
	$(RUFF) check --fix src/ tests/
	$(RUFF) format src/ tests/

format: ## Format code with ruff
	@echo "${CYAN}Formatting code...${NC}"
	$(RUFF) format src/ tests/

type-check: ## Run mypy type checking
	@echo "${CYAN}Type checking...${NC}"
	$(MYPY) src/

quality: lint format type-check ## Run all code quality checks
	@echo "${GREEN}✓ All quality checks passed${NC}"

train-local: ## Train model with local config
	@echo "${CYAN}Training with local config...${NC}"
	uv run python -m src train configs/local.yaml

train: train-prod ## Alias: full training with production config

train-prod: ## Train model with production config
	@echo "${CYAN}Training with production config...${NC}"
	uv run python -m src train configs/production.yaml

clean: ## Clean all artifacts
	@echo "${CYAN}Cleaning artifacts...${NC}"
	rm -rf .ruff_cache/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf results/models/*
	rm -rf results/metrics/*
	rm -rf results/plots/*
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "${GREEN}✓ Cleaned${NC}"

setup: ## Initial project setup
	@echo "${CYAN}Setting up project...${NC}"
	@command -v uv >/dev/null 2>&1 || { echo "${RED}uv not installed. Install from: https://github.com/astral-sh/uv${NC}" >&2; exit 1; }
	uv sync
	uv run pre-commit install
	@echo "${GREEN}✓ Project ready!${NC}"

hooks: ## Run pre-commit hooks on all files
	@echo "${CYAN}Running pre-commit hooks...${NC}"
	.venv/bin/pre-commit run --all-files || uv run pre-commit run --all-files

update: ## Update all dependencies
	@echo "${CYAN}Updating dependencies...${NC}"
	uv lock --upgrade
	uv sync
	@echo "${GREEN}✓ Dependencies updated${NC}"

notebook: ## Start Jupyter notebook
	@echo "${CYAN}Starting Jupyter...${NC}"
	uv run jupyter lab --notebook-dir=notebooks/

tensorboard: ## Start TensorBoard
	@echo "${CYAN}Starting TensorBoard...${NC}"
	uv run tensorboard --logdir=results/

# Development shortcuts
t: test-fast ## Shortcut for test-fast
f: format ## Shortcut for format
l: lint ## Shortcut for lint
q: quality ## Shortcut for quality checks
