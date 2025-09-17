# Seizure Detection Pipeline - Modern ML Makefile
.PHONY: help install test lint format train clean setup hooks

# Colors for beautiful output
CYAN := \033[0;36m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

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
	uv sync --all-extras
	@echo "${GREEN}✓ Dependencies installed${NC}"

dev: ## Install dev dependencies and pre-commit hooks
	@echo "${CYAN}Setting up development environment...${NC}"
	uv sync --all-extras
	uv run pre-commit install
	@echo "${GREEN}✓ Development environment ready${NC}"

test: ## Run tests with coverage
	@echo "${CYAN}Running tests...${NC}"
	uv run pytest -n auto --cov=src --cov-report=term-missing --cov-report=html

test-fast: ## Run tests without coverage (faster)
	@echo "${CYAN}Running fast tests...${NC}"
	uv run pytest -n auto -x

lint: ## Run ruff linter
	@echo "${CYAN}Linting code...${NC}"
	uv run ruff check src/ tests/ --fix

format: ## Format code with ruff
	@echo "${CYAN}Formatting code...${NC}"
	uv run ruff format src/ tests/

type-check: ## Run mypy type checking
	@echo "${CYAN}Type checking...${NC}"
	uv run mypy src/

quality: lint format type-check ## Run all code quality checks
	@echo "${GREEN}✓ All quality checks passed${NC}"

train-local: ## Train model with local config
	@echo "${CYAN}Training with local config...${NC}"
	uv run python -m src.experiment.pipeline --config configs/local.yaml

train: train-prod ## Alias: full training with production config

train-prod: ## Train model with production config
	@echo "${CYAN}Training with production config...${NC}"
	uv run python -m src.experiment.pipeline --config configs/production.yaml

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
	uv sync --all-extras
	uv run pre-commit install
	@echo "${GREEN}✓ Project ready!${NC}"

hooks: ## Run pre-commit hooks on all files
	@echo "${CYAN}Running pre-commit hooks...${NC}"
	uv run pre-commit run --all-files

update: ## Update all dependencies
	@echo "${CYAN}Updating dependencies...${NC}"
	uv lock --upgrade
	uv sync --all-extras
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
