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

test: ## Run tests with coverage (excludes performance benchmarks)
	@echo "${CYAN}Running non-serial CPU tests (xdist)...${NC}"
	$(PYTEST) -n auto -m "not serial and not performance and not gpu" --cov=src --cov-append --cov-report=term-missing:skip-covered
	@echo "${CYAN}Running GPU tests (serial)...${NC}"
	$(PYTEST) -n 1 -m "gpu and not performance" --cov=src --cov-append --cov-report=term-missing:skip-covered
	@echo "${CYAN}Running serial tests (excluding performance and GPU)...${NC}"
	$(PYTEST) -n 0 -m "serial and not performance and not gpu" --cov=src --cov-append --cov-report=term-missing:skip-covered --cov-report=html

test-fast: ## Run tests without coverage (faster, excludes performance)
	@echo "${CYAN}Running fast tests (CPU only)...${NC}"
	$(PYTEST) -n 4 --dist=loadfile -m "not performance and not gpu and not serial" -q

test-cov: ## Run tests with full coverage report
	@echo "${CYAN}Running tests with full coverage...${NC}"
	$(PYTEST) -n auto -m "not performance and not gpu" --cov=src --cov-append --cov-report=term-missing:skip-covered
	@echo "${CYAN}Running GPU tests with coverage (serial)...${NC}"
	$(PYTEST) -n 1 -m "gpu and not performance" --cov=src --cov-append --cov-report=term-missing --cov-report=html

test-integration: ## Run only integration tests (excludes performance)
	@echo "${CYAN}Running non-GPU integration tests (parallel)...${NC}"
	$(PYTEST) -n auto -m "integration and not performance and not gpu" -v
	@echo "${CYAN}Running GPU integration tests (serial)...${NC}"
	$(PYTEST) -n 1 -m "integration and gpu and not performance" -v

test-performance: ## Run only performance benchmarks (serial)
	@echo "${CYAN}Running performance benchmarks (serial)...${NC}"
	$(PYTEST) -n 0 -m performance -v

test-gpu: ## Run tests optimized for GPU (serial)
	@echo "${CYAN}Running GPU tests (serial)...${NC}"
	$(PYTEST) -n 1 -v -k "mamba or cuda"

test-cpu: ## Run CPU tests in parallel
	@echo "${CYAN}Running CPU tests (parallel)...${NC}"
	$(PYTEST) -n 4 --dist=loadfile -k "not (mamba or cuda)" -q

test-safe: ## Run tests safely to avoid OOM (serial, excludes heavy models)
	@echo "${CYAN}Running tests safely (serial to avoid OOM)...${NC}"
	@echo "${CYAN}Step 1: Unit tests without heavy models...${NC}"
	$(PYTEST) tests/unit -n 1 -k "not (v3 or V3 or detector_from_config)" -m "not gpu and not performance" --tb=short -q
	@echo "${CYAN}Step 2: V3 tests in serial...${NC}"
	$(PYTEST) tests/unit/models/test_detector_v3.py -n 0 -m "not gpu and not performance" --tb=short
	@echo "${CYAN}Step 3: Lightweight integration tests...${NC}"
	$(PYTEST) tests/integration -n 1 -k "not (from_config or tcn_integration or gnn_integration)" -m "not gpu and not performance" --tb=short -q
	@echo "${GREEN}✅ Safe test run complete!${NC}"

test-edge: ## Run edge case tests for data robustness
	@echo "${CYAN}Running edge case tests...${NC}"
	$(PYTEST) tests/unit/data/test_io_edge_cases.py tests/unit/post/test_hysteresis_edge.py -v

test-clinical: ## Run clinical validation suite
	@echo "${CYAN}Running clinical validation...${NC}"
	$(PYTEST) tests/clinical/ -v

test-all: ## Run ALL tests including performance (comprehensive)
	@echo "${CYAN}Running ALL tests comprehensively...${NC}"
	$(PYTEST) -n auto -m "not serial and not gpu" --cov=src --cov-append
	@echo "${CYAN}Running GPU tests (serial)...${NC}"
	$(PYTEST) -n 1 -m "gpu" --cov=src --cov-append
	@echo "${CYAN}Running serial tests (including performance, excluding GPU)...${NC}"
	$(PYTEST) -n 0 -m "serial and not gpu" --cov=src --cov-append --cov-report=term-missing --cov-report=html

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

train-local: ## Train model with v2.6 local config
	@echo "${CYAN}Training with v2.6 stack (TCN+BiMamba+GNN+LPE)...${NC}"
	@echo "${YELLOW}NaN protections enabled: BGB_NAN_DEBUG=1, BGB_SANITIZE_INPUTS=1${NC}"
	BGB_NAN_DEBUG=1 BGB_SANITIZE_INPUTS=1 .venv/bin/python -m src train configs/local/train.yaml

smoke-local: ## Run local smoke test (1 epoch)
	@echo "${CYAN}Running v2.6 smoke test...${NC}"
	@echo "${YELLOW}NaN protections enabled: BGB_NAN_DEBUG=1, BGB_SANITIZE_INPUTS=1${NC}"
	BGB_NAN_DEBUG=1 BGB_SANITIZE_INPUTS=1 BGB_SMOKE_TEST=1 .venv/bin/python -m src train configs/local/smoke.yaml

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
	rm -f .coverage .coverage.*
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

setup-gpu: ## Setup GPU support with mamba-ssm and PyG (requires CUDA 12.1)
	@echo "${CYAN}Setting up GPU support for v2.6 stack...${NC}"
	@echo "${YELLOW}Checking CUDA versions...${NC}"
	@.venv/bin/python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')" || echo "${RED}PyTorch not installed${NC}"
	@nvcc --version 2>/dev/null | grep "release" || echo "${RED}CUDA toolkit not found!${NC}"
	@echo "${CYAN}Installing Mamba-SSM components...${NC}"
	@export CUDA_HOME=/usr/local/cuda-12.1 && \
		uv pip install --no-build-isolation causal-conv1d==1.4.0 && \
		uv pip install --no-build-isolation mamba-ssm==2.2.2
	@echo "${CYAN}Installing PyG with pre-built wheels...${NC}"
	@.venv/bin/pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
	@.venv/bin/pip install torch-geometric==2.6.1
	@echo "${CYAN}Installing TCN...${NC}"
	@uv pip install pytorch-tcn==1.2.3
	@echo "${CYAN}Verifying v2.6 stack...${NC}"
	@.venv/bin/python -c "from mamba_ssm import Mamba2; print('${GREEN}✓ Mamba-SSM working${NC}')" || echo "${RED}⚠️  Mamba-SSM failed${NC}"
	@.venv/bin/python -c "import torch_geometric; print(f'${GREEN}✓ PyG {torch_geometric.__version__} installed${NC}')" || echo "${RED}⚠️  PyG failed${NC}"
	@.venv/bin/python -c "import pytorch_tcn; print('${GREEN}✓ TCN installed${NC}')" || echo "${RED}⚠️  TCN failed${NC}"
	@echo "${GREEN}✓ v2.6 stack ready (TCN + BiMamba + GNN + LPE)${NC}"

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
ti: test-integration ## Shortcut for integration tests
tp: test-performance ## Shortcut for performance benchmarks
f: format ## Shortcut for format
l: lint ## Shortcut for lint
q: quality ## Shortcut for quality checks
s: smoke-local ## Shortcut for smoke test
g: setup-gpu ## Shortcut for GPU setup
