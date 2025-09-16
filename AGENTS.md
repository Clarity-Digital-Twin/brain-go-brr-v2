# Repository Guidelines

## Project Structure & Module Organization
- `src/experiment/`: Core code (`schemas.py`, `data.py`, `models.py`, `pipeline.py`, `infra.py`).
- `src/cli.py`: CLI entry point for running/validating configs.
- `configs/`: YAML experiment configs (e.g., `configs/local.yaml`).
- `tests/`: Pytest suite (unit, integration, pipeline, schema tests).
- `notebooks/`: Exploratory analysis; keep lightweight and data-agnostic.
- `data/`, `results/`, `cache/`: Artifacts and datasets (git-ignored).

## Build, Test, and Development Commands
- Setup (Python 3.10+): `pip install -r requirements.txt` then `pre-commit install`.
- Run CLI: `python -m src.cli --config configs/local.yaml` (see `--help`).
- Validate config: `python -m src.cli validate configs/local.yaml`.
- Tests: `pytest -v` (markers: `-m unit`, `-m integration`; coverage: `--cov=src --cov-report=html`).
- Format/lint/type: `black src tests`; `isort src tests`; `flake8 src tests`; `mypy src`.

## Coding Style & Naming Conventions
- Python: 4-space indent, Black line length 88, isort profile Black.
- Lint/type: flake8 (ignore `E203`, `W503`), mypy strict settings in `pyproject.toml`.
- Naming: modules `snake_case.py`; classes `CamelCase`; functions/vars `snake_case`.
- Imports: standard → third-party → first-party (`experiment`), sorted by isort.

## Testing Guidelines
- Framework: pytest; tests live under `tests/` only.
- Discovery: files `test_*.py`, classes `Test*`, functions `test_*`.
- Marks: use `@pytest.mark.unit` / `integration` / `slow` as appropriate.
- Aim for meaningful coverage on core paths; add tests with new code.

## Commit & Pull Request Guidelines
- History is mixed; adopt Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`).
- Before PR: run `pre-commit run --all-files` and `pytest -v` locally.
- PRs must include: clear description, linked issue, test updates, and notes on configs/CLI changes.

## Security & Configuration Tips
- Do not commit secrets or large artifacts; keep data under `data/` and outputs under `results/`, `cache/`.
- Use YAML in `configs/`; prefer config-driven changes over hardcoded paths.
- Keep runs reproducible (respect seeds/devices from config).

## Caching & ExCa
- Cached stages use a hash over `{experiment, stage, _cache_context}`; change config or context to refresh.
- Data preparation injects `_cache_context=self.config.model_dump()` so config edits produce new cache keys.
- For custom cached stages, pass `_cache_context=...` explicitly; ensure returned values are picklable to persist.

## Agent-Specific Instructions
- Keep changes minimal and focused; preserve public APIs under `src/experiment/`.
- Follow tooling pinned in `pyproject.toml`; don’t alter formatting/type settings without discussion.
- If adding modules or flags, update docs and tests in the same PR.