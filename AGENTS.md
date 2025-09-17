# Repository Guidelines

## Project Structure & Module Organization
- `src/experiment/`: Core code (`schemas.py`, `data.py`, `models.py`, `pipeline.py`, `infra.py`).
- `src/cli.py`: CLI entry point for running/validating configs.
- `configs/`: YAML experiment configs (e.g., `configs/local.yaml`).
- `tests/`: Pytest suite (unit, integration, pipeline, schema tests).
- `notebooks/`: Exploratory analysis; keep lightweight and data-agnostic.
- `data/`, `results/`, `cache/`: Artifacts and datasets (git-ignored).

## Build, Test, and Development Commands
- Setup (Python 3.11+): `make setup` (uses UV), or `make dev` for dev tools + hooks.
- Run CLI: `python -m src.cli --config configs/local.yaml` (validate/run commands to be added).
- Validate config: `python -m src.cli validate configs/local.yaml` (planned; not implemented yet).
- Tests: `make test` (markers: `-m unit`, `-m integration`; coverage: HTML + terminal).
- Format/lint/type: `make lint`; `make format`; `make type-check` (Ruff + mypy).

## Coding Style & Naming Conventions
- Python: 4-space indent; formatter via Ruff (`ruff format`), line length 100.
- Lint/type: Ruff rules (see `pyproject.toml`), mypy strict settings in `pyproject.toml`.
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
- Prefer UV workflows and Ruff for lint/format to stay consistent.
