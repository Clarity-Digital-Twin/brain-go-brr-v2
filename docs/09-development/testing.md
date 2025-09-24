# Testing

Commands

- `make t` — quick tests
- `make test` — full coverage
- `make test-gpu` — GPU-specific

Focus areas

- Adjacency assembly edge cases
- GNN vectorized path integration
- Resume correctness for V3 components

Test stability and performance tips

- Default timeouts and warnings are configured in `pyproject.toml` for faster, quieter runs.
- Keep unit tests memory‑safe by using small batches; fixtures use conservative defaults.
- Useful env vars:
  - `TEST_BATCH_SIZE=1` to constrain test batch size
  - `TEST_LOW_MEMORY=true` to skip memory‑intensive checks
  - WSL2: `UV_LINK_MODE=copy` for install, `data.num_workers: 0` in configs
- Debugging NaNs: set `BGB_NAN_DEBUG=1` to enable extra checks; `SEIZURE_MAMBA_FORCE_FALLBACK=1` to fallback Mamba to Conv1d

Summary of recent fixes

- Dynamic PE buffers are consistently registered (no attribute collisions) and numerically guarded; vectorized path has sign consistency and fallback to last valid PE.
- Lint and type checks enforced via `make q` (ruff + mypy).
