# Test Suite Snapshot – 2025-10-15

**Branch:** `main`  
**Python test files:** 65  
**Total lines (Python tests + fixtures):** 15 344
**Last refreshed:** 2025‑10‑15 (v4.0.0 FLA production + WSL2 fix)

This document captures the state of the automated tests that ship with the repository.  
All metrics below are generated from the checked-in sources. Install the project
dependencies (including `torch`) before running `pytest`—the suite imports
`torch` during collection.

---

## Directory Overview

| Path              | Python files | Lines | Notes |
|-------------------|--------------|-------|-------|
| `tests/clinical/` | 2            | 891   | TAES scoring & EEG channel order validation |
| `tests/integration/` | 11       | 1 862 | End-to-end data/graph/model flows (`test_training_edge_cases.py`, `test_streaming.py`, …) |
| `tests/performance/` | 4        | 1 222 | GPU/CPU latency & memory benchmarks (marked with `@pytest.mark.performance`) |
| `tests/unit/`     |  42           | 10 037 | Fine-grained coverage for data, models, train loop, utils (see below) |
| `tests/train/`    | 2            | 684   | Training robustness tests (recording storage, validation memory) |
| Root helpers      | 4            | 648   | `conftest.py`, `gpu_memory_guard.py`, `test_config.py`, `__init__.py` |

### Selected Highlights

* **Unit → train/**:  
  - `test_losses.py`, `test_warmup.py`, `test_gradient_sanitization.py` backstop the training loop’s safety logic.  
  - `test_loop.py` verifies stepping, gradient scaling, and checkpointing (≈ 350 LOC).
* **Unit → models/**: comprehensive coverage of TCN, Mamba, fusion/clamping, graph adjacency, PR safeguards, and NaN handling (`test_nan_robustness.py`).
* **Integration**:  
  - `test_training_edge_cases.py` exercises OOM/NaN recovery logic.  
  - `test_streaming.py` validates real-time inference pipeline.  
  - `data/test_io_edge_cases.py` keeps I/O fallbacks honest.
* **Performance**: `test_latency.py` and `test_memory.py` benchmark throughput/VRAM; opt-in via `make test-performance`.
* **Clinical**: `test_taes_metrics.py` and `test_channel_order.py` guarantee that medical scoring and electrode ordering remain stable.

---

## Fixture & Marker Essentials

* Root `tests/conftest.py` defines 18 core fixtures (model factories, synthetic EEG, CLI runner, gradient sanitiser, etc.).  
  - `test_batch_size` + `TEST_MAX_BATCH_SIZE` (from `tests/test_config.py`) guard memory usage on both CPU and GPU.  
  - `cleanup_torch_resources` & `gpu_memory_guard.gpu_memory_limit` enforce VRAM hygiene between tests.
* Marker policy is documented in `tests/MARKER_POLICY.md`. Key rules:  
  - Every CUDA-dependent test must be annotated with `@pytest.mark.gpu`.  
  - Performance benchmarks carry both `@pytest.mark.performance` and `@pytest.mark.timeout(...)`.
* Local GPU adjustments (RTX 4090 defaults, env overrides such as `BGB_TEST_GPU_FRACTION`) live in `tests/GPU_ADJUSTMENTS.md`.

---

## Running the Suite

```bash
# Full test suite with coverage (CPU-safe, excludes GPU + performance)
make test

# Fast tests without coverage (use `make t` as shortcut)
make test-fast

# Safe during long GPU training sessions
make test-safe        # pytest -m "not performance and not gpu"

# Full performance benchmarks (GPU required)
make test-performance

# Collect all tests (requires torch installed)
pytest --collect-only
```

All of the invocations above respect `tests/test_config.py`, which auto-detects the GPU
model and adjusts batch sizes and thresholds. Override via environment variables if
you need to test different budgets (e.g., `TEST_BATCH_SIZE`, `BGB_TEST_GPU_FRACTION`).

---

## Regenerating This Snapshot

```bash
# Count Python files and lines (same script used for this snapshot)
python - <<'PY'
from pathlib import Path
root = Path("tests")
files = list(root.rglob("*.py"))
total_lines = sum(sum(1 for _ in open(f, encoding="utf-8")) for f in files)
print(f"Python files: {len(files)}")
print(f"Total lines: {total_lines}")
for subtree in ("clinical", "integration", "performance", "unit"):
    sf = [p for p in files if subtree in p.parts and p.parts[p.parts.index(subtree)] == subtree]
    lines = sum(sum(1 for _ in open(p, encoding=\"utf-8\")) for p in sf)
    print(f\"{subtree}: {len(sf)} files, {lines} lines\")
PY
```

Update this document whenever new test modules are added or large suites are refactored.
If you introduce/rename markers or fixtures, cross-check `MARKER_POLICY.md`,
`GPU_ADJUSTMENTS.md`, and the tables above to keep everything consistent.
