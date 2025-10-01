# Environment Variables

**Last Updated**: October 1, 2025 (v3.4.1)
**Source**: `src/brain_brr/utils/env.py`

## Critical for Production (v3.4.1)

**REQUIRED for stable training**:
```bash
export BGB_SANITIZE_GRADS=1  # Gradient NaN protection (REQUIRED)
export BGB_NAN_DEBUG=1       # Loss monitoring (RECOMMENDED)
```

**These are automatically set by Modal** (`deploy/modal/app.py:720-723`)

---

## Core Controls

- `BGB_SMOKE_TEST=1` — enable smoke shortcuts (skip sampling, relax checks)
- `BGB_LIMIT_FILES=N` — limit file count for quick runs
- `BGB_FORCE_MANIFEST_REBUILD=1` — delete and rebuild manifest on start

Debugging and stability

- `BGB_NAN_DEBUG=1` — extra logging if loss or grads misbehave
- `BGB_NAN_DEBUG_MAX=K` — limit debug prints per epoch (default 10)
- `BGB_DISABLE_TQDM=1` — disable progress bars (Modal auto)
- `BGB_DISABLE_TB=1` — disable TensorBoard writer
- `BGB_SANITIZE_INPUTS=1` — clamp/sanitize inputs in training loop (debug)
- `BGB_ANOMALY_DETECT=1` — enable torch autograd anomaly detection

Checkpoint cadence (resume workflows)

- `BGB_MID_EPOCH_MINUTES=M` — mid‑epoch checkpoint interval (minutes)
- `BGB_MID_EPOCH_KEEP=K` — retain at most K mid‑epoch checkpoints

Model toggles

- `SEIZURE_MAMBA_FORCE_FALLBACK=1` — force Conv1d fallback instead of CUDA kernels (debug only)
- `BGB_FORCE_TCN_EXT=1` — force internal TCN implementation (bypass ext)

WSL2 and packaging

- `UV_LINK_MODE=copy` — safer linking mode for uv on Windows filesystems

Model and stability toggles

- `BGB_DEBUG_FINITE=1` — enable assert_finite checks in critical tensors (debug only)
- `BGB_SAFE_CLAMP=1` — enable extra activation clamping (debug only)
- `BGB_SAFE_CLAMP_MIN=-10.0` — minimum clamp value when safe_clamp enabled
- `BGB_SAFE_CLAMP_MAX=10.0` — maximum clamp value when safe_clamp enabled

## Training Safety/Debug (CRITICAL)

- `BGB_SANITIZE_GRADS=1` — **REQUIRED** for v3.4.1 stability (clamp/replace NaN/Inf gradients)
- `BGB_SKIP_OPT_STEP_ON_NAN=1` — skip optimizer step if NaN detected (optional, debug only)

Performance testing (tests/performance)

- `BGB_PERF_ALLOW_GPU=1` — allow GPU usage in performance tests
- `BGB_PERF_THREADS=N` — set CPU thread count in performance tests
- `BGB_PERF_TOLERANCE_FACTOR=X.Y` — tolerance factor for perf tests (default 1.2)
- `BGB_PERF_STRICT_MODE=1` — disable tolerance slack (strict comparisons)

Logging configuration

- `BGB_LOG_LEVEL=INFO|DEBUG|WARNING|ERROR` — logging verbosity (default: INFO)
- `BGB_LOG_FILE=/path/to/file.log` — optional log file output
- `BGB_LOG_FORMAT=rich|simple|json` — output format (default: auto-detect, simple in CI/Modal)
- `BGB_LOG_EVERY_N_STEPS=50` — log training progress every N steps (default: 50)
- `BGB_LOG_RING_BUFFER_SIZE=1000` — size of in-memory log buffer (default: 1000)
- `BGB_FORCE_SIMPLE=1` — force simple logging format (no Rich)
- `BGB_FORCE_RICH=0` — disable Rich format even if available

## Modal-Specific (Auto-Set)

Modal automatically sets these in `deploy/modal/app.py`:

```python
# Memory allocator (XID 31 fix)
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

# Unique Triton cache per run (prevents stale kernels)
TRITON_CACHE_DIR=f"/tmp/triton_cache_run_{run_id}"
TORCHINDUCTOR_CACHE_DIR=f"/tmp/tii_cache_run_{run_id}"

# Enhanced logging
BGB_LOG_EVERY_N_STEPS=10

# NaN protection
BGB_SANITIZE_GRADS=1
BGB_NAN_DEBUG=1
```

## Important Notes

- Environment variables are **cached at import time** by `src.brain_brr.utils.env.EnvConfig` to support `torch.compile`. Restart the process to apply changes.
- `BGB_LIMIT_FILES` is honored by both training and `build-cache` (unless `--limit-files` is passed on the CLI).
- **v3.4.1**: `BGB_SANITIZE_GRADS=1` is **REQUIRED** for stable training on PyTorch 2.5.0.
