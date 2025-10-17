# Environment Variables

**Last Updated**: October 4, 2025 (v3.6.1)
**Source of Truth**: [`src/brain_brr/utils/env.py`](../../src/brain_brr/utils/env.py)

## Key Takeaways

- 🛡️ **Primary gradient protection** is `training.gradient_clip` in your config (default: `0.5`).
- 🛠️ Environment flags are **optional debugging tools**. Enable them only when you need extra visibility.
- 📘 See [`docs/08-operations/gradient-protection-guide.md`](../08-operations/gradient-protection-guide.md) for the full protection strategy.

---

## Core Training Controls

- `BGB_SMOKE_TEST=1` — Enable smoke-mode shortcuts (3 files, relaxed checks)
- `BGB_LIMIT_FILES=N` — Limit dataset size for quick experiments
- `BGB_FORCE_MANIFEST_REBUILD=1` — Rebuild cache manifest on startup
- `BGB_DISABLE_TQDM=1` — Disable progress bars (set automatically on Modal)
- `BGB_DISABLE_TB=1` — Disable TensorBoard writer

## Debugging & Stability

- `BGB_NAN_DEBUG=1` — Log NaN/Inf tensors with additional context (recommended when troubleshooting)
- `BGB_DEBUG_FINITE=1` — Enable assert_finite() checks inside the model (slower; debug only)
- `BGB_ANOMALY_DETECT=1` — Turn on `torch.autograd.set_detect_anomaly(True)` (very slow; use sparingly)
- `BGB_SANITIZE_GRADS=1` — **Optional**: zero-out/log non-finite gradients after `backward()` (disabled by default)

> ❗️ `BGB_SANITIZE_GRADS` is no longer required for normal training. Gradient clipping handles stability. Enable the flag only when you want to investigate why gradients went non-finite.

### DEPRECATED/UNUSED Flags (Documented but Never Implemented)

The following flags were documented in various places but **never actually checked in the code**. They are listed here for historical reference:

- ❌ `BGB_SANITIZE_INPUTS` — **REMOVED**: Never implemented. Input sanitization masks data quality issues. Use preprocessing outlier clipping (±10σ in `preprocess.py`) instead.
- ❌ `BGB_SKIP_OPT_STEP_ON_NAN` — **REMOVED**: Never implemented. Skipping optimizer steps breaks LR schedules and distributed training. Investigate root cause instead.
- ❌ `BGB_SAFE_CLAMP` — **REMOVED**: Never implemented. Global activation clamping indicates architectural problems. LayerNorm is the correct solution (always enabled via config).

**Note**: Gradient clipping (`training.gradient_clip: 0.5` in config) is the REAL protection mechanism and has always been the primary safeguard.

## Checkpoint Cadence

- `BGB_MID_EPOCH_MINUTES=M` — Save mid-epoch checkpoints every _M_ minutes
- `BGB_MID_EPOCH_KEEP=K` — Keep at most _K_ mid-epoch checkpoints (default: 2)

## Model Toggles

- `SEIZURE_MAMBA_FORCE_FALLBACK=1` — Force Conv1d fallback instead of CUDA kernels (debug only)
- `BGB_FORCE_TCN_EXT=1` — Force internal TCN implementation (bypass extension module)

## Performance Testing

**Scope:** Performance test suite only (`tests/performance/`) — NOT used during training

- `BGB_PERF_ALLOW_GPU=1` — Allow GPU usage in performance tests
- `BGB_PERF_THREADS=N` — Pin CPU thread count for perf tests
- `BGB_PERF_TOLERANCE_FACTOR=X.Y` — Set tolerance factor for perf comparisons (default: 1.2)
- `BGB_PERF_STRICT_MODE=1` — Disable the tolerance slack (strict comparisons)

## Logging Configuration

- `BGB_LOG_LEVEL=INFO|DEBUG|WARNING|ERROR` — Global logging verbosity (default: INFO)
- `BGB_LOG_FILE=/path/to/file.log` — Write logs to a file in addition to stdout
- `BGB_LOG_FORMAT=rich|simple|json` — Output format (default: auto-detect; simple in CI/Modal)
- `BGB_LOG_EVERY_N_STEPS=N` — Progress logging cadence (default: 50 local, Modal overrides to 10)
- `BGB_LOG_RING_BUFFER_SIZE=1000` — In-memory log buffer size
- `BGB_FORCE_SIMPLE=1` — Force simple logging format (no Rich)
- `BGB_FORCE_RICH=0` — Disable Rich formatting even if available

## Modal-Specific Defaults

Modal sets environment variables automatically in [`deploy/modal/app.py`](../../deploy/modal/app.py):

```python
# Memory allocator tweaks (prevent XID 31)
PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:512"

# Unique Triton/Inductor cache per run
TRITON_CACHE_DIR = f"/tmp/triton_cache_run_{run_id}"
TORCHINDUCTOR_CACHE_DIR = f"/tmp/tii_cache_run_{run_id}"

# Logging
BGB_LOG_EVERY_N_STEPS = "10"              # Faster logging on A100
BGB_NAN_DEBUG = "1"                       # Always enabled for cloud debugging

# Training Control
BGB_WALL_CLOCK_LIMIT_S = "82800"          # 23h limit (1h safety margin before Modal 24h kill)
BGB_DISABLE_TQDM = "1"                    # Disable progress bars in cloud logs
BGB_LIMIT_FILES = <unset for full training> # Explicitly removed to prevent accidents
```

> Modal no longer forces `BGB_SANITIZE_GRADS=1`. Gradient clipping already keeps training stable.

## Notes

- Environment variables are cached at import time by `EnvConfig` for `torch.compile` compatibility. **Restart the process** after changing any `BGB_*` flag.
- `BGB_LIMIT_FILES` affects both training and cache-building (unless a CLI flag overrides it).
- For the full protection story—including clipping, sanitization, and architectural safeguards—read the [Gradient Protection Guide](../08-operations/gradient-protection-guide.md).
