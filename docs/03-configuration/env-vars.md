# Environment Variables

Core controls

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

Training safety/debug

- `BGB_SANITIZE_GRADS=1` — clamp/replace NaN/Inf gradients (**RECOMMENDED for TCN stability**)
- `BGB_SKIP_OPT_STEP_ON_NAN=1` — skip optimizer step if NaN detected (debug only)

Performance testing (tests/performance)

- `BGB_PERF_ALLOW_GPU=1` — allow GPU usage in performance tests
- `BGB_PERF_THREADS=N` — set CPU thread count in performance tests
- `BGB_PERF_TOLERANCE_FACTOR=X.Y` — tolerance factor for perf tests (default 1.2)
- `BGB_PERF_STRICT_MODE=1` — disable tolerance slack (strict comparisons)

Important notes

- Environment variables are cached at import time by `src.brain_brr.utils.env.EnvConfig` to support `torch.compile`. Restart the process to apply changes.
- `BGB_LIMIT_FILES` is honored by both training and `build-cache` (unless `--limit-files` is passed on the CLI).
