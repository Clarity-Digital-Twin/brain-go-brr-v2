# Environment Variables

Core controls

- `BGB_SMOKE_TEST=1` — enable smoke shortcuts (skip sampling, relax checks)
- `BGB_LIMIT_FILES=N` — limit file count for quick runs
- `BGB_FORCE_MANIFEST_REBUILD=1` — delete and rebuild manifest on start

Debugging and stability

- `BGB_NAN_DEBUG=1` — extra logging if loss or grads misbehave
- `BGB_NAN_DEBUG_MAX=K` — limit debug prints per epoch (default 3)
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
