# NaN Prevention Reference

**Last Updated**: October 6, 2025
**Scope**: Production safeguards for V3 dual-stream model (TCN + BiMamba + GNN)

---

## Quick Start

Before any long run (local or Modal):

```bash
export BGB_NAN_DEBUG=1          # Verbose NaN logging (recommended)
# export BGB_SANITIZE_GRADS=1   # Optional debugging helper
```

Gradient clipping (`training.gradient_clip: 0.5`) is always enabled from the config and provides the primary protection; the environment flags add observability while you investigate issues.

---

## Three-Layer Defence System

### 1. Data Preprocessing (Always On)

- `preprocess.py` performs per-channel z-score normalisation, clips outliers to ±10σ, and replaces NaN/Inf with zeros.
- Use `python -m src build-cache` to regenerate caches after any preprocessing change; stale data can reintroduce NaNs.

### 2. Model Boundaries (Always On)

- LayerNorm + LayerScale guarding each major component (TCN → Node Mamba → Edge Mamba → GNN → Decoder).
- Edge features clamp cosine similarities within ±(1 − margin) and ensure non-zero norms.
- Laplacian eigenvectors are detached to avoid gradient explosions during eigendecomposition (PR‑3 fix).

### 3. Gradient Handling (Config Driven)

- `training.gradient_clip: 0.5` is mandatory for both local and Modal configs; PyTorch clips the pre-scaled gradients before each optimiser step.
- Optional: set `BGB_SANITIZE_GRADS=1` to zero-out/log non-finite gradients for forensic runs (disabled by default to avoid masking issues).

---

## Architectural Hardening (PR-1 → PR-5)

| PR | Summary | Code Reference |
|----|---------|----------------|
| PR-1 | Boundary LayerNorm + LayerScale at every stream junction | `models/detector.py` norms block |
| PR-2 | Bounded edge stream with activation + norm + initial gain | `models/edge_features.py`, constants wired |
| PR-3 | Stable Laplacian eigendecomposition (detach eigenvectors, clamp eigenvalues) | `models/gnn_pyg.py` |
| PR-4 | Fusion constants & LayerScale fallback centralised | `models/builders` modules |
| PR-5 | Edge similarity clamp margin (0.01) to prevent ±1 blow-ups | Config & constants |

All five protections are active in every shipping config (`configs/local/*`, `configs/modal/*`).

---

## Gradient Logging

- Default log line: `[GRADIENTS] Last 100 batches: P50=2.19 | IQR=2.39 | P95=11.38 | Max=14.82`.
- Metrics are computed on finite pre-clip norms; overflow batches are reported separately (`[GRADIENTS] 15/100 batches had inf pre-clip norm ...`).
- For long-term monitoring, log the same metrics to W&B:

```python
wandb.log({
    "gradients/pre_clip_p50": p50,
    "gradients/pre_clip_iqr": iqr,
    "gradients/pre_clip_p95": p95,
    "gradients/overflow_pct": overflow_pct,
}, step=batch_idx)
```

See `docs/08-operations/gradient-monitoring.md` for interpretation tips.

---

## Environment Flags (Current State)

| Flag | Default | Purpose |
|------|---------|---------|
| `BGB_NAN_DEBUG` | 0 | Log non-finite tensors with extra context. Modal sets this to 1 automatically. |
| `BGB_SANITIZE_GRADS` | 0 | Optional debugging: replace non-finite gradients with zeros after `backward()`. Use sparingly. |
| `BGB_SANITIZE_INPUTS` | Removed | Never implemented; rely on preprocessing instead. |
| `BGB_SKIP_OPT_STEP_ON_NAN` | Removed | Skipping optimiser steps breaks LR schedules; investigate root causes instead. |
| `BGB_SAFE_CLAMP` | Removed | LayerNorm-based safeguards made it redundant. |

Remember to restart the process after toggling any `BGB_*` flag; `EnvConfig` memoises values on import.

---

## Troubleshooting Checklist

1. **Run `make q` and `make test`** to ensure baseline passes locally.
2. **Inspect logs** for `[GRADIENTS]` lines; if P95 skyrockets or overflow % grows, lower the learning rate or batch size.
3. **Rebuild caches** if you changed preprocessing or observed corrupted windows:
   ```bash
   rm -rf cache/tusz_mmap
   python scripts/convert_cache_to_mmap.py ...
   python -m src scan-cache --cache-dir cache/tusz_mmap/train
   ```
4. **Use `BGB_SANITIZE_GRADS=1` temporarily** to narrow down problematic batches; disable it once you have a repro so the optimiser sees real gradients.
5. **Modal-specific**: ensure `/results/cache/tusz_mmap` exists with manifest + *_data.npy files; run `modal run deploy/modal/app.py --action check-cache` if in doubt.

---

## Reference Documents

- `docs/08-operations/gradient-protection-guide.md` — Detailed rationale for each safeguard.
- `docs/archive/nan-prevention-complete.md` — Historical record of the investigation (kept for provenance).
- `docs/archive/PROTECTION_IMPLEMENTATION_SSOT.md` — Future work proposal for opt-in sanitisation tooling.

Maintain this checklist to avoid costly reruns and keep the 100-epoch campaigns stable.
