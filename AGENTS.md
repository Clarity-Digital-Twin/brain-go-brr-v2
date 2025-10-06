# AGENTS.md

This document is automatically consumed by AI coding agents. It summarises the **current** Brain-Go-Brr v3.8.2 baseline so every task starts with the right context.

---

## 🧠 Project Overview

Brain-Go-Brr v3.8.2 – "Zero Warnings" – is a clinical EEG seizure detector built on the **V3 dual-stream architecture**:

- **TCN** (8 layers, stride_down=16) for multi-scale temporal encoding.
- **Node BiMamba-2** (6 layers, d_model=64) for O(N) global context per electrode.
- **Edge BiMamba-2** (2 layers, d_model=16) for learned adjacency dynamics.
- **GNN (SSGConv)** with **dynamic Laplacian positional encoding** (α=0.05, k=16, sign-consistent eigenvectors).
- **Gated fusion** for node/edge streams, plus clamping safeguards (edge_similarity_margin = 0.01).

Key properties:
- Entire cache pipeline now uses **memory-mapped `_data.npy` / `_labels.npy` pairs** (no NPZ writes anywhere).
- Datasets operate in **read-only** mode with **NumPy copy-on-read tensors** so PyTorch never emits writable warnings; cache misses raise helpful errors directing the user to repopulate.
- Technical debt is at zero: lint, type checks, and full test/clinical suites are green (104 unit/integration + clinical, 83.8% cov) with **zero runtime warnings**.
- Mixed-precision runs now **guard the LR scheduler** so it only advances after a real optimizer step—no skipped-step warnings, accurate warmup/cosine decay.
- Modal automation: `check-cache` validates counts, and `clean_stray_npz.py` removes accidental NPZ files after aborted runs.

See `docs/04-model/v3-architecture.md` and `docs/04-model/v3-stability-evolution.md` for architecture and safeguard details.

---

## 🚀 Quick Commands

### Core Tooling
| Command | Purpose |
|---------|---------|
| `make q` | Run lint (ruff), format check, mypy – **run after every change** |
| `make t` | Fast unit + integration tests |
| `make test` | Full test suite with coverage |
| `make s` | Local smoke test (3 files, 1 epoch) |
| `make train-local` | Full local training loop (RTX 4090) |
| `make setup` | Base environment (uv) |
| `make setup-gpu` | GPU extras (Mamba-SSM, PyG) – required for V3 |

### Local Training (RTX 4090, 24 GB)
```bash
# Optional debugging when chasing NaNs
export BGB_NAN_DEBUG=1

# Smoke test (3 files, 1 epoch)
make s

# Full training (recommended in tmux)
tmux new -s train
make train-local
# Detach: Ctrl+B, D   |   Reattach: tmux attach -t train
```

### Modal Cloud (A100-80 GB)
```bash
# Populate mmap cache from S3 → Modal SSD (use --detach)
modal run --detach deploy/modal/app.py --action populate-cache

# Verify CUDA availability
modal run deploy/modal/app.py --action test-mamba

# Smoke test (50 files, 1 epoch)
modal run --detach deploy/modal/app.py --action train --config configs/modal/smoke.yaml

# Full training (detached long run)
modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml

# Monitoring helpers
modal app list
modal app logs <app-id>
modal app stop <app-id>
```

Use `modal run deploy/modal/clean_stray_npz.py --confirm` if `check-cache` warns about residual `*_windows.npz` files.

---

## 📁 Project Layout
```
src/brain_brr/          # Core implementation
├── models/             # Detector, TCN, Mamba (node/edge), GNN, fusion
├── data/               # IO, preprocessing, datasets (mmap read-only)
├── train/              # Training loop, optimisers, warmup, sampling
├── post/               # Post-processing (hysteresis + morphology)
└── config/             # Pydantic runtime config schemas

configs/                # YAML experiment configs
├── local/              # RTX 4090 (smoke/train)
└── modal/              # Modal A100 (smoke/train)

cache/tusz_mmap/        # Local mmap cache (`*_data.npy` + `*_labels.npy`)
└── {train,dev}/        # 4,667 train + 1,832 dev stems + manifests

/results/cache/tusz_mmap/  # Modal persistent SSD cache (same layout)
```

---

## ⚙️ Critical Configuration Snapshots

### Local (RTX 4090)
```yaml
data:
  cache_dir: cache/tusz_mmap
  num_workers: 0               # WSL2 stability
training:
  batch_size: 8                # Verified safe (~20 GB VRAM)
  mixed_precision: false       # FP16 unreliable on 4090
  loss: focal
  use_balanced_sampling: true
model:
  graph:
    edge_similarity_margin: 0.01
```

### Modal (A100-80 GB)
```yaml
data:
  cache_dir: /results/cache/tusz_mmap
  num_workers: 4
  persistent_workers: true     # Keeps mmap pages warm
  prefetch_factor: 2
training:
  batch_size: 48               # ~58 GB peak VRAM
  gradient_accumulation_steps: 1
  mixed_precision: true
  gradient_clip: 0.5
model:
  graph:
    edge_similarity_margin: 0.01
resources:
  cpu: 24
  memory: 98304
```

Both environments rely on the same mmap cache produced by `populate-cache`. Datasets never regenerate cache files.

---

## 🔧 Installation Requirements

Exact versions (pinned in `pyproject.toml`):
```
PyTorch==2.5.0+cu124
CUDA Toolkit==12.4
mamba-ssm==2.2.5
causal-conv1d==1.5.2
torch-geometric==2.6.1
numpy==1.26.4
```

Order of operations:
1. Install CUDA 12.4 toolchain (WSL2/Ubuntu: `sudo apt-get install -y cuda-toolkit-12-4`).
2. `make setup`
3. `make setup-gpu`
4. Verify: `python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('✅')"`

PyG wheels must match the Torch/CUDA version (`https://data.pyg.org/whl/torch-2.5.0+cu124.html`).

---

## 🛠️ Development Guidelines

- Python ≥ 3.11 with full type hints.
- Ruff (line length 100) + mypy enforced via `make q`.
- Import order: stdlib → third-party → first-party (isort-compatible).
- Prefer minimal inline comments; only add clarifying notes for non-obvious logic.
- Never rewrite cached data in code paths – rely on read-only mmap cache.
- Tests: `make t` for quick runs, `make test` for full coverage, `make test-gpu` when touching CUDA logic.

### Useful Environment Flags
```bash
# Debugging
export BGB_NAN_DEBUG=1          # Extra NaN logging (optional)
export BGB_SANITIZE_GRADS=1     # Debug utility (logs/zeros NaN grads) – optional

# Data limiting
export BGB_SMOKE_TEST=1         # Local smoke helper (3 files)
export BGB_LIMIT_FILES=50       # Custom file cap

# Cache maintenance
export BGB_FORCE_MANIFEST_REBUILD=1   # Force manifest regeneration

# WSL2
export UV_LINK_MODE=copy
```
Modal sets `BGB_NAN_DEBUG=1`, `BGB_LIMIT_FILES=50`, and logging cadence automatically.

---

## 🚨 Common Issues & Solutions

| Issue | Resolution |
|-------|------------|
| Wrong cache path | Local: `cache/tusz_mmap/`; Modal: `/results/cache/tusz_mmap/` |
| Stray NPZ files after aborted runs | `modal run deploy/modal/clean_stray_npz.py --confirm` |
| Cache miss errors | Run `populate-cache` (Modal) or rebuild locally; datasets no longer create NPZ fallbacks |
| A100 OOM with batch 64 | Use `batch_size: 48`, `gradient_accumulation_steps: 1` |
| NaNs on RTX 4090 | Keep `mixed_precision: false`; optional `BGB_SANITIZE_GRADS=1` when investigating |
| PyG install failures | Use prebuilt wheels matching Torch 2.5.0 + cu124 |
| Modal hangs | Ensure `/results` volume has ≥600 GB free; `check-cache` validates counts |

---

## 📊 Expected Performance

| Scenario | Time/Epoch | Notes |
|----------|------------|-------|
| Local train (batch 8) | ~3 h | ~300 h total for 100 epochs |
| Modal train (batch 48) | ~1 h | ~100 h total (~$319 @ $3.19/hr blended) |
| Smoke tests | ~5 min | Local (3 files) or Modal (50 files) |

Resource usage:
- VRAM: 20 GB (4090), 58 GB (A100).
- Cache size: ~50 GB (mmap NPY).
- Checkpoints: ~125 MB per epoch, mid-epoch snapshots every 30 min (keep last 3).

---

## 📌 Current Release – v3.8.2 Summary
- ✅ **Zero warnings**: NumPy copy-on-read pattern removes read-only tensor warnings; AMP scheduler guard only advances after real optimizer steps.
- ✅ **Complete tensor safety**: All 3 datasets keep read-only mmap semantics without ever mutating cache data.
- ✅ Read-only mmap cache pipeline (no NPZ drift possible).
- ✅ Shared mmap loader (`cache_utils.load_cache_mmap`) with uniform logging.
- ✅ Modal `check-cache`/`clean_stray_npz.py` health tooling.
- ✅ Type safety and lint clean (0 blockers).
- ✅ Documentation + configs fully aligned with reality.
- ✅ Tests: 104 automated + clinical; 83.8% coverage.
- 🚧 Next ideas (post-training): optional gradient sanitisation filter, print→logging sweep.

Mission remains unchanged: **<1 FA/24h clinical-grade seizure detection** with a fully reproducible, debt-free codebase.
