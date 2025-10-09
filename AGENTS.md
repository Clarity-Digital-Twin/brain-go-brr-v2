# AGENTS.md

This document is automatically consumed by AI coding agents. Start every task by skimming this sheet, then drill into the linked docs for specifics. It reflects the **current v3.9.1 baseline (validation OOM fix) plus the in-flight Flash Linear Attention (FLA) upgrade** state.

---

## 🧠 Project Overview

Brain-Go-Brr v3.9.1 – “Validation OOM Fix” – is a clinical EEG seizure detector built on the **V3 dual-stream architecture**:

- **TCN** (8 layers, stride_down=16) for multi-scale temporal encoding.
- **Node BiMamba-2** (6 layers, d_model=64) for O(N) global context per electrode. **Phase 1b** validated a drop-in BiGatedDeltaNet option (same d_model, headdim=8, 0.75× heads constraint).
- **Edge BiMamba-2** (2 layers, baseline d_model=16) for learned adjacency dynamics. **Phase 1a** introduced a BiGatedDeltaNet variant that requires d_model=32 for Triton compatibility; choose per-config.
- **GNN (SSGConv)** with **dynamic Laplacian positional encoding** (α=0.05, k=16, sign-consistent eigenvectors).
- **Gated fusion** for node/edge streams, plus clamping safeguards (edge_similarity_margin = 0.01).
- **Atomic checkpoints** capture model, optimizer, scheduler, AMP scaler, and RNG for deterministic resume.
- **Timeout guard** exits ~23 h (before Modal’s 24 h kill) and writes `timeout_exit.pt` with full state.
- **W&B persistence** stores `.wandb_run_id` in the checkpoint directory so resumes continue the same run.

Key properties:
- Cache pipeline uses **memory-mapped `_data.npy` / `_labels.npy` pairs** only (datasets are read-only and fail-fast on cache miss).
- NumPy copy-on-read tensors eliminate PyTorch read-only warnings.
- Technical debt is zero: lint, format, mypy, config validation, and 104 tests (83.8 % cov) are green with **zero runtime warnings**.
- Mixed-precision runs guard the LR scheduler so it only advances after a real optimizer step—no skipped-step warnings, accurate warmup/cosine decay.
- Modal automation: `check-cache` validates cache health / manifests; `clean_stray_npz.py` removes accidental NPZ files; training sets `BGB_WALL_CLOCK_LIMIT_S=82800` for the timeout guard; **disk-backed validation** prevents Modal OOMs (v3.9.1).

See `docs/05-training/modal.md`, `docs/05-training/checkpoint-strategy.md`, and `docs/04-model/v3-architecture.md` for deep dives on architecture and safeguards. FLA-specific plans live in:
- `FLASH_LINEAR_ATTENTION_RESEARCH.md` (SSOT)
- `FLASH_LINEAR_ATTENTION_DOC1_EDGE_MIGRATION.md` (Phase 1a – complete)
- `FLASH_LINEAR_ATTENTION_DOC2_NODE_MIGRATION.md` (Phase 1b – complete)
- `FLASH_LINEAR_ATTENTION_DOC3_FULL_MIGRATION.md` (Phase 2 – complete; Modal A/B pending)
- `FLASH_LINEAR_ATTENTION_DOC4_HYBRID_SWA.md` (optional post-Phase 2 experiments)
- `FLA_ROADMAP.md` / `FLA_QUICK_REFERENCE.md` (current strategy + status)

---

## 🚀 Quick Commands

### Core Tooling
| Command | Purpose |
|---------|---------|
| `make q` | Run lint (ruff), format check, mypy – **run after every change** |
| `make t` | Fast unit + integration tests |
| `make test` | Full test suite with coverage |
| `make smoke-bimamba` | Local BiMamba2 smoke test (3 files, 1 epoch) |
| `make smoke-fla` | Local FLA smoke test (3 files, 1 epoch) |
| `make train-bimamba` | Local BiMamba2 full training (RTX 4090) |
| `make train-fla` | Local FLA full training (RTX 4090) |
| `make train-local` | Alias for `make train-bimamba` |
| `make setup` | Base environment (uv) |
| `make setup-gpu` | GPU extras (Mamba-SSM, PyG) – required for V3 |

### Local Training (RTX 4090, 24 GB)
```bash
# Optional debugging when chasing NaNs
export BGB_NAN_DEBUG=1

# BiMamba2 smoke test (3 files, 1 epoch)
make smoke-bimamba

# FLA smoke test (3 files, 1 epoch)
make smoke-fla

# Full training (recommended in tmux)
tmux new -s train-bimamba
make train-bimamba
# Detach: Ctrl+B, D   |   Reattach: tmux attach -t train-bimamba

# FLA full training (optional)
tmux new -s train-fla
make train-fla
# Detach: Ctrl+B, D   |   Reattach: tmux attach -t train
```

### Modal Cloud (A100-80 GB)
```bash
# Populate mmap cache from S3 → Modal SSD (use --detach)
modal run --detach deploy/modal/app.py --action populate-cache

# Verify CUDA availability
modal run deploy/modal/app.py --action test-mamba

# Cache health / manifest sanity
modal run deploy/modal/app.py --action check-cache

# BiMamba2 smoke test (50 files, 1 epoch)
modal run --detach deploy/modal/app.py --action train \
  --config configs/modal/smoke_bimamba.yaml

# FLA smoke test
modal run --detach deploy/modal/app.py --action train \
  --config configs/modal/smoke_fla.yaml

# BiMamba2 full training (exits ~23h with timeout_exit.pt, resume required)
modal run --detach deploy/modal/app.py --action train \
  --config configs/modal/train_bimamba.yaml

# FLA full training (launch after BiMamba2 baseline completes)
modal run --detach deploy/modal/app.py --action train \
  --config configs/modal/train_fla.yaml

# Resume after timeout / interrupts (BiMamba2 example)
modal run --detach deploy/modal/app.py --action train \
  --config configs/modal/train_bimamba.yaml --resume true

# Monitoring helpers
modal app list
modal app logs <app-id>
modal app stop <app-id>
```

Use `modal run deploy/modal/clean_stray_npz.py --confirm` if `check-cache` warns about residual `*_windows.npz` files.

---

## 🤖 Agent Workflow Playbook

- **Always** run `git status` first; never revert unrelated changes.
- Skim the relevant FLA doc before touching configs or docs; they are kept in lock-step with implementation and list the exact evidence required.
- Default validation ladder (Oct 9 2025): smoke (3 files, ~5 min) → single medium run (40–50 files, 2–3 h) → Modal full train. Phase 2 smoke + medium complete; Modal A/B (BiMamba2 baseline → FLA) is next.
- Edge GDN configs must set `edge_mamba_d_model: 32` and satisfy the 0.75× head constraint; node GDN mirrors these rules with d_model=64.
- Capture logs to `/tmp/phase*_*.log` and link them in status docs; they anchor the CRAV methodology.
- Run `make q` before pushing; follow with `make t` (or targeted subsets) when touching core logic.

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
  mid_checkpoint_interval_s: 1800
  mid_epoch_keep: 3
model:
  graph:
    edge_similarity_margin: 0.01
resources:
  cpu: 24
  memory: 98304
```

Both environments rely on the same mmap cache produced by `populate-cache`. Datasets never regenerate cache files; cache misses raise a descriptive error instructing you to rebuild.

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

# Cache maintenance / manifest rebuild
export BGB_FORCE_MANIFEST_REBUILD=1   # Force manifest regeneration locally

# Timeout guard (Modal sets 82800 automatically)
export BGB_WALL_CLOCK_LIMIT_S=82800

# WSL2
export UV_LINK_MODE=copy
```
Modal sets `BGB_NAN_DEBUG=1`, `BGB_LIMIT_FILES=50`, timeout guard, and logging cadence automatically.

---

## 🚨 Common Issues & Solutions

| Issue | Resolution |
|-------|------------|
| Wrong cache path | Local: `cache/tusz_mmap/`; Modal: `/results/cache/tusz_mmap/` |
| Stray NPZ files after aborted runs | `modal run deploy/modal/clean_stray_npz.py --confirm` |
| Dev manifest stale / 0 validation windows | `modal run deploy/modal/app.py --action check-cache` and rebuild manifest if prompted |
| Cache miss errors | Run `populate-cache` (Modal) or rebuild locally; datasets fail fast and never create NPZ fallbacks |
| A100 OOM with batch 64 | Use `batch_size: 48`, `gradient_accumulation_steps: 1` |
| Modal 24 h timeout | Timeout guard writes `timeout_exit.pt`; rerun training with `--resume true` |
| W&B run duplicates after resume | `.wandb_run_id` saved alongside checkpoints; ensure `wandb` section enabled |
| NaNs on RTX 4090 | Keep `mixed_precision: false`; optional `BGB_SANITIZE_GRADS=1` when investigating |
| PyG install failures | Use prebuilt wheels matching Torch 2.5.0 + cu124 |
| Modal hangs | Ensure `/results` volume has ≥600 GB free; `check-cache` validates counts |

---

## 📊 Expected Performance

| Scenario | Time/Epoch | Notes |
|----------|------------|-------|
| Local train (batch 8) | ~3 h | ~300 h total for 100 epochs |
| Modal train (batch 48) | ~1 h | ~100 h total (~$319 at $3.19/hr blended) |
| Smoke tests | ~5 min | Local (3 files) or Modal (50 files) |

Resource usage:
- VRAM: 20 GB (4090), 58 GB (A100).
- Cache size: ~50 GB (mmap NPY).
- Checkpoints: ~195 MB each; atomic saves keep `best.pt`, `last.pt`, rotating `mid_epoch_*.pt`, plus `timeout_exit.pt` on guard-triggered exits.

---

## 📌 Current Release – v3.9.1 Summary
- ✅ **Validation OOM fix**: Disk-backed validation + manifest guard eliminate 120 GB spikes (Modal training stable).
- ✅ **Bulletproof checkpoints**: Atomic saves (temp + fsync + rename), AMP scaler capture, RNG persistence.
- ✅ **Timeout guard**: 23 h wall-clock limit with 10 min safety buffer; writes `timeout_exit.pt`.
- ✅ **Metric key normalization**: `metrics_utils.normalize_metrics_dict` stops “New best 0.0000” logs.
- ✅ **W&B persistence**: Run ID saved to `.wandb_run_id` for continuous dashboards across resumes.
- ✅ **Zero warnings / zero debt**: Lint, type, and tests (104) all green with 83.8 % coverage.
- 🔄 **FLA upgrade track**: Phase 0–2 complete (smoke + medium); Modal A/B (BiMamba2 baseline → FLA) queued after baseline run.
- 🚧 Next ideas (post-training): optional gradient sanitisation filter, print→logging sweep.

Mission remains unchanged: **<1 FA/24 h clinical-grade seizure detection** with a fully reproducible, debt-free codebase.
