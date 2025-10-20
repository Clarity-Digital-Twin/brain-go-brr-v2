# Brain-Go-Brr Documentation (v4.0.0)

**Last Updated**: October 17, 2025
**Codebase Version**: v4.0.0 (Dual Stack: BiMamba2 baseline + Flash Linear Attention variant)
**Architecture**: V3 Dual-Stream (TCN + SSM + GNN + Dynamic LPE) with interchangeable BiMamba2 / BiGatedDeltaNet streams
**Status**: 🟢 Zero active debt; Modal A100 + local RTX 4090 training live with deterministic resume and ext4-backed cache
**Historical Archive**: Legacy incident reports and PR plans live under `archived_docs/docs_v4_archive/reference/`

---

## 🚀 Quick Start (New Users)

### 5-Minute Smoke Test
**Start here** if this is your first time:
- [**Quickstart Guide**](getting-started/quickstart.md) - Run smoke test in 5 minutes

### Full Training Walkthrough
Ready for production training:
- [**Your First Training Run**](getting-started/first-run.md) - Complete 100-epoch guide

---

## 📖 Documentation Structure

### 🎓 Getting Started (Tutorials)
Learn by doing:
- [Quickstart](getting-started/quickstart.md) - 5-minute smoke test
- [First Training Run](getting-started/first-run.md) - Full local training walkthrough

### 🛠️ How-To Guides (Task-Oriented)

**Installation & Setup**:
- [Environment Setup](01-installation/env-setup.md) - Python, uv, dependencies
- [GPU Stack](01-installation/gpu-stack.md) - PyTorch, Mamba, PyG installation
- [Preflight Checks](01-installation/preflight-checks.md) - Verify setup

**Training**:
- [Local Training (RTX 4090)](05-training/local.md) - WSL2, batch size, memory
- [Modal Training (A100)](05-training/modal.md) - Cloud deployment, monitoring
- [Smoke Tests](05-training/smoke-tests.md) - Quick validation runs
- [Resume from Checkpoint](05-training/resume.md) - Handle interruptions

**Troubleshooting**:
- [NaN Prevention & Handling](08-operations/nan-prevention-complete.md) - **Canonical guide** ⭐
- [Gradient Monitoring](08-operations/gradient-monitoring.md) - Understanding "Large grad norm"
- [General Troubleshooting](08-operations/troubleshooting.md) - Common issues
- [WSL2 Specific](08-operations/wsl2-notes.md) - Windows users (see also [WSL2 SIGBUS Fix](08-operations/wsl2-sigbus-fix.md))

**Optimization**:
- [Performance Optimization](08-operations/performance-optimization.md) - Speed & memory
- [Warmup Schedules](05-training/warmup-schedules.md) - Optional gradient stabilization

### 📚 Reference (Information-Oriented)

**Architecture**:
- [V3 Architecture](04-model/v3-architecture.md) - **Complete specification** ⭐
- [V3 Stability Evolution](04-model/v3-stability-evolution.md) - v3.3.0 → v4.0.0 journey
- [TCN](04-model/tcn.md) - Temporal convolutional network
- [State-Space Streams](04-model/mamba.md) - BiMamba2 + Flash Linear Attention
- [FLA Research Documentation](04-model/flash-linear-attention/) - Detailed validation logs and methodology
- [GNN](04-model/gnn.md) - Graph neural network with PyG
- [Laplacian PE](04-model/laplacian-pe.md) - Dynamic positional encoding
- [Edge Features](04-model/edge-features-and-adjacency.md) - Learned adjacency
- [Post-processing](04-model/postprocess.md) - Hysteresis + morphology

**Configuration**:
- [Config Schema](03-configuration/config-schema.md) - Complete YAML reference
- [Environment Variables](03-configuration/env-vars.md) - BGB_* flags
- [Local Configs](03-configuration/local-configs.md) - RTX 4090 profiles
- [Modal Configs](03-configuration/modal-configs.md) - A100 profiles

**Data**:
- [Data Overview](02-data/overview.md) - TUH corpus, preprocessing
- [Cache Layout](02-data/cache-layout.md) - **NPY mmap structure** (v3.8.0)
- [Preprocessing](02-data/preprocessing.md) - Filters, z-score, clipping

**CLI Tools**:
- [CLI Usage](07-cli-tools/cli-usage.md) - `python -m src` commands
- [Makefile Commands](07-cli-tools/makefile-commands.md) - `make` shortcuts

**Evaluation**:
- [Metrics & TAES](06-evaluation/metrics-and-taes.md) - Sensitivity, FA rate

### 💡 Explanations (Understanding-Oriented)

**Architecture Design**:
- [Architecture Summary](00-overview/architecture-summary.md) - High-level overview
- [Performance Targets](00-overview/performance-targets.md) - Clinical goals

**Training Concepts**:
- [Gradient Monitoring](08-operations/gradient-monitoring.md) - Why P95=20 is normal
- [Warmup Schedules](05-training/warmup-schedules.md) - When/why to use
- [Checkpoint Strategy](05-training/checkpoint-strategy.md) - Save frequency, resume

**Operations**:
- [Modal Volume Architecture](08-operations/modal-volume-architecture.md) - Persistent SSD
- [Streaming](08-operations/streaming.md) - Real-time inference

### 🧑‍💻 Development (For Contributors)

- [Coding Standards](09-development/coding-standards.md) - Style, types, ruff
- [Testing](09-development/testing.md) - Test strategy, GPU tests
- [Versioning & Roadmap](09-development/versioning-and-roadmap.md) - Release history
- [Bug Tracker](09-development/bug-tracker.md) - Known issues
- [Technical Debt](09-development/technical-debt.md) - Future improvements

---

## 🔥 Common Tasks

### I want to...

**...start training immediately**
→ [Quickstart Guide](getting-started/quickstart.md)

**...fix NaN losses**
→ [NaN Prevention Complete](08-operations/nan-prevention-complete.md)

**...understand "Large grad norm" messages**
→ [Gradient Monitoring](08-operations/gradient-monitoring.md)

**...train on Modal (A100)**
→ [Modal Training Guide](05-training/modal.md)

**...optimize training speed**
→ [Performance Optimization](08-operations/performance-optimization.md)

**...resume from checkpoint**
→ [Resume Guide](05-training/resume.md)

**...understand the architecture**
→ [V3 Architecture](04-model/v3-architecture.md)

**...troubleshoot WSL2 issues**
→ [WSL2 Notes](08-operations/wsl2-notes.md)

---

## ⚙️ Key Technical Details

### Architecture (v4.0.0)
- **TCN**: 8 layers, channels [64,128,256,512], stride=16
- **Node SSM**: BiMamba2 (6 layers, d_model=64) OR BiGatedDeltaNet (FLA variant)
- **Edge SSM**: BiMamba2 (2 layers, d_model=16) OR BiGatedDeltaNet with d_model=32
- **GNN**: 2× SSGConv, α=0.05, Laplacian PE (k=16)
- **Parameters**: ~31M total (BiMamba2) or ~29M (FLA variant)
- **Dynamic PE**: Always enabled with safeguards (v3.3.1+)
- **Dual Stacks**: BiMamba2 (Modal A100) + FLA (Local RTX 4090) both training

### Training Stability (v4.0.0)
- ✅ **Atomic checkpoints**: Temp-file + fsync + rename with AMP scaler & RNG capture (deterministic resume)
- ✅ **Timeout guard**: Wall-clock monitor exits ~23 h with `timeout_exit.pt` before Modal’s 24 h limit
- ✅ **Metric normalization**: `metrics_utils.normalize_metrics_dict` eliminates “New best 0.0000” logs
- ✅ **W&B persistence**: `.wandb_run_id` stored in checkpoint dir for seamless resumes
- ✅ **Complete tensor safety**: Datasets use copy-on-read tensors for mmap safety (no PyTorch warnings)
- ✅ **Zero NPZ contamination**: Datasets read-only, fail-fast on cache miss
- ✅ **Memory-mapped cache**: <1 GB RAM vs 387 GB NPZ, 99.6 % faster startup
- ✅ **Zero XID 31 crashes**: Unique Triton / Inductor cache per run
- ✅ **Eigendecomposition fix**: Detached eigenvectors prevents gradient explosion
- ✅ **3-tier NaN protection**: Gradient clipping (0.5) + monitoring + architectural safeguards

### Hardware Requirements
- **Local**: RTX 4090 (24GB VRAM), batch_size=8, mixed_precision=false
- **Modal**: A100-80GB, batch_size=48, mixed_precision=true
- **Cache**: ~50GB disk space (4667 train + 1832 dev NPY mmap files)

### Software Stack (Exact Versions)
```
PyTorch==2.5.0+cu124
CUDA Toolkit==12.4
mamba-ssm==2.2.5
causal-conv1d==1.5.2
torch-geometric==2.6.1
numpy==1.26.4
```

### Environment Variables (Optional)
```bash
export BGB_NAN_DEBUG=1         # Enable NaN warnings
# export BGB_SANITIZE_GRADS=1  # Debug: log gradient NaNs (not required)
```

**Note**: Gradient clipping (0.5) from config provides primary NaN protection.

---

## 📊 Performance Targets (Clinical)

| FA Rate | Target Sensitivity |
|---------|-------------------|
| 10 FA/24h | >95% |
| 5 FA/24h | >90% ⭐ |
| 1 FA/24h | >75% |

**Current status (v4.0.0)**: Dual production stacks training simultaneously with auto-restart on Modal.

---

## 🗂️ Quick Reference

### Essential Commands

```bash
# Smoke test (5 minutes)
make s

# Full training (local) - BiMamba2 baseline
export BGB_NAN_DEBUG=1
make train-bimamba

# Or FLA research variant
make train-fla

# Full training (Modal) - BiMamba2 baseline (hands-free auto-restart)
modal deploy deploy/modal/app.py
modal run --detach deploy/modal/app.py --action schedule-training --config configs/modal/train_bimamba.yaml

# Or FLA research variant (hands-free auto-restart)
modal run --detach deploy/modal/app.py --action schedule-training --config configs/modal/train_fla.yaml

# Validate config
python -m src validate configs/local/train_bimamba.yaml

# Build cache (NPY mmap format)
python -m src build-cache --data-dir data_ext4/tusz/edf/train --cache-dir cache/tusz_mmap/train

# Quality checks (lint + format + mypy)
make q
```

### File Locations

```
brain-go-brr-v2/
├── src/brain_brr/           # Core implementation
│   ├── models/              # TCN, Mamba, GNN, detector
│   ├── data/                # Preprocessing, datasets
│   ├── train/               # Training loop
│   └── config/              # Pydantic schemas
├── configs/                 # Training configurations
│   ├── local/               # RTX 4090 (smoke.yaml, train.yaml)
│   └── modal/               # A100 (smoke.yaml, train.yaml)
├── cache/tusz_mmap/         # Memory-mapped NPY cache (v3.8.0)
│   ├── train/               # 4667 _data.npy + _labels.npy + manifest.json
│   └── dev/                 # 1832 _data.npy + _labels.npy + manifest.json
├── deploy/modal/            # Modal deployment scripts
└── docs_v3/                 # You are here!
```

---

## 📖 Documentation Philosophy

This documentation follows the **Diátaxis framework**:

| Type | Purpose | Examples |
|------|---------|----------|
| **Tutorials** | Learning by doing | Quickstart, First Run |
| **How-To Guides** | Solve specific problems | NaN troubleshooting, Modal setup |
| **Reference** | Technical specifications | Config schema, architecture |
| **Explanations** | Understanding concepts | Why BiMamba+GNN, gradient norms |

**Navigation tip**: Start with **Getting Started**, use **How-To Guides** for tasks, consult **Reference** for details, read **Explanations** to understand "why".

---

## 🆘 Getting Help

### Common Issues
- **NaN losses**: [NaN Prevention Complete](08-operations/nan-prevention-complete.md)
- **GPU OOM**: Reduce `batch_size` in config
- **WSL2 hangs**: Set `num_workers: 0` in config
- **Gradient explosions**: [Gradient Monitoring](08-operations/gradient-monitoring.md)

### Additional Resources
- **Bug reports**: See [Bug Tracker](09-development/bug-tracker.md)
- **Historical incidents**: See `archived_docs/docs_v4_archive/reference/incidents-historical/`
- **Development plans**: See `archived_docs/docs_v4_archive/reference/development/`

**Current Version**: v4.0.0 (October 17, 2025) – FLA Production + WSL2 Fix

**Recent Milestones**:
- ✅ v4.0.0 (Oct 17): WSL2 SIGBUS fix (cache on native ext4), FLA stack validated, dual production training live
- ✅ v3.9.1 (Oct 9): Disk-backed validation + manifest guard eliminate Modal OOM (100-epoch runs stable)
- ✅ v3.9.0 (Oct 8): Atomic checkpoints, timeout guard, W&B persistence, Modal training live
- ✅ v3.8.3 (Oct 7): Manifest naming cleanup, zero technical debt
- ✅ v3.8.2 (Oct 6): Zero warnings – copy-on-read tensors + AMP scheduler guard
- ✅ v3.8.1 (Oct 6): Complete tensor safety – all datasets fixed (EEGWindowDataset `.clone()`)

**Current Status**:
- 🔶 **BiMamba2 (Modal A100)**: PAUSED at Epoch 6 ($1,118 spent, $18,600 projected) due to high costs
- 🟢 **FLA (Local RTX 4090)**: ACTIVE at Epoch 7/100 (7% complete, ~960h total / 40 days)
- ✅ **Smoke tests**: PASS – Local (3 files) & Modal (50 files) validated after each change
- ✅ **Technical debt**: ZERO active issues (P0/P1/P2/P3 all resolved)

**Next Steps**:
- [ ] Continue FLA local training to completion (Epoch 7 → 100)
- [ ] Full TAES evaluation on dev set once training completes
- [ ] Tune post-processing for <1 FA/24h @ >75% sensitivity

---

**Ready to start?** → [Quickstart Guide](getting-started/quickstart.md) 🚀
