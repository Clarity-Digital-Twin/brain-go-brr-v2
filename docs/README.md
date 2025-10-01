# Brain-Go-Brr Documentation (v3.4.1)

**Last Updated**: October 1, 2025
**Codebase Version**: v3.4.1 (PyTorch 2.5.0 + mamba-ssm 2.2.5)
**Architecture**: V3 Dual-Stream (TCN + BiMamba + GNN + Dynamic LPE)

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
- [WSL2 Specific](08-operations/wsl2-notes.md) - Windows users

**Optimization**:
- [Performance Optimization](08-operations/performance-optimization.md) - Speed & memory
- [Warmup Schedules](05-training/warmup-schedules.md) - Optional gradient stabilization

### 📚 Reference (Information-Oriented)

**Architecture**:
- [V3 Architecture](04-model/v3-architecture.md) - **Complete specification** ⭐
- [V3 Stability Evolution](04-model/v3-stability-evolution.md) - v3.3.0 → v3.4.1 journey
- [TCN](04-model/tcn.md) - Temporal convolutional network
- [Mamba](04-model/mamba.md) - Bidirectional SSM
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
- [Cache Layout](02-data/cache-layout.md) - NPZ structure
- [Cache Manifest Architecture](02-data/cache-manifest-architecture.md) - Balanced sampling
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

### Architecture (v3.4.1)
- **TCN**: 8 layers, channels [64,128,256,512], stride=16
- **Node Mamba**: 6 layers, d_model=64, bidirectional SSM
- **Edge Mamba**: 2 layers, d_model=16, learned adjacency
- **GNN**: 2× SSGConv, α=0.05, Laplacian PE (k=16)
- **Parameters**: ~31M total
- **Dynamic PE**: Always enabled with safeguards (v3.3.1+)

### Training Stability (v3.4.1)
- ✅ **Zero NaN/Inf** across 723+ batches (validated Oct 1, 2025)
- ✅ **Eigendecomposition fix** (v3.3.1): Detached eigenvectors prevents gradient explosion
- ✅ **3-tier NaN protection**: Data preprocessing + gradient sanitization + architectural safeguards
- ✅ **Optional warmup schedules** (v3.4.1): Adjacency temperature + focal gamma

### Hardware Requirements
- **Local**: RTX 4090 (24GB VRAM), batch_size=12, mixed_precision=false
- **Modal**: A100-80GB, batch_size=64, mixed_precision=true
- **Cache**: ~50GB disk space (4667 train + 1832 dev NPZ files)

### Software Stack (Exact Versions)
```
PyTorch==2.5.0+cu124
CUDA Toolkit==12.4
mamba-ssm==2.2.5
causal-conv1d==1.5.2
torch-geometric==2.6.1
numpy==1.26.4
```

### Critical Environment Variables
```bash
export BGB_SANITIZE_GRADS=1  # REQUIRED for PyTorch 2.5.0
export BGB_NAN_DEBUG=1       # Recommended for monitoring
```

---

## 📊 Performance Targets (Clinical)

| FA Rate | Target Sensitivity |
|---------|-------------------|
| 10 FA/24h | >95% |
| 5 FA/24h | >90% ⭐ |
| 1 FA/24h | >75% |

**Current status (v3.4.1)**: Architecture validated, full training in progress.

---

## 🗂️ Quick Reference

### Essential Commands

```bash
# Smoke test (5 minutes)
make s

# Full training (local)
export BGB_SANITIZE_GRADS=1 BGB_NAN_DEBUG=1
make train-local

# Full training (Modal)
modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml

# Validate config
python -m src validate configs/local/train.yaml

# Build cache
python -m src build-cache --data-dir data_ext4/tusz/edf/train --cache-dir cache/tusz/train

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
├── cache/tusz/              # Preprocessed NPZ files
│   ├── train/               # 4667 files + manifest.json
│   └── dev/                 # 1832 files + manifest.json
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
- **Historical incidents**: See `reference/incidents-historical/`
- **Development plans**: See `reference/development/`

---

## 🎯 Project Status

**Current Version**: v3.4.1 (October 1, 2025)

**Recent Milestones**:
- ✅ v3.3.1 (Sept 30): Eigendecomposition fix - zero gradient explosions
- ✅ v3.4.0 (Sept 30): Pre-norm Mamba alignment
- ✅ v3.4.1 (Oct 1): Optional warmup schedules
- ✅ Training validation: 723+ batches stable, zero NaN/Inf

**Next Steps**:
- [ ] Complete 100-epoch training run
- [ ] Full TAES evaluation on dev set
- [ ] Hyperparameter optimization

---

**Ready to start?** → [Quickstart Guide](getting-started/quickstart.md) 🚀
