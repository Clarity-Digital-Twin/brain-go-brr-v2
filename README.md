# 🧠 Brain-Go-Brr V3: Clinical EEG Seizure Detection

**State-of-the-art seizure detection using TCN + BiMamba + GNN with Dynamic Laplacian PE**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![PyTorch 2.2.2](https://img.shields.io/badge/pytorch-2.2.2-red.svg)](https://pytorch.org)
[![CUDA 12.1](https://img.shields.io/badge/cuda-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](LICENSE)

## 🚀 Highlights

- **O(N) Complexity**: Linear-time processing via Mamba state-space models
- **Dual-Stream Architecture**: Parallel processing of node (19×) and edge (171×) features
- **Dynamic Graph Learning**: Time-evolving brain connectivity without heuristics
- **Architectural Stability (v3.2.0)**: Edge similarity clamping at source with configurable margin
- **NaN-Robust Training**: 3-tier clamping system with gradient sanitization
- **31M Parameters**: Efficient architecture that runs on consumer GPUs

## 📊 Clinical Performance Targets

```
┌─────────────────┬──────────────┬─────────────┐
│ False Alarms    │ Sensitivity  │ Status      │
├─────────────────┼──────────────┼─────────────┤
│ 10 FA/24h       │ >95%         │ 🔄 Training │
│ 5 FA/24h        │ >90%         │ 🔄 Training │
│ 1 FA/24h        │ >75%         │ 🎯 Target   │
└─────────────────┴──────────────┴─────────────┘
```

Tip: Ensure train/dev caches are built and manifests exist before training.

## 🏗️ Architecture

```
              EEG Input (19 channels @ 256Hz)
                            │
                            ▼
                     ┌─────────────┐
                     │ TCN ENCODER │ 8 layers, stride↓16
                     └─────────────┘
                            │
                     ╔══════╧═════╗
                     ║ PROJECTION ║ 512 → 19×64
                     ╚══════╤═════╝
                      ┌─────┴─────┐
                      ▼           ▼
              ┌────────────┐ ┌────────────┐
              │ NODE MAMBA │ │ EDGE MAMBA │  Parallel
              │   19×SSM   │ │  171×SSM   │  Streams
              └─────┬──────┘ └──────┬─────┘
                    │               │
                    └───────┬───────┘
                            ▼
                    ┌──────────────┐
                    │  GNN + LPE   │ Dynamic PE
                    └──────────────┘
                            │
                    ┌──────────────┐
                    │   DECODER    │ Upsample↑16
                    └──────────────┘
                            │
                            ▼
                     Seizure Predictions

Stability by construction (implemented)

- Boundary normalization at seams (PR‑1): configurable via `model.norms.*`.
- Bounded edge lift (PR‑2): `graph.edge_lift_activation` + `graph.edge_lift_norm`, init gain 0.1.
- Adjacency conditioning (PR‑3): `graph.adj_row_softmax`, `adj_ema_beta`, `adj_force_symmetric`, `laplacian_eps`.
- Clamp at source (PR‑5): edge similarity clamped to `[-1+margin, 1-margin]` via `edge_similarity_margin`.
- Dynamic PE safeguards: FP32 eigens, sign consistency, regularization, fallback; `semi_dynamic_interval` for memory.
```

### Key Components

| Component | Description | Parameters |
|-----------|-------------|------------|
| **TCN** | 8-layer temporal encoder with dilated convolutions | 12.8M |
| **BiMamba** | Bidirectional state-space model (6 layers) | 8.4M |
| **GNN** | 2-layer SSGConv with α=0.05 for EEG graphs | 6.2M |
| **Dynamic LPE** | k=16 eigenvectors, computed per timestep | 4.1M |
| **Total** | End-to-end trainable | **31.5M** |

## ⚡ Quick Start

### Prerequisites

```bash
# System requirements
- Ubuntu 20.04+ or WSL2
- CUDA 12.1+ with cuDNN 8.9+
- 24GB+ GPU memory (RTX 4090 or better)
- 32GB+ system RAM
```

### Installation

```bash
# 1. Install UV package manager (faster than pip)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and setup environment
git clone https://github.com/clarity-digital-twin/brain-go-brr-v2.git
cd brain-go-brr-v2
make setup

# 3. Install GPU components (CRITICAL: exact versions matter!)
make setup-gpu  # Installs mamba-ssm==2.2.2, PyG, TCN
```

### Data Preparation

```bash
# Download TUH EEG Seizure Corpus (requires agreement)
# Place in: data_ext4/tusz/edf/

# Build preprocessed caches (one-time)
# Train split
python -m src build-cache \
  --data-dir data_ext4/tusz/edf/train \
  --cache-dir cache/tusz/train \
  --split train

# Dev split (CRITICAL: We use 'dev' to match TUSZ official naming, not 'val'!)
python -m src build-cache \
  --data-dir data_ext4/tusz/edf/dev \
  --cache-dir cache/tusz/dev \
  --split dev

# Build manifests
python -m src scan-cache --cache-dir cache/tusz/train
python -m src scan-cache --cache-dir cache/tusz/dev
```

### Training

```bash
# Quick smoke test (5 minutes)
make s

# Full local training (RTX 4090)
export BGB_SANITIZE_GRADS=1  # Recommended for stability
tmux new -s train
make train-local
# Ctrl+B, D to detach
# tmux attach -t train to resume

# Cloud training (Modal A100)
modal run --detach deploy/modal/app.py \
  --action train \
  --config configs/modal/train.yaml
```

## 🔧 Configuration

### Critical Settings

```yaml
# RTX 4090 (24GB) - configs/local/train.yaml
training:
  batch_size: 4         # Stable baseline on 24GB VRAM
  mixed_precision: false  # MUST be false (causes NaNs)
  gradient_clip: 0.1     # Aggressive for stability

model:
  graph:
    use_dynamic_pe: true       # Enable dynamic PE
    semi_dynamic_interval: 5   # Update every 5 timesteps (memory tradeoff)
    edge_similarity_margin: 0.01  # v3.2.0: Safety margin for edge clamps

# A100 (80GB) - configs/modal/train.yaml
training:
  batch_size: 64
  mixed_precision: true   # A100 handles FP16 safely
  gradient_clip: 0.5
```

### Environment Variables

```bash
# Debugging
export BGB_NAN_DEBUG=1        # Verbose NaN reporting
export BGB_SANITIZE_GRADS=1   # Clean gradients (RECOMMENDED)
export BGB_DEBUG_FINITE=1     # Check all tensors

# Performance
export BGB_LIMIT_FILES=50     # Limit data for testing
export BGB_SMOKE_TEST=1       # Quick validation mode
```

## 📁 Project Structure

```
brain-go-brr-v2/
├── src/brain_brr/
│   ├── models/          # Core architecture
│   │   ├── detector.py  # Main model orchestrator
│   │   ├── tcn.py      # Temporal convolutions
│   │   ├── mamba.py    # Bidirectional SSM
│   │   └── gnn_pyg.py  # Graph neural network
│   ├── data/           # Preprocessing pipeline
│   ├── train/          # Training loop
│   └── post/           # Post-processing
├── configs/            # Training configurations
├── tests/              # Comprehensive test suite
├── docs/               # Documentation
│   ├── 00-overview/    # Architecture & targets
│   ├── 03-configuration/  # Config validation
│   ├── 04-model/       # Component details
│   ├── 05-training/    # Training guides
│   └── 08-operations/  # Troubleshooting
└── cache/tusz/         # Preprocessed data
```

## 🛠️ Development

### Essential Commands

```bash
make q          # Run quality checks (lint, format, type)
make t          # Fast test suite
make test       # Full tests with coverage
make clean      # Clean all artifacts
```

### Monitoring Training

```bash
# Local monitoring
tensorboard --logdir results/
watch -n 1 nvidia-smi  # GPU usage

# Modal monitoring
modal app list         # List running apps
modal app logs <id>   # Stream logs
```

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| NaN losses | Enable `BGB_SANITIZE_GRADS=1` and rebuild cache |
| OOM errors | Reduce batch_size or increase semi_dynamic_interval |
| Slow training | Verify cache on SSD, not network mount |
| Import errors | Exact versions: torch==2.2.2, mamba-ssm==2.2.2 |

## 📊 Expected Performance

Training time varies with batch size and cache locality. As a rough guide:

- A100-80GB, batch 64: ~1 hour/epoch (~100 hours total)
- RTX 4090, batch 4–8: several hours/epoch depending on IO and PE settings

### Memory Requirements

| Component | RTX 4090 | A100 | Note |
|-----------|----------|------|------|
| Model | 4GB | 4GB | Fixed |
| Batch | 8GB | 32GB | Scales with batch_size |
| Dynamic PE | 4GB | 8GB | Scales with interval |
| **Total** | **16GB** | **44GB** | Expected peak |

## 📚 Documentation

### Must Read
- [CLAUDE.md](CLAUDE.md) - Project context for AI assistants
- [ARCHITECTURE_EVOLUTION.md](ARCHITECTURE_EVOLUTION.md) - Design decisions
- [docs/08-operations/nan-prevention-complete.md](docs/08-operations/nan-prevention-complete.md) - NaN handling

### Deep Dives
- [docs/04-model/v3-architecture.md](docs/04-model/v3-architecture.md) - Full architecture
- [docs/04-model/laplacian-pe.md](docs/04-model/laplacian-pe.md) - Dynamic PE math
- [docs/05-training/modal-deployment.md](docs/05-training/modal-deployment.md) - Cloud setup

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Run `make q` before committing
4. Add tests for new features
5. Submit a pull request

## 📈 Roadmap

- [x] V3 dual-stream architecture
- [x] Dynamic Laplacian PE
- [x] NaN-robust training
- [ ] <1 FA/24h performance
- [ ] Real-time inference optimization
- [ ] Multi-dataset validation
- [ ] Clinical deployment

## 📝 Citation

```bibtex
@software{brain-go-brr-v3,
  title = {Brain-Go-Brr V3: Clinical EEG Seizure Detection},
  author = {Clarity Digital Twin},
  year = {2025},
  url = {https://github.com/clarity-digital-twin/brain-go-brr-v2}
}
```

## 📄 License

Apache 2.0 - See [LICENSE](LICENSE)

## 🙏 Acknowledgments

- **TUH EEG Seizure Corpus** - Temple University Hospital
- **CHB-MIT** - Children's Hospital Boston & MIT
- **Modal.com** - Cloud GPU infrastructure
- **Mamba** - Gu & Dao for SSM architecture
- **PyG Team** - PyTorch Geometric library

---

<div align="center">
<b>Questions?</b> Open an issue | <b>Updates:</b> Watch the repo | <b>Discussion:</b> Start a discussion
</div>
