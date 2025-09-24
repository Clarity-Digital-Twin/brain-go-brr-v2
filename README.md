# 🧠 Brain-Go-Brr v2.6: TCN + Bi-Mamba + GNN + LPE for Clinical EEG Seizure Detection

**Pioneering O(N) complexity seizure detection with state-space models and graph neural networks**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![PyTorch 2.2.2](https://img.shields.io/badge/pytorch-2.2.2-red.svg)](https://pytorch.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

<details>
<summary><strong>Status: v2.6 (stable) + V3 (implemented)</strong></summary>

- **TCN + BiMamba + GNN + LPE** architecture fully implemented (31M parameters)
- **V3 dual‑stream path implemented**: learned adjacency via Edge Mamba + vectorized GNN
- **V2 path**: heuristic cosine similarity graphs (top_k=3) remains supported
- **PyG 2.6.1** with SSGConv (α=0.05) and Laplacian PE (k=16 eigenvectors)
- Local: RTX 4090 optimized; Modal: A100‑80GB with 24 CPU cores
- Last updated: 2025‑09‑24

</details>

## 🎯 Mission

We're deploying a **TCN + BiMamba + GNN + LPE** stack to reduce false alarms while maintaining sensitivity on long clinical EEG. Current systems trigger **>10 false alarms per day**, and while transformers work well, their O(N²) cost hinders real-time deployment. Our hybrid achieves **O(N) complexity** with superior spatiotemporal modeling.

**Why this architecture?**
- **TCN**: Multi-scale temporal features with dilated convolutions (8 layers)
- **BiMamba**: Long-range dependencies with O(N) efficiency (6 layers)
- **GNN**: Spatial electrode relationships via SSGConv (α=0.05)
- **LPE**: Laplacian positional encoding (k=16 eigenvectors)

Use either path:
- **V2 (architecture: tcn)** → heuristic cosine similarity graphs
- **V3 (architecture: v3)** → learned adjacency via Edge Mamba + vectorized GNN
See V3 details: docs/architecture/V3_ACTUAL.md

## 🏗️ Architecture

```
EEG Input (19ch, 256Hz, 60s windows)
         ↓
[TCN Encoder]       → 8 layers, channels [64,128,256,512], stride_down=16
         ↓
[Temporal SSM]      → BiMamba (O(N)) — global context
         ↓
[GNN + LPE]         → 2× SSGConv (α=0.05) + Laplacian PE (k=16)
         ↓           Adjacency: heuristic (v2) or learned via Edge Mamba (v3)
[Projection + Upsample] → Restore resolution (512→19, 960→15360)
         ↓
[Detection Head]    → Per‑timestep seizure probabilities
         ↓
[Post‑Processing]   → Hysteresis (τ_on=0.86, τ_off=0.78) + Morphology
         ↓
[TAES Evaluation]   → Clinical metrics (10 FA/24h target)
```

**Key Specifications:**
- **Input**: 19-channel 10-20 montage @ 256 Hz
- **Model**: 31M parameters (TCN + BiMamba + GNN + LPE)
- **GPU Requirements**: NVIDIA RTX 4090 (24GB) or A100-80GB
- **Training Time**: ~200-300 hours on RTX 4090, ~100 hours on A100
- **Complexity**: O(N) vs Transformer's O(N²)
- **Window**: 60s with 10s stride (83% overlap)
- **Class Imbalance**: 12:1 (focal loss + balanced sampling required)

→ Installation guide: `INSTALLATION.md`
→ Architecture evolution: `ARCHITECTURE_EVOLUTION.md`

## ⚡ Quick Start

### Installation

```bash
# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/clarity-digital-twin/brain-go-brr-v2.git
cd brain-go-brr-v2
make setup

# REQUIRED: Install v2.6 stack (Mamba + PyG + TCN)
make setup-gpu  # or: make g
```

### Training

```bash
# Local smoke test (1 epoch, 3 files)
make s  # or: make smoke-local

# Full training (local config currently uses V3)
tmux new -s train
make train-local  # configs/local/train.yaml → model.architecture: v3
# Watch: tmux attach -t train
```

### 🌩️ Cloud Training (Modal.com)

```bash
# Install Modal CLI
pip install --upgrade modal
modal setup

# Test Mamba CUDA
modal run deploy/modal/app.py --action test-mamba

# Smoke test
modal run deploy/modal/app.py --action train --config configs/modal/smoke.yaml

# Full training (detached)
modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml

# Monitor
modal app logs <app-id>
```

## 🗂️ Project Structure

```
src/brain_brr/           # Core modules
├── models/
│   ├── detector.py      # Main SeizureDetector class
│   ├── tcn.py          # TCN encoder (8 layers)
│   ├── mamba.py        # Bidirectional Mamba (6 layers)
│   ├── gnn_pyg.py      # PyG GNN with Laplacian PE
│   └── graph_builder.py # Heuristic adjacency builder
├── data/               # EEG preprocessing
├── train/              # Training pipeline
├── post/               # Post-processing
├── eval/               # TAES evaluation
└── config/             # Pydantic schemas

configs/                 # YAML configurations
├── local/              # RTX 4090 optimized
│   ├── smoke.yaml      # Quick test (1 epoch)
│   └── train.yaml      # Full training
└── modal/              # A100-80GB optimized
    ├── smoke.yaml      # Quick test
    └── train.yaml      # Full training

tests/                   # Comprehensive test suite
docs/                    # Documentation
```

## 📊 Clinical Targets

| FA Rate | Target Sensitivity | Current SOTA | Our Goal |
|---------|-------------------|--------------|----------|
| 10 FA/24h | >95% | ~90% | ✓ |
| 5 FA/24h | >90% | ~85% | ✓ |
| 1 FA/24h | >75% | ~70% | ✓ |

## 🔬 Technical Details

### Data Pipeline
1. **Input**: EDF files from TUH EEG Seizure Corpus
2. **Preprocessing**:
   - 10-20 montage standardization
   - Bandpass 0.5-120 Hz, 60 Hz notch
   - Resample to 256 Hz
   - Window: 60s with 10s stride
   - Per-channel z-score normalization
3. **Cache**: Pre-processed NPZ files (3734 train, 933 val)

### Training Strategy
- **Loss**: Focal loss (α=0.5, γ=2.0) for class imbalance
- **Sampling**: Balanced sampling ensures seizures in every batch
- **Optimizer**: AdamW with cosine schedule
- **Early Stopping**: Patience=5 on sensitivity@10FA/24h

### Critical Configuration

**Local (RTX 4090)**:
```yaml
training:
  batch_size: 12  # Conservative for 24GB VRAM
  mixed_precision: false  # Disabled - causes NaNs
data:
  cache_dir: cache/tusz  # 3734 pre-processed files
```

**Modal (A100-80GB)**:
```yaml
training:
  batch_size: 64
  mixed_precision: true  # A100 tensor cores
data:
  cache_dir: /results/cache/tusz  # Persistent SSD
```

## 🛠️ Development

```bash
# Quality checks (run after every change!)
make q  # lint + format + type check

# Testing
make t  # fast tests
make test  # full test suite with coverage
make test-gpu  # GPU-specific tests

# Utilities
make clean  # clean all artifacts
tmux ls  # list active training sessions
```

## 📖 Documentation

- **Installation**: See `INSTALLATION.md` for detailed setup
- **Architecture Evolution**: See `ARCHITECTURE_EVOLUTION.md` for design decisions
- **V3 Implementation**: See `docs/architecture/V3_ACTUAL.md` for the implemented dual‑stream path
- **Configuration**: See `configs/README.md` for config details
- **Claude AI Guide**: See `CLAUDE.md` for AI assistant instructions

## 🤝 Contributing

We welcome contributions! Please ensure:
1. Run `make q` before committing
2. Add tests for new features
3. Follow existing code patterns
4. Update documentation as needed

## 📝 License

Apache 2.0 - See [LICENSE](LICENSE) for details

## 🙏 Acknowledgments

- TUH EEG Seizure Corpus for training data
- CHB-MIT dataset for validation
- Modal.com for cloud GPU infrastructure
- PyTorch team for framework
- Mamba authors for SSM implementation

---

**Mission**: Shock the world with O(N) clinical seizure detection 🚀
