# 🧠 Brain-Go-Brr V3: TCN + Bi-Mamba + GNN + Dynamic LPE for Clinical EEG Seizure Detection

**Pioneering O(N) complexity seizure detection with dual-stream architecture and dynamic Laplacian positional encoding**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![PyTorch 2.2.2](https://img.shields.io/badge/pytorch-2.2.2-red.svg)](https://pytorch.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

<details>
<summary><strong>Status: V3 Production Ready (2025-09-24)</strong></summary>

- **V3 Dual-Stream Architecture**: Node Mamba (19×) + Edge Mamba (171×) processing in parallel
- **Dynamic Laplacian PE**: Time-evolving positional encoding computed every timestep (semi-dynamic interval configurable)
- **31M parameters**: TCN (8L) + BiMamba (6L) + GNN (2L SSGConv) + Dynamic LPE (k=16)
- **Memory Optimized**: RTX 4090 (batch=4, interval=5), A100 (batch=64, full dynamic)
- **NaN Protection**: Decoder clamping, focal loss fixes, training safeguards
- **Production Deployment**: Running on Modal A100-80GB with W&B tracking

</details>

## 🎯 Mission

Deploying **V3 dual-stream architecture** for clinical seizure detection with **<1 FA/24h** target. Our innovation: **dynamic Laplacian positional encoding** that evolves with the brain network over time, validated by EvoBrain literature.

**V3 Architecture Innovations:**
- **Dual-Stream Processing**: Node features (19 electrodes) and edge features (171 connections) processed separately
- **Dynamic LPE**: Eigendecomposition computed per timestep, capturing evolving brain connectivity
- **Edge Mamba**: Learns adjacency matrices directly from data (no heuristics)
- **Vectorized GNN**: Processes all timesteps simultaneously for 10× speedup
- **Semi-Dynamic Interval**: Configurable PE update frequency for memory/accuracy tradeoff

**Key Improvements from V2:**
- Replaced heuristic graphs with learned adjacency (Edge Mamba)
- Added dynamic PE (was static in V2)
- Vectorized GNN operations (10× faster)
- Fixed numerical stability issues (NaN protection throughout)

## 🏗️ Architecture

```
EEG Input (B, 19, 15360) @ 256Hz
         ↓
[TCN Encoder]           8 layers, [64,128,256,512], stride_down=16
         ↓              Output: (B, 512, 960)
[Projection]            512 → 19×64 electrode features
         ↓
    ┌────┴────┐
[Node Mamba]  [Edge Mamba]     PARALLEL DUAL-STREAM
19× BiMamba2  171× BiMamba2    Node: (B×19, 64, 960)
    │         │                 Edge: (B×171, 16, 960)
    │    [Adjacency]           Learned per timestep
    └────┬────┘
         ↓
[Vectorized GNN]        2-layer SSGConv (α=0.05)
+ Dynamic LPE           k=16 eigenvectors, computed every N steps
         ↓              Process all 960 timesteps at once
[Back-Projection]       19×64 → 512 bottleneck
         ↓
[Decoder + Upsample]    4 stages, restore to (B, 19, 15360)
         ↓
[Detection Head]        Per-sample logits with clamping
         ↓
[Post-Processing]       Hysteresis + Morphology
```

**Key Specifications:**
- **Model Size**: 31,475,722 parameters
- **Memory Usage**:
  - RTX 4090: 16GB with batch_size=4, semi_dynamic_interval=5
  - A100: 60GB with batch_size=64, full dynamic PE (interval=1)
- **Dynamic PE Cost**: 960 eigendecompositions per batch (7.5GB for full dynamic)
- **Training Speed**:
  - RTX 4090: ~2-3 hours/epoch (100-300 hours total)
  - A100: ~1 hour/epoch (100 hours total, ~$319)
- **Numerical Stability**: Mixed precision OFF on RTX 4090, ON for A100
- **Class Imbalance**: 34.2% seizure windows (balanced sampling critical)

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

# Install GPU stack (Mamba + PyG + TCN)
make setup-gpu  # uses prebuilt PyG wheels for torch 2.2.2+cu121
```

### Training

```bash
# Local smoke test (1 epoch, 3 files)
make s  # or: make smoke-local

# Full V3 training with optimized config
tmux new -s v3_full
make train-local  # RTX 4090: batch_size=4, semi_dynamic_interval=5
# Detach: Ctrl+B then D
# Watch: tmux attach -t v3_full
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
