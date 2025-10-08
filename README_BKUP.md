# 🧠 Brain-Go-Brr V3: Clinical EEG Seizure Detection

**O(N) complexity seizure detection via time-then-graph paradigm**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![PyTorch 2.5.0](https://img.shields.io/badge/pytorch-2.5.0-red.svg)](https://pytorch.org)
[![CUDA 12.4](https://img.shields.io/badge/cuda-12.4-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](LICENSE)
[![v3.9.0](https://img.shields.io/badge/version-3.9.0-blue.svg)](https://github.com/clarity-digital-twin/brain-go-brr-v2/releases/tag/v3.9.0-production-training-baseline)

## Overview

Epileptic seizures affect ~50 million people worldwide. Continuous clinical monitoring faces a critical bottleneck: **false alarm rates**. Target performance: <1 false alarm per 24 hours at >75% sensitivity.

### The Challenge

EEG seizure detection demands modeling two intertwined phenomena:
- **Temporal dynamics**: Multi-scale patterns from milliseconds (spike transients) to minutes (rhythmic evolution)
- **Spatial propagation**: Time-varying connectivity across 19 scalp electrodes as seizures spread through neural networks

Seizures are spatiotemporal network disorders requiring joint modeling of both dimensions.

### Our Approach

V3 implements a **dual-stream architecture** grounded in state-space models and dynamic graph theory:

1. **Time-first processing**: TCN + BiMamba extract temporal features with O(N) complexity
2. **Graph-aware fusion**: Dynamic Laplacian PE captures evolving electrode connectivity
3. **Learned adjacency**: Edge Mamba models pairwise relationships, not hand-crafted graphs

**Theoretical foundation**: [EvoBrain (NeurIPS 2025)](literature/markdown/EVOBRAIN.md) proves time-then-graph ordering achieves +23% AUROC over alternatives.

**Status**: v3.9.0 – Production Training Baseline. Atomic checkpoints + deterministic resume, timeout guard exits before Modal’s 24 h limit, W&B run persistence, and zero technical debt. See [release notes](RELEASE_NOTES.md) for details.

## 🏗️ Architecture: Theory & Design

### Why Time-Then-Graph?

[EvoBrain (NeurIPS 2025)](literature/markdown/EVOBRAIN.md) establishes two critical theorems:

**Theorem 1 (Dynamic Graphs)**: *Explicit dynamic modeling (time-varying adjacency) is strictly more expressive than implicit (static graphs).*

**Theorem 2 (Temporal Ordering)**: *time-then-graph > time-and-graph > graph-then-time*

**Intuition**: Temporal features must stabilize before graph operations. Processing graph structure first forces simultaneous learning of both patterns—a harder optimization landscape.

**Empirical**: EvoBrain achieves 95% AUROC on TUSZ (+23% over baselines).

### ⚡ Why O(N) Complexity?

**Problem scale**: 60-second EEG windows at 256Hz = **15,360 samples per channel**. Traditional Transformers:
- **Attention cost**: O(N²) = 236M operations per layer
- **Memory**: O(N²) = 900MB just for attention matrices (batch=1)
- **Inference**: 8 Hz/batch (too slow for clinical real-time)

**State-space solution**: Mamba achieves O(N) via selective state propagation:
- **Cost**: 15K operations (1500× reduction)
- **Memory**: O(N) = 60KB per layer
- **Inference**: 128 Hz/batch ([EEG-Mamba 2024](literature/markdown/EEG-BIMAMBA)) vs 8 Hz/batch for Transformers

### 🔄 Architecture Flow

```
EEG Input (B, 19 channels, 15360 samples @ 256Hz = 60s)
        │
        ▼
  ┌─────────────────────────────────────────────┐
  │ TCN ENCODER (8 layers, 16× downsampling)    │
  │ → Multi-scale temporal decomposition        │
  │ → Parallel processing (no recurrence)       │
  │ → Output: (B, 512, 960) compressed features │
  └─────────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────────┐
  │ PROJECTION → Per-Electrode Features         │
  │ → 512 channels → 19 electrodes × 64 dims    │
  │ → Output: (B, 19, 960, 64)                  │
  └─────────────────────────────────────────────┘
        │
        ├────────────┬─────────────┐
        ▼            ▼             ▼
   ┌────────┐  ┌─────────┐  ┌───────────┐
   │ NODE   │  │  EDGE   │  │ ADJACENCY │
   │ MAMBA  │  │  MAMBA  │  │ ASSEMBLY  │
   │ (19×)  │  │ (171×)  │  │ (learned) │
   └───┬────┘  └────┬────┘  └─────┬─────┘
       │            │             │
       │            └──────┬──────┘
       │                   ▼
       │         ┌────────────────────────┐
       │         │ DYNAMIC LAPLACIAN PE   │
       │         │ → k=16 eigenvectors    │
       │         │ → Time-varying graphs  │
       │         └───────────┬────────────┘
       │                     ▼
       │         ┌────────────────────────┐
       │         │ GNN (2× SSGConv)       │
       │         │ → Spatial aggregation  │
       │         └───────────┬────────────┘
       │                     │
       └─────────┬───────────┘
                 ▼
       ┌──────────────────────┐
       │ GATED FUSION         │
       │ → Learned node/GNN   │
       │   combination        │
       └──────────┬───────────┘
                  ▼
       ┌──────────────────────┐
       │ DECODER              │
       │ → Upsample 16×       │
       │ → Per-sample logits  │
       └──────────────────────┘
                  ▼
         (B, 15360) predictions
```

## ⚙️ Component Justification

### 1. TCN Encoder: Why Not RNNs?

**Temporal Convolutional Networks** ([Bai et al. 2018](literature/markdown/TCN)):
- **Parallelism**: Entire 60s window processed simultaneously (vs sequential RNN)
- **Multi-scale**: Dilated convolutions capture patterns at exponentially growing timescales:
  - Layer 1 (dilation=1): 50ms receptive field (spike detection)
  - Layer 4 (dilation=8): 400ms (rhythmic patterns)
  - Layer 8 (dilation=128): 6.4s (ictal evolution)
- **Stable gradients**: Residual connections prevent vanishing gradients

**Tradeoff**: O(N log N) complexity due to dilation, but negligible for N=15K.

### 2. BiMamba: Why Not Transformers?

**Mamba State-Space Models** ([Gu & Dao 2023](https://arxiv.org/abs/2312.00752)):

**✨ Core innovation**: Selective state propagation with data-dependent gates:
```
S_t = α_t ⊙ S_{t-1} + v_t ⊗ k_t^T    # Forget + update
o_t = S_t q_t                          # Retrieve
```

Where α_t ∈ (0,1) controls memory decay **per timestep** (not global like RNNs).

**Dual-stream design**:
- **Node stream (19 parallel SSMs)**: Independent electrode evolution
  - Captures per-channel patterns (e.g., rhythmic spiking in C3)
  - d_model=64, 6 layers bidirectional
- **Edge stream (171 pairwise SSMs)**: Inter-electrode relationships
  - Models connectivity strength evolution over time
  - d_model=16, 2 layers (lighter, more pairs)

**Why bidirectional?** 60s windows are **offline** analysis—future context improves detection. For real-time deployment, causal Mamba variants exist.

### 3. Dynamic Laplacian PE: Why Not Static Graphs?

**EvoBrain Theorem 1** proves explicit time-varying adjacency is strictly more expressive than static graphs or implicit learning.

**Implementation**:
- Compute **k=16 eigenvectors** of normalized graph Laplacian every 5 timesteps
- Eigenvectors = fixed positional coordinates in spectral space (like Transformer sinusoidal PE)
- Learning happens in GNN layers that **process** PE, not in PE itself ([best practice 2025](docs/04-model/laplacian-pe.md))

**Why top-k=3 neighbors?** Validated by [EvoBrain](literature/markdown/EVOBRAIN.md) on EEG: 3 strongest connections capture 85%+ of spatial variance.

### 4. Gated Fusion: Why Not Simple Addition?

**Problem**: Node stream and GNN produce different feature scales and semantics.

**Solution**: Multi-head gated fusion learns optimal combination:
```
g = σ(W_g [node_out; gnn_out])        # Per-feature gates
fused = g ⊙ node_out + (1-g) ⊙ gnn_out  # Weighted merge
```

This allows the model to emphasize:
- **Node features** when electrodes evolve independently (early seizure)
- **GNN features** when spatial synchronization dominates (propagated seizure)

## 📊 Model Statistics

| Component | Parameters | Complexity | Motivation |
|-----------|-----------|------------|------------|
| **TCN** | 12.8M | O(N log N) | Parallel multi-scale temporal features |
| **Node Mamba** | 7.2M | O(N) | Per-electrode O(N) sequence modeling |
| **Edge Mamba** | 1.2M | O(N) | Inter-electrode relationship evolution |
| **GNN + LPE** | 6.2M | O(N·k²) | Spatial aggregation (k=19 nodes) |
| **Decoder** | 3.1M | O(N) | Upsampling + detection head |
| **Total** | **31.5M** | **O(N)** | Mamba bottleneck dominates |

*Note: GNN is O(N·k²) but k=19 (fixed electrode count) makes it O(N) in sequence length.*

## 🏥 Dataset & Clinical Targets

### TUH EEG Seizure Corpus

**World's largest open-source seizure dataset** ([Picone et al. 2021](literature/markdown/TUSZ-DATA)):
- **504 hours** from 592 patients
- **36 hours** of seizures (~7% prevalence)
- **19-channel** 10-20 montage @ 256Hz
- **Realistic splits**: By patient (prevents data leakage)

### Performance Goals

Based on [Temple Any-Event Scoring (TAES)](literature/markdown/picone-2021-NEDC-SCORING):

| False Alarm Rate | Sensitivity | Clinical Viability |
|------------------|-------------|-------------------|
| 10 FA/24h | >95% | Initial deployment |
| 5 FA/24h | >90% | Standard care |
| **1 FA/24h** | **>75%** | **Gold standard** 🎯 |

**Note**: At 10 FA/24h, alarm fatigue leads to system abandonment. <1 FA/24h enables sustained clinical use.

## 🚀 Quick Start

```bash
# 1️⃣ Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2️⃣ Clone and setup
git clone https://github.com/clarity-digital-twin/brain-go-brr-v2.git
cd brain-go-brr-v2
make setup
make setup-gpu  # Installs mamba-ssm==2.2.5, PyG

# 3️⃣ Download TUH corpus (requires agreement)
# Place in: data_ext4/tusz/edf/

# 4️⃣ Build preprocessing cache (one-time)
python -m src build-cache --data-dir data_ext4/tusz/edf/train --cache-dir cache/tusz/train --split train
python -m src build-cache --data-dir data_ext4/tusz/edf/dev --cache-dir cache/tusz/dev --split dev

# 5️⃣ Run smoke test (5 minutes)
make s

# 6️⃣ Full training (RTX 4090)
export BGB_NAN_DEBUG=1          # Optional: extra logging
# export BGB_SANITIZE_GRADS=1   # Optional: debugging helper
tmux new -s train
make train-local
# Detach: Ctrl+B then D | Reattach: tmux attach -t train
```

**Cloud training (Modal A100-80GB)**:
```bash
modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml
```

See [installation guide](docs/01-installation/) and [training docs](docs/05-training/) for details.

## 🔮 Future Research Directions

### 1. 🎯 Gated Delta Networks (Next-Gen SSM)

**Current**: BiMamba2 uses gated memory (α_t) but lacks targeted updates.

**Next-gen**: [Gated DeltaNet (ICLR 2025)](literature/markdown/GATED-DETLA) combines:
- **Gating** (from Mamba2): Rapid memory erasure for context switches
- **Delta rule** (from DeltaNet): Selective key-value updates without forgetting others

**Why for EEG?** Seizures have abrupt onsets (need memory clearing) and persistent patterns (need selective retention). Gated Delta handles both.

**Implementation**: Available in [FLA (Flash Linear Attention)](https://github.com/fla-org/flash-linear-attention) library. Drop-in replacement for current Mamba layers.

### 2. 🌊 Frequency-Aware Enhancement

**Current limitation**: TCN learns implicit frequency decomposition, but lacks explicit seizure-critical bands.

**Proposed** ([Future Work Doc](docs/future-work/FUTURE_WORK_STFT_ENHANCEMENT.md)): Lightweight 3-band STFT side-branch:
- **Theta/Alpha** (4-12 Hz): Slow wave patterns
- **Beta/Gamma** (14-40 Hz): Fast ictal activity
- **HFO** (80-250 Hz): High-frequency oscillations

**Expected gain**: +2-3% AUROC, <10% compute overhead (based on EvoBrain, EEGM2 results).

### 3. 📐 Multi-Resolution Temporal Modeling

**Current**: Fixed 16× downsampling (960 timesteps).

**Future**: Multi-scale processing at [960, 480, 240] timesteps with late fusion. Captures seizure features at different temporal granularities without increasing complexity.

### 4. 🔗 Hybrid Architectures

**Idea**: Replace some Mamba layers with sliding window attention (like Gated DeltaNet paper).

**Rationale**: Local attention (window=256) provides explicit positional biases that SSMs lack, while Mamba handles long-range dependencies.

**Hypothesis**: Improved training efficiency and short-duration seizure detection.

## 📚 Documentation

**Getting Started**:
- [Quickstart](docs/getting-started/quickstart.md) - 5-minute smoke test
- [Your First Training Run](docs/getting-started/first-run.md) - Complete walkthrough

**Architecture Deep Dives**:
- [V3 Architecture Spec](docs/04-model/v3-architecture.md) - Full implementation details
- [Laplacian PE](docs/04-model/laplacian-pe.md) - Dynamic graph theory
- [Stability Evolution](docs/04-model/v3-stability-evolution.md) - Training fixes (v3.3.0 → v3.4.1)

**Operations**:
- [Training Guide](docs/05-training/) - Local & cloud setup
- [Troubleshooting](docs/08-operations/troubleshooting.md) - Common issues
- [NaN Prevention](docs/08-operations/nan-prevention-complete.md) - Gradient stability

## Contributing

We welcome contributions! See [development docs](docs/09-development/) for:
- Coding standards
- Testing strategy
- Architecture decisions

Run `make q` before committing (lint + format + type check).

## Training Status (v3.9.0)

**v3.9.0 – Production Training Baseline (LIVE):**
- ✅ **Full Modal A100 training launched** - 100 epochs running (app: ap-weaDyLGsgK5TEz8sLLOxO6)
- ✅ **Bulletproof checkpoints** - Atomic saves every 30min, AMP scaler + RNG capture, verified integrity
- ✅ **Timeout guard** - 23h wall-clock limit, 1h safety margin, graceful exit before Modal kill
- ✅ **Comprehensive validation** - PRE_TRAINING_VALIDATION.md, metrics pipeline verified from first principles
- ✅ **Test suite enhanced** - Manifest validation, checkpoint robustness, 75%+ coverage maintained
- ✅ **Zero technical debt** - All P0/P1/P2/P3 issues resolved, production training LIVE

**Previous Milestones:**
- v3.8.3: Manifest naming cleanup complete, zero P0/P1/P2/P3 debt achieved
- v3.8.2: Zero PyTorch warnings (NumPy copy-on-read, AMP scheduler guard)
- v3.8.1: Complete tensor safety across all 3 dataset classes
- v3.8.0: NPZ cache cleanup, type safety, code deduplication

See [release notes](RELEASE_NOTES.md) for complete history.

## Citation

```bibtex
@software{brain-go-brr-v3,
  title = {Brain-Go-Brr V3: Clinical EEG Seizure Detection},
  author = {Clarity Digital Twin},
  year = {2025},
  url = {https://github.com/clarity-digital-twin/brain-go-brr-v2}
}
```

## License

Apache 2.0 - See [LICENSE](LICENSE)

## Acknowledgments

**Datasets**: [TUH EEG Seizure Corpus](literature/markdown/TUSZ-DATA) (Temple), CHB-MIT (Boston Children's/MIT)

**Key Papers**:
- **EvoBrain** ([NeurIPS 2025](literature/markdown/EVOBRAIN.md)) - Dynamic graph theory + time-then-graph paradigm
- **Mamba** ([Gu & Dao 2023](https://arxiv.org/abs/2312.00752)) - Selective state-space models
- **EEG-Mamba** ([2024](literature/markdown/EEG-BIMAMBA)) - BiMamba for EEG classification
- **Gated DeltaNet** ([ICLR 2025](literature/markdown/GATED-DETLA)) - Next-gen SSM with delta rule
- **TCN** ([Bai et al. 2018](literature/markdown/TCN)) - Temporal convolutions
- **Focal Loss** ([Lin et al. 2017](literature/markdown/FOCAL_LOSS)) - Class imbalance handling

**Infrastructure**: [Modal.com](https://modal.com) (A100-80GB), [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/), [mamba-ssm](https://github.com/state-spaces/mamba) (Tri Dao)

---

<div align="center">
<b>Questions?</b> Open an issue | <b>Updates:</b> Watch the repo | <b>Discussion:</b> Start a discussion
</div>
