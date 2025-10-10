# 🧠 Brain-Go-Brr V3: Clinical EEG Seizure Detection

**O(N) complexity seizure detection via dual-stream temporal-spatial architecture**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![PyTorch 2.5.0](https://img.shields.io/badge/pytorch-2.5.0-red.svg)](https://pytorch.org)
[![CUDA 12.4](https://img.shields.io/badge/cuda-12.4-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](LICENSE)
[![v3.9.2](https://img.shields.io/badge/version-3.9.2-blue.svg)](https://github.com/clarity-digital-twin/brain-go-brr-v2/releases/tag/v3.9.2-ci-stability)

## 📋 Overview

Epileptic seizures affect ~50 million people worldwide. Continuous clinical monitoring faces a critical bottleneck: **false alarm rates**. Target performance: <1 false alarm per 24 hours at >75% sensitivity.

### 🎯 The Challenge

EEG seizure detection demands modeling two intertwined phenomena:
- **Temporal dynamics**: Multi-scale patterns from milliseconds (spike transients) to minutes (rhythmic evolution)
- **Spatial propagation**: Time-varying connectivity across 19 scalp electrodes as seizures spread through neural networks

Seizures are spatiotemporal network disorders requiring joint modeling of both dimensions.

### ✨ Our Approach

V3 implements a **dual-stream architecture** grounded in state-space models and dynamic graph theory:

1. **Time-first processing**: TCN + SSM extract temporal features with O(N) complexity
2. **Graph-aware fusion**: Dynamic Laplacian PE captures evolving electrode connectivity
3. **Learned adjacency**: Edge stream models pairwise relationships, not hand-crafted graphs

**Theoretical foundation**: [EvoBrain (NeurIPS 2025)](literature/markdown/EVOBRAIN.md) proves time-then-graph ordering achieves +23% AUROC over alternatives.

**Current status (v3.9.2)**: Production-ready system with bulletproof checkpointing, disk-backed validation, and zero technical debt. Full Modal A100 training live. See [STATUS.md](STATUS.md) for real-time updates.

## 🔬 Research Configuration

We're conducting an **A/B comparison** between two state-space architectures:

**Stack 1: BiMamba2 Baseline** (TRAINING NOW)
- **Node/Edge streams**: BiMamba2 (Mamba2 with bidirectional processing)
- **Status**: Modal A100-80GB, 100 epochs, ~4667 train files
- **Motivation**: Proven selective state propagation ([Gu & Dao 2023](https://arxiv.org/abs/2312.00752))

**Stack 2: Gated DeltaNet** (NEXT)
- **Node/Edge streams**: Gated DeltaNet via FLA (Flash Linear Attention)
- **Status**: Infrastructure complete, smoke tests passed
- **Motivation**: Combines gating (memory erasure) + delta rule (targeted updates) ([ICLR 2025](literature/markdown/GATED-DETLA))

**Why both?** EEG seizures have abrupt onsets (need memory clearing) *and* persistent patterns (need selective retention). Gated Delta theoretically handles both better than gating alone.

**Research goal**: Empirical comparison on full TUSZ dataset. Both results publishable regardless of outcome.

See [FLA_ROADMAP.md](docs/flash-linear-attention/FLA_ROADMAP.md) for complete strategy.

## 🏗️ Architecture: Theory & Design

### 🤔 Why Time-Then-Graph?

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

**State-space solution**: Mamba/GatedDelta achieve O(N) via selective state propagation:
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
   │  SSM   │  │   SSM   │  │ ASSEMBLY  │
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

**Key difference**: SSM boxes = BiMamba2 (Stack 1) or Gated DeltaNet (Stack 2)

## 💡 Component Justification

### 1. TCN Encoder: Why Not RNNs?

**Temporal Convolutional Networks** ([Bai et al. 2018](literature/markdown/TCN)):
- **Parallelism**: Entire 60s window processed simultaneously (vs sequential RNN)
- **Multi-scale**: Dilated convolutions capture patterns at exponentially growing timescales:
  - Layer 1 (dilation=1): 50ms receptive field (spike detection)
  - Layer 4 (dilation=8): 400ms (rhythmic patterns)
  - Layer 8 (dilation=128): 6.4s (ictal evolution)
- **Stable gradients**: Residual connections prevent vanishing gradients

**Tradeoff**: O(N log N) complexity due to dilation, but negligible for N=15K.

### 2. State-Space Models: Why Not Transformers?

**Core innovation**: Selective state propagation with data-dependent gates:
```
S_t = α_t ⊙ S_{t-1} + v_t ⊗ k_t^T    # Forget + update
o_t = S_t q_t                          # Retrieve
```

Where α_t ∈ (0,1) controls memory decay **per timestep** (not global like RNNs).

**BiMamba2 (Stack 1)**:
- Proven architecture ([Gu & Dao 2023](https://arxiv.org/abs/2312.00752))
- Fast CUDA kernels (mamba-ssm 2.2.5)
- Bidirectional processing for offline analysis

**Gated DeltaNet (Stack 2)**:
- Adds delta rule: selective key-value updates without forgetting
- Beats Mamba2 on language modeling ([ICLR 2025](literature/markdown/GATED-DETLA))
- Hypothesis: Better for EEG with abrupt context switches

**Dual-stream design** (both stacks):
- **Node stream (19 parallel SSMs)**: Independent electrode evolution
  - Captures per-channel patterns (e.g., rhythmic spiking in C3)
  - d_model=64, 6 layers bidirectional
- **Edge stream (171 pairwise SSMs)**: Inter-electrode relationships
  - Models connectivity strength evolution over time
  - d_model=16, 2 layers (lighter, more pairs)

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
| **Node SSM** | 7.2M | O(N) | Per-electrode O(N) sequence modeling |
| **Edge SSM** | 1.2M | O(N) | Inter-electrode relationship evolution |
| **GNN + LPE** | 6.2M | O(N·k²) | Spatial aggregation (k=19 nodes) |
| **Decoder** | 3.1M | O(N) | Upsampling + detection head |
| **Total** | **31.5M** | **O(N)** | SSM bottleneck dominates |

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
make setup-gpu  # Installs mamba-ssm==2.2.5, PyG, FLA

# 3️⃣ Download TUH corpus (requires agreement)
# Place in: data_ext4/tusz/edf/

# 4️⃣ Build preprocessing cache (one-time)
python -m src build-cache --data-dir data_ext4/tusz/edf/train --cache-dir cache/tusz_mmap/train --split train
python -m src build-cache --data-dir data_ext4/tusz/edf/dev --cache-dir cache/tusz_mmap/dev --split dev

# 5️⃣ Run smoke test (5 minutes, BiMamba2 stack)
make smoke-bimamba

# 6️⃣ Full training (RTX 4090)
export BGB_NAN_DEBUG=1
tmux new -s train

# BiMamba2 stack (baseline)
make train-bimamba

# OR: Gated DeltaNet stack (research)
make train-fla

# Detach: Ctrl+B then D | Reattach: tmux attach -t train
```

**Cloud training (Modal A100-80GB)**:
```bash
# BiMamba2 baseline (CURRENT)
modal run --detach deploy/modal/app.py --action train --config configs/modal/train_bimamba.yaml

# Gated DeltaNet research (NEXT)
modal run --detach deploy/modal/app.py --action train --config configs/modal/train_fla.yaml
```

See [installation guide](docs/01-installation/) and [training docs](docs/05-training/) for details.

## 🔮 Research Roadmap

### 🎯 Active Experiments

**Phase 1: A/B Architecture Comparison** (IN PROGRESS)
- ✅ BiMamba2 training LIVE (Modal A100, ~4-5 days remaining)
- ⏳ Gated DeltaNet training QUEUED (after BiMamba2 baseline completes)
- 📊 Deliverable: Direct performance comparison on full TUSZ dataset

**Research question**: Does delta rule (targeted updates) improve over pure gating (memory erasure) for clinical EEG?

**Expected outcomes** (all scientifically valuable):
1. Gated Delta > BiMamba2 → "Linear attention mechanisms show X% improvement"
2. BiMamba2 > Gated Delta → "State-space models outperform by X%"
3. Gated Delta ≈ BiMamba2 → "Architecture equivalence observed"

All three outcomes are novel—no prior work compares these stacks on clinical seizure detection.

### 🧪 Future Enhancements

**Frequency-Aware Processing** ([Future Work Doc](docs/future-work/FUTURE_WORK_STFT_ENHANCEMENT.md)):
- Lightweight 3-band STFT side-branch (theta/alpha, beta/gamma, HFO)
- Expected: +2-3% AUROC, <10% compute overhead
- Rationale: Explicit frequency decomposition vs implicit TCN learning

**Multi-Resolution Temporal Modeling**:
- Multi-scale processing at [960, 480, 240] timesteps with late fusion
- Expected: Better short-duration seizure detection
- Rationale: Captures patterns at different temporal granularities

**Hybrid Architectures**:
- Replace some SSM layers with sliding window attention (window=256)
- Expected: Improved positional biases + long-range modeling
- Rationale: Combines local attention strengths with O(N) global context

## 📚 Documentation

**Getting Started**:
- [Quickstart](docs/getting-started/quickstart.md) - 5-minute smoke test
- [Your First Training Run](docs/getting-started/first-run.md) - Complete walkthrough

**Architecture Deep Dives**:
- [V3 Architecture Spec](docs/04-model/v3-architecture.md) - Full implementation details
- [Laplacian PE](docs/04-model/laplacian-pe.md) - Dynamic graph theory
- [Stability Evolution](docs/04-model/v3-stability-evolution.md) - Gradient stability

**Research Documentation**:
- [FLA Roadmap](docs/flash-linear-attention/FLA_ROADMAP.md) - A/B comparison strategy
- [FLA Quick Reference](docs/flash-linear-attention/FLA_QUICK_REFERENCE.md) - Implementation guide

**Operations**:
- [Training Guide](docs/05-training/) - Local & cloud setup
- [Troubleshooting](docs/08-operations/troubleshooting.md) - Common issues
- [NaN Prevention](docs/08-operations/nan-prevention-complete.md) - Gradient stability

## 🤝 Contributing

We welcome contributions! See [development docs](docs/09-development/) for:
- Coding standards
- Testing strategy
- Architecture decisions

Run `make q` before committing (lint + format + type check).

## 📖 Citation

```bibtex
@software{brain-go-brr-v3,
  title = {Brain-Go-Brr V3: Clinical EEG Seizure Detection via Dual-Stream State-Space Models},
  author = {Clarity Digital Twin},
  year = {2025},
  url = {https://github.com/clarity-digital-twin/brain-go-brr-v2},
  note = {A/B comparison of BiMamba2 and Gated DeltaNet architectures}
}
```

## ⚖️ License

Apache 2.0 - See [LICENSE](LICENSE)

## 🙏 Acknowledgments

**Datasets**: [TUH EEG Seizure Corpus](literature/markdown/TUSZ-DATA) (Temple), CHB-MIT (Boston Children's/MIT)

**Key Papers**:
- **EvoBrain** ([NeurIPS 2025](literature/markdown/EVOBRAIN.md)) - Dynamic graph theory + time-then-graph paradigm
- **Mamba** ([Gu & Dao 2023](https://arxiv.org/abs/2312.00752)) - Selective state-space models
- **Gated DeltaNet** ([ICLR 2025](literature/markdown/GATED-DETLA)) - Memory erasure + delta rule
- **EEG-Mamba** ([2024](literature/markdown/EEG-BIMAMBA)) - BiMamba for EEG classification
- **TCN** ([Bai et al. 2018](literature/markdown/TCN)) - Temporal convolutions
- **Focal Loss** ([Lin et al. 2017](literature/markdown/FOCAL_LOSS)) - Class imbalance handling

**Infrastructure**: [Modal.com](https://modal.com) (A100-80GB), [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/), [mamba-ssm](https://github.com/state-spaces/mamba) (Tri Dao), [FLA](https://github.com/fla-org/flash-linear-attention) (Songlin Yang)

---

<div align="center">
<b>Questions?</b> Open an issue | <b>Updates:</b> Watch the repo | <b>Discussion:</b> Start a discussion
</div>
