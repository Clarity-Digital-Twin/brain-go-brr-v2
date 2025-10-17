# 🧠 Brain-Go-Brr V4: Clinical EEG Seizure Detection

**O(N) complexity seizure detection via dual-stack state-space architecture**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![PyTorch 2.5.0](https://img.shields.io/badge/pytorch-2.5.0-red.svg)](https://pytorch.org)
[![CUDA 12.4](https://img.shields.io/badge/cuda-12.4-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](LICENSE)
[![v4.0.0](https://img.shields.io/badge/version-4.0.0-blue.svg)](https://github.com/clarity-digital-twin/brain-go-brr-v2/releases/tag/v4.0.0-fla-production-wsl2-fix)

## 📋 The Clinical Problem

**50 million people worldwide** suffer from epilepsy. Continuous EEG monitoring in ICUs could catch seizures early—but current systems fail at a critical bottleneck: **false alarm fatigue**.

At 10 false alarms per 24 hours, clinical staff stop responding. The gold standard? **<1 false alarm per day** while maintaining >75% seizure detection.

That's what we're building.

---

## 🎯 The Technical Challenge

Seizures aren't just temporal patterns or spatial patterns—they're **both simultaneously**:

- **Temporal dynamics**: Multi-scale patterns from milliseconds (spike transients) → seconds (rhythmic activity) → minutes (ictal evolution)

- **Spatial propagation**: Time-varying electrode connectivity as seizures propagate through neural networks (e.g., C3 → C4 → P3)

Traditional approaches fail because they treat these as separate problems. We model them jointly via **time-then-graph ordering**.

---

## 🔬 Our Approach: Dual-Stack Research Experiment

**Controlled A/B comparison** of two state-space architectures:

### 🔷 Stack 1: BiMamba2 (baseline)
- **What**: Mamba2 with bidirectional processing
- **Status**: ⏸️ PAUSED at Epoch 6 (Modal A100, $1,118 spent, checkpoints backed up)
- **Foundation**: Fast CUDA kernels, selective state propagation ([Gu & Dao 2023](https://arxiv.org/abs/2312.00752))
- **Motivation**: Proven SSM architecture with O(N) efficiency

### 🔶 Stack 2: Gated DeltaNet (research variant)
- **What**: FLA (Flash Linear Attention) with gating + delta rule
- **Status**: 🟢 ACTIVE - Local RTX 4090 training (Epoch 2, progressing normally)
- **Foundation**: Beats Mamba2 on language modeling ([ICLR 2025](literature/markdown/GATED-DETLA))
- **Hypothesis**: Better for EEG's abrupt context switches (seizure onsets)

**Why both?** Seizures have **abrupt onsets** (need memory clearing via gating) *and* **persistent patterns** (need selective retention via delta rule). Gated Delta theoretically handles both. But does theory match clinical reality? That's what we're testing.

**Research transparency**: All three outcomes (Gated Delta wins, BiMamba2 wins, or tie) are scientifically valuable. No prior work compares these architectures on clinical EEG. See [FLA_ROADMAP.md](docs/flash-linear-attention/FLA_ROADMAP.md) for full strategy.

**Current status (v4.0.0)**: **FLA-FOCUSED TRAINING** - BiMamba2 (Modal, PAUSED at Epoch 6, $1,118 spent, checkpoints backed up) + FLA (Local RTX 4090, Epoch 2, ACTIVE). Budget-conscious decision: finish FLA first (free), resume BiMamba2 later if needed. See [STATUS.md](STATUS.md) and [backup README](backups/modal_bimamba2_epoch6/README.md) for details.

## 🏗️ Architecture: Theory & Design

### 🤔 Why Time-Then-Graph?

[EvoBrain (NeurIPS 2025)](literature/markdown/EVOBRAIN.md) establishes two critical theorems:

- **Theorem 1 (Dynamic Graphs)**: Explicit dynamic modeling (time-varying adjacency) is strictly more expressive than implicit (static graphs)
- **Theorem 2 (Temporal Ordering)**: time-then-graph > time-and-graph > graph-then-time

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
  │ → Dilations: 1→2→4→8→16→32→64→128           │
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
        ├──────────────┬──────────────┐
        ▼              ▼              ▼
   ┌─────────┐   ┌─────────┐   ┌───────────┐
   │  NODE   │   │  EDGE   │   │ ADJACENCY │
   │   SSM   │   │   SSM   │   │ ASSEMBLY  │
   │  (19×)  │   │ (171×)  │   │ (learned) │
   └────┬────┘   └────┬────┘   └─────┬─────┘
        │             │              │
        │             └──────┬───────┘
        │                    ▼
        │          ┌────────────────────────┐
        │          │ DYNAMIC LAPLACIAN PE   │
        │          │ → k=16 eigenvectors    │
        │          │ → Every 5 timesteps    │
        │          └──────────┬─────────────┘
        │                     ▼
        │          ┌────────────────────────┐
        │          │ GNN (2× SSGConv)       │
        │          │ → Spatial aggregation  │
        │          │ → Alpha=0.05           │
        │          └──────────┬─────────────┘
        │                     │
        └─────────────────────┴─► (B, 19, 960, 128)
                                  ▼
                        ┌──────────────────┐
                        │ GATED FUSION     │
                        │ → 4-head combine │
                        │ → Node + spatial │
                        └────────┬─────────┘
                                 ▼
                        ┌──────────────────┐
                        │ DECODER          │
                        │ → Upsample 16×   │
                        │ → Per-sample     │
                        └────────┬─────────┘
                                 ▼
                        (B, 15360) logits
```

**🔑 Key**: SSM boxes = **🔷 BiMamba2** (Stack 1) or **🔶 Gated DeltaNet** (Stack 2)

Everything else is identical—TCN frontend, GNN backend, fusion layer. Only the temporal core changes.

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

### 2. State-Space Models: The Heart of the System

**Core innovation**: Selective state propagation with data-dependent gates
```python
S_t = α_t ⊙ S_{t-1} + v_t ⊗ k_t^T    # Forget (α) + update (v⊗k)
o_t = S_t q_t                          # Retrieve
```

Where α_t ∈ (0,1) controls **per-timestep memory decay** (not global like RNNs).

#### 🔷 BiMamba2 Architecture (Stack 1)

**Node Stream** (19 parallel SSMs):
- **Purpose**: Model per-electrode temporal dynamics independently
- **Config**: 6 layers, d_model=64, d_state=16, bidirectional
- **Example**: Rhythmic spiking in C3 electrode evolves independently
- **Parameters**: 7.2M

**Edge Stream** (171 pairwise SSMs):
- **Purpose**: Model inter-electrode connectivity strength over time
- **Config**: 2 layers, d_model=16, d_state=8, bidirectional
- **Example**: C3-C4 coherence increases during seizure propagation
- **Parameters**: 1.2M

**Total SSM**: 8.4M parameters, O(N) complexity

#### 🔶 Gated DeltaNet Architecture (Stack 2)

**Key difference**: Adds **delta rule** on top of gating

**Delta rule**: Selective key-value updates without forgetting others
```python
# Mamba2: Global gate (erases everything)
S_t = α_t ⊙ S_{t-1} + update

# Gated DeltaNet: Targeted update (selective retention)
S_t = α_t ⊙ S_{t-1} + β_t ⊙ (k_t ⊗ v_t - old_memory)
```

**Configuration**:
- **Node Stream**: 6 layers, d_model=512, num_heads=6, headdim=8
- **Edge Stream**: 2 layers, d_model=32, num_heads=3, headdim=8

**Total SSM**: ~8.4M parameters (matched to BiMamba2), O(N) complexity

**Hypothesis**: Delta rule handles EEG better because:
1. **Gating** clears memory during seizure onset (abrupt context switch)
2. **Delta rule** preserves persistent patterns (rhythmic activity continues)
3. BiMamba2 has only gating → may "forget" ongoing rhythms during onset

**Reality check**: This is a hypothesis. Full TUSZ training will tell us if it's true.

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

## 📊 Model Statistics: Side-by-Side Comparison

### 🔷 Stack 1: BiMamba2

| Component | Parameters | Complexity | Details |
|-----------|-----------|------------|---------|
| **TCN Encoder** | 12.8M | O(N log N) | 8 layers, channels [64,128,256,512] |
| **Node BiMamba2** | 7.2M | O(N) | 19 parallel SSMs, 6 layers, d_model=64 |
| **Edge BiMamba2** | 1.2M | O(N) | 171 pairwise SSMs, 2 layers, d_model=16 |
| **GNN + LPE** | 6.2M | O(N·k²) | 2× SSGConv, k=16 eigenvectors |
| **Gated Fusion** | 2.1M | O(N) | 4-head attention fusion |
| **Decoder** | 1.0M | O(N) | 16× upsampling, detection head |
| **Total** | **30.5M** | **O(N)** | SSM bottleneck dominates |

### 🔶 Stack 2: Gated DeltaNet

| Component | Parameters | Complexity | Details |
|-----------|-----------|------------|---------|
| **TCN Encoder** | 12.8M | O(N log N) | *Identical to Stack 1* |
| **Node GatedDelta** | 7.2M | O(N) | 19 parallel SSMs, 6 layers, d_model=512 |
| **Edge GatedDelta** | 1.2M | O(N) | 171 pairwise SSMs, 2 layers, d_model=32 |
| **GNN + LPE** | 6.2M | O(N·k²) | *Identical to Stack 1* |
| **Gated Fusion** | 2.1M | O(N) | *Identical to Stack 1* |
| **Decoder** | 1.0M | O(N) | *Identical to Stack 1* |
| **Total** | **30.5M** | **O(N)** | Matched parameter count for fair comparison |

**🔑 Key Difference**: Only Node/Edge SSM layers differ. TCN frontend, GNN backend, fusion, and decoder are 100% identical.

**Note**: GNN is O(N·k²) but k=19 (fixed electrode count) makes it effectively O(N) in sequence length.

## 🏥 Dataset: TUSZ Clinical Reality

### TUH EEG Seizure Corpus

**World's largest open-source seizure dataset** ([Temple University](literature/markdown/TUSZ-DATA)):
- **504 hours** of continuous EEG from 592 patients
- **36 hours** of seizures (~7% prevalence) → 12:1 class imbalance
- **19-channel** 10-20 montage @ 256Hz (clinical standard)
- **Patient-based splits** (train/dev/test) → no data leakage

**Preprocessing pipeline**:
1. Bandpass filter: 0.5-120Hz
2. Notch filter: 60Hz (removes powerline noise)
3. Resample: 256Hz (standardize across recordings)
4. Windowing: 60s windows, 10s stride (83% overlap)
5. Normalization: Per-channel z-score + clip to ±10σ (removes outliers)

**Our cache system** (memory-mapped NPY format):
- **Train**: 4667 files → 61,616 balanced windows (34.2% seizure ratio via oversampling)
- **Dev**: 1832 files → 148,224 natural windows (7.7% seizure ratio, real distribution)
- **Speed**: 99.6% faster startup than NPZ (manifest-based loading)
- **Memory**: <1 GB RAM vs 387 GB for NPZ

**Why oversample training?** Standard ML practice: Train on balanced data (model learns seizure patterns), validate on natural distribution (measures real-world performance).

---

## 🎯 Performance Targets: Evidence-Based Goals

Based on verified clinical benchmarks and SOTA research (see [REALISTIC_PERFORMANCE_TARGETS.md](REALISTIC_PERFORMANCE_TARGETS.md) for comprehensive analysis):

### Primary Target (Match Temple Clinical SOTA)
**≤4 FA/24h @ ≥50% sensitivity** (NEDC OVERLAP scoring)

- **Temple NEDC verified**: 4 FA/24h @ ~50% sensitivity (real clinical deployments)
- **SeizureTransformer #1**: 26.89 FA/24h @ 45.63% sensitivity (TUSZ eval, 2025)
- **Our goal**: Match or beat Temple's verified clinical benchmark

### Stretch Goal (Clinical Deployment)
**≤10 FA/24h @ ≥75% sensitivity** (NEDC OVERLAP scoring)

- Enables ICU monitoring with manageable alarm fatigue
- Represents significant breakthrough over current systems
- Current gap: SeizureTransformer @ 10 FA = 33.90% sensitivity (42-point gap to close)

### Aspirational Gold Standard
**≤1 FA/24h @ ≥75% sensitivity** (NEDC OVERLAP scoring)

- Human reviewer performance level
- Likely impossible with current architectures (64 points above SOTA)
- Included as long-term research direction

### Additional Metrics (Threshold-Independent)

| Metric | Target | Baseline (SeizureTransformer) | Rationale |
|--------|--------|-------------------------------|-----------|
| **AUROC** | ≥0.90 | 0.902 (TUSZ eval) | Overall discrimination capability |
| **AUPRC** | ≥0.40 | Not reported | Better for 12:1 class imbalance |
| **F1 Score** | ≥0.45 | 0.414 (NEDC OVERLAP) | Balanced precision/recall |

### Realistic Success Criteria

| Outcome | Sensitivity @ 4 FA/24h | Publication Tier |
|---------|------------------------|------------------|
| **Breakthrough** | ≥60% | Top-tier venue (beats all known systems) |
| **Strong** | ≥50% | Highly publishable (matches Temple SOTA) |
| **Publishable** | ≥45% | Solid contribution (architectural novelty) |
| **Minimum** | ≥40% | Viable if architectural insights clear |

**Reality check**: Temple NEDC research confirms ROC curves are **very steep** at low FA rates. 5% absolute sensitivity change = massive FA rate shift. Our dual-stack (BiMamba2 vs Gated DeltaNet) comparison provides scientific value regardless of absolute performance.

**Scoring impact**: Same predictions can yield 3-16× different FA rates depending on scorer (SzCORE vs NEDC OVERLAP vs NEDC TAES). We use **NEDC OVERLAP** as primary metric (standard for TUSZ evaluation).

**Full analysis**: See [REALISTIC_PERFORMANCE_TARGETS.md](REALISTIC_PERFORMANCE_TARGETS.md) for comprehensive benchmark comparison, scorer differences, and architectural tables.

## 🚀 Quick Start

```bash
# 1️⃣ Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2️⃣ Clone repo
git clone https://github.com/clarity-digital-twin/brain-go-brr-v2.git
cd brain-go-brr-v2

# 3️⃣ Setup environment (installs mamba-ssm, PyG)
make setup
make setup-gpu

# Optional: Install FLA for Gated DeltaNet research
make setup-fla

# 4️⃣ Download TUSZ corpus
# Visit: https://isip.piconepress.com/projects/tuh_eeg/html/request_access.php
# Place in: data_ext4/tusz/edf/

# 5️⃣ Build preprocessing cache (one-time, ~2 hours)
python -m src build-cache \
  --data-dir data_ext4/tusz/edf/train \
  --cache-dir cache/tusz_mmap/train \
  --split train

python -m src build-cache \
  --data-dir data_ext4/tusz/edf/dev \
  --cache-dir cache/tusz_mmap/dev \
  --split dev

# 6️⃣ Smoke test (3 files, 5 minutes)
make smoke-bimamba    # Test BiMamba2 stack
make smoke-fla        # Test Gated DeltaNet stack

# 7️⃣ Full local training (RTX 4090, ~960 hours / 40 days)
export BGB_NAN_DEBUG=1
tmux new -s train
make train-bimamba    # or: make train-fla
# Ctrl+B then D to detach | tmux attach -t train to reattach
```

**Cloud training (Modal A100-80GB, ~700-1200 hours, $3,400-$5,300+)** - **EXPENSIVE due to validation overhead**:

**🚨 CRITICAL**: Use `--action schedule-training` for 100-epoch production runs (auto-restart every 23h).

```bash
# Deploy Modal functions first
modal deploy deploy/modal/app.py

# BiMamba2 production (hands-free, auto-restart)
modal run --detach deploy/modal/app.py \
  --action schedule-training \
  --config configs/modal/train_bimamba.yaml

# Gated DeltaNet production (hands-free, auto-restart)
modal run --detach deploy/modal/app.py \
  --action schedule-training \
  --config configs/modal/train_fla.yaml

# Monitor progress
modal app list
modal app logs <app-id>

# Manual mode (use ONLY for experiments, NOT production)
# Runs ONCE, requires manual restart after 23h timeout
modal run --detach deploy/modal/app.py \
  --action train \
  --config configs/modal/train_bimamba.yaml \
  --resume
```

See [installation guide](docs/01-installation/) and [training docs](docs/05-training/) for details.

---

## 🔬 Research Timeline

### Current Phase: FLA-Focused Training (October 2025)

**🔷 Stack 1 (BiMamba2)**: ⏸️ **PAUSED** at Epoch 6 (batch 647/1284, 50% through). Modal A100 training stopped Oct 13 due to budget control ($1,118 spent). Checkpoints backed up to Modal SSD + local (`backups/modal_bimamba2_epoch6/`). Resumable anytime.

**🔶 Stack 2 (Gated DeltaNet)**: 🟢 **ACTIVE** - Local RTX 4090 training on Epoch 7/100 (7% complete), progressing normally. WSL2 SIGBUS fix enables local training (cache on ext4).

**Research question**: Does the delta rule improve over pure gating for clinical EEG seizure detection?

**Current strategy**: Budget-conscious approach as independent researcher. Complete FLA training first (free, ~960 hours / 40 days), analyze results, then decide if BiMamba2 comparison needed. If needed, resume BiMamba2 incrementally ($500-1k/month). Already have 6 epochs of BiMamba2 data for early comparison.

**Analysis plan**: Evaluate FLA on sensitivity@1FA/@5FA/@10FA, AUROC, and TAES scores. Compare against BiMamba2's 6-epoch baseline if both datasets available. All three outcomes (Gated Delta wins, BiMamba2 wins, or equivalence) are scientifically valuable—no prior work compares these architectures on TUSZ.

---

## 📚 Documentation

### Getting Started
- [Quickstart](docs/getting-started/quickstart.md) - 5-minute validation
- [First Training Run](docs/getting-started/first-run.md) - Complete walkthrough

### Architecture
- [V3 Spec](docs/04-model/v3-architecture.md) - Full implementation details
- [Laplacian PE](docs/04-model/laplacian-pe.md) - Dynamic graph theory
- [Stability Evolution](docs/04-model/v3-stability-evolution.md) - NaN prevention history

### Research
- [FLA Roadmap](docs/flash-linear-attention/FLA_ROADMAP.md) - Complete A/B strategy
- [FLA Quick Reference](docs/flash-linear-attention/FLA_QUICK_REFERENCE.md) - Config guide
- [Future Work](docs/future-work/) - Post-training enhancements

### Operations
- [Training Guide](docs/05-training/) - Local + Modal setup
- [Troubleshooting](docs/08-operations/troubleshooting.md) - Common issues
- [NaN Prevention](docs/08-operations/nan-prevention-complete.md) - Gradient stability

---

## 🤝 Contributing

We welcome contributions! See [development docs](docs/09-development/) for:
- Coding standards (Ruff, mypy, no comments unless requested)
- Testing strategy (`make q` before committing)
- Architecture decision records

**Zero technical debt policy**: All P0/P1/P2 issues resolved before major releases.

---

## 📖 Citation

```bibtex
@software{brain-go-brr-v4,
  title = {Brain-Go-Brr V4: Clinical EEG Seizure Detection via Dual-Stack State-Space Models},
  author = {Clarity Digital Twin},
  year = {2025},
  version = {4.0.0},
  url = {https://github.com/clarity-digital-twin/brain-go-brr-v2},
  note = {Empirical A/B comparison of BiMamba2 and Flash Linear Attention (BiGatedDeltaNet) architectures on TUSZ}
}
```

---

## ⚖️ License

Apache 2.0 - See [LICENSE](LICENSE) for full text.

---

## 🙏 Acknowledgments

**Datasets**:
- [TUH EEG Seizure Corpus](literature/markdown/TUSZ-DATA) (Temple University)
- CHB-MIT Scalp EEG Database (Boston Children's Hospital / MIT)

**Foundational Papers**:
- **EvoBrain** ([NeurIPS 2025](literature/markdown/EVOBRAIN.md)) - Time-then-graph paradigm, dynamic graphs
- **Mamba** ([Gu & Dao 2023](https://arxiv.org/abs/2312.00752)) - Selective state-space models
- **Gated DeltaNet** ([Yang et al., ICLR 2025](literature/markdown/GATED-DETLA)) - Memory erasure + delta rule
- **EEG-Mamba** ([2024](literature/markdown/EEG-BIMAMBA)) - BiMamba for EEG classification
- **TCN** ([Bai et al. 2018](literature/markdown/TCN)) - Temporal convolutional networks
- **Focal Loss** ([Lin et al. 2017](literature/markdown/FOCAL_LOSS)) - Class imbalance handling

**Infrastructure & Libraries**:
- [Modal.com](https://modal.com) - A100-80GB GPU infrastructure
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) - Graph neural networks
- [mamba-ssm](https://github.com/state-spaces/mamba) (Tri Dao) - Mamba2 implementation
- [FLA](https://github.com/fla-org/flash-linear-attention) (Songlin Yang) - Gated DeltaNet implementation

---

<div align="center">

**Questions?** [Open an issue](https://github.com/clarity-digital-twin/brain-go-brr-v2/issues) •
**Updates?** [Watch the repo](https://github.com/clarity-digital-twin/brain-go-brr-v2) •
**Discussion?** [Start a discussion](https://github.com/clarity-digital-twin/brain-go-brr-v2/discussions)

**Current status (v4.0.0)**: FLA-focused training → BiMamba2 (Modal, PAUSED at Epoch 6, $1.1k spent, backed up) + FLA (Local RTX 4090, Epoch 7/100, ACTIVE) • Strategy: Complete FLA first (free, ~960h), resume BiMamba2 incrementally if comparison needed • See [STATUS.md](STATUS.md) for full rationale

</div>
