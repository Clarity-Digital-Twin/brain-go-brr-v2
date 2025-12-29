# 🧠 Brain-Go-Brr V4: Clinical EEG Seizure Detection

**O(N) complexity seizure detection via dual-stack state-space architecture**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![PyTorch 2.5.0](https://img.shields.io/badge/pytorch-2.5.0-red.svg)](https://pytorch.org)
[![CUDA 12.4](https://img.shields.io/badge/cuda-12.4-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](LICENSE)
[![v4.4.0](https://img.shields.io/badge/version-4.4.0-blue.svg)](https://github.com/clarity-digital-twin/brain-go-brr-v2/releases/tag/v4.4.0)

**Current Status (v4.4.0):**
- ✅ **FLA Exp4 COMPLETE**: 78 epochs trained, best @ epoch 63
- 📊 **TUSZ Eval Results**: **35.9% sensitivity @ 10 FA/24h**, AUROC 0.8654
- ⏸️ **BiMamba2**: Paused at epoch 6 (focusing on local training due to cost)

---

## 📋 The Clinical Problem

**50 million people worldwide** suffer from epilepsy. Continuous EEG monitoring in ICUs could catch seizures early—but current systems fail at a critical bottleneck: **false alarm fatigue**.

At 10 false alarms per 24 hours, clinical staff stop responding. The gold standard? **<1 false alarm per day** while maintaining >75% seizure detection. That's what we're building.

---

## 🎯 The Technical Challenge

Seizures aren't just temporal patterns or spatial patterns—they're **both simultaneously**:

- **Temporal dynamics**: Multi-scale patterns from milliseconds (spike transients) → seconds (rhythmic activity) → minutes (ictal evolution)
- **Spatial propagation**: Time-varying electrode connectivity as seizures propagate through neural networks (e.g., C3 → C4 → P3)

Traditional approaches fail because they treat these as separate problems. We model them jointly via **time-then-graph ordering**.

---

## 🔬 Our Approach: Dual-Stack Research Experiment

**Controlled A/B comparison** of two state-space architectures on identical pipeline:

### 🔷 Stack 1: BiMamba2 (Baseline)
- **What**: Mamba2 with bidirectional processing
- **Status**: ⏸️ **PAUSED** at Epoch 6 (Modal A100, $1.1k spent, checkpoints backed up in `backups/modal_bimamba2_epoch6/`)
- **Foundation**: Fast CUDA kernels, selective state propagation ([Gu & Dao 2023](https://arxiv.org/abs/2312.00752))
- **Motivation**: Proven SSM architecture with O(N) efficiency

### 🔶 Stack 2: Gated DeltaNet (Research Variant)
- **What**: FLA (Flash Linear Attention) with gating + delta rule
- **Status**: ✅ **Exp4 COMPLETE** - **35.9% sensitivity @ 10 FA/24h** on TUSZ eval (AUROC 0.8654)
- **Checkpoint/Config**: `results/local_fla_exp4_cyclic/checkpoints/best.pt`, `configs/local/train_fla_exp4_cyclic.yaml`
- **Foundation**: Beats Mamba2 on language modeling ([ICLR 2025](https://github.com/NVlabs/GatedDeltaNet))
- **Hypothesis**: Better for EEG's abrupt context switches (seizure onsets)
- **Next**: Close gap to Temple SOTA (4 FA/24h @ ~50% sensitivity)

**Why both?** Seizures have **abrupt onsets** (need memory clearing via gating) *and* **persistent patterns** (need selective retention via delta rule). Gated Delta theoretically handles both. But does theory match clinical reality? That's what we're testing.

**Research transparency**: All three outcomes (Gated Delta wins, BiMamba2 wins, or tie) are scientifically valuable. No prior work compares these architectures on clinical EEG analysis. See [`docs/04-model/flash-linear-attention/FLA_ROADMAP.md`](docs/04-model/flash-linear-attention/FLA_ROADMAP.md) for full strategy.

---

## 🏗️ Architecture: Theory & Design

### 🤔 Why Time-Then-Graph?

EvoBrain ([NeurIPS 2025](https://arxiv.org/abs/2509.15857)) establishes two critical theorems:

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
- **Inference**: 128 Hz/batch (EEG-Mamba 2024) vs 8 Hz/batch for Transformers

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

**Key**: SSM boxes = **🔷 BiMamba2** (Stack 1) or **🔶 Gated DeltaNet** (Stack 2)

Everything else is identical—TCN frontend, GNN backend, fusion layer. Only the temporal core changes.

---

## 💡 Component Justification

### 1. TCN Encoder: Multi-Scale Temporal Decomposition

**Temporal Convolutional Networks** ([Bai et al. 2018](https://arxiv.org/abs/1803.01271)):
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

### 3. Dynamic Laplacian PE: Time-Evolving Graph Structure

EvoBrain's Theorem 1 proves explicit time-varying adjacency is strictly more expressive than static graphs.

**Implementation**:
- Compute **k=16 eigenvectors** of normalized graph Laplacian every 5 timesteps
- Eigenvectors = fixed positional coordinates in spectral space (like Transformer sinusoidal PE)
- Learning happens in GNN layers that **process** PE, not in PE itself ([best practice](docs/04-model/laplacian-pe.md))

**Why top-k=3 neighbors?** 3 strongest connections capture 85%+ of spatial variance (validated by EvoBrain on EEG).

### 4. Gated Fusion: Adaptive Feature Combination

**Problem**: Node stream and GNN produce different feature scales and semantics.

**Solution**: Multi-head gated fusion learns optimal combination:
```python
g = σ(W_g [node_out; gnn_out])        # Per-feature gates
fused = g ⊙ node_out + (1-g) ⊙ gnn_out  # Weighted merge
```

This allows the model to emphasize:
- **Node features** when electrodes evolve independently (early seizure)
- **GNN features** when spatial synchronization dominates (propagated seizure)

---

## 📊 Model Statistics: Side-by-Side Comparison

| Component | BiMamba2 (Stack 1) | Gated DeltaNet (Stack 2) | Complexity |
|-----------|-------------------|-------------------------|------------|
| **TCN Encoder** | 12.8M | 12.8M (identical) | O(N log N) |
| **Node SSM** | 7.2M (d_model=64) | 7.2M (d_model=512) | O(N) |
| **Edge SSM** | 1.2M (d_model=16) | 1.2M (d_model=32) | O(N) |
| **GNN + LPE** | 6.2M | 6.2M (identical) | O(N·k²) |
| **Fusion** | 2.1M | 2.1M (identical) | O(N) |
| **Decoder** | 1.0M | 1.0M (identical) | O(N) |
| **Total** | **30.5M** | **30.5M** (matched) | **O(N)** |

**🔑 Key**: Parameter counts matched for fair comparison. Only Node/Edge SSM layers differ. TCN frontend, GNN backend, fusion, and decoder are 100% identical.

---

## 🏥 Dataset: TUSZ Clinical Reality

### TUH EEG Seizure Corpus

**World's largest open-source seizure dataset** ([Temple University](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/index.shtml)):
- **504 hours** of continuous EEG from 592 patients
- **36 hours** of seizures (~7% prevalence) → 12:1 class imbalance
- **19-channel** 10-20 montage @ 256Hz (clinical standard)
- **Patient-based splits** (train/dev/eval) → no data leakage

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

**Why oversample training?** Standard ML practice: Train on balanced data (model learns seizure patterns), validate on natural distribution (measures real-world performance). See [`docs/05-training/training-methodology.md`](docs/05-training/training-methodology.md) for detailed explanation.

---

## 🎯 Performance Targets: Evidence-Based Goals

Based on verified clinical benchmarks and SOTA research (see [`docs/00-overview/performance-targets.md`](docs/00-overview/performance-targets.md) for comprehensive analysis):

### Primary Target (Match Temple Clinical SOTA)
**≤4 FA/24h @ ≥50% sensitivity** (NEDC OVERLAP scoring)

- **Temple NEDC verified**: 4 FA/24h @ ~50% sensitivity (real clinical deployments)
- **SeizureTransformer #1**: 26.89 FA/24h @ 45.63% sensitivity (TUSZ eval, 2025)
- **Our goal**: Match or beat Temple's verified clinical benchmark

### Stretch Goal (Clinical Deployment)
**≤10 FA/24h @ ≥75% sensitivity** (NEDC OVERLAP scoring)

- Enables ICU monitoring with manageable alarm fatigue
- SeizureTransformer @ ~10 FA (OVERLAP): 33.90% sensitivity → Exp4: 35.9% (+2.0 points), still below Temple target (4 FA/24h @ ~50%)

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

**Scoring impact**: Same predictions can yield 3-16× different FA rates depending on scorer (SzCORE vs NEDC OVERLAP vs NEDC TAES). We use **NEDC OVERLAP** as primary metric. See [`docs/06-evaluation/TAES_DISAMBIGUATION.md`](docs/06-evaluation/TAES_DISAMBIGUATION.md) for critical naming collision explanation.

---

## 📊 Results: FLA Exp4 (Gated DeltaNet) - December 2025

**Training completed** on RTX 4090 local GPU over 6 weeks.

### TUSZ Eval Set (Held-Out Test)

| Metric | Value | Notes |
|--------|-------|-------|
| **AUROC** | **0.8654** | Strong discrimination |
| **PR-AUC** | 0.5409 | Handles 12:1 imbalance |
| **Sensitivity @ 10 FA/24h** | **35.9%** | Primary clinical metric |
| **Sensitivity @ 5 FA/24h** | 27.1% | Stricter threshold |
| **Sensitivity @ 2.5 FA/24h** | 18.6% | Very strict |
| **Sensitivity @ 1 FA/24h** | 5.8% | Clinical gold standard |
| **ECE** | 0.029 | Well-calibrated |
| **Val Loss** | 0.090 | Focal loss |

**Dataset**: TUSZ eval split (865 EDF/label pairs) → 836 recordings scored (29 yielded 0 windows under 60s windowing), 127.8 hours

### Training Details

| Parameter | Value |
|-----------|-------|
| **Architecture** | TCN + BiGatedDeltaNet (FLA) + GNN + Dynamic LPE |
| **Total Epochs** | 78 (early stopped, patience=15) |
| **Best Epoch** | 63 |
| **Dev Sensitivity @ 10FA** | 29.0% (validation during training) |
| **Training Time** | ~6 weeks on RTX 4090 |
| **Config** | `configs/local/train_fla_exp4_cyclic.yaml` |
| **Checkpoint** | `results/local_fla_exp4_cyclic/checkpoints/best.pt` |
| **Results JSON (SSOT)** | `results/local_fla_exp4_cyclic/eval_results_v2.json` |

### Comparison to SeizureTransformer (Same TUSZ Eval Split, OVERLAP Scoring)

| FA Rate | FLA Exp4 | SeizureTransformer | Delta |
|--------:|---------:|------------------:|------:|
| 10 FA/24h | **35.9%** | 33.90% | **+2.0%** |
| 2.5 FA/24h | **18.6%** | 14.50% | **+4.1%** |

SeizureTransformer numbers are from our run in `reference_repos/SeizureTransformer/docs/results/FINAL_COMPREHENSIVE_RESULTS_TABLE.md` (Python OVERLAP = NEDC OVERLAP).

**Key Insight**: We now beat SeizureTransformer at the two tuned clinical operating points (10 and 2.5 FA/24h), but remain below Temple's verified clinical SOTA (≈50% @ 4 FA/24h).

### Validation During Training (Dev Set)

Best epoch 63 metrics on dev (validation) set:
- Sensitivity @ 10 FA/24h: 29.0%
- AUROC: 0.7792
- TAES (metric): 1.0000

**Note**: Eval performance exceeded dev at the 10 FA operating point (35.9% vs 29.0%); this can happen due to split differences.

---

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
# Visit: https://isip.piconepress.com/projects/nedc/html/tuh_eeg/index.shtml
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

**Cloud training (Modal A100-80GB)** - See [`docs/05-training/modal.md`](docs/05-training/modal.md) for details:

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
```

**🚨 CRITICAL**: Use `--action schedule-training` for 100-epoch production runs (auto-restart every 23h). Use `--action train` ONLY for smoke tests and experiments.

See [`docs/01-installation/`](docs/01-installation/) and [`docs/05-training/`](docs/05-training/) for complete setup guides.

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
- [FLA Roadmap](docs/04-model/flash-linear-attention/FLA_ROADMAP.md) - Complete A/B strategy
- [FLA Quick Reference](docs/04-model/flash-linear-attention/FLA_QUICK_REFERENCE.md) - Config guide
- [Future Work](docs/future-work/) - Post-training enhancements

### Operations
- [Training Guide](docs/05-training/) - Local + Modal setup
- [Training Methodology](docs/05-training/training-methodology.md) - Why validation has more batches
- [Modal Timeout Guard](docs/08-operations/modal-timeout-guard.md) - Three-layer defense system
- [Troubleshooting](docs/08-operations/troubleshooting.md) - Common issues
- [NaN Prevention](docs/08-operations/gradient-protection-guide.md) - Gradient stability

---

## 🤝 Contributing

We welcome contributions! See [`docs/09-development/`](docs/09-development/) for:
- [Coding Standards](docs/09-development/coding-standards.md) (Ruff, mypy, no comments unless requested)
- [Testing Strategy](docs/09-development/testing.md) (`make q` before committing)
- [Technical Debt](docs/09-development/technical-debt.md) (currently zero!)

**Zero technical debt policy**: All P0/P1/P2 issues resolved before major releases.

---

## 📖 Citation

```bibtex
@software{brain-go-brr-v4,
  title = {Brain-Go-Brr V4: Clinical EEG Seizure Detection via Dual-Stack State-Space Models},
  author = {Clarity Digital Twin},
  year = {2025},
  version = {4.4.0},
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
- [TUH EEG Seizure Corpus](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/index.shtml) (Temple University)
- CHB-MIT Scalp EEG Database (Boston Children's Hospital / MIT)

**Foundational Papers**:
- **EvoBrain** ([Kotoge et al., NeurIPS 2025](https://arxiv.org/abs/2509.15857)) - Time-then-graph paradigm, dynamic graphs
- **Mamba** ([Gu & Dao 2023](https://arxiv.org/abs/2312.00752)) - Selective state-space models
- **Gated DeltaNet** ([Yang et al., ICLR 2025](https://github.com/NVlabs/GatedDeltaNet)) - Memory erasure + delta rule
- **SeizureTransformer** ([Wu et al. 2025](https://arxiv.org/abs/2504.00336)) - SOTA baseline, U-Net + Transformer (EpilepsyBench #1)
- **EEGMamba** ([Gui et al. 2024](https://arxiv.org/abs/2407.20254)) - Bidirectional Mamba for EEG (speed benchmark)
- **TCN** ([Bai et al. 2018](https://arxiv.org/abs/1803.01271)) - Temporal convolutional networks
- **Focal Loss** ([Lin et al. 2017](https://arxiv.org/abs/1708.02002)) - Class imbalance handling

**Infrastructure & Libraries**:
- [Modal.com](https://modal.com) - A100-80GB GPU infrastructure
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) - Graph neural networks
- [mamba-ssm](https://github.com/state-spaces/mamba) (Tri Dao) - Mamba2 implementation
- [FLA](https://github.com/fla-org/flash-linear-attention) (Songlin Yang) - Gated DeltaNet implementation

---

<div align="center">

**Questions?** [Open an issue](https://github.com/clarity-digital-twin/brain-go-brr-v2/issues) • **Updates?** [Watch the repo](https://github.com/clarity-digital-twin/brain-go-brr-v2) • **Discussion?** [Start a discussion](https://github.com/clarity-digital-twin/brain-go-brr-v2/discussions)

**Status**: v4.4.0 FLA Exp4 COMPLETE • **35.9% sensitivity @ 10 FA/24h on TUSZ eval** • BiMamba2 paused (Epoch 6) • See [`STATUS.md`](STATUS.md) for full details

</div>
