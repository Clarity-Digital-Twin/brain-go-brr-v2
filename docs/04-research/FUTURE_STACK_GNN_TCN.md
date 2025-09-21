# 🚀 Future Roadmap: Next-Generation EEG Architecture Stack

## Executive Summary

Moving beyond the current U-Net + ResCNN + Bi-Mamba-2 architecture to address fundamental limitations in EEG seizure detection, particularly the montage-dependency problem and multi-scale temporal modeling.

**Proposed Stack**: GNN → TCN → ConvNeXt → Bi-Mamba

---

## Current Architecture (v2)

```
EEG (19ch, 256Hz) → U-Net → ResCNN → Bi-Mamba-2 → Detection Head
```

### Limitations
- **Montage-dependent**: Assumes fixed electrode relationships
- **U-Net overhead**: Complex skip connections for temporal data
- **ResNet aging**: Newer architectures outperform vanilla ResNets
- **No spatial reasoning**: Treats channels as independent rather than spatially related

---

## Proposed Experimental Stack (v3)

```
EEG Input (19ch, 256Hz, 60s windows)
         ↓
[GNN - Spatial]      → Dynamic electrode relationship learning
         ↓
[TCN - Temporal]     → Multi-scale temporal feature extraction
         ↓
[ConvNeXt - Local]   → State-of-the-art local pattern refinement
         ↓
[Bi-Mamba - Global]  → O(N) long-range sequence modeling
         ↓
[Detection Head]     → Per-timestep probabilities
```

---

## Component Rationale

### 1. Graph Neural Network (GNN) - Spatial Reasoning
**Problem Solved**: Montage dependency
- **Current issue**: Fixed electrode positions assume specific montage (referential/bipolar)
- **GNN solution**: Learn electrode relationships dynamically
- **Architecture**: GAT (Graph Attention Network) or GCN with learned adjacency matrix
- **Input**: 19 nodes (electrodes) with position encoding
- **Output**: Spatially-aware feature representation

### 2. Temporal Convolutional Network (TCN) - Multi-Scale Features
**Problem Solved**: Cleaner multi-scale extraction than U-Net
- **Current issue**: U-Net designed for 2D images, not 1D temporal data
- **TCN solution**: Dilated convolutions with exponential receptive field growth
- **Architecture**:
  - Layer 1: 1ms patterns (dilation=1)
  - Layer 2: 10ms patterns (dilation=10)
  - Layer 3: 100ms patterns (dilation=100)
  - Layer 4: 1s patterns (dilation=1000)
- **Advantages**: Simpler than U-Net, native temporal design

### 3. ConvNeXt - Local Pattern Enhancement
**Problem Solved**: Outdated ResNet blocks
- **Current issue**: ResNet from 2015, surpassed by modern designs
- **ConvNeXt solution**:
  - Larger kernels (7×1 or 9×1) for longer temporal patterns
  - Fewer activation functions
  - Layer normalization > Batch normalization
  - Inverted bottleneck design
- **Performance**: Matches Vision Transformers with pure convolution

### 4. Bi-Mamba-2 - Long-Range Dependencies
**Keep from v2**: Already optimal for this role
- **Maintains**: O(N) complexity advantage
- **Enhancement**: Now operates on richer, spatially-aware features

---

## Implementation Phases

### Phase 1: Baseline Ablation (Months 1-2)
- [ ] Benchmark current U-Net + ResCNN + Bi-Mamba on TUSZ
- [ ] Document baseline metrics (TAES at 10/5/1 FA/24h)
- [ ] Profile computational requirements

### Phase 2: TCN Replacement (Months 2-3)
- [ ] Replace U-Net with TCN
- [ ] Architecture: 6-layer TCN with exponential dilation
- [ ] Compare: Parameters, FLOPs, performance
- [ ] Expected: Simpler model, comparable or better performance

### Phase 3: ConvNeXt Integration (Months 3-4)
- [ ] Replace ResCNN with ConvNeXt blocks
- [ ] Tune: Kernel sizes (7×1, 9×1, 11×1)
- [ ] Compare: Local pattern detection quality
- [ ] Expected: 5-10% performance improvement

### Phase 4: GNN Spatial Reasoning (Months 4-6)
- [ ] Add GNN as first component
- [ ] Architecture options:
  - GAT with 2-hop attention
  - GCN with learnable adjacency
  - GraphSAGE with electrode embedding
- [ ] Test montage generalization:
  - Train on referential, test on bipolar
  - Train on 19ch, test on 10-20 system variations
- [ ] Expected: Major improvement in cross-dataset transfer

### Phase 5: Full Stack Optimization (Months 6-7)
- [ ] Joint training of all components
- [ ] Hyperparameter search
- [ ] Multi-objective optimization (sensitivity vs FA rate)

---

## Evaluation Strategy

### Datasets
1. **Primary**: TUSZ v2.0.3 (train/dev/eval splits)
2. **Transfer**: CHB-MIT (pediatric, different montage)
3. **Stress test**: Siena Scalp EEG (different equipment)

### Metrics
- **Primary**: NEDC TAES scoring (proper clinical evaluation)
- **Sensitivity targets**: >75% at 1 FA/24h, >90% at 5 FA/24h
- **Montage robustness**: Performance delta between montages <10%
- **Computational**: Must maintain real-time capability (<100ms per 60s window)

### Ablation Studies
1. **Component contribution**: Remove each component, measure delta
2. **Order sensitivity**: Try different component orders
3. **Parallel vs Sequential**: Test parallel branches
4. **Depth tuning**: Optimal layers per component

---

## Expected Challenges

### Technical
- **Training complexity**: 4 components = harder optimization
- **Memory requirements**: GNN adds spatial dimension
- **Hyperparameter explosion**: Each component has its own settings
- **Gradient flow**: Ensuring gradients reach all components

### Scientific
- **Overfitting risk**: More parameters with same data
- **Interpretability**: GNN attention weights need validation
- **Generalization**: Does montage-agnostic really work?

### Practical
- **Compute requirements**: Full grid search infeasible
- **Implementation time**: 6-7 month timeline
- **Baseline comparison**: Need strong v2 baseline first

---

## Success Criteria

### Minimum Viable Improvement
- 10% reduction in FA rate at same sensitivity
- OR 10% sensitivity improvement at same FA rate
- Successful transfer to different montage without retraining

### Stretch Goals
- Sub-1 FA/24h with >80% sensitivity
- Real-time inference on CPU
- Interpretable electrode importance maps via GNN attention

---

## Alternative Architectures to Consider

### Hybrid Approaches
```
Option A: Parallel Processing
EEG → [GNN, TCN] → Concat → ConvNeXt → Bi-Mamba

Option B: Multi-Head
EEG → GNN → [TCN, ConvNeXt, Attention] → Fusion → Bi-Mamba

Option C: Hierarchical
EEG → GNN → TCN → [ConvNeXt + Bi-Mamba parallel] → Fusion
```

### Simpler Alternatives (if full stack fails)
1. Just TCN + Bi-Mamba (drop GNN and ConvNeXt)
2. GNN + existing U-Net + Bi-Mamba (only add spatial)
3. Current stack + GNN postprocessing (minimal change)

---

## Key Innovations

1. **First GNN + SSM combination for EEG** (novel to literature)
2. **Montage-agnostic by design** (solves fundamental problem)
3. **Principled multi-scale architecture** (each component has clear role)
4. **O(N) complexity maintained** (scalable to continuous monitoring)

---

## Next Steps

1. [ ] Finish v2 benchmarking (current architecture)
2. [ ] Literature review on GNNs for EEG (identify best practices)
3. [ ] Implement TCN baseline (simplest component to start)
4. [ ] Collaborate/discuss with signal processing experts
5. [ ] Secure compute resources for extensive ablations

---

## References

- TCN: "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling" (2018)
- ConvNeXt: "A ConvNet for the 2020s" (2022)
- GNN for EEG: "EEG-GCNN: Augmenting Electroencephalogram-based Neurological Disease Diagnosis using a Domain-guided Graph Convolutional Neural Network" (2020)
- Mamba: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2024)

---

*Note: This is a research roadmap for experimental architecture. Current v2 (U-Net + ResCNN + Bi-Mamba) remains the primary focus until benchmarked.*

**Last Updated**: 2025-09-21
**Status**: Proposed - Pending v2 Baseline Completion