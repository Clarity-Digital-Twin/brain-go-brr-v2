# 🚀 Future Roadmap: Next-Generation EEG Architecture Stack

## Executive Summary

Moving beyond the current U-Net + ResCNN + Bi-Mamba-2 architecture to address fundamental limitations in EEG seizure detection, particularly the montage-dependency problem and multi-scale temporal modeling.

**Proposed Stack (EvoBrain‑aligned)**: Bi‑Mamba (time) → Dynamic GNN (graph) → ConvNeXt → optional TCN

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
EEG (19ch, 256 Hz, 60 s)
  ↓
Bi‑Mamba‑2 (time encoder)
  ↓
Dynamic GCN + Laplacian PE (graph encoder; explicit A_t)
  ↓
ConvNeXt (local refinement)
  ↓
Optional TCN (extra multi‑scale)
  ↓
Detection Head
```

---

## Component Rationale

### 1. Dynamic Graph Neural Network (GNN) — Spatial Reasoning
**Problem Solved**: Montage dependency
- Fixed adjacency is insufficient; edges must evolve with state (explicit dynamics)
- Use GCN with Laplacian Positional Encoding (top‑K eigenvectors)
- Build time‑varying adjacency A_t from temporal features (cosine or learned)
- Input: per‑channel embeddings from Bi‑Mamba‑2
- Output: spatially aware features per time step

### 2. Bi‑Mamba‑2 — Temporal Encoder
Already present in v2; becomes the first stage in time‑then‑graph pipeline per EvoBrain.

### 3. ConvNeXt - Local Pattern Enhancement
**Problem Solved**: Outdated ResNet blocks
- **Current issue**: ResNet from 2015, surpassed by modern designs
- **ConvNeXt solution**:
  - Larger kernels (7×1 or 9×1) for longer temporal patterns
  - Fewer activation functions
  - Layer normalization > Batch normalization
  - Inverted bottleneck design
- **Performance**: Matches Vision Transformers with pure convolution

### 4. Optional TCN — Multi‑Scale Add‑on
Add only if multi‑scale detail is lacking after graph integration; keep latency budget in mind.

---

## Implementation Phases

### Phase 1: Baseline Ablation (Months 1-2)
- [ ] Benchmark current U-Net + ResCNN + Bi-Mamba on TUSZ
- [ ] Document baseline metrics (TAES at 10/5/1 FA/24h)
- [ ] Profile computational requirements

### Phase 2: Add Dynamic GNN after Bi‑Mamba (Months 2–3)
- [ ] Implement GraphChannelMixer (GCN+LPE) module gated by config
- [ ] Adjacency A_t from temporal features (cosine similarity / learned)
- [ ] Unit tests (shape, identity init) and integration tests

### Phase 3: ConvNeXt Integration (Months 3-4)
- [ ] Replace ResCNN with ConvNeXt blocks
- [ ] Tune: Kernel sizes (7×1, 9×1, 11×1)
- [ ] Compare: Local pattern detection quality
- [ ] Expected: 5-10% performance improvement

### Phase 4: Dual‑Stream Temporal (Months 4–6)
- [ ] Prototype node‑stream and edge‑stream Mamba (EvoBrain)
- [ ] Edge stream supervises/builds A_t; node stream feeds GCN
- [ ] Montage generalization experiments (referential ↔ bipolar)

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
