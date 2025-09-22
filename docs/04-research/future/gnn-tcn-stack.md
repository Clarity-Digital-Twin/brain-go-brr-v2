# 🚀 Future Roadmap: Next-Generation EEG Architecture Stack

## Executive Summary

Moving beyond the current U-Net + ResCNN + Bi-Mamba-2 architecture to address fundamental limitations in EEG seizure detection, particularly the montage-dependency problem and multi-scale temporal modeling.

**Proposed Stack v3.0 (EvoBrain‑aligned)**: Mamba (time) → Dynamic GNN (graph) → ConvNeXt (local) → optional TCN (multi‑scale)
**NEW: EvoBrain Validation** (Sep 2025): Time‑then‑graph with explicit dynamic graphs is superior. 🎯

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

### UPDATED AFTER EVOBRAIN PAPER (2025‑09‑22)

```
EEG (19ch, 256 Hz, 60 s)
  ↓
STFT (optional) → frequency features (helps seizures per EvoBrain)
  ↓
Bi‑Mamba‑2 (time encoder) → per‑channel temporal embeddings (time‑first)
  ↓
Dynamic GCN + Laplacian PE (graph encoder) → explicit time‑varying adjacency A_t
  ↓
ConvNeXt (local refinement) → longer kernels for morphology
  ↓
Detection head → per‑timestep probabilities
```

---

## Component Rationale

### 1. Bi‑Mamba‑2 (time‑first) — Temporal Encoder
**Problem Solved**: Long‑range temporal context with O(N) complexity
- Fits EvoBrain’s “time‑then‑graph” result (temporal first, graph second)
- Our codebase already uses Bi‑Mamba‑2; reuse as the time encoder
- Optional: dual‑stream node/edge Mamba later (EvoBrain’s two‑stream idea)
- CUDA note: our CUDA path coerces `d_conv=4→4`; EvoBrain uses `d_conv=4`. Recommend `conv_kernel=4` for parity.

### 2. Dynamic Graph Neural Network (GNN) — Spatial Reasoning
**Problem Solved**: Montage dependency
- Fixed adjacency is insufficient; edges must evolve with state (explicit dynamics)
- Architecture:
  - GCN with Laplacian Positional Encoding (LPE; top‑K eigenvectors, e.g., K=8–16)
  - Time‑varying adjacency A_t built from Mamba features (learned similarity or corr)
  - Optional GAT variant for attention over electrodes
- Input: Per‑channel temporal embeddings from Bi‑Mamba‑2
- Output: Spatially aware features per time step

Adjacency strategies (from EvoBrain):
- Individual (correlation): cross‑correlation per time slice; keep top‑k (default k=3) neighbors; filter: dual_random_walk.
- Combined (distance): distance graph; filter: Laplacian.
- Dynamic: recompute per step; sparsify with top‑k and small threshold (|w|>1e‑4).

### 3. ConvNeXt — Local Pattern Enhancement
**Problem Solved**: Outdated ResNet blocks
- **Current issue**: ResNet from 2015, surpassed by modern designs
- **ConvNeXt solution**:
  - Larger kernels (7×1 or 9×1) for longer temporal patterns
  - Fewer activation functions
  - Layer normalization > Batch normalization
  - Inverted bottleneck design
- **Performance**: Matches Vision Transformers with pure convolution

### 4. Optional TCN — Multi‑Scale Temporal Add‑on
If extra multi‑scale temporal detail is needed post‑graph, add a lightweight TCN block.
Keep total latency within real‑time bounds (<100 ms per hop).

---

## Implementation Phases

### Phase 1: Baseline Ablation (Months 1-2)
- [ ] Benchmark current U-Net + ResCNN + Bi-Mamba on TUSZ
- [ ] Document baseline metrics (TAES at 10/5/1 FA/24h)
- [ ] Profile computational requirements

### Phase 2: Add Dynamic GNN after Bi‑Mamba (Months 2–3)
- [ ] Implement GraphChannelMixer (GCN+LPE) after Bi‑Mamba‑2
- [ ] Build adjacency A_t from temporal features (cosine or learned MLP)
- [ ] Gate by config; add unit tests for shapes and identity init
- [ ] Compare TAES + AUROC vs baseline
 - [ ] Start with top_k=3 and threshold=1e‑4 for edge pruning (EvoBrain defaults)

### Phase 3: ConvNeXt Integration (Months 3–4)
- [ ] Replace ResCNN with ConvNeXt blocks (7×1, 9×1, 11×1)
- [ ] Evaluate interaction with graph features

### Phase 4: Dual‑Stream Temporal (Months 4–6)
- [ ] Prototype node‑stream and edge‑stream Mamba (EvoBrain style)
- [ ] Edge stream supervises/builds A_t; node stream feeds GCN
- [ ] Test montage generalization and robustness

---

## Implementation Hooks (Codebase)
- Insert graph stage after temporal encoder:
  - File: `src/brain_brr/models/detector.py:104` (after `temporal = self.mamba(features)`)
  - Shape contract there is `(B, 512, 960)`; preserve channel order from `src/brain_brr/constants.py:12`.
- New module for graph ops (planned):
  - File: `src/brain_brr/models/gnn.py` (GraphChannelMixer: GCN+LPE)
  - Optional adjacency builder helper: `src/brain_brr/models/graph_builder.py`
- Config flags (planned):
  - Schema: `src/brain_brr/config/schemas.py` → `experiment.graph.enabled`, `graph.k_eigs`, `graph.similarity`.
- Note: PyTorch Geometric will be required for LPE/GCN variants; not yet in dependencies.

Param guidance (EvoBrain‑informed):
- Mamba: `d_state=16, d_conv=4, expand≈2`
- LPE: `k_eigs=16` (for 19 nodes; ensure K≤N‑1)
- GNN: 2 layers (e.g., SSGConv or GCNConv) with residual skip
- Sparsity: `top_k=3`, edge threshold `1e‑4`

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

- **EvoBrain**: "Dynamic Multi-channel EEG Graph Modeling for Time-evolving Brain Network" (2025) - VALIDATES OUR APPROACH!
- TCN: "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling" (2018)
- ConvNeXt: "A ConvNet for the 2020s" (2022)
- GNN for EEG: "EEG-GCNN: Augmenting Electroencephalogram-based Neurological Disease Diagnosis using a Domain-guided Graph Convolutional Neural Network" (2020)
- Mamba: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2024)

---

*Note: This is a research roadmap for experimental architecture. Current v2 (U-Net + ResCNN + Bi-Mamba) remains the primary focus until benchmarked.*

**Last Updated**: 2025-09-22
**Status**: VALIDATED BY EVOBRAIN PAPER! Architecture ordering confirmed!

---

## 🔥 EVOBRAIN PAPER INSIGHTS (Sep 22, 2025)

### Key Validations for Our v3 Design:

1. **TIME-THEN-GRAPH SUPERIORITY PROVEN**
   - Theorem 2 proves: graph-then-time ⊂ time-and-graph ⊂ time-then-graph
   - Our TCN→GNN ordering is mathematically optimal!

2. **TWO-STREAM MAMBA ARCHITECTURE**
   - EvoBrain uses dual Mamba: one for nodes, one for edges
   - Consider upgrading our Bi-Mamba to dual-stream in v3.1

3. **FREQUENCY DOMAIN INPUT**
   - STFT with log amplitudes captures seizure markers better
   - Add frequency transform as preprocessing step

4. **DYNAMIC GRAPH CONSTRUCTION**
   - Graphs evolve at each time snapshot
   - Top-k correlation-based edges
   - Solves montage dependency!

5. **PERFORMANCE GAINS**
   - 23% AUROC improvement
   - 30% F1 score improvement
   - Over dynamic GNN baselines

### What This Means for Brain-Go-Brr v3:

✅ Our intuition about TCN-first was RIGHT
✅ GNN for spatial modeling validated
✅ Mamba for long-range dependencies confirmed
⚡ Consider: STFT preprocessing, dual-stream Mamba, dynamic graphs

### EvoBrain vs Brain-Go-Brr v3 Comparison:

| Component | EvoBrain | Brain-Go-Brr v3 | Notes |
|-----------|----------|-----------------|-------|
| Temporal | Dual Mamba | TCN + Bi-Mamba | Both O(N), different approaches |
| Spatial | GCN + Laplacian PE | GNN (flexible) | We can test multiple GNN types |
| Input | STFT frequencies | Raw EEG | Add STFT option |
| Graphs | Dynamic per snapshot | Static (planned dynamic) | Implement dynamic in v3.1 |
| Local patterns | None | ConvNeXt | Our addition for fine details |

**BOTTOM LINE**: EvoBrain validates our core ideas while suggesting enhancements!
