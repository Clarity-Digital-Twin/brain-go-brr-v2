# 🎯 CANONICAL ARCHITECTURE ROADMAP (Updated — v3 implemented)

## ⚠️ UPDATE: WE'RE ALREADY AT v3.0! (TCN IS LIVE!)

## WHERE WE ARE NOW
v3 dual‑stream is implemented and selectable via `model.architecture: v3`:
```
EEG (19ch, 256Hz)
 → TCN Encoder
 → Bi‑Mamba‑2 (node)  +  Bi‑Mamba‑2 (edge: learned lift 1→D→1, D=16)
 → Vectorized PyG GNN (SSGConv, α=0.05) + Static Laplacian PE (k=16)
 → Projection / Upsample → Detection Head
```

Baseline (`model.architecture: tcn`) remains available without the learned adjacency path.

## Prior state (v2.3)
```
EEG (19ch, 256Hz) → TCN Encoder → Bi-Mamba-2 → TCN Decoder → Detection Head
```
**Status**: TCN ALREADY REPLACED U-Net + ResCNN! Training on Modal NOW!
**Architecture**: Using `architecture: tcn` in all Modal configs
**Old Path**: U-Net + ResCNN still in codebase but NOT USED at runtime

---

## INCREMENTAL UPGRADES (Ship & Train Each)

### v2.1 - Multi-Task Artifact Suppression ⚡
```
EEG → U-Net → ResCNN → Bi-Mamba-2 → [Seizure Head + Artifact Head]
```
**Changes**: Add artifact detection head (TUAR dataset)
**Why**: Reduce false alarms from muscle/eye/electrode artifacts
**Expected**: 30-50% FA reduction (proven in literature)
**Timeline**: 1 week
**Training**: Co-train on TUSZ + TUAR

### v2.2 - Frequency Domain Input 📊
```
EEG → STFT → U-Net → ResCNN → Bi-Mamba-2 → Detection Head
```
**Changes**: Add Short-Time Fourier Transform preprocessing
**Why**: EvoBrain paper shows frequency domain captures seizure markers better
**Expected**: 5-10% sensitivity improvement
**Timeline**: 3 days
**Training**: Same TUSZ, different input representation

### ~~v2.5~~ SKIP STRAIGHT TO v2.6! 🚀

### v2.6 - Dynamic GNN + Laplacian PE 🧠🔥 (Implemented in v3)
```
EEG → TCN Encoder → Bi-Mamba-2 → [Dynamic GNN + LPE] → TCN Decoder → Detection Head
                                          ↑
                                  INSERT GNN HERE!
```
**Changes**: Add dynamic graph with time-evolving adjacency + Laplacian Positional Encoding
**Status**: Implemented (see v3).
**Implementation Details** (from EvoBrain):
- PyG's `AddLaplacianEigenvectorPE(k=16)` for positional awareness
- Dynamic adjacency per timestep: `adj[timestep, batch, nodes, nodes]`
- SSGConv with alpha=0.05 (proven best for EEG)
- Edge transform: Linear(512, 1) → Softplus() for positive weights
- Edge pruning: keep only |weight| > 0.0001 + top_k=3
- Concatenate PE to node features before GNN
**Why**: EvoBrain MATHEMATICALLY PROVES dynamic > static; skip static entirely!
**Expected**: 15-20% improvement (combines what would be v2.5 + v2.6)
**Timeline**: 1 week
**Training**: Graph structure evolves per snapshot capturing seizure dynamics

---

## MAJOR REPLACEMENTS

### ✅ v3.0 — TCN replaced U‑Net + ResCNN
```
EEG → TCN Encoder → Bi-Mamba-2 → TCN Decoder → Detection Head
```
**Changes**: TCN replaced BOTH U-Net AND ResCNN
**Status**: COMPLETE - This is what's training on Modal RIGHT NOW
**Note**: ResCNN was removed from the path entirely (TCN handles local patterns)

---

## Implementation Hooks
- Wire graph stage after Mamba:
  - `src/brain_brr/models/detector.py:104`
- New GNN module (planned):
  - `src/brain_brr/models/gnn.py` (GraphChannelMixer)
- Channel ordering source of truth:
  - `src/brain_brr/constants.py:12`
- Dependency: PyTorch Geometric required for LPE/GCN (to be added later)

### ~~v3.1 - Replace ResCNN with ConvNeXt~~ NOT APPLICABLE
```
ResCNN already removed when we switched to TCN!
```
**Status**: SKIPPED - TCN handles both encoding AND local patterns
**Note**: ConvNeXt would only be relevant for the OLD U-Net path
**Alternative**: Could add ConvNeXt as optional local refiner AFTER GNN if needed

### ✅ v3.2 — Dual‑Stream Mamba 🌊 (Node+Edge streams)
```
EEG → TCN → ConvNeXt → [Node-Mamba + Edge-Mamba] → GNN → Detection Head
                                ↑
                          DUAL STREAM
```
**Changes**: Separate Mamba for nodes and edges (EvoBrain style)
**Implementation Details** (from EvoBrain):
- Node-Mamba: `Mamba(d_model=512, d_state=16, d_conv=4, expand=2)`
- Edge-Mamba: Same config, processes edge features
- Both outputs fed to GNN
**Why**: Model channel features and relationships separately
**Status**: Implemented in v3 (edge lift 1→D→1, D=16; vectorized GNN; static PE).

---

## NEXT: Test hardening and ablations
- Unit/integration tests: vectorized vs loop equivalence; disjoint‑batch node‑id offsets; edge lift grad‑flow.
- Ablations: edge features (cosine/correlation/coherence), top‑k and threshold sweeps.
- Performance markers: forward latency/memory on GPU.

---

## EXPERIMENTAL (High Risk/Reward)

### v4.0 - Full EvoBrain-Inspired Stack
```
EEG → STFT → [TCN-Nodes + TCN-Edges] → GCN+LPE → Detection Head
```
**Changes**: Full time-then-graph with dual temporal streams
**Why**: Mathematically proven optimal expressivity
**Expected**: 20-30% total improvement
**Timeline**: 1 month
**Training**: Complete architecture overhaul

---

## TRAINING STRATEGY PER VERSION

1. **Train baseline (v2.0)** → Get TAES metrics
2. **Add one change** → Train → Compare metrics
3. **If better**: Keep and build on it
4. **If worse**: Revert and try next idea
5. **Document everything**: Params, FLOPs, FA/24h, sensitivity

## SUCCESS METRICS (Track for Each Version)

| Version | FA/24h@90% | Params | GFLOPs | Modal $/epoch | Notes |
|---------|------------|--------|--------|---------------|-------|
| v2.0    | BASELINE   | 47M    | TBD    | $0.80         | Current |
| v2.1    | Target -30%| +2M    | +0.1   | $0.85         | Artifact head |
| v2.2    | Target -5% | Same   | +0.5   | $0.85         | STFT input |
| v2.6    | Target -20%| +7M    | +3.0   | $0.95         | FULL Dynamic GNN+LPE |
| v3.0    | Target same| -10M   | -5.0   | $0.70         | TCN simpler |
| v3.1    | Target -5% | +2M    | +1.0   | $0.75         | ConvNeXt |
| v3.2    | Target -10%| +15M   | +3.0   | $0.95         | Dual Mamba |

---

## IMMEDIATE NEXT STEPS (THIS WEEK)

1. ✅ **Let v2.0 finish training** (baseline metrics)
2. 🔧 **Implement v2.1** (artifact head) - HIGHEST ROI
3. 📊 **Test v2.2** (STFT) - Easy win
4. 🧠🔥 **FULL SEND v2.6** (Dynamic GNN + LPE) - GO BIG OR GO HOME!

---

## KEY INSIGHTS FROM EVOBRAIN (Sep 2025)

✅ **Time-then-graph > Graph-then-time** (mathematically proven)
✅ **Dynamic graphs > Static graphs** (explicit modeling wins)
✅ **Frequency domain helps** (STFT preprocessing)
✅ **Laplacian Positional Encoding** (k=16 eigenvectors optimal)
✅ **Dual-stream temporal** (separate node/edge Mamba streams)
✅ **SSGConv performs best** for EEG graphs
✅ **Our Mamba (time) → GNN (graph) ordering is CORRECT!**

### Concrete Implementation Params from EvoBrain:
- Mamba: `d_state=16, d_conv=4, expand=2`
- GNN: 2-layer SSGConv with skip connections
- LPE: 16 eigenvectors
- Activation: Tanh for GNN, Softplus for edge weights

---

## DECISION TREE

```
v2.0 (current) working?
├─ YES → Add artifact head (v2.1)
│   ├─ Better? → Add STFT (v2.2)
│   │   ├─ Better? → FULL Dynamic GNN+LPE (v2.6)
│   │   │   ├─ Better? → Replace U-Net with TCN (v3.0)
│   │   │   └─ Worse? → Debug graph implementation
│   │   └─ Worse? → Skip STFT, try v2.6 directly
│   └─ Worse? → Debug why artifacts hurt
└─ NO → Fix v2.0 first!
```

---

**PHILOSOPHY**: Ship small, train often, measure everything. Each version should be a working system we can deploy.

**Last Updated**: 2025-09-22
**Next Review**: After v2.0 baseline metrics available
