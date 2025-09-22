# 🎯 CANONICAL ARCHITECTURE ROADMAP

## WHERE WE ARE NOW (v2.0 - BASELINE)
```
EEG (19ch, 256Hz) → U-Net → ResCNN → Bi-Mamba-2 → Detection Head
```
**Status**: SHIPPED & TRAINING on Modal
**Performance**: TBD (currently training)
**Cost**: ~$0.80/epoch on Modal

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

### v2.5 - Add GNN for Spatial Reasoning 🧠
```
EEG → U-Net → ResCNN → Bi-Mamba-2 → GNN → Detection Head
                                          ↑
                                    ADD THIS!
```
**Changes**: Add Graph Neural Network after Mamba
**Status**: Planned (not implemented)
**Implementation Details** (from EvoBrain):
- Use SSGConv (Simple Spectral Graph) or GCNConv
- Edge weights from correlation matrix
- Skip connections around GNN
**Why**: Learn electrode relationships, montage-agnostic
**Expected**: 10-15% cross-dataset transfer improvement
**Timeline**: 1 week
**Training**: Same data, spatial awareness added

### v2.6 - Dynamic Graphs + Laplacian PE 🔄
```
EEG → U-Net → ResCNN → Bi-Mamba-2 → [LPE + Dynamic GNN] → Detection Head
```
**Changes**: GNN with time-evolving adjacency + Laplacian Positional Encoding
**Status**: Planned (not implemented)
**Implementation Details** (from EvoBrain):
- PyG's `AddLaplacianEigenvectorPE(k=16)`
- Adjacency matrix per timestep: `adj[timestep, batch, nodes, nodes]`
- Concatenate PE to node features before GNN
- Edge pruning: keep only |weight| > 0.0001
**Why**: EvoBrain proves dynamic > static graphs; LPE stabilizes
**Expected**: Additional 5-10% improvement
**Timeline**: 1 week
**Training**: Graph structure evolves per snapshot

---

## MAJOR REPLACEMENTS (One at a Time)

### v3.0 - Replace U-Net with TCN 🚀
```
EEG → TCN → ResCNN → Bi-Mamba-2 → GNN → Detection Head
       ↑
  REPLACE U-Net
```
**Changes**: Temporal Convolutional Network replaces U-Net
**Why**: Simpler, native 1D design, proven superior for sequences
**Expected**: Faster training, comparable performance
**Timeline**: 2 weeks
**Training**: Full retrain required

---

## Implementation Hooks
- Wire graph stage after Mamba:
  - `src/brain_brr/models/detector.py:104`
- New GNN module (planned):
  - `src/brain_brr/models/gnn.py` (GraphChannelMixer)
- Channel ordering source of truth:
  - `src/brain_brr/constants.py:12`
- Dependency: PyTorch Geometric required for LPE/GCN (to be added later)

### v3.1 - Replace ResCNN with ConvNeXt 🎯
```
EEG → TCN → ConvNeXt → Bi-Mamba-2 → GNN → Detection Head
              ↑
         REPLACE ResCNN
```
**Changes**: Modern ConvNeXt blocks
**Why**: 2022 architecture > 2015 ResNet
**Expected**: 5-10% performance gain
**Timeline**: 1 week
**Training**: Retrain from v3.0

### v3.2 - Dual-Stream Mamba 🌊
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
**Expected**: 10-15% improvement
**Timeline**: 2 weeks
**Training**: More complex but powerful

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
| v2.5    | Target -10%| +5M    | +2.0   | $0.90         | Add GNN |
| v3.0    | Target same| -10M   | -5.0   | $0.70         | TCN simpler |
| v3.1    | Target -5% | +2M    | +1.0   | $0.75         | ConvNeXt |
| v3.2    | Target -10%| +15M   | +3.0   | $0.95         | Dual Mamba |

---

## IMMEDIATE NEXT STEPS (THIS WEEK)

1. ✅ **Let v2.0 finish training** (baseline metrics)
2. 🔧 **Implement v2.1** (artifact head) - HIGHEST ROI
3. 📊 **Test v2.2** (STFT) - Easy win
4. 🧠 **Prototype v2.5** (add GNN) - Validated by EvoBrain

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
│   │   ├─ Better? → Add GNN (v2.5)
│   │   │   ├─ Better? → Replace U-Net with TCN (v3.0)
│   │   │   └─ Worse? → Try dynamic GNN (v2.6)
│   │   └─ Worse? → Skip to GNN (v2.5)
│   └─ Worse? → Debug why artifacts hurt
└─ NO → Fix v2.0 first!
```

---

**PHILOSOPHY**: Ship small, train often, measure everything. Each version should be a working system we can deploy.

**Last Updated**: 2025-09-22
**Next Review**: After v2.0 baseline metrics available
