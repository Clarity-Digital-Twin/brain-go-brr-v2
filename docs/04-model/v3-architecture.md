# V3 Architecture (Single Source of Truth)

**Last Updated**: October 12, 2025
**Current Version**: v4.0.0 (dual-stream architecture with BiMamba2 baseline + Flash Linear Attention variant)
**Status**: VALIDATED - Production training on Modal A100 (BiMamba2) and RTX 4090 (FLA on ext4 cache)

## Version History

- **v3.3.0** (Sept 27, 2025): PR1-5 architectural fixes
  - PR-1: Boundary normalization layers
  - PR-2: Bounded edge stream (tanh activation, LayerNorm)
  - PR-3: Adjacency conditioning (row-softmax, EMA, symmetry)
  - PR-4: Gated fusion (replaced simple addition)
  - PR-5: Edge similarity margin (safety from ±1 boundaries)

- **v3.3.1** (Sept 30, 2025): **CRITICAL FIX** - Eigendecomposition gradient explosion
  - Detached eigenvectors in gnn_pyg.py:205
  - Prevents gradient explosion through eigendecomposition
  - Zero architectural compromise (2025 best practice)

- **v3.4.0** (Sept 30, 2025): Pre-norm Mamba
  - Aligned with reference Mamba2 implementation
  - Improved scale consistency in residual paths

- **v3.4.1** (Oct 1, 2025): Optional warmup schedules
  - Adjacency temperature warmup (2.0 → 1.0)
  - Focal gamma warmup (1.0 → 2.0)
  - Residual scale warmup (optional)

## Data Flow

- Input `(B,19,15360)` → TCN `(B,512,960)` → Electrode features `(B,19,960,64)`
- **Node stream**: BiMamba2 over `(B*19,64,960)` → `(B,19,960,64)` (or BiGatedDeltaNet when `temporal_type: gated_deltanet`)
- **Edge stream**: Cosine similarity with margin `(B,171,960,1)` → Edge SSM (BiMamba2 1→16→1 + Softplus, or BiGatedDeltaNet with `edge_mamba_d_model: 32`) → weights `(B,171,960)`
- **Adjacency assembly**: `(B,960,19,19)` with top‑k=3, threshold, symmetry, identity fallback
- **GNN processing**: Vectorized SSGConv×2 + Laplacian PE over all timesteps → `(B,19,960,64)`
- **Fusion**: Multi-head gated fusion (node + GNN) → `(B,19,960,64)`
- **Decoder**: Back‑project to `(B,512,960)` → ProjectionHead to `(B,19,15360)` → Conv1d(19→1) logits `(B,15360)`

## Optional Enhancement

- Lightweight time–frequency hybrid: add a 3‑band STFT side‑branch and fuse before `proj_to_electrodes`. See `docs_v2/04-model/time-frequency-hybrid.md`.

## Key Parameters

### Node Stream (Per-Electrode Processing)
- **d_model**: 64 (per-electrode feature dimension)
- **n_layers**: 6
- **d_state**: 16
- **d_conv**: 4 (CUDA constraint: 2-4)
- **expand**: 2
- **headdim**: 8
- **Pattern**: Pre-norm BiMamba2 (v3.4.0+)

### Edge Stream (Pairwise Relationships)
- **d_model**: 16 (learned lift from 1→16→1)
- **n_layers**: 2
- **d_state**: 8
- **d_conv**: 4
- **expand**: 2
- **headdim**: 4
- **Activation**: Softplus (positive edge weights)
- **Similarity**: Cosine with margin=0.01 (v3.3.0, PR-5)

### GNN (Graph Neural Network)
- **Layers**: 2× SSGConv
- **Alpha**: 0.05 (spectral mixing parameter)
- **Laplacian PE**: k=16 eigenvectors
- **Dynamic PE**: **ALWAYS ENABLED** with safeguards (v3.3.1+)
  - Eigenvectors detached (gnn_pyg.py:205) - prevents gradient explosion
  - Semi-dynamic interval: 5 (update every 5 timesteps)
  - PE sign consistency: enabled
- **Adjacency**: Top-k=3, threshold pruning, forced symmetry, identity fallback

### TCN (Temporal Convolutional Network)
- **Layers**: 8
- **Channels**: [64, 128, 256, 512]
- **Kernel**: 7
- **Stride**: 16 (15360 → 960 downsampling)

## Stability Features (v3.3.0 - v3.4.1)

### PR-1: Boundary Normalization (v3.3.0)
**Purpose**: Prevent unbounded information flows between components

```yaml
model:
  norms:
    boundary_norm: layernorm
    after_tcn_proj: true       # After TCN → electrode projection
    after_node_mamba: true     # After node Mamba processing
    after_edge_mamba: true     # After edge Mamba processing
    after_gnn: true            # After GNN spatial processing
    before_decoder: true       # Before final decoder
    layerscale_alpha: 0.1      # Conservative residual scaling
```

### PR-2: Bounded Edge Stream (v3.3.0)
**Purpose**: Prevent edge feature explosions

```yaml
model:
  graph:
    edge_lift_activation: tanh       # Bound to [-1, 1]
    edge_lift_norm: layernorm        # Normalize after lift
    edge_lift_init_gain: 0.1         # Conservative initialization
```

### PR-3: Adjacency Conditioning (v3.3.0)
**Purpose**: Stable, well-conditioned adjacency matrices

```yaml
model:
  graph:
    adj_row_softmax: true            # Row-wise normalization
    adj_softmax_tau: 1.0             # Temperature (warmup in v3.4.1)
    adj_ema_beta: 0.95               # Temporal smoothing
    adj_force_symmetric: true        # Symmetric Laplacian
    laplacian_eps: 1.0e-3            # Regularization for eigendecomp
```

### PR-4: Gated Fusion (v3.3.0)
**Purpose**: Learn optimal node/GNN combination

```yaml
model:
  graph:
    fusion_type: gated               # Multi-head gated fusion
    # Replaces simple addition (node + gnn)
```

### PR-5: Edge Similarity Margin (v3.3.0)
**Purpose**: Prevent ±1.0 boundary explosions in cosine similarity

```yaml
model:
  graph:
    edge_similarity_margin: 0.01     # Safety margin from boundaries
```

### v3.3.1: Eigendecomposition Fix (Sept 30, 2025)
**CRITICAL**: Prevents gradient explosion through eigendecomposition

**File**: `src/brain_brr/models/gnn_pyg.py:205`
```python
eigenvectors = eigenvectors.detach()  # NO gradients through eigendecomp
```

**Why This Works**:
- Eigenvectors are FIXED positional coordinates (like Transformer sinusoidal PE)
- Learning happens in GNN layers that PROCESS PE, not in PE itself
- Prevents `1/(λᵢ - λⱼ)` gradient explosion when eigenvalues are close
- Zero architectural compromise - this is 2025 best practice

### v3.4.1: Warmup Schedules (Oct 1, 2025)
**Optional**: Gradual gradient stabilization

```yaml
training:
  warmup_schedule:
    enabled: true
    warmup_steps: 1000
    adj_temperature_enabled: true    # Adjacency τ: 2.0 → 1.0
    adj_temperature_start: 2.0
    adj_temperature_end: 1.0
    focal_gamma_enabled: true        # Focal γ: 1.0 → 2.0
    focal_gamma_start: 1.0
    focal_gamma_end: 2.0
    residual_scale_enabled: false    # Optional
```

## Constraints and Guards

- **CUDA alignment**: `(d_model*expand)/headdim` must be integer and multiple of 8
  - Node: (64*2)/8 = 16 ✅
  - Edge: (16*2)/4 = 8 ✅
- **FLA edge alignment**: When `temporal_type: gated_deltanet`, set `edge_mamba_d_model: 32`, `edge_mamba_heads: 8` so `(32*2)/8 = 8` stays aligned with Triton kernels.
- **Vectorized GNN**: Processes all timesteps simultaneously for efficiency
- **Dynamic PE**: ALWAYS enabled (v3.3.1+ with detached eigenvectors)
- **Edge transform bypass**: `bypass_edge_transform=True` (edge weights already Softplus'ed upstream)
- **Identity fallback**: Prevents disconnected nodes in adjacency matrix

## Code Locations

| Component | File | Key Lines |
|-----------|------|-----------|
| **V3 Detector** | `src/brain_brr/models/detector.py` | Full architecture |
| **Edge features** | `src/brain_brr/models/edge_features.py` | Similarity + margin |
| **Adjacency assembly** | `src/brain_brr/models/adjacency.py` | Top-k, symmetry, conditioning |
| **GNN + Laplacian PE** | `src/brain_brr/models/gnn_pyg.py` | Line 205: eigenvector detach |
| **BiMamba2** | `src/brain_brr/models/mamba.py` | Pre-norm pattern |
| **BiGatedDeltaNet (FLA)** | `src/brain_brr/models/gated_deltanet.py` | Flash Linear Attention wrapper |
| **TCN encoder** | `src/brain_brr/models/tcn.py` | Downsampling + features |
| **Gated fusion** | `src/brain_brr/models/fusion.py` | Multi-head gating |
| **Norms** | `src/brain_brr/models/norms.py` | LayerNorm + LayerScale |

## Dual SSM Streams (v4.0.0)

- **Baseline**: `bimamba2` for both streams (production on Modal A100).
- **Variant**: `gated_deltanet` (Flash Linear Attention) for node and/or edge streams. Toggle per stream in `ModelConfig.mamba.temporal_type` and the edge builder config.
- Both options share the same TCN, GNN, fusion, and decoder components—only the temporal SSM modules swap.
- Parameter counts: Node stream drops ~29% with BiGatedDeltaNet (398k → 284k), edge stream rises ~190% (10k → 30k) to satisfy FLA alignment; total stream parameters decrease ~23%.
- Validation ladder: Phase 0 infrastructure → Phase 1a edge-only → Phase 1b node-only → Phase 2 full stack. All phases complete; local FLA training confirmed stable when cache resides on ext4 (`docs/08-operations/wsl2-sigbus-fix.md`).

## Validated Design Decisions

### Dynamic Laplacian PE (ALWAYS ENABLED)
- **Status**: Default ON with safeguards (v3.3.1+)
- **Critical fix**: Eigenvectors detached (gnn_pyg.py:205)
- **Fallback**: NOT needed - architecture is stable on all hardware
- **Evidence**: Batch 723+ training, zero NaN/Inf

### Graph Sparsity (Top-k=3)
- **Validated by**: EvoBrain reference architecture
- **Rationale**: "Top-3 neighbors kept" sufficient for EEG electrode graphs
- **Safety**: Threshold pruning + symmetry + identity fallback

### Temporal → Spatial Order
- **Pattern**: TCN → Mamba → GNN (time-then-graph)
- **Validated by**: Literature + empirical results
- **Implementation**: Vectorized over all timesteps for efficiency

### Node Stream Capacity
- **d_model**: 64 per-electrode (1216 total: 19×64)
- **Status**: Sufficient - validated in training
- **Future**: Can ablate 128 if needed

### Bidirectional Processing
- **Pattern**: BiMamba2 for 60s windows (offline processing)
- **Causal variant**: Available for streaming (future work)

## Training Validation (Oct 1, 2025)

**Platform**: RTX 4090, batch 723+
**Config**: `configs/local/train_bimamba.yaml`

| Metric | Initial | Batch 723 | Change |
|--------|---------|-----------|--------|
| **Loss** | 0.3050 | 0.1555 | -49% ↓ |
| **P95 Gradient** | 52.06 | 9.74 | -82% ↓ |
| **NaN/Inf** | 0 | 0 | ✅ STABLE |

**Conclusion**: Architecture is **ROCK SOLID** with v3.3.1+ fixes.

## References

- **Stability evolution**: `docs_v2/04-model/v3-stability-evolution.md`
- **Warmup schedules**: `docs_v2/05-training/warmup-schedules.md`
- **Gradient monitoring**: `docs_v2/08-operations/gradient-monitoring.md`
- **NaN prevention**: `docs_v2/08-operations/nan-prevention-complete.md`
- **Laplacian PE details**: `docs_v2/04-model/laplacian-pe.md`
- **GNN implementation**: `docs_v2/04-model/gnn.md`
- **Edge features**: `docs_v2/04-model/edge-features-and-adjacency.md`
