# Configuration Consistency Check (v4.0.0)

The config namespace now has **dedicated files per architecture**:

```
configs/local/
  smoke_bimamba.yaml   # BiMamba2 stack – smoke (3 files, 1 epoch)
  train_bimamba.yaml   # BiMamba2 stack – full training (100 epochs)
  smoke_fla.yaml       # FLA (Gated DeltaNet) stack – smoke
  train_fla.yaml       # FLA (Gated DeltaNet) stack – full training

configs/modal/
  smoke_bimamba.yaml   # BiMamba2 stack – smoke (50 files, 1 epoch)
  train_bimamba.yaml   # BiMamba2 stack – full training (100 epochs)
  smoke_fla.yaml       # FLA stack – smoke
  train_fla.yaml       # FLA stack – full training
```

Each pair (BiMamba2 vs FLA) is identical except for the **temporal blocks** and the
associated safeguards (`temporal_type*`, `gdn_*`, `edge_mamba_d_model`).

## Local Configs (RTX 4090 - 24GB)

### BiMamba2 – `train_bimamba.yaml`
```yaml
batch_size: 8                  # OPTIMIZED: 2× faster than batch=4
use_dynamic_pe: true          # ENABLED
semi_dynamic_interval: 5      # PE every 19.5ms (192 eigendecomps)
num_workers: 0                # WSL2 fix
mixed_precision: false        # RTX 4090 stability
learning_rate: 1.0e-4         # Conservative for stability
gradient_clip: 0.5            # Increased from 0.1 (eigendecomp fix allows this)
mid_checkpoint_interval_s: 1800  # Save every 30 min
mid_epoch_keep: 3             # Keep last 3 mid-epoch snapshots
```
**Status**: ✅ PRODUCTION STABLE

### FLA – `train_fla.yaml`
```yaml
batch_size: 8
use_dynamic_pe: true
semi_dynamic_interval: 5
temporal_type: gated_deltanet          # Applied to both node and edge streams
gdn_edge_num_heads: 3                  # 3 × 8 = 24 = 0.75 × edge_d_model
gdn_edge_headdim: 8
edge_mamba_d_model: 32                 # FLA causal_conv1d requirement
mixed_precision: false
learning_rate: 1.0e-4
gradient_clip: 0.5
mid_checkpoint_interval_s: 1800
mid_epoch_keep: 3
```
**Status**: ✅ READY FOR RESEARCH (mirrors BiMamba2 config except for temporal stack)

### BiMamba2 – `smoke_bimamba.yaml`
```yaml
batch_size: 8                  # Same as train for consistency
use_dynamic_pe: true          # ENABLED
semi_dynamic_interval: 5      # Same as train
num_workers: 0                # WSL2 fix
mixed_precision: false        # RTX 4090 stability
epochs: 1                     # Quick validation
# No mid-epoch checkpointing (1 epoch only)
```
**Status**: ✅ CORRECT

### FLA – `smoke_fla.yaml`
Same as BiMamba2 smoke config plus the FLA temporal override (`temporal_type: gated_deltanet`,
`gdn_*`, `edge_mamba_d_model: 32`). No mid-epoch checkpoints (1 epoch only).

## Modal Configs (A100 - 80GB)

### BiMamba2 – `train_bimamba.yaml`
```yaml
batch_size: 48                 # PRODUCTION: ~58GB peak (verified stable)
use_dynamic_pe: true          # ENABLED
semi_dynamic_interval: 5      # OPTIMAL: 192 eigendecomps (matches local)
num_workers: 4                # SAFE: 8 caused overhead
mixed_precision: true         # A100 tensor cores (3.8× faster)
learning_rate: 8.0e-5         # Batch-size scaled
gradient_clip: 0.5            # NaN protection
mid_checkpoint_interval_s: 1800  # Save every 30 min (CRITICAL for 6-7h epochs)
mid_epoch_keep: 3             # Keep last 3 mid-epoch snapshots
```
**Status**: ✅ PRODUCTION READY

### FLA – `train_fla.yaml`
```yaml
batch_size: 48
temporal_type: gated_deltanet          # Applied to both node and edge streams
gdn_edge_num_heads: 3
gdn_edge_headdim: 8
edge_mamba_d_model: 32
mixed_precision: true
learning_rate: 8.0e-5
gradient_clip: 0.5
mid_checkpoint_interval_s: 1800
mid_epoch_keep: 3
```
**Status**: ✅ READY FOR RESEARCH (mirrors BiMamba2 settings)

### BiMamba2 – `smoke_bimamba.yaml`
```yaml
batch_size: 48                # Same as train for consistency
use_dynamic_pe: true          # ENABLED
semi_dynamic_interval: 5      # Same as train
num_workers: 4                # Same as train
mixed_precision: true         # A100 tensor cores
epochs: 1                     # Quick validation
# No mid-epoch checkpointing (1 epoch only)
```
**Status**: ✅ VERIFIED

### FLA – `smoke_fla.yaml`
Identical to BiMamba2 smoke config with the temporal override (`temporal_type: gated_deltanet`,
`gdn_*`, `edge_mamba_d_model: 32`). One epoch, no mid-epoch checkpoints.

## Key Differences by Platform

| Setting | RTX 4090 | A100 | Reason |
|---------|----------|------|--------|
| **Batch Size (train)** | 8 | 48 | Memory: 24GB vs 80GB |
| **Batch Size (smoke)** | 8 | 48 | Consistency with train config |
| **Semi-dynamic Interval** | 5 | 5 | OPTIMAL: 192 eigendecomps (same on both) |
| **Mixed Precision** | false | true | RTX 4090 NaN issues |
| **Num Workers** | 0 | 4 | WSL2 vs cloud |
| **Learning Rate** | 1.0e-4 | 8.0e-5 | Stability vs batch-size scaling |
| **Mid-Epoch Checkpoints** | 1800s | 1800s | Crash recovery (both platforms) |
| **Temporal Stack (BiMamba2)** | implicit BiMamba2 | implicit BiMamba2 | Defaults |
| **Temporal Stack (FLA)** | `temporal_type = gated_deltanet` | Same | Explicit override |
| **Edge d_model (FLA)** | 32 | 32 | Required for FLA kernels |

## Critical Settings That MUST Match

### All Configs Must Have:
- ✅ V3 dual-stream layout (TCN → temporal streams → GNN → fusion → decoder)
- ✅ `use_dynamic_pe: true` (interval=5 for both platforms)
- ✅ `edge_top_k: 3` (validated by literature)
- ✅ `edge_similarity_margin: 0.01` (v3.3.0 safety margin)
- ✅ `focal_loss` with `alpha=0.5, gamma=2.0`
- ✅ `learning_rate`: 1e-4 (local) or 8e-5 (modal)
- ✅ `gradient_clip: 0.5` (both platforms - eigendecomp fix allows this)
- ✅ `mid_checkpoint_interval_s: 1800` (full training only, not smoke tests)
- ✅ `mid_epoch_keep: 3` (full training only)

## Memory Safety Formula (APPROXIMATE - FP32 only)

```
Memory (GB) ≈ batch_size × (3.5 + 0.94 × (960/semi_dynamic_interval))
```

**⚠️ WARNING**: This formula is APPROXIMATE and does NOT account for:
- `mixed_precision=true` (FP16 uses ~50% less memory)
- PyTorch 2.5.0 memory optimizations
- Actual empirical usage may be significantly lower

**Examples (FP32 prediction)**:
- batch=4, interval=5: 4 × (3.5 + 0.94×192) = 17.6 GB ✅ (Local)
- batch=64, interval=1: 64 × (3.5 + 0.94×960) = 284 GB ❌ (Predicts failure)

**Reality**: Modal configs with mixed_precision=true work fine on 80GB A100!
**Recommendation**: Use empirical testing over formula for mixed_precision configs.

## Recommendations

1. **BiMamba2 configs** (`*_bimamba.yaml`): Keep as production defaults (no temporal overrides).
2. **FLA configs** (`*_fla.yaml`): Copy of BiMamba2 settings + explicit Gated DeltaNet parameters.
3. **Smoke tests**: Run both stacks (`make smoke-bimamba`, `make smoke-fla`) before any long run.
4. **Full runs**: Use matching Modal/local configs for apples-to-apples comparison.
5. **Documentation**: All references should point to the new `*_bimamba.yaml` / `*_fla.yaml` filenames.
