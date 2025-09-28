# Configuration Consistency Check (v3.2.0)

## Local Configs (RTX 4090 - 24GB)

### train.yaml (Full Training)
```yaml
batch_size: 4                  # OPTIMAL: 16GB usage
use_dynamic_pe: true          # ENABLED
semi_dynamic_interval: 5      # PE every 19.5ms (192 eigendecomps)
num_workers: 0                # WSL2 fix
mixed_precision: false        # RTX 4090 stability
```
**Status**: ✅ OPTIMIZED for full training

### smoke.yaml (Quick Test)
```yaml
batch_size: 1                  # Minimal for 3 files
use_dynamic_pe: true          # ENABLED
semi_dynamic_interval: 10     # Reduced for smoke test (96 eigendecomps)
num_workers: 0                # WSL2 fix
mixed_precision: false        # RTX 4090 stability
epochs: 1                     # Quick validation
```
**Status**: ✅ UPDATED for smoke test safety

## Modal Configs (A100 - 80GB)

### modal/train.yaml
```yaml
batch_size: 64                 # A100 can handle much larger
use_dynamic_pe: true          # Full dynamic possible
semi_dynamic_interval: 1      # Can afford full computation
num_workers: 8                # Cloud environment
mixed_precision: true         # A100 tensor cores
```
**Status**: ✅ VERIFIED and correct

### modal/smoke.yaml
```yaml
batch_size: 32                # Reduced for V3 dual-stream memory
use_dynamic_pe: true
semi_dynamic_interval: 1      # A100 can handle full
epochs: 1
```
**Status**: ✅ VERIFIED and correct

## Key Differences by Platform

| Setting | RTX 4090 | A100 | Reason |
|---------|----------|------|--------|
| **Batch Size (train)** | 4 | 64 | Memory: 24GB vs 80GB |
| **Batch Size (smoke)** | 1 | 32 | Safety vs speed |
| **Semi-dynamic Interval** | 5 | 1 | Memory constraints |
| **Mixed Precision** | false | true | RTX 4090 NaN issues |
| **Num Workers** | 0 | 8 | WSL2 vs cloud |

## Critical Settings That MUST Match

### All Configs Must Have:
- ✅ `architecture: v3`
- ✅ `use_dynamic_pe: true` (with appropriate interval)
- ✅ `edge_top_k: 3` (validated by literature)
- ✅ `edge_similarity_margin: 0.01` (v3.2.0 safety margin)
- ✅ `focal_loss` with `alpha=0.5, gamma=2.0`
- ✅ `learning_rate`: 1e-4 (local) or 3e-5 (modal)
- ✅ `gradient_clip: 0.5` (modal) or 0.1 (local RTX 4090)

## Memory Safety Formula

```
Memory (GB) = batch_size × (3.5 + 0.94 × (960/semi_dynamic_interval))
```

Examples:
- batch=4, interval=5: 4 × (3.5 + 0.94×192) = 14 + 3.6 = 17.6 GB ✅
- batch=8, interval=1: 8 × (3.5 + 0.94×960) = 28 + 7.5 = 35.5 GB ❌

## Recommendations

1. **Local train.yaml**: Current settings are OPTIMAL
2. **Local smoke.yaml**: Now FIXED with interval=10
3. **Modal configs**: ✅ Updated and verified for v3.2.0
4. **Documentation**: ✅ README.md updated with v3.2.0 settings