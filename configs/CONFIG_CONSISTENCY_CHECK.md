# Configuration Consistency Check

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
**Status**: ⚠️ Should verify these settings are present

### modal/smoke.yaml
```yaml
batch_size: 16                # Larger than local smoke
use_dynamic_pe: true
semi_dynamic_interval: 1      # A100 can handle full
epochs: 1
```
**Status**: ⚠️ Should verify

## Key Differences by Platform

| Setting | RTX 4090 | A100 | Reason |
|---------|----------|------|--------|
| **Batch Size (train)** | 4 | 64 | Memory: 24GB vs 80GB |
| **Batch Size (smoke)** | 1 | 16 | Safety vs speed |
| **Semi-dynamic Interval** | 5 | 1 | Memory constraints |
| **Mixed Precision** | false | true | RTX 4090 NaN issues |
| **Num Workers** | 0 | 8 | WSL2 vs cloud |

## Critical Settings That MUST Match

### All Configs Must Have:
- ✅ `architecture: v3`
- ✅ `use_dynamic_pe: true` (with appropriate interval)
- ✅ `edge_top_k: 3` (validated by literature)
- ✅ `focal_loss` with `alpha=0.5, gamma=2.0`
- ✅ `learning_rate: 5.0e-5`
- ✅ `gradient_clip: 0.5`

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
3. **Modal configs**: Should update to match local structure
4. **Documentation**: Update README.md with new settings