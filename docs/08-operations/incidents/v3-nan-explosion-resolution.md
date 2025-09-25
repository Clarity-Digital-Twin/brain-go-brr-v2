# V3 NaN Explosion Incident & Resolution

## Incident Timeline
- **Date**: September 24-25, 2025
- **Impact**: V3 dual-stream architecture training completely broken
- **Resolution**: Multiple fixes implemented and verified

## Root Causes Identified

### 1. Dynamic Laplacian PE (PRIMARY CAUSE)
- **Issue**: Eigendecomposition on uninitialized/garbage adjacency matrices
- **Symptoms**: NaN explosion at batch 10-20
- **Fix**: Disabled dynamic PE in configs (`use_dynamic_pe: false`)

### 2. Optimizer Hygiene
- **Issue**: Weight decay applied to normalization parameters
- **Impact**: Gradual parameter corruption leading to NaNs
- **Fix**: Separated parameters into decay/no-decay groups in optimizer

### 3. Edge Projection Explosion
- **Issue**: 1→16 dimension projection without bounds
- **Impact**: Values exploding in edge Mamba stream
- **Fix**: Added configurable clamping (`BGB_EDGE_CLAMP=1`)

### 4. Gradient Accumulation
- **Issue**: Gradients not properly cleared on NaN batches
- **Impact**: Corrupted gradients propagating
- **Fix**: Added gradient sanitization and optional step skipping

## Implemented Fixes

### Code Changes

#### 1. Optimizer Parameter Groups (src/brain_brr/train/loop.py)
```python
# Separate parameters to prevent weight decay on normalization
no_decay = ["bias", "bn", "ln", "layernorm", "norm", "rmsnorm"]
decay_params = []
no_decay_params = []

for name, param in model.named_parameters():
    if any(nd in name.lower() for nd in no_decay):
        no_decay_params.append(param)
    else:
        decay_params.append(param)
```

#### 2. Edge Clamping (src/brain_brr/models/detector.py)
```python
# Configurable edge projection clamping
if os.getenv("BGB_EDGE_CLAMP", "1") == "1":
    clamp_min = float(os.getenv("BGB_EDGE_CLAMP_MIN", "-20.0"))
    clamp_max = float(os.getenv("BGB_EDGE_CLAMP_MAX", "20.0"))
    edge_in = torch.clamp(edge_in, clamp_min, clamp_max)
```

#### 3. Gradient Sanitization (src/brain_brr/train/loop.py)
```python
# Optional gradient sanitization
if os.getenv("BGB_SANITIZE_GRADS", "0") == "1":
    for name, param in model.named_parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            param.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
```

### Configuration Changes

#### configs/local/train.yaml
```yaml
graph:
  use_dynamic_pe: false  # CRITICAL: Disabled to prevent NaN
  semi_dynamic_interval: 5
  pe_sign_consistency: true
```

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `BGB_NAN_DEBUG` | 0 | Enable NaN debugging output |
| `BGB_EDGE_CLAMP` | 1 | Enable edge projection clamping |
| `BGB_EDGE_CLAMP_MIN` | -20.0 | Minimum clamp value |
| `BGB_EDGE_CLAMP_MAX` | 20.0 | Maximum clamp value |
| `BGB_SANITIZE_GRADS` | 0 | Enable gradient sanitization |
| `BGB_SKIP_OPT_STEP_ON_NAN` | 0 | Skip optimizer step on NaN gradients |

## Verification Tests

### Test Suite Created
- `test_v3_components.py`: Component isolation tests
- `test_v3_fixes.py`: Fix verification tests
- `debug_v3_training.py`: Debug script for NaN sources

### Test Results
- All component tests pass in isolation
- Integration tests pass with fixes applied
- Extended training (500+ batches) stable

## Lessons Learned

1. **Dynamic PE is unstable**: Eigendecomposition on learned adjacency is numerically dangerous
2. **Optimizer hygiene critical**: Weight decay must exclude normalization parameters
3. **Gradient guards necessary**: NaN gradients must be caught and handled
4. **Component isolation essential**: Test each part independently before integration

## Monitoring Recommendations

1. Always run with `BGB_NAN_DEBUG=1` during development
2. Monitor gradient norms for early warning
3. Use smaller batch sizes initially
4. Disable mixed precision on consumer GPUs (RTX 4090)

## Current Status
✅ V3 architecture stable and training successfully
- Local training: 600+ batches without NaN
- Loss converging normally (~0.15)
- All safeguards in place