# NaN Prevention & Handling: Complete Reference

**Last Updated**: September 26, 2025
**Architecture**: V3 dual-stream (TCN + BiMamba + GNN + LPE)
**Status**: PRODUCTION-READY with 3-tier clamping system

## Critical Fixes (September 26, 2025)

### Three Root Causes Identified & Fixed

1. **Data Preprocessing Issue**
   - **Problem**: EEG data contained extreme outliers (>100σ) after normalization
   - **Fix**: Added robust clipping in `preprocess.py:68`
   ```python
   x = np.clip(x, -10.0, 10.0)  # Clip to ±10 standard deviations
   ```

2. **Missing Output Sanitization**
   - **Problem**: Detection head output not sanitized before loss computation
   - **Fix**: Added Tier 3 clamping in `detector.py:313-314`
   ```python
   output = torch.nan_to_num(output, nan=0.0, posinf=50.0, neginf=-50.0)
   output = torch.clamp(output, -100.0, 100.0)  # Tier 3: Output clamping
   ```

3. **TCN Gradient Instability**
   - **Problem**: Gradients explode after ~30 batches during training
   - **Workaround**: Enable gradient sanitization with `BGB_SANITIZE_GRADS=1`

**ACTION REQUIRED**: Rebuild cache after preprocessing fix!
```bash
rm -rf cache/tusz
python -m src build-cache --data-dir data_ext4/tusz/edf --cache-dir cache/tusz
```

## 3-Tier Clamping System

| Tier | Range | Purpose | Where Used |
|------|-------|---------|------------|
| **Input** | [-10, 10] | Normalized inputs | TCN input, Mamba input |
| **Internal** | [-50, 50] | Feature maps | TCN features, Detector features |
| **Output** | [-100, 100] | Logits | Focal loss input |

## Environment Variables

### Core NaN Detection
| Variable | Default | Purpose |
|----------|---------|---------|
| `BGB_NAN_DEBUG` | 0 | Enable NaN debug output |
| `BGB_DEBUG_FINITE` | 0 | Check tensor finiteness |
| `BGB_ANOMALY_DETECT` | 0 | PyTorch anomaly detection |

### Gradient Handling (RECOMMENDED)
| Variable | Default | Purpose |
|----------|---------|---------|
| `BGB_SANITIZE_GRADS` | 0 | Replace NaN gradients (**Enable for training**) |
| `BGB_SKIP_OPT_STEP_ON_NAN` | 0 | Skip optimizer on NaN |

### Input/Activation Safeguards
| Variable | Default | Purpose |
|----------|---------|---------|
| `BGB_SANITIZE_INPUTS` | 0 | Clean inputs/labels |
| `BGB_SAFE_CLAMP` | 0 | Enable activation clamps |

### Model-Specific
| Variable | Default | Purpose |
|----------|---------|---------|
| `SEIZURE_MAMBA_FORCE_FALLBACK` | 0 | Force Conv1d fallback instead of CUDA |

## NaN Flow Through Model

```
1. Input → TCN [CHECK: torch.isnan(x)]
   ↓
2. TCN → Features [CHECK: assert_finite("tcn_out")]
   ↓
3. Features → Node/Edge Split [CHECK: assert_finite("proj_to_electrodes")]
   ↓
4. Node Mamba [CHECK: assert_finite("node_mamba")]
   ↓
5. Edge Features [CHECK: cosine similarity clamping]
   ↓
6. Edge Mamba [CHECK: edge clamping -3 to 3]
   ↓
7. Adjacency [CHECK: assert_finite("adjacency")]
   ↓
8. GNN + Dynamic PE [CHECK: eigendecomposition fallback]
   ↓
9. Backprojection [CHECK: assert_finite("backproj")]
   ↓
10. Decoder [CHECK: assert_finite("decoder_prelogits")]
    ↓
11. Logits [CHECK: assert_finite("final_logits")]
    ↓
12. Loss [CHECK: focal loss probability clamping]
    ↓
13. Gradients [CHECK: sanitize_grads if enabled]
```

## Key Implementation Locations

### Data Preprocessing
- `data/preprocess.py:66-71` - Outlier clipping + NaN removal
- `data/io.py` - Channel interpolation for missing electrodes

### Model Components
- `models/tcn.py:239-248` - Input validation & clamping
- `models/mamba.py:161-166, 303-319` - State management
- `models/edge_features.py:70-91` - Numerical stability
- `models/gnn_pyg.py:170-220` - Dynamic PE safeguards
- `models/detector.py:250-256, 305-314` - Edge & output clamping

### Training Loop
- `train/loop.py:180-224` - Focal loss with probability clamping
- `train/loop.py:566-606` - Input & logit sanitization
- `train/loop.py:694-739` - Gradient sanitization

### Debug Utilities
- `models/debug_utils.py` - assert_finite() checks throughout

## Training Configuration

### Local (RTX 4090)
```yaml
training:
  learning_rate: 1.0e-4     # Increased to prevent near-zero
  gradient_clip: 0.1        # Aggressive clipping
  mixed_precision: false    # Disabled to prevent NaNs
  loss: focal               # Required for class imbalance
  scheduler:
    warmup_ratio: 0.01      # 1% warmup
```

### Modal (A100)
```yaml
training:
  learning_rate: 3e-5       # Conservative for larger batch
  gradient_clip: 0.5        # Strong clipping
  mixed_precision: true     # A100 handles FP16 safely
  scheduler:
    warmup_ratio: 0.03      # 3% warmup
```

## Validation Commands

### Quick NaN Check
```bash
export BGB_NAN_DEBUG=1 BGB_DEBUG_FINITE=1 BGB_LIMIT_FILES=10
python -m src train configs/local/train.yaml
```

### Full Validation with Safeguards
```bash
export BGB_SANITIZE_INPUTS=1 BGB_SANITIZE_GRADS=1 BGB_SAFE_CLAMP=1
export BGB_DEBUG_FINITE=1 BGB_ANOMALY_DETECT=1
python -m src train configs/local/smoke.yaml
```

### Production Run (RECOMMENDED)
```bash
export BGB_SANITIZE_GRADS=1  # Recommended for TCN stability
python -m src train configs/local/train.yaml
```

## Status Summary

### Currently Active (Hardcoded)
- ✅ Data preprocessing: Outlier clipping + nan_to_num
- ✅ TCN input sanitization: Unconditional NaN replacement
- ✅ Mamba state management: Input/output/intermediate clamps
- ✅ Edge feature stability: Cosine similarity epsilon=1e-6
- ✅ Dynamic PE hardening: Regularization + fallback
- ✅ Conservative initialization: Gains 0.01-0.2 throughout
- ✅ Focal loss clamping: Probability [1e-6, 1-1e-6]
- ✅ Gradient clipping: 0.1 local, 0.5 modal

### Available but Disabled by Default
- ❌ `BGB_SAFE_CLAMP` - Extra activation clamping
- ❌ `BGB_SANITIZE_INPUTS` - Input NaN replacement
- ❌ `BGB_SANITIZE_GRADS` - Gradient NaN replacement (RECOMMEND ENABLING)
- ❌ `BGB_SKIP_OPT_STEP_ON_NAN` - Skip optimizer updates
- ❌ `BGB_NAN_DEBUG` - Verbose NaN reporting
- ❌ `BGB_DEBUG_FINITE` - Assert finite checks
- ❌ `BGB_ANOMALY_DETECT` - PyTorch anomaly mode

## Prevention Best Practices

1. **Always rebuild cache** after preprocessing changes
2. **Enable gradient sanitization** for training stability
3. **Use conservative initialization** (gain=0.01-0.2)
4. **Separate optimizer groups** for weight decay
5. **Monitor key statistics** every 100 batches
6. **Validate for 500+ batches** before production

## Related Documentation
- [NaN Troubleshooting Guide](nan-troubleshooting.md)
- [V3 Architecture](../04-model/v3-architecture.md)
- [Training Configuration](../03-configuration/local-configs.md)
- [Modal Deployment](../03-configuration/modal-configs.md)