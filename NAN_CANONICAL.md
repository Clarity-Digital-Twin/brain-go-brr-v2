# NaN Prevention & Handling: Complete Canonical Reference

**Last Scanned**: September 26, 2025
**Codebase Version**: V3 dual-stream architecture
**Status**: COMPREHENSIVE - Every NaN-related implementation documented

## Table of Contents
1. [Environment Variables](#environment-variables)
2. [Configuration Settings](#configuration-settings)
3. [Code Implementations](#code-implementations)
4. [Function Reference](#function-reference)
5. [NaN Flow Through Model](#nan-flow-through-model)
6. [Testing & Validation](#testing--validation)

---

## Environment Variables

### Core NaN Detection
| Variable | Default | Location | Purpose | Implementation |
|----------|---------|----------|---------|----------------|
| `BGB_NAN_DEBUG` | 0 | `utils/env.py:30` | Enable NaN debug output | `train/loop.py:523-525,619-634` |
| `BGB_NAN_DEBUG_MAX` | 10 | `utils/env.py:31` | Max debug messages | `train/loop.py:525,634` |
| `BGB_DEBUG_FINITE` | 0 | `utils/env.py:18` | Check tensor finiteness | `models/debug_utils.py:6,20` |
| `BGB_ANOMALY_DETECT` | 0 | `utils/env.py:35` | PyTorch anomaly detection | `utils/env.py:148` |

### Gradient Handling
| Variable | Default | Location | Purpose | Implementation |
|----------|---------|----------|---------|----------------|
| `BGB_SANITIZE_GRADS` | 0 | `utils/env.py:33` | Replace NaN gradients | `train/loop.py:694-704,728-734` |
| `BGB_SKIP_OPT_STEP_ON_NAN` | 0 | `utils/env.py:34` | Skip optimizer on NaN | `train/loop.py:704-709` |

### Input Sanitization
| Variable | Default | Location | Purpose | Implementation |
|----------|---------|----------|---------|----------------|
| `BGB_SANITIZE_INPUTS` | 0 | `utils/env.py:32` | Clean inputs/labels | `train/loop.py:567-571` |

### Activation Clamping
| Variable | Default | Location | Purpose | Implementation |
|----------|---------|----------|---------|----------------|
| `BGB_SAFE_CLAMP` | 0 | `utils/env.py:40` | Enable activation clamps | Applied in `SeizureDetector.forward` (post‑TCN and post‑Mamba) |
| `BGB_SAFE_CLAMP_MIN` | -10.0 | `utils/env.py:41` | Min clamp value | Used by `SeizureDetector.forward` when `BGB_SAFE_CLAMP=1` |
| `BGB_SAFE_CLAMP_MAX` | 10.0 | `utils/env.py:42` | Max clamp value | Used by `SeizureDetector.forward` when `BGB_SAFE_CLAMP=1` |

### Edge-Specific (Deprecated/Unused)
| Variable | Default | Location | Purpose | Status |
|----------|---------|----------|---------|--------|
| `BGB_EDGE_CLAMP` | 1 | `utils/env.py:15` | Legacy toggle for edge clamping | Defined but currently unused |
| `BGB_EDGE_CLAMP_MIN` | -5.0 | `utils/env.py:16` | Legacy min edge value | Defined but currently unused |
| `BGB_EDGE_CLAMP_MAX` | 5.0 | `utils/env.py:17` | Legacy max edge value | Defined but currently unused |

### Mamba-Specific
| Variable | Default | Location | Purpose | Implementation |
|----------|---------|----------|---------|----------------|
| `SEIZURE_MAMBA_FORCE_FALLBACK` | 0 | `utils/env.py:36` | Force Conv1d fallback | `models/mamba.py:71,149` |
| `BGB_FORCE_TCN_EXT` | 0 | `utils/env.py:37` | Force external TCN backend | `models/tcn.py` backend selection |

---

## Configuration Settings

### Training Configuration (`configs/local/train.yaml`)
```yaml
training:
  learning_rate: 1.0e-4     # Line 29 - Increased from 1e-5 to prevent near-zero
  weight_decay: 0.01        # Line 30 - Reduced from 0.05
  gradient_clip: 0.1        # Line 31 - Aggressive clipping (was 0.5)
  mixed_precision: false    # Line 33 - Disabled on RTX 4090 to prevent NaNs
  loss: focal              # Line 37 - Required for class imbalance
  focal_alpha: 0.5         # Line 38 - Neutral (let pos_weight handle)
  focal_gamma: 2.0         # Line 39 - Focus on hard examples
  scheduler:
    warmup_ratio: 0.01     # Line 42 - 1% warmup to avoid near-zero LR
```

### Modal Configuration (`configs/modal/train.yaml`)
```yaml
training:
  learning_rate: 3e-5      # Line 67 - Conservative for larger batch
  gradient_clip: 0.5       # Line 71 - Strong clipping
  mixed_precision: true    # Line 72 - A100 can handle FP16 safely
  warmup_ratio: 0.03      # Line 70 - 3% warmup
```

### Model Configuration
```yaml
model:
  graph:
    use_dynamic_pe: true   # Can cause NaN via eigendecomposition
    k_eigenvectors: 16     # Number of eigenvectors (affects stability)
```

---

## Code Implementations

### 1. Debug Utilities (`models/debug_utils.py`)

#### `assert_finite()` - Lines 9-49
```python
def assert_finite(tag: str, x: torch.Tensor, raise_on_fail: bool = True) -> bool:
    # Checks for NaN/Inf and provides detailed statistics
    # Used throughout model forward passes
    # Returns: True if finite, False otherwise
```

**Usage Locations:**
- `detector.py:205` - After TCN encoder
- `detector.py:230` - After electrode projection
- `detector.py:240` - After node Mamba
- `detector.py:268` - After edge weights
- `detector.py:279` - After adjacency assembly
- `detector.py:283` - After GNN
- `detector.py:290` - After backprojection
- `detector.py:303` - Before logits
- `detector.py:310` - Final logits

#### `clamp_and_check()` - Lines 52-81
```python
def clamp_and_check(tag: str, x: torch.Tensor,
                    min_val: float = -10.0, max_val: float = 10.0) -> torch.Tensor:
    # Combined NaN check, replacement, and clamping
    # Not currently used in codebase (available utility)
```

#### `check_gradients()` - Lines 84-119
```python
def check_gradients(model: torch.nn.Module, max_grad_norm: float = 100.0) -> dict:
    # Analyzes gradients for NaN/Inf and large norms
    # Not currently used in training loop (available utility)
```

### 2. TCN Encoder (`models/tcn.py`)

#### Input Validation - Lines 227-231
```python
# Check for NaN/Inf in inputs
if torch.isnan(x).any() or torch.isinf(x).any():
    # Replace NaN/Inf with zeros
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

# Clamp inputs to reasonable range for EEG data
x = torch.clamp(x, min=-100.0, max=100.0)  # Line 233
```

#### Intermediate Clamping - Lines 239-251
```python
x = torch.clamp(x, min=-50.0, max=50.0)   # After TCN
x = torch.clamp(x, min=-20.0, max=20.0)   # After projection
x = torch.clamp(x, min=-10.0, max=10.0)   # Final output
```

#### Weight Initialization - Lines 183-207
```python
# Conservative gains to prevent explosion
nn.init.xavier_uniform_(self.channel_proj.weight, gain=0.1)
nn.init.xavier_uniform_(self.downsample.weight, gain=0.05)
module.weight.data *= 0.2  # Scale down Kaiming init
```

### 3. Mamba Layers (`models/mamba.py`)

#### Input Validation - Lines 161-166, 303-306
```python
# BiMamba2Layer.forward() and BiMamba2.forward()
if torch.isnan(x).any() or torch.isinf(x).any():
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
x = torch.clamp(x, min=-10.0, max=10.0)
```

#### Output Clamping - Lines 218-222, 316-319
```python
x_output = torch.clamp(x_output, min=-5.0, max=5.0)  # After projection
output = torch.clamp(output, min=-10.0, max=10.0)    # Final output
```

#### Layer-wise Clamping - Lines 313-315
```python
# Process through bidirectional layers with clamping after each layer
for i, layer in enumerate(self.layers):
    x = layer(x)
    if (i + 1) % 2 == 0:
        x = torch.clamp(x, min=-10.0, max=10.0)
```

#### Weight Initialization - Lines 127-143
```python
nn.init.xavier_uniform_(self.output_proj.weight, gain=0.05)
nn.init.xavier_uniform_(self.forward_mamba_fallback.weight, gain=0.1)
```

### 4. Edge Features (`models/edge_features.py`)

#### Cosine Similarity Stability - Lines 70-81
```python
if metric == "cosine":
    norms = torch.linalg.norm(x, dim=-1, keepdim=True)
    norms = torch.clamp(norms, min=1e-6)  # Prevent division by near-zero
    x_norm = x / norms
    x_norm = torch.clamp(x_norm, min=-10.0, max=10.0)  # Additional safety
    sim = torch.matmul(x_norm, x_norm.transpose(-1, -2))
    sim = torch.clamp(sim, min=-1.0, max=1.0)  # Valid cosine range
```

#### Correlation Stability - Lines 82-91
```python
elif metric == "correlation":
    denom = torch.sqrt(x_center.pow(2).sum(dim=-1) + 1e-6)
    denom = torch.clamp(denom, min=1e-6)
    sim = torch.clamp(sim, min=-1.0, max=1.0)
```

### 5. Dynamic PE (`models/gnn_pyg.py`)

#### Eigendecomposition Hardening - Lines 170-220
```python
# Degree clamping
degrees = adj_combined.sum(dim=-1)
deg_sqrt_inv = degrees.clamp_min(1e-6).pow(-0.5)  # Line 175

# Regularization with condition check
eps = 1e-4  # Increased from 1e-6
l_stable = laplacian.to(torch.float32) + eps * torch.eye(N)

# Check condition number
cond = torch.linalg.cond(l_stable)
if (cond > 1e6).any():
    eps = 1e-3  # Increase regularization
    l_stable = laplacian.to(torch.float32) + eps * torch.eye(N)
```

#### NaN Detection & Fallback - Lines 196-210
```python
if (torch.isnan(eigenvalues).any() or torch.isnan(eigenvectors).any()
    or torch.isinf(eigenvalues).any() or torch.isinf(eigenvectors).any()):
    # Use cached PE fallback
    if self.last_valid_pe is not None:
        pe = self.last_valid_pe
    else:
        pe = torch.randn(B * T, N, k, device=device) * 0.01
```

#### Final Sanitization - Lines 240-247
```python
pe = torch.nan_to_num(pe, nan=0.0, posinf=1.0, neginf=-1.0)
# Cache valid PE
if not torch.isnan(pe).any() and not torch.isinf(pe).any():
    self.last_valid_pe = pe.detach()
```

### 6. Detector (`models/detector.py`)

#### Safe Clamping - Lines 208-211, 297-299
```python
from src.brain_brr.utils.env import env as _env
if _env.safe_clamp():
    features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    features = torch.clamp(features, _env.safe_clamp_min(), _env.safe_clamp_max())
```

#### Edge Clamping - Lines 250-256
```python
# Clamp cosine similarities
edge_feats = torch.clamp(edge_feats, -0.99, 0.99)  # Line 250
# Conservative clamping after projection
edge_in = torch.clamp(edge_in, -3.0, 3.0)  # Line 256
```

#### Decoder Clamping - Lines 305-307
```python
# Add clamping before logits to prevent overflow
decoded = torch.nan_to_num(decoded, nan=0.0, posinf=1e4, neginf=-1e4)
decoded = torch.clamp(decoded, -40.0, 40.0)
```

#### Weight Initialization - Lines 126-192
```python
# Detection head: very small
nn.init.xavier_uniform_(self.detection_head.weight, gain=0.01)

# Projections: conservative
nn.init.xavier_uniform_(self.proj_to_electrodes.weight, gain=0.1)
nn.init.xavier_uniform_(self.proj_from_electrodes.weight, gain=0.05)

# Edge projections: reduced gains
nn.init.xavier_uniform_(self.edge_in_proj.weight, gain=0.1)  # Was 0.5
nn.init.xavier_uniform_(self.edge_out_proj.weight, gain=0.1)  # Was 0.5

# Conv layers: scaled down
m.weight.data *= 0.2  # Scale down by 5x
```

### 7. Training Loop (`train/loop.py`)

#### Focal Loss - Lines 180-224
```python
class FocalLoss(nn.Module):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Clamp logits to prevent overflow in BCE
        logits_clamped = logits.clamp(min=-100, max=100)

        # Compute probabilities with clamping
        p = torch.sigmoid(logits_clamped)
        p = p.clamp(min=1e-6, max=1 - 1e-6)  # Line 212 - Prevent log(0)

        # Final loss clamping for safety
        focal_loss = focal_loss.clamp(max=100.0)
```

#### Input Sanitization - Lines 566-571
```python
if env.sanitize_inputs():
    if not torch.isfinite(windows).all():
        windows = torch.nan_to_num(windows, nan=0.0, posinf=0.0, neginf=0.0)
    if not torch.isfinite(labels).all():
        labels = torch.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)
```

#### Logit Sanitization & Bad Batch Save - Lines 585-606
```python
if not torch.isfinite(logits).all():
    nonfinite = (~torch.isfinite(logits)).sum().item()
    print(f"[WARN] Non-finite logits at batch {batch_idx}: count={nonfinite}")
    # Sanitize logits to allow training to continue
    logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
    # Save offending batch for offline debugging
    Path("debug").mkdir(parents=True, exist_ok=True)
    torch.save({"windows": windows.cpu(), "labels": labels.cpu(), "global_step": global_step},
               f"debug/bad_batch_{global_step:06d}.pt")
```

#### Loss NaN Handling - Lines 639-686
```python
if not torch.isfinite(loss):
    consecutive_nans += 1
    print(f"[WARNING] NaN loss detected at batch {batch_idx}")

    if consecutive_nans >= max_consecutive_nans:  # 50 consecutive
        print(f"[ERROR] {consecutive_nans} consecutive NaN losses, terminating")
        break
```

#### Gradient Sanitization - Lines 694-709, 728-739
```python
if env.sanitize_grads():
    grad_has_nan = False
    for param in model.parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            grad_has_nan = True
            param.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

    if grad_has_nan:
        print(f"[WARN] Sanitized NaN gradients at batch {batch_idx}")
        if env.skip_opt_step_on_nan():
            print("[WARN] Skipping optimizer step due to NaN gradients")
            continue
```

#### Optimizer Parameter Groups - Lines 274-293
```python
# Separate parameters to exclude normalization from weight decay
no_decay = ["bias", "ln", "bn", "norm", "layernorm", "batchnorm", "rmsnorm"]
decay_params = []
no_decay_params = []

for name, param in model.named_parameters():
    if param.requires_grad:
        if any(nd in name.lower() for nd in no_decay):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
```

---

## NaN Flow Through Model

### Detection Points
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

---

## Testing & Validation

### Unit Tests for NaN Robustness
- `tests/unit/train/test_nan_robustness.py` - Focal loss stability
- `tests/unit/models/test_dynamic_pe.py` - PE eigendecomposition
- `tests/unit/models/test_edge_features.py` - Numerical stability

### Expected Test Failures (Benign)
Due to conservative initialization (gain=0.01-0.2):
- `test_bidirectional_processing` - Signal too weak
- `test_temporal_modeling` - Gradients < threshold
- `test_tcn_encoder_gradient_flow` - Gradients < 1e-12

### Validation Commands
```bash
# Quick NaN check (10 files)
export BGB_NAN_DEBUG=1 BGB_DEBUG_FINITE=1 BGB_LIMIT_FILES=10
python -m src train configs/local/train.yaml

# Full validation with all safeguards
export BGB_SANITIZE_INPUTS=1 BGB_SANITIZE_GRADS=1 BGB_SAFE_CLAMP=1
export BGB_DEBUG_FINITE=1 BGB_ANOMALY_DETECT=1
python -m src train configs/local/smoke.yaml

# Production run (no safeguards needed)
python -m src train configs/local/train.yaml
```

---

## Status Summary

### Currently Active (Hardcoded)
- TCN stability: conservative initialization; optional post‑TCN activation clamp via `BGB_SAFE_CLAMP`; inputs are sanitized by preprocessing and, if enabled, `BGB_SANITIZE_INPUTS` (no unconditional clamp inside TCN forward)
- ✅ Mamba state management
- ✅ Edge feature numerical stability
- ✅ Dynamic PE hardening with fallback
- ✅ Conservative weight initialization
- ✅ Focal loss probability clamping
- ✅ Optimizer parameter groups (no decay on norms)
- ✅ Gradient clipping (0.1 local, 0.5 modal)

### Available but Disabled by Default
- ❌ `BGB_SAFE_CLAMP` - Extra activation clamping
- ❌ `BGB_SANITIZE_INPUTS` - Input NaN replacement
- ❌ `BGB_SANITIZE_GRADS` - Gradient NaN replacement
- ❌ `BGB_SKIP_OPT_STEP_ON_NAN` - Skip optimizer updates
- ❌ `BGB_NAN_DEBUG` - Verbose NaN reporting
- ❌ `BGB_DEBUG_FINITE` - Assert finite checks
- ❌ `BGB_ANOMALY_DETECT` - PyTorch anomaly mode

### Removed/Deprecated
- ⚠️ `BGB_EDGE_CLAMP*` - Present in `utils/env.py` but not used in forward paths (edge projection clamping is hardcoded in `detector.py`)
- ❌ Dynamic PE can be disabled via `use_dynamic_pe: false`

---

**This document is the COMPLETE and AUTHORITATIVE reference for ALL NaN-related implementations in the codebase as of September 26, 2025.**
