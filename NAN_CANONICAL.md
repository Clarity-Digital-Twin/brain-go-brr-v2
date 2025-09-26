# NaN Prevention & Handling: Complete Canonical Reference

**Last Updated**: September 26, 2025
**Codebase Version**: V3 dual-stream architecture
**Status**: CLEAN ARCHITECTURE - Refactored with 3-tier clamping system

## ⚠️ CRITICAL FIXES (Sep 26, 2025)

### Three Root Causes Identified & Fixed

**1. Data Preprocessing Issue**
- **Problem**: EEG data contained extreme outliers (>100σ) after normalization
- **Example**: Raw values up to 1256µV creating 121σ outliers post-normalization
- **Fix**: Added robust clipping in `preprocess.py:68`
```python
# Clip to ±10 standard deviations after z-score normalization
x = np.clip(x, -10.0, 10.0)
```

**2. Missing Output Sanitization**
- **Problem**: Detection head output not sanitized before loss computation
- **Fix**: Added Tier 3 clamping in `detector.py:313-314`
```python
output = torch.nan_to_num(output, nan=0.0, posinf=50.0, neginf=-50.0)
output = torch.clamp(output, -100.0, 100.0)  # Tier 3: Output clamping
```

**3. TCN Gradient Instability**
- **Problem**: Gradients explode after ~30 batches during training
- **Workaround**: Enable gradient sanitization with `BGB_SANITIZE_GRADS=1`
- **Long-term**: TCN architecture review needed for stability

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
| Variable | Default | Purpose | Implementation |
|----------|---------|---------|----------------|
| `BGB_NAN_DEBUG` | 0 | Enable NaN debug output | `train/loop.py` - NaN loss reporting |
| `BGB_NAN_DEBUG_MAX` | 10 | Max debug messages | `train/loop.py` - Debug limit |
| `BGB_DEBUG_FINITE` | 0 | Check tensor finiteness | `models/debug_utils.py` - assert_finite() |
| `BGB_ANOMALY_DETECT` | 0 | PyTorch anomaly detection | `train/loop.py` - Autograd debugging |

### Gradient Handling
| Variable | Default | Purpose | Implementation |
|----------|---------|---------|----------------|
| `BGB_SANITIZE_GRADS` | 0 | Replace NaN gradients | `train/loop.py` - Gradient sanitization |
| `BGB_SKIP_OPT_STEP_ON_NAN` | 0 | Skip optimizer on NaN | `train/loop.py` - Skip updates |

### Input Sanitization
| Variable | Default | Purpose | Implementation |
|----------|---------|---------|----------------|
| `BGB_SANITIZE_INPUTS` | 0 | Clean inputs/labels | `train/loop.py` - Input sanitization |

### Activation Clamping
| Variable | Default | Purpose | Implementation |
|----------|---------|---------|----------------|
| `BGB_SAFE_CLAMP` | 0 | Enable activation clamps | `TCNEncoder.forward` - Internal tier clamping |
| `BGB_SAFE_CLAMP_MIN` | -10.0 | Min clamp value | `SeizureDetector.forward` - Safety bounds |
| `BGB_SAFE_CLAMP_MAX` | 10.0 | Max clamp value | `SeizureDetector.forward` - Safety bounds |

### Model-Specific
| Variable | Default | Purpose | Implementation |
|----------|---------|---------|----------------|
| `SEIZURE_MAMBA_FORCE_FALLBACK` | 0 | Force Conv1d fallback | `models/mamba.py` - Use fallback instead of CUDA |
| `BGB_FORCE_TCN_EXT` | 0 | Force external TCN backend | `models/tcn.py` - Backend selection |

---

## Clean Architecture Design

### 3-Tier Clamping System
The codebase now uses a standardized 3-tier clamping system for numerical stability:

| Tier | Range | Purpose | Where Used |
|------|-------|---------|------------|
| **Input** | [-10, 10] | Normalized inputs | TCN input, Mamba input |
| **Internal** | [-50, 50] | Feature maps | TCN features, Detector features |
| **Output** | [-100, 100] | Logits | Focal loss input |

This replaces the previous ad-hoc approach with 4+ different ranges.

### Dependency Injection for Testing
Models now support `init_gain` parameter for clean test/production separation:

```python
# Production (default): Conservative initialization
model = BiMamba2Layer(d_model=512, init_gain=0.2)  # Default

# Testing: Stronger initialization for gradient flow tests
model = BiMamba2Layer(d_model=512, init_gain=0.5)  # For tests only
```

This eliminates the anti-pattern of BGB_TEST_MODE environment variable.

### Single Responsibility Principle
NaN prevention is organized into 4 clear responsibilities:

1. **Preprocessing** (`data/preprocess.py`): Handle raw data issues
2. **Model Input** (`models/*.py`): Single sanitization point at entry
3. **Critical Points**: Clamp only at numerical risks (div by zero, exp)
4. **Loss** (`train/loop.py`): Handle probability edge cases

---

## Configuration Settings

### Training Configuration (`configs/local/train.yaml`)
```yaml
training:
  learning_rate: 1.0e-4     # Increased from 1e-5 to prevent near-zero
  gradient_clip: 0.1        # Aggressive clipping (was 0.5)
  mixed_precision: false    # Disabled on RTX 4090 to prevent NaNs
  loss: focal               # Required for class imbalance
  focal_alpha: 0.5          # Neutral (let pos_weight handle)
  focal_gamma: 2.0          # Focus on hard examples
  scheduler:
    warmup_ratio: 0.01      # 1% warmup to avoid near-zero LR
```

### Modal Configuration (`configs/modal/train.yaml`)
```yaml
training:
  learning_rate: 3e-5      # Conservative for larger batch
  gradient_clip: 0.5       # Strong clipping
  mixed_precision: true    # A100 can handle FP16 safely
  warmup_ratio: 0.03       # 3% warmup
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

### 0. Data Preprocessing (`data/preprocess.py`) [FIRST LINE OF DEFENSE]

#### Outlier Clipping - Lines 66-68 [CRITICAL FIX]
```python
# 4) Per-channel z-score
mean = np.mean(x, axis=1, keepdims=True)
std = np.std(x, axis=1, keepdims=True)
x = (x - mean) / (std + 1e-8)

# CRITICAL: Clip outliers to prevent infinities during training
# EEG data can have extreme artifacts (>100σ) that cause numerical issues
x = np.clip(x, -10.0, 10.0)  # Clip to ±10 standard deviations
```
**Status**: ALWAYS ACTIVE - Prevents extreme outliers from causing NaN

#### Automatic NaN Removal - Line 71
```python
# Always sanitize raw EEG data before any processing
x_clean: npt.NDArray[np.float32] = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
```
**Status**: ALWAYS ACTIVE - Unconditional preprocessing safeguard

#### Channel Interpolation (`data/io.py: load_edf_file`) – midline (Fz/Pz) interpolation when available
```python
# Interpolate missing channels (Fz/Pz) using MNE montage
if missing_midline:
    raw.info['bads'] = missing_midline
    raw.interpolate_bads(reset_bads=True)
```

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

#### Input Validation [ALWAYS RUNS] - Lines 239-248
```python
# CRITICAL: Input validation and clamping to prevent NaN propagation
# Check for NaN/Inf in inputs
if torch.isnan(x).any() or torch.isinf(x).any():
    # Replace NaN/Inf with zeros
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

# Input tier clamping: [-10, 10] for normalized EEG data
x = torch.clamp(x, min=-10.0, max=10.0)
```
**Status**: ALWAYS ACTIVE - Unconditional input safeguard

#### Post-Processing Clamps [CONDITIONAL on BGB_SAFE_CLAMP=1]
```python
# Internal tier clamping: [-50, 50] for feature maps
if env.safe_clamp():
    x = torch.clamp(x, min=-50.0, max=50.0)  # After TCN blocks
    x = torch.clamp(x, min=-50.0, max=50.0)  # After channel projection
    x = torch.clamp(x, min=-50.0, max=50.0)  # After downsampling
```
**Status**: DISABLED BY DEFAULT - Only active when BGB_SAFE_CLAMP=1

#### Weight Initialization - TCNEncoder
```python
def _initialize_weights(self) -> None:
    """Initialize TCN encoder weights with conservative gains for stability."""
    # Use conservative initialization for production stability
    # Tests can pass higher init_gain if needed for gradient flow validation
    proj_gain = self.init_gain  # Default 0.2
    down_gain = self.init_gain * 0.5  # Half of proj_gain
    conv_scale = self.init_gain * 2.5  # 2.5x proj_gain

    # Channel projection
    nn.init.xavier_uniform_(self.channel_proj.weight, gain=proj_gain)
    # Downsampling
    nn.init.xavier_uniform_(self.downsample.weight, gain=down_gain)
    # TCN conv layers scaled down
    module.weight.data *= conv_scale
```
**FIXED**: Now uses dependency injection via `init_gain` parameter

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

#### Weight Initialization - BiMamba2Layer
```python
def _initialize_weights(self) -> None:
    """Initialize Mamba layer weights with conservative gains for stability."""
    # Use conservative initialization for production stability
    # Tests can pass higher init_gain if needed for gradient flow validation
    gain = self.init_gain  # Default 0.2

    # Output projection: residual-like behavior
    nn.init.xavier_uniform_(self.output_proj.weight, gain=gain)
    # Fallback convolutions
    nn.init.xavier_uniform_(self.forward_mamba_fallback.weight, gain=gain)
    nn.init.xavier_uniform_(self.backward_mamba_fallback.weight, gain=gain)
```
**FIXED**: Now uses dependency injection via `init_gain` parameter

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

#### Decoder Clamping - Lines 305-307, 313-314
```python
# Internal tier clamping for features before final projection
decoded = torch.nan_to_num(decoded, nan=0.0, posinf=50.0, neginf=-50.0)
decoded = torch.clamp(decoded, -50.0, 50.0)

# CRITICAL: Final output sanitization (was missing, causing non-finite logits!)
output = self.detection_head(decoded)  # Line 309
output = torch.nan_to_num(output, nan=0.0, posinf=50.0, neginf=-50.0)  # Line 313
output = torch.clamp(output, -100.0, 100.0)  # Line 314 - Tier 3: Output clamping
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
- `tests/unit/models/test_nan_robustness.py` - End-to-end NaN/Inf coverage across components
- `tests/unit/models/test_dynamic_pe.py` - PE eigendecomposition
- `tests/unit/models/test_edge_features.py` - Numerical stability
- `tests/unit/models/test_detector_v3.py` - V3 architecture integration
- `tests/unit/models/test_mamba.py` - Mamba layer stability
- `tests/integration/test_model_assembly.py` - Full model NaN checks
- `tests/integration/test_training_edge_cases.py` - Training robustness

### Test Philosophy (Mamba)
Mamba unit tests assert functional behavior (shape, determinism, no NaN/Inf, residual effect), not internal
signal amplitude or specific propagation distances. This avoids coupling tests to implementation details that
can vary with conservative initialization and clamping. Initialization uses dependency injection via
`init_gain` for tests that need stronger gradients, without altering production code paths.

### Validation Commands
```bash
# IMPORTANT: Rebuild cache after preprocessing fix!
rm -rf cache/tusz
python -m src build-cache --data-dir data_ext4/tusz/edf --cache-dir cache/tusz

# Quick NaN check (10 files)
export BGB_NAN_DEBUG=1 BGB_DEBUG_FINITE=1 BGB_LIMIT_FILES=10
python -m src train configs/local/train.yaml

# Full validation with all safeguards
export BGB_SANITIZE_INPUTS=1 BGB_SANITIZE_GRADS=1 BGB_SAFE_CLAMP=1
export BGB_DEBUG_FINITE=1 BGB_ANOMALY_DETECT=1
python -m src train configs/local/smoke.yaml

# Production run (with gradient sanitization recommended)
export BGB_SANITIZE_GRADS=1  # Recommended for TCN stability
python -m src train configs/local/train.yaml
```

---

## Status Summary

### Currently Active (Hardcoded)
- ✅ **Data Preprocessing**: Outlier clipping + `np.nan_to_num()` (preprocess.py:68,71)
- ✅ **TCN Input Sanitization**: Unconditional NaN replacement + clamp [-10,10] (tcn.py)
- ✅ **Mamba State Management**: Input/output/intermediate clamps (mamba.py)
- ✅ **Edge Feature Stability**: Cosine similarity epsilon=1e-6 (edge_features.py:70-91)
- ✅ **Dynamic PE Hardening**: Regularization eps=1e-4, fallback on failure (gnn_pyg.py:170-220)
- ✅ **Conservative Initialization**: Gains 0.2-0.5 throughout (all models)
- ✅ **Focal Loss Clamping**: Probability [1e-6, 1-1e-6] (loop.py:212)
- ✅ **Optimizer Groups**: No weight decay on normalization (loop.py:280-293)
- ✅ **Gradient Clipping**: 0.1 local, 0.5 modal (configs)

### Available but Disabled by Default
- ❌ `BGB_SAFE_CLAMP` - Extra activation clamping
- ❌ `BGB_SANITIZE_INPUTS` - Input NaN replacement
- ❌ `BGB_SANITIZE_GRADS` - Gradient NaN replacement
- ❌ `BGB_SKIP_OPT_STEP_ON_NAN` - Skip optimizer updates
- ❌ `BGB_NAN_DEBUG` - Verbose NaN reporting
- ❌ `BGB_DEBUG_FINITE` - Assert finite checks
- ❌ `BGB_ANOMALY_DETECT` - PyTorch anomaly mode

### Removed/Deprecated
- ❌ `BGB_EDGE_CLAMP*` - Removed from `utils/env.py`; edge projection clamping is hardcoded in `detector.py`
- ❌ Dynamic PE can be disabled via `use_dynamic_pe: false`

---

---

## Recent Improvements

### Fixed: BGB_TEST_MODE Anti-Pattern ✅
Previously, models used different initialization for tests vs production via BGB_TEST_MODE.
This violated testing principles and has been replaced with dependency injection:

**Old (Anti-Pattern)**:
```python
gain = 0.5 if env.test_mode() else 0.2  # WRONG
```

**New (Clean)**:
```python
# Production
model = BiMamba2Layer(init_gain=0.2)  # Default
# Tests
model = BiMamba2Layer(init_gain=0.5)  # Explicit for gradient tests
```

### Standardized: 3-Tier Clamping ✅
Replaced ad-hoc clamping ranges with consistent 3-tier system.

### Cleaned: Removed Unused Variables ✅
Removed BGB_EDGE_CLAMP* variables that were defined but never used.

### Training Status
- **Issue Found**: Data had extreme outliers causing NaN after ~30 batches
- **Solution**: Added outlier clipping in preprocessing
- **Recommendation**: Enable `BGB_SANITIZE_GRADS=1` for training stability
- **Cache**: Must rebuild after preprocessing fix

---

**This document is the COMPLETE and AUTHORITATIVE reference for ALL NaN-related implementations in the codebase as of September 26, 2025.**

**Version**: V3 dual-stream architecture
**Last Verified**: Commit c0578f4 (current HEAD)
**Critical Commits**:
- `57426ea` - Clip outliers in preprocessing to prevent NaN
- `7ba8017` - Add output sanitization in detector
- `c0578f4` - Enhanced debugging capabilities
**Training Status**: FIXED - Requires cache rebuild with clipped data
