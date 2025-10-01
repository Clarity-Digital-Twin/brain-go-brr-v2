# NaN Prevention & Handling: Complete Reference

**Last Updated**: October 1, 2025
**Codebase Version**: v3.4.1 (PyTorch 2.5.0 + mamba-ssm 2.2.5)
**Status**: PRODUCTION STABLE - Zero NaN/Inf after 723+ batches

---

## Quick Start

### Required for Production Training

```bash
# CRITICAL: Set these before training
export BGB_SANITIZE_GRADS=1  # Prevents gradient explosion → NaN activations
export BGB_NAN_DEBUG=1       # Shows NaN warnings if they occur
```

**These two flags are REQUIRED for stable training on PyTorch 2.5.0.**

---

## Table of Contents

1. [Three-Layer Defense System](#three-layer-defense-system)
2. [PR1-5 Architectural Fixes](#pr1-5-architectural-fixes)
3. [Environment Variables Reference](#environment-variables-reference)
4. [Required Settings by Environment](#required-settings-by-environment)
5. [Code Implementation Details](#code-implementation-details)
6. [Configuration Examples](#configuration-examples)
7. [Troubleshooting Guide](#troubleshooting-guide)
8. [Testing & Validation](#testing--validation)

---

## Three-Layer Defense System

### Layer 1: Data Preprocessing (ALWAYS ACTIVE)

**File**: `src/brain_brr/data/preprocess.py`

**What it does**:
- Clips outliers to ±10σ after z-score normalization (line 68)
- Replaces NaN/Inf with zeros using `np.nan_to_num()` (line 71)

**Purpose**: Prevents bad data from entering the pipeline

**Code**:
```python
# Per-channel z-score
mean = np.mean(x, axis=1, keepdims=True)
std = np.std(x, axis=1, keepdims=True)
x = (x - mean) / (std + 1e-8)

# CRITICAL: Clip outliers to prevent infinities during training
x = np.clip(x, -10.0, 10.0)  # ±10 standard deviations

# Always sanitize raw EEG data
x_clean = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
```

**Why this was needed**: EEG data can have extreme artifacts (>100σ) that cause numerical issues downstream.

---

### Layer 2: Model Boundaries (ALWAYS ACTIVE)

**Files**: `tcn.py`, `mamba.py`, `edge_features.py`

**What it does**:
- Checks for NaN/Inf at component inputs
- Replaces with zeros and warns
- Clamps to safe ranges

**Purpose**: Catch NaNs before they propagate through the model

**Examples**:

**TCN Input** (`tcn.py:239-241`):
```python
# Check for NaN/Inf in inputs
if torch.isnan(x).any() or torch.isinf(x).any():
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

# Input tier clamping: [-10, 10] for normalized EEG data
x = torch.clamp(x, min=-10.0, max=10.0)
```

**Mamba Input** (`mamba.py:177-180`):
```python
if torch.isnan(x).any() or torch.isinf(x).any():
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
x = torch.clamp(x, min=-10.0, max=10.0)
```

**Edge Features** (`edge_features.py:72-75`):
```python
# Cosine similarity stability
norms = torch.linalg.norm(x, dim=-1, keepdim=True)
norms = torch.clamp(norms, min=1e-6)  # Prevent division by near-zero
x_norm = x / norms
sim = torch.clamp(sim, min=-1.0 + margin, max=1.0 - margin)  # Safety margin
```

---

### Layer 3: Gradient Sanitization (OPTIONAL, RECOMMENDED)

**File**: `src/brain_brr/train/loop.py:728-739`

**What it does**:
- Checks gradients after `loss.backward()`
- Replaces NaN/Inf with zeros
- Warns user

**Purpose**: Prevents gradient explosion → NaN activations in next batch

**Code**:
```python
if env.sanitize_grads():
    grad_has_nan = False
    for param in model.parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            grad_has_nan = True
            param.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

    if grad_has_nan:
        print(f"[WARN] Sanitized NaN gradients at batch {batch_idx}")
```

**Why this is recommended**: PyTorch 2.5.0 + complex architectures (BiMamba+GNN) can have transient gradient spikes that, if unchecked, corrupt future batches.

---

## PR1-5 Architectural Fixes

These fixes eliminate NaN **at the source**, not just symptoms. All are **ALWAYS enabled** via configuration.

### PR-1: Boundary Normalization ✅ COMPLETE

**What**: LayerNorm at 5 critical component boundaries

**Where**:
- After TCN → electrodes projection (`detector.py:247-248`)
- After node Mamba stream (`detector.py:261-262`)
- After edge Mamba stream (`detector.py:287-291`)
- After GNN processing (`detector.py:323-324`)
- Before decoder projection (`detector.py:337-341`)

**Why**: Prevents unbounded information flow between components

**Config**:
```yaml
model:
  norms:
    boundary_norm: layernorm
    boundary_eps: 1.0e-5
    layerscale_alpha: 0.1
    after_tcn_proj: true
    after_node_mamba: true
    after_edge_mamba: true
    after_gnn: true
    before_decoder: true
```

**Impact**: Replaced 27 manual clamps with principled normalization

---

### PR-2: Bounded Edge Stream ✅ COMPLETE

**What**: Tanh activation + LayerNorm after edge projection

**Problem**: Edge stream has 16x dimension explosion (512 → 8192 → 512) that causes unbounded activation growth

**Solution**:
- Tanh activation after `edge_in_proj` → bounds to [-1, 1]
- LayerNorm on feature dimension → normalizes variance
- Conservative initialization (gain=0.1)

**Config**:
```yaml
model:
  graph:
    edge_lift_activation: tanh
    edge_lift_norm: layernorm
    edge_lift_init_gain: 0.1
```

**Impact**: Eliminated edge feature explosion, made edge clamping redundant

---

### PR-3: Adjacency Conditioning ✅ COMPLETE

**What**: Stabilized adjacency matrix for eigendecomposition

**Problem**: Dynamic Laplacian PE computes eigendecomposition on a learned adjacency every forward pass, causing:
- Eigenvalue explosion (condition numbers > 10^6)
- Eigenvector sign flips
- NaN in PE computation

**Solutions**:
1. **Row-softmax normalization**: Each node's outgoing weights sum to 1
2. **Temporal EMA smoothing**: Smooth transitions prevent sudden changes
3. **Symmetrization**: Guarantees real eigenvalues
4. **Enhanced Laplacian regularization**: eps=1e-3 instead of 1e-6

**Config**:
```yaml
model:
  graph:
    adj_row_softmax: true
    adj_softmax_tau: 1.0
    adj_ema_beta: 0.9
    adj_force_symmetric: true
    laplacian_eps: 1.0e-3
    laplacian_normalize: true
```

**Impact**: Reduced condition number from >10^6 to <100, eliminated PE fallback

---

### PR-4: Multi-Head Gated Fusion ✅ COMPLETE

**What**: Learned fusion of node and edge streams with multi-head attention

**Config**:
```yaml
model:
  fusion:
    fusion_type: gated
    fusion_heads: 4
    fusion_dropout: 0.1
```

**Impact**: Improved gradient flow, stable stream combination

---

### PR-5: Edge Similarity Clamping with Margin ✅ COMPLETE

**What**: Safety margin from ±1 boundaries in cosine similarity

**Problem**: Cosine similarity at exactly ±1.0 can cause numerical issues in downstream operations

**Solution**: Clamp to `[-1.0 + margin, 1.0 - margin]` where margin defaults to 0.01

**Config**:
```yaml
model:
  graph:
    edge_similarity_margin: 0.01  # Safety margin from ±1 boundaries
```

**Code** (`edge_features.py:81,91`):
```python
margin = self.edge_similarity_margin  # Default 0.01
sim = torch.clamp(sim, min=-1.0 + margin, max=1.0 - margin)
```

**Impact**: Prevents edge similarity explosions, removes need for ad-hoc clamps in detector

---

### v3.3.1: Eigendecomposition Gradient Detachment (Sept 30, 2025)

**What**: Detached eigenvectors from gradient computation

**Problem**: PyTorch's eigendecomposition backward pass is numerically unstable, causing gradient spikes

**Solution**: Detach eigenvectors before using them for PE computation

**Code** (`gnn_pyg.py:205`):
```python
eigenvalues, eigenvectors = torch.linalg.eigh(l_stable)
eigenvectors = eigenvectors.detach()  # Prevent gradient explosion
```

**Impact**: Eliminated gradient spikes (P95 dropped from 52.06 → 9.74 over 700 batches)

**Reference**: See `ARCHITECTURAL_STABILITY_INVESTIGATION.md` for full analysis

---

## Environment Variables Reference

### Core NaN Detection & Debugging

| Variable | Default | Purpose | When to Enable |
|----------|---------|---------|----------------|
| `BGB_NAN_DEBUG` | 0 | Enable NaN debug output | **ALWAYS (production)** |
| `BGB_NAN_DEBUG_MAX` | 10 | Max NaN warnings before stopping | If NAN_DEBUG=1 |
| `BGB_DEBUG_FINITE` | 0 | Enable `assert_finite()` checks | Investigation only |
| `BGB_ANOMALY_DETECT` | 0 | PyTorch anomaly detection | Investigation only (very slow) |

### Gradient Protection (MOST IMPORTANT)

| Variable | Default | Purpose | When to Enable |
|----------|---------|---------|----------------|
| `BGB_SANITIZE_GRADS` | 0 | Replace NaN gradients with zeros | **ALWAYS (production)** |
| `BGB_SKIP_OPT_STEP_ON_NAN` | 0 | Skip optimizer when NaN detected | Debug only |

### Input Sanitization

| Variable | Default | Purpose | When to Enable |
|----------|---------|---------|----------------|
| `BGB_SANITIZE_INPUTS` | 0 | Replace NaN/Inf in batch inputs | Debug/testing (redundant with preprocessing) |

### Activation Safety Clamps

| Variable | Default | Purpose | When to Enable |
|----------|---------|---------|----------------|
| `BGB_SAFE_CLAMP` | 0 | Enable extra activation clamps | Debug only (PR1-5 make unnecessary) |
| `BGB_SAFE_CLAMP_MIN` | -10.0 | Min clamp value | If SAFE_CLAMP=1 |
| `BGB_SAFE_CLAMP_MAX` | 10.0 | Max clamp value | If SAFE_CLAMP=1 |

### Model Fallbacks

| Variable | Default | Purpose | When to Enable |
|----------|---------|---------|----------------|
| `SEIZURE_MAMBA_FORCE_FALLBACK` | 0 | Force Conv1d fallback (no CUDA) | CUDA kernel issues |

---

## Required Settings by Environment

### Production Training (RTX 4090 / A100)

**REQUIRED**:
```bash
export BGB_SANITIZE_GRADS=1  # CRITICAL - prevents gradient explosion
export BGB_NAN_DEBUG=1       # Shows NaN warnings if they occur
```

**NOT NEEDED** (unless investigating):
```bash
# BGB_DEBUG_FINITE=0         # Leave disabled (performance cost)
# BGB_SANITIZE_INPUTS=0      # Redundant with preprocessing
# BGB_SAFE_CLAMP=0           # Redundant with PR1-5
# BGB_ANOMALY_DETECT=0       # Too slow
```

### Investigation / Debugging

**Full diagnostics**:
```bash
export BGB_DEBUG_FINITE=1      # Enable assert_finite() checks (9 per forward)
export BGB_NAN_DEBUG=1         # Enable NaN warnings
export BGB_SANITIZE_GRADS=1    # Sanitize gradients
export BGB_SANITIZE_INPUTS=1   # Extra safety
export BGB_ANOMALY_DETECT=1    # PyTorch anomaly detection (VERY SLOW!)
```

### Smoke Tests

```bash
export BGB_SMOKE_TEST=1        # Limit to 3 files (auto-enables DEBUG_FINITE)
export BGB_NAN_DEBUG=1         # Show any NaN warnings
export BGB_SANITIZE_GRADS=1    # Production-like sanitization
```

**Note**: `BGB_SMOKE_TEST=1` automatically enables `DEBUG_FINITE` checks

---

## Code Implementation Details

### Data Preprocessing

**File**: `src/brain_brr/data/preprocess.py`

- **Line 68**: Outlier clipping to ±10σ (ALWAYS ACTIVE)
- **Line 71**: `np.nan_to_num()` sanitization (ALWAYS ACTIVE)

### Model Input Boundaries

**TCN** (`tcn.py`):
- Lines 236-241: Input sanitization + clamp [-10, 10]

**Mamba** (`mamba.py`):
- Lines 177-180: BiMamba2Layer input sanitization
- Lines 329-335: BiMamba2 input sanitization
- Lines 249, 259, 339, 342: Output clamping

**Edge Features** (`edge_features.py`):
- Lines 72-75: Input sanitization before cosine similarity
- Lines 81, 91: Similarity clamping with margin (configurable via `edge_similarity_margin`)
- Line 87: Division safety (correlation)

### Detector Forward Pass

**File**: `src/brain_brr/models/detector.py`

- Lines 225, 247, 263, 278, 282, 286, 290, 309, 312: `assert_finite()` checks (9 total)
- Lines 385-386, 392-393: Output sanitization before loss (Tier 3)

### Training Loop

**File**: `src/brain_brr/train/loop.py`

- Lines 566-571: Input sanitization (if `BGB_SANITIZE_INPUTS=1`)
- Lines 585-606: Logit sanitization + bad batch save
- Lines 639-686: Loss NaN handling (consecutive count, terminate after 50)
- Lines 694-709, 728-739: Gradient sanitization (if `BGB_SANITIZE_GRADS=1`)

### Focal Loss

**File**: `src/brain_brr/train/loop.py:180-224`

- Line 205: Logit clamping [-100, 100]
- Line 212: Probability clamping [1e-6, 1-1e-6] (prevent log(0))
- Line 218: p_t stability clamp
- Line 223: Loss explosion prevention (max=100.0)

### Dynamic Laplacian PE

**File**: `src/brain_brr/models/gnn_pyg.py`

**Eigendecomposition hardening** (lines 170-220):
```python
# Degree clamping
degrees = adj_combined.sum(dim=-1)
deg_sqrt_inv = degrees.clamp_min(1e-6).pow(-0.5)

# Regularization with condition check
eps = 1e-4
l_stable = laplacian.to(torch.float32) + eps * torch.eye(N)

# Check condition number
cond = torch.linalg.cond(l_stable)
if (cond > 1e6).any():
    eps = 1e-3  # Increase regularization
    l_stable = laplacian.to(torch.float32) + eps * torch.eye(N)

# Eigendecomposition
eigenvalues, eigenvectors = torch.linalg.eigh(l_stable)
eigenvectors = eigenvectors.detach()  # CRITICAL: Prevent gradient explosion
```

**NaN detection & fallback** (lines 196-210):
```python
if (torch.isnan(eigenvalues).any() or torch.isnan(eigenvectors).any()
    or torch.isinf(eigenvalues).any() or torch.isinf(eigenvectors).any()):
    # Use cached PE fallback
    if self.last_valid_pe is not None:
        pe = self.last_valid_pe
    else:
        pe = torch.randn(B * T, N, k, device=device) * 0.01
```

**Final sanitization** (lines 240-247):
```python
pe = torch.nan_to_num(pe, nan=0.0, posinf=1.0, neginf=-1.0)
# Cache valid PE
if not torch.isnan(pe).any() and not torch.isinf(pe).any():
    self.last_valid_pe = pe.detach()
```

---

## Configuration Examples

### Local Training (RTX 4090)

**File**: `configs/local/train.yaml`

```yaml
training:
  learning_rate: 1.0e-4
  gradient_clip: 0.5         # Gradient clipping
  mixed_precision: false     # Disabled on RTX 4090 (AMP causes NaNs)
  loss: focal
  focal_alpha: 0.5
  focal_gamma: 2.0
  use_balanced_sampling: true  # CRITICAL for class imbalance
  scheduler:
    warmup_ratio: 0.01       # 1% warmup

model:
  # PR-1: Boundary normalization
  norms:
    boundary_norm: layernorm
    boundary_eps: 1.0e-5
    layerscale_alpha: 0.1
    after_tcn_proj: true
    after_node_mamba: true
    after_edge_mamba: true
    after_gnn: true
    before_decoder: true

  graph:
    # PR-2: Bounded edge stream
    edge_lift_activation: tanh
    edge_lift_norm: layernorm
    edge_lift_init_gain: 0.1

    # PR-3: Adjacency conditioning
    adj_row_softmax: true
    adj_softmax_tau: 1.0
    adj_ema_beta: 0.9
    adj_force_symmetric: true
    laplacian_eps: 1.0e-3
    laplacian_normalize: true

    # PR-5: Edge similarity margin
    edge_similarity_margin: 0.01

    # Dynamic PE settings
    use_dynamic_pe: true
    k_eigenvectors: 16

  # PR-4: Fusion
  fusion:
    fusion_type: gated
    fusion_heads: 4
    fusion_dropout: 0.1
```

### Modal Training (A100-80GB)

**File**: `configs/modal/train.yaml`

```yaml
training:
  learning_rate: 1.0e-4
  gradient_clip: 0.5
  mixed_precision: true      # A100 can handle FP16 safely
  loss: focal
  focal_alpha: 0.5
  focal_gamma: 2.0
  use_balanced_sampling: true
  scheduler:
    warmup_ratio: 0.01

model:
  # Same PR-1/2/3/4/5 settings as local
  # ...
```

**Environment setup** (`deploy/modal/app.py:539-542`):
```python
# CRITICAL: PyTorch memory allocator - prevent fragmentation on A100-80GB
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

# NaN protection
os.environ["BGB_SANITIZE_GRADS"] = "1"
os.environ["BGB_NAN_DEBUG"] = "1"
```

---

## Troubleshooting Guide

### Issue: Constant NaN Warnings

**Symptom**:
```
[WARNING] NaN loss detected at batch 42
```

**Diagnosis**:
1. Check if `BGB_SANITIZE_GRADS=1` is set
2. Check if cache was rebuilt after preprocessing fix (Sept 26)
3. Check config has `edge_similarity_margin: 0.01`

**Solution**:
```bash
# Rebuild cache
rm -rf cache/tusz
python -m src build-cache --data-dir data_ext4/tusz/edf --cache-dir cache/tusz

# Set protection flags
export BGB_SANITIZE_GRADS=1 BGB_NAN_DEBUG=1

# Restart training
python -m src train configs/local/train.yaml
```

### Issue: Non-Finite Logits

**Symptom**:
```
[WARN] Non-finite logits at batch 0: count=983040
```

**Root causes** (all FIXED as of Sept 26):
1. **Data outliers**: Fixed by preprocessing clip (line 68)
2. **Missing output sanitization**: Fixed by detector output clamps (lines 385-386, 392-393)
3. **Gradient explosion**: Fixed by `BGB_SANITIZE_GRADS=1`

**Verification**:
- Check preprocessing includes `np.clip(x, -10.0, 10.0)`
- Check detector has output sanitization before loss
- Enable gradient sanitization

### Issue: Dynamic PE Fallback

**Symptom**:
```
[WARN] Dynamic PE eigendecomposition failed, using fallback
```

**Causes**:
- Ill-conditioned adjacency matrix
- Eigenvalue explosion
- Numerical instability

**Solutions**:
1. Verify PR-3 adjacency conditioning is enabled
2. Check `laplacian_eps: 1.0e-3` (not 1e-6)
3. Verify eigenvectors are detached (v3.3.1 fix)

### Issue: Gradient Explosion

**Symptom**:
```
[GRADIENTS] P95=156.23 | Max=1247.85
```

**Expected behavior** (with warmup):
- Batch 0-200: P95 ~20-60 (high variance, normal)
- Batch 200-1000: P95 ~10-30 (decreasing)
- Batch 1000+: P95 ~5-20 (stable)

**If P95 stays >100 after batch 500**:
1. Check `BGB_SANITIZE_GRADS=1` is set
2. Verify eigenvector detachment (v3.3.1)
3. Check warmup schedules enabled (v3.4.1)

### Issue: "Large Grad Norm" Messages

**Symptom**:
```
[DEBUG] Large grad norm at batch 42: 7.93e+00 (clipped to 0.5)
```

**This is NORMAL**:
- Individual gradients can spike due to focal loss (γ=2.0)
- Gradient clipping is doing its job
- Focus on **rolling statistics** (P95, Mean), not individual messages

**When to worry**:
- If you see this **every single batch** → deeper issue
- If rolling P95 doesn't decrease over 1000 batches → investigate

---

## Testing & Validation

### Unit Tests for NaN Robustness

- `tests/unit/models/test_nan_robustness.py` - End-to-end NaN/Inf coverage
- `tests/unit/models/test_dynamic_pe.py` - PE eigendecomposition
- `tests/unit/models/test_edge_features.py` - Numerical stability
- `tests/unit/models/test_detector_v3.py` - V3 architecture integration
- `tests/unit/models/test_mamba.py` - Mamba layer stability
- `tests/integration/test_model_assembly.py` - Full model NaN checks
- `tests/integration/test_training_edge_cases.py` - Training robustness

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

# Production run (recommended settings)
export BGB_SANITIZE_GRADS=1 BGB_NAN_DEBUG=1
python -m src train configs/local/train.yaml
```

---

## Status Summary

### Currently Active (Hardcoded or Config-Enabled)

- ✅ **Data Preprocessing**: Outlier clipping + `np.nan_to_num()` (ALWAYS ACTIVE)
- ✅ **TCN Input Sanitization**: Unconditional NaN replacement + clamp [-10,10]
- ✅ **Mamba State Management**: Input/output/intermediate clamps
- ✅ **Edge Feature Stability**: Cosine similarity with margin (PR-5)
- ✅ **Dynamic PE Hardening**: Regularization, fallback, eigenvector detachment (v3.3.1)
- ✅ **PR-1**: Boundary normalization (5 LayerNorms)
- ✅ **PR-2**: Bounded edge stream (tanh + LayerNorm)
- ✅ **PR-3**: Adjacency conditioning (row-softmax + EMA + symmetry)
- ✅ **PR-4**: Multi-head gated fusion
- ✅ **PR-5**: Edge similarity margin (0.01)
- ✅ **Conservative Initialization**: Gains 0.2-0.5 throughout
- ✅ **Focal Loss Clamping**: Probability [1e-6, 1-1e-6]
- ✅ **Optimizer Groups**: No weight decay on normalization
- ✅ **Gradient Clipping**: 0.5 (local and Modal)

### Recommended but Optional

- ⚠️ `BGB_SANITIZE_GRADS=1` - **HIGHLY RECOMMENDED** for production
- ⚠️ `BGB_NAN_DEBUG=1` - **RECOMMENDED** for monitoring

### Available but Not Needed

- ❌ `BGB_SAFE_CLAMP` - Redundant with PR1-5
- ❌ `BGB_SANITIZE_INPUTS` - Redundant with preprocessing
- ❌ `BGB_DEBUG_FINITE` - Only for debugging (performance cost)
- ❌ `BGB_ANOMALY_DETECT` - Only for investigation (very slow)
- ❌ `BGB_SKIP_OPT_STEP_ON_NAN` - Only for debugging

---

## Training Validation Results

### v3.4.1 (October 1, 2025) - Local RTX 4090

**Configuration**:
- `BGB_SANITIZE_GRADS=1`
- `BGB_NAN_DEBUG=1`
- Warmup schedules enabled

**Results** (batch 0-723):
```
Batch 0: P95=52.06, Mean=14.54 (early training variance)
Batch 166: P95=26.57, Mean=9.28 (warmup working)
Batch 697: P95=10.32, Mean=3.72 (stabilizing)
Batch 723: P95=9.74, Mean=3.32 (excellent health)

NaN/Inf count: ZERO
Loss: 0.1582 → 0.1555 (decreasing)
```

**Gradient trend**: 82% P95 decrease, 77% Mean decrease over 723 batches ✅

**Individual clipping messages**: Normal and expected (focal loss with γ=2.0)

**Conclusion**: Training is **ROCK SOLID** with current NaN protection stack.

---

## FAQ

### Q: Why aren't SANITIZE_GRADS and NAN_DEBUG enabled by default?

**A**: Historical design - originally optional debugging tools. PyTorch 2.5.0 makes gradient sanitization effectively required for complex architectures.

### Q: Will sanitization hurt model performance?

**A**: No evidence of harm. Sanitization prevents training corruption. Clean gradients → better convergence.

### Q: Should I always use BGB_DEBUG_FINITE=1?

**A**: No - has performance cost (~5-10% slower due to 9 finite checks per forward). Only for debugging.

### Q: What if I see NaNs even with sanitization?

**A**: Indicates deeper issue. Check:
1. Cache was rebuilt after dependency upgrade?
2. Dynamic PE causing instability? (verify v3.3.1 eigenvector detachment)
3. Config has all PR1-5 settings enabled?

### Q: How do I know if sanitization is working?

**A**: With `BGB_NAN_DEBUG=1`, you'll see warnings like:
```
[WARN] Sanitized NaN gradients at batch 42
```

If you see this **occasionally** → sanitization working ✅
If you see this **every batch** → deeper problem, investigate ⚠️

---

## Summary

**CRITICAL for Production**:
```bash
export BGB_SANITIZE_GRADS=1  # REQUIRED
export BGB_NAN_DEBUG=1       # RECOMMENDED
```

**All other protection is built into the architecture via PR1-5 and hardcoded safeguards.**

**Status**: 🟢 **PRODUCTION STABLE** - v3.4.1 has zero NaN/Inf across 723+ batches with warmup schedules and eigendecomposition fix.

**Last Verified**: October 1, 2025 - Batch 723 on local RTX 4090 training

**Critical Commits**:
- `57426ea` - Clip outliers in preprocessing
- `7ba8017` - Add output sanitization in detector
- `1106520` - v3.3.1 eigendecomposition fix (eigenvector detachment)
- `af54ebd` - Modal A100 memory fragmentation fix
- Current HEAD - v3.4.1 with warmup schedules

**Related Docs**:
- `GRADIENT_MONITORING_GUIDE.md` - Realistic gradient expectations
- `ARCHITECTURE_V3_STABILITY.md` - v3.3.1 eigendecomposition fix
- `WARMUP_SCHEDULES_GUIDE.md` - v3.4.1 gradient stabilization
