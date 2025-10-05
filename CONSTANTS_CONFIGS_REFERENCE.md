# Constants vs Configs: Complete SSOT Reference

**Version**: v3.7.0
**Date**: October 5, 2025
**Purpose**: Authoritative decision matrix for EVERY constant and config value in Brain-Go-Brr v3
**Principle**: Single Source of Truth (SSOT) - Robert C. Martin's Clean Code + Google/DeepMind ML Best Practices

---

## 📋 Executive Summary

**Architectural Decision**: Following Google DeepMind and OpenAI best practices, we maintain TWO types of defaults:

1. **Config-Driven Defaults** (`src/brain_brr/config/schemas.py`)
   - User-tunable hyperparameters exposed in YAML configs
   - **SSOT**: Pydantic Field defaults in schema definitions
   - Examples: `focal_alpha`, `learning_rate`, `dropout`, `batch_size`

2. **Code-Guard Constants** (`src/brain_brr/constants.py`)
   - Safety defaults, numerical stability values, and clinical thresholds
   - NOT exposed in configs (hardcoded for safety/consistency)
   - Examples: `EPSILON_PROB_CLAMP`, `HYSTERESIS_TAU_ON`, `SAMPLING_RATE`

3. **Defensive Fallbacks** (Code-level safety when config is None/missing)
   - Hardcoded in function signatures as last resort
   - **MUST match corresponding constant** from `constants.py`
   - Examples: `alpha: float = GNN_SSGCONV_ALPHA_DEFAULT` (when graph_cfg is None)

---

## 🔍 AUDIT FINDINGS (October 5, 2025)

### ✅ CORRECTLY WIRED (No Issues)

| Constant/Config | Location | Usage | Status |
|----------------|----------|-------|--------|
| `EPSILON_PROB_CLAMP` | `constants.py:95` | Focal loss clamping | ✅ Used |
| `EPSILON_NUMERICAL` | `constants.py:99` | Schema bounds (`schemas.py:238,485`) | ✅ Used |
| `HYSTERESIS_TAU_ON` | `constants.py:123` | Schema default (`schemas.py:307`) | ✅ Config-driven |
| `FOCAL_ALPHA_PRODUCTION` | `constants.py:174` | Schema default (`schemas.py:476`) | ✅ Config-driven |
| `SAMPLING_RATE` | `constants.py:67` | Schema default + validation (`schemas.py:52,621`) | ✅ Used |
| `ECE_NUM_BINS` | `constants.py:280` | `metrics.py:39`, `val_step.py:148` | ✅ Used |
| `TAES_ALPHA_DEFAULT` | `constants.py:288` | `metrics.py:82` | ✅ Used |
| `BALANCED_SAMPLER_MAX_SAMPLE` | `constants.py:298` | `loop.py:695` | ✅ Used |
| `DATASET_DISTRIBUTION_SAMPLE_SIZE` | `constants.py:301` | `train_step.py:237` | ✅ Used |
| `PERCENTILE_P25/P50/P75/P95` | `constants.py:308-317` | `train_step.py:107,143` | ✅ Used |
| `format_sensitivity_key()` | `constants.py:344` | `loop.py:255,269,284`, `val_step.py:127,171` | ✅ Used |

---

### 🔴 GAPS IDENTIFIED (7 Total - Need Fixing)

**Summary**: 7 hardcoded literals found that should use constants from `constants.py`

#### 1. BALANCED_SAMPLER_SAMPLE_SIZE (Unused Constant)

**Location**: `constants.py:295`
```python
BALANCED_SAMPLER_SAMPLE_SIZE: int = 500
"""Number of windows to sample when checking seizure presence in balanced sampler."""
```

**Current Code** (`sampling.py:21`):
```python
def create_balanced_sampler(dataset: Any, sample_size: int = 500) -> ...
```

**Issue**: Hardcoded `500` instead of using constant

**Fix Required**:
```python
from src.brain_brr.constants import BALANCED_SAMPLER_SAMPLE_SIZE

def create_balanced_sampler(dataset: Any, sample_size: int = BALANCED_SAMPLER_SAMPLE_SIZE) -> ...
```

**Category**: **Code-Guard Constant** (NOT config-driven, defensive default)

---

#### 2. GNN_SSGCONV_ALPHA_DEFAULT (Defensive Fallback)

**Location**: `constants.py:324`
```python
GNN_SSGCONV_ALPHA_DEFAULT: float = 0.05
"""Default alpha mixing parameter for GNN SSGConv layer (when not specified in config)."""
```

**Current Code** (`gnn_pyg.py:55`):
```python
def __init__(
    self,
    ...
    alpha: float = 0.05,  # SSGConv alpha for EEG
    ...
):
```

**Architecture Context**:
- **Config-driven**: `schemas.py:210-212` has `alpha: float = Field(default=0.05)`
- **Used in practice**: `detector.py:587` passes `alpha=graph_cfg.alpha`
- **Fallback purpose**: If someone instantiates `GNNLaplacianPE()` directly (e.g., tests), use safe default

**Decision Required**:
- ✅ **OPTION A (Recommended)**: Wire constant into gnn_pyg.py for consistency
  ```python
  from src.brain_brr.constants import GNN_SSGCONV_ALPHA_DEFAULT

  def __init__(
      self,
      ...
      alpha: float = GNN_SSGCONV_ALPHA_DEFAULT,  # Defensive fallback
      ...
  ):
  ```
- ❌ **OPTION B**: Remove constant (rely only on schema default)
  - **Risk**: Tests/direct instantiation would fail without explicit alpha

**Category**: **Defensive Fallback** (duplicates schema but ensures safety)

---

#### 3. EIGENVALUE_CLAMP_MAX (Unused Constant)

**Location**: `constants.py:327`
```python
EIGENVALUE_CLAMP_MAX: float = 2.0
"""Maximum eigenvalue for Laplacian stability (prevents numerical overflow)."""
```

**Current Code** (`gnn_pyg.py:284`):
```python
eigenvalues = torch.clamp(eigenvalues, min=EPSILON_NUMERICAL, max=2.0)
```

**Issue**: Hardcoded `2.0` instead of using constant

**Fix Required**:
```python
from src.brain_brr.constants import EIGENVALUE_CLAMP_MAX

eigenvalues = torch.clamp(eigenvalues, min=EPSILON_NUMERICAL, max=EIGENVALUE_CLAMP_MAX)
```

**Category**: **Code-Guard Constant** (numerical stability, NOT config-driven)

---

#### 4. LAYERSCALE_ALPHA_FALLBACK (Defensive Fallback - Unused)

**Location**: `constants.py:334`
```python
LAYERSCALE_ALPHA_FALLBACK: float = 0.1
"""Fallback LayerScale alpha when config is missing (defensive default)."""
```

**Current Code** (`node_stream.py:30`, `edge_stream.py:65`):
```python
layerscale_init = float(norms_cfg.layerscale_alpha if norms_cfg else 0.1)
```

**Architecture Context**:
- **Config-driven**: `schemas.py:242-243` has `layerscale_alpha: float = Field(default=0.1)`
- **Used in practice**: All configs have `norms.layerscale_alpha: 0.1`
- **Fallback purpose**: If `norms_cfg` is None (old configs, tests), use safe default

**Decision Required**:
- ✅ **OPTION A (Recommended)**: Wire constant for consistency
  ```python
  from src.brain_brr.constants import LAYERSCALE_ALPHA_FALLBACK

  layerscale_init = float(norms_cfg.layerscale_alpha if norms_cfg else LAYERSCALE_ALPHA_FALLBACK)
  ```
- ❌ **OPTION B**: Remove constant (rely only on schema default)
  - **Risk**: Defensive fallback would be hardcoded `0.1` instead of centralized

**Category**: **Defensive Fallback** (duplicates schema but ensures safety)

---

#### 5. MORPHOLOGY_OPENING_KERNEL & MORPHOLOGY_CLOSING_KERNEL (Unused Constants)

**Location**: `constants.py:154-155`
```python
MORPHOLOGY_OPENING_KERNEL: int = 11
"""Opening kernel size..."""

MORPHOLOGY_CLOSING_KERNEL: int = 31
"""Closing kernel size..."""
```

**Current Code** (`postprocess.py:128-129`):
```python
def apply_morphology(
    masks: torch.Tensor,
    opening_kernel: int = 11,
    closing_kernel: int = 31,
    ...
):
```

**Issue**: Hardcoded `11` and `31` instead of using constants

**Fix Required**:
```python
from src.brain_brr.constants import MORPHOLOGY_OPENING_KERNEL, MORPHOLOGY_CLOSING_KERNEL

def apply_morphology(
    masks: torch.Tensor,
    opening_kernel: int = MORPHOLOGY_OPENING_KERNEL,
    closing_kernel: int = MORPHOLOGY_CLOSING_KERNEL,
    ...
):
```

**Category**: **Code-Guard Constant** (defensive fallback, config overrides)

---

#### 6. EDGE_INPUT_CLAMP_MAX (Missing Constant - Optional Fix)

**Current Code** (`detector.py:363`):
```python
edge_in = torch.clamp(edge_in, -3.0, 3.0)
```

**Issue**: Hardcoded `3.0` clamp range (3σ safety for normalized features)

**Decision Required**:
- **OPTION A**: Create constant `EDGE_INPUT_CLAMP_MAX = 3.0` and wire it
- **OPTION B**: Leave as-is (rarely-used fallback path when `edge_lift_activation: none`)

**Recommendation**: OPTION A for completeness (centralizes all magic numbers)

**Category**: **Code-Guard Constant** (safety clamp for edge features)

---

## 📊 Complete Categorization Matrix

### Category 1: Config-Driven (Schema = SSOT)

| Value | Schema Location | Config Location | Code Usage | Notes |
|-------|-----------------|-----------------|------------|-------|
| `focal_alpha` | `schemas.py:476` (default=0.5) | `train.yaml:149` | `loop.py` via config | Config overrides |
| `focal_gamma` | `schemas.py:482` (default=2.0) | `train.yaml:150` | `loop.py` via config | Config overrides |
| `learning_rate` | `schemas.py:485` (default=3e-4) | `train.yaml:132` | `loop.py` via config | Config overrides |
| `batch_size` | `schemas.py:470` (default=16) | `train.yaml:128` | `loop.py` via config | Config overrides |
| `dropout_mamba` | `schemas.py:124` (default=0.1) | `train.yaml:58` | `mamba.py` via config | Config overrides |
| `dropout_tcn` | `schemas.py:136` (default=0.15) | `train.yaml:48` | `tcn.py` via config | Config overrides |
| `graph.alpha` | `schemas.py:210` (default=0.05) | `train.yaml:92` | `detector.py:587` | Config overrides |
| `norms.layerscale_alpha` | `schemas.py:242` (default=0.1) | `train.yaml:38` | `node_stream.py:30` | Config overrides |
| `hysteresis.tau_on` | `schemas.py:307` (default=0.86) | `train.yaml:189` | `postprocess.py` via config | Config overrides |
| `hysteresis.tau_off` | `schemas.py:309` (default=0.78) | `train.yaml:190` | `postprocess.py` via config | Config overrides |

**Decision**: ✅ **Keep in schemas ONLY** - These are user-tunable, configs are SSOT

---

### Category 2: Code-Guard Constants (Never in Configs)

| Constant | Location | Purpose | Usage | Notes |
|----------|----------|---------|-------|-------|
| `EPSILON_PROB_CLAMP` | `constants.py:95` | Focal loss stability | `train_step.py` | 1e-7 for [0,1] domain |
| `EPSILON_NUMERICAL` | `constants.py:99` | General stability | `schemas.py:238,485`, `gnn_pyg.py` | 1e-6 standard |
| `EPSILON_ZERO_CHECK` | `constants.py:103` | Zero detection | `sampling.py:68` | 1e-8 coarse check |
| `EPSILON_NORM` | `constants.py:106` | LayerNorm eps | `schemas.py:237` | PyTorch default |
| `EPSILON_LAPLACIAN` | `constants.py:110` | Graph stability | `schemas.py:161,197` | 1e-4 for eigendecomp |
| `SAMPLING_RATE` | `constants.py:67` | EEG standard | `schemas.py:52,621` | 256 Hz fixed |
| `N_CHANNELS` | `constants.py:64` | 10-20 montage | `schemas.py:55,625` | 19 channels fixed |
| `WINDOW_SIZE_SEC` | `constants.py:68` | Clinical standard | `schemas.py:57,629` | 60s fixed |
| `STRIDE_SIZE_SEC` | `constants.py:69` | Clinical standard | `schemas.py:59,633` | 10s fixed |
| `CHANNEL_NAMES_10_20` | `constants.py:31` | Spatial order | `io.py` | CRITICAL contract |
| `ECE_NUM_BINS` | `constants.py:280` | Calibration bins | `metrics.py:39`, `val_step.py:148` | Guo et al. 2017 |
| `TAES_ALPHA_DEFAULT` | `constants.py:288` | FA penalty weight | `metrics.py:82` | 0.15 from TUSZ |
| `BALANCED_SAMPLER_MAX_SAMPLE` | `constants.py:298` | Safety limit | `loop.py:695` | ✅ 20000 max |
| `DATASET_DISTRIBUTION_SAMPLE_SIZE` | `constants.py:301` | Sampling size | `train_step.py:237` | ✅ 100 windows |
| `PERCENTILE_P25/P50/P75/P95` | `constants.py:308-317` | Gradient stats | `train_step.py:107,143` | ✅ Used |
| **🔴 BALANCED_SAMPLER_SAMPLE_SIZE** | `constants.py:295` | Default sampling | `sampling.py:21` | ❌ **NOT USED** (hardcoded 500) |
| **🔴 EIGENVALUE_CLAMP_MAX** | `constants.py:327` | Laplacian clamp | `gnn_pyg.py:284` | ❌ **NOT USED** (hardcoded 2.0) |
| **🔴 MORPHOLOGY_OPENING_KERNEL** | `constants.py:154` | Opening kernel | `postprocess.py:128` | ❌ **NOT USED** (hardcoded 11) |
| **🔴 MORPHOLOGY_CLOSING_KERNEL** | `constants.py:155` | Closing kernel | `postprocess.py:129` | ❌ **NOT USED** (hardcoded 31) |
| **🔴 EDGE_INPUT_CLAMP_MAX** | N/A (missing) | Edge clamp | `detector.py:363` | ❌ **NOT DEFINED** (hardcoded 3.0) |

**Decision**: ✅ **Keep in constants.py** - These are safety/clinical standards, never user-tunable

---

### Category 3: Defensive Fallbacks (Duplicates Schema for Safety)

| Constant | Schema Default | Code Fallback | Purpose | Status |
|----------|----------------|---------------|---------|--------|
| **🔴 GNN_SSGCONV_ALPHA_DEFAULT** | `schemas.py:210` (0.05) | `gnn_pyg.py:55` (0.05) | Direct instantiation safety | ❌ **Hardcoded, should use constant** |
| **🔴 LAYERSCALE_ALPHA_FALLBACK** | `schemas.py:242` (0.1) | `node_stream.py:30`, `edge_stream.py:65` (0.1) | Missing norms_cfg safety | ❌ **Hardcoded, should use constant** |

**Decision**:
- ✅ **OPTION A (RECOMMENDED)**: Wire constants into code for consistency
  - Ensures fallbacks match constants (SSOT)
  - Defensive programming best practice (Google/DeepMind style)
  - Example: `alpha: float = GNN_SSGCONV_ALPHA_DEFAULT`

- ❌ **OPTION B**: Remove constants, rely only on schemas
  - Risk: Hardcoded fallbacks in code diverge from schema
  - Not recommended for production codebases

---

## 🛠️ RECOMMENDED FIXES

### Fix 1: sampling.py - Wire BALANCED_SAMPLER_SAMPLE_SIZE

**File**: `src/brain_brr/train/sampling.py:21`

**BEFORE**:
```python
def create_balanced_sampler(dataset: Any, sample_size: int = 500) -> ...
```

**AFTER**:
```python
from src.brain_brr.constants import BALANCED_SAMPLER_SAMPLE_SIZE

def create_balanced_sampler(dataset: Any, sample_size: int = BALANCED_SAMPLER_SAMPLE_SIZE) -> ...
```

---

### Fix 2: gnn_pyg.py - Wire GNN_SSGCONV_ALPHA_DEFAULT

**File**: `src/brain_brr/models/gnn_pyg.py:55`

**BEFORE**:
```python
def __init__(
    self,
    ...
    alpha: float = 0.05,  # SSGConv alpha for EEG
    ...
):
```

**AFTER**:
```python
from src.brain_brr.constants import GNN_SSGCONV_ALPHA_DEFAULT

def __init__(
    self,
    ...
    alpha: float = GNN_SSGCONV_ALPHA_DEFAULT,  # Defensive fallback (config overrides)
    ...
):
```

---

### Fix 3: gnn_pyg.py - Wire EIGENVALUE_CLAMP_MAX

**File**: `src/brain_brr/models/gnn_pyg.py:284`

**BEFORE**:
```python
eigenvalues = torch.clamp(eigenvalues, min=EPSILON_NUMERICAL, max=2.0)
```

**AFTER**:
```python
from src.brain_brr.constants import EIGENVALUE_CLAMP_MAX

eigenvalues = torch.clamp(eigenvalues, min=EPSILON_NUMERICAL, max=EIGENVALUE_CLAMP_MAX)
```

---

### Fix 4: node_stream.py - Wire LAYERSCALE_ALPHA_FALLBACK

**File**: `src/brain_brr/models/builders/node_stream.py:30`

**BEFORE**:
```python
layerscale_init = float(norms_cfg.layerscale_alpha if norms_cfg else 0.1)
```

**AFTER**:
```python
from src.brain_brr.constants import LAYERSCALE_ALPHA_FALLBACK

layerscale_init = float(norms_cfg.layerscale_alpha if norms_cfg else LAYERSCALE_ALPHA_FALLBACK)
```

---

### Fix 5: edge_stream.py - Wire LAYERSCALE_ALPHA_FALLBACK

**File**: `src/brain_brr/models/builders/edge_stream.py:65`

**BEFORE**:
```python
layerscale_init = float(norms_cfg.layerscale_alpha if norms_cfg else 0.1)
```

**AFTER**:
```python
from src.brain_brr.constants import LAYERSCALE_ALPHA_FALLBACK

layerscale_init = float(norms_cfg.layerscale_alpha if norms_cfg else LAYERSCALE_ALPHA_FALLBACK)
```

---

### Fix 6: postprocess.py - Wire MORPHOLOGY Constants

**File**: `src/brain_brr/post/postprocess.py:128-129`

**BEFORE**:
```python
def apply_morphology(
    masks: torch.Tensor,
    opening_kernel: int = 11,
    closing_kernel: int = 31,
    use_gpu: bool = False,
    kernel_size: int | None = None,
) -> torch.Tensor:
```

**AFTER**:
```python
from src.brain_brr.constants import MORPHOLOGY_OPENING_KERNEL, MORPHOLOGY_CLOSING_KERNEL

def apply_morphology(
    masks: torch.Tensor,
    opening_kernel: int = MORPHOLOGY_OPENING_KERNEL,
    closing_kernel: int = MORPHOLOGY_CLOSING_KERNEL,
    use_gpu: bool = False,
    kernel_size: int | None = None,
) -> torch.Tensor:
```

**Category**: **Code-Guard Constant** (defensive fallback, config overrides)

---

### Fix 7: detector.py - Edge Input Clamp (Optional)

**File**: `src/brain_brr/models/detector.py:363`

**CURRENT**:
```python
edge_in = torch.clamp(edge_in, -3.0, 3.0)
```

**Issue**: Hardcoded `3.0` clamp range (3σ safety for normalized features)

**Decision Required**:
- **OPTION A**: Create constant and wire it
  ```python
  # In constants.py
  EDGE_INPUT_CLAMP_MAX: float = 3.0
  """Maximum edge input range (±3σ for normalized features)."""

  # In detector.py
  from src.brain_brr.constants import EDGE_INPUT_CLAMP_MAX
  edge_in = torch.clamp(edge_in, -EDGE_INPUT_CLAMP_MAX, EDGE_INPUT_CLAMP_MAX)
  ```
- **OPTION B**: Leave as-is (fallback path, may be deprecated in future)

**Recommendation**: Implement OPTION A for completeness (currently this is in a rarely-used fallback path when `edge_lift_activation: none`)

---

## ✅ VALIDATION CHECKLIST

After implementing all fixes:

```bash
# 1. Verify no hardcoded fallbacks remain
rg '\bsample_size: int = 500\b' src/brain_brr/train/sampling.py  # Should fail ✅
rg '\balpha: float = 0\.05\b' src/brain_brr/models/gnn_pyg.py  # Should fail ✅
rg 'max=2\.0\b' src/brain_brr/models/gnn_pyg.py  # Should fail ✅
rg 'else 0\.1\b' src/brain_brr/models/builders/  # Should fail ✅
rg 'opening_kernel: int = 11\b' src/brain_brr/post/postprocess.py  # Should fail ✅
rg 'closing_kernel: int = 31\b' src/brain_brr/post/postprocess.py  # Should fail ✅
rg '\-3\.0, 3\.0\b' src/brain_brr/models/detector.py  # Should fail if fix #7 applied ✅

# 2. Verify constants are imported
rg 'BALANCED_SAMPLER_SAMPLE_SIZE' src/brain_brr/train/sampling.py  # Should match ✅
rg 'GNN_SSGCONV_ALPHA_DEFAULT' src/brain_brr/models/gnn_pyg.py  # Should match ✅
rg 'EIGENVALUE_CLAMP_MAX' src/brain_brr/models/gnn_pyg.py  # Should match ✅
rg 'LAYERSCALE_ALPHA_FALLBACK' src/brain_brr/models/builders/  # Should match ✅
rg 'MORPHOLOGY_OPENING_KERNEL' src/brain_brr/post/postprocess.py  # Should match ✅
rg 'MORPHOLOGY_CLOSING_KERNEL' src/brain_brr/post/postprocess.py  # Should match ✅

# 3. Run quality checks
make q  # Lint, format, type checking

# 4. Run targeted tests
.venv/bin/pytest tests/unit/train/ tests/unit/models/ -v

# 5. Smoke test end-to-end
make s  # 1 epoch, 3 files
```

---

## 📈 IMPACT ANALYSIS

### Before Fixes (Current State)
- ❌ **5 hardcoded literals** instead of constants
- ❌ **2 unused constants** (defined but not imported)
- ❌ **SSOT violations**: Fallback values duplicated in code

### After Fixes (Target State)
- ✅ **100% constant usage** - All values centralized
- ✅ **Zero hardcoded fallbacks** - SSOT enforced
- ✅ **Clear separation**:
  - Config schemas = User-tunable SSOT
  - Constants = Safety defaults SSOT
  - Code = Uses both appropriately

---

## 🎯 ARCHITECTURAL PRINCIPLES (Clean Code + ML Best Practices)

### 1. Config-Driven Development (Google/DeepMind)
- **User-tunable hyperparameters** → Pydantic schemas with `Field(default=...)`
- **YAML configs override schemas** for experiments
- **Code receives validated config objects**, trusts schema defaults

### 2. Code-Guard Constants (Robert C. Martin)
- **Numerical stability values** → Never exposed to users (safety-critical)
- **Clinical thresholds** → Validated by domain experts, not tunable
- **File/metric naming** → Consistency across codebase

### 3. Defensive Fallbacks (Production ML)
- **Function signature defaults** → Match constants for tests/direct use
- **Config-None handling** → Graceful degradation, not hard failure
- **Example**: `alpha: float = GNN_SSGCONV_ALPHA_DEFAULT` ensures tests work without config

---

## 🚀 FINAL DECISION SUMMARY

### ✅ APPROVED APPROACH

**For values in BOTH schema AND code**:
1. **Schema defines config default** (SSOT for user-facing value)
2. **Constant defines code fallback** (SSOT for defensive default)
3. **Code imports constant** for function signatures
4. **Config always wins** when present

**Example (LayerScale Alpha)**:
- Schema: `layerscale_alpha: float = Field(default=0.1)` ← User can tune
- Constant: `LAYERSCALE_ALPHA_FALLBACK = 0.1` ← Code safety net
- Code: `layerscale_init = float(norms_cfg.layerscale_alpha if norms_cfg else LAYERSCALE_ALPHA_FALLBACK)`
- Result: Config overrides constant, constant overrides hardcoded `0.1`

**Why This Works**:
- ✅ Single source of truth for each context (config vs code)
- ✅ Defensive programming (tests/minimal configs still work)
- ✅ No magic numbers (all values documented + centralized)
- ✅ Professional ML codebase standard (OpenAI/Google/DeepMind style)

---

## 📝 FILES REQUIRING CHANGES

| File | Lines | Change Type | Risk |
|------|-------|-------------|------|
| `src/brain_brr/train/sampling.py` | 21 | Add import + use constant | Low |
| `src/brain_brr/models/gnn_pyg.py` | 55, 284 | Add imports + use constants | Low |
| `src/brain_brr/models/builders/node_stream.py` | 30 | Add import + use constant | Low |
| `src/brain_brr/models/builders/edge_stream.py` | 65 | Add import + use constant | Low |
| `src/brain_brr/post/postprocess.py` | 128-129 | Add imports + use constants | Low |
| `src/brain_brr/models/detector.py` | 363 | Create constant + use it (optional) | Low |
| `src/brain_brr/constants.py` | N/A | Add `EDGE_INPUT_CLAMP_MAX = 3.0` (optional) | None |

**Total Changes**: 7 fixes across 5-6 files (6th/7th optional but recommended)

---

## 🔄 NEXT STEPS (After Consensus Validation)

1. ✅ **Get AI consensus** on this document (Task tool with general-purpose agent)
2. ✅ **Implement 5 fixes** (wire constants into code)
3. ✅ **Update root docs** (POLISH_ITEMS.md, DEBT_STATUS_TRUE.md reflect reality)
4. ✅ **Run validation suite** (make q + tests)
5. ✅ **Smoke test** (make s - 1 epoch, 3 files)
6. ✅ **Production ready** - Deploy to A100-80GB

---

**Document Status**: 📋 **AWAITING CONSENSUS VALIDATION**
**Last Updated**: October 5, 2025
**Audit Completed**: 100% codebase coverage (constants.py + schemas.py + all usage sites)
**Verification**: Line-by-line code reads + grep searches for all values
