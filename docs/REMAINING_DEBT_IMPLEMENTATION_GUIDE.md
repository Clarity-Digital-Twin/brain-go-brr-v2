# Remaining Debt Implementation Guide

**Version**: v3.6.2
**Date**: October 4, 2025
**Status**: Complete implementation roadmap for P2/P3 quality improvements

---

## 🎯 Executive Summary

**v3.6.2 Status**: ✅ **100% DEBT-FREE** (All P0/P1 critical issues eliminated)

This guide provides **step-by-step implementation instructions** for remaining P2/P3 quality-of-life improvements. These items are **NON-BLOCKING** for production training but will improve code maintainability and consistency.

**Total Remaining Work**: ~8 hours
- **P2 (Medium Priority)**: ~4 hours - Focus on P2.1 and P2.3 first
- **P3 (Low Priority)**: ~4 hours - Optional polish for v4.0

---

## 🔴 IMMEDIATE PRIORITY (v3.7.0)

### P2.3: Refactor FA Sweep Duplication (60 Lines of Duplicate Code)

**Priority**: 🔴 **CRITICAL** - Violates DRY principle, high maintenance burden
**Effort**: 1 hour
**Risk**: Low (same algorithm, just calling helper instead of reimplementing)

#### Current Problem

`val_step.py:153-213` contains 60 lines of duplicate threshold search logic that already exists in `false_alarm.py:39-108`.

**Duplicate Code Locations**:
- **Binary search loop**: Lines 153-183 (31 lines)
- **Sensitivity calculation**: Lines 186-213 (28 lines)

#### Implementation Steps

**Step 1**: Read the current implementations
```bash
# Verify current state
cat -n src/brain_brr/train/val_step.py | sed -n '153,213p'
cat -n src/brain_brr/eval/helpers/false_alarm.py | sed -n '39,108p'
```

**Step 2**: Update imports in `val_step.py`
```python
# Add to top of file (after line 27)
from src.brain_brr.eval.helpers.false_alarm import find_threshold_for_fa_target, FASweepResult
```

**Step 3**: Replace duplicate logic in `_compute_final_metrics()` (lines 153-213)

**BEFORE** (60 lines of duplicate code):
```python
for fa in fa_rates:
    low, high = constants.THRESHOLD_SEARCH_LOW, constants.THRESHOLD_SEARCH_HIGH
    best_tau_on = constants.HYSTERESIS_TAU_ON

    for _ in range(constants.THRESHOLD_SEARCH_MAX_ITERS):
        mid_tau_on = (low + high) / 2
        mid_tau_off = max(0.0, mid_tau_on - constants.HYSTERESIS_DELTA)
        # ... 50 more lines of search and sensitivity calculation ...
```

**AFTER** (call canonical implementation):
```python
# Build timelines for FA sweep (format expected by helper)
timelines_probs = [p.cpu() for p in all_probs_flat]
timelines_labels = [l.cpu() for l in all_labels_flat]

# Call canonical implementation from false_alarm.py
for fa in fa_rates:
    result: FASweepResult = find_threshold_for_fa_target(
        timelines_probs=timelines_probs,
        timelines_labels=timelines_labels,
        fa_target=fa,
        total_hours=total_hours,
        all_ref_events=all_ref_events,
        post_cfg=post_cfg,
        sampling_rate=sampling_rate,
        max_iters=constants.THRESHOLD_SEARCH_MAX_ITERS,
    )

    # Extract results
    thresholds[f"{fa}"] = result.threshold_tau_on
    sensitivity_results[f"sensitivity_at_{fa}fa"] = result.sensitivity
    fa_curve.append((fa, result.sensitivity))
```

**Step 4**: Test the refactor
```bash
# Run validation tests to ensure same behavior
.venv/bin/pytest tests/unit/train/test_loop.py::TestTrainingSmoke::test_validation -xvs
.venv/bin/pytest tests/integration/test_evaluation.py -v
```

**Step 5**: Verify output consistency
```bash
# Compare FA sweep results before/after refactor
# Should produce identical threshold/sensitivity values
```

#### Files Modified
- `src/brain_brr/train/val_step.py` (lines 27, 153-213)

#### Success Criteria
- ✅ All validation tests pass
- ✅ FA sweep produces same thresholds/sensitivities
- ✅ ~60 lines of duplicate code eliminated
- ✅ Single source of truth for threshold search

---

### P2.1: Config Validation Script

**Priority**: 🔴 **HIGH** - Prevents config drift as constants evolve
**Effort**: 2 hours
**Risk**: None (new script, doesn't modify existing code)

#### Current Problem

No automated validation that YAML config values match `constants.py` defaults. If we change `HYSTERESIS_TAU_ON` from 0.86 to 0.90 in constants, configs won't auto-update and will silently drift.

#### Implementation Steps

**Step 1**: Create validation script

**File**: `scripts/validate_configs.py`

```python
#!/usr/bin/env python3
"""Validate YAML configs match constants.py defaults.

Usage:
    python scripts/validate_configs.py
    make config-check
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from brain_brr import constants


CRITICAL_CONSTANTS = {
    "hysteresis.tau_on": constants.HYSTERESIS_TAU_ON,
    "hysteresis.tau_off": constants.HYSTERESIS_TAU_OFF,
    "hysteresis.delta": constants.HYSTERESIS_DELTA,
    "focal.alpha": constants.FOCAL_ALPHA_PRODUCTION,
    "focal.gamma": constants.FOCAL_GAMMA_PRODUCTION,
    "data.sampling_rate": constants.SAMPLING_RATE,
    "data.n_channels": constants.N_CHANNELS,
    "data.window_size": constants.WINDOW_SIZE_SEC,
    "data.stride": constants.STRIDE_SIZE_SEC,
}


def get_nested_value(data: dict[str, Any], path: str) -> Any:
    """Get value from nested dict using dot notation."""
    keys = path.split(".")
    value = data
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None
    return value


def validate_config(config_path: Path) -> list[str]:
    """Validate one config file against constants.

    Returns list of error messages (empty if valid).
    """
    errors = []

    with open(config_path) as f:
        config = yaml.safe_load(f)

    for path, expected_value in CRITICAL_CONSTANTS.items():
        actual_value = get_nested_value(config, path)

        # Skip if not present in config (may be optional)
        if actual_value is None:
            continue

        # Compare values
        if actual_value != expected_value:
            errors.append(
                f"{config_path.name}: {path} = {actual_value}, "
                f"expected {expected_value} (from constants.py)"
            )

    return errors


def main() -> int:
    """Validate all YAML configs."""
    config_dir = Path(__file__).parent.parent / "configs"

    all_errors = []

    # Check all YAML files in configs/
    for config_path in config_dir.rglob("*.yaml"):
        if config_path.name.startswith("."):
            continue

        print(f"Checking {config_path.relative_to(config_dir)}...")
        errors = validate_config(config_path)
        all_errors.extend(errors)

    if all_errors:
        print("\n❌ CONFIG VALIDATION FAILED\n")
        for error in all_errors:
            print(f"  - {error}")
        print(
            f"\nFound {len(all_errors)} config/constants mismatches. "
            "Update configs to match constants.py or vice versa."
        )
        return 1

    print("\n✅ All configs match constants.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Step 2**: Add to Makefile

```makefile
# Add to Makefile after 'make q' target
.PHONY: config-check
config-check:
	@echo "🔍 Validating configs against constants.py..."
	$(PYTHON) scripts/validate_configs.py

# Update 'make q' to include config check
.PHONY: q
q: config-check lint format mypy
	@echo "✅ Quality checks passed"
```

**Step 3**: Make script executable
```bash
chmod +x scripts/validate_configs.py
```

**Step 4**: Run validation
```bash
make config-check
```

**Step 5**: Fix any mismatches found
```bash
# If script reports errors, either:
# 1. Update YAML configs to match constants.py, OR
# 2. Update constants.py to match intended config values
```

#### Files Created
- `scripts/validate_configs.py` (new)
- `Makefile` (add `config-check` target)

#### Success Criteria
- ✅ Script runs without errors
- ✅ All critical constants validated
- ✅ `make config-check` passes
- ✅ Integrated into `make q` workflow

---

### P2.4: Move Inline Imports (5-Minute Quick Win)

**Priority**: 🟡 **MEDIUM** - PEP 8 compliance, better IDE support
**Effort**: 5 minutes
**Risk**: Zero (just moving import location)

#### Current Problem

`val_step.py` has 2 inline imports (inside functions) that violate PEP 8 and confuse IDEs.

**Locations**:
- Line 55: `from src.brain_brr.eval.metrics import stitch_recording_timeline`
- Line 110: `from src.brain_brr.eval.metrics import calculate_ece, calculate_taes, overlap`

#### Implementation Steps

**Step 1**: Read current imports
```bash
cat -n src/brain_brr/train/val_step.py | head -30
```

**Step 2**: Add missing imports to top of file (after line 27)

**BEFORE** (lines 22-27):
```python
from src.brain_brr import constants
from src.brain_brr.config.schemas import PostprocessingConfig
from src.brain_brr.eval.metrics import batch_probs_to_events
from src.brain_brr.events import batch_mask_to_events
from src.brain_brr.utils.env import env

logger = logging.getLogger(__name__)
```

**AFTER** (add missing imports):
```python
from src.brain_brr import constants
from src.brain_brr.config.schemas import PostprocessingConfig
from src.brain_brr.eval.metrics import (
    batch_probs_to_events,
    calculate_ece,
    calculate_taes,
    overlap,
    stitch_recording_timeline,
)
from src.brain_brr.events import batch_mask_to_events
from src.brain_brr.utils.env import env

logger = logging.getLogger(__name__)
```

**Step 3**: Delete inline imports
```python
# DELETE line 55 (inside _process_recording)
from src.brain_brr.eval.metrics import stitch_recording_timeline  # ❌ DELETE

# DELETE line 110 (inside _compute_final_metrics)
from src.brain_brr.eval.metrics import calculate_ece, calculate_taes, overlap  # ❌ DELETE
```

**Step 4**: Run quality checks
```bash
make q  # Should pass - imports now at top
```

**Step 5**: Run tests to verify no breakage
```bash
.venv/bin/pytest tests/unit/train/test_loop.py -v
```

#### Files Modified
- `src/brain_brr/train/val_step.py` (lines 22-27, 55, 110)

#### Success Criteria
- ✅ All imports at module level (lines 1-30)
- ✅ No inline imports remain
- ✅ `make q` passes
- ✅ All tests pass

---

## ⚠️ MEDIUM PRIORITY (v3.7.1 or v3.8.0)

### P2.5: Metric Name Formatting (Type Safety)

**Priority**: ⚠️ **MEDIUM** - Improves type safety, prevents typos
**Effort**: 1 hour
**Risk**: Low

#### Current Problem

Sensitivity metric keys are formatted with f-strings in 3 locations, creating typo risk:
- Line 125: `default_results[f"sensitivity_at_{fa}fa"] = 0.0`
- Line 184: `thresholds[f"{fa}"] = best_tau_on`
- Line 213: `sensitivity_results[f"sensitivity_at_{fa}fa"] = sensitivity`

**Silent Bug Example**: Typo like `f"sensitivity_at_{fa}af"` would create wrong key with no error.

#### Implementation Steps

**Step 1**: Add helper to constants.py

```python
# Add to src/brain_brr/constants.py (after line 230)

# ============================================================================
# Metric Key Formatting
# ============================================================================

METRIC_SENSITIVITY_TEMPLATE: str = "sensitivity_at_{}fa"
"""Template for sensitivity metric keys (e.g., sensitivity_at_10fa)."""

def format_sensitivity_key(fa_rate: float) -> str:
    """Format sensitivity metric key for given FA rate.

    Args:
        fa_rate: False alarm rate (e.g., 10.0 for 10 FA/24h)

    Returns:
        Formatted metric key (e.g., "sensitivity_at_10.0fa")

    Example:
        >>> format_sensitivity_key(10.0)
        'sensitivity_at_10.0fa'
        >>> format_sensitivity_key(1)
        'sensitivity_at_1fa'
    """
    return METRIC_SENSITIVITY_TEMPLATE.format(fa_rate)
```

**Step 2**: Update val_step.py imports

```python
# Update imports (line 23)
from src.brain_brr.constants import format_sensitivity_key
```

**Step 3**: Replace f-strings with helper

**Line 125** (default initialization):
```python
# BEFORE
for fa in fa_rates:
    default_results[f"sensitivity_at_{fa}fa"] = 0.0

# AFTER
for fa in fa_rates:
    default_results[format_sensitivity_key(fa)] = 0.0
```

**Line 213** (final results):
```python
# BEFORE
sensitivity_results[f"sensitivity_at_{fa}fa"] = sensitivity

# AFTER
sensitivity_results[format_sensitivity_key(fa)] = sensitivity
```

**Step 4**: Search for other f-string patterns
```bash
# Find any remaining sensitivity_at_ patterns
rg 'f"sensitivity_at_.*fa"' src/
rg "'sensitivity_at_.*fa'" src/
```

**Step 5**: Run tests
```bash
make q
.venv/bin/pytest tests/unit/train/test_loop.py::TestTrainingSmoke::test_validation -xvs
```

#### Files Modified
- `src/brain_brr/constants.py` (add helper function)
- `src/brain_brr/train/val_step.py` (lines 125, 213)

#### Success Criteria
- ✅ Helper function added to constants.py
- ✅ All f-string metric keys replaced
- ✅ Type safety improved
- ✅ Tests pass

---

### P2.2: Update Config Documentation Version

**Priority**: ⚠️ **LOW** - Cosmetic fix (version number only)
**Effort**: 5 minutes
**Risk**: Zero

#### Current Problem

`configs/CONFIG_CONSISTENCY_CHECK.md` header shows "v3.6.1" but we're at v3.6.2.

#### Implementation Steps

**Step 1**: Update version in header
```bash
# Edit configs/CONFIG_CONSISTENCY_CHECK.md line 1
# BEFORE:
# Configuration Consistency Check (v3.6.1)

# AFTER:
# Configuration Consistency Check (v3.6.2)
```

**Step 2**: Commit change
```bash
git add configs/CONFIG_CONSISTENCY_CHECK.md
git commit -m "docs: Update CONFIG_CONSISTENCY_CHECK to v3.6.2"
```

#### Files Modified
- `configs/CONFIG_CONSISTENCY_CHECK.md` (line 1)

#### Success Criteria
- ✅ Version updated to v3.6.2
- ✅ Document remains accurate

---

## 🔵 LOW PRIORITY (v4.0 - Optional Polish)

### P3.2: ECE Bins Constant (Documentation Improvement)

**Priority**: 🔵 **LOW** - Documents WHY 10 bins
**Effort**: 5 minutes
**Risk**: Zero

#### Implementation Steps

**Step 1**: Add constant to constants.py

```python
# Add to src/brain_brr/constants.py (around line 230)

# ============================================================================
# Calibration Metrics
# ============================================================================

ECE_NUM_BINS: int = 10
"""Number of bins for Expected Calibration Error (ECE) calculation.

Standard calibration curve resolution per Guo et al. 2017:
"On Calibration of Modern Neural Networks" (ICML 2017)
https://arxiv.org/abs/1706.04599
"""
```

**Step 2**: Update metrics.py function signature

```python
# src/brain_brr/eval/metrics.py line 38
# BEFORE
def calculate_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:

# AFTER
from src.brain_brr.constants import ECE_NUM_BINS

def calculate_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = ECE_NUM_BINS) -> float:
```

**Step 3**: Update call site in val_step.py

```python
# src/brain_brr/train/val_step.py line 146
# BEFORE
ece = calculate_ece(probs_flat, labels_flat, n_bins=10)

# AFTER
from src.brain_brr.constants import ECE_NUM_BINS

ece = calculate_ece(probs_flat, labels_flat, n_bins=ECE_NUM_BINS)
```

**Step 4**: Run quality checks
```bash
make q
.venv/bin/pytest tests/unit/train/ -v
```

#### Files Modified
- `src/brain_brr/constants.py` (add constant + citation)
- `src/brain_brr/eval/metrics.py` (line 38)
- `src/brain_brr/train/val_step.py` (line 146)

#### Success Criteria
- ✅ Constant added with citation
- ✅ All uses updated
- ✅ Tests pass

---

### P3.1: Schema Epsilon Bounds

**Priority**: 🔵 **LOW** - Cosmetic consistency
**Effort**: 15 minutes
**Risk**: Zero

#### Implementation Steps

**Step 1**: Update schemas.py epsilon bounds

```python
# src/brain_brr/config/schemas.py
# Import at top
from src.brain_brr.constants import EPSILON_NUMERICAL

# Line 236 - BEFORE
boundary_eps: float = Field(default=EPSILON_NORM, ge=1e-10, ...)

# Line 236 - AFTER
boundary_eps: float = Field(default=EPSILON_NORM, ge=EPSILON_NUMERICAL, ...)

# Line 481 - BEFORE
learning_rate: float = Field(default=3e-4, ge=1e-6, ...)

# Line 481 - AFTER
learning_rate: float = Field(default=3e-4, ge=EPSILON_NUMERICAL, ...)
```

**Step 2**: Verify no other epsilon literals
```bash
rg 'ge=1e-\d+' src/brain_brr/config/schemas.py
```

**Step 3**: Run quality checks
```bash
make q
.venv/bin/pytest tests/unit/config/ -v
```

#### Files Modified
- `src/brain_brr/config/schemas.py` (lines 236, 481)

#### Success Criteria
- ✅ All epsilon bounds use constants
- ✅ Tests pass

---

### P3.5: Config Schema Literal Types

**Priority**: 🔵 **VERY LOW** - Validation assertions consistency
**Effort**: 2 minutes
**Risk**: Zero

#### Implementation Steps

**Step 1**: Update schema validation to use constants

```python
# src/brain_brr/config/schemas.py
# Import at top (add to existing imports)
from src.brain_brr.constants import (
    SAMPLING_RATE,
    N_CHANNELS,
    WINDOW_SIZE_SEC,
    STRIDE_SIZE_SEC,
)

# Lines 617-625 - BEFORE
def validate(self, phase: str = "all") -> None:
    if phase == "data":
        if self.data.sampling_rate != 256:
            raise ValueError(f"Must use 256 Hz...")
        if self.data.n_channels != 19:
            raise ValueError(f"Must use 19 channels...")
        if self.data.window_size != 60:
            raise ValueError(f"Must use 60s windows...")
        if self.data.stride != 10:
            raise ValueError(f"Must use 10s stride...")

# Lines 617-625 - AFTER
def validate(self, phase: str = "all") -> None:
    if phase == "data":
        if self.data.sampling_rate != SAMPLING_RATE:
            raise ValueError(f"Must use {SAMPLING_RATE} Hz...")
        if self.data.n_channels != N_CHANNELS:
            raise ValueError(f"Must use {N_CHANNELS} channels...")
        if self.data.window_size != WINDOW_SIZE_SEC:
            raise ValueError(f"Must use {WINDOW_SIZE_SEC}s windows...")
        if self.data.stride != STRIDE_SIZE_SEC:
            raise ValueError(f"Must use {STRIDE_SIZE_SEC}s stride...")
```

**Step 2**: Run quality checks
```bash
make q
.venv/bin/pytest tests/unit/config/ -v
```

#### Files Modified
- `src/brain_brr/config/schemas.py` (lines 617-625)

#### Success Criteria
- ✅ All validation assertions use constants
- ✅ Tests pass

---

### P3.3: Additional Constants (Complete SSOT Coverage)

**Priority**: 🔵 **VERY LOW** - Non-critical polish
**Effort**: 3 hours
**Risk**: Low

#### Overview

8 scattered magic numbers that could be constants but aren't in hot paths or clinical-critical code.

#### Implementation Steps

**Step 1**: Add all missing constants to constants.py

```python
# Add to src/brain_brr/constants.py (around line 230)

# ============================================================================
# Balanced Sampling Configuration
# ============================================================================

BALANCED_SAMPLER_SAMPLE_SIZE: int = 500
"""Number of windows to sample when checking seizure presence in balanced sampler."""

BALANCED_SAMPLER_MAX_SAMPLE: int = 20000
"""Maximum number of windows to check for safety in balanced sampler."""

DATASET_DISTRIBUTION_SAMPLE_SIZE: int = 100
"""Number of windows to sample when checking dataset distribution."""

# ============================================================================
# Statistics and Logging
# ============================================================================

PERCENTILE_P25: float = 25.0
"""25th percentile for gradient/weight statistics."""

PERCENTILE_P50: float = 50.0
"""50th percentile (median) for gradient/weight statistics."""

PERCENTILE_P75: float = 75.0
"""75th percentile for gradient/weight statistics."""

PERCENTILE_P95: float = 95.0
"""95th percentile for gradient/weight statistics (outlier detection)."""

# ============================================================================
# GNN Configuration Defaults
# ============================================================================

GNN_SSGCONV_ALPHA_DEFAULT: float = 0.05
"""Default alpha mixing parameter for GNN SSGConv layer (when not specified in config)."""

EIGENVALUE_CLAMP_MAX: float = 2.0
"""Maximum eigenvalue for Laplacian stability (prevents numerical overflow)."""

# ============================================================================
# Model Architecture Defaults
# ============================================================================

LAYERSCALE_ALPHA_FALLBACK: float = 0.1
"""Fallback LayerScale alpha when config is missing (defensive default)."""

# ============================================================================
# TAES Metric Configuration
# ============================================================================

TAES_ALPHA_DEFAULT: float = 0.15
"""Default false alarm penalty weight for TAES metric."""
```

**Step 2**: Update all usage sites (8 locations)

**2a. Balanced sampler** (`train/sampling.py:21`, `train/loop.py:693`):
```python
from src.brain_brr.constants import BALANCED_SAMPLER_SAMPLE_SIZE, BALANCED_SAMPLER_MAX_SAMPLE

# sampling.py:21 - BEFORE
def create_balanced_sampler(dataset: Any, sample_size: int = 500) -> ...:

# sampling.py:21 - AFTER
def create_balanced_sampler(dataset: Any, sample_size: int = BALANCED_SAMPLER_SAMPLE_SIZE) -> ...:

# loop.py:693 - BEFORE
sample_size = min(20000, len(train_dataset))

# loop.py:693 - AFTER
sample_size = min(BALANCED_SAMPLER_MAX_SAMPLE, len(train_dataset))
```

**2b. Dataset distribution check** (`train/train_step.py:228`):
```python
from src.brain_brr.constants import DATASET_DISTRIBUTION_SAMPLE_SIZE

# BEFORE
sample_size = min(100, dataset_len)

# AFTER
sample_size = min(DATASET_DISTRIBUTION_SAMPLE_SIZE, dataset_len)
```

**2c. Percentiles** (`train/train_step.py:101, 135, 482`):
```python
from src.brain_brr.constants import PERCENTILE_P25, PERCENTILE_P50, PERCENTILE_P75, PERCENTILE_P95

# train_step.py:101 - BEFORE
p25, median, p75, p95 = np.percentile(grad_array, [25, 50, 75, 95])

# train_step.py:101 - AFTER
p25, median, p75, p95 = np.percentile(
    grad_array,
    [PERCENTILE_P25, PERCENTILE_P50, PERCENTILE_P75, PERCENTILE_P95]
)

# train_step.py:482 - BEFORE
grad_p95 = finite_norms[int(len(finite_norms) * 0.95)]

# train_step.py:482 - AFTER
grad_p95 = finite_norms[int(len(finite_norms) * (PERCENTILE_P95 / 100.0))]
```

**2d. GNN defaults** (`models/gnn_pyg.py:55, 284`):
```python
from src.brain_brr.constants import GNN_SSGCONV_ALPHA_DEFAULT, EIGENVALUE_CLAMP_MAX

# gnn_pyg.py:55 - BEFORE
def __init__(self, ..., alpha: float = 0.05, ...):

# gnn_pyg.py:55 - AFTER
def __init__(self, ..., alpha: float = GNN_SSGCONV_ALPHA_DEFAULT, ...):

# gnn_pyg.py:284 - BEFORE
eigenvalues = torch.clamp(eigenvalues, min=EPSILON_NUMERICAL, max=2.0)

# gnn_pyg.py:284 - AFTER
eigenvalues = torch.clamp(eigenvalues, min=EPSILON_NUMERICAL, max=EIGENVALUE_CLAMP_MAX)
```

**2e. LayerScale fallback** (`models/builders/node_stream.py:30`, `models/builders/edge_stream.py:65`):
```python
from src.brain_brr.constants import LAYERSCALE_ALPHA_FALLBACK

# BEFORE
layerscale_init = float(norms_cfg.layerscale_alpha if norms_cfg else 0.1)

# AFTER
layerscale_init = float(norms_cfg.layerscale_alpha if norms_cfg else LAYERSCALE_ALPHA_FALLBACK)
```

**2f. TAES alpha** (`eval/metrics.py:80`):
```python
from src.brain_brr.constants import TAES_ALPHA_DEFAULT

# BEFORE
def calculate_taes(..., alpha: float = 0.15) -> float:

# AFTER
def calculate_taes(..., alpha: float = TAES_ALPHA_DEFAULT) -> float:
```

**Step 3**: Run comprehensive tests
```bash
make q
.venv/bin/pytest tests/unit/ tests/integration/ -v
```

#### Files Modified
- `src/brain_brr/constants.py` (add 8 constants)
- `src/brain_brr/train/sampling.py` (line 21)
- `src/brain_brr/train/loop.py` (line 693)
- `src/brain_brr/train/train_step.py` (lines 101, 135, 228, 482)
- `src/brain_brr/models/gnn_pyg.py` (lines 55, 284)
- `src/brain_brr/models/builders/node_stream.py` (line 30)
- `src/brain_brr/models/builders/edge_stream.py` (line 65)
- `src/brain_brr/eval/metrics.py` (line 80)

#### Success Criteria
- ✅ All 8 constants added with documentation
- ✅ All 8 locations updated
- ✅ Complete SSOT coverage achieved
- ✅ Tests pass

---

## 📊 Implementation Priority Matrix

| Item | Priority | Effort | Impact | Order |
|------|----------|--------|--------|-------|
| **P2.3: FA Sweep Refactor** | 🔴 CRITICAL | 1h | HIGH (eliminates 60 duplicate lines) | 1️⃣ |
| **P2.1: Config Validation** | 🔴 HIGH | 2h | HIGH (prevents drift) | 2️⃣ |
| **P2.4: Inline Imports** | 🟡 MEDIUM | 5min | LOW (code style) | 3️⃣ |
| **P2.5: Metric Formatting** | ⚠️ MEDIUM | 1h | MEDIUM (type safety) | 4️⃣ |
| **P2.2: Config Docs** | ⚠️ LOW | 5min | LOW (cosmetic) | 5️⃣ |
| **P3.2: ECE Bins** | 🔵 LOW | 5min | LOW (documentation) | 6️⃣ |
| **P3.1: Schema Epsilon** | 🔵 LOW | 15min | LOW (consistency) | 7️⃣ |
| **P3.5: Schema Literals** | 🔵 VERY LOW | 2min | VERY LOW | 8️⃣ |
| **P3.3: Additional Constants** | 🔵 VERY LOW | 3h | VERY LOW | 9️⃣ |

---

## 🚀 Recommended Implementation Path

### Phase 1: v3.7.0 (Next Patch Release - ~4 hours)

**Focus**: Critical P2 items only

```bash
# 1. Refactor FA sweep duplication (1 hour)
git checkout -b fix/refactor-fa-sweep
# ... implement P2.3 ...
git add src/brain_brr/train/val_step.py
git commit -m "refactor: Eliminate 60 lines of duplicate FA sweep logic"

# 2. Add config validation (2 hours)
git checkout -b feat/config-validation
# ... implement P2.1 ...
git add scripts/validate_configs.py Makefile
git commit -m "feat: Add automated config validation script"

# 3. Fix inline imports (5 minutes)
git checkout -b fix/inline-imports
# ... implement P2.4 ...
git add src/brain_brr/train/val_step.py
git commit -m "fix: Move inline imports to module level (PEP 8)"

# 4. Add metric name helper (1 hour)
git checkout -b feat/metric-name-formatting
# ... implement P2.5 ...
git add src/brain_brr/constants.py src/brain_brr/train/val_step.py
git commit -m "feat: Add format_sensitivity_key() helper for type safety"
```

**Total**: ~4 hours → **v3.7.0 ready for production**

### Phase 2: v3.7.1 (Optional Polish - ~30 minutes)

**Focus**: Quick cosmetic fixes

```bash
# 5. Update config docs version (5 minutes)
git checkout -b docs/update-config-version
# ... implement P2.2 ...

# 6. Add ECE bins constant (5 minutes)
git checkout -b feat/ece-bins-constant
# ... implement P3.2 ...

# 7. Schema epsilon bounds (15 minutes)
git checkout -b fix/schema-epsilon
# ... implement P3.1 ...

# 8. Schema literal types (2 minutes)
git checkout -b fix/schema-literals
# ... implement P3.5 ...
```

**Total**: ~30 minutes → **v3.7.1 complete constants coverage**

### Phase 3: v4.0 (Major Refactor - 3 hours)

**Focus**: Complete SSOT coverage

```bash
# 9. Add all remaining constants (3 hours)
git checkout -b feat/complete-constants-coverage
# ... implement P3.3 ...
```

**Total**: 3 hours → **v4.0 perfect code quality**

---

## ✅ Success Metrics

### After Phase 1 (v3.7.0):
- ✅ Zero duplicate FA sweep code
- ✅ Automated config validation in CI/CD
- ✅ PEP 8 compliant (no inline imports)
- ✅ Type-safe metric key formatting

### After Phase 2 (v3.7.1):
- ✅ All schema bounds use constants
- ✅ All validation assertions use constants
- ✅ ECE bins documented with citation

### After Phase 3 (v4.0):
- ✅ **100% SSOT coverage** - Zero magic numbers
- ✅ All defaults documented and centralized
- ✅ Perfect code quality baseline

---

## 📝 Testing Strategy

**For each change**:
```bash
# 1. Quality checks
make q

# 2. Unit tests
.venv/bin/pytest tests/unit/ -v

# 3. Integration tests
.venv/bin/pytest tests/integration/ -v

# 4. Smoke test (fast validation)
make s
```

**Before merging**:
```bash
# Full test suite
make test

# Config validation
make config-check

# Local smoke test
make s
```

---

## 🎯 Final Notes

**Current Status**: v3.6.2 is **100% debt-free** for P0/P1 critical items.

**These P2/P3 items are**:
- ✅ **Non-blocking** for production training
- ⚠️ **Recommended** for code quality and maintainability
- 🔵 **Optional** for low-priority cosmetic polish

**Priority**: Focus on **P2.3** (duplicate code) and **P2.1** (config validation) first. These provide the highest ROI for maintenance burden reduction.

---

**Document Version**: 1.0
**Last Updated**: October 4, 2025
**Next Review**: After v3.7.0 release
