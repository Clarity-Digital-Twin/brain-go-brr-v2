# Brain-Go-Brr v3.5.0 - Remaining Polish Items

**Date**: October 3, 2025
**Status**: POST-AUDIT CLEANUP (Non-Blocking)
**Priority**: Medium/Low - Can be deferred to v3.5.1 or v4.0

**NOTE**: All P0 (critical) and P1 (high-priority) issues from the constants audit have been FIXED and VERIFIED. The system is production-ready. These are optional quality-of-life improvements.

---

## P2: Medium Priority (Quality Improvements)

### P2.1: Config Consistency Automation

**Issue**: No automated validation that YAML config defaults match `constants.py` values.

**Current State**:
- `configs/local/train.yaml`, `configs/modal/train.yaml` manually specify values like:
  ```yaml
  hysteresis:
    tau_on: 0.86
    tau_off: 0.78
  ```
- If we change `HYSTERESIS_TAU_ON` in `constants.py`, YAMLs won't auto-update
- Risk: Config drift over time as constants evolve

**Solution**:
Create `scripts/validate_configs.py` that:
1. Loads all YAML configs
2. Compares numeric values to `constants.py`
3. Fails if any mismatch detected
4. Run as part of `make q` or CI/CD

**Files to Create**:
- `scripts/validate_configs.py`
- Update `Makefile` to add `make config-check` target
- Add to `.github/workflows/tests.yml` (if CI/CD exists)

**Effort**: 1-2 hours
**Benefit**: Prevents silent config drift

---

### P2.2: Outdated Config Documentation

**Issue**: `configs/CONFIG_CONSISTENCY_CHECK.md` still documents v3.2.0 defaults.

**Current State**:
```markdown
# configs/CONFIG_CONSISTENCY_CHECK.md (lines 1-70)
# Still mentions:
batch_size: 4          # Actually 12 in local/train.yaml
gradient_clip: 0.1     # Actually 1.0 in current configs
```

**Solution**:
Either:
1. **Option A**: Update manually to reflect current v3.5.0 defaults
2. **Option B**: Auto-generate from YAML parsing + constants.py
3. **Option C**: Delete and replace with `scripts/validate_configs.py` output

**Files to Update**:
- `configs/CONFIG_CONSISTENCY_CHECK.md`

**Effort**: 30 minutes (manual update) or 1 hour (auto-generation)
**Benefit**: Accurate developer documentation

---

### P2.3: Duplicate FA Sweep Logic in val_step.py

**Issue**: `val_step.py:135-198` reimplements threshold search instead of calling `false_alarm.py` helpers.

**Current State**:
```python
# val_step.py has its own binary search loop
for fa in fa_rates:
    low, high = constants.THRESHOLD_SEARCH_LOW, constants.THRESHOLD_SEARCH_HIGH
    for _ in range(constants.THRESHOLD_SEARCH_MAX_ITERS):
        # ... 30 lines of search logic ...

# false_alarm.py has the canonical implementation
def find_threshold_for_fa_target(...):
    # ... same logic ...
```

**Why Duplicated**: Historical reasons - `val_step.py` predates `false_alarm.py` refactor.

**Solution**:
Refactor `_compute_final_metrics` to call `compute_fa_sweep()` from `false_alarm.py`:
```python
from src.brain_brr.eval.helpers.false_alarm import compute_fa_sweep

def _compute_final_metrics(...):
    # Build timelines for sweep
    timelines_probs = [p.flatten() for p in all_probs_flat]
    timelines_labels = [l.flatten() for l in all_labels_flat]

    # Use canonical implementation
    results = compute_fa_sweep(
        timelines_probs, timelines_labels, fa_rates,
        total_hours, all_ref_events, post_cfg, sampling_rate
    )

    # Extract thresholds and sensitivities
    for result in results:
        thresholds[f"{result.fa_target}"] = result.threshold_tau_on
        sensitivity_results[f"sensitivity_at_{result.fa_target}fa"] = result.sensitivity
```

**Files to Modify**:
- `src/brain_brr/train/val_step.py` (lines 135-198)

**Effort**: 1 hour (careful refactor + testing)
**Benefit**: Single source of truth for FA sweep logic, easier to maintain
**Risk**: Low (same algorithm, just calling helper instead of reimplementing)

---

### P2.4: Inline Imports in Hot Loops

**Issue**: `val_step.py` imports functions inside per-recording loops.

**Current State**:
```python
# Line 54 (inside _process_recording function)
from src.brain_brr.eval.metrics import stitch_recording_timeline

# Line 109 (inside _compute_final_metrics function)
from src.brain_brr.eval.metrics import calculate_ece, calculate_taes, overlap
```

**Why It Works**: Python caches module imports, so no actual performance hit.

**Why It's Confusing**:
- Violates PEP 8 (imports at top)
- Harder to track dependencies
- IDEs may complain

**Solution**:
Move all imports to module level (lines 1-27):
```python
from src.brain_brr.eval.metrics import (
    batch_probs_to_events,
    calculate_ece,
    calculate_taes,
    overlap,
    stitch_recording_timeline,
)
```

**Files to Modify**:
- `src/brain_brr/train/val_step.py` (lines 54, 109)

**Effort**: 5 minutes
**Benefit**: Cleaner code, better IDE support
**Risk**: Zero (just moving import location)

---

### P2.5: Metric Name String Formatting

**Issue**: Sensitivity metric keys are manually formatted with f-strings.

**Current State**:
```python
# Line 196
sensitivity_results[f"sensitivity_at_{fa}fa"] = sensitivity

# Used throughout codebase as magic strings
if "sensitivity_at_10fa" in results:
    ...
```

**Why It's Bad**:
- Typo risk (`"sensitivity_at_10af"` would be a silent bug)
- No autocomplete or type checking

**Solution Option A**: Add constants for common metric keys
```python
# In constants.py
METRIC_SENSITIVITY_TEMPLATE: str = "sensitivity_at_{}fa"

def format_sensitivity_key(fa_rate: float) -> str:
    """Format sensitivity metric key for FA rate."""
    return METRIC_SENSITIVITY_TEMPLATE.format(fa_rate)

# Usage
sensitivity_results[format_sensitivity_key(fa)] = sensitivity
```

**Solution Option B**: Use TypedDict (Python 3.11+)
```python
from typing import TypedDict

class ValidationMetrics(TypedDict, total=False):
    auroc: float
    taes: float
    pr_auc: float
    ece: float
    sensitivity_at_10fa: float
    sensitivity_at_5fa: float
    sensitivity_at_1fa: float
    # ...
```

**Files to Modify**:
- `src/brain_brr/constants.py` (add helper)
- `src/brain_brr/train/val_step.py` (line 196)
- `src/brain_brr/eval/metrics.py` (similar patterns)

**Effort**: 1 hour (add helper + update all call sites)
**Benefit**: Type safety, no typo risk
**Risk**: Low

---

## P3: Low Priority (Nice to Have)

### P3.1: Schema Epsilon Bounds

**Issue**: `config/schemas.py:474` uses literal `ge=1e-6` instead of constant.

**Current State**:
```python
some_field: float = Field(default=0.001, ge=1e-6, description="...")
```

**Solution**:
```python
from src.brain_brr.constants import EPSILON_NUMERICAL

some_field: float = Field(default=0.001, ge=EPSILON_NUMERICAL, description="...")
```

**Files to Search**: `src/brain_brr/config/schemas.py` (find all `ge=1e-` patterns)

**Effort**: 15 minutes
**Benefit**: Consistency with epsilon policy
**Risk**: Zero

---

### P3.2: ECE Bins Not Justified

**Issue**: `calculate_ece()` uses `n_bins=10` with no documentation.

**Current State**:
```python
# eval/metrics.py:27
def calculate_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
```

**Solution**:
```python
# In constants.py
ECE_NUM_BINS: int = 10  # Standard calibration curve resolution (Guo et al. 2017)

# In metrics.py
def calculate_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = ECE_NUM_BINS) -> float:
```

**Files to Modify**:
- `src/brain_brr/constants.py` (add constant)
- `src/brain_brr/eval/metrics.py` (line 27)

**Effort**: 5 minutes
**Benefit**: Documents WHY 10 bins
**Risk**: Zero

---

### P3.3: Additional Time/Preprocessing Constants

**Issue**: Various scattered literals that could be constants but aren't critical.

**Examples**:
```python
# Warmup fraction (0.1 = 10% of training)
warmup_fraction = 0.1

# Balanced sampler sample size
sample_size = 500

# Memory warning threshold
if swap_usage > 0.1:  # 0.1 GB

# Gradient percentile logging
p95 = np.percentile(grads, 0.95)

# Temperature defaults
adj_temperature = 1.0

# Alpha mixing (GNN SSGConv)
alpha = 0.05

# Eigenvalue clamp range
eigenvalues = torch.clamp(eigenvalues, 1e-6, 2.0)

# LayerScale init
layer_scale = 0.1

# TAES alpha (false alarm penalty)
taes_alpha = 0.15
```

**Solution**: Add to `constants.py` as needed, but NOT urgent.

**Effort**: 2-3 hours (find all, document, refactor)
**Benefit**: Complete SSOT coverage
**Priority**: Very low - these aren't in hot paths or clinical-critical code

---

## P3.4: Function Signature Consistency (Remaining)

**Issue**: Some functions still use `sampling_rate: int = 256` instead of `SAMPLING_RATE`.

**Verified Remaining** (from audit):
- `src/brain_brr/streaming/streaming.py:41` ✅ (uses SAMPLING_RATE)
- `src/brain_brr/events/events.py:80` ✅ (uses SAMPLING_RATE)
- All others verified to use constants

**Status**: COMPLETE - This was actually fixed! (I verified by reading the files)

**No action needed.**

---

## P3.5: Config Schema Literal Types

**Issue**: Some schemas still use `Literal[256]` instead of `Literal[SAMPLING_RATE]`.

**Current State**:
```python
# Line 614 of schemas.py
assert self.data.sampling_rate == 256, "Must use 256 Hz"
```

**Why It Exists**: Safety assertion to catch config errors.

**Solution Options**:
1. **Keep as-is**: This is actually fine - it's a runtime validation that sampling rate is correct
2. **Use constant**: `assert self.data.sampling_rate == SAMPLING_RATE`

**Files to Review**: `src/brain_brr/config/schemas.py` (line 614)

**Effort**: 2 minutes
**Benefit**: Marginal (assertion is for safety, literal is clearer)
**Priority**: Very low

---

## Summary

| Priority | Item | Files | Effort | Benefit |
|----------|------|-------|--------|---------|
| P2.1 | Config consistency script | 1 new | 2h | High - prevents drift |
| P2.2 | Update config docs | 1 edit | 30m | Medium - accuracy |
| P2.3 | Refactor FA sweep duplication | 1 edit | 1h | High - maintainability |
| P2.4 | Move inline imports | 1 edit | 5m | Low - code style |
| P2.5 | Metric name formatting | 3 edits | 1h | Medium - type safety |
| P3.1 | Schema epsilon constants | 1 edit | 15m | Low - consistency |
| P3.2 | ECE bins constant | 2 edits | 5m | Low - documentation |
| P3.3 | Additional constants | Many | 3h | Low - completeness |
| P3.4 | Function signatures | 0 | 0m | ✅ DONE |
| P3.5 | Schema literals | 1 edit | 2m | Very low |

**Total P2 Effort**: ~5 hours
**Total P3 Effort**: ~4 hours

---

## Recommendation

**For v3.5.1 (next minor release):**
- ✅ P2.1: Config consistency automation (prevents future bugs)
- ✅ P2.3: Refactor FA sweep (single source of truth)
- ✅ P2.4: Move inline imports (5 minutes, easy win)

**For v4.0 (major refactor):**
- P2.2, P2.5, P3.1-P3.5: Complete polish pass

**For now (v3.5.0):**
- ✅ **SHIP IT!** All blockers fixed, system is production-ready 🚀

---

**Last Updated**: October 3, 2025
**Status**: COMPLETE - All critical work done, these are optional improvements
**Next Milestone**: Production training on TUSZ v2.0.3
