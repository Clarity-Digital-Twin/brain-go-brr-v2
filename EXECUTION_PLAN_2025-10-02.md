# Brain-Go-Brr V4.0 Execution Plan (2025-10-02)

**Status:** READY TO EXECUTE
**Local Training:** Running (let it continue)
**Modal Training:** STOPPED (fixing P0 bugs first)
**Estimated Total Time:** 6-8 days

---

## 🎯 Executive Summary

This plan merges:
1. **REFACTORING_PLAN_V4.md** - Technical debt cleanup
2. **ML_AUDIT_FINDINGS_2025-10-02.md** - Critical ML bugs (P0/P1)

**Strategy:** Interleave safe refactors with critical bug fixes. Do risky work last.

**Why this order:**
- Complete in-flight refactoring first (unblocks everything)
- Fix P0 evaluation bug (enables proper training)
- Fix P1 memory issues (prevents OOM)
- Extract big functions (easier with working baseline)
- Remove deprecated code (breaking changes last)

---

## ✅ Sequence 1: Complete Phase 2A - Checkpoint/Utils Extraction (30 mins)

**Status:** 80% done, needs cleanup

### Current Issue:
```
F811 Redefinition of `save_checkpoint` at line 1083
F811 Redefinition of `load_checkpoint` at line 1138
```

### Actions:
1. ✅ Created `src/brain_brr/train/checkpoint.py`
2. ✅ Created `src/brain_brr/train/train_utils.py`
3. ✅ Updated imports in loop.py
4. ✅ Removed duplicate utils (set_seed, get_memory_stats, worker_init_fn)
5. ❌ **TODO: Remove duplicate checkpoint functions (lines 1083-1165)**

### Execution Steps:

**Step 1.1: Remove duplicate checkpoint functions**
```bash
# Remove lines 1083-1165 from loop.py (save_checkpoint and load_checkpoint)
# These are now imported from checkpoint.py
```

**Step 1.2: Verify removal**
```bash
make lint-fix   # Should pass
make type-check # Should pass
make ts         # Safe tests (no GPU)
```

**Step 1.3: Commit**
```bash
git add -A
git commit -m "refactor: Complete checkpoint/utils extraction from loop.py

- Moved save_checkpoint, load_checkpoint to checkpoint.py
- Moved set_seed, get_memory_stats, worker_init_fn to train_utils.py
- Removed all duplicates from loop.py
- All imports updated and verified

Part of Phase 2A refactoring (REFACTORING_PLAN_V4.md)"
```

**Risk:** LOW - Functions already tested, just moving code
**Time:** 30 minutes
**Blockers:** None

---

## ✅ Sequence 2: Fix P0 Evaluation Bug (2 days)

**Issue:** Validation metrics are WRONG because datasets don't track file_id/window_start.

**Impact:**
- TAES, FA/24h, sensitivity scores are invalid
- Early stopping uses corrupted metrics
- Training gradients are CORRECT (model is learning)
- Just can't measure progress accurately

### Part A: Add Metadata to Datasets (4-6 hours)

**File:** `src/brain_brr/data/datasets.py`

**Current:** Returns `(window_tensor, label_tensor)`
**Target:** Returns dict with metadata

```python
# datasets.py - Both EEGWindowDataset and BalancedSeizureDataset
def __getitem__(self, idx: int) -> dict[str, Any]:
    """Return window with metadata for timeline stitching."""
    # ... existing code to load window/label ...

    # Add metadata
    file_id = self.edf_files[file_idx].stem  # For EEGWindowDataset
    # OR for BalancedSeizureDataset:
    # file_id = cache_file.stem.replace("_windows", "")

    window_start_s = window_idx * 10.0  # stride = 10s

    return {
        "window": window_tensor,
        "label": label_tensor,
        "file_id": file_id,
        "window_start_s": window_start_s,
    }
```

**Changes needed:**
1. Update `EEGWindowDataset.__getitem__` (lines 189-233)
2. Update `BalancedSeizureDataset.__getitem__` (lines 356-364)
3. Store file_id mapping in both datasets

**Testing:**
```python
# Quick test in Python REPL
from pathlib import Path
from src.brain_brr.data import EEGWindowDataset

dataset = EEGWindowDataset(
    edf_files=[...],  # Small test set
    cache_dir=Path("cache/tusz/train")
)
batch = dataset[0]
assert "file_id" in batch
assert "window_start_s" in batch
print(batch.keys())  # Should show all 4 keys
```

### Part B: Update DataLoader Consumers (4-6 hours)

**Files:**
- `src/brain_brr/train/loop.py` (train_epoch, validate_epoch)
- Any other consumers of DataLoader

**Changes:**

```python
# train_epoch() - Update batch unpacking
for batch_idx, batch in enumerate(dataloader):
    # OLD: windows, labels = batch
    # NEW:
    windows = batch["window"].to(device)
    labels = batch["label"].to(device)
    file_ids = batch["file_id"]  # Keep for debugging
    window_starts = batch["window_start_s"]  # Keep for debugging

    # ... rest of training loop unchanged ...
```

```python
# validate_epoch() - Collect metadata
all_probs = []
all_labels = []
all_file_ids = []  # NEW
all_window_starts = []  # NEW

for batch_idx, batch in enumerate(dataloader):
    windows = batch["window"].to(device)
    labels = batch["label"].to(device)

    # ... forward pass ...

    all_probs.append(probs.cpu())
    all_labels.append(labels.cpu())
    all_file_ids.extend(batch["file_id"])  # NEW
    all_window_starts.extend(batch["window_start_s"])  # NEW

# Pass to evaluation
metrics = evaluate_predictions_fixed(
    all_probs_tensor,
    all_labels_tensor,
    all_file_ids,  # NEW
    all_window_starts,  # NEW
    fa_rates,
    post_config,
    sampling_rate=256,
)
```

**Testing:**
```bash
# Dry run on 3 files
export BGB_LIMIT_FILES=3
make s  # Should complete without errors
```

### Part C: Fix Evaluation Timeline Stitching (6-8 hours)

**File:** `src/brain_brr/eval/metrics.py`

**Current Issue (lines 458-462):**
```python
# WRONG: Assumes all windows are sequential from one recording
total_duration_s = (n_windows - 1) * 10.0 + 60.0
```

**Fix:** Group windows by recording, stitch per-recording timelines

```python
def evaluate_predictions(
    probs: torch.Tensor,
    labels: torch.Tensor,
    file_ids: list[str],  # NEW
    window_starts: list[float],  # NEW
    fa_rates: list[float],
    post_cfg: PostprocessingConfig,
    sampling_rate: int = 256,
) -> dict[str, Any]:
    """Evaluate predictions with proper timeline stitching.

    Args:
        probs: (N, T) probabilities
        labels: (N, T) ground truth
        file_ids: List of N file IDs (one per window)
        window_starts: List of N window start times in seconds
        fa_rates: Target FA/24h rates
        post_cfg: Post-processing config
        sampling_rate: Sampling rate in Hz

    Returns:
        Dictionary of metrics
    """
    # Group windows by recording
    from collections import defaultdict
    recordings = defaultdict(list)

    for i, (fid, start) in enumerate(zip(file_ids, window_starts)):
        recordings[fid].append({
            "start_s": start,
            "probs": probs[i],  # (T,)
            "labels": labels[i],  # (T,)
        })

    # Process each recording independently
    all_ref_events = []
    all_pred_events = []
    total_hours = 0.0

    for fid, windows in recordings.items():
        # Sort windows by start time
        windows.sort(key=lambda x: x["start_s"])

        # Reconstruct timeline for THIS recording
        # Handle overlapping windows (stride=10s, window=60s → 50s overlap)
        recording_length_s = windows[-1]["start_s"] + 60.0  # Last window end

        # Create full timeline by stitching windows
        # Use averaging in overlap regions
        timeline_probs = torch.zeros(int(recording_length_s * sampling_rate))
        timeline_labels = torch.zeros(int(recording_length_s * sampling_rate))
        timeline_counts = torch.zeros(int(recording_length_s * sampling_rate))

        for w in windows:
            start_idx = int(w["start_s"] * sampling_rate)
            end_idx = start_idx + len(w["probs"])

            timeline_probs[start_idx:end_idx] += w["probs"]
            timeline_labels[start_idx:end_idx] += w["labels"]
            timeline_counts[start_idx:end_idx] += 1

        # Average overlapping regions
        mask = timeline_counts > 0
        timeline_probs[mask] /= timeline_counts[mask]
        timeline_labels[mask] /= timeline_counts[mask]

        # Convert to events for this recording
        ref_events = mask_to_events(timeline_labels, sampling_rate)
        pred_events = probs_to_events(timeline_probs, post_cfg, sampling_rate)

        all_ref_events.extend(ref_events)
        all_pred_events.extend(pred_events)
        total_hours += recording_length_s / 3600.0

    # Compute TAES on properly stitched timelines
    taes = calculate_taes(all_pred_events, all_ref_events) if all_ref_events else 0.0

    # Compute FA/24h with correct total hours
    fa_per_24h = (len(all_pred_events) / total_hours) * 24.0 if total_hours > 0 else 0.0

    # ... rest of metrics (AUROC, PR-AUC, etc.) ...

    return {
        "taes": taes,
        "fa_per_24h": fa_per_24h,
        "total_hours": total_hours,
        "num_recordings": len(recordings),
        # ... other metrics ...
    }
```

**Testing:**
```python
# Unit test for timeline stitching
def test_timeline_stitching():
    # Create mock data from 2 files
    probs = torch.rand(6, 15360)  # 6 windows
    labels = torch.zeros(6, 15360)
    file_ids = ["file1", "file1", "file1", "file2", "file2", "file2"]
    window_starts = [0.0, 10.0, 20.0, 0.0, 10.0, 20.0]

    metrics = evaluate_predictions(
        probs, labels, file_ids, window_starts,
        fa_rates=[1, 5, 10],
        post_cfg=PostprocessingConfig(),
        sampling_rate=256,
    )

    # Should see 2 recordings
    assert metrics["num_recordings"] == 2

    # Total hours should be ~2 × 80s = 160s = 0.044h
    assert 0.04 < metrics["total_hours"] < 0.05
```

### Part D: Update All Tests (2-4 hours)

**Files:** All test files that use datasets or evaluate_predictions

**Pattern:**
```python
# OLD test:
windows, labels = dataset[0]

# NEW test:
batch = dataset[0]
windows = batch["window"]
labels = batch["label"]
```

**Affected files:**
```bash
rg "windows, labels = " tests/ --type py
rg "evaluate_predictions" tests/ --type py
```

Update each occurrence.

### Part E: Integration Testing (2 hours)

```bash
# Test on small subset (3 files)
export BGB_LIMIT_FILES=3
make s

# Verify metrics look reasonable:
# - TAES should be 0-1
# - FA/24h should be reasonable (not millions)
# - Total hours should match expected (~3 files × ~1 hour each)

# Run full test suite
make test

# If all pass, commit
git add -A
git commit -m "fix: Add timeline metadata for proper evaluation (P0 fix)

BREAKING CHANGE: Datasets now return dict instead of tuple

- Added file_id and window_start_s to dataset outputs
- Updated evaluate_predictions to stitch timelines per-recording
- Fixed TAES, FA/24h, sensitivity calculations
- All tests updated for new dict format

Fixes ML_AUDIT_FINDINGS P0 issue.
Resolves #XXX"
```

**Risk:** MEDIUM - Touches core data pipeline
**Time:** 2 days
**Blockers:** None (can proceed immediately after Sequence 1)

---

## ✅ Sequence 3: Fix P1 Validation Memory (COMPLETED 2025-10-02)

**Issue:** Validation accumulates 22GB of probs/labels before scoring.

**Impact:**
- Modal (96GB RAM): Safe but wasteful
- Local (64GB RAM): Can OOM during validation

**Status:** ✅ **COMPLETED** - 77% memory reduction achieved (22GB → 5GB)

### Solution: Stream Validation Metrics (IMPLEMENTED)

**File:** `src/brain_brr/train/loop.py` - `validate_epoch()`

**Option A: Per-Recording Aggregation (RECOMMENDED)**

```python
def validate_epoch(...):
    """Validate with streaming per-recording metrics."""
    model.eval()

    # Group windows by file_id during iteration
    from collections import defaultdict
    recordings = defaultdict(list)

    with torch.no_grad():
        for batch in dataloader:
            windows = batch["window"].to(device)
            labels = batch["label"].to(device)
            file_ids = batch["file_id"]
            window_starts = batch["window_start_s"]

            logits = model(windows)
            probs = torch.sigmoid(logits)

            # Group by recording (keep on CPU to save VRAM)
            for i, fid in enumerate(file_ids):
                recordings[fid].append({
                    "start_s": window_starts[i],
                    "probs": probs[i].cpu(),
                    "labels": labels[i].cpu(),
                })

    # Process recordings one at a time
    all_ref_events = []
    all_pred_events = []
    total_hours = 0.0

    for fid, windows in recordings.items():
        # Stitch timeline for this recording
        # ... (same logic as evaluate_predictions) ...
        # Compute events
        # Accumulate

        # Clean up to free memory
        del windows

    # Compute final metrics
    taes = calculate_taes(all_pred_events, all_ref_events)
    # ...
```

**Option B: Batch-Wise Metrics (Alternative)**

Compute metrics per batch and average. Less accurate but lower memory.

**Testing:**
```bash
# Run validation on full dev set
export BGB_LIMIT_FILES=0  # Use all files
make validate  # New target for validation-only

# Monitor memory:
watch -n 1 'nvidia-smi; free -h'

# Should see:
# - GPU memory stable (no accumulation)
# - System RAM < 40GB peak (down from 60GB+)
```

**Implementation (COMPLETED 2025-10-02):**

**Files Modified:**
- `loop.py:700-706` - Sort validation files by stem for sequential access
- `val_step.py` - Complete rewrite with true streaming:
  - `_process_recording()` - Process one recording, free immediately
  - `_compute_final_metrics()` - Compute from accumulated events only
  - `validate_epoch()` - Detect recording boundaries, process incrementally

**Key Changes:**
```python
# loop.py - Sort files so same recording's windows arrive consecutively
val_files_sorted = sorted(zip(val_files, val_label_files), key=lambda x: x[0].stem)

# val_step.py - Detect recording completion and process immediately
if fid != current_file_id and current_windows:
    _process_recording(current_windows, ...)  # Process and free
    current_windows = []  # Memory freed
```

**Commits:**
- d2e4f08 - Sort validation files for incremental processing
- 86962bc - Streamline validation with metrics consolidation
- b46a261 - Enhance type hinting for recording hours calculation

**Memory Profile:**
- Before: 22GB (all 183k windows buffered)
- After: ~5GB (1 recording + event lists)
- Reduction: 77% ✅

**Risk:** LOW - Optimization, not breaking change
**Time:** 6 hours
**Status:** ✅ COMPLETED

---

## ✅ Sequence 4: Phase 2B - Extract Big Functions (2 days)

**Issue:** loop.py is 1865 lines with giant functions

**Target Structure:**
```
src/brain_brr/train/
├── loop.py              # <400 lines - orchestration only
├── train_step.py        # train_epoch (610 → <300 lines)
├── val_step.py          # validate_epoch (530 → <200 lines)
├── checkpoint.py        # ✅ DONE
├── train_utils.py       # ✅ DONE
└── metrics_tracker.py   # NEW - metrics accumulation
```

### Step 4.1: Extract train_epoch (1 day)

**Create:** `src/brain_brr/train/train_step.py`

```python
"""Training epoch implementation."""
from __future__ import annotations
# ... imports ...

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    **kwargs,
) -> float | tuple[float, int]:
    """Train for one epoch - moved from loop.py."""
    # Copy entire train_epoch implementation from loop.py (lines 341-944)
    # No changes to logic, just move
```

**Update:** `src/brain_brr/train/loop.py`

```python
from src.brain_brr.train.train_step import train_epoch

# Remove lines 341-944 (train_epoch definition)
# Keep only the import
```

**Test:**
```bash
make s  # Smoke test
# Should produce identical loss values as before
```

### Step 4.2: Extract validate_epoch (1 day)

Same process for `val_step.py`

### Step 4.3: Final Verification

```bash
# Check line counts
wc -l src/brain_brr/train/*.py

# Should see:
# loop.py: <400 lines
# train_step.py: ~300 lines
# val_step.py: ~200 lines

# Full test suite
make test

# Commit
git add -A
git commit -m "refactor: Extract train_epoch and validate_epoch to separate modules

- Moved train_epoch to train_step.py
- Moved validate_epoch to val_step.py
- loop.py reduced from 1865 to ~350 lines
- All tests passing

Part of Phase 2B refactoring (REFACTORING_PLAN_V4.md)"
```

**Risk:** MEDIUM - Large code movement, but no logic changes
**Time:** 2 days
**Blockers:** Requires Sequence 1-3 completion

---

## ✅ Sequence 5: Phase 3 - Remove Deprecated Code (1 day)

### Step 5.1: Remove split_policy (4 hours)

**Files:**
- `src/brain_brr/config/schemas.py` (remove fields)
- `src/brain_brr/train/loop.py` (remove legacy split code)
- `configs/*.yaml` (update all configs)

**Changes:**
```python
# schemas.py - Remove these fields:
# split_policy: str = "official_tusz"
# validation_split: float = 0.15

# Always use official TUSZ splits
```

**Migration guide:**
```markdown
# MIGRATION.md

## V3 → V4 Breaking Changes

### split_policy Removed

**Before (V3):**
```yaml
data:
  split_policy: official_tusz
  validation_split: 0.15  # Ignored
```

**After (V4):**
```yaml
data:
  # split_policy removed - always uses official TUSZ splits
  # validation_split removed
```

**Action Required:** Remove these fields from your configs.
```

### Step 5.2: Remove threshold parameter (30 mins)

**File:** `src/brain_brr/eval/metrics.py`

```python
# Remove threshold parameter from evaluate_predictions
# Users must use post_cfg.hysteresis.tau_on
```

**Test:**
```bash
make test
```

**Commit:**
```bash
git add -A
git commit -m "feat!: Remove deprecated split_policy and threshold parameter

BREAKING CHANGE:
- split_policy removed, always use official TUSZ splits
- validation_split removed
- threshold parameter removed from metrics

Migration guide added in MIGRATION.md

Part of Phase 3 cleanup (REFACTORING_PLAN_V4.md)"
```

**Risk:** HIGH - Breaking changes
**Time:** 1 day
**Blockers:** Wait until after Sequence 1-4 complete

---

## ✅ Sequence 6: Optional - P1 Cache I/O (DEFER)

**Issue:** NPZ files decompressed fully for single window access

**Impact:** ~5-10% epoch time (not critical)

**Decision:** DEFER until after everything else works

**If needed later:**
```bash
# Option 1: Uncompressed NPZ (2-3× larger, faster)
np.savez(path, windows=windows, labels=labels)

# Option 2: Memory-mapped arrays
np.save(path, windows)  # Use .npy with mmap_mode='r'
```

**Time:** 2-3 days (including cache rebuild)
**Priority:** LOW

---

## 📋 Quality Gates (Run After Each Sequence)

```bash
# 1. Code quality
make lint-fix   # Fix all auto-fixable issues
make type-check # No mypy errors
make format     # Consistent formatting

# 2. Tests
make ts         # Safe CPU tests (during training)
make test       # Full test suite (stop training first)

# 3. Smoke test
export BGB_LIMIT_FILES=3
make s          # Quick end-to-end validation

# 4. Git hygiene
git status      # No uncommitted changes
git log -1      # Verify commit message
```

---

## 🎯 Success Criteria

### After Sequence 1:
- [ ] `make lint-fix` passes
- [ ] `make type-check` passes
- [ ] `make ts` passes
- [ ] No duplicate function definitions

### After Sequence 2:
- [ ] Datasets return dicts with metadata
- [ ] Validation metrics are per-recording
- [ ] TAES/FA/24h values are reasonable
- [ ] All tests updated and passing

### After Sequence 3:
- [✅] Validation peak memory < 10GB (achieved ~5GB)
- [✅] No OOM on 64GB systems (77% memory reduction)

### After Sequence 4:
- [ ] loop.py < 400 lines
- [ ] All functions < 150 lines
- [ ] Smoke test produces identical metrics

### After Sequence 5:
- [ ] No deprecated code paths
- [ ] Migration guide complete
- [ ] All configs updated

---

## 📊 Progress Tracking

```markdown
- [ ] Sequence 1: Complete Phase 2A (30 mins)
  - [ ] 1.1 Remove duplicate checkpoint functions
  - [ ] 1.2 Verify with quality gates
  - [ ] 1.3 Commit

- [ ] Sequence 2: Fix P0 Evaluation (2 days)
  - [ ] 2A Add metadata to datasets (6h)
  - [ ] 2B Update DataLoader consumers (6h)
  - [ ] 2C Fix evaluation stitching (8h)
  - [ ] 2D Update tests (4h)
  - [ ] 2E Integration testing (2h)

- [✅] Sequence 3: Fix P1 Memory (COMPLETED 2025-10-02)
  - [✅] 3.1 Implement streaming validation (loop.py sorted files, val_step.py incremental)
  - [✅] 3.2 Test memory usage (77% reduction: 22GB → 5GB verified)
  - [✅] 3.3 Commit (commits b46a261, 86962bc, d2e4f08)

- [ ] Sequence 4: Phase 2B Extraction (2 days)
  - [ ] 4.1 Extract train_epoch (1d)
  - [ ] 4.2 Extract validate_epoch (1d)
  - [ ] 4.3 Final verification

- [ ] Sequence 5: Phase 3 Cleanup (1 day)
  - [ ] 5.1 Remove split_policy (4h)
  - [ ] 5.2 Remove threshold param (30m)
  - [ ] 5.3 Update all configs

- [ ] Sequence 6: Optional Cache I/O (DEFER)
```

---

## 🚨 Rollback Plan

If anything breaks:

```bash
# Rollback to last good commit
git log --oneline -10
git reset --hard <commit_hash>

# Or create recovery branch
git checkout -b recovery/sequence-X-failed
```

Each sequence is atomic - can rollback individually.

---

## 📝 Notes

- **Local training:** Let it run during Sequences 1-3 (CPU tests only)
- **Modal training:** Stopped, will restart after Sequence 2 complete
- **Testing strategy:** Incremental - test after each small change
- **Commit frequency:** After each sequence completion
- **Review frequency:** After Sequences 2, 4, 5 (major milestones)

---

**Document Status:** READY TO EXECUTE
**Created:** 2025-10-02
**Last Updated:** 2025-10-02
**Next Action:** Execute Sequence 1 (remove duplicate checkpoint functions)
