# VALIDATION STREAMING FIX - PRODUCTION IMPLEMENTATION PLAN

**Status**: RFC - Ready for external validation
**Priority**: P0 - Blocks Modal training
**Estimated Implementation**: 60-90 minutes
**Risk Level**: LOW (backward compatible, no metric changes)

---

## 🎯 **EXECUTIVE SUMMARY**

**Problem**: Validation accumulates 60-90GB of tensors in RAM, causing OOM kills on Modal (exit 137).

**Root Cause**: Lines 105-106, 136-137, 159-160 in `val_step.py` accumulate ALL validation data across 3088 batches before computing metrics.

**Solution**: Convert to **streaming/online metrics computation** using running statistics and per-recording processing.

**Impact**: Reduces peak validation RAM from **90GB → 5-10GB** (18x reduction).

---

## 🔬 **TECHNICAL ANALYSIS**

### **Current Memory Profile (3088 batches, batch_size=48)**

| Component | Memory | Lifecycle |
|-----------|--------|-----------|
| `all_probs_flat` | 30GB | Accumulated across ALL batches |
| `all_labels_flat` | 30GB | Accumulated across ALL batches |
| `probs_flat` (concat) | 30GB | Created at line 136 for metrics |
| `labels_flat` (concat) | 30GB | Created at line 137 for metrics |
| **Peak (during FA sweep)** | **90-120GB** | **EXCEEDS 96GB Modal limit** ❌ |

### **Target Memory Profile (Streaming)**

| Component | Memory | Lifecycle |
|-----------|--------|-----------|
| Running stats buffers | 1-2GB | Updated per-batch, never grows |
| Per-recording timelines | 2-3GB | ~1832 recordings, cleared after FA sweep |
| Histogram bins (AUROC/ECE) | <100MB | Fixed-size bins |
| **Peak (streaming)** | **5-10GB** | **Safe for 96GB Modal** ✅ |

---

## 🛠️ **IMPLEMENTATION PLAN**

### **Phase 1: Add Streaming Metrics Classes (NEW FILE)**

**File**: `src/brain_brr/train/streaming_metrics.py`

**Classes to implement:**

#### **1. `StreamingAUROC`**
```python
class StreamingAUROC:
    """Compute AUROC online using histogram bins.

    Strategy:
    - Maintain fixed bins for probabilities [0.0, 0.01, 0.02, ..., 1.0]
    - Count positives/negatives per bin
    - Compute AUROC from cumulative counts (trapezoidal rule)

    Memory: O(num_bins) = ~100 bins × 8 bytes = <1KB
    """
    def __init__(self, num_bins: int = 100):
        self.num_bins = num_bins
        self.pos_bins = np.zeros(num_bins, dtype=np.int64)
        self.neg_bins = np.zeros(num_bins, dtype=np.int64)

    def update(self, probs: np.ndarray, labels: np.ndarray):
        """Update histogram bins with new batch."""
        # Bin probabilities into [0, num_bins)
        # Increment pos_bins or neg_bins based on labels
        pass

    def compute(self) -> float:
        """Compute AUROC from accumulated histogram."""
        # Cumulative sum approach (same as sklearn.roc_auc_score)
        pass
```

#### **2. `StreamingPRAUC`**
```python
class StreamingPRAUC:
    """Compute PR-AUC online using histogram bins.

    Strategy:
    - Same binning as AUROC
    - Track TP/FP/FN per threshold bin
    - Compute precision-recall curve from bins

    Memory: O(num_bins) = <1KB
    """
    def __init__(self, num_bins: int = 100):
        self.num_bins = num_bins
        self.tp_bins = np.zeros(num_bins, dtype=np.int64)
        self.fp_bins = np.zeros(num_bins, dtype=np.int64)
        self.fn = 0  # Total false negatives

    def update(self, probs: np.ndarray, labels: np.ndarray):
        pass

    def compute(self) -> float:
        pass
```

#### **3. `StreamingECE`**
```python
class StreamingECE:
    """Compute Expected Calibration Error online.

    Strategy:
    - Maintain bins for probabilities
    - Track (sum_probs, sum_labels, count) per bin
    - Compute ECE from bin statistics

    Memory: O(num_bins) = ~10 bins × 24 bytes = <1KB
    """
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.bin_probs = np.zeros(n_bins, dtype=np.float64)
        self.bin_labels = np.zeros(n_bins, dtype=np.float64)
        self.bin_counts = np.zeros(n_bins, dtype=np.int64)

    def update(self, probs: np.ndarray, labels: np.ndarray):
        pass

    def compute(self) -> float:
        pass
```

**Validation Strategy:**
- Unit tests comparing streaming vs batch metrics (should match within 0.1%)
- Test on synthetic data with known AUROC/PR-AUC/ECE values
- Verify with small validation subset before full deployment

---

### **Phase 2: Refactor `_compute_final_metrics()` to Use Streaming**

**File**: `src/brain_brr/train/val_step.py`

**BEFORE (Lines 87-190):**
```python
def _compute_final_metrics(
    all_probs_flat: list[torch.Tensor],  # ← 30GB accumulated
    all_labels_flat: list[torch.Tensor],  # ← 30GB accumulated
    all_ref_events: list[tuple[float, float]],
    all_pred_events: list[tuple[float, float]],
    total_hours: float,
    fa_rates: list[float],
    post_cfg: PostprocessingConfig,
    sampling_rate: int,
    num_recordings: int,
) -> dict[str, Any]:
    # Lines 136-137: Concatenate all tensors (60GB peak!)
    probs_flat = torch.cat(all_probs_flat).cpu().numpy()
    labels_flat = torch.cat(all_labels_flat).cpu().numpy()

    # Compute metrics from giant arrays
    auroc = roc_auc_score(labels_flat, probs_flat)
    # ...
```

**AFTER:**
```python
def _compute_final_metrics(
    streaming_auroc: StreamingAUROC,  # ← 1KB buffer
    streaming_pr: StreamingPRAUC,      # ← 1KB buffer
    streaming_ece: StreamingECE,       # ← 1KB buffer
    per_recording_timelines: list[dict],  # ← ~2-3GB (cleared after FA sweep)
    all_ref_events: list[tuple[float, float]],
    all_pred_events: list[tuple[float, float]],
    total_hours: float,
    fa_rates: list[float],
    post_cfg: PostprocessingConfig,
    sampling_rate: int,
    num_recordings: int,
) -> dict[str, Any]:
    # Compute from streaming buffers (no concatenation!)
    auroc = streaming_auroc.compute()
    pr_auc = streaming_pr.compute()
    ece = streaming_ece.compute()

    # FA sweep uses per-recording timelines (CRITICAL: keep this)
    for fa in fa_rates:
        result = find_threshold_for_fa_target(
            timelines_probs=[rec["probs"] for rec in per_recording_timelines],
            timelines_labels=[rec["labels"] for rec in per_recording_timelines],
            fa_target=fa,
            total_hours=total_hours,
            all_ref_events=all_ref_events,
            post_cfg=post_cfg,
            sampling_rate=sampling_rate,
            max_iters=constants.THRESHOLD_SEARCH_MAX_ITERS,
        )

    # CRITICAL: Clear timelines immediately after FA sweep
    del per_recording_timelines
    import gc
    gc.collect()

    # Return same metrics structure (backward compatible)
    return {
        "taes": taes,
        "auroc": auroc,
        "pr_auc": pr_auc,
        "ece": ece,
        "fa_curve": fa_curve,
        "num_recordings": num_recordings,
        "total_hours": total_hours,
        "thresholds": thresholds,
        **sensitivity_results,
    }
```

---

### **Phase 3: Update `_process_recording()` to Feed Streaming Buffers**

**BEFORE (Lines 42-84):**
```python
def _process_recording(
    windows: list[dict[str, Any]],
    all_probs_flat: list[torch.Tensor],  # ← Accumulator
    all_labels_flat: list[torch.Tensor],  # ← Accumulator
    all_ref_events: list[tuple[float, float]],
    all_pred_events: list[tuple[float, float]],
    post_cfg: PostprocessingConfig,
    sampling_rate: int,
) -> float:
    timeline_probs, timeline_labels = stitch_recording_timeline(windows, sampling_rate)

    # Extract events
    ref_events_list = batch_mask_to_events(timeline_labels.unsqueeze(0), sampling_rate)
    pred_events_list = batch_probs_to_events(timeline_probs.unsqueeze(0), post_cfg, sampling_rate)

    # PROBLEM: Accumulate for later (never freed!)
    all_probs_flat.append(timeline_probs.flatten())
    all_labels_flat.append(timeline_labels.flatten())

    return recording_hours
```

**AFTER:**
```python
def _process_recording(
    windows: list[dict[str, Any]],
    streaming_auroc: StreamingAUROC,  # ← Update buffers
    streaming_pr: StreamingPRAUC,      # ← Update buffers
    streaming_ece: StreamingECE,       # ← Update buffers
    per_recording_timelines: list[dict],  # ← For FA sweep only
    all_ref_events: list[tuple[float, float]],
    all_pred_events: list[tuple[float, float]],
    post_cfg: PostprocessingConfig,
    sampling_rate: int,
) -> float:
    timeline_probs, timeline_labels = stitch_recording_timeline(windows, sampling_rate)

    # Extract events (same as before)
    ref_events_list = batch_mask_to_events(timeline_labels.unsqueeze(0), sampling_rate)
    pred_events_list = batch_probs_to_events(timeline_probs.unsqueeze(0), post_cfg, sampling_rate)

    if ref_events_list:
        for event_obj in ref_events_list[0]:
            all_ref_events.append((float(event_obj.start_s), float(event_obj.end_s)))
    if pred_events_list:
        all_pred_events.extend(pred_events_list[0])

    # SOLUTION 1: Update streaming buffers (no accumulation!)
    probs_np = timeline_probs.flatten().cpu().numpy()
    labels_np = timeline_labels.flatten().cpu().numpy()

    streaming_auroc.update(probs_np, labels_np)
    streaming_pr.update(probs_np, labels_np)
    streaming_ece.update(probs_np, labels_np)

    # SOLUTION 2: Store ONLY for FA sweep (cleared after use)
    per_recording_timelines.append({
        "probs": timeline_probs.cpu(),  # Keep per-recording for FA sweep
        "labels": timeline_labels.cpu(),
    })

    # Free immediately
    del timeline_probs, timeline_labels, probs_np, labels_np

    return recording_hours
```

---

### **Phase 4: Update `validate_epoch()` Main Loop**

**BEFORE (Lines 193-236):**
```python
def validate_epoch(...):
    # Lines 105-106: Initialize accumulators
    all_probs_flat: list[torch.Tensor] = []  # ← PROBLEM
    all_labels_flat: list[torch.Tensor] = []  # ← PROBLEM

    with torch.no_grad():
        for batch_idx, batch in enumerate(iterator):
            # Process batch...
            for i, fid in enumerate(file_ids):
                if fid != current_file_id and current_windows:
                    recording_hours = _process_recording(
                        current_windows,
                        all_probs_flat,  # ← Accumulates
                        all_labels_flat,  # ← Accumulates
                        ...
                    )

    # Lines 226-236: Compute metrics from accumulated data
    metrics = _compute_final_metrics(
        all_probs_flat,  # ← 30GB
        all_labels_flat,  # ← 30GB
        ...
    )
```

**AFTER:**
```python
def validate_epoch(...):
    # SOLUTION: Initialize streaming buffers
    from src.brain_brr.train.streaming_metrics import (
        StreamingAUROC,
        StreamingPRAUC,
        StreamingECE,
    )

    streaming_auroc = StreamingAUROC(num_bins=100)
    streaming_pr = StreamingPRAUC(num_bins=100)
    streaming_ece = StreamingECE(n_bins=ECE_NUM_BINS)
    per_recording_timelines: list[dict] = []  # For FA sweep only

    all_ref_events: list[tuple[float, float]] = []
    all_pred_events: list[tuple[float, float]] = []
    total_hours = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(iterator):
            # Process batch (same as before)...
            for i, fid in enumerate(file_ids):
                if fid != current_file_id and current_windows:
                    recording_hours = _process_recording(
                        current_windows,
                        streaming_auroc,  # ← Update buffers
                        streaming_pr,
                        streaming_ece,
                        per_recording_timelines,  # ← For FA sweep
                        all_ref_events,
                        all_pred_events,
                        post_config,
                        constants.SAMPLING_RATE,
                    )
                    total_hours += recording_hours
                    num_recordings += 1
                    current_windows = []

    # Compute final metrics from streaming buffers
    metrics = _compute_final_metrics(
        streaming_auroc,  # ← 1KB
        streaming_pr,     # ← 1KB
        streaming_ece,    # ← 1KB
        per_recording_timelines,  # ← 2-3GB (cleared inside)
        all_ref_events,
        all_pred_events,
        total_hours,
        fa_rates,
        post_config,
        constants.SAMPLING_RATE,
        num_recordings,
    )

    return metrics
```

---

## 🧪 **TESTING STRATEGY**

### **Unit Tests (Required)**

**File**: `tests/train/test_streaming_metrics.py`

```python
def test_streaming_auroc_matches_sklearn():
    """Verify StreamingAUROC matches sklearn.roc_auc_score."""
    np.random.seed(42)
    probs = np.random.rand(10000)
    labels = (np.random.rand(10000) > 0.5).astype(float)

    # Batch computation (sklearn)
    expected = roc_auc_score(labels, probs)

    # Streaming computation
    streaming = StreamingAUROC(num_bins=100)
    streaming.update(probs, labels)
    actual = streaming.compute()

    # Should match within 1% (binning approximation)
    assert abs(expected - actual) < 0.01, f"AUROC mismatch: {expected} vs {actual}"


def test_streaming_incremental_updates():
    """Verify streaming gives same result as batch when updated incrementally."""
    np.random.seed(42)
    probs = np.random.rand(10000)
    labels = (np.random.rand(10000) > 0.5).astype(float)

    # Batch update
    streaming_batch = StreamingAUROC(num_bins=100)
    streaming_batch.update(probs, labels)
    batch_result = streaming_batch.compute()

    # Incremental updates (10 batches of 1000)
    streaming_incr = StreamingAUROC(num_bins=100)
    for i in range(10):
        start = i * 1000
        end = (i + 1) * 1000
        streaming_incr.update(probs[start:end], labels[start:end])
    incr_result = streaming_incr.compute()

    # Should match exactly
    assert abs(batch_result - incr_result) < 1e-6


def test_streaming_vs_current_validation(integration_test):
    """Integration test: Run validation with BOTH methods, compare outputs."""
    # Run current validation (accumulation)
    metrics_old = validate_epoch_old(model, dataloader, ...)

    # Run streaming validation
    metrics_new = validate_epoch(model, dataloader, ...)

    # Metrics should match within tolerance
    assert abs(metrics_old["auroc"] - metrics_new["auroc"]) < 0.01
    assert abs(metrics_old["pr_auc"] - metrics_new["pr_auc"]) < 0.01
    assert abs(metrics_old["ece"] - metrics_new["ece"]) < 0.01

    # FA sweep results should match
    for fa in [10, 5, 1]:
        key = format_sensitivity_key(fa)
        assert abs(metrics_old[key] - metrics_new[key]) < 0.02  # 2% tolerance
```

### **Memory Profiling (Validation)**

```python
import tracemalloc

def test_memory_usage():
    """Verify streaming uses <10GB peak memory."""
    tracemalloc.start()

    # Run validation
    metrics = validate_epoch(model, val_loader, ...)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_gb = peak / (1024 ** 3)
    assert peak_gb < 10.0, f"Peak memory {peak_gb:.1f}GB exceeds 10GB target"
```

---

## 📋 **IMPLEMENTATION CHECKLIST**

### **Phase 1: Streaming Metrics Infrastructure**
- [ ] Create `src/brain_brr/train/streaming_metrics.py`
- [ ] Implement `StreamingAUROC` class
- [ ] Implement `StreamingPRAUC` class
- [ ] Implement `StreamingECE` class
- [ ] Write unit tests (`test_streaming_auroc_matches_sklearn()`)
- [ ] Write unit tests (`test_streaming_incremental_updates()`)
- [ ] Verify tests pass with synthetic data

### **Phase 2: Refactor Validation Loop**
- [ ] Update `_process_recording()` signature
- [ ] Add streaming buffer updates in `_process_recording()`
- [ ] Update `_compute_final_metrics()` signature
- [ ] Remove `torch.cat()` calls (lines 136-137)
- [ ] Add per-recording timeline handling
- [ ] Add explicit cleanup after FA sweep

### **Phase 3: Update Main Validation Function**
- [ ] Replace `all_probs_flat` / `all_labels_flat` with streaming buffers
- [ ] Update `validate_epoch()` to initialize streaming classes
- [ ] Update all calls to `_process_recording()`
- [ ] Verify no accumulation in validation loop

### **Phase 4: Testing & Validation**
- [ ] Run `make test` - all tests pass
- [ ] Run integration test on small validation subset (10 files)
- [ ] Compare metrics: streaming vs old implementation (<1% diff)
- [ ] Profile memory: peak < 10GB
- [ ] Run `make q` - quality checks pass

### **Phase 5: Deployment**
- [ ] Test on local RTX 4090 with full validation set
- [ ] Verify no OOM, comparable speed
- [ ] Deploy to Modal with `--resume true`
- [ ] Monitor first validation completion
- [ ] Verify W&B metrics match previous epochs

---

## 🎯 **SUCCESS CRITERIA**

| Metric | Target | Validation Method |
|--------|--------|-------------------|
| **Peak RAM** | <10GB | `tracemalloc` profiling |
| **AUROC Match** | <1% diff | Compare with sklearn on synthetic data |
| **PR-AUC Match** | <1% diff | Unit tests with known distributions |
| **ECE Match** | <1% diff | Calibration test datasets |
| **FA Sensitivity** | <2% diff | Integration test on small val set |
| **Speed** | ±10% | Validation time should not increase >10% |
| **Quality** | 100% pass | `make q` all green |

---

## 🔍 **VALIDATION WITH EXTERNAL AGENT**

**Questions for External Review:**

1. **Correctness**: Do the streaming algorithms match batch computation semantics?
2. **Memory Safety**: Any edge cases where memory could still spike?
3. **Numerical Stability**: Binning approach accurate enough for clinical metrics?
4. **FA Sweep**: Is keeping per-recording timelines acceptable? (Alternative: streaming FA sweep is complex)
5. **Backward Compatibility**: Will metrics match within clinical tolerance?
6. **Testing Strategy**: Are unit tests comprehensive enough?
7. **Edge Cases**: Empty validation sets, single-class batches, NaN handling?

---

## 📊 **EXPECTED OUTCOMES**

**Before Fix:**
```
[VAL] Batch 3088/3088 | RAM: 90.5GB used
[VAL] Computing metrics from 148,224,000 samples...
[MODAL] Runner ran out of memory, exit code: 137
```

**After Fix:**
```
[VAL] Batch 3088/3088 | RAM: 8.2GB used
[VAL] Computing metrics from streaming buffers...
[VAL] Done! Val Loss: 0.1153 | AUROC: 0.892 | PR-AUC: 0.456
[VAL] Sensitivity@10FA: 0.847 | @5FA: 0.793 | @1FA: 0.612
✅ Validation complete in 42.3 minutes
```

---

## 🚀 **NEXT STEPS**

1. **External validation** of this implementation plan
2. **Approve plan** with any modifications
3. **Implement Phase 1-4** (60-90 min estimated)
4. **Test thoroughly** (Phase 5)
5. **Deploy to Modal**
6. **Monitor success**

---

**Document Version**: 1.0
**Author**: Claude (AI Assistant)
**Date**: 2025-10-08
**Review Status**: AWAITING EXTERNAL VALIDATION
