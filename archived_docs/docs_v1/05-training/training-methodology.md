# Training Methodology - Balanced Sampling and Validation Strategy

**Last Updated**: October 20, 2025

**Purpose**: Explain why validation has more batches than training and why this is professional standard practice

**TL;DR**: Training downsamples to 20% for class balance, validation uses 100% for accurate metrics. **This is correct** and matches industry best practices.

---

## 📊 Key Numbers

```
TRAINING (BalancedSeizureDataset):
├─ 7,702 batches (61,616 windows)
├─ Uses 20.3% of available windows (aggressive downsampling)
├─ Oversamples seizures to ~30% per batch
└─ Purpose: Learn patterns with enough positive examples

VALIDATION (ValidationDataset):
├─ 18,528 batches (148,224 windows) ← 2.41× MORE than training
├─ Uses 100% of dev split (natural ~8% seizure rate)
├─ Groups windows by file for timeline reconstruction
└─ Purpose: Measure real-world performance with accurate TAES

WHY VALIDATION HAS MORE BATCHES:
Training downsamples to 20.3% for class balance
Validation uses 100% for accurate metrics (TAES needs full timelines)
This is BY DESIGN and matches literature best practices!
```

---

## 🎯 The Problem: Severe Class Imbalance

### TUSZ Dataset Statistics

```
Total windows: 303,990 (train split)
Seizure windows: ~8% (24,319 windows)
Background windows: ~92% (279,671 windows)
```

### What Happens Without Balancing?

**Problem**: If we use all 303,990 windows in natural distribution:
- Each batch (size 8) would have 0-1 seizures
- Model learns to predict "no seizure" → 92% accuracy
- **This is useless for clinical seizure detection!**

**Example batch (unbalanced)**:
```
Batch: [❌ ❌ ❌ ❌ ❌ ❌ ❌ ❌]  ← 0 seizures
Batch: [❌ ❌ ❌ ❌ ❌ ❌ ❌ ✅]  ← 1 seizure (rare!)
Batch: [❌ ❌ ❌ ❌ ❌ ❌ ❌ ❌]  ← 0 seizures
```

**Model learns**: "Always predict background" → 92% accuracy but 0% seizure detection!

---

## ✅ The Solution: Balanced Sampling

### BalancedSeizureDataset Strategy

**Implementation**: `src/brain_brr/data/datasets.py:330-540`

**Sampling formula**:
```python
# Oversample seizures to ~30% per batch
Partial seizure windows: ALL (most informative)
Full seizure windows: 0.3× count
Background windows: 2.5× partial seizure count

Total windows = P + 0.3P + 2.5P = 3.8P windows
```

**Result**:
- Total: 61,616 windows (20.3% of available 303,990)
- Seizure rate: ~30% per batch (up from natural 8%)
- 7,702 batches @ batch_size=8

**Example batch (balanced)**:
```
Batch: [❌ ❌ ✅ ❌ ❌ ✅ ❌ ✅]  ← 3 seizures
Batch: [❌ ✅ ❌ ❌ ✅ ❌ ❌ ❌]  ← 2 seizures
Batch: [✅ ❌ ❌ ✅ ❌ ❌ ❌ ✅]  ← 3 seizures
```

**Model learns**: "Seizures have distinct patterns" → learns to discriminate!

---

## 🔍 Validation: Natural Distribution

### Why Validation Uses ALL Windows

**Implementation**: `src/brain_brr/data/datasets.py:542-701`

**Requirements for TAES Metrics**:
1. **Timeline reconstruction**: Must stitch windows back together by file
2. **Post-processing**: Hysteresis + morphology operate on full timelines
3. **Natural distribution**: Measures real-world performance (~8% seizures)
4. **No missing data**: Skipping windows breaks timeline continuity

**Result**:
- Total: 148,224 windows (100% of dev split)
- Seizure rate: ~8% (natural distribution)
- 18,528 batches @ batch_size=8

### Why More Batches Than Training

```
Training:   61,616 windows / 8 = 7,702 batches
Validation: 148,224 windows / 8 = 18,528 batches
Ratio: 2.41× more validation batches
```

**This is NOT a bug!** Training uses aggressive downsampling (20.3%) for class balance, validation uses ALL data for accurate metrics.

---

## 📚 Professional Practice Validation

### SeizureTransformer (2023 - SOTA)

✅ **Training**: Balanced sampling with oversampling of seizures
✅ **Validation**: Full test set with natural distribution
✅ **Metrics**: TAES (requires full timeline reconstruction)
✅ **Result**: Same approach as ours

### Facebook AI Seizure Detection (2019)

✅ **Training**: WeightedRandomSampler for class balance
✅ **Validation**: Full validation set, every epoch
✅ **Metrics**: Time-aligned metrics (like TAES)
✅ **Result**: Same approach as ours

### Industry Standard Practice

1. **Train on balanced data**: Oversample minority class (seizures) to ~30%
2. **Validate on natural distribution**: Use full dataset with ~8% seizures
3. **Calculate timeline metrics**: TAES requires full recordings
4. **Validate every epoch**: For early stopping and monitoring

**Our pipeline matches this exactly!** ✓

---

## ⏱️ Validation Cost Reality

### Why Validation is Expensive

**Local (RTX 4090, batch_size=8)**:
- Training: ~4.1h per epoch (7,702 batches)
- Validation: ~5.5h per epoch (18,528 batches)
- **Validation is 1.3× longer than training!**

**Modal (A100-80GB, batch_size=48)**:
- Training: 1-2h per epoch (1,284 batches)
- Validation: ~5.8h per epoch (3,088 batches)
- **Validation is 3-5× longer than training!**

### Why We Can't Skip Windows

**TAES Requirements**:
```python
# Must stitch windows by file for timeline reconstruction
for file_id in recordings:
    windows = get_all_windows(file_id)  # MUST be complete!
    timeline = stitch_windows(windows)  # Needs ALL windows
    events = postprocess(timeline)      # Hysteresis needs continuity
    taes = calculate_taes(events, refs) # Accurate only with full timeline
```

**If we skip windows**:
- Timeline reconstruction breaks (missing data)
- Post-processing fails (discontinuous signals)
- TAES calculation is inaccurate
- Can't compare to literature (different methodology)

**Conclusion**: The validation overhead is **unavoidable** for accurate TAES metrics.

---

## 🎯 Dataset Structure (TUSZ Official Splits)

### File Counts

```
Train: 4,667 EDF files (579 patients)
Dev:   1,832 EDF files (53 patients)  ← 39% as many files
Eval:  865 EDF files (43 patients)    ← Blind test set
```

**Patient Disjointness**: ✅ Enforced (see `tusz_splits.py:validate_patient_disjointness`)

### Window Counts (60s windows, 10s stride = 83% overlap)

```
Train cache: 303,990 windows (from 4,667 files)
Dev cache:   148,224 windows (from 1,832 files)
```

**Key Insight**: Dev has 39% as many files but **49% as many windows** (148k vs 304k). This is normal - file lengths vary, and dev happens to have longer recordings.

---

## ❓ Common Questions

### Q: Can we speed up validation?

**Options**:
1. **Subset validation** (e.g., 20% of dev set most epochs, 100% every 10 epochs)
   - Pros: 5× faster
   - Cons: Less accurate early stopping, might miss best checkpoint
   - Verdict: Possible P4 optimization, but adds complexity

2. **Cache predictions** (save to disk, recalculate TAES less frequently)
   - Pros: Timeline stitching only runs every N epochs
   - Cons: Can't use TAES for early stopping
   - Verdict: Possible P4 optimization if disk I/O is fast

3. **Accept the cost** (full validation every epoch)
   - Pros: Most accurate monitoring, simplest implementation
   - Cons: Expensive on Modal, long wall-clock time
   - Verdict: **This is what we do now, and it's correct!**

### Q: Why not reduce validation frequency?

**Current**: Validate every epoch
**Alternative**: Validate every 2-3 epochs

**Trade-off**:
- Saves time: Yes (50-66% reduction)
- Risks missing: Best checkpoint for early stopping
- Professional practice: Most teams validate every epoch
- **Decision**: Keep every-epoch validation for accuracy

### Q: Is this the right dataset split?

**Our splits** (TUSZ official):
```
Train: 4,667 files (72.5%)
Dev:   1,832 files (28.5%)
Eval:  865 files (test set, separate)
```

**Standard practice**: 70/15/15 or 80/10/10 splits

**Why ours is different**: We use TUSZ official splits with patient disjointness enforced. The dev split is larger than typical (28.5%) to ensure robust validation metrics. This is **correct** for TUSZ evaluation.

---

## 🔑 Key Takeaways

1. **Training downsamples to 20.3%** for class balance (seizures → 30% per batch)
2. **Validation uses 100%** for accurate TAES metrics (natural 8% distribution)
3. **Validation has 2.41× more batches** - this is BY DESIGN, not a bug!
4. **This matches professional practice** - SeizureTransformer, Facebook AI, all SOTA systems
5. **Validation overhead is unavoidable** - TAES requires full timeline reconstruction

**No action needed. Pipeline validated. Keep training. 🚀**

---

## 📚 References

1. **SeizureTransformer (2023)** - Uses balanced training, full validation with TAES
2. **Facebook AI Seizure Detection (2019)** - Same balanced/natural split approach
3. **Industry Practice** - Oversample minority class for training, validate on natural distribution
4. **Our Implementation** - Matches all of the above ✓

**See Also**:
- `src/brain_brr/data/datasets.py` - BalancedSeizureDataset (lines 330-540), ValidationDataset (lines 542-701)
- `docs/archive/TRAINING_VALIDATION_PIPELINE_ANALYSIS.md` - Full first-principles analysis
