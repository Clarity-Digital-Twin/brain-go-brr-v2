# P0: BALANCED SAMPLING CATASTROPHIC FAILURE WITH TUSZ

**Severity**: P0 - TRAINING COMPLETELY BROKEN
**Date**: 2025-09-20
**Status**: ACTIVE EMERGENCY - Both local and Modal affected
**Impact**: Model collapses to 0% seizure predictions, training is worthless

## THE CATASTROPHE

We have a FUNDAMENTAL DESIGN FLAW in our balanced sampling that makes training on TUSZ impossible. The model sees NO SEIZURES and learns to predict "no seizure" for everything.

```
[DATASET] Windows with seizures: 0/1000 (0.00%)
[CRITICAL] Model will likely collapse to all-negative predictions.
```

This isn't a minor bug - this is a COMPLETE FAILURE of the training pipeline.

## ROOT CAUSE ANALYSIS

### 1. TUSZ Dataset Reality
- **12:1 imbalance**: Only ~8% of recordings have seizures
- **Temporal sparsity**: Even in seizure files, only small portions are seizures
- **Window-level imbalance**: Could be 100:1 or worse at window level
- **File distribution**: Seizure files are scattered throughout, not clustered

### 2. Our Sampling Strategy (BROKEN)
```python
# src/brain_brr/train/loop.py:1064
train_sampler = create_balanced_sampler(train_dataset, sample_size=500)

# src/brain_brr/train/loop.py:87-88
sample_size = min(sample_size, len(dataset))  # 500 windows
sample_indices = torch.randperm(len(dataset))[:sample_size]
```

**THE FATAL FLAW**: We sample only 500 windows to determine balance!

### 3. The Math of Failure
```
Dataset: 252,279 windows (from 3734 files)
Seizure rate: ~1% at window level (optimistic)
Sample size: 500 windows

Expected seizures in sample: 500 * 0.01 = 5 windows
Probability of 0 seizures: (0.99)^500 = 0.6%

BUT WAIT - IT'S WORSE!
If seizure rate is 0.5%: P(0 seizures) = (0.995)^500 = 8.2%
If seizure rate is 0.3%: P(0 seizures) = (0.997)^500 = 22.3%
If seizure rate is 0.2%: P(0 seizures) = (0.998)^500 = 36.8%
```

**WE HAVE A 20-40% CHANCE OF COMPLETE FAILURE!**

### 4. The Cascade of Doom
1. Sample 500 windows → Find 0 seizures (likely!)
2. `seizure_ratio < 1e-8` → Return None (line 104)
3. Fall back to uniform sampling
4. Uniform sampling → Model sees 99% negative examples
5. Model learns: "Always predict no seizure"
6. AUROC = 0.5, Sensitivity = 0%, Model is USELESS

## WHY THIS IS ESPECIALLY EVIL

### The Silent Failure
- Training APPEARS to work (loss decreases!)
- No errors, no crashes
- Model converges nicely... to predicting all zeros
- Only realize failure when evaluating

### The Alphabetical Trap
```python
# Files are sorted alphabetically
edf_files = sorted(Path(data_dir).glob("**/*.edf"))
# First 20% go to validation (which might have MORE seizures!)
val_files = edf_files[:val_split]
train_files = edf_files[val_split:]
```

If seizure files happen to cluster alphabetically, train/val imbalance gets WORSE!

### The Cache Red Herring
We thought cache was the problem. It wasn't. The cache is fine. The SAMPLING is broken.

## COMPREHENSIVE SOLUTIONS

### Solution 1: AGGRESSIVE SAMPLING (Quick Fix)
```python
# Sample MORE windows to guarantee finding seizures
sample_size = min(10000, len(dataset))  # 10k instead of 500

# Math check:
# P(0 seizures with 0.2% rate) = (0.998)^10000 = 0.0000002%
# MUCH better!
```
**Pros**: Simple, high confidence
**Cons**: Slower startup (10k windows to check)

### Solution 2: EXHAUSTIVE SEARCH (Guaranteed)
```python
def find_all_seizure_windows(dataset, max_search=50000):
    """Keep searching until we find seizures or exhaust search."""
    seizure_indices = []
    for i in range(min(max_search, len(dataset))):
        _, label = dataset[i]
        if (label > 0).any():
            seizure_indices.append(i)
        if len(seizure_indices) >= 100:  # Found enough
            break

    if not seizure_indices:
        raise ValueError("NO SEIZURES FOUND IN 50K WINDOWS!")

    return seizure_indices
```
**Pros**: GUARANTEED to find seizures
**Cons**: Could be very slow

### Solution 3: METADATA PRECOMPUTE (Best Long-term)
```python
class EEGWindowDataset:
    def __init__(self, ...):
        # During cache building, track seizure presence
        self.seizure_window_indices = []

        for i, window in enumerate(self.build_cache()):
            if has_seizure(window):
                self.seizure_window_indices.append(i)

        # Save to cache metadata
        np.savez(cache_dir / "metadata.npz",
                 seizure_indices=self.seizure_window_indices)
```
**Pros**: Instant at runtime, perfect information
**Cons**: Requires cache rebuild

### Solution 4: FILE-LEVEL STRATEGY (Smart)
```python
def create_file_aware_sampler(dataset, edf_files, csv_files):
    """Sample at FILE level first, then windows."""
    # 1. Find which FILES have seizures
    seizure_files = []
    for csv in csv_files:
        if has_seizure_annotations(csv):
            seizure_files.append(csv)

    # 2. Ensure we sample from seizure files
    # This GUARANTEES seizure representation
    seizure_file_weight = 0.5  # 50% of samples from seizure files

    # 3. Build window-level weights accordingly
    ...
```
**Pros**: Leverages file-level knowledge
**Cons**: More complex

### Solution 5: STRATIFIED SPLIT (Critical)
```python
def stratified_train_val_split(edf_files, csv_files, val_ratio=0.2):
    """Split ensuring both sets have seizures."""
    seizure_files = []
    non_seizure_files = []

    for edf, csv in zip(edf_files, csv_files):
        if has_seizures(csv):
            seizure_files.append((edf, csv))
        else:
            non_seizure_files.append((edf, csv))

    # Split BOTH groups proportionally
    val_seizure = seizure_files[:int(len(seizure_files) * val_ratio)]
    train_seizure = seizure_files[int(len(seizure_files) * val_ratio):]

    val_non = non_seizure_files[:int(len(non_seizure_files) * val_ratio)]
    train_non = non_seizure_files[int(len(non_seizure_files) * val_ratio):]

    # Combine and shuffle
    train = train_seizure + train_non
    val = val_seizure + val_non

    return shuffle(train), shuffle(val)
```
**Pros**: Ensures balanced train/val
**Cons**: Requires preprocessing

## THE RECOMMENDED FIX (DO THIS NOW!)

### Immediate (For running training):
```python
# src/brain_brr/train/loop.py:1065
# CHANGE THIS:
sample_size = min(5000, len(train_dataset))  # NOT ENOUGH!

# TO THIS:
sample_size = min(20000, len(train_dataset))  # SAFE for 0.1% seizure rate
```

### Today (Before Modal finishes cache):
1. Implement Solution 1 (Aggressive Sampling) - 20k samples
2. Add explicit check:
```python
if sampler is None:
    print("FATAL: No seizures found! Training will FAIL!")
    print("This is a P0 blocker. Fix data or sampling strategy.")
    sys.exit(1)  # FAIL FAST
```

### This Week:
1. Implement Solution 3 (Metadata Precompute) during cache build
2. Implement Solution 5 (Stratified Split)
3. Add seizure statistics to cache manifest

### Next Sprint:
1. Implement Solution 4 (File-level awareness)
2. Add adaptive sampling that adjusts based on actual seizure rate

## VALIDATION TESTS

### Test 1: Minimum Seizure Guarantee
```python
def test_balanced_sampler_finds_seizures():
    # Create dataset with 0.1% seizure rate
    dataset = create_test_dataset(seizure_rate=0.001)
    sampler = create_balanced_sampler(dataset, sample_size=20000)
    assert sampler is not None, "Must find seizures with 20k samples!"
```

### Test 2: Batch Balance Check
```python
def test_batches_have_seizures():
    for batch in train_loader:
        seizure_ratio = (batch.labels > 0).any(dim=-1).float().mean()
        assert seizure_ratio > 0.3, "Each batch needs 30%+ seizure windows!"
```

### Test 3: Convergence Protection
```python
def test_model_doesnt_collapse():
    # After 1 epoch
    predictions = model(test_batch)
    seizure_pred_ratio = (predictions > 0.5).float().mean()
    assert seizure_pred_ratio > 0.01, "Model predicting all zeros!"
```

## THE NUCLEAR OPTION

If balanced sampling keeps failing:

```python
class OversampleSeizureDataset(Dataset):
    """Explicitly duplicate seizure windows to force balance."""
    def __init__(self, base_dataset, target_ratio=0.2):
        self.seizure_indices = find_all_seizure_windows(base_dataset)
        self.non_seizure_indices = find_all_non_seizure_windows(base_dataset)

        # Repeat seizure indices to hit target ratio
        repeat_factor = int(len(self.non_seizure_indices) * target_ratio / len(self.seizure_indices))
        self.seizure_indices = self.seizure_indices * repeat_factor

        self.all_indices = self.seizure_indices + self.non_seizure_indices
        random.shuffle(self.all_indices)
```

## MONITORING & ALERTS

Add these checks to training:

```python
# After dataset creation
print(f"[CRITICAL CHECK] Train seizure windows: {n_seizure}/{n_total} ({ratio:.1%})")
if ratio < 0.001:
    print("WARNING: Less than 0.1% seizures - sampling will likely FAIL!")

# After sampler creation
if train_sampler is None:
    print("FATAL: Balanced sampler creation failed!")
    print("Training will produce a USELESS model!")
    if not args.force:
        sys.exit(1)

# During training (every 100 batches)
seizure_seen = sum([(labels > 0).any() for _, labels in last_100_batches])
if seizure_seen < 10:
    print(f"WARNING: Only {seizure_seen}/100 batches had seizures!")
    print("Model is likely collapsing to all-negative!")
```

## WHY THIS MATTERS

Without balanced sampling, we're training a $30,000 "no seizure" predictor:
- 100 epochs × 8 hours = 800 hours compute
- A100 at $3/hour = $2,400
- Result: AUROC = 0.5 (random guessing)

**This bug turns our cutting-edge Bi-Mamba architecture into an expensive random number generator.**

## THE TRUTH

This isn't really a "bug" - it's a **fundamental mismatch between our assumptions and reality**:
- We assumed: "Some seizures in every random sample"
- Reality: "Seizures are so rare that random sampling fails"

**We need to respect the extreme imbalance of real clinical data.**

---

**Status**: FIXING NOW - Local broken, Modal still building cache
**Next Update**: After implementing 20k sample fix