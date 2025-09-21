# P0: THE REAL SOLUTION - WHAT SEIZURETRANSFORMER DID RIGHT

**They won 1st place in 2025 competition. We're failing to even train.**

## SEIZURETRANSFORMER'S APPROACH (WHAT WORKS)

### Step 1: Pre-categorize ALL windows BEFORE dataset creation
```python
# BEFORE creating the dataset, scan ALL windows
full_seizure_windows = []    # 100% of window is seizure
partial_seizure_windows = []  # Some seizure in window
no_seizure_windows = []       # No seizure at all

for edf_file, csv_file in all_files:
    windows = extract_windows(edf_file)
    labels = extract_labels(csv_file)

    for window, label in zip(windows, labels):
        seizure_ratio = label.mean()
        if seizure_ratio == 0:
            no_seizure_windows.append(window)
        elif seizure_ratio == 1:
            full_seizure_windows.append(window)
        else:
            partial_seizure_windows.append(window)
```

### Step 2: Create BALANCED dataset
```python
# Use ALL partial seizure windows (they're rare and informative)
dataset = partial_seizure_windows

# Add subset of full seizure windows
n_full = int(0.3 * len(partial_seizure_windows))
dataset += random.sample(full_seizure_windows, n_full)

# Add subset of no-seizure windows
n_no_seizure = int(2.5 * len(partial_seizure_windows))
dataset += random.sample(no_seizure_windows, n_no_seizure)
```

### Result:
- **Guaranteed seizure representation** in EVERY epoch
- **Known class balance** (not hoping to randomly find seizures)
- **Reproducible** training

## OUR BROKEN APPROACH (WHAT FAILS)

### We're doing this:
```python
# Create dataset with ALL windows
dataset = EEGWindowDataset(all_files)  # 250k+ windows

# HOPE to find seizures by random sampling
sampler = create_balanced_sampler(dataset, sample_size=20000)
# If unlucky: NO SEIZURES FOUND → Training fails
```

### Problems:
1. **Random sampling** might miss all seizures (even with 20k samples!)
2. **No pre-knowledge** of which windows have seizures
3. **Wasting time** checking windows during training startup
4. **Non-deterministic** - might work one run, fail the next

## THE CORRECT IMPLEMENTATION

### Option 1: Pre-categorize during cache building (BEST)
```python
class EEGWindowDataset:
    def __init__(self, edf_files, csv_files, cache_dir, balance_mode='seizuretransformer'):
        if balance_mode == 'seizuretransformer':
            # Build or load categorized cache
            cache_manifest = cache_dir / 'manifest.json'

            if not cache_manifest.exists():
                self._build_categorized_cache(edf_files, csv_files, cache_dir)

            # Load pre-categorized indices
            manifest = json.load(open(cache_manifest))
            partial_indices = manifest['partial_seizure']
            full_indices = manifest['full_seizure']
            no_seizure_indices = manifest['no_seizure']

            # Create balanced dataset like SeizureTransformer
            self.indices = partial_indices  # Use ALL partial
            self.indices += random.sample(full_indices, int(0.3 * len(partial_indices)))
            self.indices += random.sample(no_seizure_indices, int(2.5 * len(partial_indices)))
            random.shuffle(self.indices)
```

### Option 2: Quick fix - Scan cache for seizures FIRST
```python
def create_balanced_dataset(cache_dir):
    """Pre-scan all cached files to categorize windows."""
    print("[BALANCE] Pre-scanning cache for seizure distribution...")

    partial_files = []
    full_files = []
    no_seizure_files = []

    for npz_file in cache_dir.glob("*.npz"):
        data = np.load(npz_file)
        labels = data['labels']

        for window_idx in range(labels.shape[0]):
            seizure_ratio = (labels[window_idx] > 0).mean()

            file_window = (npz_file, window_idx)

            if seizure_ratio == 0:
                no_seizure_files.append(file_window)
            elif seizure_ratio > 0.9:  # Mostly seizure
                full_files.append(file_window)
            else:
                partial_files.append(file_window)

    print(f"[BALANCE] Found {len(partial_files)} partial, {len(full_files)} full, {len(no_seizure_files)} no-seizure")

    # Balance like SeizureTransformer
    balanced = partial_files
    balanced += random.sample(full_files, min(len(full_files), int(0.3 * len(partial_files))))
    balanced += random.sample(no_seizure_files, min(len(no_seizure_files), int(2.5 * len(partial_files))))

    return balanced
```

## WHY OUR CURRENT "FIX" IS STILL WRONG

Even with 20k samples, we're still:
1. **Guessing** at runtime instead of KNOWING
2. **Wasting time** checking 20k windows at every training start
3. **Not controlling** the actual class balance in training
4. **Missing the point** - SeizureTransformer doesn't "find" seizures, they PRE-CATEGORIZE

## THE PROFESSIONAL FIX (DO THIS NOW)

### 1. Add categorization to cache building
```python
# When building cache, also build manifest
manifest = {
    'full_seizure': [],     # [(file_idx, window_idx), ...]
    'partial_seizure': [],
    'no_seizure': [],
    'total_windows': 0,
    'seizure_statistics': {}
}
```

### 2. Use manifest for deterministic balanced loading
```python
# At training time, just load the manifest and create balanced dataset
# No random sampling, no hoping, no guessing
```

### 3. Match SeizureTransformer's ratio
- ALL partial seizure windows (most informative)
- 30% of that number from full seizure
- 250% of that number from no-seizure
- This gives roughly 1:3 seizure:background ratio

## THE UGLY TRUTH

**We've been trying to solve the wrong problem.**

We thought: "Sample more windows to find seizures"
Reality: "Pre-categorize ALL windows, then sample strategically"

SeizureTransformer didn't get lucky - they were SYSTEMATIC.

---

**Next Step**: Implement Option 2 (quick scan) NOW while cache is ready, then implement Option 1 for future runs.