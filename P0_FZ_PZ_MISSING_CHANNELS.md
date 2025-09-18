# P0: Fz/Pz Missing Channels in Dev Set

**STATUS**: 🔴 BLOCKING TRAINING
**PRIORITY**: P0 - Highest
**DATE**: 2025-09-18
**ERROR**: `ValueError: Missing required channels: ['Fz', 'Pz']`

## Executive Summary

Training crashes immediately when loading first EDF file from dev set. The system successfully found 1466 training files but fails on channel validation claiming Fz and Pz are missing. This is the FINAL barrier to training.

## Error Context

```
Loading 1466 train, 366 val files
File: src/experiment/data.py, line 165
ValueError: Missing required channels: ['Fz', 'Pz']
```

## Initial Hypotheses

### 1. Canonicalization Bug (MOST LIKELY)
- Our canonicalization might be filtering out Fz/Pz incorrectly
- These channels might have different prefixes/suffixes in dev set
- Possible variations: "EEG FZ-REF", "EEG PZ-REF", "FZ-LE", "PZ-LE"

### 2. MNE Handling Issue
- MNE might be reading these channels differently
- Channel names might be getting modified during MNE's read process

### 3. Actual Missing Channels (UNLIKELY)
- Based on our previous analysis showing ALL files have all 19 channels
- But dev set might be different from our sample analysis

### 4. Case Sensitivity Issue
- Fz/Pz vs FZ/PZ handling in canonicalization
- The 'z' suffix channels might need special handling

## Investigation Plan

1. **Identify the failing file**
   - First file in dev set that triggers the error
   - Get exact channel names from that file

2. **Debug canonicalization**
   - Print raw channel names before canonicalization
   - Track what happens to Fz/Pz during canonicalization
   - Check if they're being filtered as non-EEG

3. **Test with MNE directly**
   - Load the file with raw MNE
   - List all channels without any processing
   - Verify Fz/Pz presence

4. **Review canonicalization logic**
   - Special handling for single-letter suffix channels (Fz, Cz, Pz, Oz)
   - Ensure they're not caught by non-EEG filters

## Potential Solutions

### Solution A: Fix Canonicalization
```python
def _to_canonical(name: str) -> str | None:
    s = name.strip().upper()

    # DON'T filter if it's a potential EEG channel with Z suffix
    if any(s.startswith(ch) or f"EEG {ch}" in s for ch in ["FZ", "CZ", "PZ", "OZ"]):
        # These are definitely EEG channels
        pass
    elif any(s.startswith(p) for p in non_eeg_prefixes):
        return None

    # Continue with canonicalization...
```

### Solution B: More Robust Channel Matching
```python
# Add special cases for z-line channels
canonical_map.update({
    'FZ': 'Fz', 'CZ': 'Cz', 'PZ': 'Pz', 'OZ': 'Oz'
})
```

### Solution C: Debug Mode First
Add verbose logging to understand exactly what's happening:
```python
print(f"Raw channels from MNE: {raw.ch_names}")
print(f"After canonicalization: {canonical_names}")
print(f"Missing: {set(CHANNEL_NAMES_10_20) - set(available)}")
```

## Quick Debug Script

```python
# debug_fz_pz.py
import mne
from pathlib import Path

# Get first dev file
dev_files = list(Path('data/tusz/edf/dev').glob('**/*.edf'))
test_file = dev_files[0]

print(f"Testing: {test_file}")

# Load with MNE
raw = mne.io.read_raw_edf(test_file, preload=False, verbose=False)
print(f"Channel count: {len(raw.ch_names)}")
print(f"Channel names: {raw.ch_names}")

# Check for Fz/Pz variations
fz_variants = [ch for ch in raw.ch_names if 'FZ' in ch.upper()]
pz_variants = [ch for ch in raw.ch_names if 'PZ' in ch.upper()]

print(f"\nFz variants found: {fz_variants}")
print(f"Pz variants found: {pz_variants}")

# Test our canonicalization
from src.experiment.data import _to_canonical, CHANNEL_NAMES_10_20

canonical_names = {}
for ch in raw.ch_names:
    canon = _to_canonical(ch)
    if canon:
        canonical_names[ch] = canon
        if canon in ['Fz', 'Pz']:
            print(f"  {ch} -> {canon}")

missing = set(CHANNEL_NAMES_10_20) - set(canonical_names.values())
print(f"\nMissing after canonicalization: {missing}")
```

## The Mystery

Why do Fz and Pz specifically fail when:
1. Our previous analysis showed all channels present
2. Other channels (F3, F4, etc.) work fine
3. The 'z' suffix channels are standard 10-20

**Theory**: The 'z' midline channels might have unique formatting in TUSZ that our canonicalization doesn't handle.

## Action Items

- [ ] Run debug script to identify exact channel names
- [ ] Fix canonicalization to handle Fz/Pz variants
- [ ] Test on multiple dev files
- [ ] Verify all 19 channels load correctly
- [ ] Resume training

## Bottom Line

This is likely a simple canonicalization bug where Fz/Pz are either:
1. Being filtered out as non-EEG (if they have unexpected prefixes)
2. Not matching our canonical map (case sensitivity or format issue)
3. Have TUSZ-specific naming we haven't accounted for

Let's debug and fix this final barrier to training!