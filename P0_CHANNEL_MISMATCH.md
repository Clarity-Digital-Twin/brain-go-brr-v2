# P0 BLOCKER: TUSZ Channel Name Mismatch

**STATUS**: 🔴 BLOCKING TRAINING
**PRIORITY**: P0 - Highest
**ETA**: 2-4 hours
**DATE**: 2025-09-18

## The Problem

Our training pipeline expects standard 10-20 channel names but TUSZ uses different naming conventions:
- **Expected**: `Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6, Fz, Cz, Pz`
- **TUSZ Has**: `EEG FP1-REF, EEG FP2-REF, EEG F3-REF, EEG F4-REF...` or `01_tcp_ar` montages

**Error**: `ValueError: Missing required channels: ['C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'Fz', 'O1', 'O2', 'P3', 'P4', 'Pz', 'T3', 'T4', 'T5', 'T6']`

## Evidence from Investigation

### 1. SeizureTransformer Solution
From `reference_repos/seizure-transformer-braingobrr`:
- Uses `epilepsy2bids.Eeg.loadEdfAutoDetectMontage()` - but this is for their specific format
- Has EDF header repair utilities for malformed dates
- **Key insight**: They handle channel mapping INSIDE their loader

### 2. NEDC-Bench Analysis
From `reference_repos/nedc-bench`:
- Works with CSV_BI annotations only (channel="TERM")
- Doesn't handle raw EDF signal processing
- No channel mapping utilities found

### 3. TUSZ Dataset Structure
```
data/tusz/edf/dev/aaaaaajy/s001_2003/02_tcp_le/  # tcp_le montage
data/tusz/edf/dev/aaaaaajy/s002_2003/01_tcp_ar/  # tcp_ar montage
```

## Root Cause

TUSZ EDF files use Temple's channel naming:
1. **Referential montages**: `EEG <CHANNEL>-REF` (e.g., `EEG FP1-REF`)
2. **TCP montages**: `01_tcp_ar` (average reference) or `02_tcp_le` (linked ears)
3. **Case differences**: `FP1` vs `Fp1`, `T3/T4` vs `T7/T8` in newer standards

## Solutions (Ranked by Feasibility)

### Solution 1: Use MNE Fallback (RECOMMENDED) ✅
**Time**: 2 hours
```python
# In src/experiment/data.py
def load_edf_file(edf_path):
    # Try MNE first - it handles channel standardization
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    raw.pick_channels(raw.ch_names[:19])  # Take first 19 channels
    raw.rename_channels(lambda x: x.replace('EEG ', '').replace('-REF', '').upper())
    # Map to standard names
    channel_map = {
        'FP1': 'Fp1', 'FP2': 'Fp2',
        'T3': 'T3', 'T4': 'T4', 'T5': 'T5', 'T6': 'T6',
        # ... complete mapping
    }
    raw.rename_channels(channel_map)
```

### Solution 2: Channel Name Mapping
**Time**: 3 hours
```python
# Create flexible channel matcher
def find_channel_match(available_channels, target):
    # Handle variations: Fp1, FP1, EEG FP1-REF, etc.
    for ch in available_channels:
        clean = ch.upper().replace('EEG', '').replace('-REF', '').strip()
        if target.upper() in clean:
            return ch
    return None
```

### Solution 3: Import SeizureTransformer's Loader
**Time**: 4+ hours (requires adapting their epilepsy2bids dependency)
- Copy their `Eeg.loadEdf()` implementation
- Risk: Introduces external dependencies

## Immediate Action Items

1. **IMPLEMENT SOLUTION 1** - MNE fallback with channel mapping
2. **TEST** on a few TUSZ files to verify channels load correctly
3. **UPDATE** EEGWindowDataset to use new loader
4. **VALIDATE** shapes match expected (19, n_samples)

## Testing Commands

```bash
# Test channel loading
python -c "
from src.experiment.data import load_edf_file
data, fs = load_edf_file('data/tusz/edf/dev/aaaaaajy/s001_2003/02_tcp_le/aaaaaajy_s001_t000.edf')
print(f'Loaded shape: {data.shape}')
"

# Then run smoke test
python -m src.experiment.pipeline --config configs/smoke_test.yaml
```

## Success Criteria

✅ `load_edf_file()` returns (19, n_samples) for TUSZ files
✅ Channel order matches CHANNEL_NAMES_10_20
✅ Training pipeline runs without channel errors
✅ Smoke test completes 1 epoch

## Notes

- SeizureTransformer solved this but their solution is tightly coupled to epilepsy2bids
- MNE is already a dependency and handles this gracefully
- This is THE ONLY blocker for training on TUSZ

---

**Bottom Line**: We need channel name mapping. MNE can handle this in 2 hours. Let's do it.