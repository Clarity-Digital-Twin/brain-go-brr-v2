# PREPROCESSING STRATEGY: MNE-Python Hybrid Approach

## Executive Decision
**USE MNE-Python for EDF I/O only. Use scipy for signal processing.**

## The Hybrid Pipeline

### Phase 1: Robust File I/O (MNE-Python)
```python
import mne
import numpy as np
from scipy.signal import butter, lfilter, iirnotch, resample

# STEP 1: MNE for bulletproof EDF reading
def load_edf(file_path, channel_order):
    """Use MNE for robust EDF loading and channel selection"""
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose='WARNING')

    # Enforce exact 19-channel 10-20 montage
    CHANNEL_ORDER_19 = ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1',
                         'Fz', 'Cz', 'Pz',
                         'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2']

    # Handle missing channels gracefully
    available = [ch for ch in CHANNEL_ORDER_19 if ch in raw.ch_names]
    if len(available) != 19:
        raise ValueError(f"Expected 19 channels, got {len(available)}")

    raw.pick_channels(available, ordered=True)

    # Extract to numpy and continue with scipy
    data = raw.get_data()  # Shape: (19, n_samples)
    fs = raw.info['sfreq']

    return data, fs
```

### Phase 2: SeizureTransformer-Compatible Processing (scipy)
```python
# STEP 2: Use scipy for exact parity with SeizureTransformer
def preprocess_recording(data, fs_original):
    """Pure scipy processing to match SeizureTransformer exactly"""

    # 2.1 Resample to 256 Hz (scipy)
    if fs_original != 256:
        n_samples_new = int(data.shape[1] * 256 / fs_original)
        data = resample(data, n_samples_new, axis=1)

    # 2.2 Bandpass filter 0.5-120 Hz (scipy butter)
    nyq = 256 / 2
    b, a = butter(3, [0.5/nyq, 120/nyq], btype='band')
    data = lfilter(b, a, data, axis=1)

    # 2.3 Notch filter at 60 Hz (scipy iirnotch)
    b_notch, a_notch = iirnotch(60, Q=30, fs=256)
    data = lfilter(b_notch, a_notch, data, axis=1)

    # 2.4 Per-channel z-score normalization (numpy)
    data = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)

    return data
```

### Phase 3: Custom Windowing
```python
# STEP 3: Our custom windowing logic
def extract_windows(data, window_sec=60, stride_sec=10, fs=256):
    """Extract overlapping windows for model input"""
    window_samples = window_sec * fs  # 15360
    stride_samples = stride_sec * fs   # 2560

    windows = []
    n_samples = data.shape[1]

    for start in range(0, n_samples - window_samples + 1, stride_samples):
        window = data[:, start:start + window_samples]
        windows.append(window)

    return np.array(windows)  # Shape: (n_windows, 19, 15360)
```

## Why This Hybrid Approach?

### MNE-Python Strengths (USE THESE)
- **EDF reading:** Handles corrupted files, varied formats, missing data
- **Channel management:** Robust montage handling, automatic reordering
- **Error handling:** Years of edge cases handled

### scipy/numpy Strengths (USE THESE)
- **Exact reproducibility:** Matches SeizureTransformer's preprocessing
- **Lightweight:** No heavy dependencies for inference
- **Fast:** Optimized C implementations

### What NOT to Use from MNE
- **MNE filtering:** Different defaults than SeizureTransformer
- **MNE resampling:** May differ from scipy.signal.resample
- **MNE epoching:** Our window/stride logic is specific

## Implementation Priority

1. **Start with pure scipy** (like SeizureTransformer) for quick prototype
2. **Add MNE for EDF reading** when you hit real-world file issues
3. **Keep scipy for DSP** to ensure reproducibility

## Dependencies

### Required
```txt
numpy>=1.24.0
scipy>=1.10.0
torch>=2.5.0
```

### Optional (for robust EDF handling)
```txt
mne>=1.5.0  # Only if dealing with messy real-world EDFs
```

## Decision Tree

```
Are you training from scratch?
├─ YES → Use this hybrid approach
│   ├─ MNE for EDF I/O
│   └─ scipy for processing
│
└─ NO (using pretrained) → Must match exact preprocessing
    └─ Use pure scipy like original model
```

## TL;DR

**For our custom architecture (U-Net + ResCNN + Bi-Mamba-2):**
- Use MNE's `read_raw_edf()` for robust file loading
- Use scipy for all signal processing (filtering, resampling)
- This balances robustness (I/O) with reproducibility (DSP)
