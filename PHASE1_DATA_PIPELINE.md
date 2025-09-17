# PHASE1_DATA_PIPELINE.md - EEG Data Loading & Preprocessing

## 🎯 Phase 1 Goal
Build a bulletproof data pipeline that loads any EEG file (TUH/CHB-MIT/custom) and outputs standardized windows ready for model training.

## 📋 Phase 1 Checklist
- [ ] EDF file reader with MNE
- [ ] Channel validation & ordering
- [ ] Signal preprocessing (filters, resample, normalize)
- [ ] Window extraction with metadata
- [ ] Dataset class for PyTorch
- [ ] Validation script on sample files

## 🔧 Implementation Files
```
src/experiment/data.py         # Core data functions
src/experiment/constants.py    # Channel names, sampling rates
tests/test_data.py            # Unit tests
scripts/validate_data.py      # Validation script
```

## 📊 Data Specifications

### Input Requirements
| Specification | Value | Notes |
|--------------|-------|-------|
| **File Format** | EDF/EDF+ | European Data Format |
| **Channels** | 19 (10-20 system) | Fixed order required |
| **Original Fs** | Variable | Usually 250-500 Hz |
| **Duration** | Any | 30s to 24+ hours |

### Output Requirements
| Specification | Value | Notes |
|--------------|-------|-------|
| **Target Fs** | 256 Hz | Fixed for all data |
| **Window Size** | 60 seconds | 15,360 samples |
| **Window Stride** | 10 seconds | 2,560 samples |
| **Data Type** | float32 | Memory efficient |
| **Units** | Microvolts (μV) | Consistent scale |
| **Shape** | (B, 19, 15360) | B = batch size |

## 🏗️ Core Components

### 1. Channel Configuration
```python
# src/experiment/constants.py

CHANNEL_NAMES_10_20 = [
    'Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1',  # Left hemisphere
    'Fz', 'Cz', 'Pz',                                   # Midline
    'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2'   # Right hemisphere
]

SAMPLING_RATE = 256  # Target Hz
WINDOW_SIZE_SEC = 60
STRIDE_SIZE_SEC = 10
WINDOW_SAMPLES = WINDOW_SIZE_SEC * SAMPLING_RATE  # 15,360
STRIDE_SAMPLES = STRIDE_SIZE_SEC * SAMPLING_RATE  # 2,560
```

### 2. EDF Reader Function
```python
# src/experiment/data.py

def load_edf_file(
    file_path: Path,
    target_channels: list[str] = CHANNEL_NAMES_10_20
) -> tuple[np.ndarray, float]:
    """
    Load EDF file and extract specified channels.

    Args:
        file_path: Path to EDF file
        target_channels: Expected channel names in order

    Returns:
        data: Array of shape (n_channels, n_samples)
        fs: Original sampling frequency

    Raises:
        ValueError: If required channels missing
    """
    # Use MNE for robust loading
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose='WARNING')

    # Validate channels exist
    available = [ch for ch in target_channels if ch in raw.ch_names]
    if len(available) != len(target_channels):
        missing = set(target_channels) - set(available)
        raise ValueError(f"Missing channels: {missing}")

    # Extract in exact order
    raw.pick_channels(target_channels, ordered=True)

    # Get data and sampling rate
    data = raw.get_data()  # (n_channels, n_samples)
    fs = raw.info['sfreq']

    # Convert to microvolts if needed
    if raw.info['chs'][0]['unit'] != FIFF.FIFF_UNIT_V:
        data *= 1e6  # Convert to μV

    return data.astype(np.float32), fs
```

### 3. Preprocessing Pipeline
```python
# src/experiment/data.py

def preprocess_recording(
    data: np.ndarray,
    fs_original: float,
    target_fs: int = SAMPLING_RATE,
    notch_freq: int = 60  # or 50 for Europe
) -> np.ndarray:
    """
    Apply standard EEG preprocessing.

    Processing steps:
    1. Resample to target frequency
    2. Bandpass filter (0.5-120 Hz)
    3. Notch filter (50/60 Hz)
    4. Per-channel z-score normalization

    Args:
        data: Raw EEG data (n_channels, n_samples)
        fs_original: Original sampling rate
        target_fs: Target sampling rate (256 Hz)
        notch_freq: Power line frequency to remove

    Returns:
        Preprocessed data (n_channels, n_samples_new)
    """
    from scipy.signal import resample, butter, filtfilt, iirnotch

    # 1. Resample if needed
    if fs_original != target_fs:
        n_samples_new = int(data.shape[1] * target_fs / fs_original)
        data = resample(data, n_samples_new, axis=1)

    # 2. Bandpass filter (0.5-120 Hz)
    nyquist = target_fs / 2
    low_freq, high_freq = 0.5, 120
    b, a = butter(3, [low_freq/nyquist, high_freq/nyquist], btype='band')
    data = filtfilt(b, a, data, axis=1)

    # 3. Notch filter (remove power line noise)
    b_notch, a_notch = iirnotch(notch_freq, Q=30, fs=target_fs)
    data = filtfilt(b_notch, a_notch, data, axis=1)

    # 4. Per-channel z-score normalization
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    data = (data - mean) / (std + 1e-8)  # Avoid division by zero

    # Handle NaN/inf
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    return data.astype(np.float32)
```

### 4. Window Extraction
```python
# src/experiment/data.py

def extract_windows(
    data: np.ndarray,
    window_size: int = WINDOW_SAMPLES,
    stride: int = STRIDE_SAMPLES,
    labels: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """
    Extract sliding windows from continuous data.

    Args:
        data: Preprocessed EEG (n_channels, n_samples)
        window_size: Window size in samples (15,360)
        stride: Stride in samples (2,560)
        labels: Optional labels (n_samples,) for supervised

    Returns:
        windows: Array (n_windows, n_channels, window_size)
        window_labels: Labels per window if provided
        metadata: Dict with start_sample for each window
    """
    n_channels, n_samples = data.shape
    n_windows = (n_samples - window_size) // stride + 1

    # Pre-allocate output
    windows = np.zeros((n_windows, n_channels, window_size), dtype=np.float32)
    metadata = {'start_samples': []}

    # Extract windows
    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        windows[i] = data[:, start:end]
        metadata['start_samples'].append(start)

    # Handle labels if provided
    window_labels = None
    if labels is not None:
        window_labels = np.zeros((n_windows, window_size), dtype=np.float32)
        for i in range(n_windows):
            start = i * stride
            end = start + window_size
            window_labels[i] = labels[start:end]

    return windows, window_labels, metadata
```

### 5. PyTorch Dataset Class
```python
# src/experiment/data.py

class EEGWindowDataset(torch.utils.data.Dataset):
    """PyTorch dataset for EEG windows."""

    def __init__(
        self,
        edf_files: list[Path],
        label_files: list[Path] | None = None,
        cache_dir: Path | None = None,
        transform: callable | None = None
    ):
        """
        Args:
            edf_files: List of EDF file paths
            label_files: Optional list of label files
            cache_dir: Directory to cache preprocessed data
            transform: Optional transform for augmentation
        """
        self.edf_files = edf_files
        self.label_files = label_files
        self.cache_dir = cache_dir
        self.transform = transform

        # Process all files and collect windows
        self.windows = []
        self.labels = []
        self.file_ids = []

        for i, edf_path in enumerate(edf_files):
            # Check cache
            if cache_dir:
                cache_path = cache_dir / f"{edf_path.stem}_windows.npz"
                if cache_path.exists():
                    cached = np.load(cache_path)
                    windows = cached['windows']
                    labels = cached.get('labels')
                else:
                    windows, labels = self._process_file(edf_path, i)
                    np.savez_compressed(cache_path, windows=windows, labels=labels)
            else:
                windows, labels = self._process_file(edf_path, i)

            # Add to dataset
            for w_idx in range(len(windows)):
                self.windows.append(windows[w_idx])
                if labels is not None:
                    self.labels.append(labels[w_idx])
                self.file_ids.append(i)

    def _process_file(self, edf_path: Path, file_idx: int):
        """Process single EDF file."""
        # Load and preprocess
        data, fs = load_edf_file(edf_path)
        data = preprocess_recording(data, fs)

        # Load labels if available
        labels = None
        if self.label_files and file_idx < len(self.label_files):
            labels = self._load_labels(self.label_files[file_idx], data.shape[1])

        # Extract windows
        windows, window_labels, _ = extract_windows(data, labels=labels)

        return windows, window_labels

    def _load_labels(self, label_path: Path, n_samples: int) -> np.ndarray:
        """Load and format labels."""
        # Implementation depends on label format (CSV, TSV, etc.)
        # Return binary array of shape (n_samples,)
        pass

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = torch.from_numpy(self.windows[idx])

        if self.transform:
            window = self.transform(window)

        if self.labels:
            label = torch.from_numpy(self.labels[idx])
            return window, label

        return window
```

## 🧪 Validation Script
```python
# scripts/validate_data.py

def validate_data_pipeline():
    """Validate data pipeline on sample files."""

    # Test files
    test_files = [
        Path("data/samples/tuh_sample.edf"),
        Path("data/samples/chb_sample.edf"),
    ]

    for file_path in test_files:
        print(f"\n{'='*50}")
        print(f"Testing: {file_path}")

        try:
            # Load
            data, fs = load_edf_file(file_path)
            print(f"✅ Loaded: {data.shape} @ {fs} Hz")

            # Preprocess
            processed = preprocess_recording(data, fs)
            print(f"✅ Preprocessed: {processed.shape} @ 256 Hz")

            # Extract windows
            windows, _, metadata = extract_windows(processed)
            print(f"✅ Windows: {windows.shape}")

            # Validate output
            assert windows.dtype == np.float32
            assert windows.shape[1] == 19  # Channels
            assert windows.shape[2] == 15360  # Samples
            assert not np.any(np.isnan(windows))
            assert not np.any(np.isinf(windows))

            print("✅ All validations passed!")

        except Exception as e:
            print(f"❌ Error: {e}")
            raise

    print(f"\n{'='*50}")
    print("🎉 Data pipeline validation complete!")

if __name__ == "__main__":
    validate_data_pipeline()
```

## 📈 Quality Metrics

### Data Quality Checks
- [ ] No NaN or Inf values
- [ ] Channels in correct order
- [ ] Sampling rate exactly 256 Hz
- [ ] Windows have 50-second overlap
- [ ] Z-score normalized (mean≈0, std≈1)

### Performance Targets
- [ ] Load 1-hour EDF < 5 seconds
- [ ] Preprocess 1-hour < 2 seconds
- [ ] Extract windows < 1 second
- [ ] Support parallel loading

## 🚨 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| **Missing channels** | Implement channel mapping/interpolation |
| **Different montages** | Add montage conversion utilities |
| **Corrupted samples** | Add validation and skip bad segments |
| **Memory overflow** | Implement chunked processing |
| **Slow loading** | Add multiprocessing support |

## ✅ Phase 1 Completion Criteria

1. **Code complete**: All functions implemented with type hints
2. **Tests pass**: `pytest tests/test_data.py -v`
3. **Quality check**: `make q` passes
4. **Validation**: Successfully processes 10 sample files
5. **Documentation**: Docstrings for all functions

## 📝 Next Steps
After Phase 1 completion:
1. Move to PHASE2_MODEL_ARCHITECTURE.md
2. Build model components using these data shapes
3. Ensure model input matches data output exactly

---
**Status**: Ready for implementation
**Estimated Time**: 2-3 days
**Dependencies**: MNE, scipy, numpy, torch