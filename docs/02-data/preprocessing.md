# Preprocessing and Windowing

Pipeline

- Bandpass 0.5–120Hz, 60Hz notch
- Resample to 256Hz
- Windows: 60s with 10s stride (83% overlap)
- Per-channel z-score normalization

Where implemented

- Loader: `src/brain_brr/data/loader.py`
- Dataset: `src/brain_brr/data/dataset.py`
