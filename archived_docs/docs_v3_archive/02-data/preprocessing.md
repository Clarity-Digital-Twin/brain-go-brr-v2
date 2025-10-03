# Preprocessing and Windowing

Goals

- Standardize signals to 256Hz, remove drift/noise, normalize channels, and produce fixed windows.

Steps (in order)

1) Load EDF via MNE (`read_raw_edf`) with preload; repair rare header issues (colon date separator) by in‑place correction on a temp copy.
2) Clean channel names (TUSZ style like `EEG FP1-LE` → `FP1`), apply case normalization, and synonyms (T7→T3, T8→T4, P7→T5, P8→T6).
3) Ensure canonical channel set; best‑effort interpolate missing midline channels (Fz, Pz) using standard_1020 montage when needed.
4) Resample to 256Hz using `scipy.signal.resample`.
5) Bandpass 0.5–120Hz (Butterworth order=3) via `lfilter`.
6) Notch filter at 60Hz (`iirnotch`, Q=30) via `lfilter`.
7) Per‑channel z‑score normalization; then clip outliers to ±10σ to prevent extreme artifacts; sanitize NaN/Inf to 0.0; cast to float32.
8) Convert to windows with `window_size=60s` and `stride=10s`; labels sliced to window extents when available.

Shapes and dtypes

- Continuous recording returns `(n_channels, n_samples)` in µV as float32.
- Windows: `(n_windows, 19, 15360)`; labels `(n_windows, 15360)` if present.

Label handling (TUSZ CSV_BI)

- Parser reads `channel,start,stop,label,confidence`; builds binary seizure mask at 256Hz.
- Default seizure labels: `{seiz, gnsz, fnsz, cpsz, absz, spsz, tcsz, tnsz, mysz}`.

Code anchors

- I/O and CSV: `src/brain_brr/data/io.py`
- Preprocessing: `src/brain_brr/data/preprocess.py`
- Windowing: `src/brain_brr/data/windows.py`
- Constants (channels, rates): `src/brain_brr/constants.py`
