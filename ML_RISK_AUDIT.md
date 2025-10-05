# ML / Data Pipeline Risk Audit

## P0 – Dynamic PE hard-codes CUDA autocast
- **Where**: `src/brain_brr/models/gnn_pyg.py:244-274`
- **What**: `_compute_dynamic_pe_vectorized` wraps the eigendecomposition in `torch.amp.autocast("cuda", enabled=False)`. On CPU-only (or MPS-only) builds, PyTorch raises a runtime error as soon as CUDA autocast is referenced, even with `enabled=False`.
- **Impact**: Any run that enables the GNN on a machine without CUDA support crashes before the forward pass. This blocks smoke tests, CI, or CPU training/debug flows.
- **Fix**: Derive the device type from `laplacian.device.type` (e.g. `device_type = laplacian.device.type`) and only enter a CUDA autocast block when that type is actually available; otherwise skip the context entirely.

## P1 – NPZ cache reads rehydrate whole recordings per window fetch
- **Where**: `src/brain_brr/data/datasets.py:258-276`, `src/brain_brr/data/datasets.py:329-376`
- **What**: Each `__getitem__` (both `EEGWindowDataset` and `BalancedSeizureDataset`) calls `np.load(...).astype(...)` on a `.npz` that was written with `np.savez_compressed`. Accessing a single window decompresses the entire `(windows, labels)` arrays (~10–20 MB per recording) on every lookup.
- **Impact**: Training becomes I/O bound with thousands of redundant decompressions per epoch (orders of magnitude slower once the manifest-backed datasets or the sampler iterate repeatedly). Memory spikes and disk thrash show up immediately when multiple DataLoader workers or gradient accumulation are used.
- **Fix**: Store caches in an uncompressed, chunked format (e.g. `.npy` per window block or HDF5/zarr) or open the `.npz` once per process (keep it in memory / memory-map) instead of reloading inside `__getitem__`.

## P1 – Balanced sampler injects random windows as “positives”
- **Where**: `src/brain_brr/train/sampling.py:42-104`
- **What**: After sampling a subset to detect seizures, the code assigns the positive weight to a random subset of *unsampled* windows (`n_unsampled_seizures = int(...); weights[random_indices] = pos_weight`). The vast majority of those windows are background, so the sampler substantially oversamples false negatives.
- **Impact**: Batches skew heavily toward background despite the intent to upweight seizures; precision-focused curriculum collapses, and reproducibility drifts with every run (`torch.randperm`). On large caches this completely defeats the “balanced” fallback.
- **Fix**: Either fall back to uniform sampling when the manifest is unavailable, or perform a second pass to actually inspect additional windows instead of randomly guessing which unsampled entries contain seizures.

---

These issues merit fixes before further optimisation, otherwise CPU/GNN runs fail outright and the refactored data/ML stack spends most of its time undoing its own caching strategy.
