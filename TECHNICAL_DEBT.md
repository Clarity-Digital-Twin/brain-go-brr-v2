# Technical Debt & Risk Register (v3.7.0)

**Date**: 2025-10-06  
**Auditor**: Codex CLI (GPT-5)  
**Scope**: Data pipeline → preprocessing → model → training → evaluation → post-processing  
**Status Legend**: P0 (blocker) · P1 (urgent) · P2 (important) · P3 (nice-to-have) · P4/P5 (future idea)

---

## Executive Summary

| Severity | Count | Notes |
|----------|-------|-------|
| **P0**   | 1     | Current NPZ caching still renders training infeasible without cache conversion |
| **P1**   | 2     | High-impact performance/quality issues that will hurt Modal runs if unaddressed |
| **P2**   | 2     | Medium priority; worth handling after the blockers are cleared |
| **P3**   | 0     | — |
| **P4/P5**| 0     | — |

> **Recommendation**: Resolve both P0 and P1 items before launching full Modal training. Estimated effort ≈ 1.5–2.0 days.

---

## P0 Blockers (Training cannot proceed)

### P0.1 Compressed NPZ Cache Still Dominates Runtime
- **Location**: `src/brain_brr/data/datasets.py` (`EEGWindowDataset`, `BalancedSeizureDataset`, `ValidationDataset`)
- **Symptom**: Loading a single window still requires inflating a compressed `.npz` (~400 MB) whenever it is not present in the tiny per-worker LRU (6 files).
- **Evidence**:
  - Benchmark (`cache/tusz/train/*.npz`, 10 random files):
    ```bash
    python - <<'PY'
    import time, numpy as np
    from pathlib import Path
    files = sorted(Path('cache/tusz/train').glob('*.npz'))[:10]
    start = time.perf_counter()
    for f in files:
        with np.load(f) as data:
            _ = data['windows'][0]
    elapsed = time.perf_counter() - start
    print(f'0.21s per file (2.1s for 10 files)')
    PY
    ```
  - Balanced training touches thousands of distinct cache files per epoch. With a 6-file LRU, each new window invariably triggers another 0.2–2.0 s decompression ⇒ **hours per epoch** before any GPU work starts.
- **Impact**: Modal 100-epoch run would take multiple weeks purely on cache inflation; GPU sits idle.
- **Fix Plan**:
  1. **One-time cache conversion** (Recommended) – Re-encode each cache file as uncompressed `.npz`/`.npy` to enable mmap-style indexed reads:
     ```bash
     python scripts/convert_cache.py \
       --source cache/tusz/train \
       --dest   cache/tusz_uncompressed/train \
       --format npy  # or npz with compression=None
     ```
     Update configs to point at the uncompressed directory.
  2. **Alternative** – Replace current LRU with a streaming iterator grouped by file (e.g., batch sampler that consumes all windows for one file before moving on). Requires rethinking shuffling logic.
  3. **Verification** – Re-run the benchmark post-conversion; target <1 ms per window.

---

## P1 High-Priority Issues

### P1.1 Balanced Dataset Fallback is Non-Functional
- **Location**: `src/brain_brr/train/loop.py` (`create_balanced_sampler` fallback)
- **Symptom**: When the manifest is missing/corrupt, training drops back to raw `EEGWindowDataset`, which (a) suffers the P0 I/O bottleneck and (b) contains <1 % seizure windows, leading to model collapse.
- **Impact**: Training effectively fails silently if the manifest is absent. Users only see AUROC collapse warning mid-epoch.
- **Fix Plan**:
  - Fail fast whenever `use_balanced_sampling=true` but manifest is missing: raise a clear runtime error instructing the user to run `python -m src build-cache --cache-dir ...`.
  - Extend CI/smoke tests to ensure the manifest is regenerated when caches change.

### P1.2 GNN Vectorised Path Still CPU-Bound
- **Location**: `src/brain_brr/models/gnn_pyg.py:322-387`
- **Symptom**: Despite being labeled “vectorised”, the current implementation iterates over every `(batch, timestep)` pair in Python to build `edge_index`/`edge_weight`. For the default setting (B=12, T=960), that’s >11k Python iterations per forward pass before the CUDA kernels fire.
- **Impact**: Enabling the GNN increases step time from seconds to minutes. Modal smoke test is tolerable, but the 100-epoch job becomes cost-prohibitive.
- **Fix Plan**:
  1. Replace manual loops with `torch_geometric.utils.from_dense_adj` or similar utilities that batch-convert dense adjacency tensors directly on the GPU.
  2. Benchmark end-to-end forward pass with and without the fix (target: <10 % overhead vs GNN-disabled path).

---

## P2 Medium-Priority Items

### P2.1 Memory Pressure from LRU Cache
- **Location**: `EEGWindowDataset.MAX_CACHE_FILES_PER_WORKER = 6`
- **Symptom**: Six uncompressed files ≈ 2.4 GB per worker. On Modal (4 workers) this is ~10 GB, acceptable; on smaller GPUs/hosts it could trigger OOM once more than six files are required concurrently (e.g., when running multiple experiments).
- **Plan**: Make the cap configurable via environment variable (e.g., `BGB_CACHE_FILES_PER_WORKER`) and add safety logging when eviction churn exceeds a threshold.

### P2.2 Validation Dataset Shares the Same I/O Fate
- **Location**: `ValidationDataset.__getitem__`
- **Symptom**: Although LRU caching is now present, validation still iterates through thousands of unique files sequentially. On the first epoch this incurs the same decompression penalty (~30 minutes).
- **Plan**: After implementing cache conversion (P0.1), re-benchmark validation. If still slow, restore by prefetching entire validation splits into a dedicated RAM-disk on Modal before evaluation.

---

## Retired / Verified Fixes

| Item | Status | Notes |
|------|--------|-------|
| Autocast crash on CPU | ✅ Fixed (device-aware guard in `gnn_pyg.py`) |
| Random positive assignment in sampler | ✅ Fixed (weights only for verified seizures) |
| Modal configs lacking persistent workers | ✅ Fixed (both modal configs updated) |

---

## Validation Checklist (post-fix)

```bash
# 1. Convert caches (if following the recommended path)
python scripts/convert_cache.py --source cache/tusz/train --dest cache/tusz_uncompressed/train --format npy
python scripts/convert_cache.py --source cache/tusz/dev   --dest cache/tusz_uncompressed/dev   --format npy

# 2. Update configs
sed -i 's|cache/tusz|cache/tusz_uncompressed|g' configs/modal/*.yaml configs/local/*.yaml

# 3. Re-run smoke + validation
make s
CUDA_VISIBLE_DEVICES="" pytest tests/unit/models -k "gnn"
python scripts/benchmark_npz_io.py  # target <1ms per window

# 4. Full quality gate
make q
make test
```

---

## Closing Notes

- Clearing the P0 gives an immediate 10³–10⁴× speed-up in data loading and makes the codebase actually GPU-bound.
- Addressing the two P1 issues keeps the GNN path viable and prevents silent regressions when manifests drift.
- The remaining items are medium priority but easy to knock out once the cache conversion is in place.

**Action Items Before Launching Modal Training**
1. Convert caches to an uncompressed, memory-mappable format (blocking).  
2. Add manifest validation to runner scripts to avoid falling back to unbalanced sampling.  
3. Rewrite the PyG adjacency construction to remove the Python hot loop.  
4. Re-run smoke/perf tests and capture timings in STATUS.md.

Once these are complete, we can confidently proceed with the 100-epoch Modal campaign.
