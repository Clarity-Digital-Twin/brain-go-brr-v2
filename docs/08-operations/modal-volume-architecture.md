# Modal Volume Architecture (Mmap Edition)

**Last Updated**: October 6, 2025
**Goal**: Keep the Modal A100 runtime fast and reliable now that the cache uses memory-mapped NPY files.

---

## 1. Cache Locations

| Environment | Path | Contents |
|-------------|------|----------|
| Local | `cache/tusz_mmap/{train,dev}` | `_data.npy` / `_labels.npy` pairs + `manifest.json` + `_dataset_index.json` |
| Modal SSD | `/results/cache/tusz_mmap/{train,dev}` | Same structure as local (populated once, reused) |
| S3 Backup | `s3://brain-go-brr-eeg-data-20250919/cache/tusz_mmap/` | Authoritative backup used by `populate-cache` |

All configs (local + Modal) now point at the mmap cache paths. The original NPZ cache (`cache/tusz/`) should be treated as a legacy backup only.

---

## 2. Why the mmap cache matters

- **Zero-copy reads**: DataLoader workers open files with `np.load(..., mmap_mode='r')`; the OS page cache keeps hot pages in RAM and evicts cold pages automatically.
- **Shared pages**: Multiple workers (and epochs) reuse the same cached pages, keeping per-worker RSS below 2 GB instead of the 85 GB required by compressed NPZ queues.
- **Latency**: Window access falls from ~1.1 s (decompress NPZ) to ~0.01 ms (mmap page fault), which is critical for validation and Modal throughput.

---

## 3. Modal volume layout

```
/results/
├── cache/
│   └── tusz_mmap/        # Memory-mapped cache (train/dev)
├── checkpoints/          # Optional global checkpoint stash
├── smoke/                # Smoke job outputs (small)
├── train/                # Full training outputs (checkpoints, TB, W&B)
└── tensorboard/, wandb/  # Created on demand
```

Guidelines:
- Keep **only** the mmap cache under `/results/cache/`; delete the legacy NPZ cache (`/results/cache/tusz/`) to free ~450 GB once you have migrated.
- Prune old training runs (checkpoints, TB logs, W&B artifacts) regularly so the 958 GiB quota is never exhausted before new populate jobs run.

---

## 4. Function mounts

```python
@app.function(
    volumes={
        "/data": data_mount,              # S3 (EDF corpus, read-only)
        "/results": results_volume,       # Modal SSD with cache + outputs
    }
)
```

- `/data` remains a CloudBucketMount pointed at `tusz/edf/`; streaming EDFs from S3 is fine.
- `/results` is the persistent SSD volume holding both the mmap cache and run artefacts.

---

## 5. Populate / clean workflow

### One-time populate

```bash
# Copy mmap cache from S3 → Modal SSD (use --detach)
modal run --detach deploy/modal/app.py --action populate-cache
```

The populate routine:
1. Mounts the S3 bucket at `/s3_cache/cache/tusz_mmap`.
2. Removes existing `/results/cache/tusz_mmap` directories (train + dev).
3. Copies train and dev splits, including manifests and index files.
4. Logs counts of `_data.npy` / `_labels.npy` pairs for verification.

### Cleaning the cache

```bash
modal run deploy/modal/app.py --action clean-cache
```

Use this before repopulating if you suspect corruption or want to reclaim space. Training itself never removes cache files; only the populate/clean commands manage them.

### Remove stray NPZ files after aborted jobs

```bash
modal run deploy/modal/clean_stray_npz.py --confirm
```

This utility deletes any lingering `*_windows.npz` files that might be left behind if a run was started before the mmap cache was copied. It verifies the matching `_data.npy/_labels.npy` pair exists before deleting, so it is safe to run when cache checks warn about NPZ contamination.

---

## 6. When to copy vs stream

| Scenario | Recommendation |
|----------|----------------|
| Standard 100-epoch run | Copy to `/results/cache/tusz_mmap` and reuse (fastest, proven) |
| Experimenting with new cache | Rebuild locally → upload to S3 → populate Modal |
| Emergency / no SSD space | You can experiment with mounting S3 directly, but expect higher latency and potential throttling. Clean old runs first instead. |

If populate-cache fails with “No space left on device”, delete legacy directories and stale checkpoints before retrying:
```bash
modal run deploy/modal/app.py --action clean-cache                # optional
modal volume ls brain-go-brr-results                              # inspect size
modal run deploy/modal/app.py --action cleanup-old-runs --days 14  # custom util if available
```

---

## 7. Smoke vs full training

- **Smoke jobs** (`configs/modal/smoke.yaml`) use the same cache but limit file consumption with `BGB_LIMIT_FILES=50`.
- **Full training** (`configs/modal/train.yaml`) accesses every file and relies on the mmap cache for performance.
- No separate “smoke cache” is required; keeping a single mmap cache avoids duplication.

---

## 8. Quick health checks

```bash
# Verify cache file counts
modal run deploy/modal/app.py --action check-cache

# Inspect volume usage
modal volume ls brain-go-brr-results

# Download manifests for inspection
modal run deploy/modal/app.py --action dump-manifest --split train > train_manifest.json
```

Use these before expensive runs or after cleanups to ensure the cache is populated and manifests are present.

---

## 9. FAQ

**Q: Why not stream mmap files directly from S3?**  
A: Page faults would incur S3 GET requests; balanced sampling touches thousands of files per epoch. The SSD cache keeps everything hot and avoids network noise.

**Q: Do I still need the NPZ cache?**  
A: Only as a legacy backup. Once the mmap migration is stable, delete `/results/cache/tusz/` on Modal and `cache/tusz/` locally to reclaim space.

**Q: How big should the Modal volume be?**  
A: Keep at least 600 GB free (≈500 GB cache + headroom for checkpoints). Routine cleanup of old runs prevents populate jobs from failing.

---

By keeping the mmap cache warm on the Modal SSD and pruning legacy artefacts, the A100 jobs stay fast and predictable.
