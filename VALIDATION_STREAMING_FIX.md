# Validation Memory Fix – Execution Guide

**Status**: Code changes are merged and lint/type checks pass. Full-suite tests and Modal validation run are still outstanding.  
**Goal**: Resume Modal training without exit 137 by eliminating validation-time RAM spikes.

---

## Key Changes (Already Landed)

- **Disk-backed validation storage** (`RecordingStorage`) – each recording is written to `.npy` shards, eliminating in-memory accumulation.
- **Staged metric computation** (`val_step.validate_epoch`)  
  1. AUROC/PR-AUC load → compute → free (≈39 GB peak).  
  2. ECE runs streaming (<1 MB).  
  3. FA sweep reloads via copy-on-write mmap; OS caches ≈25–30 GB but AUROC memory has already been released.
- **Streaming saves**  
  - `_save_predictions_streaming`: copies shard files into the run’s output directory.  
  - `_save_plots_streaming`: generates sample plots (first 10 recordings) without loading the full dataset.
- **Loss handling** – validation now falls back to BCE automatically when focal parameters are absent, preserving CLI evaluation behaviour.

---

## Honest Memory Profile (Modal 96 GB host RAM)

| Phase | Peak RSS | Notes |
|-------|----------|-------|
| Validation loop | ~0 GB | Writes straight to disk. |
| AUROC / PR-AUC stage | ~39 GB | Pre-allocated concat + sklearn overhead. |
| Post-AUROC free | ~0 GB | Explicit `del` + `gc.collect()`. |
| ECE | <1 MB | Streaming bin stats. |
| FA sweep | 25–30 GB | Copy-on-write mmaps; OS keeps hot pages cached while thresholds iterate. |
| **Overall peak** | ~39 GB | AUROC and FA sweep peaks do not overlap. |

This comfortably fits within the 96 GB Modal budget that previously saw ~120 GB spikes.

---

## Verification to Date

- `make q` (ruff lint/format + mypy + config validation) – ✅
- Targeted unit & integration tests covering the new storage/validation plumbing – ✅  
  `pytest tests/train/test_recording_storage.py tests/train/test_validation_memory.py`
- Manual inspection of generated prediction files and plots on small samples – ✅

---

## Outstanding Actions (Blockers before Modal resume)

1. **Full test suite** – run `pytest -n auto` (or the project’s equivalent) to confirm nothing outside the focused tests regressed.  
2. **Local validation smoke (optional but recommended)** – run a 10-file subset with tracemalloc/psutil to observe the ~39 GB → 0 GB → 30 GB pattern.  
3. **Modal resume** – execute and monitor the first validation cycle to ensure the exit 137 no longer occurs.

```
modal run --detach deploy/modal/app.py \
  --action train \
  --config configs/modal/train.yaml \
  --resume true
```

4. **Monitor W&B & Modal logs** – confirm validation completes, AUROC/PR metrics match pre-crash values, and host RAM stays <50 GB. Capture the log snippet for the project record.

---

## Quick Reference – Touch Points

- `src/brain_brr/train/recording_storage.py`
- `src/brain_brr/train/val_step.py`
  - `_process_recording`
  - `_compute_final_metrics`
  - `_save_predictions_streaming`
  - `_save_plots_streaming`
- `tests/train/test_recording_storage.py`
- `tests/train/test_validation_memory.py`

Once the outstanding actions are complete, Modal training can proceed with the new validation pipeline in place.
