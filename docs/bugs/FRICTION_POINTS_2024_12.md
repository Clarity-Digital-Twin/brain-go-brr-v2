# Friction Points & Bugs - December 2024 Session

Documented during FLA Exp4 training cycle completion and eval run.

## 1. CUDA Async Race Condition (FIXED)

**Symptom**: Training crashed intermittently with "CUDA error: unknown error" at random batch numbers.

**Root Cause**: FLA Triton kernels (`chunk_gated_delta_rule`) racing for memory under concurrent bidirectional execution.

**Fix Applied**: Added `torch.cuda.synchronize()` calls around FLA forward passes in `gated_deltanet.py`.

**PR**: #2 (merged)

**Status**: FIXED

---

## 2. `build-cache` CLI Doesn't Actually Build NPY Cache

**Symptom**: `python -m src build-cache --split eval` creates only `_dataset_index.json`, not NPY cache files.

**Expected**: Should create `*_data.npy` and `*_labels.npy` files.

**Actual**: Only creates index file, says "Cache incomplete: X missing after build".

**Workaround**: Eval command can work directly with EDF files + index.

**Fix Needed**: Either:
- Make `build-cache` actually create NPY files, OR
- Rename command to `build-index` to avoid confusion

**Priority**: Medium

---

## 3. `build-cache --split` Parameter Confusion

**Symptom**: Running `build-cache --data-dir data_ext4/tusz/edf --split eval` processes ALL files from index, not just eval split.

**Expected**: Should only process files in `data_ext4/tusz/edf/eval/`.

**Actual**: Loads stale global index and processes all 7364 files instead of 865 eval files.

**Workaround**: Must specify split-specific paths:
```bash
# Wrong
python -m src build-cache --data-dir data_ext4/tusz/edf --split eval

# Correct
python -m src build-cache --data-dir data_ext4/tusz/edf/eval --cache-dir cache/tusz_mmap/eval --split eval
```

**Fix Needed**: `--split` parameter should automatically scope to correct subdirectory.

**Priority**: High (confusing UX)

---

## 4. Stale `_dataset_index.json` Causes Wrong Split Processing

**Symptom**: Build command loads cached index from parent dir instead of building fresh for specified split.

**Root Cause**: Global `cache/tusz_mmap/_dataset_index.json` takes precedence over split-specific index.

**Workaround**: Manually delete stale index: `rm cache/tusz_mmap/_dataset_index.json`

**Fix Needed**: Index should be scoped per-split, or `--split` should ignore global index.

**Priority**: High

---

## 5. `evaluate` CLI Expects EDF Dir, Not Cache Dir

**Symptom**: `python -m src evaluate checkpoint.pt cache/tusz_mmap/eval` fails with "No EDF files found".

**Expected**: Should work with cache directory since we have preprocessed data.

**Actual**: Requires raw EDF directory path.

**Workaround**: Use EDF path directly:
```bash
python -m src evaluate checkpoint.pt data_ext4/tusz/edf/eval
```

**Fix Needed**: Either:
- Accept cache dir and use cached NPY files for faster eval, OR
- Update docs/help text to clarify EDF dir required

**Priority**: Medium

---

## 6. GNN `last_valid_pe` Buffer Shape Mismatch on Load

**Symptom**: Warning on checkpoint load:
```
[CHECKPOINT] Skipping dynamic buffer with shape mismatch: gnn.last_valid_pe
(checkpoint: torch.Size([8, 192, 19, 16]), model: torch.Size([1, 1, 1, 16]))
```

**Root Cause**: Dynamic PE buffer is saved with batch-specific shape, doesn't match new model init.

**Impact**: PE fallback mechanism may not work correctly after resume.

**Fix Needed**: Either:
- Don't save dynamic buffers in checkpoint, OR
- Handle shape mismatch gracefully during load

**Priority**: Low (doesn't affect training, just warning)

---

## 7. WandB Cloud Runs Accumulate Without Auto-Cleanup

**Symptom**: WandB cloud storage fills up with crashed/failed runs.

**Workaround**: Manual cleanup via Python API:
```python
import wandb
api = wandb.Api()
run = api.run('entity/project/run_id')
run.delete()
```

**Fix Needed**: Consider adding `make clean-wandb` or auto-delete crashed runs.

**Priority**: Low

---

---

## 8. Eval Doesn't Pass Label Files to Dataset (FIXED)

**Symptom**: Eval returned AUROC=0.5, Sensitivity=0% on all metrics.

**Root Cause**: `evaluation.py` didn't pass `label_files` to `EEGWindowDataset`, so all labels were zeros.

**Fix Applied**: Added label file discovery and passing in `evaluation.py:create_dataloader()`.

**Status**: FIXED

See `EVAL_LABELS_BUG_2024_12.md` for full details.

---

## 9. `_load_labels` Doesn't Recognize `.csv_bi` Suffix (FIXED)

**Symptom**: Even with label files passed, labels were all zeros.

**Root Cause**: `datasets.py:_load_labels()` only checked for `.csv` suffix, but TUSZ uses `.csv_bi`.

**Fix Applied**: Changed suffix check from `== ".csv"` to `in (".csv", ".csv_bi")`.

**Status**: FIXED

See `EVAL_LABELS_BUG_2024_12.md` for full details.

---

## Summary

| Bug | Severity | Status |
|-----|----------|--------|
| CUDA race condition | High | FIXED |
| Eval doesn't pass label files | **CRITICAL** | **FIXED** |
| `_load_labels` suffix check | **CRITICAL** | **FIXED** |
| build-cache doesn't build cache | Medium | TODO |
| --split parameter confusion | High | TODO |
| Stale index precedence | High | TODO |
| evaluate expects EDF not cache | Medium | TODO |
| PE buffer shape mismatch | Low | TODO |
| WandB run accumulation | Low | TODO |
