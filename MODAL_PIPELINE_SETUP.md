# MODAL PIPELINE SETUP - CLOUD GPU TRAINING

## CRITICAL: Modal Pipeline IS READY with all fixes!

### Modal Setup (`deploy/modal/app.py`)

1. **Image includes all dependencies** (line 12-54)
   - PyTorch with CUDA 12.1
   - Mamba-SSM with CUDA kernels
   - All project dependencies
   - **INCLUDES FIXED CODE**: Copies `/src` and `/configs` with CSV parser fixes

2. **Data mounted from S3** (line 68-73)
   ```python
   data_mount = modal.CloudBucketMount(
       "brain-go-brr-eeg-data-20250919",  # Your S3 bucket
       key_prefix="tusz/",
       read_only=True,
   )
   ```

3. **Config uses persistent storage** (`configs/tusz_train_a100.yaml`)
   ```yaml
   data:
     data_dir: /data/edf/train         # S3 mount
     cache_dir: /results/cache/tusz    # Persistent volume
   ```

4. **Training function** (line 90-93)
   ```python
   def train(config_path="configs/tusz_train_a100.yaml", resume=False):
   ```

## HOW IT WORKS:

1. **First run**: Builds cache from S3 data using FIXED parser
2. **Cache persists** in `/results/cache/tusz`
3. **Next runs**: Reuses cache, builds manifest, uses BalancedSeizureDataset

## RUNNING ON MODAL:

```bash
# Deploy and run
modal run deploy/modal/app.py::train

# Run with different config
modal run deploy/modal/app.py::train --config-path configs/smoke_test.yaml

# Resume training
modal run deploy/modal/app.py::train --resume
```

## KEY POINTS:

1. **ALL FIXES INCLUDED**: The Modal image copies your fixed `/src` directory
2. **Parser fixes apply**: CSV_BI format, all seizure types
3. **Same pipeline logic**: Auto-builds cache, manifest, uses balanced dataset
4. **Persistent cache**: Survives between runs on Modal volumes

## CACHE BEHAVIOR ON MODAL:

### First run:
```
/results/cache/tusz/  (empty)
→ Builds cache from /data/edf/train (S3)
→ Uses FIXED CSV parser
→ Creates manifest
→ Saves to persistent volume
```

### Subsequent runs:
```
/results/cache/tusz/  (has cache)
→ Finds existing cache
→ Uses manifest
→ BalancedSeizureDataset with seizures
```

## MODAL vs LOCAL:

| Aspect | Local | Modal |
|--------|-------|-------|
| GPU | RTX 4090 | A100-80GB |
| Data source | `data_ext4/tusz/` | S3: `/data/` |
| Cache location | `cache/tusz/` | `/results/cache/tusz/` |
| Config | `tusz_train_wsl2.yaml` | `tusz_train_a100.yaml` |
| Parser fixes | ✅ Applied | ✅ Applied |
| Balanced dataset | ✅ Works | ✅ Works |

## IMPORTANT NOTES:

1. **S3 data must match local structure**: `edf/train/` with EDFs and CSVs
2. **Modal image rebuilds if `/src` changes**: Picks up all fixes
3. **Cache persists across runs**: No need to rebuild
4. **24-hour timeout**: Modal max runtime

---

**TLDR: Modal pipeline is READY. It has all fixes and will work exactly like local training.**