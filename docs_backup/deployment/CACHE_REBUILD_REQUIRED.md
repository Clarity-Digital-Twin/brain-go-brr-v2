# ⚠️ CRITICAL: Cache Rebuild Required (CSV_BI + seizure labels)

STATUS: In progress — caches are currently rebuilding locally and on Modal. Do not start training until rebuilds complete and manifests show seizures (partial>0 or full>0).

## Immediate Action Required

**ALL EXISTING CACHES MUST BE DELETED AND REBUILT** after fixing CSV_BI parsing and completing the seizure label set (including `mysz`).

## Why This Is Critical

1. CSV_BI columns were misread in prior builds, causing all-zero labels in some caches; fixed parser now reads `channel,start,stop,label,confidence`.
2. Seizure labels expanded to include rare types like `mysz` (myoclonic); old caches missed these windows.
3. Training correctness requires rebuilt caches with correct labels and balanced manifest.

## Affected Files

### Files with mysz seizures in training set:
- `aaaaajrs_s001_t000.csv` - contains mysz events
- `aaaaajrs_s001_t001.csv` - contains mysz events

While only 2 files in train have mysz (44 total events across all TUSZ), these represent RARE seizures that are critical for model completeness.

## Cache Locations to Delete

### Local Caches
```bash
# DELETE ALL OF THESE:
rm -rf /home/jj/proj/brain-go-brr-v2/cache/tusz/
rm -rf /home/jj/proj/brain-go-brr-v2/cache/smoke/
rm -rf /home/jj/proj/brain-go-brr-v2/cache/dev/
rm -rf /home/jj/proj/brain-go-brr-v2/cache/eval/
```

### Modal Caches
```bash
# On Modal volumes:
rm -rf /results/cache/tusz/
rm -rf /results/cache/smoke/
rm -rf /results/cache/dev_tuning/
rm -rf /results/cache/eval_final/
```

## How Caches Are Built

The cache building pipeline:
1. Reads EDF files and CSV annotations
2. Calls `parse_tusz_csv()` to extract events
3. Calls `events_to_binary_mask()` which uses seizure_labels set
4. **OLD**: `{"seiz", "gnsz", "fnsz", "spsz", "cpsz", "absz", "tnsz", "tcsz", "spkz"}` - MISSING mysz!
5. **NEW**: `{"seiz", "gnsz", "fnsz", "cpsz", "absz", "spsz", "tcsz", "tnsz", "mysz"}` - COMPLETE!

## Verification After Rebuild (required)

After rebuilding caches, verify mysz detection:

```bash
# Check seizure windows exist in manifest
python -c "
import json
with open('cache/tusz/train/manifest.json') as f:
    manifest = json.load(f)
print(f'Partial seizure windows: {len(manifest[\"partial_seizure\"])}')
print(f'Full seizure windows: {len(manifest[\"full_seizure\"])}')
print(f'No seizure windows: {len(manifest[\"no_seizure\"])}')
"
```

Expect partial>0 and/or full>0. If both are zero, stop and investigate CSV paths/parsing.

## Timeline

1. **v0.1.0**: Built caches without mysz - INCORRECT
2. **v0.2.0**: Fixed parser to include mysz - CORRECT
3. **NOW**: Rebuilding caches with fixed parser + labels (in progress)

## Commands to Rebuild

### Local rebuild:
```bash
# Delete old caches
rm -rf cache/

# Build manifest on existing cache (optional)
python -m src scan-cache --cache-dir cache/tusz/train

# Smoke test (1 epoch)
python -m src train configs/smoke_test.yaml

# Full training (WSL2-safe)
python -m src train configs/tusz_train_wsl2.yaml
```

### Modal rebuild:
```bash
# Delete caches on Modal volume
modal run --detach deploy/modal/app.py::delete_caches

# Rebuild
modal run --detach deploy/modal/app.py -- --action train --config configs/tusz_train_a100.yaml
```

## Impact Assessment

- **Local training cache**: 1589 NPZ files built - MUST REBUILD
- **Modal training**: Unknown number of files - MUST REBUILD
- **Any models trained on old caches**: Potentially missing mysz seizure patterns

---

**DO NOT PROCEED WITH TRAINING UNTIL SCAN SHOWS partial>0 OR full>0 AND THE BALANCED DATASET LOADS WITHOUT ERROR**
