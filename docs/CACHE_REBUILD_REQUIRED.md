# ⚠️ CRITICAL: Cache Rebuild Required After mysz Fix

## Immediate Action Required

**ALL EXISTING CACHES MUST BE DELETED AND REBUILT** after the mysz seizure type fix.

## Why This Is Critical

1. **Missing Seizure Data**: The old caches were built without detecting `mysz` (myoclonic) seizures
2. **Wrong Manifest Classification**: Windows containing mysz seizures are incorrectly marked as "no_seizure"
3. **Training Corruption**: Models trained on old caches are missing seizure training data

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

## Verification After Rebuild

After rebuilding caches, verify mysz detection:

```bash
# Check if mysz seizures are now detected in manifest
python -c "
import json
with open('cache/tusz/train/manifest.json') as f:
    manifest = json.load(f)
print(f'Partial seizure windows: {len(manifest[\"partial_seizure\"])}')
print(f'Full seizure windows: {len(manifest[\"full_seizure\"])}')
print(f'No seizure windows: {len(manifest[\"no_seizure\"])}')
"
```

The numbers should change slightly - windows previously marked as "no_seizure" that contain mysz will move to "partial_seizure" or "full_seizure".

## Timeline

1. **v0.1.0**: Built caches without mysz - INCORRECT
2. **v0.2.0**: Fixed parser to include mysz - CORRECT
3. **NOW**: Must rebuild all caches with v0.2.0+ code

## Commands to Rebuild

### Local rebuild:
```bash
# Delete old caches
rm -rf cache/

# Rebuild with smoke test first
python -m src train configs/local/smoke.yaml

# Then full training
python -m src train configs/local/train.yaml
```

### Modal rebuild:
```bash
# Delete caches on Modal volume
modal run --detach deploy/modal/app.py::delete_caches

# Rebuild
modal run --detach deploy/modal/app.py::train --config configs/modal/train_a100.yaml
```

## Impact Assessment

- **Local training cache**: 1589 NPZ files built - MUST REBUILD
- **Modal training**: Unknown number of files - MUST REBUILD
- **Any models trained on old caches**: Potentially missing mysz seizure patterns

---

**DO NOT PROCEED WITH TRAINING UNTIL CACHES ARE REBUILT WITH THE mysz FIX**