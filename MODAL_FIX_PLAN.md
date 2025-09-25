# Modal Fix Plan for Patient-Disjoint Splits

## Current Modal Setup ✅
- **S3 Bucket**: `/data/edf` contains `train/`, `dev/`, `eval/` directories
- **Cache Location**: `/results/cache/tusz` with `train/` and `dev/` subdirs
- **Configs**: Already have `split_policy: official_tusz` ✅

## Issues to Fix

### 1. ❌ OLD CACHE CONTAMINATED
The Modal persistent volume `/results/cache/tusz` likely contains:
- Cache built BEFORE patient-disjoint fix
- Patient "aaaaagxr" may have sessions in BOTH train and val
- **MUST DELETE AND REBUILD**

### 2. ⚠️ Cache Directory Confusion
Modal configs show:
- `data.cache_dir: /results/cache/tusz`
- But app.py creates `train/` and `val/` subdirs
- Should align with new `train/` and `dev/` convention

### 3. ✅ Data Directory Correct
- `data.data_dir: /data/edf` - parent directory
- Contains `train/`, `dev/`, `eval/` TUSZ official splits

## Action Plan

### Step 1: Clear Modal Cache
```bash
# In Modal function, run:
rm -rf /results/cache/tusz/*
rm -rf /results/cache/smoke/*
```

### Step 2: Update Modal Configs
Both `configs/modal/smoke.yaml` and `configs/modal/train.yaml`:
- ✅ Already have `split_policy: official_tusz`
- ✅ Already have `data_dir: /data/edf`
- ✅ Cache will rebuild with correct splits

### Step 3: Test Smoke First
```bash
modal run deploy/modal/app.py --action test-mamba  # Verify setup
modal run deploy/modal/app.py --action train --config configs/modal/smoke.yaml
```

### Step 4: Full Training
```bash
modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml
```

## Expected Behavior After Fix

1. **First Run**: Will be slow - builds cache from scratch
2. **Cache Structure**:
   - `/results/cache/tusz/train/` - 579 patients from train/
   - `/results/cache/tusz/dev/` - 53 patients from dev/
3. **Patient Disjointness**: GUARANTEED by official TUSZ splits

## Verification Commands

Add to Modal train function:
```python
# Verify patient disjointness
train_patients = set(p.name for p in Path("/data/edf/train").iterdir())
dev_patients = set(p.name for p in Path("/data/edf/dev").iterdir())
overlap = train_patients & dev_patients
assert len(overlap) == 0, f"PATIENT LEAKAGE: {overlap}"
print(f"✅ VERIFIED: {len(train_patients)} train, {len(dev_patients)} dev patients - NO OVERLAP")
```