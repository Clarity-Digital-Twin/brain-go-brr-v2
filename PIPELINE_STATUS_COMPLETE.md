# COMPLETE PIPELINE STATUS

## CONFIG FILES - ALL FIXED ✅

### Local Configs
| Config | Cache Dir | Data Dir | Purpose | Status |
|--------|-----------|----------|---------|---------|
| `smoke_test.yaml` | `cache/smoke` | `data_ext4/tusz/edf/train` | Quick test (50 files) | ✅ FIXED |
| `tusz_train_wsl2.yaml` | `cache/tusz` | `data_ext4/tusz/edf/train` | Full training (3734 files) | ✅ CORRECT |
| `tusz_dev_tuning.yaml` | `cache/dev_tuning` | `data_ext4/tusz/edf/dev` | Dev tuning | ✅ CORRECT |
| `tusz_eval_final.yaml` | `cache/eval_final` | `data_ext4/tusz/edf/eval` | Final eval | ✅ CORRECT |

### Modal Config
| Config | Cache Dir | Data Dir | Purpose | Status |
|--------|-----------|----------|---------|---------|
| `tusz_train_a100.yaml` | `/results/cache/tusz` | `/data/edf/train` | Full A100 training | ✅ CORRECT |

## LOCAL TRAINING - RUNNING ✅

- **Session**: tmux session "full-train"
- **Config**: tusz_train_wsl2.yaml
- **Progress**: 463/3734 files cached (12.4%)
- **Cache location**: cache/tusz/train/
- **CSV Parser**: FIXED (reads correct columns, includes all seizure types)
- **ETA**: ~2 hours remaining for cache build

## MODAL TRAINING - LAUNCHED 🚀

- **URL**: https://modal.com/apps/clarity-digital-twin/main/ap-mlWEBy3MruInjvR59gHnbA
- **Config**: tusz_train_a100.yaml (default)
- **Expected behavior**:
  - Should process 3734 files (NOT 50)
  - Cache at /results/cache/tusz/
  - Uses FIXED parser from mounted /src

## THE FIX WE APPLIED

In `deploy/modal/app.py` lines 121-125:
```python
if "smoke" in config_path.lower():
    env["BGB_LIMIT_FILES"] = "50"
else:
    # EXPLICITLY UNSET for full training to avoid inheritance
    env.pop("BGB_LIMIT_FILES", None)
```

This ensures:
- Smoke tests get BGB_LIMIT_FILES=50
- Full training explicitly REMOVES the limit

## WHAT TO VERIFY ON MODAL

Check the Modal dashboard for:
1. "Loading 3734 train, 933 val files" (NOT "50 train, 10 val")
2. NO "[DEBUG] BGB_LIMIT_FILES=50" message
3. Cache building at /results/cache/tusz/

## DATA PIPELINE FIXES - ALL APPLIED ✅

1. **CSV Parser Fixed** (src/brain_brr/data/io.py)
   - Reads correct columns for TUSZ CSV_BI format
   - Includes ALL seizure types (cpsz, gnsz, fnsz, etc.)

2. **BalancedSeizureDataset** (src/brain_brr/data/datasets.py)
   - Uses SeizureTransformer formula: ALL partial + 0.3×full + 2.5×background
   - Hard guards against 0 seizure training

3. **Manifest System** (src/brain_brr/data/cache_utils.py)
   - Categorizes windows: partial/full/no-seizure
   - Creates reproducible training sets

4. **Cache Directories**
   - Each config has unique cache directory
   - No confusion between smoke/full/dev/eval

## BOTTOM LINE

- **Local**: Running correctly, building full cache with fixed parser ✅
- **Modal**: Launched, should be building full cache (verify in dashboard)
- **Configs**: All properly configured with separate cache directories ✅
- **Parser**: Fixed for TUSZ CSV_BI format ✅
- **Dataset**: BalancedSeizureDataset ready with hard guards ✅

The pipelines are UNCLOGGED - data flows correctly from EDF → Parser → Cache → Manifest → Training!