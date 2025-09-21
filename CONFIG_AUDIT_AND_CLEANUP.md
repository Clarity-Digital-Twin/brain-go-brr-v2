# CONFIG AUDIT AND PROFESSIONAL CLEANUP PLAN

## CURRENT MESSY STATE - 8 CONFIGS WITH UNCLEAR PURPOSE

| Config File | Purpose | Data Dir | Cache Dir | Status |
|-------------|---------|----------|-----------|---------|
| `local.yaml` | Old dev config? | data_ext4/tusz/edf/train | cache | ❓ UNCLEAR |
| `production.yaml` | Old prod? | data/raw | cache | ❌ BROKEN PATH |
| `smoke_test.yaml` | Quick test | data_ext4/tusz/edf/train | cache/smoke | ✅ FIXED |
| `tusz_train.yaml` | Generic train? | data_ext4/tusz/edf/train | NOT SET | ❓ REDUNDANT? |
| `tusz_train_wsl2.yaml` | Local full train | data_ext4/tusz/edf/train | cache/tusz | ✅ RUNNING |
| `tusz_train_a100.yaml` | Modal full train | /data/edf/train | /results/cache/tusz | ✅ RUNNING |
| `tusz_dev_tuning.yaml` | Dev set tuning | data_ext4/tusz/edf/dev | cache/dev_tuning | ✅ CLEAR |
| `tusz_eval_final.yaml` | Final evaluation | data_ext4/tusz/edf/eval | cache/eval_final | ✅ CLEAR |

## PROBLEMS

1. **Naming confusion**: local.yaml vs tusz_train_wsl2.yaml - which is THE local config?
2. **Redundancy**: tusz_train.yaml vs tusz_train_wsl2.yaml - why both?
3. **Broken paths**: production.yaml points to "data/raw" which doesn't exist
4. **Platform mixing**: WSL2 configs mixed with generic configs
5. **No clear Modal configs**: Only one A100 config for Modal

## PROFESSIONAL STRUCTURE - CLEAR & CLEAN

```
configs/
├── local/                    # Local WSL2/Linux configs
│   ├── smoke.yaml           # Quick 50-file test
│   ├── train.yaml           # Full training (3734 files)
│   ├── dev.yaml             # Dev set tuning (55 patients)
│   └── eval.yaml            # Final evaluation (45 patients)
│
├── modal/                    # Modal cloud GPU configs
│   ├── smoke_a100.yaml      # Quick cloud test
│   ├── train_a100.yaml      # Full cloud training
│   ├── dev_a100.yaml        # Cloud dev tuning
│   └── eval_a100.yaml       # Cloud final eval
│
└── base/                     # Shared base configs (optional)
    ├── model.yaml           # Model architecture defaults
    └── training.yaml        # Training hyperparameters
```

## WHAT EACH CONFIG SHOULD BE

### Local Configs (WSL2/Linux with RTX 4090)
```yaml
# local/smoke.yaml
data:
  data_dir: data_ext4/tusz/edf/train
  cache_dir: cache/smoke
training:
  epochs: 1
  batch_size: 8
experiment:
  description: "Local smoke test - 50 files via BGB_LIMIT_FILES"

# local/train.yaml
data:
  data_dir: data_ext4/tusz/edf/train
  cache_dir: cache/tusz
training:
  epochs: 100
  batch_size: 8
  gradient_accumulation: 8
experiment:
  description: "Local full training - RTX 4090 optimized"

# local/dev.yaml
data:
  data_dir: data_ext4/tusz/edf/dev
  cache_dir: cache/dev
training:
  epochs: 50
  batch_size: 8
experiment:
  description: "Local hyperparameter tuning on dev set"

# local/eval.yaml
data:
  data_dir: data_ext4/tusz/edf/eval
  cache_dir: cache/eval
experiment:
  description: "Local final evaluation - no training"
```

### Modal Configs (Cloud A100-80GB)
```yaml
# modal/smoke_a100.yaml
data:
  data_dir: /data/edf/train
  cache_dir: /results/cache/smoke
training:
  epochs: 1
  batch_size: 64
experiment:
  description: "Modal smoke test - verify pipeline"

# modal/train_a100.yaml
data:
  data_dir: /data/edf/train
  cache_dir: /results/cache/tusz
training:
  epochs: 100
  batch_size: 64
  gradient_accumulation: 1
experiment:
  description: "Modal full training - A100 optimized"

# modal/dev_a100.yaml
data:
  data_dir: /data/edf/dev
  cache_dir: /results/cache/dev
training:
  epochs: 50
  batch_size: 64
experiment:
  description: "Modal hyperparameter search on dev set"

# modal/eval_a100.yaml
data:
  data_dir: /data/edf/eval
  cache_dir: /results/cache/eval
experiment:
  description: "Modal final evaluation - inference only"
```

## MIGRATION PLAN

1. **Keep for now** (currently running):
   - `tusz_train_wsl2.yaml` → will become `local/train.yaml`
   - `tusz_train_a100.yaml` → will become `modal/train_a100.yaml`

2. **Delete/Archive** (confusing/broken):
   - `local.yaml` - redundant with tusz_train_wsl2
   - `production.yaml` - broken paths
   - `tusz_train.yaml` - redundant generic version

3. **Rename/Move**:
   - `smoke_test.yaml` → `local/smoke.yaml`
   - `tusz_dev_tuning.yaml` → `local/dev.yaml`
   - `tusz_eval_final.yaml` → `local/eval.yaml`

## WHY THIS IS BETTER

1. **Clear platform separation**: local/ vs modal/
2. **Consistent naming**: smoke/train/dev/eval
3. **No redundancy**: Each config has ONE clear purpose
4. **Easy to find**: "I need local training" → local/train.yaml
5. **Professional**: Any engineer can understand the structure

## IMMEDIATE ACTION

For now, your running pipelines are CORRECT:
- Local: Using `tusz_train_wsl2.yaml` ✅
- Modal: Using `tusz_train_a100.yaml` ✅

After training completes, we should reorganize into the clean structure above!