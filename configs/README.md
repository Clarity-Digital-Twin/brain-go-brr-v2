# Configuration Structure

## 📁 Clean Professional Structure

```
configs/
├── local/                    # Local WSL2/Linux configs (RTX 4090 optimized)
│   ├── smoke.yaml           # Quick 50-file test (1 epoch)
│   ├── train.yaml           # Full training (3734 files, 100 epochs)
│   ├── dev.yaml             # Dev set hyperparameter tuning (55 patients)
│   └── eval.yaml            # Final evaluation (45 patients) - ONE SHOT ONLY!
│
└── modal/                    # Modal cloud GPU configs (A100-80GB optimized)
    ├── smoke_a100.yaml      # Quick cloud test (1 epoch)
    ├── train_a100.yaml      # Full cloud training (3734 files, 100 epochs)
    ├── dev_a100.yaml        # Cloud hyperparameter tuning
    └── eval_a100.yaml       # Cloud final evaluation - ONE SHOT ONLY!
```

## 🚀 Usage Examples

### Local Training
```bash
# Quick smoke test (50 files via BGB_LIMIT_FILES env)
BGB_LIMIT_FILES=50 python -m src train configs/local/smoke.yaml

# Full training
python -m src train configs/local/train.yaml

# Hyperparameter tuning on dev set (after training)
python -m src evaluate configs/local/dev.yaml --checkpoint results/run/best.pt

# Final evaluation (ONE TIME ONLY!)
python -m src evaluate configs/local/eval.yaml --checkpoint results/run/best.pt
```

### Modal Cloud Training
```bash
# Smoke test
modal run deploy/modal/app.py::train --config-path configs/modal/smoke_a100.yaml

# Full training (detached)
modal run --detach deploy/modal/app.py::train --config-path configs/modal/train_a100.yaml

# Dev set tuning
modal run deploy/modal/app.py::evaluate --config-path configs/modal/dev_a100.yaml

# Final evaluation
modal run deploy/modal/app.py::evaluate --config-path configs/modal/eval_a100.yaml
```

## 🔑 Key Differences

| Aspect | Local (RTX 4090) | Modal (A100-80GB) |
|--------|------------------|-------------------|
| **Data Path** | `data_ext4/tusz/edf/` | `/data/edf/` (S3 mount) |
| **Cache Path** | `cache/` | `/results/cache/` (persistent volume) |
| **Batch Size** | 8-32 | 64-128 (8x VRAM) |
| **Workers** | 0 (WSL2 safe) | 4-8 (cloud optimized) |
| **Gradient Accumulation** | 4-8 | 1-2 |

## ⚠️ Important Notes

1. **Smoke Tests**: Automatically set `BGB_LIMIT_FILES=50` to use only 50 training files
2. **Dev Set**: For hyperparameter tuning ONLY - do not train on this
3. **Eval Set**: FINAL TEST - use only once after all tuning is complete
4. **Cache Separation**: Each config uses its own cache directory to avoid conflicts

## 🏗️ Architecture (All Configs)

- **Model**: Bi-Mamba-2 (6 layers) + U-Net + ResCNN
- **Input**: 19-channel EEG @ 256 Hz
- **Window**: 60s with 10s stride
- **Loss**: Focal loss (for 12:1 class imbalance)
- **Post-processing**: Hysteresis (tau_on=0.86, tau_off=0.78)

## 📊 Training Progression

1. **Smoke Test** → Verify pipeline works (1 epoch, 50 files)
2. **Full Training** → Train model (100 epochs, 3734 files)
3. **Dev Tuning** → Optimize thresholds (55 patients)
4. **Final Eval** → Report results (45 patients) - NO RETRAINING AFTER THIS!

## 🔧 Migrated From

Previous messy structure with 8 configs:
- `local.yaml`, `production.yaml`, `tusz_train.yaml` → **DELETED** (redundant/broken)
- `smoke_test.yaml` → `local/smoke.yaml`
- `tusz_train_wsl2.yaml` → `local/train.yaml`
- `tusz_dev_tuning.yaml` → `local/dev.yaml`
- `tusz_eval_final.yaml` → `local/eval.yaml`
- `tusz_train_a100.yaml` → `modal/train_a100.yaml`
- **NEW**: `modal/smoke_a100.yaml`, `modal/dev_a100.yaml`, `modal/eval_a100.yaml`

All old configs backed up in `configs/archive/` for reference.