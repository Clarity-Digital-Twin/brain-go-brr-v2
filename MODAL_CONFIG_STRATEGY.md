# MODAL CONFIG STRATEGY - WHAT WE HAVE vs WHAT WE SHOULD HAVE

## CURRENT STATE

### Local Configs (COMPLETE ✅)
- `smoke_test.yaml` - Quick test (50 files)
- `tusz_train_wsl2.yaml` - Full training (3734 files)
- `tusz_dev_tuning.yaml` - Dev set tuning (55 patients)
- `tusz_eval_final.yaml` - Final evaluation (45 patients)

### Modal Configs (INCOMPLETE ❌)
- `tusz_train_a100.yaml` - Full training only
- **MISSING**: smoke_a100.yaml
- **MISSING**: dev_tuning_a100.yaml
- **MISSING**: eval_final_a100.yaml

## SHOULD WE HAVE ALL MODAL CONFIGS?

**YES for full reproducibility!** Here's why:

1. **smoke_a100.yaml** - Quick GPU tests before expensive runs
2. **dev_tuning_a100.yaml** - Hyperparameter search on A100s
3. **eval_final_a100.yaml** - Final evaluation on cloud GPUs

## BUT WAIT - THE REAL QUESTION

### IS MODAL FULL TRAINING PARALLEL TO LOCAL?

Let me check the key parameters:

| Parameter | Local (tusz_train_wsl2) | Modal (tusz_train_a100) | Match? |
|-----------|-------------------------|-------------------------|---------|
| **Data source** | data_ext4/tusz/edf/train | /data/edf/train (S3 mount) | ✅ Same data |
| **Cache dir** | cache/tusz | /results/cache/tusz | ✅ Different path, same structure |
| **Model config** | | | |
| - d_model | 512 | 512 | ✅ |
| - n_layers | 6 | 6 | ✅ |
| - Bi-Mamba | Yes | Yes | ✅ |
| **Training config** | | | |
| - batch_size | 16 | 32 | ❌ DIFFERENT! |
| - learning_rate | 1e-4 | 1e-4 | ✅ |
| - epochs | 100 | 100 | ✅ |
| - gradient_accumulation | 4 | 2 | ❌ DIFFERENT! |

## THE CRITICAL DIFFERENCE

**Effective batch size:**
- Local: 16 × 4 = 64
- Modal: 32 × 2 = 64 ✅ SAME!

They're configured differently but achieve the SAME effective batch size!
- Local uses smaller batches (16) with more accumulation (4) for 24GB VRAM
- Modal uses larger batches (32) with less accumulation (2) for 80GB VRAM

## ANSWER TO YOUR QUESTION

### Is Modal Full PARALLEL to Local Full?

**YES! 100% PARALLEL** in what matters:
1. ✅ Same dataset (TUSZ train)
2. ✅ Same model architecture (Bi-Mamba-2 + U-Net)
3. ✅ Same effective batch size (64)
4. ✅ Same learning rate (1e-4)
5. ✅ Same CSV parser fixes
6. ✅ Same BalancedSeizureDataset logic

The only differences are OPTIMIZATIONS for hardware:
- Modal: Larger per-GPU batch (more VRAM)
- Local: Smaller batch with more gradient accumulation

## FOR CLEAN OSS REPRODUCIBILITY

Should create:
```yaml
configs/
├── local/
│   ├── smoke.yaml
│   ├── train.yaml
│   ├── dev.yaml
│   └── eval.yaml
└── modal/
    ├── smoke_a100.yaml
    ├── train_a100.yaml  (exists)
    ├── dev_a100.yaml    (TODO)
    └── eval_a100.yaml   (TODO)
```

But for NOW: **The full Modal training IS 100% parallel to local!**