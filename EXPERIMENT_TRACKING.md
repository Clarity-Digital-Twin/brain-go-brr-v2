# 🧪 EXPERIMENT TRACKING & VERSIONING

## Current Training Status

### v2.0 - U-Net Baseline (IN PROGRESS on Modal)
- **Branch**: `development`
- **Tag**: `v2.0-unet-baseline-training`
- **Architecture**: U-Net + ResNet + Bi-Mamba-2
- **Config**: `configs/modal/train_a100.yaml`
- **Status**: Training on Modal A100
- **W&B Project**: seizure-detection-a100
- **Expected Completion**: ~15 epochs with early stopping

## Branch Strategy

```
main
  └── stable, tested releases only

development
  └── v2.0 U-Net baseline (frozen during training)

feature/v2.3-tcn-architecture (CURRENT)
  └── TCN replacement for U-Net + ResNet

feature/v2.6-dynamic-gnn (FUTURE)
  └── Dynamic GNN + Laplacian PE addition
```

## Version Roadmap

| Version | Architecture | Status | Branch/Tag |
|---------|-------------|---------|------------|
| v2.0 | U-Net + ResNet + Bi-Mamba | Training | `v2.0-unet-baseline-training` |
| v2.1 | + Artifact detection head | Planned | - |
| v2.2 | + STFT preprocessing | Planned | - |
| v2.3 | TCN (replaces U-Net+ResNet) | Development | `feature/v2.3-tcn-architecture` |
| v2.6 | + Dynamic GNN + LPE | Planned | - |

## Weight Naming Convention

```
seizure_detector_v{VERSION}_{ARCHITECTURE}_epoch{EPOCH}_val{METRIC}.pt

Examples:
- seizure_detector_v2.0_unet_epoch50_val0.85.pt
- seizure_detector_v2.3_tcn_epoch30_val0.91.pt
```

## How to Resume from Specific Version

```bash
# To run inference with v2.0 U-Net weights:
git checkout v2.0-unet-baseline-training
python -m src.inference --weights /path/to/seizure_detector_v2.0_unet_final.pt

# To continue development on TCN:
git checkout feature/v2.3-tcn-architecture

# To see all experiment tags:
git tag -l "v*"
```

## Metrics Tracking

All experiments tracked in W&B:
- Project: `seizure-detection-a100`
- Entity: `jj-vcmcswaggins-novamindnyc`
- Key metrics: TAES @ 10/5/1 FA/24h, AUROC, Training time

## Important Notes

1. **NEVER** modify `development` branch while training is running
2. **ALWAYS** tag commits that produce training runs
3. **DOCUMENT** weight files with corresponding git commit/tag
4. **TEST** on same commit that produced weights

---

Last Updated: 2025-09-22
Current Training: v2.0 on Modal (DO NOT INTERRUPT)