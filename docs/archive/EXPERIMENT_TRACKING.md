# 🧪 EXPERIMENT TRACKING & VERSIONING

## Current Training Status

### v2.3 - TCN Architecture (TRAINING NOW on Modal)
- **Branch**: `feature/v2.3-tcn-architecture`
- **Tag**: `v2.3-tcn-training`
- **Architecture**: TCN + Bi-Mamba-2 (replaced U-Net + ResCNN)
- **Config**: `configs/modal/train.yaml` with `architecture: tcn`
- **Status**: 100-epoch training on Modal A100
- **W&B Project**: seizure-detection-a100
- **Started**: 2025-09-22
- **Modal App**: https://modal.com/apps/clarity-digital-twin/main

## Branch Strategy

```
main
  └── stable, tested releases only

feature/v2.3-tcn-architecture (CURRENT)
  └── TCN replacement for U-Net + ResNet (TRAINING NOW)

feature/v2.6-dynamic-gnn (NEXT)
  └── Dynamic GNN + Laplacian PE addition
```

## Version Roadmap

| Version | Architecture | Status | Notes |
|---------|-------------|---------|-------|
| ~~v2.0~~ | ~~U-Net + ResNet + Bi-Mamba~~ | Skipped | Went straight to TCN |
| ~~v2.1~~ | ~~+ Artifact detection~~ | Skipped | - |
| ~~v2.2~~ | ~~+ STFT preprocessing~~ | Skipped | - |
| **v2.3** | **TCN + Bi-Mamba** | **🔥 TRAINING** | Current production |
| v2.6 | + Dynamic GNN + LPE | Next | From EvoBrain paper |

## Weight Naming Convention

```
seizure_detector_v{VERSION}_{ARCHITECTURE}_epoch{EPOCH}_val{METRIC}.pt

Examples:
- seizure_detector_v2.3_tcn_epoch30_val0.91.pt
- seizure_detector_v2.6_tcn_gnn_epoch50_val0.95.pt
```

## How to Resume from Specific Version

```bash
# To continue development on TCN:
git checkout feature/v2.3-tcn-architecture

# To start GNN development (after TCN training):
git checkout -b feature/v2.6-dynamic-gnn

# To see all experiment tags:
git tag -l "v*"
```

## Metrics Tracking

All experiments tracked in W&B:
- Project: `seizure-detection-a100`
- Entity: `jj-vcmcswaggins-novamindnyc`
- Key metrics: TAES @ 10/5/1 FA/24h, AUROC, Training time

## Important Notes

1. **NEVER** interrupt Modal training runs
2. **ALWAYS** tag commits that produce training runs
3. **DOCUMENT** weight files with corresponding git commit/tag
4. **TEST** on same commit that produced weights

---

Last Updated: 2025-09-22
Current Training: v2.3 TCN on Modal A100 (DO NOT INTERRUPT)