# Documentation History and Migration Index

Last updated: 2025-09-21

This document tracks where documentation has moved during the 2025 reorganization.

## Current Documentation Structure

```
docs/
├── 01-data-pipeline/    # TUSZ handling, preprocessing, caching
├── 02-architecture/     # Models, specs, design decisions
├── 03-operations/       # Training, deployment, evaluation
├── 04-reference/        # Commands, configs, checklists
└── 05-research/         # Future work, experiments
```

## Where Did X Documentation Go?

### From Phase Docs → Components
- `PHASE1_DATA_PIPELINE.md` → `01-data-pipeline/data-io.md`
- `PHASE2.1_UNET_ENCODER.md` → `02-architecture/model-unet.md`
- `PHASE2.2_RESCNN_STACK.md` → `02-architecture/model-rescnn.md`
- `PHASE2.3_BIMAMBA.md` → `02-architecture/model-mamba.md`
- `PHASE2.4_DECODER.md` → `02-architecture/model-decoder.md`
- `PHASE3_TRAINING_PIPELINE.md` → `03-operations/training.md`
- `PHASE4_POSTPROCESSING.md` → `03-operations/postprocessing.md`
- `PHASE5_EVALUATION.md` → `03-operations/evaluation.md`

### From TUSZ Docs → Renamed
- `TUSZ_CSV_BI_PARSER.md` → `01-data-pipeline/tusz-csv-parser.md`
- `TUSZ_CHANNELS.md` → `01-data-pipeline/tusz-channels.md`
- `TUSZ_EDF_HEADER_FIX.md` → `01-data-pipeline/tusz-edf-repair.md`
- `TUSZ_SAMPLING_STRATEGY.md` → `01-data-pipeline/tusz-cache-sampling.md`

### Critical Issues (Consolidated)
- All P0/P1 bug reports → `01-data-pipeline/CRITICAL-ISSUES-RESOLVED.md`
- Cache disasters, focal loss bugs, WSL2 hangs all documented there

### Deployment & WSL2
- `WSL2/` subdirectory flattened → `03-operations/wsl2-*.md`
- Modal deployment notes → `03-operations/deploy-modal.md`

## What Was Added During Migration

1. **Empirical seizure type frequencies** - Added to `tusz-csv-parser.md`
2. **Cache configuration warnings** - Added to `04-reference/configs.md`
3. **Architecture rationale** (why no pretrained weights) - Added to `canonical-spec.md`
4. **Critical issues history** - New file `CRITICAL-ISSUES-RESOLVED.md`

## Archives Status

The `/docs_archive/` directory (if still present) contained 43 files of historical documentation that has been reviewed, relevant content extracted, and can be safely deleted. Everything valuable has been preserved in the current `/docs/` structure.