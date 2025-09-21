# Brain-Go-Brr v2 Documentation

Organized documentation for the Bi-Mamba-2 + U-Net + ResCNN seizure detection pipeline.

## Quick Navigation

### 01-data-pipeline/
**TUSZ dataset handling and preprocessing**
- `tusz-overview.md` - Dataset overview and structure
- `tusz-csv-parser.md` - **CRITICAL**: CSV_BI label parsing (includes mysz fix)
- `tusz-channels.md` - 19-channel canonical ordering
- `tusz-data-flow.md` - End-to-end data pipeline
- `tusz-mysz-crisis.md` - Postmortem on missing seizure type

### 02-architecture/
**Model components and technical specifications**
- `canonical-spec.md` - **SOURCE OF TRUTH** for full architecture
- `model-unet.md` - U-Net encoder/decoder (×16 downsample)
- `model-mamba.md` - Bidirectional Mamba-2 (O(N) complexity)
- `model-rescnn.md` - Residual CNN stack
- `pipeline-diagram.md` - ASCII visualization

### 03-operations/
**Training, deployment, and evaluation**
- `deploy-modal.md` - Cloud deployment on Modal
- `deploy-local-wsl2.md` - Local WSL2/Linux setup
- `training.md` - Training pipeline and strategies
- `postprocessing.md` - Hysteresis and morphology
- `evaluation.md` - TAES metrics and benchmarks

### 04-reference/
**Commands, configs, and checklists**
- `commands.md` - All CLI commands (train, eval, etc.)
- `configs.md` - YAML configuration guide
- `evaluation-checklist.md` - Pre-deployment validation

### 05-research/
**Future work and experiments**
- `streaming.md` - Real-time inference plans
- `channel-annotations.md` - Annotation integration
- `benchmark-plans.md` - Performance targets

## Entry Points

**Getting Started:**
1. Read `01-data-pipeline/tusz-overview.md`
2. Review `02-architecture/canonical-spec.md`
3. Follow `03-operations/deploy-local-wsl2.md`

**Key Commands:**
```bash
# Training
python -m src train configs/smoke_test.yaml

# Evaluation
python -m src evaluate <checkpoint> <data_dir> --config <config>

# Quality checks
make q  # Run after every change!
```

## Critical Notes

- **TUSZ mysz seizure type**: Was missing from parser - now fixed (see `tusz-mysz-crisis.md`)
- **Mamba CUDA**: Set `SEIZURE_MAMBA_FORCE_FALLBACK=1` for Conv1d fallback
- **Cache rebuild**: Required after any data pipeline changes

See `HISTORY.md` for archived documentation index.