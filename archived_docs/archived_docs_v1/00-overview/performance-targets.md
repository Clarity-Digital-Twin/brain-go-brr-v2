# Performance Targets and System Profile

Targets (TAES)

**🚨 CRITICAL METRICS NOTE**: "TAES" has TWO different meanings! See `TAES_DISAMBIGUATION.md` for complete explanation.
- TAES score = quality metric [0,1] from `calculate_taes()`
- Sensitivity @ FA rates = uses NEDC OVERLAP scoring (NOT NEDC TAES scoring!)

Clinical targets (sensitivity at FA rates):
- 10 FA/24h → Sensitivity > 95%
- 5 FA/24h → Sensitivity > 90%
- 1 FA/24h → Sensitivity > 75%

Current measured performance (Epoch 6, FLA on RTX 4090):
- TAES: **0.9450** (quality score, not sensitivity!)
- AUROC: **0.8141**
- Sensitivity @ 10 FA/24h: **24.84%** (OVERLAP scoring)
- Sensitivity @ 5 FA/24h: **16.81%**
- Sensitivity @ 1 FA/24h: **7.38%**

Training times (MEASURED from production runs)

**Local FLA (RTX 4090, batch 8)**:
- **~9.6h/epoch average** (Epochs 1-6 measured)
  - Training: ~4.1h/epoch (7702 batches @ ~2.1s/batch)
  - Validation: ~5.5h/epoch (18528 batches, disk-backed) - **1.3× longer than training!**
- Epoch 1: 7.20h (faster due to warmup)
- Epochs 2-6: 10.1h avg (consistent performance)
- 100 epochs: **~960 hours (40 days)** - FREE (only electricity)

**Modal BiMamba2 (A100-80GB, batch 48)**:
- **TRAINING PAUSED at Epoch 6 due to costs**
- 7-12 hours/epoch documented (training 1-2h + validation 5.8h)
- **Cost: $18,600 for 100 epochs** ($186/epoch measured × 100)
  - Rate: $4.40/hr (GPU $2.50 + 24 CPU $1.13 + 96GB RAM $0.77)
  - Modal training paused; local FLA training proceeding

**Smoke tests**:
- Local/Docker: ~5 minutes (3 files via BGB_SMOKE_TEST=1)
- Modal: ~5-10 minutes (50 files via BGB_LIMIT_FILES=50)

Resource usage (current configs)

- VRAM: ~20 GB on RTX 4090 (batch 8, mixed precision off); ~55–60 GB on A100-80GB (batch 48, mixed precision on)
- Cache size: **≈520 GB** memory-mapped NPY pairs (`cache/tusz_mmap/*`)
- Checkpoints: ~125 MB per epoch (per run)

Complexity summary

- TCN: O(N); Node/Edge Mamba: O(N) per stream; GNN: O(E+V) per timestep (vectorized over time)

Post-processing defaults

- Hysteresis: τ_on=0.86, τ_off=0.78
- Morphology: Opening(11), Closing(31)
- Duration: 3–600s; Merge within 2s
