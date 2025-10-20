# Performance Targets and System Profile

**Last Updated**: October 20, 2025

**🚨 CRITICAL METRICS NOTE**: "TAES" has TWO different meanings! See `docs/06-evaluation/TAES_DISAMBIGUATION.md` for complete explanation.
- TAES score = quality metric [0,1] from `calculate_taes()`
- Sensitivity @ FA rates = uses NEDC OVERLAP scoring (NOT NEDC TAES scoring!)

---

## 🎯 Gold Standard: Temple NEDC (October 2025)

> **"Our best systems operate at about 4 FAs/24 hours at a sensitivity of about 50% for the seizure class in a two-class problem - seizure vs. background."**
>
> — Temple University NEDC Research Team (creators of TUSZ dataset)

**Verified Clinical SOTA**: **4 FA/24h @ ~50% sensitivity**

**Key Context**:
- Verified on **real clinical data** (72-hour continuous patient monitors)
- ROC curve is **very steep** in low FA rate area
- 5% absolute sensitivity change → **huge change in FA rate**
- ML algorithms claiming better at high FA rates **don't hold at low FA rates**
- Transformers are promising but **no systems yet that can control FA rates** for newer architectures

**Source**: Personal correspondence with Temple NEDC research team (October 2025)

---

## 📊 Current State-of-the-Art (2025)

### SeizureTransformer (EpilepsyBench #1 Winner)

**Model**: U-Net + Transformer, ~41M parameters
**Dataset**: TUSZ v2.0.3 eval set (865 files, 127.7 hours, 469 seizures)

#### Performance by Scoring System

| Scoring System | Sensitivity | FA/24h | Notes |
|----------------|-------------|--------|-------|
| **SzCORE Event** (most permissive) | 52.35% | **8.59** | ±30s/60s tolerances, merges events <90s |
| **NEDC OVERLAP** (standard) | 45.63% | **26.89** | Any temporal overlap = TP |
| **NEDC TAES** (strictest) | 65.21% | **136.73** | Partial credit, time-aligned |

**🔥 Key Insight**: Same predictions, different scorers → **3.1× difference in FA/24h** (NEDC OVERLAP vs SzCORE Event)

**⚠️ Warning**: Always check which scorer papers use! A model claiming 1 FA/24h with SzCORE Event might be 3-5 FA/24h with NEDC OVERLAP.

---

## 🎯 Realistic Performance Targets

### Aspirational Targets (Original README - NEEDS REVISION)

| FA Rate | Target Sensitivity | Reality Check |
|---------|-------------------|---------------|
| 10 FA/24h | >95% | SeizureTransformer: 34% ❌ (**61 points gap**) |
| 5 FA/24h | >90% | SeizureTransformer: ~25% ❌ (**65 points gap**) |
| 1 FA/24h | >75% | Temple SOTA: 50% @ 4 FA ❌ (**impossible with current architectures**) |

**Reality**: These targets are **64 percentage points above SOTA**. Would require fundamental breakthrough, not incremental improvement.

### Revised Realistic Targets

#### Optimistic Scenario (Architectural Edge)
**Target**: **≤4 FA/24h @ ≥55% sensitivity** (NEDC OVERLAP)
- Matches Temple's verified clinical SOTA
- 5% better sensitivity than Temple's baseline
- **Highly publishable** (beats verified clinical systems)

#### Realistic Scenario (Incremental Improvements)
**Target**: **≤8 FA/24h @ ≥50% sensitivity** (NEDC OVERLAP)
- Matches SeizureTransformer's SzCORE Event performance on NEDC OVERLAP scale
- Solid improvement over current TUSZ baselines
- **Publishable**

#### Conservative Scenario (Similar to SOTA)
**Target**: **≤15 FA/24h @ ≥45% sensitivity** (NEDC OVERLAP)
- Better than SeizureTransformer's 26.89 FA/24h
- Demonstrates architectural value
- **Publishable with strong ablation studies**

### Minimum Viable Result (Publishable)
**Either**:
- Match SeizureTransformer on **SzCORE Event** (8.59 FA/24h @ 52.35% sensitivity)
- **OR** beat SeizureTransformer on **NEDC OVERLAP** (≤20 FA/24h @ ≥45% sensitivity)

**Why Publishable**:
- First BiMamba2 + FLA comparison on TUSZ
- First dual-stream (node + edge Mamba) architecture for seizure detection
- First dynamic Laplacian PE for EEG
- Empirical validation even if not SOTA

---

## 📈 Current Measured Performance

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
