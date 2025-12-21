# Performance Targets and System Profile

**Last Updated**: 2025-12-20

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
| 10 FA/24h | >95% | Brain-Go-Brr (FLA Exp4): 35.9% (OVERLAP); SeizureTransformer: 33.90% |
| 5 FA/24h | >90% | Brain-Go-Brr (FLA Exp4): 27.1% (OVERLAP) |
| 1 FA/24h | >75% | Brain-Go-Brr (FLA Exp4): 5.8% (OVERLAP); Temple SOTA: 50% @ 4 FA |

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

**FLA Exp4 (Gated DeltaNet) – TUSZ eval (held-out)**:
- AUROC: **0.8654**
- PR-AUC: **0.5409**
- Sensitivity @ 10 FA/24h: **35.9%** (OVERLAP scoring)
- Sensitivity @ 5 FA/24h: 27.1%
- Sensitivity @ 2.5 FA/24h: 18.6%
- Sensitivity @ 1 FA/24h: 5.8%
- ECE: 0.029
- Source of truth: `results/local_fla_exp4_cyclic/eval_results_v2.json`

**Status**: Training complete (78 epochs, best epoch 63); see `docs/06-evaluation/REALISTIC_PERFORMANCE_TARGETS.md` for updated target framing.

---

## 🚨 Reality Check: Why 1 FA/24h @ 75% is Probably Impossible

### Evidence

1. **Temple's SOTA**: 4 FA/24h @ 50% sensitivity (best real-world system)
2. **SeizureTransformer**: 26.89 FA/24h @ 45.63% sensitivity (2025 #1 model)
3. **ROC Curve Reality**: 5% absolute sensitivity change = massive FA rate change in low FA region

### What This Means

- Going from 4 FA/24h → 1 FA/24h requires **~25% sensitivity drop** (not improvement!)
- Current aspirational targets (1 FA/24h @ 75%) are **64 percentage points above SOTA**
- Would require **fundamental breakthrough**, not incremental improvement

### Neureka 2020 Competition Evidence

**Dataset**: TUSZ v1.5.1 eval set, 16 participating systems

- **Best system (sia)**: 1 FA/24h @ **11.37% sensitivity** (TAES scoring)
- **Baseline (nedc)**: 17 FA/24h @ **35.54% sensitivity** (TAES scoring)
- **Key Insight**: Achieving 1 FA/24h with good sensitivity (>50%) is **extremely difficult**

**Conclusion**: 1 FA/24h @ 75% sensitivity is likely impossible with current architectures. More realistic to target 4-8 FA/24h @ 50-55% sensitivity.

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

---

## 📚 References

1. **Temple NEDC Research Team** - Personal correspondence, October 2025 (4 FA/24h @ 50% sensitivity benchmark from clinical deployments)
2. **Wu et al. 2025** - SeizureTransformer: arXiv:2504.00336, EpilepsyBench #1 winner
3. **Shah et al. 2021** - "Validation of Temporal Scoring Metrics for Automatic Seizure Detection" (Neureka 2020 results)
4. **Our SeizureTransformer Evaluation** - NEDC v6.0.0 on TUSZ v2.0.3 eval set
5. **TUSZ Dataset** - v2.0.3, patient-disjoint splits (865 files, 127.7 hours, 469 seizures in eval set)

**See Also**:
- `docs/06-evaluation/TAES_DISAMBIGUATION.md` - CRITICAL naming collision explanation
- `docs/06-evaluation/REALISTIC_PERFORMANCE_TARGETS.md` - Full analysis with additional tables and comparisons
