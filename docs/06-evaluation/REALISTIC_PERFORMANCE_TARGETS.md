# Realistic Performance Targets for EEG Seizure Detection

**Date Created**: October 14, 2025
**Last Updated**: 2025-12-20
**Status**: Updated with FLA Exp4 eval results
**Purpose**: Establish achievable performance goals based on SOTA results and clinical reality

---

## 🎯 THE GOLD STANDARD TO BEAT

### Temple University NEDC Research Team - October 2025

> **"Our best systems operate at about 4 FAs/24 hours at a sensitivity of about 50% for the seizure class in a two-class problem - seizure vs. background."**

**Target**: **4 FA/24h @ ~50% sensitivity**

**Source**: Personal correspondence with Temple NEDC research team (October 2025)

**Context from NEDC Research Team**:
- Verified on **real clinical data** (72-hour continuous patient monitors)
- ROC curve is **very steep** in low FA rate area
- 5% absolute sensitivity change → **huge change in FA rate**
- ML algorithms claiming better at high FA rates **don't hold at low FA rates**
- Transformers are promising but **no systems yet that can control FA rates** for newer architectures

**Why This Matters**:
- From Temple University's Neural Engineering Data Consortium (creators of TUSZ dataset and NEDC scorer)
- Based on REAL clinical deployments, not just benchmark datasets
- Represents current SOTA for production systems (not just research claims)

---

## 📊 CURRENT STATE-OF-THE-ART (2025)

### SeizureTransformer (EpilepsyBench #1 Winner)

**Model**: U-Net + Transformer architecture, ~41M parameters
**Training**: TUSZ v2.0.3 subset (~910 hours) + Siena dataset (128 hours)
**Source**: Our evaluation using NEDC v6.0.0 on TUSZ eval set (865 files, 127.7 hours, 469 seizures)

#### Performance by Scoring System

| Scoring System | Sensitivity | FA/24h | Notes |
|----------------|-------------|--------|-------|
| **SzCORE Event** (most permissive) | 52.35% | **8.59** | ±30s/60s tolerances, merges events <90s |
| **NEDC OVERLAP** (standard) | 45.63% | **26.89** | Any temporal overlap = TP |
| **NEDC TAES** (strictest) | 65.21% | **136.73** | Partial credit, time-aligned |

**Key Insight**: Same predictions, different scorers → **3.1× difference** in FA/24h (NEDC OVERLAP vs SzCORE Event)

#### Clinical Target Attempts

**10 FA/24h Target** (threshold=0.88, kernel=5, duration=3.0s):
- NEDC OVERLAP: 33.90% sensitivity @ 10.27 FA/24h ❌ (far below 75% goal)
- SzCORE Event: 40.59% sensitivity @ 3.36 FA/24h ✅ (meets FA target but low sensitivity)

**2.5 FA/24h Target** (threshold=0.95, kernel=5, duration=5.0s):
- NEDC OVERLAP: 14.50% sensitivity @ 2.05 FA/24h ❌ (too low for clinical use)
- SzCORE Event: 19.71% sensitivity @ 0.75 FA/24h ❌ (too low for clinical use)

**1 FA/24h Target**: NOT TESTED (would require extreme parameters yielding <5% sensitivity)

**AUROC**: 0.9019 (excellent discrimination capability)

---

### Brain-Go-Brr (FLA Exp4 - BiGatedDeltaNet)

**Model**: TCN + BiGatedDeltaNet (FLA) + GNN + Dynamic LPE (~31M params)  
**Source of truth**: `results/local_fla_exp4_cyclic/eval_results_v2.json` (run 2025-12-20)  
**Scoring**: OVERLAP-style sensitivity at fixed FA/24h targets (binary any-overlap TP counting; see `TAES_DISAMBIGUATION.md`)

#### TUSZ Eval (Held-Out)

| Metric | Value |
|--------|-------|
| **AUROC** | **0.8654** |
| **PR-AUC** | 0.5409 |
| **Sensitivity @ 10 FA/24h** | **35.9%** |
| **Sensitivity @ 5 FA/24h** | 27.1% |
| **Sensitivity @ 2.5 FA/24h** | 18.6% |
| **Sensitivity @ 1 FA/24h** | 5.8% |
| **ECE** | 0.029 |

**Dataset notes**:
- TUSZ eval split has 865 EDF/label pairs; 29 yield 0 windows under 60s windowing
- Metrics above are computed on 836 scored recordings totaling 127.8 hours

#### Head-to-Head vs SeizureTransformer (Same TUSZ Eval Split, OVERLAP Scoring)

| FA Rate | FLA Exp4 | SeizureTransformer | Delta |
|--------:|---------:|------------------:|------:|
| 10 FA/24h | **35.9%** | 33.90% | **+2.0%** |
| 2.5 FA/24h | **18.6%** | 14.50% | **+4.1%** |

SeizureTransformer numbers are from our run in `reference_repos/SeizureTransformer/docs/results/FINAL_COMPREHENSIVE_RESULTS_TABLE.md` (Python OVERLAP = NEDC OVERLAP).

---

## 🔍 UNDERSTANDING SCORING SYSTEMS

**🚨 CRITICAL**: "TAES" has TWO different meanings (metric vs scoring system)! See `TAES_DISAMBIGUATION.md` for full explanation.

Different scorers measure different aspects of performance. **Choice of scorer can create 3-16× differences in reported FA rates for identical predictions.**

### 1. NEDC TAES (Strictest)
**Philosophy**: Research precision, temporal accuracy
**Method**: Partial credit based on overlap percentage
**Use Case**: Algorithm development, exact timing matters
**Impact**: Penalizes over/under-segmentation, highest FA rates

### 2. NEDC OVERLAP (Clinical Standard)
**Philosophy**: Clinical review, any detection is useful
**Method**: Binary - any overlap = full TP
**Use Case**: Standard TUSZ evaluation, balanced approach
**Impact**: Middle ground - most commonly reported for TUSZ

### 3. SzCORE Event (Most Permissive)
**Philosophy**: Clinical deployment, early warning valuable
**Method**: ±30s/60s tolerances around events, merges <90s gaps
**Use Case**: Screening, alarm fatigue reduction
**Impact**: Lowest FA rates - rewards early detection

**Critical Reality**: When comparing results, **ALWAYS check which scorer was used**. A model claiming 1 FA/24h with SzCORE Event might be 3-5 FA/24h with NEDC OVERLAP and 15-20 FA/24h with NEDC TAES.

---

## 🎯 OUR PERFORMANCE TARGETS

### Primary Target (Match/Beat Temple SOTA)
**Goal**: **≤4 FA/24h @ ≥50% sensitivity** (using NEDC OVERLAP)

**Rationale**:
- Matches Temple NEDC's verified clinical systems
- Uses standard TUSZ scorer (NEDC OVERLAP)
- Realistic and achievable based on proven systems

**Current Gap**:
- SeizureTransformer (2025 #1): 26.89 FA/24h @ 45.63% sensitivity (NEDC OVERLAP on TUSZ eval)
- Brain-Go-Brr (FLA Exp4, Dec 2025): 35.9% sensitivity @ 10 FA/24h (OVERLAP-style) on TUSZ eval
- Temple's SOTA: 4 FA/24h @ 50% sensitivity (verified on clinical data)
- Need: **6.7× reduction in FA rate** to match Temple's benchmark

### Stretch Goal (Clinical Deployment)
**Goal**: **≤10 FA/24h @ ≥75% sensitivity** (using NEDC OVERLAP)

**Rationale**:
- Meets clinical deployment thresholds from literature
- 75% sensitivity = acceptable for ICU monitoring
- 10 FA/24h = manageable alarm fatigue

**Current Reality**:
- SeizureTransformer @ 10 FA: 33.90% sensitivity (NEDC OVERLAP)
- Brain-Go-Brr (FLA Exp4) @ 10 FA: 35.9% sensitivity (OVERLAP-style)
- Need: **~2.1× improvement in sensitivity** to reach 75%

### Gold Standard (Human Reviewer Level)
**Goal**: **≤1 FA/24h @ ≥75% sensitivity** (using NEDC OVERLAP)

**Rationale**:
- Human reviewers achieve ~1 FA/24h
- Gold standard for unattended monitoring
- Enables 24/7 clinical use without constant supervision

**Current Reality**:
- SeizureTransformer: NOT TESTED (likely <10% sensitivity)
- Brain-Go-Brr (FLA Exp4) @ 1 FA: 5.8% sensitivity (OVERLAP-style)
- This may be **impossible with current architectures**

---

## 📈 REALISTIC EXPECTATIONS FOR OUR STACK

### Brain-Go-Brr Architecture (TCN + BiMamba2 / BiGatedDeltaNet + GNN + Dynamic LPE)
**Parameters**: ~31M (similar to SeizureTransformer's ~41M)
**Training**: Full TUSZ v2.0.3 train split (4667 files, official patient-disjoint)

### Optimistic Scenario
**Assumption**: Our architecture improvements (dual-stream, dynamic PE, GNN) provide edge

**Target**: **≤4 FA/24h @ ≥55% sensitivity** (NEDC OVERLAP)
- Matches Temple's verified clinical SOTA
- 5% better sensitivity than Temple's baseline
- Would be **highly publishable result** (beats verified clinical systems)

### Realistic Scenario
**Assumption**: Incremental improvements, similar to other modern architectures

**Target**: **≤8 FA/24h @ ≥50% sensitivity** (NEDC OVERLAP)
- Matches SeizureTransformer's SzCORE Event performance on NEDC OVERLAP scale
- Solid improvement over current TUSZ baselines
- **Still publishable**

### Conservative Scenario
**Assumption**: Similar to SeizureTransformer on TUSZ with NEDC scoring

**Target**: **≤15 FA/24h @ ≥45% sensitivity** (NEDC OVERLAP)
- Better than SeizureTransformer's 26.89 FA/24h
- Demonstrates architectural value
- **Publishable if ablation studies show clear improvements**

---

## 🔬 WHAT WOULD CONSTITUTE "SUCCESS"?

### Minimum Viable Result (Publishable)
**Either**:
- Match SeizureTransformer on **SzCORE Event** (8.59 FA/24h @ 52.35% sensitivity)
- **OR** beat SeizureTransformer on **NEDC OVERLAP** (≤20 FA/24h @ ≥45% sensitivity)

**Why Publishable**:
- First BiMamba2 + FLA comparison on TUSZ
- First dual-stream (node + edge Mamba) architecture for seizure detection
- First dynamic Laplacian PE for EEG
- Empirical validation even if not SOTA

### Strong Result (Highly Publishable)
**Either**:
- Beat Temple SOTA: **≤4 FA/24h @ ≥50% sensitivity** (NEDC OVERLAP)
- **OR** beat SeizureTransformer on SzCORE: **≤5 FA/24h @ ≥55% sensitivity**

**Why Highly Publishable**:
- Matches/beats verified clinical systems
- Clear architectural contribution
- Ready for clinical validation studies

### Breakthrough Result (Top-Tier Publication)
**Goal**: **≤4 FA/24h @ ≥60% sensitivity** (NEDC OVERLAP)

**Why Top-Tier**:
- Significantly beats all known systems
- Closes gap toward clinical deployment (75% goal)
- Demonstrates clear path to 10 FA/24h @ 75% sensitivity

---

## 🚨 CRITICAL REALITY CHECK

### Why 1 FA/24h @ 75% Sensitivity is Probably Impossible

**Evidence**:
1. **Temple's SOTA**: 4 FA/24h @ 50% sensitivity (best real-world system)
2. **SeizureTransformer**: 26.89 FA/24h @ 45.63% sensitivity (2025 #1 model)
3. **ROC Curve Reality**: 5% absolute sensitivity change = massive FA rate change in low FA region

**What This Means**:
- Going from 4 FA/24h → 1 FA/24h requires **~25% sensitivity drop** (not improvement!)
- Current README targets (1 FA/24h @ 75%) are **64 percentage points above SOTA**
- Would require **fundamental breakthrough**, not incremental improvement

### Our README Currently Claims (NEEDS REVISION)

| FA Rate | Target Sensitivity | Reality Check |
|---------|-------------------|---------------|
| 10 FA/24h | >95% | SeizureTransformer: 34% ❌ (61 points off) |
| 5 FA/24h | >90% | SeizureTransformer: ~25% ❌ (65 points off) |
| 1 FA/24h | >75% | Temple SOTA: 50% @ 4 FA ❌ (impossible) |

**Action Required**: Update README with realistic targets before training completes

---

## 📚 REFERENCES

### Primary Sources
1. **Temple NEDC Research Team** - Personal correspondence, October 2025 (4 FA/24h @ 50% sensitivity benchmark from clinical deployments)
2. **Wu et al. 2025** - SeizureTransformer: arXiv:2504.00336, EpilepsyBench #1 winner, Table I (competition results)
3. **Shah et al. 2021** - "Validation of Temporal Scoring Metrics for Automatic Seizure Detection," Table 4 (Neureka 2020 results)
4. **Our SeizureTransformer Evaluation** - reference_repos/SeizureTransformer/docs/results/FINAL_COMPREHENSIVE_RESULTS_TABLE.md (NEDC v6.0.0 on TUSZ v2.0.3 eval)
5. **NEDC Scoring Tools** - v6.0.0, Temple University official scorer
6. **TUSZ Dataset** - v2.0.3, patient-disjoint splits (865 files, 127.7 hours, 469 seizures in eval set)

### Key Papers
- **Shah et al. 2018**: TUSZ dataset description and annotation methodology
- **Shah et al. 2021**: NEDC TAES/OVERLAP/EPOCH scoring methodology validation (Neureka 2020 results)
- **Dan et al. 2024**: SzCORE scoring framework with clinical tolerances (EpilepsyBench standard)
- **Roy et al. 2021**: Clinical deployment goals (75% sensitivity, ~1 FA/24h human reviewer level)

### Neureka 2020 Competition (Shah et al. 2021)
**Dataset**: TUSZ v1.5.1 eval set, 16 participating systems

- **Best system (sia)**: 1 FA/24h @ **11.37% sensitivity** (TAES scoring)
- **Baseline (nedc)**: 17 FA/24h @ **35.54% sensitivity** (TAES scoring)
- **Key Insight**: Achieving 1 FA/24h with good sensitivity (>50%) is **extremely difficult**
- **Scoring**: Competition used TAES (strictest temporal alignment) as primary metric

---

## 📋 ARCHITECTURE COMPARISON TABLES

### Table 1: EpilepsyBench 2025 Challenge Leaderboard

**Dataset**: Dianalund (Danish EMU, 4360 hours, 65 patients)
**Scoring**: SzCORE Event (most permissive - ±30s/60s tolerances, merges <90s)

| Rank | Model | Architecture | Input (s) | Sensitivity | FA/24h | F1 | Params |
|------|-------|--------------|-----------|-------------|--------|-----|---------|
| 1 | **SeizureTransformer** | U-Net + CNN + Transformer | 60 | **37%** | **1.0** | **0.43** | ~41M |
| 2 | Van Gogh Detector | CNN + Transformer | N×10 | 39% | 3.0 | 0.36 | - |
| 3 | S4Seizure | S4 (SSM) | 12 | 30% | 2.0 | 0.34 | - |
| 4 | DeepSOZ-HEM | LSTM + Transformer | 600 | 58% | 14.0 | 0.31 | - |
| 5 | HySEIZa | Hyena-Hierarchy + CNN | 12 | 60% | 13.0 | 0.26 | - |
| 6 | Zhu-Transformer | CNN + Transformer | 25 | 46% | 24.0 | 0.20 | - |
| 7 | SeizUnet | U-Net + LSTM | 30 | 16% | 4.0 | 0.19 | - |

**Key Observations**:
- **SeizureTransformer wins** with lowest FA rate (1.0) but moderate sensitivity (37%)
- Trade-off visible: HySEIZa/DeepSOZ-HEM have 60% sensitivity but 13-14 FA/24h
- Different operating points on ROC curve - no single "best" architecture

---

### Table 2: Neureka 2020 Competition Top Systems

**Dataset**: TUSZ v1.5.1 eval set
**Scoring**: TAES (strictest - partial credit, time-aligned)

| System | Architecture | Sensitivity (TAES) | FA/24h (TAES) | Sensitivity (OVLP) | FA/24h (OVLP) |
|--------|--------------|-------------------|---------------|-------------------|---------------|
| **sia** (winner) | Attention-Gated U-Net | **11.37%** | **1.0** | ~26% | ~1.6 |
| lzk | - | ~20% | ~8 | ~40% | ~5 |
| pnc98 | - | ~18% | ~14 | ~26% | ~14 |
| yff | Time Delay NN + LSTM | 14.03% | ~10 | ~26% | ~14 |
| **nedc** (baseline) | Channel-Dependent | **35.54%** | **17.2** | **51.57%** | ~17 |

**Key Observations**:
- **sia** optimized for low FA (1.0) at cost of sensitivity (11.37%)
- **nedc** baseline shows higher sensitivity (35.54%) with moderate FA (17.2)
- TAES vs OVLP: Same model, 2-3× difference in metrics
- Best system (sia) still only 11% sensitivity at 1 FA/24h

---

### Table 3: SeizureTransformer Performance Across Datasets

**Source**: Wu et al. 2025 (paper) + Our NEDC evaluation (TUSZ)

| Dataset | Scorer | Hours | Sensitivity | FA/24h | AUROC | F1 |
|---------|--------|-------|-------------|--------|-------|-----|
| **Dianalund** (Nordic EMU) | SzCORE Event | 4360 | **37%** | **1.0** | - | **0.43** |
| **TUSZ eval** (Temple) | SzCORE Event | 127.7 | **52.35%** | **8.59** | 0.902 | 0.485 |
| **TUSZ eval** (Temple) | NEDC OVERLAP | 127.7 | **45.63%** | **26.89** | 0.902 | 0.414 |
| **TUSZ eval** (Temple) | NEDC TAES | 127.7 | **65.21%** | **136.73** | 0.902 | 0.240 |
| **TUSZ test** (AUROC only) | - | 42.7 | - | - | **0.876** | - |

**Key Observations**:
- **8.6× FA degradation** across datasets: 1.0 (Dianalund/SzCORE) → 8.59 (TUSZ/SzCORE)
- **3.1× scorer impact** on same dataset: 8.59 (SzCORE) → 26.89 (OVERLAP) → 136.73 (TAES)
- AUROC (0.876-0.902) stays high despite FA/sensitivity variations
- **Dataset shift + scorer choice** compounds to create 137× FA difference!

---

### Table 4: Recent SSM-Based Architectures (2024-2025)

**Models using State Space Models (Mamba, S4) for EEG**

| Model | Architecture | Innovation | Dataset | Best Result | Year |
|-------|--------------|------------|---------|-------------|------|
| **SeizureTransformer** | U-Net + Res CNN + Transformer | Time-step detection | TUSZ + Siena | AUROC 0.876 | 2025 |
| **EEGMamba** | BiMamba + MoE + ST-Adaptive | Multi-task learning | 8 datasets | SOTA on multiple tasks | 2024 |
| **EvoBrain** | Dual Mamba + GCN + Dynamic LPE | Explicit dynamic graphs | TUSZ | +23% AUROC vs baseline | 2025 |
| **S4Seizure** | S4 (SSM) | Linear complexity | Dianalund | 30% @ 2 FA/24h | 2025 |
| **Brain-Go-Brr (FLA Exp4)** | TCN + BiGatedDeltaNet + GNN + Dynamic LPE | Dual-stream + dynamic PE | TUSZ eval | 35.9% @ 10 FA/24h (AUROC 0.8654) | 2025 |

**Key Observations**:
- **SSM renaissance**: Mamba/S4 replacing Transformers for linear complexity
- **EEGMamba**: First universal multi-task EEG model (seizure, emotion, sleep, MI)
- **EvoBrain**: First to prove time-then-graph > graph-then-time theoretically
- **Our V3**: First dual-stream (19× node + 171× edge) with dynamic LPE

---

### Table 5: Operating Point Comparisons (Clinical Targets)

**How different models perform at clinically relevant FA thresholds**

| Model | 1 FA/24h | 4 FA/24h | 10 FA/24h | Notes |
|-------|----------|----------|-----------|-------|
| **Temple SOTA** | - | **50% sens** | - | Personal correspondence |
| **SeizureTransformer** (Dianalund) | **37% sens** | - | - | SzCORE Event scorer |
| **SeizureTransformer** (TUSZ) | NOT TESTED | - | **34% sens** | NEDC OVERLAP @ 10.27 FA |
| **Brain-Go-Brr (FLA Exp4)** | **5.8% sens** | - | **35.9% sens** | OVERLAP-style scorer on TUSZ eval (also reports 5 FA and 2.5 FA) |
| **Neureka sia** (TUSZ) | **11% sens** | - | - | NEDC TAES |
| **Neureka nedc** (TUSZ) | - | - | - | 35.54% @ 17.2 FA (TAES)* |
| **Our Target** | - | **≥55% sens** | **≥75% sens** | Optimistic goal |

*Note: Neureka nedc baseline measured at 17.2 FA/24h, not 4 FA target. Included for reference.

**Reality Check**:
- **1 FA/24h**: Best achieved is 37% (SeizureTransformer on permissive SzCORE)
- **4 FA/24h**: Temple's 50% is the clinical gold standard (verified clinical data)
- **10 FA/24h**: Brain-Go-Brr (FLA Exp4) hits 35.9% (OVERLAP-style) vs SeizureTransformer 33.90% (NEDC/Python OVERLAP)
- **Gap to 75% @ 10 FA**: Would need **39.1 percentage point improvement** over current Exp4

---

## 🎯 NEXT STEPS (Post-Exp4)

1. Add a 4 FA/24h operating point for Exp4 (current eval exports 10/5/2.5/1 only)
2. Run official NEDC v6.0.0 scoring on Exp4 outputs for publication-ready OVERLAP/TAES/SzCORE numbers
3. Iterate toward Temple SOTA (≈50% @ 4 FA/24h) via post-processing + calibration + model changes

---

## ✅ SUCCESS CRITERIA SUMMARY

### Must Achieve (Minimum for Publication)
- **NEDC OVERLAP**: ≤20 FA/24h @ ≥45% sensitivity
- **OR SzCORE Event**: ≤10 FA/24h @ ≥50% sensitivity

### Should Achieve (Strong Publication)
- **NEDC OVERLAP**: ≤8 FA/24h @ ≥50% sensitivity
- Matches/beats Temple verified systems

### Dream Scenario (Top-Tier Publication)
- **NEDC OVERLAP**: ≤4 FA/24h @ ≥60% sensitivity
- Clear path to clinical deployment

---

**Last Updated**: 2025-12-20
**Next Review**: After next full eval sweep (including 4 FA/24h)
