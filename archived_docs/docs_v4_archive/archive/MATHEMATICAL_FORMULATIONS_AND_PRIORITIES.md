# Mathematical Formulations and Priority Analysis

**Brain-Go-Brr v3.9.2 (CI/CD Stability) - October 9, 2025**

This document catalogs all mathematical formulations, priority blocks, validation metrics, and critical parameters in the codebase, providing a comprehensive reference for understanding the system's technical foundation.

---

## Table of Contents

1. [Priority System Overview](#priority-system-overview)
2. [Mathematical Formulations by Component](#mathematical-formulations-by-component)
3. [TAES Metrics and Clinical Validation](#taes-metrics-and-clinical-validation)
4. [Critical Parameters and Constants](#critical-parameters-and-constants)
5. [Complexity Analysis](#complexity-analysis)
6. [Validation Methodology](#validation-methodology)

---

## Priority System Overview

### Priority Definitions

The codebase uses a 6-tier priority system for issue tracking and technical debt management:

- **P0**: Breaks training/inference or corrupts results (URGENT - blocks deployment)
- **P1**: High risk of silent degradation; not a hard block (HIGH - quality impact)
- **P2**: Medium risk/confusion; suboptimal defaults (MEDIUM - maintenance burden)
- **P3**: Minor cleanup (LOW - polish and refinement)
- **P4**: Nice-to-have (OPTIONAL - future enhancements)
- **P5**: Research ideas (EXPERIMENTAL - post-training optimization)

### Current Status (v3.9.2)

```
✅ P0: 0 issues (all blockers resolved)
✅ P1: 0 issues (all quality risks mitigated)
✅ P2: 0 issues (all code quality debt paid)
✅ P3: 0 issues (manifest naming cleanup complete)
🟡 P4/P5: Optional ideas (post-training optimization only)
```

**Zero technical debt** - All critical (P0-P3) issues have been resolved across the entire project.

### Resolved Critical Issues (Historical)

#### P0 Blockers (All Fixed in v3.8.0-v3.8.3)
1. **NPZ Cache Contamination** - Fixed by migration to memory-mapped NPY cache
2. **Non-finite Logits** - Fixed by gradient clipping and outlier clamping (±10σ)
3. **Edge Similarity Explosions** - Fixed by safety margin (0.01) from ±1 boundaries
4. **Gradient Spikes (7.03+)** - Fixed by detaching eigenvectors in gnn_pyg.py:268

#### P1 Quality Issues (All Fixed in v3.8.3)
1. **Mixed Precision on RTX 4090** - Disabled for local training (causes NaNs)
2. **Modal XID 31 GPU Crashes** - Fixed by unique Triton cache directories
3. **Validation OOM** - Fixed by disk-backed validation storage (v3.9.1)

#### P2 Maintenance Issues (All Fixed)
1. **Code duplication** - 120 lines eliminated
2. **Magic numbers** - All moved to constants.py
3. **Type annotations** - Complete coverage

---

## Mathematical Formulations by Component

### 1. TAES (Time-Aligned Event Scoring)

**Location**: `src/brain_brr/eval/metrics.py:79-146`

**Formula**:
```
TAES = base_score - penalty

where:
    base_score = (1/|R|) × Σᵣ min(1, overlap_duration(r, P) / duration(r))
    penalty = α × (fp_duration / total_pred_duration)

    R = reference events
    P = predicted events
    α = 0.15 (false alarm penalty weight)
```

**Purpose**: Measures temporal alignment quality between predicted and reference seizure events, accounting for both overlap quality and false alarm duration.

**Key Properties**:
- Range: [0, 1]
- Penalizes both missed detections (low overlap) and false alarms (excessive duration)
- Each reference event contributes equally (average, not sum)

### 2. False Alarms per 24 Hours (FA/24h)

**Location**: `src/brain_brr/eval/metrics.py:149-178`

**Formula**:
```
FA/24h = (FA_count / total_hours) × 24

where:
    FA_count = |{p ∈ P : ∀r ∈ R, overlap(p, r) = 0}|

    P = predicted events
    R = reference events
```

**Purpose**: Clinical metric for alarm fatigue assessment. Target: <1 FA/24h for sustained clinical use.

### 3. Expected Calibration Error (ECE)

**Location**: `src/brain_brr/eval/metrics.py:40-76`

**Formula**:
```
ECE = Σᵢ |accuracy(Bᵢ) - confidence(Bᵢ)| × P(Bᵢ)

where:
    Bᵢ = bins of predicted probabilities [i/n, (i+1)/n]
    n = number of bins (default: 15)
    accuracy(Bᵢ) = mean(labels ∈ Bᵢ)
    confidence(Bᵢ) = mean(probs ∈ Bᵢ)
    P(Bᵢ) = proportion of samples in bin i
```

**Purpose**: Measures calibration quality of probability predictions. Lower is better (0 = perfect calibration).

### 4. Sensitivity at FA Thresholds

**Location**: `src/brain_brr/eval/metrics.py:313-422`

**Algorithm** (Binary Search for Threshold):
```
Input: probabilities P, labels L, FA target τ
Output: sensitivity at operating point τ

1. Binary search over hysteresis thresholds [0.08, 1.0]:
   - For each tau_on ∈ [low, high]:
     - Set tau_off = max(0, tau_on - 0.08)
     - Apply postprocessing: hysteresis → morphology → duration filtering
     - Convert to events E
     - Compute FA_rate = FA/24h(E, L)
     - If FA_rate > τ: increase tau_on
     - Else: decrease tau_on, save as best_tau_on

2. At best_tau_on, compute event-level sensitivity:
   sensitivity = |{r ∈ R : ∃p ∈ P, overlap(r, p) > 0}| / |R|
```

**Convergence**: Max 20 iterations (THRESHOLD_SEARCH_MAX_ITERS), tolerance 1e-4

### 5. Edge Similarity (Cosine)

**Location**: `src/brain_brr/models/edge_features.py:41-107`

**Formula**:
```
sim(i, j, t) = clamp(cos(xᵢₜ, xⱼₜ), -1+ε, 1-ε)

where:
    xᵢₜ = electrode i features at time t (normalized: ||xᵢₜ|| = 1)
    cos(xᵢₜ, xⱼₜ) = xᵢₜᵀ xⱼₜ
    ε = 0.01 (safety margin from ±1 boundaries)

Result: (B, E, T, 1) where E = 19×18/2 = 171 unique pairs
```

**Critical Safety Measures** (v3.3.0+):
1. Input sanitization: clamp to [-10, 10] before normalization
2. Robust normalization: denominator ≥ EPSILON_NUMERICAL = 1e-6
3. Similarity clamping: [-0.99, 0.99] to prevent boundary explosions
4. NaN replacement: nan_to_num before returning

**Why Safety Margin?**: Without clamping, edge similarities of exactly ±1.0 can propagate through Mamba layers and cause gradient explosions (observed in v3.2.x).

### 6. Laplacian Positional Encoding (Dynamic)

**Location**: `src/brain_brr/models/gnn_pyg.py:214-368`

**Algorithm**:
```
Input: Adjacency A(t) ∈ ℝ^(N×N) (dynamic, per timestep)
Output: Positional encoding PE(t) ∈ ℝ^(N×k)

1. Compute degree matrix: D(t) = diag(Σⱼ A(t)ᵢⱼ)

2. Compute normalized Laplacian:
   L(t) = I - D(t)^(-1/2) A(t) D(t)^(-1/2)

   Stability: Add ε = 1e-6 to diagonal before inversion

3. Eigendecomposition (fp32 only):
   L(t) = V Λ Vᵀ

   where Λ = diag(λ₁, λ₂, ..., λₙ), λ₁ ≤ λ₂ ≤ ... ≤ λₙ

4. Extract k smallest eigenvectors:
   PE(t) = [v₁, v₂, ..., vₖ] ∈ ℝ^(N×k)

   Default: k = 16 (EvoBrain baseline)

5. **CRITICAL FIX (v3.8.1)**: Detach eigenvectors
   PE(t) = PE(t).detach()

   Reason: PyTorch eigendecomposition backward uses 1/(λᵢ - λⱼ)
   which explodes when eigenvalues are close (near-degenerate)

6. Sign consistency (optional):
   PE(t) ← PE(t) × sign(Σᵢ PE(t)ᵢⱼ)
```

**Computational Cost**: O(N³) per graph (eigendecomposition), vectorized across B×T graphs.

**Stability Measures**:
- Eigenvalue clamping: [EPSILON_NUMERICAL, EIGENVALUE_CLAMP_MAX] = [1e-6, 1e8]
- FP32 computation (no AMP) for numerical precision
- Fallback to cached PE if NaN/Inf detected
- Random PE initialization if no valid cache (σ=0.01)

### 7. SSGConv (Simple Spectral Graph Convolution)

**Location**: `src/brain_brr/models/gnn_pyg.py:159-166` (PyG implementation)

**Formula**:
```
h⁽ˡ⁺¹⁾ = (1 - α) Σₖ₌₀ᴷ θₖ Lᵏ h⁽ˡ⁾ + α h⁽ˡ⁾

where:
    L = normalized Laplacian
    K = hop count (default: 2)
    α = skip connection weight (default: 0.05, from EvoBrain)
    θₖ = learnable parameters for k-hop aggregation
```

**Purpose**: Aggregates features from K-hop neighborhoods while maintaining a skip connection to preserve local information.

**EvoBrain Reference**: Line 332 (α=0.05), Lines 331-334 (2-layer architecture)

### 8. Focal Loss

**Location**: `src/brain_brr/train/loop.py:212` (application), `src/brain_brr/constants.py:188` (parameters)

**Formula**:
```
FL(p, y) = -αₜ (1 - pₜ)^γ log(pₜ)

where:
    pₜ = p     if y = 1 (seizure)
    pₜ = 1 - p if y = 0 (background)

    α = 0.5  (neutral class weighting for 12:1 imbalance)
    γ = 2.0  (focusing parameter for hard examples)
```

**Purpose**: Addresses class imbalance (12:1 background:seizure) by:
1. Down-weighting easy examples: (1 - pₜ)^γ → 0 as pₜ → 1
2. Focusing on hard misclassifications: (1 - pₜ)^γ → 1 as pₜ → 0

**Why Not Weighted CE?**: Focal loss provides better gradient flow for rare seizure events than simple class weights.

### 9. Hysteresis Thresholding

**Location**: `src/brain_brr/post/postprocess.py` (implementation), configs (parameters)

**Algorithm**:
```
Input: Probabilities p(t) ∈ [0, 1], thresholds τ_on, τ_off
Output: Binary mask m(t) ∈ {0, 1}

State machine:
    m(t) = 1  if m(t-1) = 0 and p(t) ≥ τ_on   (seizure onset)
    m(t) = 0  if m(t-1) = 1 and p(t) < τ_off  (seizure offset)
    m(t) = m(t-1) otherwise                    (maintain state)

Default values (clinical):
    τ_on  = 0.86  (high threshold to reduce false alarms)
    τ_off = 0.78  (lower threshold to capture full duration)
    δ     = τ_on - τ_off = 0.08  (hysteresis gap)
```

**Purpose**: Prevents rapid oscillations in predictions, ensuring stable event boundaries. Critical for clinical acceptance.

### 10. Balanced Sampling (BalancedSeizureDataset)

**Location**: `src/brain_brr/data/datasets.py` (implementation)

**Formula** (Oversampling Ratio):
```
For each seizure window w with class c:
    sample_weight(w) = total_samples / (n_classes × count(c))

This achieves approximately:
    P(seizure window) ≈ 0.30-0.35 in batches
    P(background)     ≈ 0.65-0.70 in batches

versus natural distribution:
    P(seizure)    ≈ 0.08 (8% of data)
    P(background) ≈ 0.92 (92% of data)
```

**Purpose**: Ensures sufficient seizure exposure during training without completely eliminating class imbalance (which would harm generalization).

**Critical Note**: Validation uses natural distribution (8% seizures) to measure real-world performance.

---

## TAES Metrics and Clinical Validation

### Clinical Performance Targets

Based on [Temple Any-Event Scoring (TAES)](literature/markdown/picone-2021-NEDC-SCORING):

| False Alarm Rate | Target Sensitivity | Clinical Viability |
|------------------|-------------------|-------------------|
| 10 FA/24h        | >95%              | Initial deployment |
| 5 FA/24h         | >90%              | Standard care     |
| **1 FA/24h**     | **>75%**          | **Gold standard** 🎯 |

**Critical Insight**: At 10 FA/24h, alarm fatigue leads to system abandonment. <1 FA/24h enables sustained clinical use.

### TAES vs. Other Metrics

**Correlation Analysis** (from Picone et al.):
- ATWV ↔ DPALIGN: High correlation (0.85 sensitivity, 0.90 specificity)
- OVLP ↔ TAES: Moderate correlation (0.78 sensitivity, 0.95 specificity)
- EPOCH ↔ Other: Low correlation (~0.5-0.6) - biased to long events

**Why TAES?**:
1. Penalizes both missed detections AND excessive false alarm duration
2. Event-level scoring (not sample-level) matches clinical workflow
3. Proven correlation with user acceptance (Picone et al., 2021)
4. Standard in TUH EEG Seizure Corpus benchmarks

### Validation Pipeline

**Location**: `src/brain_brr/train/val_step.py`, `src/brain_brr/eval/metrics.py`

**Algorithm**:
```
1. Window-level predictions:
   - Model outputs: (N_windows, T_samples) probabilities
   - Each window: 60s duration, 10s stride

2. Per-recording stitching:
   - Group windows by file_id
   - Sort by window_start_s
   - Reconstruct continuous timeline (average overlaps)
   - Result: (T_total,) probability series per recording

3. Event extraction:
   - Apply hysteresis thresholding
   - Morphological operations: Opening(11), Closing(31)
   - Duration filtering: 3-600s valid range
   - Merge events within 2s
   - Result: List of (start_s, end_s) events

4. Metrics computation:
   - TAES: Event overlap quality + FA penalty
   - AUROC: Sample-level discrimination
   - Sensitivity@FA: Binary search over thresholds
   - FA/24h: Event-level false alarm rate

5. Disk-backed aggregation (v3.9.1):
   - Store predictions per recording to disk
   - Compute AUROC on accumulated samples (memory-efficient)
   - Run FA sweep over thresholds (one pass)
   - Peak memory: <5GB (vs. 22GB full retention)
```

### Metrics Verification (v3.9.0)

**Status**: ✅ All metrics verified from first principles

**Verification Checklist** (`PRE_TRAINING_VALIDATION.md`):
- [x] TAES formula matches Picone et al. (2021)
- [x] FA/24h correctly normalizes to 24-hour rate
- [x] Sensitivity@FA uses event-level (not sample-level) TP count
- [x] Threshold search converges monotonically
- [x] Window stitching preserves temporal order
- [x] Duration filtering uses correct sampling rate (256 Hz)

---

## Critical Parameters and Constants

### Numerical Stability Constants

**Location**: `src/brain_brr/constants.py`

```python
EPSILON_NUMERICAL = 1e-6      # General numerical stability
EPSILON_LAPLACIAN = 1e-6      # Laplacian regularization
EPSILON_ZERO_CHECK = 1e-9     # Zero division guards
EIGENVALUE_CLAMP_MAX = 1e8    # Eigenvalue upper bound
```

**Usage**:
- **EPSILON_NUMERICAL**: Denominator safety in normalization (edge_features.py:83, :96)
- **EPSILON_LAPLACIAN**: Diagonal regularization in Laplacian (gnn_pyg.py:248)
- **EPSILON_ZERO_CHECK**: Duration checks in TAES (metrics.py:109)
- **EIGENVALUE_CLAMP_MAX**: Prevent eigenvalue overflow (gnn_pyg.py:295, :340)

### Architecture Parameters (EvoBrain-Aligned)

**SSGConv (GNN)**:
```python
GNN_SSGCONV_ALPHA_DEFAULT = 0.05  # Skip connection strength (EvoBrain line 332)
```

**Laplacian PE**:
```python
k_eigenvectors = 16               # Number of PE dimensions (EvoBrain line 858)
```

**Edge Transform**:
```python
edge_transform = nn.Linear(1, 1)  # Learnable edge weights (EvoBrain lines 869-870)
edge_activate = nn.Softplus()     # Non-negative edge weights
```

**Safety Margins** (v3.3.0):
```python
edge_similarity_margin = 0.01     # Safety from ±1 boundaries in edge_scalar_series
```

### Post-Processing Defaults

**Hysteresis** (Clinical Tuned):
```python
HYSTERESIS_TAU_ON = 0.86          # Onset threshold (high to reduce FAs)
HYSTERESIS_TAU_OFF = 0.78         # Offset threshold (lower for full capture)
HYSTERESIS_DELTA = 0.08           # Gap between thresholds
```

**Morphology**:
```python
MORPH_OPENING_SIZE = 11           # Remove small noise bursts
MORPH_CLOSING_SIZE = 31           # Fill brief gaps in events
```

**Duration Filtering**:
```python
MIN_EVENT_DURATION_SEC = 3.0      # Minimum valid seizure (too short = noise)
MAX_EVENT_DURATION_SEC = 600.0    # Maximum valid seizure (10 minutes)
```

**Event Merging**:
```python
TAU_MERGE_SEC = 2.0               # Merge events within 2 seconds
```

### Training Hyperparameters

**Optimization**:
```python
learning_rate = 1e-3              # AdamW default
weight_decay = 1e-4               # L2 regularization
gradient_clip = 0.5               # Global norm clipping (PRIMARY NaN protection)
warmup_epochs = 5                 # Linear warmup (optional)
```

**Loss**:
```python
loss_fn = FocalLoss(alpha=0.5, gamma=2.0)  # For 12:1 imbalance
```

**Regularization**:
```python
dropout = 0.1                     # Standard dropout rate
```

### Hardware-Specific Settings

**Local (RTX 4090)**:
```python
batch_size = 8                    # Optimized for 24GB VRAM
mixed_precision = False           # Disabled - causes NaNs on RTX 4090
num_workers = 0                   # WSL2 multiprocessing fix
```

**Modal (A100-80GB)**:
```python
batch_size = 48                   # Optimized for 80GB VRAM
mixed_precision = True            # Tensor cores enabled
num_workers = 4                   # Safe multiprocessing
gradient_accumulation_steps = 1   # No accumulation needed (batch=48 fits)
```

**Memory Estimates** (FP32 only, approximate):
```
RTX 4090 peak:  ~20GB  (batch=8,  no AMP)
A100 peak:      ~58GB  (batch=48, AMP enabled)
```

**Warning**: Mixed precision on RTX 4090 causes non-finite loss due to Ampere fp16 range limitations. A100 has wider fp16 range and handles AMP correctly.

---

## Complexity Analysis

### Model Complexity (Per Forward Pass)

**Notation**:
- B = batch size
- T = sequence length (960 samples = 3.75s at 256 Hz)
- N = nodes (19 electrodes)
- E = edges (171 pairs)
- D = feature dimension (64)
- k = PE dimension (16)

**Component-wise**:

1. **TCN Encoder**: O(B × T × D)
   - 8 layers, exponentially increasing dilation
   - Stride-down by 16: T_out = T / 16 = 60 timesteps

2. **Node Stream (Mamba)**: O(B × N × T_out × D)
   - 6 layers, d_model=64 per electrode
   - Linear recurrence: O(N × T) per layer (not O(N × T²))

3. **Edge Stream (Mamba)**: O(B × E × T_out × D)
   - 6 layers, d_model=64 per edge
   - Linear recurrence: O(E × T) per layer

4. **Edge Similarity**: O(B × T × N² × D)
   - Cosine similarity: matrix multiplication (N × N) at each timestep
   - Dominant cost: matmul in edge_features.py:89

5. **Adjacency Assembly**: O(B × T × E)
   - Top-k sparsification: O(E log k) per timestep (k=3)

6. **Dynamic Laplacian PE**: O(B × T × N³)
   - Eigendecomposition: O(N³) per graph
   - **Vectorized**: Process B×T graphs in parallel
   - **Semi-dynamic**: Compute every `semi_dynamic_interval` timesteps (default: 1)

7. **GNN (SSGConv)**: O(B × T × (E + N) × D)
   - Message passing: O(E × D) per timestep
   - Node updates: O(N × D) per timestep
   - 2 layers

**Total Complexity**: O(B × T × N³) dominated by eigendecomposition

**Optimization Opportunities**:
- **Semi-dynamic PE**: Reduce to O(B × T/k × N³) by updating PE every k timesteps
- **Static PE**: O(N³) once (pre-compute from structural graph)
- **Spectral methods**: Fast Laplacian eigendecomposition (not yet implemented)

### Training Time Estimates

**Empirical Measurements** (100 epochs on TUSZ train split):

- **Local (RTX 4090)**: ~300 hours (~3h/epoch)
  - Bottleneck: Eigendecomposition on older GPU architecture

- **Modal (A100-80GB)**: ~100 hours (~1h/epoch)
  - Faster eigendecomposition + tensor cores
  - Cost: ~$319 for 100 epochs

**Smoke Test** (Fast Validation):
- **3 files** (local/Docker): ~5 minutes
- **50 files** (Modal): ~10 minutes

### Memory Complexity

**Model Parameters**: 31M total

**Activation Memory** (Training):
```
TCN:     B × T × D × 8_layers      ≈ 8 × B × T × D
Mamba:   B × N × T × D × 6_layers  ≈ 6 × B × N × T × D
GNN:     B × T × N × D × 2_layers  ≈ 2 × B × T × N × D

Total:   O(B × T × N × D)
```

**Cache Size** (Preprocessed Data):
```
Train:   4667 NPY files  ≈ 250 GB memory-mapped
Dev:     1832 NPY files  ≈ 100 GB memory-mapped
Total:   ~350 GB on disk (minimal RAM usage due to mmap)
```

**Checkpoint Size**: ~125 MB per epoch (model + optimizer states)

---

## Validation Methodology

### Test Suite Structure

**Coverage**: 75%+ across all modules

**Test Categories**:
1. **Unit Tests** (Fast, Isolated)
   - `tests/unit/`: Component-level correctness
   - Examples: edge_features, metrics, datasets
   - Run time: ~30 seconds

2. **Integration Tests** (Components Together)
   - `tests/integration/`: Multi-component workflows
   - Examples: TCN → Mamba → GNN pipeline, postprocessing
   - Run time: ~2 minutes (CPU only)

3. **Clinical Tests** (Domain Validation)
   - `tests/clinical/`: TAES metrics, channel order, EEG standards
   - Examples: Clinical sensitivity targets, seizure duration validation
   - Run time: ~1 minute

4. **Performance Tests** (GPU Required)
   - `tests/performance/`: Speed and memory benchmarks
   - Examples: TCN throughput, batch size limits
   - Run time: ~3 minutes (requires GPU)
   - **WARNING**: Do not run during training (OOM risk)

### Testing During Training

**Critical Rule**: Use `make ts` (training-safe tests) while training is running.

**Why?**:
- Training uses ~20GB VRAM on RTX 4090
- Performance tests allocate another ~15-20GB
- Combined load → OOM and crash

**Solution**:
```bash
# During training
make ts  # or: pytest -m "not gpu and not performance and not slow"

# After training stopped
make test  # Full test suite with coverage
```

### Continuous Integration

**GitHub Actions Workflows**:

1. **Main CI** (`.github/workflows/ci.yml`)
   - Triggers: Push to development/main, PRs
   - Tests: All except `@pytest.mark.slow`
   - Timeout: 5 minutes per test
   - Status: ✅ Passing

2. **Release CI** (`.github/workflows/release.yml`)
   - Triggers: Git tags matching `v*.*.*`
   - Tests: Fast subset (skip slow, GPU, performance)
   - Timeout: 60 seconds per test
   - Purpose: Quick validation before GitHub release creation
   - Status: ✅ Passing (v3.9.2)

3. **FLA CI** (`.github/workflows/fla.yml`)
   - Triggers: Pushes to feature/flash-linear-attention
   - Tests: Full FLA code path (both BiMamba2 and FLA configs)
   - Purpose: Ensure FLA research doesn't break existing BiMamba2 pipeline
   - Status: ✅ Passing

### Smoke Test Standards

**Purpose**: Fast validation that pipeline is functional before committing to full training.

**Configurations**:
```yaml
# Local/Docker: 3 files
configs/local/smoke_bimamba.yaml:
  data.limit_files: 3  # or set BGB_SMOKE_TEST=1
  training.epochs: 1

# Modal: 50 files
configs/modal/smoke_bimamba.yaml:
  data.limit_files: 50  # or set BGB_LIMIT_FILES=50
  training.epochs: 1
```

**Expected Results**:
- Local: ~5 minutes, loss decreases, no NaN/Inf
- Modal: ~10 minutes, validation completes, metrics logged

**When to Run**:
- After code changes affecting training loop
- After dependency upgrades
- Before starting 100-epoch runs

### Pre-Training Validation Checklist

**Location**: `PRE_TRAINING_VALIDATION.md`

Comprehensive checklist covering:
- ✅ Data pipeline correctness (cache, sampling, augmentation)
- ✅ Model architecture sanity (forward pass, gradients, parameter counts)
- ✅ Metrics pipeline accuracy (TAES, FA/24h, sensitivity@FA)
- ✅ Training stability (NaN protection, gradient clipping, checkpoints)
- ✅ Configuration consistency (local vs. Modal, smoke vs. full)

**Status**: All items verified for v3.9.2 release.

---

## Summary and Quick Reference

### Zero Technical Debt Milestone (v3.9.2)

```
Total Issues Resolved: 47
- P0 Blockers:     12 fixed
- P1 Quality:      18 fixed
- P2 Maintenance:  14 fixed
- P3 Polish:        3 fixed

Remaining: 0 critical issues
```

### Key Mathematical Insights

1. **TAES = Overlap Quality - FA Penalty**
   - Clinically meaningful: reflects user acceptance
   - Proven correlation with alarm fatigue (Picone et al.)

2. **Gradient Explosion Root Cause = Eigendecomposition Backward**
   - Solution: Detach eigenvectors (gnn_pyg.py:268)
   - Rationale: PE is positional, not learned (fixed coordinates)

3. **Edge Similarity Boundary Explosions**
   - Cause: cos(x, y) = ±1.0 → Mamba amplification
   - Solution: Clamp to [-0.99, 0.99] with margin=0.01

4. **Focal Loss > Weighted CE for Rare Events**
   - Down-weights easy examples: (1 - p)^2 term
   - Better gradient flow for 8% seizure prevalence

5. **Hysteresis Prevents Oscillations**
   - Critical for clinical acceptance (stable event boundaries)
   - Gap = 0.08 (empirically tuned on TUSZ)

### Quick Access Locations

**Formulas**:
- TAES: `src/brain_brr/eval/metrics.py:79-146`
- Laplacian PE: `src/brain_brr/models/gnn_pyg.py:214-368`
- Edge Similarity: `src/brain_brr/models/edge_features.py:41-107`

**Constants**:
- All numerical: `src/brain_brr/constants.py`
- All configs: `configs/README.md`

**Validation**:
- Test suite: `tests/` (75%+ coverage)
- Pre-training checklist: `PRE_TRAINING_VALIDATION.md`
- Metrics verification: `docs/06-evaluation/metrics-and-taes.md`

### Performance Quick Check

```bash
# Verify model forward pass (no GPU required)
pytest tests/unit/models/ -v

# Verify metrics pipeline (includes TAES)
pytest tests/clinical/test_taes_metrics.py -v

# Smoke test full pipeline (3 files, ~5 min)
make smoke-bimamba

# Full training (100 epochs)
# Local:  make train-bimamba  (~300 hours)
# Modal:  modal run --detach deploy/modal/app.py --action train --config configs/modal/train_bimamba.yaml  (~100 hours)
```

---

**Document Version**: v1.0
**Last Updated**: October 9, 2025 (v3.9.2 - CI/CD Stability)
**Maintained By**: Brain-Go-Brr Development Team
**Purpose**: Comprehensive reference for all mathematical formulations, priority tracking, and validation methodology in the codebase.
