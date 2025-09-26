# COMPREHENSIVE NaN FIXES - FIXING EVERYTHING AT ONCE

## Strategy: Fix ALL potential NaN sources rather than iterative testing

### 1. EIGENDECOMPOSITION (Dynamic PE) - HIGHEST RISK
**File**: `src/brain_brr/models/gnn_pyg.py`

**Issues**:
- Eigendecomposition can produce NaN/Inf on singular matrices
- Float precision issues
- Sign ambiguity

**Fixes to Apply**:
```python
# Line ~200 in compute_dynamic_pe()
# Add these safeguards:
1. Double regularization: L = L + 1e-5 * I (increase from 1e-6)
2. Add condition number check before eigh
3. Add fallback to identity matrix if eigh fails
4. Clamp eigenvalues more aggressively [1e-6, 2.0]
5. Add try-except with fallback PE
```

### 2. TCN INPUT HANDLING
**File**: `src/brain_brr/models/tcn.py`

**Issues**:
- No input validation
- Weight norm can explode
- Padding issues

**Fixes to Apply**:
```python
# At start of forward()
1. Input clamping: x = torch.clamp(x, -100, 100)
2. Check for NaN/Inf inputs and replace
3. Add LayerNorm after input projection
4. Initialize weights with smaller variance
```

### 3. EDGE FEATURES CALCULATION
**File**: `src/brain_brr/models/detector.py`

**Issues**:
- Cosine similarity can produce NaN
- Edge projection not initialized properly
- Clamping happens too late

**Fixes to Apply**:
```python
# Lines ~200-220
1. Normalize inputs BEFORE cosine_similarity
2. Add epsilon to denominators
3. Clamp immediately after calculation
4. Initialize edge_in_proj with xavier_uniform_(gain=0.1)
```

### 4. MAMBA STABILITY
**File**: `src/brain_brr/models/mamba.py`

**Issues**:
- CUDA kernels unstable on RTX 4090
- SSM states can accumulate

**Fixes to Apply**:
```python
1. Force fallback on RTX 4090 by default
2. Add state clamping every N steps
3. Use smaller dt_init
4. Add gradient checkpointing
```

### 5. DECODER/PROJECTION HEAD
**File**: `src/brain_brr/models/detector.py`

**Issues**:
- Final projection can explode
- No bounds on logits

**Fixes to Apply**:
```python
# Lines ~260
1. Add LayerNorm before final projection
2. Initialize projection with gain=0.01
3. Clamp more aggressively [-20, 20]
```

### 6. LOSS COMPUTATION
**File**: `src/brain_brr/train/loop.py`

**Issues**:
- Focal loss can produce extreme gradients
- Log operations on near-zero values

**Fixes to Apply**:
```python
1. Increase epsilon in focal loss (1e-6 → 1e-5)
2. Clamp probabilities [1e-5, 1-1e-5]
3. Add gradient clipping per-parameter
4. Reduce focal_gamma (2.0 → 1.5)
```

### 7. DATA NORMALIZATION
**File**: `src/brain_brr/data/preprocess.py`

**Issues**:
- Some windows might have extreme values
- Z-score can fail on constant channels

**Fixes to Apply**:
```python
1. Add robust scaling option
2. Clip outliers before normalization
3. Add small epsilon to std
4. Check for dead channels
```

### 8. INITIALIZATION
**All model files**

**Issues**:
- Default initialization too large for deep networks
- No special handling for residual connections

**Fixes to Apply**:
```python
1. Use smaller gains (0.1-0.5)
2. Zero-initialize residual projections
3. Careful init for layer norms
4. Special handling for positional encodings
```

## IMPLEMENTATION ORDER

1. **Most Critical First**:
   - Fix eigendecomposition (most likely culprit)
   - Fix TCN input handling
   - Fix edge features

2. **Then Stability**:
   - Add all clamping
   - Fix initializations
   - Add LayerNorms

3. **Finally Polish**:
   - Loss improvements
   - Data preprocessing
   - Comprehensive checks

## TESTING STRATEGY

After implementing ALL fixes:
```bash
# Test with all safeguards ENABLED
export BGB_SAFE_CLAMP=1
export BGB_SANITIZE_INPUTS=1
export BGB_SANITIZE_GRADS=1
export BGB_DEBUG_FINITE=1
export SEIZURE_MAMBA_FORCE_FALLBACK=1

# Run for 500 batches
timeout 600 .venv/bin/python -m src train configs/local/train.yaml
```

## SUCCESS CRITERIA
- No NaN warnings for 500 batches
- Loss decreases normally
- LR follows schedule
- All components enabled (no disabling)

## FINAL FALLBACK
If STILL failing after all fixes:
1. Check CUDA/cuDNN versions
2. Test on different GPU
3. Use Modal A100 exclusively