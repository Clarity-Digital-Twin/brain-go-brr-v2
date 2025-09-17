# REFERENCE_REPOS.md - Critical Code References for Brain-Go-Brr v2

## 🎯 Purpose
These repositories contain reference implementations for our Bi-Mamba-2 + U-Net + ResCNN architecture. Clone these into `/reference_repos/` to study implementation patterns.

## 🔥 CRITICAL REPOS TO CLONE

### 1. **NEDC-BENCH** (OUR EVALUATION FRAMEWORK) ⭐⭐⭐⭐⭐
```bash
git clone https://github.com/Clarity-Digital-Twin/nedc-bench.git reference_repos/nedc-bench
```
**WHY THIS IS GENIUS:**
- **ACTUAL SCORING CODE**: TAES, OVERLAP, DP algorithms we'll be evaluated against
- **CSV_BI FORMAT**: Exact output format for Temple evaluation
- **TRAINING INTEGRATION**: Can use Beta pipeline for real-time scoring during training
- **100% PARITY**: Validated against NEDC v6.0.0
- **KEY FILES**:
  - `src/nedc_bench/algorithms/taes.py` - TAES implementation
  - `src/nedc_bench/models/annotations.py` - CSV_BI format
  - `nedc_eeg_eval/v6.0.0/` - Original Temple code

### 2. **STATE-SPACES/MAMBA** (CORE SSM) ⭐⭐⭐⭐⭐
```bash
git clone https://github.com/state-spaces/mamba.git reference_repos/mamba
```
**WHAT TO STEAL:**
- `mamba_ssm/modules/mamba2.py` - Mamba-2 implementation
- `mamba_ssm/models/mixer_seq_simple.py` - Model architecture
- SSD (Structured State Space Duality) algorithm
- Hardware-aware optimizations

### 3. **SEIZURETRANSFORMER** (U-NET + RESCNN) ⭐⭐⭐⭐⭐
```bash
# Already cloned
```
**WHAT TO STEAL:**
- `architecture.py:120-212` - U-Net encoder/decoder
- `architecture.py:215-260` - ResCNN blocks
- Skip connection patterns
- Channel progression [64, 128, 256, 512]

### 4. **HUSTVL/VIM** (BIDIRECTIONAL MAMBA) ⭐⭐⭐⭐
```bash
git clone https://github.com/hustvl/Vim.git reference_repos/vision-mamba
```
**WHAT TO STEAL:**
- Bidirectional SSM wiring
- Forward/backward concatenation patterns
- Position embedding strategies
- Middle token placement

### 5. **BRAINDECODE** (EEG CNN PATTERNS) ⭐⭐⭐⭐
```bash
git clone https://github.com/braindecode/braindecode.git reference_repos/braindecode
```
**WHAT TO STEAL:**
- `braindecode/models/eegnet.py` - Temporal convolutions
- `braindecode/models/deep4.py` - Deep ConvNet patterns
- `braindecode/models/shallow_fbcsp.py` - Filter bank patterns
- EEG-specific normalization

## 📦 WHAT WE DON'T NEED

### ❌ SKIP THESE:
- **seizy** - Hysteresis is trivial with scipy/scikit-image
- **EEGNet standalone repos** - Braindecode has better implementations
- **Random seizure repos** - We have SeizureTransformer + NEDC-BENCH

## 🧪 STANDARD LIBRARIES WE'LL USE

```python
# Hysteresis & Morphology (no repo needed)
from skimage.filters import apply_hysteresis_threshold
from scipy.ndimage import binary_opening, binary_closing

# These give us:
# - Double thresholding (τ_on=0.86, τ_off=0.78)
# - Morphological operations
# - Minimum duration filtering
```

## 🏗️ COMPONENT MAPPING

| Our Component | Reference Repo | Specific Files |
|---------------|----------------|----------------|
| **U-Net Encoder** | SeizureTransformer | `architecture.py:120-170` |
| **U-Net Decoder** | SeizureTransformer | `architecture.py:171-212` |
| **ResCNN Stack** | SeizureTransformer | `architecture.py:215-260` |
| **Bi-Mamba-2** | mamba + Vim | `mamba2.py` + bidirectional patterns |
| **TAES Scoring** | nedc-bench | `algorithms/taes.py` |
| **CSV_BI Format** | nedc-bench | `models/annotations.py` |
| **Temporal Conv** | braindecode | `eegnet.py` |
| **Hysteresis** | scipy/skimage | Built-in functions |

## 🚀 CLONE COMMANDS

```bash
# Run these from project root
mkdir -p reference_repos
cd reference_repos

# Clone all critical repos
git clone https://github.com/Clarity-Digital-Twin/nedc-bench.git
git clone https://github.com/state-spaces/mamba.git
git clone https://github.com/hustvl/Vim.git vision-mamba
git clone https://github.com/braindecode/braindecode.git

# SeizureTransformer already exists
cd ..
```

## 📝 INTEGRATION STRATEGY

### During Development:
1. **Study** reference implementations
2. **Adapt** patterns to our architecture
3. **Preserve** APIs from `src/experiment/`

### During Training:
1. **Import** nedc-bench Beta pipeline
2. **Score** outputs in real-time
3. **Optimize** directly for TAES metrics

### During Evaluation:
1. **Format** outputs as CSV_BI
2. **Run** through nedc-bench
3. **Validate** against Temple benchmarks

---
**Mission: Build O(N) seizure detection with these battle-tested components** 🚀