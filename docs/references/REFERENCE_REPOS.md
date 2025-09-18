# REFERENCE_REPOS.md - Critical Code References for Brain-Go-Brr v2

## 🎯 Purpose
These repositories contain reference implementations for our Bi-Mamba-2 + U-Net + ResCNN architecture. We distinguish between:
- **📚 STUDY-ONLY**: Clone to `/reference_repos/` to study patterns, then write your own
- **📦 IMPORTABLE**: Install via pip/uv and use directly in code

## 🔥 REPOS BY USAGE TYPE

### 1. **NEDC-BENCH** 📦 IMPORTABLE (Our Custom Package) ⭐⭐⭐⭐⭐
**STATUS**: GitHub-only (not on PyPI yet)
**INSTALL**:
```bash
# Install as dependency
uv add "nedc-bench @ git+https://github.com/Clarity-Digital-Twin/nedc-bench.git"

# Or clone for reference
git clone https://github.com/Clarity-Digital-Twin/nedc-bench.git reference_repos/nedc-bench
```
**USE IN CODE**:
```python
from nedc_bench.algorithms import taes
from nedc_bench.models import annotations
```
**KEY FEATURES**:
- TAES, OVERLAP, DP scoring algorithms
- CSV_BI format for Temple evaluation
- Beta pipeline for real-time training metrics
- 100% parity with NEDC v6.0.0

### 2. **MAMBA-SSM** 📦 IMPORTABLE (PyPI Official) ⭐⭐⭐⭐⭐
**STATUS**: On PyPI + GitHub
**INSTALL**:
```bash
# Install official package (requires CUDA)
uv sync -E gpu  # Adds mamba-ssm with CUDA kernels

# Clone for studying patterns
git clone https://github.com/state-spaces/mamba.git reference_repos/mamba
```
**USE IN CODE**:
```python
from mamba_ssm import Mamba2
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
```
**STUDY FILES**:
- `mamba_ssm/modules/mamba2.py` - Core Mamba-2 layer
- `mamba_ssm/models/mixer_seq_simple.py` - Full model architecture
- Hardware-aware SSD algorithm

### 3. **SEIZURETRANSFORMER** 📚 STUDY-ONLY ⭐⭐⭐⭐⭐
**STATUS**: Research code (not packaged)
**CLONE**:
```bash
# Already cloned in reference_repos/SeizureTransformer
```
**WHAT TO COPY** (write your own version):
- `architecture.py:120-212` - U-Net encoder/decoder patterns
- `architecture.py:215-260` - ResCNN block structure
- Skip connection wiring
- Channel progression [64, 128, 256, 512]

### 4. **VIM (Vision Mamba)** 📚 STUDY-ONLY ⭐⭐⭐⭐
**STATUS**: Research code (not packaged)
**CLONE**:
```bash
git clone https://github.com/hustvl/Vim.git reference_repos/vision-mamba
```
**WHAT TO COPY** (adapt patterns):
- Bidirectional SSM wiring
- Forward/backward state concatenation
- Position embedding for sequences
- Middle token strategies

### 5. **BRAINDECODE** 📦 IMPORTABLE (PyPI Official) ⭐⭐⭐⭐
**STATUS**: On PyPI + GitHub
**INSTALL**:
```bash
# Install official package
uv add braindecode

# Clone for deeper study
git clone https://github.com/braindecode/braindecode.git reference_repos/braindecode
```
**USE IN CODE**:
```python
from braindecode.models import EEGNetv4, Deep4Net, ShallowFBCSPNet
from braindecode.augmentation import TimeReverse, SignFlip
```
**ALSO STUDY**:
- `models/eegnet.py` - Temporal/spatial convolution patterns
- `models/deep4.py` - Deep ConvNet architecture
- EEG-specific preprocessing

## 📦 ADDITIONAL IMPORTABLE PACKAGES

### Standard Scientific Stack (PyPI)
```bash
# All available via pip/uv add
torch           # Deep learning framework
mne             # EEG I/O and preprocessing
scipy           # Signal processing, morphology
scikit-image    # Hysteresis thresholding
numpy           # Numerical arrays
pandas          # CSV_BI output formatting
```

### Key Functions We'll Use
```python
# Hysteresis (from scikit-image)
from skimage.filters import apply_hysteresis_threshold

# Morphology (from scipy)
from scipy.ndimage import binary_opening, binary_closing

# EEG (from MNE)
import mne
raw = mne.io.read_raw_edf(...)
```

## ❌ REPOS TO SKIP
- **seizy** - We have scipy/skimage for hysteresis
- **Standalone EEGNet** - Braindecode has better version
- **Random seizure detection repos** - We have SeizureTransformer patterns

## 🏗️ COMPONENT IMPLEMENTATION STRATEGY

| Our Component | Strategy | Source |
|---------------|----------|--------|
| **U-Net Encoder** | 📚 COPY PATTERN | SeizureTransformer `architecture.py:120-170` |
| **U-Net Decoder** | 📚 COPY PATTERN | SeizureTransformer `architecture.py:171-212` |
| **ResCNN Stack** | 📚 COPY PATTERN | SeizureTransformer `architecture.py:215-260` |
| **Mamba-2 Core** | 📦 IMPORT | `from mamba_ssm import Mamba2` |
| **Bidirectional** | 📚 ADAPT PATTERN | Vim forward/backward wiring |
| **TAES Scoring** | 📦 IMPORT | `from nedc_bench.algorithms import taes` |
| **CSV_BI Format** | 📦 IMPORT | `from nedc_bench.models import annotations` |
| **EEG Models** | 📦 IMPORT | `from braindecode.models import EEGNetv4` |
| **Hysteresis** | 📦 IMPORT | `from skimage.filters import apply_hysteresis_threshold` |

## 🚀 SETUP COMMANDS

### Step 1: Install Importable Packages
```bash
# From project root
cd /path/to/brain-go-brr-v2

# Core packages
uv add torch mne scipy scikit-image numpy pandas

# Braindecode (EEG models)
uv add braindecode

# Mamba-SSM (requires CUDA)
uv sync -E gpu

# NEDC-Bench (from GitHub)
uv add "nedc-bench @ git+https://github.com/Clarity-Digital-Twin/nedc-bench.git"
```

### Step 2: Clone Study-Only Repos
```bash
# Already done, but for reference:
mkdir -p reference_repos
cd reference_repos
git clone https://github.com/hustvl/Vim.git vision-mamba
# SeizureTransformer already exists
cd ..
```

## 📝 DEVELOPMENT WORKFLOW

### When Building Components:
1. **Check Strategy Table** above
2. **If 📦 IMPORT**: Just `import` and use
3. **If 📚 STUDY**: Read code, understand pattern, write your own

### Example: Building Bi-Mamba-2
```python
# IMPORT the core Mamba-2 layer
from mamba_ssm import Mamba2

# STUDY Vim patterns for bidirectional wiring
# Then WRITE your own BiMamba2 class

class BiMamba2(nn.Module):
    def __init__(self, d_model=512):
        super().__init__()
        self.forward_mamba = Mamba2(d_model)  # Import!
        self.backward_mamba = Mamba2(d_model)  # Import!
        # Your bidirectional logic here (studied from Vim)
```

### During Training:
```python
# IMPORT scoring directly
from nedc_bench.algorithms import taes
metrics = taes.score(ref_csv, hyp_csv)
```

---
**Mission: Build O(N) seizure detection with these battle-tested components** 🚀