# Critical Version Pins for Brain-Go-Brr v2

## Why Version Pinning Matters
This project uses GPU-accelerated CUDA extensions that require EXACT version alignment.

## Required Versions (DO NOT CHANGE)

### System Requirements
- **CUDA Toolkit**: 12.1 (NOT 12.0, 12.2, 12.6, etc.)
- **Python**: 3.11.x (tested with 3.11.13)

### Core Dependencies
```python
# pyproject.toml should have:
torch==2.2.2+cu121        # NOT >=2.2.2, EXACT version
numpy==1.26.4             # NOT >=1.24, numpy 2.x breaks mamba
scipy==1.11.4             # Tested version
scikit-learn==1.3.2       # Tested version
```

### GPU Extensions (Critical)
```python
mamba-ssm==2.2.4          # NOT 2.2.5 (has undefined symbol bug)
causal-conv1d==1.5.2      # Must match mamba-ssm
```

### EEG Processing
```python
mne==1.7.1                # Tested with TUSZ
pyedflib==0.1.38          # EDF file reading
```

## Version Compatibility Matrix

| Component      | Version  | CUDA | Notes                           |
|---------------|----------|------|----------------------------------|
| PyTorch       | 2.2.2    | 12.1 | Must use +cu121 wheel           |
| CUDA Toolkit  | 12.1     | -    | For building extensions          |
| mamba-ssm     | 2.2.4    | 12.1 | 2.2.5 has ABI issues            |
| causal-conv1d | 1.5.2    | 12.1 | Dependency of mamba-ssm          |
| numpy         | 1.26.4   | -    | Last 1.x version (2.x breaks)   |
| Python        | 3.11.13  | -    | 3.12 works but less tested      |

## Installation Commands

### For Exact Reproducibility
```bash
# Install exact PyTorch version
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# Install exact mamba-ssm (after CUDA 12.1 toolkit)
export CUDA_HOME=/usr/local/cuda-12.1
pip install --no-build-isolation causal-conv1d==1.5.2
pip install --no-build-isolation mamba-ssm==2.2.4
```

## What Breaks With Wrong Versions

### PyTorch != 2.2.2
- mamba-ssm may not compile
- CUDA kernels may fail

### CUDA Toolkit != 12.1
- `undefined symbol: _ZN3c104cuda14ExchangeDeviceEa`
- Falls back to Conv1d (not equivalent!)

### numpy >= 2.0
- mamba-ssm import fails
- Selective scan operations break

### mamba-ssm == 2.2.5
- Import error with undefined symbols
- Known regression in 2.2.5

## Verification Script
```python
#!/usr/bin/env python
"""Verify all versions are correct"""

import sys
import torch
import numpy as np

print("Checking versions...")

# Check Python
assert sys.version_info[:2] == (3, 11), f"Need Python 3.11, got {sys.version}"

# Check PyTorch
assert torch.__version__.startswith("2.2.2"), f"Need PyTorch 2.2.2, got {torch.__version__}"
assert "+cu121" in torch.__version__, "Need CUDA 12.1 build of PyTorch"

# Check numpy
assert np.__version__.startswith("1.26"), f"Need numpy 1.26.x, got {np.__version__}"

# Check CUDA
assert torch.cuda.is_available(), "CUDA not available"
assert torch.version.cuda == "12.1", f"Need CUDA 12.1, got {torch.version.cuda}"

# Check mamba
try:
    from mamba_ssm import Mamba2
    print("✅ All versions correct!")
except ImportError as e:
    print(f"❌ mamba-ssm not working: {e}")
    sys.exit(1)
```

## For CI/CD

### GitHub Actions
```yaml
- name: Setup CUDA
  uses: Jimver/cuda-toolkit@v0.2.11
  with:
    cuda: '12.1.0'

- name: Install exact deps
  run: |
    pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
    pip install numpy==1.26.4
```

### Docker
```dockerfile
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
# ... rest of Dockerfile
```

## DO NOT UPGRADE WITHOUT TESTING
Any version changes require:
1. Full test suite pass
2. Verify mamba-ssm loads (no fallback)
3. Training smoke test
4. Check CUDA kernel dispatch

## Emergency Rollback
If upgrades break:
```bash
git checkout main -- pyproject.toml uv.lock
uv sync --reinstall
```