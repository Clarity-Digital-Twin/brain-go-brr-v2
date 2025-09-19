# GPU Training on Windows + WSL2 (Reliable Setup)

This guide fixes PyTorch/Triton import hangs under WSL2 and gets your 4090 training. The issue isn’t the model — it’s running CUDA wheels and heavy .so imports from `/mnt/c` (Windows filesystem). The fix: run from the WSL ext4 filesystem and install the correct CUDA wheel.

Use this decision guide, then follow the steps for your path.

## TL;DR

- Do NOT run Python from `/mnt/c/...` in WSL — move the repo to `~/...` (ext4).
- Install a CUDA-enabled PyTorch wheel that matches your driver (cu121 or cu124).
- If you don’t need Linux-only GPU deps, native Windows Python is simplest.
- When in doubt, you can always develop in a CPU-only WSL env and train later on GPU.

## Choose Your Path

- Native Windows (GPU, simplest): Works great for PyTorch CUDA wheels. Some research libs (e.g., `mamba-ssm` CUDA kernels) are Linux-only; they’ll fall back to CPU on Windows.
- WSL2 with GPU (GPU + Linux-only deps): Best if you need Linux-only CUDA deps (e.g., `mamba-ssm`). Requires running from ext4 and ensuring WSL CUDA works.
- WSL2 CPU-only (dev only): If you just need to develop/run tests without GPU, install the CPU-only torch wheel.

---

## Option B (Recommended here): WSL2 with GPU

Works with Linux-only CUDA deps (`mamba-ssm`) and uses your Windows GPU via WSL.

### 0) One-time checks

- Update NVIDIA Windows driver (WSL CUDA support).
- In WSL terminal, run: `nvidia-smi` (should report your GPU and a CUDA version, e.g., 12.x)
- Ensure you run from your Linux home, NOT `/mnt/c`.

### 1) Move repo to ext4 (safe & fast)

```bash
mkdir -p ~/proj
rsync -a --delete \
  --exclude 'data/' --exclude 'results/' --exclude '.venv/' \
  "/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brr-v2/" ~/proj/brain-go-brr-v2
cd ~/proj/brain-go-brr-v2
```

If your dataset lives on Windows, you can symlink it (data is OK on `/mnt/c`; just avoid running Python packages from there):

```bash
ln -s /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brr-v2/data data
```

### 2) Create a fresh venv

```bash
uv venv .venv
source .venv/bin/activate
python -m ensurepip --upgrade
python -m pip install --upgrade pip wheel setuptools
```

### 3) Install CUDA-enabled PyTorch

Pick the wheel index that matches your CUDA toolchain shown by `nvidia-smi`:

- CUDA 12.1 → `cu121`
- CUDA 12.4 or newer → `cu124`

```bash
# Example for CUDA 12.4+
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio

# Project + GPU extra (installs mamba-ssm, etc.)
pip install -e .[gpu] tensorboard
```

### 4) Verify Torch sees the GPU

```bash
python - <<'PY'
import torch
print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'is_available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))
PY
```

If `is_available` is False:

- Ensure you’re running from `~/proj/...` (ext4), not `/mnt/c/...`.
- Check `nvidia-smi` in WSL works.
- Export loader path once (if needed): `export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH`
- Confirm `/dev/nvidia*` exists: `ls -l /dev/nvidia*`

### 5) Run the pipeline

```bash
# Full GPU path (Mamba-SSM kernels if available)
python -m src.experiment.pipeline --config configs/local.yaml

# If you need to force CPU fallback for Mamba (debug/dev)
SEIZURE_MAMBA_FORCE_FALLBACK=1 python -m src.experiment.pipeline --config configs/local.yaml
```

---

## Option A: Native Windows (GPU, simplest)

Use Windows Python when you don’t need Linux-only CUDA deps.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip

# Choose the CUDA wheel that matches your driver (cu121 or cu124)
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio

# Install the project
pip install -e .

# Verify
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Notes:

- `mamba-ssm` GPU kernels are Linux-only. Our code auto-falls back to CPU for Mamba on Windows.

---

## Option C: WSL2 CPU-only (dev)

Unblocks imports/tests when you can’t or don’t want GPU in WSL yet.

```bash
uv venv .venv && source .venv/bin/activate
python -m ensurepip --upgrade && pip install --upgrade pip wheel setuptools

pip uninstall -y torch torchvision torchaudio 'nvidia-*'
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0+cpu
pip install -e . tensorboard

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"  # expect cpu, None, False
```

You can develop/run tests on CPU, then train on GPU later via Option A or B.

---

## Troubleshooting

- Torch import “hangs” in WSL:
  - You’re likely running from `/mnt/c`. Move repo + venv to `~/...` (ext4).
- `torch.cuda.is_available() == False` with CUDA wheel:
  - Run `nvidia-smi` in WSL (must work).
  - Ensure `/usr/lib/wsl/lib` is on loader path: `export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH`.
  - Confirm `/dev/nvidia*` exists.
  - Avoid running from `/mnt/c`.
- Triton error: `RuntimeError: 0 active drivers ([])`:
  - GPU driver not visible to Python in WSL. Fix `nvidia-smi`/driver/LD paths as above.
  - As a temporary bypass, set `SEIZURE_MAMBA_FORCE_FALLBACK=1`.
- SciPy/NumPy first import slow or “hung”:
  - Don’t use a strict `timeout` on first imports; they can take a few seconds.
  - Run from ext4; DrvFS (`/mnt/c`) is notoriously slow for large .so mmaps.
- Windows path works but Mamba-SSM is slow:
  - Expected — Windows lacks the Linux CUDA kernels; the code falls back to CPU for Mamba.

---

## Quick Cheat Sheet

```bash
# Move to ext4 and set up CUDA Torch (WSL)
mkdir -p ~/proj && rsync -a --delete --exclude 'data/' --exclude 'results/' --exclude '.venv/' \
  "/mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brr-v2/" ~/proj/brain-go-brr-v2
cd ~/proj/brain-go-brr-v2 && uv venv .venv && source .venv/bin/activate
python -m ensurepip --upgrade && pip install --upgrade pip wheel setuptools
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
pip install -e .[gpu] tensorboard
ln -s /mnt/c/Users/JJ/Desktop/Clarity-Digital-Twin/brain-go-brr-v2/data data
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m src.experiment.pipeline --config configs/local.yaml
```

---

## Why this works

WSL2’s `/mnt/c` mount (DrvFS) performs poorly when Python mmaps large native libraries (CUDA, SciPy, OpenBLAS). Importing those wheels from `/mnt/c` can stall or appear hung. Running from the WSL ext4 filesystem and installing a matching CUDA wheel fixes the problem. If GPU still isn’t visible, `nvidia-smi`/loader path fixes make it available to Torch and Triton.

