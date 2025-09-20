# Modal Cloud Deployment - Single Source of Truth

**Status: ✅ WORKING (smoke test verified)**
**Last updated: 2025-09-20**
**Location: `/home/jj/proj/brain-go-brr-v2/docs/deployment/MODAL_DEPLOYMENT_SSOT.md`**

## 🎯 Current Working State

### What's Working
- ✅ Modal deployment with A100-80GB GPU
- ✅ Mamba-SSM compilation with CUDA kernels
- ✅ S3 bucket mount for TUSZ data (79GB)
- ✅ Smoke test training runs successfully
- ✅ PyTorch 2.2.2 + CUDA 12.1
- ✅ Loss decreases properly (0.91 → 0.09 in smoke test)

### What's Pending
- ⚠️ Logging visibility (needs flush=True on prints)
- ⚠️ W&B integration for real-time metrics
- ⚠️ Full TUSZ training run (24+ hours)

## ⚡ Quick Commands

```bash
# Smoke test (2 files, 1 epoch) - VERIFIED WORKING
modal run --detach deploy/modal/app.py -- --action train --config configs/smoke_test.yaml

# Full A100 training (100 epochs, ~24 hours)
modal run --detach deploy/modal/app.py -- --action train --config configs/tusz_train_a100.yaml

# Evaluate checkpoint
modal run deploy/modal/app.py -- --action evaluate --checkpoint /results/checkpoints/best.pt
```

**CRITICAL**: Always use `--detach` flag to prevent disconnection from killing training!

## 📦 Required Files

### 1. Modal App (`/home/jj/proj/brain-go-brr-v2/deploy/modal/app.py`)
- ✅ Working configuration with all fixes applied
- ✅ CUDA devel image for mamba-ssm compilation
- ✅ S3 bucket mount configured
- ✅ Volumes for results persistence

### 2. Configs
- `configs/smoke_test.yaml` - Quick test (2 files, 1 epoch)
- `configs/tusz_train_a100.yaml` - Full training optimized for A100
- `configs/tusz_train_wsl2.yaml` - Local WSL2 training (reference)

## 🚀 First-Time Setup

### Step 1: Install Modal CLI
```bash
pip install --upgrade modal
modal --version  # Should be 1.1.4+
modal setup      # Login via browser
```

### Step 2: Configure Modal Settings
1. Go to https://modal.com/settings/image-config
2. Select **"2025.06"** (NOT legacy!)
3. Click Save
4. **Why**: Legacy version causes Mamba-SSM compilation failures

### Step 3: Setup AWS S3 (for TUSZ data)
```bash
# Create S3 secret in Modal
modal secret create aws-s3-secret \
  --env AWS_ACCESS_KEY_ID=your_key \
  --env AWS_SECRET_ACCESS_KEY=your_secret \
  --env AWS_DEFAULT_REGION=us-east-1

# Verify secret exists
modal secret list | grep aws-s3-secret
```

### Step 4: (Optional) Setup W&B
```bash
# Create W&B secret for metrics tracking
modal secret create wandb-secret \
  --env WANDB_API_KEY=your_wandb_api_key
```

## 🔧 Key Configuration Details

### Image Build (Working Recipe)
```python
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("build-essential", "ninja-build")  # CRITICAL for compilation
    .pip_install("torch==2.2.2", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("packaging")  # CRITICAL: before mamba-ssm
    .run_commands("export CC=gcc CXX=g++ && pip install -v --no-build-isolation 'mamba-ssm>=2.0.0'")
)
```

### GPU & Resources
- **GPU**: A100-80GB (3x faster than 4090)
- **Memory**: 32GB RAM
- **CPUs**: 8 cores
- **Timeout**: 86400s (24 hours max - Modal limit)

### Data Mounts
- **S3 Bucket**: `brain-go-brr-eeg-data-20250919` (79GB TUSZ)
- **Mount Path**: `/data/tusz/`
- **Results Volume**: `/results/` (persistent across runs)

## 📊 Training Configs Comparison

| Config | Files | Epochs | Batch Size | Workers | Purpose |
|--------|-------|--------|------------|---------|---------|
| smoke_test.yaml | 2 | 1 | 4 | 0 | Quick test (~5 min) |
| tusz_train_a100.yaml | ALL | 100 | 32 | 4 | Full training (~24h) |
| tusz_train_wsl2.yaml | ALL | 100 | 8 | 0 | Local reference |

## ⚠️ Known Issues & Solutions

### Issue 1: Mamba-SSM Symbol Linking Error
**Error**: `undefined symbol: _ZN3c104cuda14ExchangeDeviceEa`
**Solution**: Use `--no-build-isolation` with proper build deps (implemented in app.py)

### Issue 2: PyTorch Scheduler Warning
**Warning**: `UserWarning: Detected call of lr_scheduler.step() before optimizer.step()`
**Solution**: False positive - scheduler called correctly at epoch level (documented)

### Issue 3: No Real-Time Logs
**Problem**: Training output buffered for hours
**Solution**: Add `flush=True` to prints (see MODAL_LOGGING_TODO.md)

## 💰 Cost Estimates

- **A100-80GB**: ~$3.34/hour
- **Smoke test**: ~$0.30 (5 minutes)
- **Full training**: ~$80-140 (24-42 hours)
- **Modal credits**: Check with `modal status`

## 🔍 Monitoring Training

### Check Running Jobs
```bash
# List all running functions
modal app list

# Stream logs from specific run
modal app logs brain-go-brr-v2
```

### Download Results
```bash
# List files in results volume
modal volume ls brain-go-brr-results

# Download checkpoint
modal volume get brain-go-brr-results checkpoints/best.pt ./best.pt
```

## 📝 Next Steps

1. **Immediate**: Implement logging fixes (flush=True)
2. **Before Full Run**: Setup W&B for metrics tracking
3. **Testing**: Run smoke test to verify setup
4. **Production**: Launch full TUSZ training with monitoring

## 🗂️ Related Documentation

- **Logging improvements**: `/docs/deployment/MODAL_LOGGING_TODO.md`
- **Mamba issues (resolved)**: `/docs/deployment/MODAL_MAMBA_DEPLOYMENT_ISSUES.md`
- **Local training**: See WSL2 configs in `/configs/`

---

**Status**: This deployment configuration is VERIFIED WORKING as of 2025-09-20.
Modal smoke test successfully completed with proper loss convergence.