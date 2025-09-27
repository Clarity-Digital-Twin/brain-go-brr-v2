# Brain-Go-Brr V3 Configuration Files

## 🧠 Architecture: V3 Dual-Stream (TCN + BiMamba + GNN)

All configs use the V3 dual-stream architecture:
- **TCN**: Multi-scale temporal feature extraction (8 layers, stride=16)
- **Node Stream**: Per-electrode BiMamba (d_model=64, 6 layers, headdim=8)
- **Edge Stream**: Per-edge BiMamba (d_model=16, 2 layers, headdim=4)
- **GNN**: Vectorized SSGConv with **DYNAMIC** Laplacian PE (α=0.05, k=16)
  - **Dynamic PE**: Recomputed per timestep from evolving adjacency (EvoBrain approach)
  - **Vectorized**: All 960 timesteps computed in parallel (100-1000x faster than loops)
  - **Numerical Stability**: FP32 eigendecomposition with sign consistency
- **Total Parameters**: ~31.5M

## 📁 Directory Structure

```
configs/
├── local/                    # Local WSL2/Linux configs (RTX 4090 optimized)
│   ├── smoke.yaml           # Quick test (1 epoch, 3 files via BGB_LIMIT_FILES=3)
│   └── train.yaml           # Full training (100 epochs, 3734 files)
│
└── modal/                    # Modal cloud GPU configs (A100-80GB optimized)
    ├── smoke.yaml           # Quick cloud test (1 epoch, 50 files)
    └── train.yaml           # Full cloud training (100 epochs, 3734 files)
```

## ⚡ Critical Cache Configuration

### Local (RTX 4090)
```yaml
data:
  cache_dir: cache/tusz     # MUST use existing cache with 3734 files!
```
- **Location**: `cache/tusz/train/` (4667 NPZ) + `cache/tusz/dev/` (1832 NPZ)
- **Warning**: Do NOT use `cache/v2.6_full/` - it's empty!

### Modal (A100)
```yaml
data:
  cache_dir: /results/cache/tusz  # Persistent SSD volume
```
- **Location**: `/results/cache/tusz/train/` + `/results/cache/tusz/dev/`
- **Built once**: First run builds cache, all subsequent runs reuse
- **NOT on S3**: Cache is on fast Modal SSD, never touches S3 after build

## 🚀 Usage Examples

### Local Training (RTX 4090)
```bash
# Smoke test (requires environment variables)
BGB_LIMIT_FILES=3 BGB_SMOKE_TEST=1 python -m src train configs/local/smoke.yaml
# Or use the helper script:
./run_smoke_test.sh

# Full training (watch in tmux recommended)
tmux new -s train
python -m src train configs/local/train.yaml
```

### Modal Cloud Training (A100)
```bash
# Test Mamba CUDA first
modal run deploy/modal/app.py --action test-mamba

# Smoke test (app.py sets BGB_LIMIT_FILES=50 automatically)
modal run --detach deploy/modal/app.py --action train --config configs/modal/smoke.yaml

# Full training (detached)
modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml

# Monitor
modal app list
modal app logs <app-id>
```

## 🔑 Key Configuration Differences

| Setting | Local (RTX 4090) | Modal (A100-80GB) | Why Different |
|---------|------------------|-------------------|---------------|
| **Batch Size** | 8 | 48 | V3 dual-stream uses more memory |
| **Mixed Precision** | false | true | RTX 4090 FP16 causes NaNs |
| **Learning Rate** | 5e-5 | 5e-5 | Reduced for V3 stability |
| **Workers** | 0 | 8 | WSL2 multiprocessing issues |
| **Cache Location** | `cache/tusz/` | `/results/cache/tusz/` | Different filesystems |

## ⚠️ Common Pitfalls

1. **Wrong Cache Directory**:
   - ❌ Local: `cache/v2.6_full/` (empty)
   - ✅ Local: `cache/tusz/{train,dev}/` (4667 + 1832 files) - Using TUSZ's 'dev' naming!

2. **Modal Cache Misconception**:
   - ❌ "Cache is on S3 causing slowdowns"
   - ✅ Cache is on Modal SSD from first run

3. **Mixed Precision on RTX 4090**:
   - ❌ `mixed_precision: true` causes NaN losses
   - ✅ Keep `mixed_precision: false` for stability

4. **PyG Not Installed**:
   - Run `make setup-gpu` locally (installs PyG from prebuilt wheels for torch 2.2.2+cu121)
   - Modal image includes PyG automatically

## 🏗️ Model Configuration (All Configs)

```yaml
model:
  architecture: v3  # V3 dual-stream architecture

  tcn:
    num_layers: 8
    kernel_size: 7
    stride_down: 16

  mamba:
    n_layers: 6
    d_model: 512
    d_state: 16
    conv_kernel: 4  # CUDA constraint
    # Node/Edge streams use different params (see detector.py)

  graph:
    enabled: true
    # PyG is required; no separate toggle needed
    alpha: 0.05    # SSGConv mixing parameter
    k_eigenvectors: 16  # Static Laplacian PE

    # V3-specific edge stream config:
    edge_mamba_layers: 2
    edge_mamba_d_state: 8
    edge_mamba_d_model: 16  # Must be multiple of 8
```

## 📊 Expected Training Times

| Config | Platform | Time/Epoch | Total Time | Cost |
|--------|----------|------------|------------|------|
| Local Train | RTX 4090 | ~2-3 hours | ~200-300 hours | Electricity |
| Modal Train | A100-80GB | ~1 hour | ~100 hours | ~$319 |
| Smoke Test | Both | ~5 mins | 5 mins | Minimal |

## 🔧 Environment Variables

| Variable | Purpose | When to Use |
|----------|---------|-------------|
| `BGB_SMOKE_TEST=1` | Skip seizure sampling | Local smoke tests |
| `BGB_LIMIT_FILES=3` | Limit to 3 files | Local smoke (required!) |
| `BGB_LIMIT_FILES=N` | Limit to N files | Testing |
| `BGB_DISABLE_TQDM=1` | Disable progress bars | Modal (automatic) |
| `BGB_NAN_DEBUG=1` | Debug NaN losses | If training fails |
| `BGB_FORCE_MANIFEST_REBUILD=1` | Rebuild cache manifest | If cache corrupted |

## 📈 Post-Processing (All Configs)

```yaml
postprocessing:
  hysteresis:
    tau_on: 0.86   # Seizure onset threshold
    tau_off: 0.78  # Seizure offset threshold
  morphology:
    opening_kernel: 11
    closing_kernel: 31
  duration:
    min_duration_s: 3.0
    max_duration_s: 600.0
```

## 🎯 Training Strategy

1. **Focal Loss**: Essential for 12:1 class imbalance
2. **Balanced Sampling**: Ensures seizures in every batch (train.yaml only)
3. **Cosine Schedule**: Smooth learning rate decay
4. **Early Stopping**: Patience=5 on sensitivity@10FA/24h

## 🚨 Critical Notes

- **V3 Architecture**: Dual-stream with learned edge dynamics
- **BiMamba2 headdim**: Node=8, Edge=4 (prevents CUDA fallback)
- **Cache Reuse**: Both platforms reuse existing preprocessed cache
- **Smoke tests**: Local needs manual env vars, Modal sets automatically
- **Full state-space modeling**: No Conv1d fallbacks with proper headdim
