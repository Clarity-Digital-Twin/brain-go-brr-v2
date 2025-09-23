# Brain-Go-Brr v2.6 Configuration Files

## 🧠 Architecture: TCN + BiMamba + GNN + LPE

All configs use the v2.6 stack with:
- **TCN**: Multi-scale temporal feature extraction (8 layers)
- **BiMamba**: Bidirectional state-space model (O(N) complexity, 6 layers)
- **GNN**: Graph neural network with SSGConv (α=0.05, 2 layers)
- **LPE**: Laplacian positional encoding (k=16 eigenvectors)
- **Total Parameters**: ~31M

## 📁 Directory Structure

```
configs/
├── local/                    # Local WSL2/Linux configs (RTX 4090 optimized)
│   ├── smoke.yaml           # Quick test (1 epoch, 3 files via BGB_SMOKE_TEST)
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
- **Location**: `cache/tusz/train/` (3734 NPZ) + `cache/tusz/val/` (933 NPZ)
- **Warning**: Do NOT use `cache/v2.6_full/` - it's empty!

### Modal (A100)
```yaml
data:
  cache_dir: /results/cache/tusz  # Persistent SSD volume
```
- **Location**: `/results/cache/tusz/train/` + `/results/cache/tusz/val/`
- **Built once**: First run builds cache, all subsequent runs reuse
- **NOT on S3**: Cache is on fast Modal SSD, never touches S3 after build

## 🚀 Usage Examples

### Local Training (RTX 4090)
```bash
# Smoke test (uses BGB_SMOKE_TEST=1 internally)
python -m src train configs/local/smoke.yaml

# Full training (watch in tmux recommended)
tmux new -s train
python -m src train configs/local/train.yaml
```

### Modal Cloud Training (A100)
```bash
# Test Mamba CUDA first
modal run deploy/modal/app.py --action test-mamba

# Smoke test
modal run deploy/modal/app.py --action train --config configs/modal/smoke.yaml

# Full training (detached)
modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml

# Monitor
modal app list
modal app logs <app-id>
```

## 🔑 Key Configuration Differences

| Setting | Local (RTX 4090) | Modal (A100-80GB) | Why Different |
|---------|------------------|-------------------|---------------|
| **Batch Size** | 12 | 64 | A100 has 3.3x more VRAM |
| **Mixed Precision** | false | true | RTX 4090 FP16 causes NaNs |
| **Learning Rate** | 1.5e-4 | 3e-4 | Conservative for stability |
| **Workers** | 4 | 8 | A100 handles more parallel IO |
| **Cache Location** | `cache/tusz/` | `/results/cache/tusz/` | Different filesystems |

## ⚠️ Common Pitfalls

1. **Wrong Cache Directory**:
   - ❌ Local: `cache/v2.6_full/` (empty)
   - ✅ Local: `cache/tusz/` (has 3734 files)

2. **Modal Cache Misconception**:
   - ❌ "Cache is on S3 causing slowdowns"
   - ✅ Cache is on Modal SSD from first run

3. **Mixed Precision on RTX 4090**:
   - ❌ `mixed_precision: true` causes NaN losses
   - ✅ Keep `mixed_precision: false` for stability

4. **PyG Not Installed**:
   - Run `uv sync -E graph` locally
   - Modal image includes PyG automatically

## 🏗️ Model Configuration (All Configs)

```yaml
model:
  architecture: tcn

  tcn:
    num_layers: 8
    channels: [64, 128, 256, 512]
    kernel_size: 7
    stride_down: 16

  mamba:
    n_layers: 6
    d_model: 512
    d_state: 16
    conv_kernel: 4  # CUDA constraint

  graph:
    enabled: true
    use_pyg: true  # CRITICAL for Laplacian PE
    alpha: 0.05    # SSGConv mixing (EvoBrain proven)
    k_eigenvectors: 16  # Laplacian PE dimension
    top_k: 3       # Sparse connectivity
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
| `BGB_SMOKE_TEST=1` | Limit to 3 files | Smoke tests |
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
2. **Balanced Sampling**: Ensures seizures in every batch
3. **Cosine Schedule**: Smooth learning rate decay
4. **Early Stopping**: Patience=5 on sensitivity@10FA/24h

## 🚨 Critical Notes

- **v2.6 Stack**: All configs use TCN+BiMamba+GNN+LPE
- **Cache Reuse**: Both platforms reuse existing preprocessed cache
- **No Edge Stream Yet**: Using heuristic graphs (cosine similarity)
- **Future v3.0**: Will add edge Mamba stream for learned adjacency