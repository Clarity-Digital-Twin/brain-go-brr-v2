# Deployment Architecture & CUDA Implementation

## Overview
This document consolidates all deployment-specific architecture decisions and CUDA kernel implementation details discovered during Modal deployment.

## Critical Version Requirements

### PyTorch & CUDA Stack
**MUST use exact versions - NO flexibility**:
- `torch==2.2.2+cu121` (NOT 2.8.0 from Modal's mirror!)
- `mamba-ssm==2.2.2` (NOT 2.2.4/2.2.5 which have bugs)
- `causal-conv1d==1.4.0` (1.5+ requires PyTorch 2.4+)
- `numpy<2.0` (2.x breaks mamba-ssm compilation)

### Why These Specific Versions Matter
1. **PyTorch 2.2.2**: Mamba CUDA kernels compiled against specific PyTorch C++ API
2. **Modal's Mirror Issue**: Default PyPI mirror serves 2.8.0, causing kernel failures
3. **causal-conv1d 1.4.0**: Provides CUDA kernels for efficient convolution (104MB wheel)
4. **mamba-ssm 2.2.2**: Compiled extension (323MB wheel) with SSM CUDA kernels

## Mamba CUDA Kernel Architecture

### d_conv Parameter Coercion
```python
# Configured value
d_conv = 5  # ~19.5ms temporal window at 256Hz

# CUDA kernel constraint
CUDA_SUPPORTED = {2, 3, 4}  # Hardware optimization limitation

# Runtime coercion
actual_d_conv = 4 if using_cuda else 5
```

### Why We Use d_conv=4 (But Get 4)
1. **Original Design**: 5 matches middle ResCNN kernel for multi-scale consistency
2. **CUDA Reality**: Kernels only support {2,3,4}, auto-coerces to 4
3. **Impact**: 15.6ms vs 19.5ms window - negligible difference
4. **Lesson**: Should have just used 4 everywhere

### Kernel Compilation Process
```bash
# Build phase (NO CUDA)
- Install build deps
- Set CUDA_HOME=/usr/local/cuda-12.1
- Compile with --no-build-isolation
- Produces .whl files with .so libraries

# Runtime phase (CUDA AVAILABLE)
- Load compiled kernels
- Dispatch to CUDA implementation
- Falls back to Conv1d if kernels fail
```

## Modal Deployment Architecture

### Container Structure
```
Modal Image Build:
├── System packages (CUDA toolkit)
├── Python environment
├── Compiled CUDA extensions (.so files)
├── Project code (.add_local_dir)
└── Config files (configs/modal/)

Runtime Mounts:
├── /data (S3 EDF files, read-only)
├── /results (persistent volume)
└── /cache (derived from /results)
```

### Critical Build Steps
1. **Force correct PyTorch**: Override Modal's mirror
2. **Set CUDA environment**: Before ANY pip installs
3. **Compile without isolation**: Access installed PyTorch
4. **Verify kernels work**: Test forward/backward pass

### File Organization
```
deploy/modal/
├── app.py          # Modal entrypoint
└── requirements.txt # Pinned versions

configs/modal/
├── smoke_a100.yaml  # 1 epoch test
└── train_a100.yaml  # Full 100 epochs
```

## Performance Characteristics

### Model Complexity
- **Parameters**: ~13.4M (not 25M as initially estimated)
- **Memory**: ~4GB for batch_size=64 on A100
- **Throughput**: ~0.5s per batch on A100-80GB

### A100 Optimizations
```yaml
# configs/modal/train.yaml
batch_size: 64        # Larger than local (16)
num_workers: 4        # Parallel data loading
pin_memory: true      # Faster GPU transfer
mixed_precision: true # FP16 training
```

### Balanced Dataset Performance
- **Manifest-driven**: Pre-computed window classifications
- **Fixed ratios**: ALL partial + 0.3× full + 2.5× none
- **Result**: ~34% seizure ratio in training batches
- **No sampler needed**: Dataset handles balancing internally

## Deployment Commands

### Test Mamba CUDA
```bash
modal run deploy/modal/app.py --action test-mamba
```

### Smoke Test (1 epoch)
```bash
modal run --detach deploy/modal/app.py \
  --action train \
  --config configs/modal/smoke.yaml
```

### Full Training (100 epochs)
```bash
modal run --detach deploy/modal/app.py \
  --action train \
  --config configs/modal/train.yaml
```

## Troubleshooting Decision Tree

### "'NoneType' object is not callable"
1. Check PyTorch version: `torch.__version__` must be 2.2.2
2. Verify CUDA compilation: Check build logs for .so files
3. Test kernels: Run test-mamba action
4. Force fallback: Set `SEIZURE_MAMBA_FORCE_FALLBACK=1`

### "No partial seizure windows found"
1. Check manifest exists: `/results/cache/.../manifest.json`
2. Force rebuild: Set `BGB_FORCE_MANIFEST_REBUILD=1`
3. Verify CSV annotations: Must have seizure events
4. Check cache files: `.npz` files must exist

### Slow Training Performance
1. Verify GPU allocation: Should see A100-80GB
2. Check batch composition: ~30-40% seizure ratio expected
3. Monitor GPU utilization: Should be >80%
4. Ensure pin_memory=true and num_workers>0

## Architecture Evolution

### What Changed from Original Design
1. **Parameter Count**: 25M → 13.4M (more efficient)
2. **Activation**: ELU → ReLU in ConvBlocks
3. **d_conv**: Effectively 4 on GPU (not 5)
4. **Sampling**: Manifest-driven balanced dataset

### What Stayed the Same
1. **Core Architecture**: U-Net + ResCNN + Bi-Mamba
2. **Window Parameters**: 60s @ 256Hz with 10s stride
3. **Channel Order**: Canonical 10-20 montage
4. **O(N) Complexity**: Linear sequence modeling

## Lessons Learned

### Do This
- Check hardware constraints BEFORE design
- Pin exact versions for compiled extensions
- Test CUDA kernels separately from training
- Use manifest-driven datasets for reproducibility

### Don't Do This
- Assume PyPI mirrors serve correct versions
- Design for theoretical optimality over hardware reality
- Mix PyTorch versions between build and runtime
- Trust build isolation with compiled extensions

## Performance Comparison

### Local (RTX 4090) vs Modal (A100)
| Metric | Local | Modal | Speedup |
|--------|-------|-------|---------|
| Batch/s | ~2.5s | ~0.5s | 5× |
| Epoch | ~8h | ~1.5h | 5.3× |
| Memory | 24GB | 80GB | 3.3× |
| Cost | $0 | ~$3/h | - |

### Key Differences
- **Modal**: Purpose-built for ML, optimized drivers
- **Local**: WSL2 overhead, consumer GPU
- **Winner**: Modal for production, local for development

## Future Optimizations

### Potential Improvements
1. **FlashAttention**: For any transformer components
2. **Triton Kernels**: Custom CUDA for morphology ops
3. **Quantization**: INT8 inference for deployment
4. **Multi-GPU**: Data parallel training on 8×A100

### Not Worth It
1. **d_conv=4 Support**: Would require custom CUDA kernel
2. **FP8 Training**: Insufficient precision for medical
3. **Graph Compilation**: Dynamic shapes prevent optimization
