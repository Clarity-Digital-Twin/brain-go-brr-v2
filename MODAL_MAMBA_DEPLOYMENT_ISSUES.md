# Modal Mamba-SSM Deployment Issues & Solutions

## Critical Issue: Mamba-SSM CUDA Compilation Failures

### 🔴 Problem 1: Symbol Linking Error
**Error**: `undefined symbol: _ZN3c104cuda14ExchangeDeviceEa`
- **Cause**: Mamba-SSM pre-built wheel compiled against different PyTorch version
- **Solution**: Force recompilation with `--no-build-isolation` flag

### 🔴 Problem 2: Missing Build Dependencies
**Error**: `ModuleNotFoundError: No module named 'packaging'`
- **Cause**: `--no-build-isolation` requires all build deps pre-installed
- **Solution**: Install `packaging` before mamba-ssm

### 🔴 Problem 3: Missing C++ Compiler
**Error**: `Command '['which', 'clang++']' returned non-zero exit status 1`
- **Cause**: CUDA devel image lacks build tools for compilation
- **Solution**: Install `build-essential` and `ninja-build` via apt

### 🔴 Problem 4: Modal Timeout Limit
**Error**: `Timeout must be between 10s and 86400s`
- **Cause**: Tried to set 30 hours (108000s) but Modal max is 24 hours
- **Solution**: Set timeout to 86400 (24 hours max)

## ✅ FINAL WORKING SOLUTION

```python
# deploy/modal/app.py
image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    # CRITICAL: Install build tools FIRST
    .apt_install("build-essential", "ninja-build")
    # Install PyTorch with exact CUDA version
    .pip_install(
        "torch==2.2.2",
        "torchvision==0.17.2",
        "numpy<2.0",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    # CRITICAL: Install packaging and ninja for build process
    .pip_install("packaging", "ninja")
    # CRITICAL: Use --no-build-isolation to link with our PyTorch
    .run_commands(
        "pip install --no-build-isolation 'mamba-ssm>=2.0.0'"
    )
    # Rest of dependencies...
)
```

## Key Lessons Learned

### 1. Mamba-SSM Compilation Requirements
- **MUST** compile from source to match PyTorch version
- **CANNOT** use pre-built wheels (they link to wrong PyTorch)
- **NEEDS** full build toolchain: gcc, g++, ninja
- **REQUIRES** `--no-build-isolation` to use installed PyTorch

### 2. Modal Image Building Order Matters
1. Base CUDA image with nvcc
2. System build tools (apt packages)
3. PyTorch with exact CUDA version
4. Build dependencies (packaging, ninja)
5. Mamba-SSM compilation from source
6. Other Python packages

### 3. Debugging Steps That Helped
- Check Modal logs at: https://modal.com/apps/clarity-digital-twin/
- Test with smoke_test.yaml first (1 epoch, minimal cost)
- Monitor image build logs for compilation errors
- Verify CUDA availability in runtime

## Environment Specifications

### Working Configuration
- **Base Image**: `nvidia/cuda:12.1.0-devel-ubuntu22.04`
- **Python**: 3.11
- **PyTorch**: 2.2.2+cu121
- **CUDA**: 12.1
- **Mamba-SSM**: >=2.0.0 (compiled from source)
- **Build Tools**: gcc, g++, ninja-build

### Failed Configurations
- ❌ Using pip's default build isolation
- ❌ Missing ninja-build
- ❌ Using pre-built mamba-ssm wheels
- ❌ PyTorch/CUDA version mismatches

## Fallback Behavior

If Mamba-SSM CUDA fails to load at runtime:
- Code has Conv1d fallback in `src/brain_brr/models/mamba.py`
- Performance degradation but training still works
- Set `SEIZURE_MAMBA_FORCE_FALLBACK=1` to force fallback

## Cost & Time Estimates

### Smoke Test (1 epoch)
- **Time**: ~15 minutes
- **Cost**: ~$1.40
- **Purpose**: Verify pipeline before full run

### Full Training (100 epochs)
- **Time**: ~25 hours (may hit 24h limit)
- **Cost**: ~$140
- **GPU**: A100-80GB
- **Early stopping**: Likely completes in 50-70 epochs

## Commands for Testing

```bash
# Quick smoke test
modal run deploy/modal/app.py --config configs/smoke_test.yaml

# Full A100 training
modal run deploy/modal/app.py --config configs/tusz_train_a100.yaml --detach

# Resume from checkpoint
modal run deploy/modal/app.py --resume true

# Check Modal credits
modal status
```

## Troubleshooting Checklist

- [ ] AWS S3 secret configured: `modal secret list | grep aws-s3-secret`
- [ ] Modal credits sufficient: Need ~$140 for full run
- [ ] Build tools installed in image
- [ ] PyTorch CUDA version matches image CUDA
- [ ] Mamba-SSM compiles without errors
- [ ] Smoke test completes successfully

## References

- GitHub Issue: https://github.com/state-spaces/mamba/issues/764
- PyTorch Forums: https://discuss.pytorch.org/t/importerror-while-installing-mamba-ssm/215907
- Solution: `pip install --no-build-isolation` with proper build env

---

**Last Updated**: 2025-01-19
**Status**: WORKING - Mamba-SSM compiling with CUDA support