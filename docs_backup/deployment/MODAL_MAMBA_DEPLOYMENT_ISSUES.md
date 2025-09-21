# Modal Mamba-SSM Deployment Issues & Solutions

**Status: ✅ RESOLVED**
**Last updated: 2025-09-20**
**Location: `/home/jj/proj/brain-go-brr-v2/docs/deployment/MODAL_MAMBA_DEPLOYMENT_ISSUES.md`**

## Summary
All Mamba-SSM compilation issues have been resolved. The working solution is implemented in `/deploy/modal/app.py`.

## Historical Issues (ALL RESOLVED)

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

## ✅ FINAL WORKING SOLUTION (IMPLEMENTED)

```python
# WORKING CODE - Already in deploy/modal/app.py
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
    # CRITICAL: Install packaging for build process
    .pip_install("packaging")
    # CRITICAL: Use CC/CXX env vars AND --no-build-isolation AND verbose output
    .run_commands(
        "export CC=gcc CXX=g++ && pip install -v --no-build-isolation 'mamba-ssm>=2.0.0'"
    )
    # Rest of dependencies...
)
```

### 🔴 Problem 5: Legacy Image Builder Warning
**Warning**: "Using legacy Image Builder version. We suggest upgrading"
- **Cause**: Workspace using old Image Builder version
- **Solution**: Go to https://modal.com/settings/image-config → Select 2025.06 → Save
- **Benefits**: Better dependency isolation, no conflicts, faster builds

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

### 3. Critical Setup Steps (IN ORDER!)

#### A. Configure Modal Workspace
1. **Set Image Builder Version** (REQUIRED - Do this FIRST!)
   - Go to https://modal.com/settings/image-config
   - Select "2025.06" (NOT the legacy version)
   - Click Save
   - This avoids dependency conflicts and compilation issues

2. **Create AWS S3 Secret**
   ```bash
   modal secret create aws-s3-secret \
     --env AWS_ACCESS_KEY_ID=your_key \
     --env AWS_SECRET_ACCESS_KEY=your_secret \
     --env AWS_DEFAULT_REGION=us-east-1
   ```

3. **Verify Modal Credits**
   ```bash
   modal status  # Need ~$140 for full training
   ```

#### B. Debugging Steps That Helped
- Check Modal logs at: https://modal.com/apps/clarity-digital-twin/
- Test with smoke_test.yaml first (1 epoch, minimal cost)
- Monitor image build logs for compilation errors
- Use verbose output (-v) for pip install to debug compilation
- Export CC=gcc CXX=g++ for compiler specification

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
# FIRST TIME SETUP (REQUIRED!)
modal setup  # Login to Modal
# Then go to https://modal.com/settings/image-config and set to 2025.06!

# Quick smoke test (do this first!)
cd deploy/modal
modal run --detach deploy/modal/app.py -- --action train --config configs/smoke_test.yaml

# Full A100 training (after smoke test succeeds) - Modal's --detach BEFORE script!
modal run --detach deploy/modal/app.py -- --action train --config configs/tusz_train_a100.yaml

# Resume from checkpoint
modal run --detach deploy/modal/app.py -- --action train --resume true

# Check Modal credits
modal status

# List running apps
modal app list

# Stop an app
modal app stop <app-id>
```

## Pre-Flight Checklist (DO IN ORDER!)

1. [ ] **Image Builder Version**: Set to 2025.06 at https://modal.com/settings/image-config
2. [ ] **Modal CLI**: Installed and authenticated (`modal setup`)
3. [ ] **AWS S3 Secret**: Created with `modal secret create aws-s3-secret ...`
4. [ ] **S3 Bucket**: Data uploaded to `brain-go-brr-eeg-data-20250919` bucket
5. [ ] **Modal Credits**: Have $140+ available (`modal status`)
6. [ ] **Local Test**: Run `make q` to ensure code quality
7. [ ] **Smoke Test**: Complete successfully before full training
8. [ ] **Clean Old Apps**: Use `modal app stop` to clean up failed runs

## References

- GitHub Issue: https://github.com/state-spaces/mamba/issues/764
- PyTorch Forums: https://discuss.pytorch.org/t/importerror-while-installing-mamba-ssm/215907
- Solution: `pip install --no-build-isolation` with proper build env

---

**Last Updated**: 2025-01-19 (Evening)
**Status**: FULLY WORKING - Successfully deployed with Image Builder 2025.06

## Complete Deployment Timeline

1. **Initial attempts**: Failed with legacy Image Builder
2. **Mamba-SSM compilation**: Fixed with CC/CXX env vars + --no-build-isolation
3. **Timeout issues**: Fixed by setting to 86400s (Modal's 24h limit)
4. **Image Builder upgrade**: Switched to 2025.06 - SOLVED ALL ISSUES!
5. **Success**: Smoke test running on A100-80GB with full CUDA support

## Key Takeaways

- **ALWAYS** set Image Builder to latest version BEFORE deployment
- **ALWAYS** use verbose output (-v) for debugging compilation
- **ALWAYS** run smoke test before committing to full training
- **NEVER** skip the CC/CXX environment variables for Mamba-SSM
- **NEVER** use legacy Image Builder with complex CUDA dependencies
