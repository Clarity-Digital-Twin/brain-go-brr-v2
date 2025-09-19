# Modal Deployment Lessons Learned

## Key Deployment Issues We Solved

### 1. Image Build Order (Modal Best Practice)
**Problem**: Modal throws `InvalidError` if you run commands after `add_local_*`
**Solution**: Always put `add_local_dir` and `add_local_file` LAST in your image chain

### 2. Path Resolution in Containers
**Problem**: `Path(__file__).parents[2]` fails in containers where file is at `/root/app.py`
**Solution**: Use relative paths that Modal resolves from the run location

### 3. Missing Dependencies
**Problem**: `tensorboard` wasn't included but is imported by training code
**Solution**: Add all dependencies including optional ones like tensorboard

## The Mamba-SSM Compilation Challenge

### What We Tried That Failed

1. **debian_slim image** ❌
   - No CUDA development tools
   - No nvcc compiler
   - Can't compile mamba-ssm CUDA kernels

2. **Installing mamba-ssm without nvcc** ❌
   - Error: `nvcc was not found`
   - Error: `bare_metal_version is not defined`
   - PyTorch version conflicts (base had 2.8, we needed 2.2.2)

### What Actually Works

**Use NVIDIA CUDA Development Images** ✅
```python
modal.Image.from_registry(
    "nvidia/cuda:12.1.0-devel-ubuntu22.04",
    add_python="3.11"
)
```

### Key Insights

1. **Mamba-SSM requires compilation**
   - Not just a pip package
   - Builds custom CUDA kernels
   - Needs full CUDA toolkit, not just runtime

2. **Order matters**
   - Install PyTorch first
   - Then numpy (with version constraint)
   - Then mamba-ssm
   - Otherwise get version conflicts

3. **First build is slow, then fast**
   - Initial: ~10-15 minutes (compilation)
   - Subsequent: ~30 seconds (cached image)
   - Worth the wait for A100 performance

4. **Modal caches aggressively**
   - Image layers cached across runs
   - Volumes persist data between runs
   - No need to re-upload data each time

## Performance Comparison

| Environment | GPU | VRAM | Speed | Cost/hr |
|------------|-----|------|-------|---------|
| Local | RTX 4090 | 24GB | 1x baseline | $0 (owned) |
| Modal | A100-80GB | 80GB | ~3x faster | $5.59 |
| Modal | A100-40GB | 40GB | ~2x faster | $3.99 |
| Modal | T4 | 16GB | ~0.3x slower | $0.59 |

**Recommendation**: Use A100-80GB for serious training, T4 for debugging.

## Current Status

✅ **Deployment Working**: Modal successfully builds and runs our code
✅ **CUDA Compilation**: Mamba-SSM compiles with nvcc (takes ~2 min first time)
✅ **Path Resolution**: Fixed all container path issues
✅ **Dependencies**: Added all required packages including tensorboard
⚠️ **Mamba-SSM Runtime**: Warning about fallback to Conv1d - may need debugging

## Future Improvements

1. Debug why mamba-ssm falls back to Conv1d despite successful installation
2. Consider pre-building mamba-ssm wheel to avoid compilation time
3. Explore Modal's new image builder (vs legacy)
4. Test with H100 for even faster training
5. Add spot instance support for 70% cost savings