# CI/CD Configuration

## Overview

Brain-Go-Brr V3 has specific CI requirements due to GPU-dependent components (Mamba-SSM, PyTorch Geometric). This document explains the CI setup and how to handle local vs CI divergence.

## CI Workflows

### 1. Standard CI (`ci.yml`)
- **Runs on**: Every push/PR to main/development
- **Environment**: Ubuntu latest, CPU-only
- **Python versions**: 3.11, 3.12
- **Key features**:
  - PyTorch Geometric installed from CPU wheels
  - Mamba forced to Conv1d fallback mode
  - Skips GPU and performance tests

### 2. GPU CI (`ci-gpu.yml`)
- **Runs on**: Self-hosted runners with GPU (optional)
- **Environment**: CUDA 12.1+ required
- **Key features**:
  - Full Mamba-SSM CUDA kernels
  - PyTorch Geometric with CUDA support
  - Runs performance benchmarks

## Dependency Handling

### PyTorch Geometric (V3 Architecture)
```bash
# CPU version (CI)
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cpu.html --no-deps
pip install torch-geometric==2.6.1 --no-deps

# GPU version (local/self-hosted)
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cu121.html --no-deps
pip install torch-geometric==2.6.1 --no-deps
```

### Mamba-SSM
- **CI**: Uses Conv1d fallback (set `SEIZURE_MAMBA_FORCE_FALLBACK=1`)
- **Local**: Full CUDA kernels with `mamba-ssm==2.2.2`

## Environment Variables

| Variable | CI Value | Local Value | Purpose |
|----------|----------|-------------|---------|
| `SEIZURE_MAMBA_FORCE_FALLBACK` | `1` | `0` | Force Conv1d fallback in CI |
| `BGB_LIMIT_FILES` | `2` | varies | Limit data files for speed |
| `CUDA_VISIBLE_DEVICES` | N/A | `0` | Single GPU for tests |

## Test Categories

### Markers
- `@pytest.mark.gpu` - Requires CUDA GPU
- `@pytest.mark.performance` - Performance benchmarks
- `@pytest.mark.serial` - Must run serially (not parallel)

### Test Exclusions in CI
```bash
# Standard CI excludes GPU/performance tests
pytest -m "not gpu and not performance"

# GPU CI runs only GPU/performance tests
pytest -m "gpu or performance"
```

## Known CI/Local Differences

1. **Mamba Implementation**:
   - CI: Conv1d fallback (kernel_size=4)
   - Local: Real Mamba2 SSM

2. **GNN Performance**:
   - CI: CPU-only, slower
   - Local: CUDA accelerated

3. **Memory Usage**:
   - CI: Limited to GitHub runner specs
   - Local: Full GPU VRAM available

## Troubleshooting

### Test Failures Only in CI

1. **ImportError: PyTorch Geometric not installed**
   - Solution: CI workflow installs PyG with `--no-deps` flag

2. **Mamba kernel_size assertion failure**
   - Solution: Test expects d_conv=4 (not 5)

3. **CUDA out of memory**
   - Solution: Tests use smaller batch sizes

### Running CI Tests Locally

```bash
# Simulate CI environment
export SEIZURE_MAMBA_FORCE_FALLBACK=1
export BGB_LIMIT_FILES=2

# Run CI test suite
pytest -m "not gpu and not performance" --cov=src

# Reset environment
unset SEIZURE_MAMBA_FORCE_FALLBACK
unset BGB_LIMIT_FILES
```

## Future Improvements

1. **GitHub-hosted GPU runners**: When available, migrate GPU tests
2. **Dependency caching**: Cache PyG wheels between runs
3. **Matrix testing**: Test multiple CUDA versions
4. **Nightly runs**: Full integration tests on schedule

## Maintenance

- Update PyG wheels URL when upgrading PyTorch
- Keep `mamba-ssm==2.2.2` pinned (later versions have issues)
- Test both CPU and GPU paths locally before PR