# Mamba d_conv Kernel Size Decision

## Summary
We use d_conv=4 throughout the codebase - the maximum value supported by CUDA kernels.

## Why d_conv=4

### Temporal Resolution Analysis
- At 256 Hz sampling rate:
  - Kernel 4 = ~15.6ms temporal window (4/256 ≈ 0.0156 seconds)
  - Kernel 3 = ~11.7ms temporal window (3/256 ≈ 0.0117 seconds)
  - Kernel 2 = ~7.8ms temporal window (2/256 ≈ 0.0078 seconds)

### Clinical Rationale
- ~15.6ms window captures high-frequency epileptiform activities
- Balances local temporal pattern capture with computational efficiency
- Matches the scale of fast ripples and spikes in EEG

## CUDA Kernel Support

```python
# mamba-ssm CUDA kernels support d_conv in {2, 3, 4}
# We use 4 as it provides the best temporal coverage
```

## Runtime Options

### Force CPU Fallback
To force the Conv1d fallback regardless of CUDA availability (e.g., for testing):
```bash
export SEIZURE_MAMBA_FORCE_FALLBACK=1
```

This uses a simpler Conv1d implementation that preserves shapes but isn't a true SSM.

### Edge Stream Alignment (v3)
- The edge temporal stream uses a learned 1→D→1 projection around Bi‑Mamba2 with `D` a multiple‑of‑8 (default 16) to satisfy CUDA stride/alignment constraints and avoid fast‑path exits. This keeps the fused kernels active instead of falling back to Conv1d, and increases modeling capacity on edges.

## Architecture Alignment
- Vision Mamba uses d_conv=4
- Original Mamba paper tested d_conv=4
- Industry standard for sequence modeling

The choice of d_conv=4 provides optimal balance between temporal resolution and hardware efficiency.
