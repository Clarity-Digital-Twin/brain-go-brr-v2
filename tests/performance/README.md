# Performance Tests

## Overview

These tests measure inference latency and throughput to ensure the model meets real-time requirements for clinical seizure detection.

## Requirements

- **Real-time processing**: < 1000ms latency (we target much better)
- **P95 latency**: 100-125ms on GPU (varies by hardware)
- **Throughput**: > 15x real-time on GPU

## Running Tests

```bash
# Run all performance tests
make test-performance

# Skip performance tests (for CI or slow systems)
SKIP_PERF_TESTS=1 make test

# Convert failures to warnings (useful for CI)
PERF_TESTS_WARN_ONLY=1 make test-performance
```

## Hardware Expectations

### GPU Performance
- **RTX 4090/3090**: P95 ~110-125ms, Median ~50-65ms
- **A100/V100**: P95 ~80-110ms, Median ~40-55ms
- **Older GPUs**: May need threshold adjustments

### Why These Tests Matter
1. **Clinical Safety**: Seizure detection must be near real-time
2. **Regression Detection**: Catch performance degradations early
3. **Hardware Planning**: Validate deployment requirements

## Troubleshooting

### Test Failures
- **Thermal throttling**: Ensure GPU cooling is adequate
- **Background processes**: Close other GPU applications
- **Driver issues**: Update to latest CUDA drivers
- **Memory pressure**: Reduce batch size if OOM

### Making Tests More Stable
- Increase warmup iterations
- Use median instead of P95 for primary metric
- Add variance tolerance
- Test relative performance vs absolute

## CI/CD Considerations

For CI environments, consider:
- Using `PERF_TESTS_WARN_ONLY=1` to avoid flaky failures
- Running on dedicated hardware for consistency
- Tracking performance trends over time vs hard thresholds
- Separating "smoke" performance tests from comprehensive benchmarks