# 🧪 Brain-Go-Brr v2 Test Plan

## 📊 Current Coverage Analysis

### ✅ What We Have (Strong Coverage)
- **Models**: Mamba (84%), ResCNN (98%), UNet (95%), Detector (98%)
- **Core Pipeline**: Streaming (97%), Events (86%), Metrics (93%)
- **Integration**: Smoke tests, model assembly, streaming tests
- **Performance**: Latency benchmarks, memory profiling
- **Clinical**: Channel ordering, TAES metrics validation

### 🚨 Critical Gaps (0% Coverage)
1. **WandB Integration** (`train/wandb_integration.py` - 0%)
2. **Modal Deployment** (No tests for distributed training)
3. **S3 Data Loading** (No tests for cloud data paths)

### ⚠️ Moderate Gaps (60-70% Coverage)
1. **Data IO** (`data/io.py` - 71%)
   - Missing: Corrupted EDF handling
   - Missing: Channel missing scenarios
   - Missing: Multi-file batch loading

2. **Dataset Loading** (`data/datasets.py` - 62%)
   - Missing: Cache invalidation
   - Missing: Parallel worker edge cases
   - Missing: Memory-mapped loading

3. **CLI Commands** (`cli/cli.py` - 69%)
   - Missing: Modal submission paths
   - Missing: Error recovery
   - Missing: Config validation edge cases

## 🎯 Priority Test Additions

### Priority 1: Data Robustness 🔴
```python
# tests/unit/data/test_io_edge_cases.py
- test_corrupted_edf_header_recovery()
- test_missing_critical_channels()
- test_channel_interpolation_limits()
- test_extreme_sampling_rates()
- test_malformed_annotations()
- test_unicode_channel_names()
- test_zero_duration_files()
```

### Priority 2: Training Stability 🟠
```python
# tests/integration/test_training_edge_cases.py
- test_nan_gradient_recovery()
- test_loss_explosion_handling()
- test_checkpoint_corruption_recovery()
- test_mixed_precision_stability()
- test_class_imbalance_extreme()  # 99.9% negative
- test_empty_batch_handling()
- test_oom_recovery()
```

### Priority 3: Production Deployment 🟡
```python
# tests/integration/test_modal_deployment.py
- test_modal_volume_mounting()
- test_distributed_data_loading()
- test_multi_gpu_synchronization()
- test_s3_data_streaming()
- test_checkpoint_s3_sync()
- test_wandb_logging_integration()
```

### Priority 4: Post-Processing Edge Cases 🟢
```python
# tests/unit/post/test_hysteresis_edge.py
- test_rapid_oscillations()
- test_boundary_conditions()
- test_single_sample_spike()
- test_morphology_extremes()
- test_duration_filter_edge()
```

### Priority 5: Error Recovery 🔵
```python
# tests/integration/test_error_recovery.py
- test_cuda_oom_batch_reduction()
- test_data_loading_retry_logic()
- test_checkpoint_partial_save()
- test_wandb_offline_fallback()
- test_config_validation_errors()
```

## 🏗️ Test Infrastructure Improvements

### 1. Fixture Enhancements
```python
# tests/conftest.py additions
@pytest.fixture
def corrupted_edf_path():
    """Generate corrupted EDF for testing."""

@pytest.fixture
def extreme_class_imbalance_data():
    """99.9% negative, 0.1% positive."""

@pytest.fixture
def modal_mock_environment():
    """Mock Modal cloud environment."""
```

### 2. Performance Regression Guards
```python
# tests/performance/test_regression.py
- test_inference_latency_regression()  # <100ms/window
- test_memory_leak_detection()  # Stable over 1000 iterations
- test_throughput_regression()  # >1000 windows/sec
```

### 3. Clinical Validation Suite
```python
# tests/clinical/test_clinical_validation.py
- test_seizure_onset_latency()  # <2s detection
- test_seizure_offset_accuracy()  # ±5s tolerance
- test_minimum_seizure_duration()  # Detect 5s seizures
- test_maximum_false_alarm_rate()  # <10 FA/24h
```

## 📈 Coverage Targets

| Module | Current | Target | Priority |
|--------|---------|--------|----------|
| data/io.py | 71% | 95% | HIGH |
| data/datasets.py | 62% | 90% | HIGH |
| cli/cli.py | 69% | 85% | MEDIUM |
| train/loop.py | 86% | 95% | MEDIUM |
| wandb_integration.py | 0% | 80% | LOW* |
| post/postprocess.py | 89% | 98% | HIGH |

*Low only if not using WandB in production

## 🔧 Implementation Strategy

### Phase 1 (Immediate)
1. Data robustness tests (prevent training crashes)
2. Hysteresis edge cases (clinical accuracy)
3. Error recovery tests (production stability)

### Phase 2 (This Week)
1. Modal deployment tests
2. Performance regression suite
3. Clinical validation expansion

### Phase 3 (Nice to Have)
1. WandB integration (if using)
2. Comprehensive CLI testing
3. Stress testing suite

## 🚀 Execution Commands

```bash
# Run new edge case tests
make test-edge

# Run performance regression
make test-perf

# Run clinical validation
make test-clinical

# Full test suite with coverage
make test-all
```

## 📝 Notes

- Focus on **data corruption** first - most common production failure
- **Class imbalance** tests critical - real data is 99%+ negative
- Modal tests can use mocks initially, then integration tests
- Performance tests should run nightly, not on every commit
- Clinical tests need real EDF samples from TUSZ

---

**Mission**: Bulletproof the pipeline for production deployment 🛡️