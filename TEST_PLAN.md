# 🧪 Brain-Go-Brr v2 Test Plan - REAL TESTS ONLY

## 📊 Current Coverage Reality Check

### ✅ What We Have (Actually Good Tests)
- **Models**: REAL GPU tests with actual tensors (84-98% coverage)
- **Performance**: REAL latency benchmarks on CUDA (test_latency.py)
- **Integration**: REAL model forward passes, no mocking
- **Clinical**: REAL TAES metrics with actual predictions

### 🚨 Critical Missing Tests (REAL IMPLEMENTATION NEEDED)
1. **Corrupted EDF Recovery** - REAL malformed files from TUSZ
2. **Extreme Class Imbalance** - REAL 99.9% negative batches
3. **GPU Memory Pressure** - REAL OOM scenarios with batch scaling
4. **Multi-GPU Training** - REAL distributed data parallel tests

### ⚠️ Coverage Gaps That Matter
1. **Data IO** (`data/io.py` - 71%)
   - Need: REAL corrupted TUSZ files that crash MNE
   - Need: REAL missing channel scenarios from clinical data
   - Need: REAL parallel loading stress tests

2. **Training Loop** (`train/loop.py` - 60-70%)
   - Need: REAL gradient explosion with extreme LR
   - Need: REAL focal loss collapse detection
   - Need: REAL checkpoint corruption recovery

3. **Post-processing** (`post/postprocess.py` - 89%)
   - Need: REAL rapid oscillation sequences
   - Need: REAL boundary condition stress tests

## 🎯 REAL Tests We Need NOW

### Priority 1: Data Corruption (REAL FILES) 🔴
```python
# tests/integration/data/test_io_edge_cases.py
def test_real_corrupted_tusz_files():
    """Test ACTUAL corrupted TUSZ files that crash in production"""
    # Use REAL problematic files from TUSZ that have:
    # - Malformed headers
    # - Missing channels (no Fz, Pz)
    # - Unicode channel names (T7→T3 synonyms)
    # - Zero duration segments

def test_extreme_class_imbalance_real_data():
    """Load REAL TUSZ files with 99.9% background"""
    # Not synthetic - REAL seizure-free recordings
    # Assert balanced sampler actually finds the 0.1% seizure windows

def test_parallel_loading_stress():
    """Stress test with num_workers=8 on REAL data"""
    # Load 100GB of EDF files in parallel
    # Assert no deadlocks, no corruption
```

### Priority 2: Training Explosions (REAL GPU) 🟠
```python
# tests/integration/test_training_edge_cases.py
def test_focal_loss_collapse_real_imbalance():
    """Train 5 epochs on REAL 99.9% negative data"""
    # Use REAL focal loss with alpha=0.999
    # Assert AUROC doesn't collapse to 0.5
    # Assert positive class gets non-zero gradients

def test_gradient_explosion_extreme_lr():
    """REAL training with LR=10.0 for 10 steps"""
    # No mocking - let it explode
    # Assert gradient clipping prevents NaN

def test_cuda_oom_recovery():
    """Progressively increase batch size until OOM"""
    # Start at batch_size=32, double each epoch
    # Assert graceful batch reduction on OOM
```

### Priority 3: GPU Memory & Performance (REAL CUDA) 🟡
```python
# tests/performance/test_memory_pressure.py
def test_memory_leak_1000_epochs():
    """Train 1000 mini-epochs, monitor GPU memory"""
    # REAL training loop, REAL GPU memory tracking
    # Assert memory usage stable (no leak)

def test_multi_gpu_data_parallel():
    """REAL multi-GPU training if available"""
    # Use torch.nn.DataParallel on 2+ GPUs
    # Assert gradients synchronized correctly

def test_mixed_precision_numerical_stability():
    """Train with AMP on REAL data"""
    # Assert no NaN/Inf in 100 epochs
    # Assert final loss within 1% of FP32
```

### Priority 4: Post-Processing Stress (REAL PREDICTIONS) 🟢
```python
# tests/integration/post/test_hysteresis_edge.py
def test_rapid_oscillations_stress():
    """Generate 1M samples oscillating at Nyquist frequency"""
    # Probability flips every sample between 0.1 and 0.9
    # Assert hysteresis doesn't crash or leak memory

def test_morphology_extreme_kernels():
    """Test with kernel_size=1001 on REAL predictions"""
    # Assert doesn't crash, produces valid output

def test_stitching_100_overlapping_windows():
    """Stitch 100 windows with 90% overlap"""
    # REAL detector output, not synthetic
    # Assert final prediction smooth and valid
```

### Priority 5: Production Scenarios (REAL DEPLOYMENT) 🔵
```python
# tests/integration/test_production_scenarios.py
def test_24_hour_continuous_inference():
    """Run inference on 24 hours of REAL EEG"""
    # Load actual 24-hour recording
    # Assert <10 FA/24h at tau_on=0.86
    # Assert memory usage stable

def test_checkpoint_recovery_mid_epoch():
    """Kill training at step 500, resume"""
    # REAL checkpoint save/load
    # Assert continues from exact step
    # Assert metrics identical post-resume
```

## 🏗️ NO MOCKS - REAL TEST INFRASTRUCTURE

### 1. REAL Data Fixtures
```python
# tests/conftest.py additions
@pytest.fixture
def real_corrupted_edf():
    """Return ACTUAL corrupted EDF from TUSZ that crashes MNE"""
    # data_ext4/tusz/edf/train/01_tcp_ar/002/00000258/*.edf
    # These files have known issues we need to handle

@pytest.fixture
def real_imbalanced_dataset():
    """REAL dataset with 99.9% background from TUSZ"""
    # Use actual seizure-free recordings
    # Not synthetic bullshit

@pytest.fixture
def gpu_memory_tracker():
    """Track REAL GPU memory usage during tests"""
    return torch.cuda.memory_stats()
```

### 2. REAL Performance Benchmarks
```python
# tests/performance/test_real_performance.py
def test_latency_under_load():
    """100ms latency with 8 concurrent streams"""
    # REAL concurrent inference threads
    # REAL GPU contention

def test_memory_stability_24h():
    """No leaks over 24 hour simulation"""
    # REAL 24 hours of windows
    # REAL memory tracking

def test_throughput_saturation():
    """Find actual throughput limit on GPU"""
    # Keep increasing batch size until GPU saturates
    # Report actual windows/sec
```

### 3. REAL Clinical Validation
```python
# tests/clinical/test_real_clinical.py
def test_tusz_test_set():
    """Run on ENTIRE TUSZ test set"""
    # REAL test files, not cherry-picked
    # Assert TAES metrics match paper claims

def test_chb_mit_cross_validation():
    """Full CHB-MIT dataset validation"""
    # All 24 patients
    # Assert generalization across patients
```

## 📈 REAL Coverage Targets

| Module | Current | Target | Method |
|--------|---------|--------|--------|
| data/io.py | 71% | 95% | Test REAL corrupted TUSZ files |
| data/datasets.py | 62% | 90% | Test REAL parallel loading |
| train/loop.py | 60% | 90% | Test REAL training explosions |
| post/postprocess.py | 89% | 98% | Test REAL edge predictions |

## 🔧 Implementation NOW

### TODAY - Do These First
1. **test_io_edge_cases.py** - REAL corrupted file handling
2. **test_training_edge_cases.py** - REAL gradient explosions
3. **test_hysteresis_edge.py** - REAL oscillation stress

### TOMORROW - GPU Stress
1. **test_memory_pressure.py** - REAL memory leak detection
2. **test_multi_gpu.py** - REAL DataParallel if 2+ GPUs
3. **test_mixed_precision.py** - REAL AMP stability

## 🚀 Run Commands

```bash
# Run REAL edge cases on GPU
CUDA_VISIBLE_DEVICES=0 pytest tests/unit/data/test_io_edge_cases.py -xvs

# Run REAL performance tests
pytest tests/performance/ -m performance -xvs

# Run REAL clinical validation
pytest tests/clinical/ -m clinical -xvs

# FULL test suite with GPU
make test-all
```

## 📝 CRITICAL RULES

- **NO MOCKS in integration/perf tests** - Use real data, real models, real GPU
- **Unit tests stay pure** - Fast, isolated, no disk/GPU
- **Mark tests properly** - @pytest.mark.integration, @pytest.mark.gpu, @pytest.mark.performance
- **STRESS TEST** - Push to actual failure points in integration tests
- **MEASURE REAL** - Actual latency, actual memory, actual throughput in perf tests

---

**Mission**: REAL tests that break REAL things so we can fix them 🔥