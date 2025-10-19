# Testing Requirements - Complete Specifications

**Purpose**: Comprehensive testing requirements for NEDC integration

**Status**: UPDATED to reflect existing infrastructure reuse (October 19, 2025)

---

## 🔄 UPDATE: Reduced Scope Due to Code Reuse

**Original Plan**: 23 tests across 3 new components (CSVBIConverter, NEDCScorer, ModelEvaluator)

**Revised Plan**: ~10 tests for 1 new component (NEDCScorer only)

**Why**: CSVBIConverter and ModelEvaluator already exist as `export_csv_bi()` and `run_evaluation()`

---

## Test Suite Summary (REVISED)

| Component | Unit Tests | Integration Tests | Total | Status |
|-----------|------------|-------------------|-------|--------|
| ~~CSVBIConverter~~ | ~~10~~ | ~~0~~ | ~~10~~ | ✅ Already exists (export_csv_bi) |
| NEDCScorer | 8 | 1 | 9 | 🆕 NEW - implement these |
| ~~ModelEvaluator~~ | ~~0~~ | ~~4~~ | ~~4~~ | ✅ Already exists (run_evaluation) |
| **TOTAL** | **8** | **1** | **9** | **~60% reduction!** |

---

## Test Coverage Requirements (REVISED)

| Component | Target Coverage | Rationale |
|-----------|----------------|-----------|
| ~~format_converter.py~~ | ~~≥ 95%~~ | N/A - already exists as export_csv_bi |
| nedc_wrapper.py | ≥ 90% | Critical integration point with NEDC |
| ~~evaluator.py~~ | ~~≥ 85%~~ | N/A - already exists as run_evaluation |
| **nedc_wrapper.py** | **≥ 90%** | **Only new component** |

---

## Performance Requirements

### Format Conversion (CSVBIConverter)

| Operation | Target Time | Max Memory | Notes |
|-----------|-------------|------------|-------|
| Convert 1 recording to CSV_BI | < 100ms | < 100MB | Per-recording overhead |
| Convert 100 recordings | < 10s | < 500MB | Batch processing |
| Convert 1000 recordings | < 60s | < 500MB | Dev set scale |
| Convert 2000 recordings (eval) | < 3 min | < 1GB | Full eval set |

**Measurement**:
```python
import time
start = time.time()
converter.convert_directory(pred_dir, out_dir, metadata)
elapsed = time.time() - start
assert elapsed < 180  # < 3 minutes for 2000 files
```

### NEDC Scoring (NEDCScorer)

| Operation | Target Time | Max Memory | Notes |
|-----------|-------------|------------|-------|
| Score 1 file pair (overlap) | < 30ms | < 50MB | Per-pair overhead |
| Score 100 file pairs | < 3s | < 200MB | Small batch |
| Score 1000 file pairs | < 30s | < 1GB | Dev set scale |
| Score 2000 file pairs (eval) | < 60s | < 2GB | Full eval set |

**Measurement**:
```python
start = time.time()
metrics = scorer.score_predictions(ref_dir, hyp_dir, algorithm="overlap")
elapsed = time.time() - start
assert elapsed < 60  # < 1 minute for 2000 pairs
```

### End-to-End Evaluation (ModelEvaluator)

| Operation | Target Time | Max Memory | Notes |
|-----------|-------------|------------|-------|
| Load checkpoint | < 5s | < 500MB | Model initialization |
| Inference (dev, 1832 files) | < 20 min | < 24GB | GPU-dependent |
| Inference (eval, ~2000 files) | < 25 min | < 24GB | GPU-dependent |
| Full eval pipeline (dev) | < 30 min | < 24GB | Includes conversion + scoring |
| Full eval pipeline (eval) | < 40 min | < 24GB | Includes conversion + scoring |

**Measurement**:
```python
start = time.time()
results = evaluator.evaluate_on_split(split="dev", algorithm="overlap")
elapsed = time.time() - start
assert elapsed < 1800  # < 30 minutes
```

---

## Error Handling Specifications

### Component 1: CSVBIConverter

| Error Condition | Exception | Message Template | Recovery Strategy |
|-----------------|-----------|------------------|-------------------|
| Probs file not found | FileNotFoundError | "Probabilities file not found: {path}" | Skip file, log WARNING, continue |
| Invalid probs shape | ValueError | "Probabilities must be 1D array, got shape {shape}" | Skip file, log ERROR, continue |
| Duration mismatch | ValueError | "Duration mismatch: expected {expected}s, got {actual}s" | Skip file, log ERROR, continue |
| Cannot write output | IOError | "Failed to write CSV_BI file to {path}: {error}" | Skip file, log ERROR, continue |
| Missing metadata | KeyError | "No metadata found for file_id: {file_id}" | Skip file, log WARNING, continue |
| Invalid sampling rate | ValueError | "sampling_rate must be > 0, got {rate}" | Fail fast (init) |
| Invalid post-processing config | ValueError | "Invalid config: tau_on ({tau_on}) >= tau_off ({tau_off})" | Fail fast (init) |

**Test Coverage**:
```python
# Test each error condition
def test_convert_recording_file_not_found(...):
    with pytest.raises(FileNotFoundError, match="Probabilities file not found"):
        converter.convert_recording(missing_path, ...)

def test_convert_recording_wrong_shape(...):
    with pytest.raises(ValueError, match="must be 1D array"):
        converter.convert_recording(probs_2d_path, ...)

# ... etc for all 7 error conditions
```

### Component 2: NEDCScorer

| Error Condition | Exception | Message Template | Recovery Strategy |
|-----------------|-----------|------------------|-------------------|
| nedc-bench not found | ImportError | "NEDC-BENCH not found at {path}. Clone from https://github.com/..." | Fail fast (init) |
| nedc-bench import failed | RuntimeError | "Failed to import nedc-bench: {error}" | Fail fast (init) |
| Reference dir not found | FileNotFoundError | "Reference directory not found: {path}" | Fail fast |
| Hypothesis dir not found | FileNotFoundError | "Hypothesis directory not found: {path}" | Fail fast |
| No CSV_BI files in ref | ValueError | "No .csv_bi files found in reference directory: {path}" | Fail fast |
| No CSV_BI files in hyp | ValueError | "No .csv_bi files found in hypothesis directory: {path}" | Fail fast |
| File count mismatch | ValueError | "File count mismatch: {ref_count} reference vs {hyp_count} hypothesis" | Fail fast |
| NEDC scoring failed | RuntimeError | "NEDC scoring failed: {error}" | Fail fast |
| Invalid CSV_BI format | RuntimeError | "Invalid CSV_BI format in {file}: {error}" | Skip file, log ERROR, continue |

**Test Coverage**:
```python
def test_init_nedc_bench_not_found(...):
    with pytest.raises(ImportError, match="NEDC-BENCH not found"):
        NEDCScorer()

def test_score_predictions_dir_not_found(...):
    with pytest.raises(FileNotFoundError):
        scorer.score_predictions(nonexistent_ref, nonexistent_hyp)

# ... etc for all 9 error conditions
```

### Component 3: ModelEvaluator

| Error Condition | Exception | Message Template | Recovery Strategy |
|-----------------|-----------|------------------|-------------------|
| Checkpoint not found | FileNotFoundError | "Checkpoint not found: {path}" | Fail fast (init) |
| Checkpoint load failed | RuntimeError | "Failed to load checkpoint: {error}" | Fail fast (init) |
| Invalid checkpoint format | RuntimeError | "Invalid checkpoint format: missing '{key}' key" | Fail fast (init) |
| Test data not found | FileNotFoundError | "Test data not found for split '{split}'" | Fail fast |
| Inference failed (file) | RuntimeError | "Inference failed on {file_id}: {error}" | Skip file, log ERROR, continue |
| Conversion failed (file) | RuntimeError | "Conversion to CSV_BI failed for {file_id}: {error}" | Skip file, log ERROR, continue |
| Scoring failed (overall) | RuntimeError | "NEDC scoring failed: {error}" | Fail fast |
| No predictions generated | ValueError | "No predictions generated for split '{split}'" | Fail fast |

**Test Coverage**:
```python
def test_init_checkpoint_not_found(...):
    with pytest.raises(FileNotFoundError):
        ModelEvaluator(nonexistent_ckpt, output_dir)

# ... etc for all 8 error conditions
```

---

## Logging Specifications

### Log Levels

**DEBUG** (verbose mode only):
```python
logger.debug(f"Loading probs from {probs_path}")
logger.debug(f"Converted {len(events)} events from {n_samples} samples")
logger.debug(f"Writing CSV_BI to {output_path}")
```

**INFO** (always):
```python
logger.info(f"[CSVBIConverter] Converting {len(probs_files)} recordings...")
logger.info(f"[CSVBIConverter] Progress: {done}/{total} ({pct:.1f}%)")
logger.info(f"[CSVBIConverter] Converted {len(converted)} files in {elapsed:.1f}s")

logger.info(f"[NEDCScorer] Scoring {len(hyp_files)} file pairs with '{algorithm}' algorithm...")
logger.info(f"[NEDCScorer] Sensitivity@10FA: {metrics.sensitivity_at_10FA_24h:.2f}%")
logger.info(f"[NEDCScorer] Scoring complete in {elapsed:.1f}s")

logger.info(f"[ModelEvaluator] Loading checkpoint: {checkpoint_path}")
logger.info(f"[ModelEvaluator] Running inference on {split} split...")
logger.info(f"[ModelEvaluator] Inference complete: {len(predictions)} recordings")
logger.info(f"[ModelEvaluator] Converting predictions to CSV_BI...")
logger.info(f"[ModelEvaluator] Scoring with NEDC-BENCH...")
logger.info(f"[ModelEvaluator] Evaluation complete! Results saved to {output_dir}")
```

**WARNING** (non-fatal issues):
```python
logger.warning(f"No metadata found for {file_id}, skipping")
logger.warning(f"Duration mismatch for {file_id}: {actual}s != {expected}s, skipping")
logger.warning(f"Skipped {skipped}/{total} files due to missing metadata")
```

**ERROR** (failed operations):
```python
logger.error(f"Failed to convert {file_id}: Invalid probs shape {shape}")
logger.error(f"NEDC scoring failed for {file_id}: {error}")
logger.error(f"Inference failed on {file_id}: {error}")
```

### Logging Configuration

```python
import logging

def setup_logging(verbose: bool = False):
    """Configure logging for evaluation pipeline"""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Suppress verbose third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
```

---

## Test Fixture Specifications

### Pytest Configuration

**File**: `tests/conftest.py` (or component-specific conftest)

```python
import pytest
from pathlib import Path
import numpy as np
from src.brain_brr.config.schemas import PostprocessingConfig

@pytest.fixture
def post_config():
    """Standard post-processing config for testing"""
    return PostprocessingConfig(
        tau_on=0.86,
        tau_off=0.78,
        morphology_opening_kernel_size=11,
        morphology_closing_kernel_size=31,
        min_duration_sec=3.0,
        max_duration_sec=600.0,
        merge_threshold_sec=2.0,
    )

@pytest.fixture
def sample_metadata():
    """Standard recording metadata"""
    from src.brain_brr.eval.format_converter import RecordingMetadata
    return RecordingMetadata(
        file_id="test_file_s001_t000",
        patient="test_file",
        session="s001",
        token="t000",
        duration_sec=300.0,
        sampling_rate=256,
    )

@pytest.fixture
def tmp_probs_file(tmp_path):
    """Create temporary .npy file with sample probabilities"""
    probs = np.random.rand(300 * 256).astype(np.float32)
    probs_path = tmp_path / "test_probs.npy"
    np.save(probs_path, probs)
    return probs_path
```

---

## Integration Test Requirements

### Test Environment Setup

```python
@pytest.fixture(scope="session")
def nedc_bench_available():
    """Check if nedc-bench is available for integration tests"""
    nedc_bench_path = Path("reference_repos/nedc-bench/src")
    if not nedc_bench_path.exists():
        pytest.skip("nedc-bench not found, skipping integration tests")
    return nedc_bench_path

@pytest.fixture(scope="session")
def gpu_available():
    """Check if GPU is available for inference tests"""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("GPU not available, skipping GPU tests")
```

### Pytest Markers

```python
# In pytest.ini or pyproject.toml
[tool.pytest.ini_options]
markers = [
    "integration: Integration tests (slower, require external dependencies)",
    "gpu: GPU-dependent tests",
    "slow: Slow tests (> 10 seconds)",
]
```

**Usage**:
```bash
# Run only unit tests (fast)
pytest -v -m "not integration and not gpu"

# Run all tests including integration
pytest -v

# Run only integration tests
pytest -v -m integration
```

---

## Test Data Requirements

### Minimal Test Dataset

**For component testing**, create minimal synthetic data:

```python
@pytest.fixture
def minimal_test_dataset(tmp_path):
    """
    Create minimal test dataset (3 recordings).

    Structure:
    test_dataset/
    ├── predictions/
    │   ├── file_001_probs.npy
    │   ├── file_002_probs.npy
    │   └── file_003_probs.npy
    ├── reference/
    │   ├── file_001.csv_bi
    │   ├── file_002.csv_bi
    │   └── file_003.csv_bi
    └── metadata.json
    """
    # Create synthetic data
    dataset_dir = tmp_path / "test_dataset"
    dataset_dir.mkdir()

    # ... create files ...

    return dataset_dir
```

### Real Data for Integration Tests

**For full integration testing**, use actual dev set:
- Location: `cache/tusz_mmap/dev/`
- Files: 1832 recordings
- Ground truth: `/data_ext4/tusz/edf/dev/*/*.csv_bi`

---

## CI/CD Test Configuration

### GitHub Actions Workflow

```yaml
# .github/workflows/test_evaluation.yml
name: Test Evaluation Pipeline

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"

      - name: Run unit tests (fast)
        run: |
          pytest -v -m "not integration and not gpu" \
            tests/unit/eval/ \
            --cov=src/brain_brr/eval \
            --cov-report=term-missing

      - name: Check coverage
        run: |
          coverage report --fail-under=90
```

---

## Acceptance Criteria (Complete)

### Component 1: CSVBIConverter

- [ ] All 10 unit tests pass
- [ ] Code coverage ≥ 95%
- [ ] Can convert 1000 recordings in < 60s
- [ ] Output .csv_bi files validate in nedc-bench parser
- [ ] Error messages are clear and actionable
- [ ] Logging provides visibility

### Component 2: NEDCScorer

- [ ] All 8 unit tests + 1 integration test pass
- [ ] Code coverage ≥ 90%
- [ ] Can score 1000 file pairs in < 30s (overlap)
- [ ] All 5 algorithms work correctly
- [ ] Integration test with nedc-bench sample data passes
- [ ] Error messages indicate format issues clearly

### Component 3: ModelEvaluator

- [ ] All 4 integration tests pass (with mocking)
- [ ] Full end-to-end test on dev set passes (manual)
- [ ] Can evaluate full dev set in < 30 min
- [ ] CLI interface works (`--help`, all arguments)
- [ ] Generates publication-ready tables
- [ ] JSON output is well-formatted

### Overall Pipeline

- [ ] Total 23 tests pass
- [ ] Overall coverage ≥ 90%
- [ ] No warnings in production logs (INFO level)
- [ ] Performance targets met for all operations
- [ ] Documentation complete with examples

---

## Next Steps

**Week 1-3**: Implement components with TDD
**Week 4**: Full integration testing and production evaluation

**Testing is the foundation!** Write tests FIRST, then make them pass! 🧪
