# Component 2: NEDCScorer - Technical Specification

**File**: `src/brain_brr/eval/nedc_wrapper.py` (~100 lines)

**Purpose**: Direct Python integration with nedc-bench for official NEDC v6.0.0 scoring

**Status**: Specification complete - Ready for TDD implementation

**Key Innovation**: ✨ **NO Docker! NO subprocess! Just `sys.path.insert()` + Python imports!** ✨

---

## Architecture: Direct Python Import

```python
# ULTRA GOD BANGER: Direct Python import (NO Docker!)
import sys
from pathlib import Path

# Add nedc-bench to Python path
NEDC_BENCH_PATH = Path(__file__).resolve().parents[3] / "reference_repos" / "nedc-bench" / "src"
sys.path.insert(0, str(NEDC_BENCH_PATH))

# Direct imports - FAST! NO overhead!
from nedc_bench.orchestration import DualPipelineOrchestrator, BetaPipeline
from nedc_bench.models.annotations import AnnotationFile
```

**Why this is PERFECT**:
- 🚀 **FAST**: Pure Python, no Docker/subprocess overhead
- 💪 **SIMPLE**: Just add to path + import
- 🧠 **CLEAN**: All in ONE process
- 📊 **RICH**: Full API access, structured data

---

## Type Definitions

```python
from typing import Literal

AlgorithmType = Literal["overlap", "taes", "dp", "epoch", "ira", "all"]
PipelineType = Literal["alpha", "beta", "dual"]
```

---

## Class: NEDCMetrics

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class NEDCMetrics:
    """
    Structured NEDC evaluation metrics.

    All metrics directly from NEDC v6.0.0 scoring.
    """
    algorithm: str
    sensitivity_at_10FA_24h: float
    sensitivity_at_5FA_24h: float
    sensitivity_at_1FA_24h: float
    taes_score: Optional[float]  # Only for "taes" algorithm
    f1: float
    precision: float
    recall: float
    tp: int
    fp: int
    fn: int
    total_seizure_duration_sec: float
    total_recording_duration_sec: float
```

---

## Class: NEDCScorer

### Responsibilities
1. Import nedc-bench Python modules via sys.path
2. Use BetaPipeline for evaluation with .csv_bi files
3. Compute FA-targeted sensitivities from raw counts (hits/misses/FAs)
4. Parse NEDC results into structured metrics
5. Handle all 5 scoring algorithms (overlap recommended)

### Dependencies
- nedc-bench at `reference_repos/nedc-bench/` (must exist!)

---

## Method Signatures

### `__init__()`

```python
def __init__(self):
    """
    Initialize NEDC scorer with dual pipeline.

    Raises:
        ImportError: If nedc-bench not found at reference_repos/
        RuntimeError: If nedc-bench import fails (dependency issues)

    Behavior:
    1. Verify NEDC_BENCH_PATH exists
    2. Import DualPipelineOrchestrator and BetaPipeline from nedc-bench
    3. Initialize pipeline instances (beta for evaluation)
    """
```

---

### `score_predictions()`

```python
def score_predictions(
    self,
    reference_dir: Path,
    hypothesis_dir: Path,
    algorithm: AlgorithmType = "overlap",
    pipeline: PipelineType = "beta",
) -> NEDCMetrics:
    """
    Score predictions using NEDC-BENCH Python API.

    Args:
        reference_dir: Directory with ground truth .csv_bi files
        hypothesis_dir: Directory with model prediction .csv_bi files
        algorithm: NEDC scoring algorithm to use
            - "overlap": Any-overlap detection (recommended for seizures)
            - "taes": Time-Aligned Event Scoring
            - "dp": Dynamic Programming alignment
            - "epoch": 250ms epoch-based sampling
            - "ira": Inter-Rater Agreement (Cohen's κ)
            - "all": Run all algorithms (returns list)
        pipeline: Which NEDC pipeline to use
            - "beta": Modern reimplementation (faster, recommended)
            - "alpha": Legacy wrapper (100% parity with NEDC v6.0.0)
            - "dual": Both (for parity validation)

    Returns:
        NEDCMetrics object with official NEDC scores

    Raises:
        FileNotFoundError: If reference_dir or hypothesis_dir doesn't exist
        ValueError: If no .csv_bi files found in directories
        ValueError: If file count mismatch (ref vs hyp)
        RuntimeError: If NEDC scoring fails (invalid CSV_BI format)

    Behavior:
    1. Validate directories exist and contain .csv_bi files
    2. Match reference and hypothesis files by filename
    3. Call BetaPipeline.evaluate() (returns hits, misses, false_alarms, durations)
    4. Compute FA-targeted sensitivities from raw counts:
       - Extract total_recording_duration from results
       - For each FA target (10, 5, 1 FA/24h):
         * This is NOT provided by NEDC API - we need to compute it
         * May require threshold sweeping OR use existing model thresholds
         * sensitivity = hits / (hits + misses)
    5. Parse raw counts into NEDCMetrics structure
    6. Log summary statistics

    CRITICAL NOTE: BetaPipeline returns raw counts (hits, misses, false_alarms),
    NOT FA-targeted sensitivities. The NEDCMetrics fields like sensitivity_at_10FA_24h
    must be computed separately, potentially via threshold sweeping similar to what
    src/brain_brr/eval/helpers/false_alarm.py:find_threshold_for_fa_target() does.

    Performance:
    - Overlap algorithm: ~100-200 file pairs per second
    - TAES algorithm: ~50-100 file pairs per second
    - Scales linearly with number of recordings

    Example:
        scorer = NEDCScorer()
        metrics = scorer.score_predictions(
            reference_dir=Path("csv_bi/reference/"),
            hypothesis_dir=Path("csv_bi/hypothesis/"),
            algorithm="overlap"
        )
        print(f"Sensitivity@10FA: {metrics.sensitivity_at_10FA_24h:.2f}%")
    """
```

---

### `score_predictions_batch()`

```python
def score_predictions_batch(
    self,
    reference_dir: Path,
    hypothesis_dir: Path,
    algorithms: List[AlgorithmType],
) -> Dict[str, NEDCMetrics]:
    """
    Score predictions with multiple algorithms.

    Args:
        reference_dir: Ground truth .csv_bi directory
        hypothesis_dir: Predictions .csv_bi directory
        algorithms: List of algorithms to run

    Returns:
        Dict mapping algorithm name to NEDCMetrics

    Example:
        results = scorer.score_predictions_batch(
            ref_dir, hyp_dir,
            algorithms=["overlap", "taes", "epoch"]
        )
        print(f"Overlap: {results['overlap'].sensitivity_at_10FA_24h:.2f}%")
        print(f"TAES: {results['taes'].taes_score:.4f}")
    """
```

---

### `validate_csv_bi_format()`

```python
def validate_csv_bi_format(
    self,
    csv_bi_path: Path,
) -> bool:
    """
    Validate CSV_BI file format using nedc-bench parser.

    Args:
        csv_bi_path: Path to .csv_bi file to validate

    Returns:
        True if valid, False otherwise

    Behavior:
    1. Attempt to parse with AnnotationFile.from_csv_bi()
    2. Check for required headers
    3. Validate event format
    4. Return True if valid, False if any errors

    Use this to debug format conversion issues!
    """
```

---

## NEDC Algorithms Reference

### 1. Overlap (Recommended)
**Use case**: Seizure detection (standard for literature)

**Logic**: Any temporal overlap between reference and hypothesis events counts as TP

**Best for**: Events with variable durations (seizures)

### 2. TAES (Time-Aligned Event Scoring)
**Use case**: Precise event timing

**Logic**: Uses multi-overlap sequencing for complex event matching

**Best for**: Multiple overlapping events

### 3. DP (Dynamic Programming)
**Use case**: Optimal event alignment

**Logic**: Finds best match between reference and hypothesis using dynamic programming

**Best for**: Complex event sequences

### 4. Epoch (250ms sampling)
**Use case**: Sample-by-sample comparison

**Logic**: Divides timeline into 250ms epochs, compares labels

**Best for**: High-resolution temporal analysis

### 5. IRA (Inter-Rater Agreement)
**Use case**: Multi-class agreement

**Logic**: Cohen's κ for multi-class labels

**Best for**: Measuring annotator agreement

---

## Test Specifications

**File**: `tests/unit/eval/test_nedc_wrapper.py`

### Test 1: Initialization

```python
def test_init_success(self, scorer):
    """NEDCScorer initializes successfully with nedc-bench found"""
    assert scorer is not None
    assert hasattr(scorer, 'pipeline')

def test_init_nedc_bench_not_found(self, monkeypatch, tmp_path):
    """NEDCScorer raises ImportError if nedc-bench not found"""
    # Mock NEDC_BENCH_PATH to nonexistent location
    with pytest.raises(ImportError, match="NEDC-BENCH not found"):
        NEDCScorer()
```

### Test 2: Perfect Match Scoring

```python
def test_score_predictions_perfect_match(self, scorer, sample_csv_bi_ref, sample_csv_bi_hyp_perfect):
    """
    Score perfect predictions (100% match).

    Expected:
    - Sensitivity: 100%
    - Precision: 100%
    - F1: 1.0
    - TP: 2, FP: 0, FN: 0
    """
    metrics = scorer.score_predictions(
        reference_dir=sample_csv_bi_ref,
        hypothesis_dir=sample_csv_bi_hyp_perfect,
        algorithm="overlap",
    )

    assert metrics.algorithm == "overlap"
    assert metrics.sensitivity_at_10FA_24h == 100.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.tp == 2
    assert metrics.fp == 0
    assert metrics.fn == 0
```

### Test 3: Partial Match Scoring

```python
def test_score_predictions_partial_match(self, scorer, sample_csv_bi_ref, sample_csv_bi_hyp_partial):
    """
    Score partial predictions (1 TP, 1 FN, 1 FP).

    Reference: 10-30s, 100-150s
    Hypothesis: 10-30s, 200-220s

    Expected:
    - TP: 1, FN: 1, FP: 1
    - Sensitivity: 50%, Precision: 50%
    """
    metrics = scorer.score_predictions(
        reference_dir=sample_csv_bi_ref,
        hypothesis_dir=sample_csv_bi_hyp_partial,
        algorithm="overlap",
    )

    assert metrics.tp == 1
    assert metrics.fn == 1
    assert metrics.fp == 1
    assert abs(metrics.recall - 0.5) < 0.01
    assert abs(metrics.precision - 0.5) < 0.01
```

### Test 4: Multiple Algorithms

```python
def test_score_predictions_batch(self, scorer, sample_csv_bi_ref, sample_csv_bi_hyp_perfect):
    """Score with multiple algorithms simultaneously"""
    results = scorer.score_predictions_batch(
        reference_dir=sample_csv_bi_ref,
        hypothesis_dir=sample_csv_bi_hyp_perfect,
        algorithms=["overlap", "epoch", "ira"],
    )

    assert isinstance(results, dict)
    assert "overlap" in results
    assert "epoch" in results
    assert "ira" in results
```

### Test 5-6: Error Handling

```python
def test_score_predictions_dir_not_found(self, scorer, tmp_path):
    """NEDCScorer raises FileNotFoundError for missing directories"""

def test_score_predictions_no_files(self, scorer, tmp_path):
    """NEDCScorer raises ValueError if no .csv_bi files found"""
```

### Test 7: CSV_BI Validation

```python
def test_validate_csv_bi_format_valid(self, scorer, sample_csv_bi_ref):
    """validate_csv_bi_format() returns True for valid file"""
    csv_bi_file = list(sample_csv_bi_ref.glob("*.csv_bi"))[0]
    assert scorer.validate_csv_bi_format(csv_bi_file) is True

def test_validate_csv_bi_format_invalid(self, scorer, tmp_path):
    """validate_csv_bi_format() returns False for invalid file"""
    invalid_file = tmp_path / "invalid.csv_bi"
    invalid_file.write_text("This is not valid CSV_BI format!")
    assert scorer.validate_csv_bi_format(invalid_file) is False
```

### Test 8: Integration Test

```python
@pytest.mark.integration
def test_integration_with_nedc_bench_sample_data(self, scorer):
    """
    Integration test with nedc-bench sample data.

    Uses actual sample files from nedc-bench repository.
    Verifies we can call nedc-bench and get real results.
    """
    nedc_bench_root = Path("reference_repos/nedc-bench")
    sample_ref = nedc_bench_root / "data/csv_bi_parity/csv_bi_export_clean/ref"
    sample_hyp = nedc_bench_root / "data/csv_bi_parity/csv_bi_export_clean/hyp"

    if not sample_ref.exists() or not sample_hyp.exists():
        pytest.skip("nedc-bench sample data not found")

    metrics = scorer.score_predictions(
        reference_dir=sample_ref,
        hypothesis_dir=sample_hyp,
        algorithm="overlap",
    )

    # Should get real results (values from nedc-bench parity tests)
    assert metrics.tp > 0
    assert metrics.fp > 0
    assert 0 < metrics.sensitivity_at_10FA_24h < 100
```

---

## Test Fixtures

### sample_csv_bi_ref

```python
@pytest.fixture
def sample_csv_bi_ref(self, tmp_path):
    """
    Create sample reference .csv_bi file.

    Recording: test_001.csv_bi
    Duration: 300s
    Events: 10-30s, 100-150s
    """
    csv_bi = """version = csv_bi_v01.00.00
patient = test
session = s001
duration = 300.0000 secs

channel,start_time,stop_time,label,confidence
TERM,10.0000,30.0000,seiz,1.0
TERM,100.0000,150.0000,seiz,1.0
"""
    ref_dir = tmp_path / "reference"
    ref_dir.mkdir()
    ref_file = ref_dir / "test_001.csv_bi"
    ref_file.write_text(csv_bi)
    return ref_dir
```

### sample_csv_bi_hyp_perfect

```python
@pytest.fixture
def sample_csv_bi_hyp_perfect(self, tmp_path):
    """Hypothesis matching reference perfectly (100% TP)"""
    # Same as reference
```

### sample_csv_bi_hyp_partial

```python
@pytest.fixture
def sample_csv_bi_hyp_partial(self, tmp_path):
    """
    Hypothesis with partial match:
    - Detects 10-30s (TP)
    - Misses 100-150s (FN)
    - False alarm 200-220s (FP)
    """
```

---

## Acceptance Criteria

**NEDCScorer is complete when**:
- [ ] All 8 unit tests + 1 integration test pass
- [ ] Code coverage ≥ 90% for nedc_wrapper.py
- [ ] Integration test with real nedc-bench sample data passes
- [ ] Can score 1000 file pairs in < 30 seconds (overlap algorithm)
- [ ] Error messages clearly indicate format issues
- [ ] Logging provides visibility into scoring process
- [ ] Documentation includes example usage for all 5 algorithms

---

## Error Handling

| Error Condition | Exception | Message | Recovery |
|-----------------|-----------|---------|----------|
| nedc-bench not found | ImportError | "NEDC-BENCH not found at {path}. Clone from ..." | Fail fast |
| Reference dir not found | FileNotFoundError | "Reference directory not found: {path}" | Fail fast |
| No CSV_BI files | ValueError | "No .csv_bi files found in {path}" | Fail fast |
| File count mismatch | ValueError | "File count mismatch: {ref_count} ref vs {hyp_count} hyp" | Fail fast |
| NEDC scoring failed | RuntimeError | "NEDC scoring failed: {error}" | Fail fast |
| Invalid CSV_BI format | RuntimeError | "Invalid CSV_BI format in {file}: {error}" | Skip file, log error |

---

## Performance Target

| Operation | Target | Notes |
|-----------|--------|-------|
| Score 1 file pair (overlap) | < 30ms | Per-pair overhead |
| Score 1000 file pairs (overlap) | < 30s | Batch processing |
| Score 2000 file pairs (eval set) | < 60s | Full eval set |

---

## Implementation Notes

1. **Path resolution**: Use `Path(__file__).resolve().parents[3]` for relative path
2. **Import checking**: Verify nedc-bench imports work in `__init__()`
3. **Error messages**: Include full path to nedc-bench in ImportError
4. **Logging**: INFO for scoring progress, ERROR for failures

---

## Next Steps

**Week 2 Implementation**:
- Day 1-2: Write all 8+1 tests (TDD - tests fail initially)
- Day 3-4: Implement NEDCScorer (make tests pass)
- Day 5: Integration test with nedc-bench sample data

**Ready to begin after Week 1 complete!**
