# NEDC Evaluation - TDD Implementation Guide

**Purpose**: Step-by-step TDD implementation guide for NEDC integration

**Approach**: Test-Driven Development (write tests first, then implement)

**Total Effort**: ~150 lines code, ~500 lines tests, 2 weeks

**CRITICAL**: This guide uses ACTUAL codebase patterns (Click CLI, EvaluationRequest dataclass, file-level nedc-bench API)

---

## Phase 0: Prerequisites (Before Coding)

### 0.1: Verify TUSZ Eval Set Exists

**Location**: `/home/jj/proj/brain-go-brr-v2/data_ext4/tusz/edf/eval/`

```bash
# Verify eval set exists
ls /home/jj/proj/brain-go-brr-v2/data_ext4/tusz/edf/eval/

# Count EDF files
find /home/jj/proj/brain-go-brr-v2/data_ext4/tusz/edf/eval/ -name "*.edf" | wc -l
# Expected: ~2000 files

# Verify ground truth CSV_BI files exist (with # headers!)
find /home/jj/proj/brain-go-brr-v2/data_ext4/tusz/edf/eval/ -name "*.csv_bi" | head -1 | xargs head -10
# Expected output:
# # version = csv_v1.0.0
# # bname = aaaaaaghb_s009_t002
# # duration = 210.0000 secs
# # montage_file = nedc_eas_default_montage.txt
# #
# channel,start_time,stop_time,label,confidence
# TERM,0.0091,1.0009,bckg,1.0000
```

**File Structure**:
```
eval/
├── {patient}/           # e.g., aaaaaaaq
│   └── {session_year}/  # e.g., s006_2014
│       └── 01_tcp_ar/   # Montage type
│           ├── {file_id}.edf      # EEG data
│           ├── {file_id}.csv      # Labels (old format)
│           └── {file_id}.csv_bi   # Labels (NEDC format) ✅
```

### 0.2: Verify nedc-bench Exists

```bash
# Verify nedc-bench cloned
ls reference_repos/nedc-bench/src/nedc_bench/

# Check for key modules
ls reference_repos/nedc-bench/src/nedc_bench/orchestration/dual_pipeline.py
ls reference_repos/nedc-bench/src/nedc_bench/algorithms/overlap.py
ls reference_repos/nedc-bench/src/nedc_bench/models/annotations.py

# Verify sample data for integration test
ls reference_repos/nedc-bench/nedc_eeg_eval/v6.0.0/data/csv/ref | head
ls reference_repos/nedc-bench/nedc_eeg_eval/v6.0.0/data/csv/hyp | head
```

### 0.3: Extend build-cache CLI to Support Eval Split

**File**: `src/brain_brr/cli/cli.py`

**Current code** (line 207):
```python
@click.option("--split", type=click.Choice(["train", "dev"]), default="train")
```

**Change to**:
```python
@click.option("--split", type=click.Choice(["train", "dev", "eval"]), default="train")
```

**Test the change**:
```bash
# Should not error
python -m src build-cache --help
# Check that --split shows [train|dev|eval]
```

### 0.4: Preprocess Eval Set (Generate Cache)

**This takes 2-4 hours** - run in tmux!

```bash
# Start tmux session
tmux new -s preprocess-eval

# Run preprocessing
python -m src build-cache \
  --data-dir data_ext4/tusz/edf/eval/ \
  --cache-dir cache/tusz_mmap/eval/ \
  --split eval

# Detach: Ctrl+B then D
# Reattach: tmux attach -t preprocess-eval
```

**Verify cache created**:
```bash
# Check files created
ls cache/tusz_mmap/eval/
# Expected: _dataset_index.json + ~4000 NPY files (data + labels)

# Count data files
ls cache/tusz_mmap/eval/*_data.npy | wc -l
# Expected: ~2000 files

# Verify manifest
cat cache/tusz_mmap/eval/_dataset_index.json | head
```

**Prerequisites complete!** ✅

---

## Phase 1: Write Tests (TDD - Days 4-5 of Week 1)

### 1.1: Create Test File Structure

```bash
# Create test directories
mkdir -p tests/unit/eval
mkdir -p tests/integration/eval

# Create test files
touch tests/unit/eval/test_nedc_wrapper.py
touch tests/integration/eval/test_nedc_integration.py
```

### 1.2: Write NEDCScorer Unit Tests (8 tests)

**File**: `tests/unit/eval/test_nedc_wrapper.py`

**Complete test file** (copy this - includes # headers in fixtures!):

```python
"""
Unit tests for NEDCScorer wrapper.

Tests NEDCScorer class that provides Python API to nedc-bench (FILE-LEVEL API).
"""

import pytest
from pathlib import Path
import tempfile

from src.brain_brr.eval.nedc_wrapper import NEDCScorer, NEDCMetrics


class TestNEDCScorerInit:
    """Test NEDCScorer initialization"""

    def test_init_success(self):
        """NEDCScorer initializes successfully with nedc-bench found"""
        scorer = NEDCScorer()
        assert scorer is not None
        assert hasattr(scorer, 'beta')  # BetaPipeline instance

    def test_init_nedc_bench_not_found(self, monkeypatch, tmp_path):
        """NEDCScorer raises ImportError if nedc-bench not found"""
        import src.brain_brr.eval.nedc_wrapper as wrapper_module
        fake_path = tmp_path / "nonexistent"
        monkeypatch.setattr(wrapper_module, "NEDC_BENCH_PATH", fake_path)

        with pytest.raises(ImportError, match="NEDC-BENCH not found"):
            NEDCScorer()


class TestNEDCScorerScoring:
    """Test NEDCScorer scoring functionality (FILE-LEVEL API)"""

    @pytest.fixture
    def scorer(self):
        """Create NEDCScorer instance"""
        return NEDCScorer()

    @pytest.fixture
    def sample_csv_bi_ref(self, tmp_path):
        """
        Create sample reference .csv_bi file WITH # HEADERS.

        Recording: test_001.csv_bi
        Duration: 300s
        Events: 10-30s, 100-150s
        """
        # CRITICAL: Must include # headers to match TUSZ format!
        csv_bi = """# version = csv_v1.0.0
# bname = test_s001_t000
# duration = 300.0000 secs
# montage_file = nedc_eas_default_montage.txt
#
channel,start_time,stop_time,label,confidence
TERM,10.0000,30.0000,seiz,1.0
TERM,100.0000,150.0000,seiz,1.0
"""
        ref_dir = tmp_path / "reference"
        ref_dir.mkdir()
        ref_file = ref_dir / "test_001.csv_bi"
        ref_file.write_text(csv_bi)
        return ref_dir

    @pytest.fixture
    def sample_csv_bi_hyp_perfect(self, tmp_path):
        """Hypothesis matching reference perfectly (100% TP)"""
        csv_bi = """# version = csv_v1.0.0
# bname = test_s001_t000
# duration = 300.0000 secs
# montage_file = nedc_eas_default_montage.txt
#
channel,start_time,stop_time,label,confidence
TERM,10.0000,30.0000,seiz,1.0
TERM,100.0000,150.0000,seiz,1.0
"""
        hyp_dir = tmp_path / "hypothesis"
        hyp_dir.mkdir()
        hyp_file = hyp_dir / "test_001.csv_bi"
        hyp_file.write_text(csv_bi)
        return hyp_dir

    @pytest.fixture
    def sample_csv_bi_hyp_partial(self, tmp_path):
        """
        Hypothesis with partial match:
        - Detects 10-30s (TP)
        - Misses 100-150s (FN)
        - False alarm 200-220s (FP)
        """
        csv_bi = """# version = csv_v1.0.0
# bname = test_s001_t000
# duration = 300.0000 secs
# montage_file = nedc_eas_default_montage.txt
#
channel,start_time,stop_time,label,confidence
TERM,10.0000,30.0000,seiz,1.0
TERM,200.0000,220.0000,seiz,1.0
"""
        hyp_dir = tmp_path / "hypothesis"
        hyp_dir.mkdir()
        hyp_file = hyp_dir / "test_001.csv_bi"
        hyp_file.write_text(csv_bi)
        return hyp_dir

    def test_score_predictions_perfect_match(
        self, scorer, sample_csv_bi_ref, sample_csv_bi_hyp_perfect
    ):
        """
        Score perfect predictions (100% match).

        Expected:
        - Recall: 100% (2/2 seizures detected)
        - Precision: 100% (2/2 detections correct)
        - F1: 1.0
        - TP: 2, FP: 0, FN: 0
        """
        metrics = scorer.score_predictions(
            reference_dir=sample_csv_bi_ref,
            hypothesis_dir=sample_csv_bi_hyp_perfect,
            algorithm="overlap",
        )

        assert metrics.algorithm == "overlap"
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1 == 1.0
        assert metrics.tp == 2
        assert metrics.fp == 0
        assert metrics.fn == 0

    def test_score_predictions_partial_match(
        self, scorer, sample_csv_bi_ref, sample_csv_bi_hyp_partial
    ):
        """
        Score partial predictions (1 TP, 1 FN, 1 FP).

        Reference: 10-30s, 100-150s
        Hypothesis: 10-30s, 200-220s

        Expected:
        - TP: 1, FN: 1, FP: 1
        - Recall: 50%, Precision: 50%
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

    def test_score_predictions_batch(
        self, scorer, sample_csv_bi_ref, sample_csv_bi_hyp_perfect
    ):
        """Score with multiple algorithms simultaneously"""
        results = scorer.score_predictions_batch(
            reference_dir=sample_csv_bi_ref,
            hypothesis_dir=sample_csv_bi_hyp_perfect,
            algorithms=["overlap", "epoch"],
        )

        assert isinstance(results, dict)
        assert "overlap" in results
        assert "epoch" in results


class TestNEDCScorerErrors:
    """Test error handling"""

    @pytest.fixture
    def scorer(self):
        return NEDCScorer()

    def test_score_predictions_dir_not_found(self, scorer, tmp_path):
        """NEDCScorer raises FileNotFoundError for missing directories"""
        nonexistent = tmp_path / "nonexistent"

        with pytest.raises(FileNotFoundError):
            scorer.score_predictions(
                reference_dir=nonexistent,
                hypothesis_dir=nonexistent,
                algorithm="overlap",
            )

    def test_score_predictions_no_files(self, scorer, tmp_path):
        """NEDCScorer raises ValueError if no .csv_bi files found"""
        empty_ref = tmp_path / "empty_ref"
        empty_hyp = tmp_path / "empty_hyp"
        empty_ref.mkdir()
        empty_hyp.mkdir()

        with pytest.raises(ValueError, match="No .csv_bi files found"):
            scorer.score_predictions(
                reference_dir=empty_ref,
                hypothesis_dir=empty_hyp,
                algorithm="overlap",
            )


class TestNEDCScorerValidation:
    """Test CSV_BI validation"""

    @pytest.fixture
    def scorer(self):
        return NEDCScorer()

    def test_validate_csv_bi_format_valid(self, scorer, tmp_path):
        """validate_csv_bi_format() returns True for valid file"""
        valid_csv_bi = """# version = csv_v1.0.0
# bname = test_s001_t000
# duration = 300.0000 secs
# montage_file = nedc_eas_default_montage.txt
#
channel,start_time,stop_time,label,confidence
TERM,10.0000,30.0000,seiz,1.0
"""
        csv_bi_file = tmp_path / "valid.csv_bi"
        csv_bi_file.write_text(valid_csv_bi)

        assert scorer.validate_csv_bi_format(csv_bi_file) is True

    def test_validate_csv_bi_format_invalid(self, scorer, tmp_path):
        """validate_csv_bi_format() returns False for invalid file"""
        invalid_file = tmp_path / "invalid.csv_bi"
        invalid_file.write_text("This is not valid CSV_BI format!")

        assert scorer.validate_csv_bi_format(invalid_file) is False
```

**Run tests (they should ALL FAIL)**:
```bash
.venv/bin/pytest tests/unit/eval/test_nedc_wrapper.py -v
# Expected: 8 failed (nedc_wrapper.py doesn't exist yet!)
```

### 1.3: Write NEDC Integration Test (1 test)

**File**: `tests/integration/eval/test_nedc_integration.py`

```python
"""
Integration test for NEDCScorer with real nedc-bench sample data.

Uses actual sample files from nedc-bench repository to verify:
1. We can import nedc-bench successfully
2. We can call BetaPipeline.evaluate_*() methods
3. We get real results back
"""

import pytest
from pathlib import Path

from src.brain_brr.eval.nedc_wrapper import NEDCScorer


@pytest.mark.integration
class TestNEDCBenchIntegration:
    """Integration tests with nedc-bench"""

    def test_integration_with_nedc_bench_sample_data(self):
        """
        Integration test with nedc-bench sample data.

        Uses actual sample files from nedc-bench repository.
        Verifies we can call nedc-bench and get real results.
        """
        # Initialize scorer
        scorer = NEDCScorer()

        # Find nedc-bench sample data (ships with repo)
        nedc_bench_root = Path("reference_repos/nedc-bench")
        sample_ref = nedc_bench_root / "nedc_eeg_eval/v6.0.0/data/csv/ref"
        sample_hyp = nedc_bench_root / "nedc_eeg_eval/v6.0.0/data/csv/hyp"

        # Skip if sample data not found
        if not sample_ref.exists() or not sample_hyp.exists():
            pytest.skip("nedc-bench sample data not found")

        # Run scoring
        metrics = scorer.score_predictions(
            reference_dir=sample_ref,
            hypothesis_dir=sample_hyp,
            algorithm="overlap",
        )

        # Should get real results (values from nedc-bench parity tests)
        assert metrics.tp > 0
        assert metrics.fp >= 0
        assert 0 < metrics.f1 < 1
```

**Run integration test (should FAIL)**:
```bash
.venv/bin/pytest tests/integration/eval/test_nedc_integration.py -v -m integration
# Expected: 1 failed (nedc_wrapper.py doesn't exist yet!)
```

**Phase 1 Complete!** ✅ All 9 tests written and failing (as expected for TDD)

---

## Phase 2: Implement NEDCScorer (Days 1-2 of Week 2)

### 2.1: Create nedc_wrapper.py File

```bash
# Create eval module directory
mkdir -p src/brain_brr/eval

# Create __init__.py
touch src/brain_brr/eval/__init__.py

# Create nedc_wrapper.py
touch src/brain_brr/eval/nedc_wrapper.py
```

### 2.2: Implement NEDCScorer Class

**File**: `src/brain_brr/eval/nedc_wrapper.py`

**Complete implementation** (~150 lines - includes file-level loop):

```python
"""
NEDCScorer - Direct Python integration with nedc-bench for NEDC v6.0.0 scoring.

NO Docker! NO subprocess! Just sys.path.insert() + Python imports!

CRITICAL: nedc-bench has FILE-LEVEL API, not directory-level.
Must loop over matched file pairs and accumulate counts.
"""

import sys
from pathlib import Path
from typing import Literal, List, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Add nedc-bench to Python path (relative to this file)
NEDC_BENCH_PATH = Path(__file__).resolve().parents[3] / "reference_repos" / "nedc-bench" / "src"


@dataclass
class NEDCMetrics:
    """
    Structured NEDC evaluation metrics.

    CRITICAL: NEDC API only provides counts (tp/fp/fn).
    We compute precision/recall/f1 from counts.
    sensitivity_at_*FA must be computed separately (threshold sweep).
    """
    algorithm: str
    tp: int                               # True positives (event count)
    fp: int                               # False positives (event count)
    fn: int                               # False negatives (event count)
    precision: float                      # TP / (TP + FP)
    recall: float                         # TP / (TP + FN)
    f1: float                             # 2 * (P * R) / (P + R)
    total_recording_duration_sec: float   # Sum of all recording durations
    num_files: int                        # Number of file pairs scored


AlgorithmType = Literal["overlap", "taes", "dp", "epoch", "ira"]


class NEDCScorer:
    """
    Direct Python integration with nedc-bench for official NEDC v6.0.0 scoring.

    Uses sys.path.insert() to import nedc-bench modules directly.
    NO Docker overhead, NO subprocess complexity!

    CRITICAL: nedc-bench has FILE-LEVEL API (evaluate_overlap(ref_file, hyp_file)).
    We must loop over file pairs and accumulate counts.
    """

    def __init__(self):
        """
        Initialize NEDC scorer with BetaPipeline.

        Raises:
            ImportError: If nedc-bench not found at reference_repos/
            RuntimeError: If nedc-bench import fails
        """
        # Verify nedc-bench exists
        if not NEDC_BENCH_PATH.exists():
            raise ImportError(
                f"NEDC-BENCH not found at {NEDC_BENCH_PATH}. "
                f"Clone from https://github.com/Clarity-Digital-Twin/nedc-bench"
            )

        # Add to path
        sys.path.insert(0, str(NEDC_BENCH_PATH))

        # Import nedc-bench modules
        try:
            from nedc_bench.orchestration.dual_pipeline import BetaPipeline
            from nedc_bench.models.annotations import AnnotationFile
            self.BetaPipeline = BetaPipeline
            self.AnnotationFile = AnnotationFile
            self.beta = BetaPipeline()
            logger.info("NEDC-BENCH imported successfully")
        except ImportError as e:
            raise RuntimeError(f"Failed to import nedc-bench: {e}")

    def score_predictions(
        self,
        reference_dir: Path,
        hypothesis_dir: Path,
        algorithm: AlgorithmType = "overlap",
    ) -> NEDCMetrics:
        """
        Score predictions using NEDC-BENCH Python API (FILE-LEVEL).

        CRITICAL: nedc-bench API is FILE-LEVEL, not directory-level!
        We loop over matched file pairs and accumulate counts.

        Args:
            reference_dir: Directory with ground truth .csv_bi files
            hypothesis_dir: Directory with model prediction .csv_bi files
            algorithm: NEDC scoring algorithm (overlap recommended)

        Returns:
            NEDCMetrics with aggregated NEDC scores

        Raises:
            FileNotFoundError: If directories don't exist
            ValueError: If no .csv_bi files found
            RuntimeError: If NEDC scoring fails
        """
        # Validate directories exist
        if not reference_dir.exists():
            raise FileNotFoundError(f"Reference directory not found: {reference_dir}")
        if not hypothesis_dir.exists():
            raise FileNotFoundError(f"Hypothesis directory not found: {hypothesis_dir}")

        # Find CSV_BI files
        ref_files = sorted(reference_dir.glob("*.csv_bi"))
        hyp_files = sorted(hypothesis_dir.glob("*.csv_bi"))

        if len(ref_files) == 0:
            raise ValueError(f"No .csv_bi files found in reference directory: {reference_dir}")
        if len(hyp_files) == 0:
            raise ValueError(f"No .csv_bi files found in hypothesis directory: {hypothesis_dir}")

        # Match files by name
        ref_dict = {f.stem: f for f in ref_files}
        hyp_dict = {f.stem: f for f in hyp_files}
        common_files = set(ref_dict.keys()) & set(hyp_dict.keys())

        if len(common_files) == 0:
            raise ValueError("No matching file pairs found between reference and hypothesis")

        logger.info(f"Scoring {len(common_files)} file pairs with '{algorithm}' algorithm...")

        # Accumulate counts across all file pairs
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_duration = 0.0

        for file_id in sorted(common_files):
            ref_file = ref_dict[file_id]
            hyp_file = hyp_dict[file_id]

            # Call nedc-bench FILE-LEVEL API
            if algorithm == "overlap":
                result = self.beta.evaluate_overlap(ref_file, hyp_file)
            elif algorithm == "taes":
                result = self.beta.evaluate_taes(ref_file, hyp_file)
            elif algorithm == "dp":
                result = self.beta.evaluate_dp(ref_file, hyp_file)
            elif algorithm == "epoch":
                result = self.beta.evaluate_epoch(ref_file, hyp_file)
            elif algorithm == "ira":
                result = self.beta.evaluate_ira(ref_file, hyp_file)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

            # Accumulate counts (OverlapResult has total_hits, total_misses, total_false_alarms)
            total_tp += result.total_hits
            total_fp += result.total_false_alarms
            total_fn += result.total_misses

            # Accumulate duration from AnnotationFile
            ref_ann = self.AnnotationFile.from_csv_bi(ref_file)
            total_duration += ref_ann.duration

        # Compute metrics from aggregated counts
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        logger.info(f"NEDC scoring complete: TP={total_tp}, FP={total_fp}, FN={total_fn}")
        logger.info(f"Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

        return NEDCMetrics(
            algorithm=algorithm,
            tp=total_tp,
            fp=total_fp,
            fn=total_fn,
            precision=precision,
            recall=recall,
            f1=f1,
            total_recording_duration_sec=total_duration,
            num_files=len(common_files),
        )

    def score_predictions_batch(
        self,
        reference_dir: Path,
        hypothesis_dir: Path,
        algorithms: List[AlgorithmType],
    ) -> Dict[str, NEDCMetrics]:
        """Score predictions with multiple algorithms."""
        results = {}
        for algorithm in algorithms:
            logger.info(f"Scoring with algorithm: {algorithm}")
            results[algorithm] = self.score_predictions(reference_dir, hypothesis_dir, algorithm)
        return results

    def validate_csv_bi_format(self, csv_bi_path: Path) -> bool:
        """
        Validate CSV_BI file format using nedc-bench parser.

        Returns:
            True if valid, False otherwise
        """
        try:
            ann = self.AnnotationFile.from_csv_bi(str(csv_bi_path))
            return True
        except Exception as e:
            logger.error(f"Invalid CSV_BI format in {csv_bi_path}: {e}")
            return False
```

**Run tests (should PASS now!)**:
```bash
# Run unit tests
.venv/bin/pytest tests/unit/eval/test_nedc_wrapper.py -v
# Expected: 8 passed

# Run integration test
.venv/bin/pytest tests/integration/eval/test_nedc_integration.py -v -m integration
# Expected: 1 passed

# Check coverage
.venv/bin/pytest tests/unit/eval/test_nedc_wrapper.py --cov=src/brain_brr/eval/nedc_wrapper --cov-report=term
# Expected: ≥ 90% coverage
```

**Phase 2 Complete!** ✅ NEDCScorer implemented and all tests passing

---

## Phase 3: Extend CLI + run_evaluation Service (Day 3 of Week 2)

### 3.0: Wire CLI flag and request payload

**File**: `src/brain_brr/cli/cli.py`

1. Add the NEDC flag to the Click command:
   ```python
   @cli.command("evaluate")
   @click.argument("checkpoint_path", type=click.Path(exists=False, path_type=Path))
   @click.argument("data_path", type=click.Path(exists=False, path_type=Path))
   ...
   @click.option(
       "--nedc-score",
       is_flag=True,
       help="Run NEDC v6.0.0 scoring via nedc-bench after inference",
   )
   def evaluate(
       checkpoint_path: Path,
       data_path: Path,
       config: Path | None,
       device: str,
       output_json: Path | None,
       output_csv_bi: Path | None,
       dry_run: bool,
       nedc_score: bool,
   ) -> None:
   ```

2. When constructing the request, pass the new flag:
   ```python
       request = EvaluationRequest(
           checkpoint_path=checkpoint_path,
           data_path=data_path,
           config_path=config,
           device=device,
           output_json=output_json,
           output_csv_bi=output_csv_bi,
           nedc_score=nedc_score,
       )
   ```

**File**: `src/brain_brr/cli/services/evaluation.py`

- Extend the dataclass:
  ```python
  @dataclass
  class EvaluationRequest:
      ...
      output_csv_bi: Path | None
      nedc_score: bool = False
  ```

### 3.1: Prepare evaluation outputs and persist predictions

**File**: `src/brain_brr/cli/services/evaluation.py`

1. Update imports at the top of the file:
   ```python
   # Add new imports (add to existing import section):
   import logging
   import shutil

   import numpy as np
   import torch  # Already imported at line 12

   # Extend existing import (line 19):
   # FROM: from src.brain_brr.events import SeizureEvent
   # TO:   from src.brain_brr.events import SeizureEvent, calculate_event_confidence
   ```

   Then add logger after imports:
   ```python
   logger = logging.getLogger(__name__)
   ```

2. In `run_evaluation()`:
   - Capture the EDF list returned from `create_dataloader`.
   - Ensure predictions are saved to a dedicated evaluation directory before the call to `validate_epoch`.
   - Invoke a helper to attach NEDC metrics after inference when `request.nedc_score` is true.

   ```python
   def run_evaluation(request: EvaluationRequest) -> EvaluationResult:
       ...
       checkpoint, cfg = load_checkpoint_and_config(request.checkpoint_path, request.config_path)

       # Ensure evaluation outputs land in a dedicated directory
       eval_output_dir = (
           request.output_json.parent
           if request.output_json is not None
           else Path(cfg.experiment.output_dir) / "evaluation" / request.data_path.name
       )
       eval_output_dir = Path(eval_output_dir)
       eval_output_dir.mkdir(parents=True, exist_ok=True)
       cfg.experiment.output_dir = eval_output_dir
       cfg.evaluation.save_predictions = True

       model = SeizureDetector.from_config(cfg.model)
       ...

       dataloader, edf_files = create_dataloader(request.data_path, cfg, device)
       ...
       metrics = validate_epoch(
           model,
           dataloader,
           cfg.postprocessing,
           ...
           output_dir=cfg.experiment.output_dir,
       )

       if request.output_json:
           _export_metrics_json(metrics, request, device)

       if request.output_csv_bi:
           _export_events_csv_bi(model, dataloader, metrics, cfg, device, request.output_csv_bi)

       if request.nedc_score:
           _attach_nedc_metrics(request, cfg, metrics, edf_files, eval_output_dir)

       return EvaluationResult(...)
   ```

### 3.2: Generate CSV_BI hypotheses and score with nedc-bench

Add the helper to the bottom of `evaluation.py` (after existing helpers):

```python
def _attach_nedc_metrics(
    request: EvaluationRequest,
    cfg: Config,
    metrics: dict[str, Any],
    edf_files: list[Path],
    eval_output_dir: Path,
) -> None:
    """Convert predictions to CSV_BI, copy references, and run NEDC scoring."""
    predictions_dir = eval_output_dir / "predictions"
    if not predictions_dir.exists():
        logger.warning("Predictions directory missing (%s); skipping NEDC scoring", predictions_dir)
        return

    reference_dir = eval_output_dir / "nedc" / "reference"
    hypothesis_dir = eval_output_dir / "nedc" / "hypothesis"
    reference_dir.mkdir(parents=True, exist_ok=True)
    hypothesis_dir.mkdir(parents=True, exist_ok=True)

    try:
        from src.brain_brr.eval.nedc_wrapper import NEDCScorer
    except Exception as exc:
        logger.warning("nedc-bench unavailable (%s); skipping NEDC scoring", exc)
        return

    scorer = NEDCScorer()
    prepared_pairs = 0

    for edf_path in edf_files:
        file_id = edf_path.stem
        probs_path = predictions_dir / f"{file_id}_probs.npy"

        if not probs_path.exists():
            logger.warning("No predictions saved for %s; skipping", file_id)
            continue

        ref_csv = edf_path.with_suffix(".csv_bi")
        if not ref_csv.exists():
            logger.warning("Reference CSV_BI missing for %s; skipping", file_id)
            continue

        probs = np.load(probs_path)
        probs_tensor = torch.from_numpy(probs).unsqueeze(0)
        pred_event_ranges = batch_probs_to_events(
            probs_tensor,
            cfg.postprocessing,
            cfg.data.sampling_rate,
        )[0]

        events: list[SeizureEvent] = []
        for start_s, end_s in pred_event_ranges:
            event = SeizureEvent(start_s=start_s, end_s=end_s)
            event.confidence = calculate_event_confidence(
                probs,
                event,
                sampling_rate=cfg.data.sampling_rate,
                method="mean",
            )
            events.append(event)

        patient_id, recording_suffix = (file_id.split("_", 1) + ["unknown"])[:2]
        export_csv_bi(
            events,
            hypothesis_dir / f"{file_id}.csv_bi",
            patient_id=patient_id,
            recording_id=recording_suffix,
            duration_s=len(probs) / cfg.data.sampling_rate,
        )

        shutil.copy2(ref_csv, reference_dir / f"{file_id}.csv_bi")
        prepared_pairs += 1

    if prepared_pairs == 0:
        logger.warning("No prediction/reference pairs prepared for NEDC scoring")
        return

    nedc_metrics = scorer.score_predictions(reference_dir, hypothesis_dir, algorithm="overlap")
    duration_hours = nedc_metrics.total_recording_duration_sec / 3600.0
    fa_per_24h = (nedc_metrics.fp / duration_hours) * 24.0 if duration_hours > 0 else 0.0

    metrics["nedc_overlap"] = {
        "tp": nedc_metrics.tp,
        "fp": nedc_metrics.fp,
        "fn": nedc_metrics.fn,
        "precision": nedc_metrics.precision,
        "recall": nedc_metrics.recall,
        "f1": nedc_metrics.f1,
        "fa_per_24h": fa_per_24h,
        "num_files": nedc_metrics.num_files,
    }

    logger.info(
        "NEDC overlap: TP=%s FP=%s FN=%s F1=%.3f FA/24h=%.2f (files=%s)",
        nedc_metrics.tp,
        nedc_metrics.fp,
        nedc_metrics.fn,
        nedc_metrics.f1,
        fa_per_24h,
        nedc_metrics.num_files,
    )
```

### 3.3: Verify CLI + service end-to-end

```bash
# Re-run unit + integration suites
.venv/bin/pytest tests/unit/eval/test_nedc_wrapper.py -v
.venv/bin/pytest tests/integration/eval/test_nedc_integration.py -v -m integration

# Exercise CLI on dev split (fast sanity check)
python -m src evaluate \
  results/local_fla_training/checkpoints/best.pt \
  data_ext4/tusz/edf/dev/ \
  --output-json results/test_nedc_integration/metrics.json \
  --nedc-score

cat results/test_nedc_integration/metrics.json | jq .nedc_overlap
```

**Phase 3 Complete!** ✅ CLI flag + service updated, NEDC metrics attached to evaluation results

---

## Final Step: Run Baseline Evaluation on Eval Set (Day 5 of Week 2)

```bash
# Full evaluation on eval set with NEDC scoring
python -m src evaluate \
  results/local_fla_training/checkpoints/best.pt \
  data_ext4/tusz/edf/eval/ \
  --output-json results/eval_baseline/metrics.json

# View results
cat results/eval_baseline/metrics.json | jq .nedc_overlap

# Expected output:
# {
#   "tp": 145,
#   "fp": 198,
#   "fn": 456,
#   "precision": 0.42,
#   "recall": 0.24,
#   "f1": 0.31,
#   "num_files": 2000
# }
```

---

## Success Metrics

**All phases complete when**:
- [x] Phase 0: Eval cache built, nedc-bench verified
- [x] Phase 1: 9 tests written (8 unit + 1 integration)
- [x] Phase 2: NEDCScorer implemented, all tests passing, ≥90% coverage
- [x] Phase 3: run_evaluation() service extended with NEDC scoring
- [x] Final: Baseline evaluated on eval set, official NEDC metrics obtained

---

## Troubleshooting

### Tests fail with "NEDC-BENCH not found"
```bash
# Verify nedc-bench cloned
ls reference_repos/nedc-bench/

# If not, clone it
git clone https://github.com/Clarity-Digital-Twin/nedc-bench.git reference_repos/nedc-bench
```

### Import errors from nedc-bench
```bash
# Install nedc-bench dependencies
cd reference_repos/nedc-bench
pip install -e .
cd ../..
```

### "No .csv_bi files found" error
```bash
# Verify ground truth files exist
find data_ext4/tusz/edf/eval/ -name "*.csv_bi" | head

# If empty, check TUSZ download
```

### Tests fail with "Invalid CSV_BI format"
```bash
# Verify fixtures include # headers
grep "# version" tests/unit/eval/test_nedc_wrapper.py
# Should find multiple matches in fixture strings
```

---

**DONE!** Follow this guide step-by-step for iron-clad TDD implementation with ACTUAL codebase patterns.
