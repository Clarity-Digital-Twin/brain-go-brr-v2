# Component 3: ModelEvaluator - Technical Specification

**File**: `src/brain_brr/eval/evaluator.py` (~200 lines)

**Purpose**: End-to-end evaluation pipeline orchestrator with CLI

**Status**: Specification complete - Ready for TDD implementation

---

## Class: EvaluationResults

```python
from dataclasses import dataclass, asdict
from typing import Optional, Literal, Dict
from pathlib import Path
import json

@dataclass
class EvaluationResults:
    """Complete evaluation results for publication"""
    experiment_name: str
    checkpoint_path: str
    checkpoint_epoch: int
    split: Literal["dev", "eval"]
    algorithm: str
    metrics: NEDCMetrics
    comparison_to_dev: Optional[Dict[str, float]]  # If eval split
    timestamp: str

    def to_json(self, path: Path):
        """Save results to JSON file"""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    def to_markdown_table(self) -> str:
        """
        Format as markdown table row for publication.

        Returns:
            Markdown table row string
        """
        pass
```

---

## Class: ModelEvaluator

### Responsibilities
1. Load checkpoint and initialize model
2. Run inference on test/eval split
3. Convert predictions to CSV_BI format
4. Score with NEDC-BENCH
5. Generate publication-ready results

### Dependencies
- `CSVBIConverter` (Component 1)
- `NEDCScorer` (Component 2)
- `SeizureDetector` (model)
- `run_validation()` (for inference)

---

## Method Signatures

### `__init__()`

```python
def __init__(
    self,
    checkpoint_path: Path,
    output_dir: Path,
    config_override: Optional[TrainingConfig] = None,
):
    """
    Initialize evaluator with checkpoint and output directory.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        output_dir: Directory for outputs (predictions, metrics, etc.)
        config_override: Optional config override (uses checkpoint config by default)

    Raises:
        FileNotFoundError: If checkpoint_path doesn't exist
        RuntimeError: If checkpoint loading fails

    Behavior:
    1. Validate checkpoint exists
    2. Load checkpoint and extract config
    3. Initialize CSVBIConverter with post-processing config
    4. Initialize NEDCScorer
    5. Create output directory structure
    """
```

---

### `evaluate_on_split()`

```python
def evaluate_on_split(
    self,
    split: Literal["dev", "eval"],
    algorithm: str = "overlap",
    save_predictions: bool = True,
    save_csv_bi: bool = True,
) -> EvaluationResults:
    """
    Complete evaluation pipeline on dev or eval split.

    Args:
        split: Which data split to evaluate on
            - "dev": Validation set (used during training)
            - "eval": Official test set (held-out, never seen)
        algorithm: NEDC scoring algorithm
        save_predictions: Save .npy predictions to disk
        save_csv_bi: Save .csv_bi files to disk (needed for NEDC)

    Returns:
        EvaluationResults with official NEDC metrics

    Workflow:
    1. Load test data for split
    2. Run inference (reuse validation code from train/val_step.py)
    3. Extract metadata from data (patient, session, duration)
    4. Convert predictions to CSV_BI format
    5. Copy ground truth CSV_BI files to reference/ dir
    6. Score with NEDC-BENCH
    7. Generate comparison to dev results (if split=="eval")
    8. Save results to JSON
    9. Return EvaluationResults

    Output Structure:
        output_dir/
        ├── predictions/
        │   └── {split}/
        │       ├── file_001_probs.npy
        │       ├── file_001_labels.npy
        │       └── ...
        ├── csv_bi/
        │   ├── reference/
        │   │   ├── file_001.csv_bi (ground truth)
        │   │   └── ...
        │   └── hypothesis/
        │       ├── file_001.csv_bi (predictions)
        │       └── ...
        ├── metrics/
        │   └── {split}_{algorithm}_metrics.json
        └── results_summary.md

    Example:
        evaluator = ModelEvaluator(
            checkpoint_path=Path("results/baseline/checkpoints/best.pt"),
            output_dir=Path("results/eval_baseline")
        )
        results = evaluator.evaluate_on_split(split="eval", algorithm="overlap")
        print(f"Sensitivity@10FA: {results.metrics.sensitivity_at_10FA_24h:.2f}%")
    """
```

---

### `compare_experiments()`

```python
@staticmethod
def compare_experiments(
    baseline_results: EvaluationResults,
    experiment_results: EvaluationResults,
) -> Dict[str, float]:
    """
    Compare two experiment results (e.g., baseline vs Exp1).

    Args:
        baseline_results: Results from baseline experiment
        experiment_results: Results from comparison experiment

    Returns:
        Dict with comparison metrics:
        - sensitivity_improvement: Absolute % improvement
        - sensitivity_relative: Relative % improvement
        - overfitting_reduction: Dev-test gap reduction
        - f1_improvement: Absolute F1 improvement

    Example:
        baseline = evaluator1.evaluate_on_split("eval")
        exp1 = evaluator2.evaluate_on_split("eval")
        comparison = ModelEvaluator.compare_experiments(baseline, exp1)
        print(f"Sensitivity improved by {comparison['sensitivity_improvement']:.2f}%")
    """
```

---

### `generate_publication_table()`

```python
@staticmethod
def generate_publication_table(
    results_list: List[EvaluationResults],
    literature_benchmarks: Optional[Dict[str, float]] = None,
) -> str:
    """
    Generate publication-ready markdown table.

    Args:
        results_list: List of evaluation results to include
        literature_benchmarks: Optional dict of literature results

    Returns:
        Markdown table string ready for papers/docs

    Example Output:
        | Model | Dev Sens@10FA | Test Sens@10FA | Dev-Test Gap | F1 | Notes |
        |-------|---------------|----------------|--------------|----|----|
        | Baseline | 28.01% | 24.3% | 3.7% | 0.31 | Overfitting |
        | Exp1 | 26.5% | 25.8% | 0.7% | 0.33 | Better gen. |
        | Shah et al. [1] | - | 89% | - | - | SOTA |

    Example:
        results = [baseline_results, exp1_results]
        table = ModelEvaluator.generate_publication_table(
            results,
            literature_benchmarks={"Shah 2018": 89.0}
        )
        print(table)
    """
```

---

## CLI Interface

### Command Structure

```bash
python -m src.brain_brr.eval.evaluator [SUBCOMMAND] [OPTIONS]
```

### Subcommand: evaluate (default)

```bash
python -m src.brain_brr.eval.evaluator \
  --checkpoint results/baseline/checkpoints/best.pt \
  --split eval \
  --algorithm overlap \
  --output results/eval_baseline/
```

**Arguments**:
```
--checkpoint PATH       Path to .pt checkpoint file (required)
--split {dev,eval}      Which data split to evaluate (required)
--algorithm {overlap,taes,dp,epoch,ira,all}  NEDC algorithm (default: overlap)
--output PATH           Output directory (default: results/eval_{timestamp}/)
--save-predictions      Save .npy predictions (default: True)
--save-csv-bi           Save .csv_bi files (default: True)
--verbose               Enable debug logging
```

### Subcommand: compare

```bash
python -m src.brain_brr.eval.evaluator compare \
  --baseline results/eval_baseline/metrics/eval_overlap_metrics.json \
  --experiment results/eval_exp1/metrics/eval_overlap_metrics.json \
  --output results/comparison_baseline_vs_exp1.md
```

**Arguments**:
```
--baseline PATH         Path to baseline metrics.json (required)
--experiment PATH       Path to experiment metrics.json (required)
--output PATH           Output markdown file (default: stdout)
```

### CLI Implementation

```python
# In evaluator.py

def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="NEDC Evaluation Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Subcommand")

    # Evaluate subcommand
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate checkpoint")
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--split", choices=["dev", "eval"], required=True)
    eval_parser.add_argument("--algorithm", default="overlap")
    eval_parser.add_argument("--output", type=Path)
    eval_parser.add_argument("--save-predictions", action="store_true", default=True)
    eval_parser.add_argument("--save-csv-bi", action="store_true", default=True)
    eval_parser.add_argument("--verbose", action="store_true")

    # Compare subcommand
    compare_parser = subparsers.add_parser("compare", help="Compare experiments")
    compare_parser.add_argument("--baseline", type=Path, required=True)
    compare_parser.add_argument("--experiment", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path)

    args = parser.parse_args()

    if args.command == "evaluate":
        # ... implementation
    elif args.command == "compare":
        # ... implementation

if __name__ == "__main__":
    main()
```

---

## Test Specifications

**File**: `tests/integration/eval/test_evaluator.py`

### Test 1: Initialization

```python
def test_init_success(self, sample_checkpoint, tmp_path):
    """ModelEvaluator initializes with valid checkpoint"""
    output_dir = tmp_path / "eval_output"

    evaluator = ModelEvaluator(
        checkpoint_path=sample_checkpoint,
        output_dir=output_dir,
    )

    assert evaluator is not None
    assert evaluator.checkpoint_path == sample_checkpoint
    assert evaluator.output_dir == output_dir

def test_init_checkpoint_not_found(self, tmp_path):
    """ModelEvaluator raises FileNotFoundError for missing checkpoint"""
    nonexistent_ckpt = tmp_path / "nonexistent.pt"
    output_dir = tmp_path / "eval_output"

    with pytest.raises(FileNotFoundError):
        ModelEvaluator(nonexistent_ckpt, output_dir)
```

### Test 2: End-to-End Evaluation (Mocked)

```python
@pytest.mark.integration
def test_evaluate_on_split_dev(self, sample_checkpoint, tmp_path, monkeypatch):
    """
    End-to-end evaluation on dev split.

    Note: This test mocks heavy operations (inference, scoring)
    to keep runtime reasonable. Full integration test requires
    real data and GPU.
    """
    output_dir = tmp_path / "eval_output"
    evaluator = ModelEvaluator(sample_checkpoint, output_dir)

    # Mock inference
    def mock_inference(*args, **kwargs):
        return {"file_001": {"probs": ..., "labels": ...}}
    monkeypatch.setattr(evaluator, "_run_inference", mock_inference)

    # Mock NEDC scoring
    def mock_score(*args, **kwargs):
        return NEDCMetrics(
            algorithm="overlap",
            sensitivity_at_10FA_24h=25.0,
            # ... other fields
        )
    monkeypatch.setattr(evaluator.scorer, "score_predictions", mock_score)

    # Run evaluation
    results = evaluator.evaluate_on_split(split="dev", algorithm="overlap")

    assert isinstance(results, EvaluationResults)
    assert results.split == "dev"
    assert results.algorithm == "overlap"
    assert results.metrics.sensitivity_at_10FA_24h == 25.0
```

### Test 3: Experiment Comparison

```python
def test_compare_experiments(self, tmp_path):
    """Compare baseline vs experiment results"""
    baseline_metrics = NEDCMetrics(
        algorithm="overlap",
        sensitivity_at_10FA_24h=24.3,
        # ... other fields
    )

    exp1_metrics = NEDCMetrics(
        algorithm="overlap",
        sensitivity_at_10FA_24h=27.8,
        # ... other fields
    )

    baseline_results = EvaluationResults(
        experiment_name="baseline",
        checkpoint_path="results/baseline/best.pt",
        checkpoint_epoch=9,
        split="eval",
        algorithm="overlap",
        metrics=baseline_metrics,
        comparison_to_dev={"dev_sens_10FA": 28.01, "gap": 3.71},
        timestamp="2025-10-19T12:00:00",
    )

    exp1_results = EvaluationResults(
        experiment_name="exp1_reg",
        checkpoint_path="results/exp1/best.pt",
        checkpoint_epoch=12,
        split="eval",
        algorithm="overlap",
        metrics=exp1_metrics,
        comparison_to_dev={"dev_sens_10FA": 29.2, "gap": 1.4},
        timestamp="2025-10-19T14:00:00",
    )

    comparison = ModelEvaluator.compare_experiments(baseline_results, exp1_results)

    # Exp1 improved sensitivity by 3.5% absolute
    assert abs(comparison["sensitivity_improvement"] - 3.5) < 0.1

    # Exp1 reduced overfitting gap from 3.71% to 1.4% (2.31% reduction)
    assert abs(comparison["overfitting_reduction"] - 2.31) < 0.1
```

### Test 4: Publication Table Generation

```python
def test_generate_publication_table(self):
    """Generate markdown table for publication"""
    # Create sample results list
    results = [baseline_results, exp1_results]

    table = ModelEvaluator.generate_publication_table(
        results,
        literature_benchmarks={"Shah 2018": 89.0}
    )

    # Verify table format
    assert "| Model |" in table
    assert "| Baseline |" in table
    assert "| Exp1 |" in table
    assert "| Shah 2018 |" in table
```

---

## Acceptance Criteria

**ModelEvaluator is complete when**:
- [ ] All integration tests pass (with mocking for speed)
- [ ] Full end-to-end test on real dev set passes (manual, with GPU)
- [ ] Can evaluate full dev set (1832 files) in < 30 minutes
- [ ] Generates publication-ready markdown tables
- [ ] JSON output is well-formatted and complete
- [ ] CLI interface works (`python -m src.brain_brr.eval.evaluator --help`)
- [ ] Documentation includes complete usage examples

---

## Error Handling

| Error Condition | Exception | Message | Recovery |
|-----------------|-----------|---------|----------|
| Checkpoint not found | FileNotFoundError | "Checkpoint not found: {path}" | Fail fast |
| Checkpoint load failed | RuntimeError | "Failed to load checkpoint: {error}" | Fail fast |
| Test data not found | FileNotFoundError | "Test data not found for split {split}" | Fail fast |
| Inference failed | RuntimeError | "Inference failed on {file_id}: {error}" | Skip file, continue |
| Conversion failed | RuntimeError | "Conversion to CSV_BI failed: {error}" | Log error, continue |
| Scoring failed | RuntimeError | "NEDC scoring failed: {error}" | Fail fast |

---

## Performance Target

| Operation | Target | Notes |
|-----------|--------|-------|
| Load checkpoint | < 5s | Model initialization |
| Inference (1832 files, dev) | < 20 min | Depends on GPU |
| Convert to CSV_BI (1832 files) | < 3 min | Batch processing |
| Score with NEDC (1832 pairs) | < 1 min | Overlap algorithm |
| **Full eval on dev set** | **< 30 min** | **End-to-end** |
| **Full eval on eval set (~2000)** | **< 40 min** | **End-to-end** |

---

## Output Files Structure

```
results/eval_baseline/
├── predictions/
│   └── eval/
│       ├── file_001_probs.npy
│       ├── file_001_labels.npy
│       └── ...
├── csv_bi/
│   ├── reference/                # Ground truth (copied from TUSZ)
│   │   ├── file_001.csv_bi
│   │   └── ...
│   └── hypothesis/               # Model predictions (converted)
│       ├── file_001.csv_bi
│       └── ...
├── metrics/
│   └── eval_overlap_metrics.json
└── results_summary.md            # Human-readable summary
```

---

## Implementation Notes

1. **Reuse validation code**: Use `run_validation()` for inference
2. **Metadata extraction**: Get patient/session/duration from EDF files or cache
3. **Ground truth copy**: Copy .csv_bi files from TUSZ to reference/ dir
4. **Atomic operations**: Save predictions before conversion, save CSV_BI before scoring
5. **Progress logging**: INFO logs for each major step

---

## Next Steps

**Week 3 Implementation**:
- Day 1-2: Write all 4 integration tests (TDD with mocking)
- Day 3-5: Implement ModelEvaluator + CLI
- Day 6-7: Full end-to-end test on dev set

**Ready to begin after Week 2 complete!**
