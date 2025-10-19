# NEDC Evaluation - Reference Documentation

**Purpose**: Reference material for implementation - dataclasses, format examples, error tables, performance benchmarks

**Use during coding**: Look up CSV_BI format, error messages, dataclass fields, etc.

---

## Table of Contents

1. [Dataclass Definitions](#dataclass-definitions)
2. [CSV_BI Format Specification](#csv_bi-format-specification)
3. [TUSZ File Structure](#tusz-file-structure)
4. [NEDC Algorithms Reference](#nedc-algorithms-reference)
5. [Performance Benchmarks](#performance-benchmarks)
6. [Error Handling Tables](#error-handling-tables)
7. [Bash Commands Reference](#bash-commands-reference)

---

## Dataclass Definitions

### NEDCMetrics

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class NEDCMetrics:
    """
    Structured NEDC evaluation metrics.

    CRITICAL: NEDC API only provides counts (tp/fp/fn).
    sensitivity_at_*FA fields must be computed by NEDCScorer!
    """
    algorithm: str                        # "overlap", "taes", etc.
    sensitivity_at_10FA_24h: float        # % seizures detected at 10 FA/24h
    sensitivity_at_5FA_24h: float         # % seizures detected at 5 FA/24h
    sensitivity_at_1FA_24h: float         # % seizures detected at 1 FA/24h
    taes_score: Optional[float]           # Only for "taes" algorithm
    f1: float                             # F1 score (harmonic mean of P/R)
    precision: float                      # TP / (TP + FP)
    recall: float                         # TP / (TP + FN)
    tp: int                               # True positives (event count)
    fp: int                               # False positives (event count)
    fn: int                               # False negatives (event count)
    total_seizure_duration_sec: float     # Total seizure time in dataset
    total_recording_duration_sec: float   # Total recording time in dataset
```

### RecordingMetadata

```python
from dataclasses import dataclass

@dataclass
class RecordingMetadata:
    """Metadata for one EEG recording needed for CSV_BI format"""
    file_id: str          # e.g., "aaaaaaaa_s001_t000"
    patient: str          # e.g., "aaaaaaaa"
    session: str          # e.g., "s001"
    token: str            # e.g., "t000"
    duration_sec: float   # e.g., 300.0
    sampling_rate: int    # e.g., 256
```

**Parse from filename**:
```python
def parse_file_id(file_id: str) -> RecordingMetadata:
    """Parse TUSZ file_id into metadata."""
    parts = file_id.split("_")
    if len(parts) != 3:
        raise ValueError(f"Invalid file_id: {file_id}")

    return RecordingMetadata(
        file_id=file_id,
        patient=parts[0],
        session=parts[1],
        token=parts[2],
        duration_sec=0.0,  # Extract from EDF or NPY
        sampling_rate=256,
    )
```

### EvaluationResults

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
        """Format as markdown table row for publication"""
        return (
            f"| {self.experiment_name} "
            f"| {self.metrics.sensitivity_at_10FA_24h:.2f}% "
            f"| {self.metrics.f1:.3f} "
            f"| {self.metrics.tp} "
            f"| {self.metrics.fp} "
            f"| {self.metrics.fn} |"
        )
```

---

## CSV_BI Format Specification

### Format Structure

CSV_BI (CSV with header comments for NEDC compatibility)

**Header lines** (MUST have `#` prefix! Matches TUSZ ground truth format):
```
# version = csv_v1.0.0
# bname = {patient}_{session}_{token}
# duration = {duration_sec:.4f} secs
# montage_file = nedc_eas_default_montage.txt
#
```

**Column header line** (NO `#` prefix):
```
channel,start_time,stop_time,label,confidence
```

**Event lines** (one per seizure):
```
TERM,{start_sec:.4f},{stop_sec:.4f},seiz,1.0
```

### Complete Example

**File**: `aaaaaaaq_s006_t000.csv_bi`

```
# version = csv_v1.0.0
# bname = aaaaaaaq_s006_t000
# duration = 300.0000 secs
# montage_file = nedc_eas_default_montage.txt
#
channel,start_time,stop_time,label,confidence
TERM,45.2000,67.8000,seiz,1.0
TERM,102.5000,115.3000,seiz,1.0
TERM,200.1250,245.7890,seiz,1.0
```

### Python Generation Code

```python
def write_csv_bi(
    events: List[Tuple[float, float]],
    metadata: RecordingMetadata,
    output_path: Path,
):
    """Write events to CSV_BI format file."""
    with open(output_path, 'w') as f:
        # Header (with # prefix!)
        f.write("# version = csv_v1.0.0\n")
        f.write(f"# bname = {metadata.patient}_{metadata.session}_{metadata.token}\n")
        f.write(f"# duration = {metadata.duration_sec:.4f} secs\n")
        f.write("# montage_file = nedc_eas_default_montage.txt\n")
        f.write("#\n")

        # Column header (NO # prefix)
        f.write("channel,start_time,stop_time,label,confidence\n")

        # Events
        for start_sec, end_sec in events:
            f.write(f"TERM,{start_sec:.4f},{end_sec:.4f},seiz,1.0\n")
```

---

## TUSZ File Structure

### Directory Structure

```
data_ext4/tusz/edf/
├── train/
│   ├── aaaaaaaa/
│   │   └── s001_2003/
│   │       └── 01_tcp_ar/
│   │           ├── aaaaaaaa_s001_t000.edf
│   │           ├── aaaaaaaa_s001_t000.csv
│   │           └── aaaaaaaa_s001_t000.csv_bi  ← Ground truth
│   └── ...
├── dev/
│   └── ...
└── eval/
    └── ...
```

### Path Pattern

**Pattern**: `{tusz_root}/edf/{split}/{patient}/{session}_YYYY/01_tcp_ar/{file_id}.{ext}`

**Example**:
```
/home/jj/proj/brain-go-brr-v2/data_ext4/tusz/edf/eval/aaaaaaaq/s006_2014/01_tcp_ar/aaaaaaaq_s006_t000.csv_bi
```

**Glob pattern** (handles year suffix):
```python
tusz_root = Path("data_ext4/tusz/edf")
patient, session, token = "aaaaaaaq", "s006", "t000"
file_id = f"{patient}_{session}_{token}"

# Glob handles year suffix (e.g., s006_2014)
csv_bi_files = list(tusz_root.glob(f"eval/{patient}/{session}_*/01_tcp_ar/{file_id}.csv_bi"))
```

### Cache Structure

```
cache/tusz_mmap/
├── train/
│   ├── _dataset_index.json              ← Manifest (4667 files)
│   ├── aaaaaaaa_s001_t000_data.npy     ← EEG data (19 channels × samples)
│   ├── aaaaaaaa_s001_t000_labels.npy   ← Binary labels (samples,)
│   └── ...
├── dev/
│   ├── _dataset_index.json              ← Manifest (1832 files)
│   └── ...
└── eval/
    ├── _dataset_index.json              ← Manifest (~2000 files)
    └── ...
```

**Manifest format** (`_dataset_index.json`):
```json
{
  "files": [
    "data_ext4/tusz/edf/eval/aaaaaaaq/s006_2014/01_tcp_ar/aaaaaaaq_s006_t000.edf",
    "data_ext4/tusz/edf/eval/aaaaaaaq/s006_2014/01_tcp_ar/aaaaaaaq_s006_t001.edf",
    ...
  ]
}
```

---

## NEDC Algorithms Reference

### 1. Overlap (Recommended for Seizures)

**Algorithm**: Any temporal overlap between reference and hypothesis events counts as TP

**Use case**: Seizure detection (standard for literature)

**Best for**: Events with variable durations

**Scoring logic**:
- If hypothesis overlaps ANY part of reference event → TP
- If hypothesis doesn't overlap any reference event → FP
- If reference event has no overlapping hypothesis → FN

**Example**:
```
Reference: [10s - 30s]
Hypothesis: [25s - 35s]  ← TP (overlap exists)
Hypothesis: [40s - 50s]  ← FP (no overlap)
```

### 2. TAES (Time-Aligned Event Scoring)

**Algorithm**: Multi-overlap sequencing for complex event matching

**Use case**: Precise event timing

**Best for**: Multiple overlapping events

**Returns**: TAES score (0-1, higher is better)

### 3. DP (Dynamic Programming)

**Algorithm**: Finds optimal match between reference and hypothesis

**Use case**: Optimal event alignment

**Best for**: Complex event sequences

### 4. Epoch (250ms sampling)

**Algorithm**: Divides timeline into 250ms epochs, compares labels

**Use case**: Sample-by-sample comparison

**Best for**: High-resolution temporal analysis

**Note**: More granular than event-based methods

### 5. IRA (Inter-Rater Agreement)

**Algorithm**: Cohen's κ for multi-class labels

**Use case**: Measuring annotator agreement

**Best for**: Multi-class problems (not just seizure vs background)

---

## Performance Benchmarks

### Format Conversion (CSV_BI Export)

| Operation | Target Time | Max Memory | Notes |
|-----------|-------------|------------|-------|
| Convert 1 recording | < 100ms | < 100MB | Per-recording overhead |
| Convert 100 recordings | < 10s | < 500MB | Batch processing |
| Convert 1000 recordings | < 60s | < 500MB | Dev set scale |
| Convert 2000 recordings (eval) | < 3 min | < 1GB | Full eval set |

**Measurement**:
```python
import time
start = time.time()
# ... conversion code ...
elapsed = time.time() - start
assert elapsed < 180  # < 3 minutes for 2000 files
```

### NEDC Scoring

| Operation | Target Time | Max Memory | Notes |
|-----------|-------------|------------|-------|
| Score 1 file pair (overlap) | < 30ms | < 50MB | Per-pair overhead |
| Score 100 file pairs | < 3s | < 200MB | Small batch |
| Score 1000 file pairs | < 30s | < 1GB | Dev set scale |
| Score 2000 file pairs (eval) | < 60s | < 2GB | Full eval set |

### End-to-End Evaluation

| Operation | Target Time | Max Memory | Notes |
|-----------|-------------|------------|-------|
| Load checkpoint | < 5s | < 500MB | Model initialization |
| Inference (dev, 1832 files) | < 20 min | < 24GB | GPU-dependent |
| Inference (eval, ~2000 files) | < 25 min | < 24GB | GPU-dependent |
| Full eval pipeline (dev) | < 30 min | < 24GB | Includes conversion + scoring |
| Full eval pipeline (eval) | < 40 min | < 24GB | Includes conversion + scoring |

---

## Error Handling Tables

### NEDCScorer Errors

| Error Condition | Exception | Message Template | Recovery Strategy |
|-----------------|-----------|------------------|-------------------|
| nedc-bench not found | `ImportError` | `"NEDC-BENCH not found at {path}. Clone from https://github.com/..."` | Fail fast (init) |
| nedc-bench import failed | `RuntimeError` | `"Failed to import nedc-bench: {error}"` | Fail fast (init) |
| Reference dir not found | `FileNotFoundError` | `"Reference directory not found: {path}"` | Fail fast |
| Hypothesis dir not found | `FileNotFoundError` | `"Hypothesis directory not found: {path}"` | Fail fast |
| No CSV_BI files in ref | `ValueError` | `"No .csv_bi files found in reference directory: {path}"` | Fail fast |
| No CSV_BI files in hyp | `ValueError` | `"No .csv_bi files found in hypothesis directory: {path}"` | Fail fast |
| File count mismatch | `ValueError` | `"File count mismatch: {ref_count} reference vs {hyp_count} hypothesis"` | Fail fast |
| NEDC scoring failed | `RuntimeError` | `"NEDC scoring failed: {error}"` | Fail fast |
| Invalid CSV_BI format | `RuntimeError` | `"Invalid CSV_BI format in {file}: {error}"` | Skip file, log ERROR, continue |

### CSV_BI Conversion Errors

| Error Condition | Exception | Message Template | Recovery Strategy |
|-----------------|-----------|------------------|-------------------|
| Probs file not found | `FileNotFoundError` | `"Probabilities file not found: {path}"` | Skip file, log WARNING, continue |
| Invalid probs shape | `ValueError` | `"Probabilities must be 1D array, got shape {shape}"` | Skip file, log ERROR, continue |
| Duration mismatch | `ValueError` | `"Duration mismatch: expected {expected}s, got {actual}s"` | Skip file, log ERROR, continue |
| Cannot write output | `IOError` | `"Failed to write CSV_BI file to {path}: {error}"` | Skip file, log ERROR, continue |
| Missing metadata | `KeyError` | `"No metadata found for file_id: {file_id}"` | Skip file, log WARNING, continue |

### Evaluation Pipeline Errors

| Error Condition | Exception | Message Template | Recovery Strategy |
|-----------------|-----------|------------------|-------------------|
| Checkpoint not found | `FileNotFoundError` | `"Checkpoint not found: {path}"` | Fail fast (init) |
| Checkpoint load failed | `RuntimeError` | `"Failed to load checkpoint: {error}"` | Fail fast (init) |
| Invalid checkpoint format | `RuntimeError` | `"Invalid checkpoint format: missing '{key}' key"` | Fail fast (init) |
| Test data not found | `FileNotFoundError` | `"Test data not found for split '{split}'"` | Fail fast |
| Inference failed (file) | `RuntimeError` | `"Inference failed on {file_id}: {error}"` | Skip file, log ERROR, continue |
| No predictions generated | `ValueError` | `"No predictions generated for split '{split}'"` | Fail fast |

---

## Bash Commands Reference

### Data Verification

```bash
# Verify TUSZ eval set exists
ls /home/jj/proj/brain-go-brr-v2/data_ext4/tusz/edf/eval/

# Count EDF files
find /home/jj/proj/brain-go-brr-v2/data_ext4/tusz/edf/eval/ -name "*.edf" | wc -l
# Expected: ~2000

# Count CSV_BI ground truth files
find /home/jj/proj/brain-go-brr-v2/data_ext4/tusz/edf/eval/ -name "*.csv_bi" | wc -l
# Expected: ~2000

# View sample CSV_BI file
head -20 $(find data_ext4/tusz/edf/eval/ -name "*.csv_bi" | head -1)
```

### Cache Operations

```bash
# Preprocess eval set
python -m src build-cache \
  --data-dir data_ext4/tusz/edf/eval/ \
  --cache-dir cache/tusz_mmap/eval/ \
  --split eval

# Verify cache created
ls cache/tusz_mmap/eval/

# Count cached data files
ls cache/tusz_mmap/eval/*_data.npy | wc -l
# Expected: ~2000

# Check manifest
cat cache/tusz_mmap/eval/_dataset_index.json | head
```

### NEDC-BENCH Verification

```bash
# Verify nedc-bench cloned
ls reference_repos/nedc-bench/src/nedc_bench/

# Check for key modules
ls reference_repos/nedc-bench/src/nedc_bench/orchestration.py
ls reference_repos/nedc-bench/src/nedc_bench/models/annotations.py

# Verify sample data for integration test
ls reference_repos/nedc-bench/data/csv_bi_parity/
```

### Testing

```bash
# Run NEDCScorer unit tests
.venv/bin/pytest tests/unit/eval/test_nedc_wrapper.py -v

# Run integration test
.venv/bin/pytest tests/integration/eval/test_nedc_integration.py -v -m integration

# Check coverage
.venv/bin/pytest tests/unit/eval/test_nedc_wrapper.py \
  --cov=src/brain_brr/eval/nedc_wrapper \
  --cov-report=term \
  --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Evaluation

```bash
# Run evaluation with NEDC scoring (dev set)
python -m src evaluate \
  --checkpoint results/local_fla_training/checkpoints/best.pt \
  --split dev \
  --nedc-score \
  --output results/test_nedc/

# Run evaluation on eval set (official)
python -m src evaluate \
  --checkpoint results/local_fla_training/checkpoints/best.pt \
  --split eval \
  --nedc-score \
  --output results/eval_baseline/

# View results
cat results/eval_baseline/metrics/eval_overlap_metrics.json | jq .

# Compare experiments
cat results/eval_baseline/metrics/eval_overlap_metrics.json \
    results/eval_exp1/metrics/eval_overlap_metrics.json | jq -s '.'
```

---

## Logging Specifications

### Log Levels

**DEBUG** (verbose mode only):
```python
logger.debug(f"Loading probs from {probs_path}")
logger.debug(f"Converted {len(events)} events from {n_samples} samples")
```

**INFO** (always):
```python
logger.info(f"[NEDCScorer] Scoring {len(hyp_files)} file pairs with '{algorithm}' algorithm...")
logger.info(f"[NEDCScorer] Sensitivity@10FA: {metrics.sensitivity_at_10FA_24h:.2f}%")
logger.info(f"[NEDCScorer] Scoring complete in {elapsed:.1f}s")
```

**WARNING** (non-fatal issues):
```python
logger.warning(f"No metadata found for {file_id}, skipping")
logger.warning(f"Duration mismatch for {file_id}: {actual}s != {expected}s, skipping")
```

**ERROR** (failed operations):
```python
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

**Use this doc as reference during implementation!**
