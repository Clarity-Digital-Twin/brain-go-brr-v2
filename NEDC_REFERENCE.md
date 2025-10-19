# NEDC Evaluation - Reference Documentation

**Purpose**: Quick reference for implementation - dataclasses, format examples, bash commands

**Use during coding**: Look up CSV_BI format, NEDCMetrics fields, error messages, etc.

---

## NEDCMetrics Dataclass

```python
from dataclasses import dataclass

@dataclass
class NEDCMetrics:
    """
    Structured NEDC evaluation metrics.

    CRITICAL: NEDC API only provides counts (tp/fp/fn).
    We compute precision/recall/f1 from counts.
    """
    algorithm: str                        # "overlap", "taes", "dp", "epoch", "ira"
    tp: int                               # True positives (event count)
    fp: int                               # False positives (event count)
    fn: int                               # False negatives (event count)
    precision: float                      # TP / (TP + FP)
    recall: float                         # TP / (TP + FN)
    f1: float                             # 2 * (P * R) / (P + R)
    total_recording_duration_sec: float   # Sum of all file durations
    num_files: int                        # Number of file pairs scored
```

**Compute metrics from counts**:
```python
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
```

---

## CSV_BI Format (WITH # HEADERS!)

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

**CRITICAL**: Header lines MUST have `#` prefix! This matches TUSZ ground truth format.

### Python Generation

```python
from src.brain_brr.events.export import export_csv_bi
from src.brain_brr.events import SeizureEvent

events = [
    SeizureEvent(start_s=45.2, end_s=67.8, confidence=1.0),
    SeizureEvent(start_s=102.5, end_s=115.3, confidence=1.0),
]

export_csv_bi(
    events=events,
    output_path="output.csv_bi",
    patient_id="aaaaaaaq",
    recording_id="s006_t000",
    duration_s=300.0,
)
```

---

## TUSZ File Structure

### Directory Layout

```
data_ext4/tusz/edf/
├── train/
├── dev/
└── eval/
    ├── aaaaaaaq/
    │   └── s006_2014/           # Session with year suffix
    │       └── 01_tcp_ar/       # Montage type
    │           ├── aaaaaaaq_s006_t000.edf
    │           ├── aaaaaaaq_s006_t000.csv
    │           └── aaaaaaaq_s006_t000.csv_bi  ← Ground truth (with # headers)
    └── ...
```

### Path Pattern

**Pattern**: `{tusz_root}/edf/{split}/{patient}/{session}_YYYY/01_tcp_ar/{file_id}.{ext}`

**Glob pattern** (handles year suffix):
```python
tusz_root = Path("data_ext4/tusz/edf")
patient, session, token = "aaaaaaaq", "s006", "t000"
file_id = f"{patient}_{session}_{token}"

# Glob handles year suffix (e.g., s006_2014)
csv_bi_files = list(tusz_root.glob(f"eval/{patient}/{session}_*/01_tcp_ar/{file_id}.csv_bi"))
```

---

## nedc-bench API (FILE-LEVEL!)

### BetaPipeline Methods

**CRITICAL**: All methods are FILE-LEVEL, not directory-level. Must loop over pairs.

```python
from nedc_bench.orchestration.dual_pipeline import BetaPipeline

beta = BetaPipeline()

# FILE-LEVEL API (not directory-level!)
result = beta.evaluate_overlap(ref_file: Path, hyp_file: Path) -> OverlapResult
result = beta.evaluate_taes(ref_file: Path, hyp_file: Path) -> TAESResult
result = beta.evaluate_dp(ref_file: Path, hyp_file: Path) -> DPResult
result = beta.evaluate_epoch(ref_file: Path, hyp_file: Path) -> EpochResult
result = beta.evaluate_ira(ref_file: Path, hyp_file: Path) -> IRAResult
```

### OverlapResult Structure

```python
@dataclass
class OverlapResult:
    hits: dict[str, int]              # Per-label hits
    misses: dict[str, int]            # Per-label misses
    false_alarms: dict[str, int]      # Per-label false alarms
    insertions: dict[str, int]        # = false_alarms (NEDC mapping)
    deletions: dict[str, int]         # = misses (NEDC mapping)
    total_hits: int                   # Sum of all hits
    total_misses: int                 # Sum of all misses
    total_false_alarms: int           # Sum of all false alarms
```

---

## Bash Commands (Click CLI)

### Data Verification

```bash
# Verify TUSZ eval set exists
ls data_ext4/tusz/edf/eval/

# Count EDF files
find data_ext4/tusz/edf/eval/ -name "*.edf" | wc -l
# Expected: ~2000

# Count CSV_BI ground truth files (with # headers!)
find data_ext4/tusz/edf/eval/ -name "*.csv_bi" | wc -l
# Expected: ~2000

# View sample CSV_BI file (verify # headers)
find data_ext4/tusz/edf/eval/ -name "*.csv_bi" | head -1 | xargs head -10
```

### Cache Operations

```bash
# Extend CLI to support eval split (FIRST!)
# Edit src/brain_brr/cli/cli.py line 207:
# FROM: type=click.Choice(["train", "dev"])
# TO:   type=click.Choice(["train", "dev", "eval"])

# Preprocess eval set
python -m src build-cache \
  --data-dir data_ext4/tusz/edf/eval/ \
  --cache-dir cache/tusz_mmap/eval/ \
  --split eval

# Verify cache created
ls cache/tusz_mmap/eval/
ls cache/tusz_mmap/eval/*_data.npy | wc -l  # Expected: ~2000
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
  --cov-report=term
```

### Evaluation (Click CLI - positional args!)

```bash
# Run evaluation with NEDC scoring (dev set)
python -m src evaluate \
  results/local_fla_training/checkpoints/best.pt \
  data_ext4/tusz/edf/dev/ \
  --output-json results/test_nedc/metrics.json

# Run evaluation on eval set (official)
python -m src evaluate \
  results/local_fla_training/checkpoints/best.pt \
  data_ext4/tusz/edf/eval/ \
  --output-json results/eval_baseline/metrics.json

# View NEDC results
cat results/eval_baseline/metrics.json | jq .nedc_overlap
```

---

## Error Handling

### NEDCScorer Errors

| Error | Exception | Message | Fix |
|-------|-----------|---------|-----|
| nedc-bench not found | `ImportError` | `"NEDC-BENCH not found at {path}"` | Clone nedc-bench to reference_repos/ |
| Reference dir not found | `FileNotFoundError` | `"Reference directory not found: {path}"` | Check TUSZ download |
| No CSV_BI files | `ValueError` | `"No .csv_bi files found in {dir}"` | Check file generation |
| Invalid CSV_BI format | `RuntimeError` | `"Invalid CSV_BI format in {file}"` | Check # headers |

---

## Performance Benchmarks

| Operation | Target Time | Notes |
|-----------|-------------|-------|
| Score 1 file pair (overlap) | < 30ms | Per-pair overhead |
| Score 1000 file pairs | < 30s | Dev set scale |
| Score 2000 file pairs (eval) | < 60s | Full eval set |

---

**Use this doc as quick reference during implementation!**
