# NEDC Evaluation Pipeline - All Questions Resolved

**Status**: All questions answered - Ready for implementation

**Date**: October 19, 2025

---

## Critical Questions & Answers

### Q1: Where is the TUSZ eval/test set?

**Answer**: ✅ **FOUND**

**Location**: `/home/jj/proj/brain-go-brr-v2/data_ext4/tusz/edf/eval/`

**Structure**:
```
eval/
├── {patient}/           # e.g., aaaaaaaq
│   └── {session_year}/  # e.g., s006_2014
│       └── 01_tcp_ar/   # Montage type
│           ├── {file_id}.edf      # EEG data
│           ├── {file_id}.csv      # Labels (one format)
│           └── {file_id}.csv_bi   # Labels (CSV_BI format) ✅
```

**Example**:
```
/home/jj/proj/brain-go-brr-v2/data_ext4/tusz/edf/eval/aaaaaaaq/s006_2014/01_tcp_ar/
├── aaaaaaaq_s006_t000.edf
├── aaaaaaaq_s006_t000.csv
└── aaaaaaaq_s006_t000.csv_bi  ← Ground truth in CSV_BI format!
```

---

### Q2: Do ground truth labels exist in CSV_BI format?

**Answer**: ✅ **YES! Already in CSV_BI format!**

**Discovery**: TUSZ provides BOTH `.csv` and `.csv_bi` label files

**Impact**: **NO conversion needed!** We can use ground truth labels directly

**File count check**:
```bash
# Count .csv_bi files in eval set
find /home/jj/proj/brain-go-brr-v2/data_ext4/tusz/edf/eval/ -name "*.csv_bi" | wc -l
# Expected: ~2000+ files
```

---

### Q3: Are EDF files available for metadata extraction?

**Answer**: ✅ **YES**

**Available files per recording**:
- `.edf` → EEG data (can extract duration from header)
- `.csv` → Labels (original format)
- `.csv_bi` → Labels (NEDC format)

**Metadata extraction strategy**:
```python
from pathlib import Path
import pyedflib

def get_recording_metadata(file_id: str, tusz_root: Path):
    """
    Extract metadata from TUSZ filename and EDF file.

    file_id: e.g., "aaaaaaaq_s006_t000"
    Returns: (patient, session, token, duration_sec)
    """
    # Parse filename
    parts = file_id.split("_")
    patient, session, token = parts[0], parts[1], parts[2]

    # Find EDF file
    # Structure: tusz/edf/eval/{patient}/{session}_YYYY/01_tcp_ar/{file_id}.edf
    # Glob pattern handles the year suffix
    edf_path = next(tusz_root.glob(f"edf/eval/{patient}/{session}_*/*/{file_id}.edf"))

    # Extract duration
    with pyedflib.EdfReader(str(edf_path)) as edf:
        duration_sec = edf.file_duration

    return patient, session, token, duration_sec
```

**Alternative (if pyedflib fails)**: Read duration from NPY cache metadata (recording length in samples / sampling_rate)

---

### Q4: Does eval cache exist?

**Answer**: ❌ **Empty (needs preprocessing)**

**Cache status**:
```
cache/tusz_mmap/
├── train/          ← Preprocessed (4667 files)
├── dev/            ← Preprocessed (1832 files)
└── eval/           ← Empty! Needs preprocessing
```

**Action required BEFORE evaluation**:
1. Preprocess eval set (same pipeline as train/dev)
2. Generate eval cache (memory-mapped NPY files)
3. Create `_dataset_index.json` manifest

**Command** (estimate):
```bash
# Preprocess eval set (one-time setup)
python -m src.brain_brr.data.preprocess \
  --split eval \
  --output cache/tusz_mmap/eval/

# Expected time: ~2-4 hours (depending on eval set size)
```

---

### Q5: What metadata format does the cache use?

**Answer**: `_dataset_index.json` lists all file paths

**Format**:
```json
{
  "files": [
    "data_ext4/tusz/edf/dev/aaaaaajy/s001_2003/02_tcp_le/aaaaaajy_s001_t000.edf",
    "data_ext4/tusz/edf/dev/aaaaaajy/s002_2003/01_tcp_ar/aaaaaajy_s002_t000.edf",
    ...
  ]
}
```

**Cache files per recording**:
- `{file_id}_data.npy` → Preprocessed EEG (19 channels × n_samples)
- `{file_id}_labels.npy` → Binary labels (n_samples,)

**Example**:
```
cache/tusz_mmap/dev/
├── _dataset_index.json
├── aaaaaajy_s001_t000_data.npy
├── aaaaaajy_s001_t000_labels.npy
├── ...
```

---

### Q6: How to extract patient/session/token from filename?

**Answer**: Simple string split on underscore

**Naming convention**: `{patient}_{session}_{token}.edf`

**Examples**:
```
aaaaaaaq_s006_t000.edf  → patient="aaaaaaaq", session="s006", token="t000"
aaaaahie_s019_t015.edf  → patient="aaaaahie", session="s019", token="t015"
```

**Implementation**:
```python
def parse_file_id(file_id: str):
    """Parse TUSZ file_id into components."""
    parts = file_id.split("_")
    if len(parts) != 3:
        raise ValueError(f"Invalid file_id: {file_id}")
    return {
        "patient": parts[0],
        "session": parts[1],
        "token": parts[2],
    }
```

---

### Q7: Where are model predictions currently saved?

**Answer**: `.npy` files in `results/*/predictions/epoch_XXX/`

**Format** (from `src/brain_brr/train/val_step.py:267-301`):
```
results/local_fla_training/predictions/
└── epoch_009/
    ├── aaaa_s001_t000_probs.npy    # Continuous probability timeline
    ├── aaaa_s001_t000_labels.npy   # Ground truth labels
    ├── ...
```

**probs.npy structure**:
- Shape: `(n_samples,)` where `n_samples = duration_sec × 256`
- Values: Float32, range [0,1] (sigmoid outputs)
- Example: 300s recording → 76,800 samples

**KEY INSIGHT**: These are continuous timelines, NOT event lists!

---

### Q8: Where are ground truth CSV_BI files?

**Answer**: Same directory as EDF files

**Path pattern**:
```
{tusz_root}/edf/eval/{patient}/{session}_YYYY/01_tcp_ar/{file_id}.csv_bi
```

**Example**:
```
/home/jj/proj/brain-go-brr-v2/data_ext4/tusz/edf/eval/aaaaaaaq/s006_2014/01_tcp_ar/aaaaaaaq_s006_t000.csv_bi
```

**Content example**:
```
version = csv_bi_v01.00.00
patient = aaaaaaaq
session = s006
duration = 300.0000 secs

channel,start_time,stop_time,label,confidence
TERM,45.2000,67.8000,seiz,1.0
TERM,102.5000,115.3000,seiz,1.0
```

---

### Q9: Should post-processing config be separate or in main config?

**Answer**: **In main config** (already exists!)

**Location**: `src/brain_brr/config/schemas.py` → `PostprocessingConfig`

**Fields**:
- `tau_on`: Hysteresis upper threshold (0.86)
- `tau_off`: Hysteresis lower threshold (0.78)
- `morphology_opening_kernel_size`: 11
- `morphology_closing_kernel_size`: 31
- `min_duration_sec`: 3.0
- `max_duration_sec`: 600.0
- `merge_threshold_sec`: 2.0

**Usage**: Load from checkpoint config
```python
checkpoint = torch.load("results/baseline/checkpoints/best.pt")
post_config = checkpoint["config"].postprocessing
```

---

### Q10: Should NEDC algorithm be configurable or hardcoded?

**Answer**: **Configurable CLI argument**

**Recommended default**: `"overlap"` (recommended for seizure detection)

**All options**:
- `"overlap"`: Any-overlap detection (best for seizures)
- `"taes"`: Time-Aligned Event Scoring
- `"dp"`: Dynamic Programming alignment
- `"epoch"`: 250ms epoch-based sampling
- `"ira"`: Inter-Rater Agreement (Cohen's κ)
- `"all"`: Run all algorithms

**CLI design**:
```bash
# Default to overlap
python -m src.brain_brr.eval.evaluator --checkpoint best.pt --split eval

# Specify algorithm
python -m src.brain_brr.eval.evaluator --checkpoint best.pt --split eval --algorithm taes

# Run all algorithms
python -m src.brain_brr.eval.evaluator --checkpoint best.pt --split eval --algorithm all
```

---

### Q11: Should we parallelize CSV_BI conversion?

**Answer**: **No (for simplicity)**

**Rationale**:
- Serial conversion is fast enough (~100ms per file)
- 2000 files × 100ms = 200 seconds (~3 minutes total)
- Multiprocessing adds complexity (WSL2 issues)
- Simplicity > micro-optimization

**Future optimization** (if needed):
```python
# Could use ThreadPoolExecutor (I/O bound)
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(convert_recording, ...) for ...]
    results = [f.result() for f in futures]
```

---

### Q12: Should we cache CSV_BI files?

**Answer**: **Yes, save to disk**

**Rationale**:
- Avoid recomputation if re-running scorer
- Debugging (can inspect CSV_BI format manually)
- Disk space is cheap (~1KB per file = ~2MB total)

**Output structure**:
```
results/eval_baseline/
├── predictions/
│   └── eval/
│       ├── file_001_probs.npy    # Original predictions
│       └── ...
└── csv_bi/
    ├── reference/                # Ground truth (copied from TUSZ)
    │   ├── file_001.csv_bi
    │   └── ...
    └── hypothesis/               # Model predictions (converted)
        ├── file_001.csv_bi
        └── ...
```

---

## Summary Table

| Question | Answer | Impact |
|----------|--------|--------|
| Eval set location | `/data_ext4/tusz/edf/eval/` | ✅ Found |
| Ground truth format | `.csv_bi` files exist | ✅ No conversion needed! |
| EDF files available | Yes | ✅ Can extract metadata |
| Eval cache exists | No (empty) | ⚠️ Needs preprocessing |
| Metadata extraction | Parse filename + read EDF | ✅ Straightforward |
| Prediction format | `.npy` continuous timelines | ℹ️ Need conversion |
| Post-processing config | In main config (exists) | ✅ Reuse existing |
| NEDC algorithm | CLI argument, default "overlap" | ✅ Flexible |
| Parallelization | No (serial is fast enough) | ✅ Keep simple |
| Cache CSV_BI files | Yes (save to disk) | ✅ Debugging + reuse |

---

## Pre-Implementation Checklist

Before starting Week 1 (CSVBIConverter):

- [ ] **Preprocess eval set** (generate cache)
  ```bash
  python -m src.brain_brr.data.preprocess --split eval --output cache/tusz_mmap/eval/
  ```

- [ ] **Verify ground truth CSV_BI files exist**
  ```bash
  ls /home/jj/proj/brain-go-brr-v2/data_ext4/tusz/edf/eval/aaaaaaaq/s006_2014/01_tcp_ar/*.csv_bi
  ```

- [ ] **Verify nedc-bench is accessible**
  ```bash
  ls reference_repos/nedc-bench/src/nedc_bench/
  ```

- [ ] **Run baseline inference with save_predictions=true** (if not done)
  ```yaml
  # In config: training.save_predictions: true
  ```

---

## Next Steps

1. **AI audit**: Review all 5 evaluation docs for accuracy
2. **Preprocess eval set**: One-time setup (~2-4 hours)
3. **Start Week 1**: Write tests for CSVBIConverter (TDD)

**All questions resolved!** Ready to proceed with implementation.
