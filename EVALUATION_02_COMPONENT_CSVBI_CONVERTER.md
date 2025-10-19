# Component 1: CSV_BI Export - Technical Specification

## 🚨 STATUS: OBSOLETE - DO NOT IMPLEMENT THIS SPEC

**Why Obsolete**: CSV_BI export functionality ALREADY EXISTS in the codebase!

**Existing Implementation**:
- ✅ `export_csv_bi()` function: `src/brain_brr/events/export.py:15-52`
- ✅ CLI integration: `python -m src evaluate --output-csv-bi path.csv`
- ✅ Service integration: `cli/services/evaluation.py:220-278`

**What This Doc Described**: Creating a new `CSVBIConverter` class and `format_converter.py` file (~150 lines)

**What We Actually Need**: NOTHING - just use existing `export_csv_bi()` or extend it if needed

**For NEDC Integration**: See `EVALUATION_03_COMPONENT_NEDC_SCORER.md` for how to use existing CSV_BI export with NEDC scorer

---

## Historical Spec (For Reference Only - DO NOT IMPLEMENT)

The sections below describe a proposed `CSVBIConverter` class that was planned before discovering the existing `export_csv_bi()` implementation. **This spec is preserved for audit purposes only.**

---

## Class: RecordingMetadata

```python
from dataclasses import dataclass

@dataclass
class RecordingMetadata:
    """Metadata for one recording needed for CSV_BI format"""
    file_id: str          # e.g., "aaaaaaaa_s001_t000"
    patient: str          # e.g., "aaaaaaaa"
    session: str          # e.g., "s001"
    token: str            # e.g., "t000"
    duration_sec: float   # e.g., 300.0
    sampling_rate: int    # e.g., 256
```

---

## Class: CSVBIConverter

### Responsibilities
1. Load .npy probability files
2. Apply existing post-processing (`batch_probs_to_events`)
3. Format events as CSV_BI text
4. Write to disk with proper NEDC metadata

### Dependencies
- `batch_probs_to_events()` from `metrics.py` (EXISTING code - reuse!)
- `PostprocessingConfig` (from `config.schemas`)

---

## Method Signatures

### `__init__()`

```python
def __init__(
    self,
    post_config: PostprocessingConfig,
    sampling_rate: int = 256,
):
    """
    Initialize converter with post-processing configuration.

    Args:
        post_config: Configuration for hysteresis + morphology
        sampling_rate: Sampling rate in Hz (default: 256)

    Raises:
        ValueError: If sampling_rate <= 0
        ValueError: If post_config invalid (tau_on >= tau_off)
    """
```

---

### `convert_recording()`

```python
def convert_recording(
    self,
    probs_path: Path,
    output_path: Path,
    metadata: RecordingMetadata,
) -> Path:
    """
    Convert one recording from .npy to .csv_bi format.

    Args:
        probs_path: Path to *_probs.npy file
        output_path: Path where .csv_bi file will be written
        metadata: Recording metadata for CSV_BI header

    Returns:
        Path to created .csv_bi file (same as output_path)

    Raises:
        FileNotFoundError: If probs_path doesn't exist
        ValueError: If probs.npy has wrong shape (must be 1D)
        ValueError: If duration mismatch (probs length vs metadata)
        IOError: If cannot write output_path

    CSV_BI Output Format:
        # version = csv_v1.0.0
        # bname = {metadata.patient}_{metadata.session}
        # duration = {metadata.duration_sec:.4f} secs
        # montage_file = nedc_eas_default_montage.txt
        #
        channel,start_time,stop_time,label,confidence
        TERM,{start:.4f},{end:.4f},seiz,1.0
        ...

    NOTE: Header lines MUST have "#" prefix! This matches NEDC/TUSZ ground truth format.
    See src/brain_brr/events/export.py:36-44 for reference implementation.

    Workflow:
    1. Load probs from probs_path
    2. Validate shape: (n_samples,) matches duration × sampling_rate
    3. Convert to events via batch_probs_to_events()
    4. Format as CSV_BI text
    5. Write to output_path atomically
    """
```

**KEY IMPLEMENTATION NOTE**: Reuse `batch_probs_to_events()` - don't rewrite it!

```python
from src.brain_brr.eval.metrics import batch_probs_to_events

# Inside convert_recording():
probs = np.load(probs_path)
events = batch_probs_to_events(
    torch.from_numpy(probs).unsqueeze(0),  # Add batch dim
    self.post_config,
    self.sampling_rate
)[0]  # Get first (only) recording

# events is now list of (start_sec, end_sec) tuples
for start, end in events:
    f.write(f"TERM,{start:.4f},{end:.4f},seiz,1.0\n")
```

---

### `convert_directory()`

```python
def convert_directory(
    self,
    predictions_dir: Path,
    output_dir: Path,
    metadata_dict: Dict[str, RecordingMetadata],
) -> List[Path]:
    """
    Convert all *_probs.npy files in directory to .csv_bi format.

    Args:
        predictions_dir: Directory containing *_probs.npy files
        output_dir: Directory where .csv_bi files will be written
        metadata_dict: Map from file_id to RecordingMetadata

    Returns:
        List of paths to created .csv_bi files

    Raises:
        FileNotFoundError: If predictions_dir doesn't exist
        ValueError: If no *_probs.npy files found
        KeyError: If file_id not in metadata_dict

    Behavior:
    1. Create output_dir if doesn't exist
    2. Find all *_probs.npy files
    3. For each file:
       - Extract file_id from filename (strip "_probs.npy")
       - Lookup metadata in metadata_dict
       - Call convert_recording()
       - Log success/failure
    4. Return list of successfully created files
    5. Warn (don't fail) if some files skip due to missing metadata
    """
```

---

## Test Specifications

**File**: `tests/unit/eval/test_format_converter.py`

### Test 1: Initialization

```python
def test_init_valid_config(self, post_config):
    """CSVBIConverter initializes with valid config"""
    converter = CSVBIConverter(post_config, sampling_rate=256)
    assert converter.sampling_rate == 256
    assert converter.post_config == post_config

def test_init_invalid_sampling_rate(self, post_config):
    """CSVBIConverter raises ValueError for invalid sampling rate"""
    with pytest.raises(ValueError, match="sampling_rate must be > 0"):
        CSVBIConverter(post_config, sampling_rate=0)
```

### Test 2: Simple Recording Conversion

```python
def test_convert_recording_simple(self, converter, sample_probs_simple, sample_metadata, tmp_path):
    """
    Convert simple recording with one seizure event.

    Input: 300s recording, seizure from 100-150s (prob=0.95)
    Expected: One CSV_BI event line: TERM,100.0000,150.0000,seiz,1.0
    """
    output_path = tmp_path / "test_output.csv_bi"
    result = converter.convert_recording(sample_probs_simple, output_path, sample_metadata)

    assert result == output_path
    assert output_path.exists()

    content = output_path.read_text()
    assert "# version = csv_v1.0.0" in content
    assert "# bname = aaaaaaaa_s001_t000" in content
    assert "# duration = 300.0000 secs" in content
    assert "# montage_file = nedc_eas_default_montage.txt" in content

    lines = [l for l in content.split("\n") if l.startswith("TERM,")]
    assert len(lines) == 1
```

### Test 3: Complex Recording (Merging + Filtering)

```python
def test_convert_recording_complex(self, converter, sample_probs_complex, sample_metadata, tmp_path):
    """
    Convert complex recording with multiple events, merging, filtering.

    Input events: 10-20s, 22-24s, 25-45s, 100-101s, 200-250s
    Expected output (after post-processing):
    - Event 1: ~10-20s
    - Event 2: ~22-45s (merged 22-24s and 25-45s)
    - Event 3: ~200-250s
    - Event at 100-101s filtered (too short < 3s)
    """
    # Test implementation checks for 3 events with correct times
```

### Test 4-6: Error Handling

```python
def test_convert_recording_file_not_found(self, converter, sample_metadata, tmp_path):
    """CSVBIConverter raises FileNotFoundError for missing probs file"""

def test_convert_recording_wrong_shape(self, converter, sample_metadata, tmp_path):
    """CSVBIConverter raises ValueError for wrong probs shape (2D instead of 1D)"""

def test_convert_recording_duration_mismatch(self, converter, sample_metadata, tmp_path):
    """CSVBIConverter raises ValueError if probs length != duration × sampling_rate"""
```

### Test 7-8: Directory Conversion

```python
def test_convert_directory_success(self, converter, tmp_path):
    """Convert multiple recordings in directory (3 files)"""

def test_convert_directory_missing_metadata(self, converter, tmp_path):
    """convert_directory() warns but continues if metadata missing for some files"""
```

### Test 9: Edge Case

```python
def test_convert_recording_no_events(self, converter, sample_metadata, tmp_path):
    """Convert recording with no seizure events detected (all background)"""
```

### Test 10: Format Compliance

```python
def test_csv_bi_format_compliance(self, converter, sample_probs_simple, sample_metadata, tmp_path):
    """Verify exact CSV_BI format compliance with NEDC spec"""
    # Checks:
    # - Line 1: "version = csv_bi_v01.00.00"
    # - Line 2: "patient = ..."
    # - Line 3: "session = ..."
    # - Line 4: "duration = X.XXXX secs"
    # - Line 5: blank
    # - Line 6: "channel,start_time,stop_time,label,confidence"
    # - Event lines: "TERM,X.XXXX,Y.YYYY,seiz,1.0"
```

---

## Test Fixtures

### sample_probs_simple

```python
@pytest.fixture
def sample_probs_simple(self, tmp_path):
    """
    Simple probability timeline: 300s recording
    - 0-100s: background (prob=0.1)
    - 100-150s: seizure (prob=0.95)
    - 150-300s: background (prob=0.1)

    Expected events (with tau_on=0.86, tau_off=0.78):
    - One event: ~100.0s to ~150.0s
    """
    n_samples = 300 * 256  # 300s × 256 Hz
    probs = np.full(n_samples, 0.1, dtype=np.float32)
    probs[100*256:150*256] = 0.95  # Seizure

    probs_path = tmp_path / "test_probs.npy"
    np.save(probs_path, probs)
    return probs_path
```

### sample_probs_complex

```python
@pytest.fixture
def sample_probs_complex(self, tmp_path):
    """
    Complex probability timeline with:
    - Multiple seizures
    - Short events (< 3s) that should be filtered
    - Close events (< 2s apart) that should be merged
    """
    # Implementation creates 5 seizure regions
    # Post-processing should merge/filter → 3 final events
```

---

## Acceptance Criteria

**CSVBIConverter is complete when**:
- [ ] All 10 unit tests pass
- [ ] Code coverage ≥ 95% for format_converter.py
- [ ] Can convert sample dev set predictions (manual integration test)
- [ ] Output .csv_bi files validated against NEDC-BENCH parser (no errors)
- [ ] Performance: Can convert 1000 recordings in < 60 seconds
- [ ] Error messages are clear and actionable
- [ ] Logging provides visibility into conversion process

---

## Error Handling

| Error Condition | Exception | Message | Recovery |
|-----------------|-----------|---------|----------|
| probs file not found | FileNotFoundError | "Probabilities file not found: {path}" | Skip file, log warning |
| Invalid probs shape | ValueError | "Probabilities must be 1D array, got shape {shape}" | Skip file, log error |
| Duration mismatch | ValueError | "Duration mismatch: {actual}s != {expected}s" | Skip file, log error |
| Cannot write output | IOError | "Failed to write CSV_BI file: {path}" | Skip file, log error |
| Missing metadata | KeyError | "No metadata found for {file_id}" | Skip file, log warning |

---

## Performance Target

| Operation | Target | Notes |
|-----------|--------|-------|
| Convert 1 recording | < 100ms | Per-recording overhead |
| Convert 1000 recordings | < 60s | Batch processing |
| Convert 2000 recordings (eval set) | < 3 min | Full eval set |

---

## Implementation Notes

1. **Reuse existing code**: `batch_probs_to_events()` already does hysteresis + morphology
2. **Simple is better**: No multiprocessing needed (serial is fast enough)
3. **Atomic writes**: Write to temp file, then rename (avoid partial writes)
4. **Logging**: INFO for progress, WARNING for skipped files, ERROR for failures

---

## Next Steps

**Week 1 Implementation**:
- Day 1-2: Write all 10 tests (TDD - tests fail initially)
- Day 3-4: Implement CSVBIConverter (make tests pass)
- Day 5: Test on real dev predictions, validate CSV_BI format

**Ready to begin!**
