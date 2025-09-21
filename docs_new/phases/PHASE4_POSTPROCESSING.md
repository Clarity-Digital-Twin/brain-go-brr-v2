# PHASE4_POSTPROCESSING.md — Clinical-Grade Post-Processing Pipeline

Note (2025-09-19): This doc may reference `src/experiment/*`; current implementation is
under `src/brain_brr/*`. See `DOCS_SSOT.md` for canonical commands/configs.

## 🎯 Phase 4 Goal
Convert raw per‑timestep probabilities into clinically actionable seizure events using dual‑tau hysteresis with temporal stability, morphology, duration constraints, window stitching, and event merging, targeting sensitivity at specific FA/24h operating points.

## 📋 Phase 4 Checklist
- [x] Hysteresis thresholding (dual‑tau with min_onset/min_offset)
- [x] Morphology (opening → closing; configurable kernels; CPU/GPU)
- [x] Duration filtering (≥3s min; ≤600s max; long‑event segmentation)
- [x] Window stitching (overlap‑add; optional triangular weighting)
- [x] Event merging (merge gaps ≤ tau_merge)
- [x] Confidence scoring (mean/peak/percentile)
- [x] CSV_BI export (Temple‑compliant)
- [ ] Real‑time streaming (future; stateful hysteresis)
- [x] TDD: unit + integration tests
- [x] Phase 3 integration (schemas + evaluate adapters)

## 🧭 Scope & Status

- Scope: Final design and implementation status for Phase 4 post‑processing (APIs, algorithms, configs).
- Status: Fully implemented and wired into evaluation on CPU; GPU morphology path implemented via pooling. Tests cover hysteresis,
  morphology, duration filtering, stitching, eventization, merging, and confidence. Streaming remains future work.

## 🏗️ Architecture Overview

```
Raw Probabilities (B, T) @ 256 Hz
        ↓
[1] Hysteresis Thresholding
    - τ_on = 0.86 (onset), τ_off = 0.78 (offset)
    - Stability: min_onset=128 (0.5s), min_offset=256 (1.0s)
        ↓
[2] Morphology (opening → closing)
    - Opening: remove spikes (k≈11 → ~43ms)
    - Closing: fill gaps (k≈31 → ~121ms)
        ↓
[3] Duration Filtering
    - Keep: 3s ≤ duration ≤ 600s; segment longer
        ↓
[4] Window Stitching
    - Overlap‑add (uniform/triangular), 60s window, 10s stride
        ↓
[5] Eventization
    - Mask → intervals → merge gaps ≤ tau_merge
        ↓
Clinical Events [(start_s, end_s, confidence)]
```

## 🔧 Implementation Files

```
src/experiment/postprocess.py     # Core post‑processing ops (hysteresis, morphology, duration, stitching)
src/experiment/events.py          # Intervals, merging, confidence
src/experiment/export.py          # CSV_BI and JSON exports
src/experiment/streaming.py       # Real‑time state (future)

src/experiment/evaluate.py        # Uses Phase 4 APIs for eventization and metrics

tests/test_postprocess.py         # Unit tests (hysteresis, morphology, duration, stitching)
tests/test_events.py              # Event tests (merge + confidence)
tests/test_export.py              # CSV_BI compliance tests
```

## 📐 Algorithms and Specifications

### 1) Hysteresis Thresholding (dual‑tau)
- OFF→ON: prob ≥ τ_on to enter seizure state
- ON→OFF: prob < τ_off to exit seizure state
- Threshold convention: ≥ τ_on to enter; < τ_off to exit
- Output: binary mask `(B, T)` bool
- Performance: implemented reference path (loop with retroactive onset). Vectorization is optional future optimization.

Parameters (defaults):
- tau_on=0.86, tau_off=0.78

Implemented stability windows:
- min_onset_samples=128 (0.5s @ 256 Hz) — require sustained τ_on crossing; retroactive onset marking
- min_offset_samples=256 (1.0s @ 256 Hz) — require sustained τ_off crossing

### 2) Morphology
Implemented sequence: Opening (erosion→dilation) THEN closing (dilation→erosion)
- Opening removes isolated spikes
- Closing fills short gaps
- Defaults (tunable): opening_kernel=11 (~43ms), closing_kernel=31 (~121ms)
- CPU: SciPy ndimage binary ops; GPU: pooling‑based morphology (max/min pooling)
- Output: cleaned mask `(B, T)` bool

Notes:
- GPU path uses max_pool1d and its negation for dilation/erosion and matches CPU semantics within tolerance.

### 3) Duration Filtering
- Identify contiguous True regions (intervals)
- Keep intervals with duration in [min_duration_s, max_duration_s]
- For intervals > max_duration_s, segment into ≤ max_duration_s chunks
- Return filtered intervals or mask depending on pipeline stage

Implemented: minimum and maximum duration filtering; long events are segmented into chunks ≤ max_duration_s.

### 4) Window Stitching
- Inputs: `window_probs: list[(T_window,)]`, `window_starts: list[int]`, `total_length`
- Methods:
  - overlap_add (uniform): sum contributions and divide by count
  - overlap_add_weighted (triangular): weight by distance to window center
  - max: elementwise maximum (optional)
- Edge cases: partial windows at boundaries; no division by zero
- Output: continuous probability `(total_length,)`

### 5) Eventization and Merging
Implemented:
- Transitions: pad with zeros → diff → starts where diff==+1; ends where diff==−1
- Convert to seconds via `idx / sampling_rate` (default 256)
- Merge: if gap ≤ tau_merge (default 2.0s), merge consecutive intervals
- Confidence: mean/peak/percentile probability within event, clamped to [0,1]
- Output: `SeizureEvent(start_s, end_s, confidence)` per record

### Units & Conventions
- Time unit: seconds
- Inputs: `probs` in [0,1], shape `(B, T)`
- Masks: bool tensors, shape `(B, T)`
- Determinism: no in‑place mutation of inputs

## 🧰 Configuration (Pydantic; extend src/experiment/schemas.py)

Target Python models (to add):

```python
from typing import Literal
from pydantic import BaseModel, Field, model_validator, field_validator

class HysteresisConfig(BaseModel):
    tau_on: float = Field(default=0.86, ge=0.5, le=1.0)
    tau_off: float = Field(default=0.78, ge=0.5, le=1.0)
    min_onset_samples: int = Field(default=128, ge=1)
    min_offset_samples: int = Field(default=256, ge=1)

    @model_validator(mode="after")
    def _check(self) -> "HysteresisConfig":
        if self.tau_on <= self.tau_off:
            raise ValueError("tau_on must be > tau_off")
        return self

class MorphologyConfig(BaseModel):
    opening_kernel: int = Field(default=11, ge=1)
    closing_kernel: int = Field(default=31, ge=1)
    use_gpu: bool = Field(default=False)

    @field_validator("opening_kernel", "closing_kernel")
    @classmethod
    def _odd(cls, v: int) -> int:
        if v % 2 == 0:
            raise ValueError("kernel sizes must be odd")
        return v

class DurationConfig(BaseModel):
    min_duration_s: float = Field(default=3.0, ge=1.0)
    max_duration_s: float = Field(default=600.0, ge=1.0)

    @model_validator(mode="after")
    def _check(self) -> "DurationConfig":
        if self.max_duration_s < self.min_duration_s:
            raise ValueError("max_duration_s must be ≥ min_duration_s")
        return self

class EventsConfig(BaseModel):
    tau_merge: float = Field(default=2.0, ge=0.0)
    confidence_method: Literal["mean", "peak", "percentile"] = Field(default="mean")
    confidence_percentile: float = Field(default=0.75, gt=0.0, lt=1.0)

class StitchingConfig(BaseModel):
    method: Literal["overlap_add", "overlap_add_weighted", "max"] = Field(default="overlap_add")
    window_size: int = Field(default=15360, ge=1)
    stride: int = Field(default=2560, ge=1)

class PostprocessingConfig(BaseModel):
    hysteresis: HysteresisConfig = Field(default_factory=HysteresisConfig)
    morphology: MorphologyConfig = Field(default_factory=MorphologyConfig)
    duration: DurationConfig = Field(default_factory=DurationConfig)
    events: EventsConfig = Field(default_factory=EventsConfig)
    stitching: StitchingConfig = Field(default_factory=StitchingConfig)
```

Current schema (today):
- PostprocessingConfig contains:
  - hysteresis: HysteresisConfig(tau_on, tau_off)
  - morphology: dict with keys like {"kernel_size": int, "operation": "closing"|"opening"}
  - min_duration: float (seconds)

Phase 4 will replace the morphology dict with a typed MorphologyConfig, add min_onset/min_offset in
HysteresisConfig, and introduce Duration/Events/Stitching configs as shown above. Tests already
target the current interim schema; migrations will be done TDD‑first in Phase 4.

Example YAML (configs/postprocess.yaml):

```yaml
postprocessing:
  hysteresis:
    tau_on: 0.86
    tau_off: 0.78
    min_onset_samples: 128
    min_offset_samples: 256

  morphology:
    opening_kernel: 11
    closing_kernel: 31
    use_gpu: false

  duration:
    min_duration_s: 3.0
    max_duration_s: 600.0

  events:
    tau_merge: 2.0
    confidence_method: mean
    confidence_percentile: 0.75

  stitching:
    method: overlap_add
    window_size: 15360
    stride: 2560
```

## 🚦 CSV_BI Export (Temple‑Compliant)

Required header lines (exact):
- `# version = csv_v1.0.0`
- `# bname = {patient_id}_{recording_id}`
- `# duration = {duration_seconds:.4f} secs`
- `# montage_file = nedc_eas_default_montage.txt`
- `#`

Columns:
- `channel,start_time,stop_time,label,confidence`

Rows:
- `channel = TERM` (whole‑record events)
- `start_time`, `stop_time` in seconds; `label = seiz`; `confidence ∈ [0,1]`

Notes:
- Sort rows by start_time; ensure duration header matches actual length
- Use dot decimal separator; 4 decimal places recommended

Example exporter (sketch):

```python
from pathlib import Path

def export_csv_bi(
    events: list[SeizureEvent],
    outfile: Path,
    patient_id: str,
    recording_id: str,
    duration_s: float,
) -> None:
    header = (
        "# version = csv_v1.0.0\n"
        f"# bname = {patient_id}_{recording_id}\n"
        f"# duration = {duration_s:.4f} secs\n"
        "# montage_file = nedc_eas_default_montage.txt\n"
        "#\n"
        "channel,start_time,stop_time,label,confidence\n"
    )
    lines = [header]
    for ev in sorted(events, key=lambda e: e.start_s):
        lines.append(f"TERM,{ev.start_s:.4f},{ev.end_s:.4f},seiz,{ev.confidence:.4f}\n")
    outfile.write_text("".join(lines), encoding="utf-8")
```

## 📊 Integration with Phase 3

Entry point (target Phase 4 API):

```python
def postprocess_predictions(
    raw_probs: torch.Tensor,  # (B, T)
    config: PostprocessingConfig,
    sampling_rate: int = 256,
) -> list[list[SeizureEvent]]:
    # 1) Hysteresis
    masks = apply_hysteresis(
        raw_probs,
        tau_on=config.hysteresis.tau_on,
        tau_off=config.hysteresis.tau_off,
        min_onset_samples=config.hysteresis.min_onset_samples,
        min_offset_samples=config.hysteresis.min_offset_samples,
    )

    # 2) Morphology
    masks = apply_morphology(
        masks,
        opening_kernel=config.morphology.opening_kernel,
        closing_kernel=config.morphology.closing_kernel,
    )

    # 3) Duration filtering (produce intervals)
    # filter_duration can return intervals directly for efficiency

    # 4) Eventization + merging + confidence
    all_events: list[list[SeizureEvent]] = []
    for b in range(masks.shape[0]):
        events = mask_to_events(
            masks[b], sampling_rate=sampling_rate, tau_merge=config.events.tau_merge
        )
        for ev in events:
            ev.confidence = calculate_event_confidence(raw_probs[b], ev, sampling_rate)
        all_events.append(events)
    return all_events
```

Interim integration (today):

```python
from src.experiment.evaluate import (
    batch_probs_to_events,
    batch_masks_to_events,
    evaluate_predictions,
)

# TAES/AUROC and sensitivity@FA are computed via evaluate_predictions
results = evaluate_predictions(probs, labels, fa_rates=[10, 5, 1], post_cfg, sampling_rate=256)

# If you only need events at a given threshold
events = batch_probs_to_events(probs, post_cfg, fs=256, threshold=0.5)
```

## 🧪 Test-Driven Development

Unit tests:
- Hysteresis: OFF→ON→OFF; oscillation robustness; min_onset/min_offset enforcement; equality semantics: ≥ τ_on enters; = τ_off stays on
- Morphology: opening removes spikes; closing fills gaps; GPU≈CPU within tolerance
- Duration: remove <3s; segment >600s; keep exactly 3.0s/600.0s
- Stitching: uniform vs triangular correctness; partial window handling
- Events: merge gaps ≤ tau_merge; confidence mean/peak/percentile
- Export: exact Temple header/columns; sorted rows; duration consistency

Integration tests:
- E2E on synthetic traces: stitched probs → events satisfy duration/merging; confidence in [0,1]

## ✅ Definition of Done (Phase 4 Implementation)
**Current Status**: Design complete, partial implementation in evaluate.py

**Completed**:
- [x] Basic hysteresis (without stability windows)
- [x] Single morphology operation (needs sequence fix)
- [x] Minimum duration filtering
- [x] Basic eventization (ndimage.label)

**TODO for Phase 4 completion**:
- [ ] Fix morphology bug (line 201) and implement opening→closing sequence
- [ ] Add typed Pydantic schemas (MorphologyConfig, DurationConfig, EventsConfig)
- [ ] Implement stability windows for hysteresis (min_onset/min_offset)
- [ ] Add max_duration with segmentation (currently only min_duration)
- [ ] Implement event merging (tau_merge)
- [ ] Add confidence scoring (mean/peak/percentile)
- [ ] Create SeizureEvent dataclass
- [ ] Window stitching (overlap_add methods)
- [ ] CSV_BI export with Temple compliance
- [ ] Extract to separate modules (postprocess.py, events.py, export.py)
- [ ] Tests >95% coverage for new modules
- [ ] Performance optimization (vectorization, optional GPU)

## ⚠️ Critical Considerations
- Determinism: pure functions; no in‑place modification
- Numerical stability: clamp divisions with eps; probs in [0,1]
- Edge cases: empty masks; single‑sample events; boundary windows
- Memory: avoid full‑day materialization; enable per‑record processing
- Compatibility: GPU path optional; match CPU outputs within tolerance

## 🔮 Future: Streaming Mode (Stateful)
- Maintain `HysteresisState` across chunks (carry buffers + in_seizure)
- Emit completed events once offset stability satisfied
- Configurable buffer (e.g., 10s)

---

**Status**: Ready for TDD implementation 🚀
**Estimated Time**: 3–4 days
**Prerequisites**: Phase 3 complete, model outputs available
**Owner**: Post‑processing specialist 🔬

**Next Steps**:
1) Extend Pydantic schemas + YAML
2) Write unit tests (hysteresis, morphology, duration, stitching, events, export)
3) Implement core functions one by one
4) Integrate with evaluation adapters
5) Benchmark and tune kernels/thresholds on validation set
