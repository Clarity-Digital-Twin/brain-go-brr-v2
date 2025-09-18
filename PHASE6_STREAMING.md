# PHASE 6 — Streaming & Real‑Time Inference (Iron‑Clad, TDD)

Purpose: turn per‑timestep probabilities into clinically usable, low‑latency events in real time with stateful hysteresis, morphology, and duration logic across chunk boundaries.

## Scope & Status
- Target: online (streaming) inference with O(1) memory per channel and bounded latency, matching offline semantics within tolerance.
- Current state: baseline streaming utilities exist (see `src/experiment/streaming.py`) with stateful hysteresis; GPU morphology path available via pooling. This document locks the end‑state APIs, tests, and DOD to finish and harden Phase 6.

## Goals
- Stateful hysteresis across chunk boundaries (τ_on/τ_off with min_onset/min_offset stability windows).
- Optional morphology (opening → closing) with boundary padding to avoid edge artifacts (CPU SciPy; GPU pooling).
- Duration filtering across chunks (min/max); finalize events only when offset stability satisfied; segment long events.
- Deterministic, low‑latency outputs; serialization of state for resumable streaming.
- Equivalence-to-offline tests: streaming ≈ offline within one‑sample tolerance at chunk seams.

## Architecture
- StreamingState (dataclass)
  - in_seizure: bool
  - onset_count, offset_count: int (stability windows carried over)
  - tail_buffer: optional Tensor/ndarray for morphology context
  - active_event_start_idx: optional int (global index within record/stream)
  - sample_offset: int (global time index of current chunk start)

- StreamingProcessor
  - Config: ties to PostprocessingConfig + StreamingConfig (see below)
  - process_chunk(probs: Tensor[B, T]) -> list[list[SeizureEvent]] (completed), StreamingState
  - flush() -> finalize any open events at end of stream
  - serialize_state()/deserialize_state() for checkpointing
  - Device-aware morphology (CPU SciPy, GPU pooling)

Data flow per chunk (per batch):
1) Optional pre/post padding for morphology; carry tail_buffer to next chunk
2) Hysteresis with stateful counters (min_onset/min_offset) → mask
3) Morphology opening→closing (if enabled) on mask; crop padding
4) Duration filtering with cross‑chunk awareness: do not finalize active event until offset stability is satisfied
5) Convert finalized True regions to SeizureEvent with absolute time (seconds)

## APIs (Target)
Location: `src/experiment/streaming.py`

- class StreamingConfig(BaseModel)
  - chunk_size_s: float = 10.0
  - hop_s: float = 1.0
  - pre_pad_s: float = 0.25
  - post_pad_s: float = 0.25
  - use_gpu: bool = False
  - latency_budget_s: float = 0.5

- class StreamingState(TypedDict | dataclass)
  - in_seizure: bool
  - onset_count: int
  - offset_count: int
  - tail_buffer: torch.Tensor | None
  - active_event_start_idx: int | None
  - sample_offset: int

- class StreamingProcessor:
  - __init__(post_cfg: PostprocessingConfig, stream_cfg: StreamingConfig, fs: int = 256)
  - process_chunk(probs: torch.Tensor) -> tuple[list[list[SeizureEvent]], StreamingState]
  - flush() -> list[list[SeizureEvent]]
  - state property with serialize()/deserialize() helpers

Notes
- Probabilities are `B×T` (batch of channels/records per stream context). If batch is 1 for online use, API remains consistent.
- Confidence scores: compute on finalized events with the same method as offline (mean/peak/percentile); configurable via `post_cfg.events`.

## Configuration
- Extend `src/experiment/schemas.py`:
  - Add `StreamingConfig` (fields above) under Postprocessing/Evaluation section.
  - Add `post_cfg.streaming_enabled: bool = False` to gate streaming path in evaluation if desired.
  - All sizes expressed in seconds; convert to samples via `fs` internally; enforce odd kernels.

## TDD Plan
Unit tests (new: `tests/test_streaming.py`)
- Hysteresis state carryover
  - Enter on τ_on only after min_onset_samples across chunk boundary
  - Exit on τ_off only after min_offset_samples across boundary
  - Equality at thresholds: probs == τ_on/τ_off does not flip state

- Morphology boundary correctness
  - With pre/post padding, streaming morphology output matches offline within 1 sample at seams
  - GPU pooling path ≈ CPU SciPy within tolerance for multiple odd kernels

- Duration filtering across chunks
  - Short blips < min_duration_s spanning boundary are removed
  - Long events segmented at max_duration_s uniformly across chunks

- Serialization
  - serialize()/deserialize() round‑trip without changing subsequent outputs
  - State resumes correctly after process restart

Integration tests
- Streaming ≈ Offline
  - Generate synthetic traces; compare events from StreamingProcessor over chunks vs offline `postprocess_predictions` + `mask_to_events`
  - Assert same number of events, similar start/end within tolerance, identical confidence scores within 1e‑6 when windows align

- Throughput/latency budget
  - With `chunk_size_s=10` and `hop_s=1`, ensure processing time per chunk < latency_budget_s on CPU for small B

## CLI (Optional)
Add subcommand in `src/cli.py`:
- `stream` command
  - Inputs: EDF file or live source adapter
  - Outputs: JSONL of events or CSV_BI per rolling horizon
  - Flags: `--chunk-size`, `--hop`, `--latency-budget`, `--gpu/--cpu`, `--confidence-method`

## Performance & Memory Targets
- O(B×C) memory (no growth with stream length)
- CPU baseline: < 100 ms per 10 s chunk at B=1 (typical laptop), morphology enabled
- GPU optional path: parity within tolerance; faster morphology/dilation/erosion

## Edge Cases
- Start/end of stream: active event may begin in first chunk or end at final chunk; flush() must close an active event if policy dictates
- Gaps between chunks (dropped frames): treat as silence unless explicitly handled by source adapter
- Device mismatch: if model on CUDA but streaming configured for CPU morphology, keep morphology on CPU (prob tensors are on CPU post‑sigmoid)
- Variable sampling rates: `fs` is required; convert s→samples consistently; default 256 Hz

## Definition of Done (DOD)
- APIs implemented as above in `src/experiment/streaming.py` with full type hints
- Unit + integration tests added and passing (`tests/test_streaming.py`), including GPU morphology parity (skipped if no CUDA)
- make q passes (ruff + format + mypy)
- CLI (optional) flows; documentation updated (this file + README link)
- Equivalence tests vs offline eventization pass within tolerance
- Example script/notebook demonstrates end‑to‑end streaming on a short EDF

## Risks & Mitigations
- Boundary artifacts: use padding and carry tail_buffer; validate with seam tests
- Latency overruns: expose morphology toggle and kernel sizes; benchmark; gate GPU path by availability
- Divergence vs offline: enforce identical τ_on/τ_off semantics and stability windows; unit tests guard regressions

## References
- Hysteresis and morphology specs: `PHASE4_POSTPROCESSING.md`
- Evaluation/TAES and FA targets: `PHASE5_EVALUATION.md`
- Model dispatch notes (CPU/GPU): `AGENTS.md`

