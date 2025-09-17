# PHASE4_POSTPROCESSING.md — Clinical-Grade Post-Processing Pipeline

## 🎯 Phase 4 Goal
Transform raw model probabilities into clinically actionable seizure events through hysteresis thresholding, morphological operations, and intelligent window stitching, achieving target sensitivity at specific FA/24h operating points.

## 📋 Phase 4 Checklist
- [ ] Hysteresis thresholding (dual-tau state machine)
- [ ] Morphological operations (opening/closing with configurable kernels)
- [ ] Minimum duration filtering (≥3s clinical requirement)
- [ ] Window stitching for continuous timelines
- [ ] Event merging (close events within tau_merge seconds)
- [ ] Confidence scoring per event
- [ ] CSV_BI export for Temple evaluation
- [ ] Real-time streaming capability (for future deployment)
- [ ] TDD: comprehensive tests for each component
- [ ] Integration with Phase 3 evaluation pipeline

## 🏗️ Architecture Overview

```
Raw Probabilities (B, T) @ 256 Hz
        ↓
[1] Hysteresis Thresholding
    - τ_on = 0.86 (seizure onset)
    - τ_off = 0.78 (seizure offset)
        ↓
[2] Morphological Operations
    - Binary opening (remove noise)
    - Binary closing (fill gaps)
    - Kernel size: 5 samples (~20ms)
        ↓
[3] Duration Filtering
    - Minimum: 3 seconds (768 samples)
    - Maximum: 600 seconds (clinical)
        ↓
[4] Window Stitching
    - Overlap-add averaging
    - 60s windows, 10s stride
        ↓
[5] Event Generation
    - Continuous mask → event intervals
    - Merge events < tau_merge apart
        ↓
Clinical Events [(start_s, end_s, confidence)]
```

## 🔧 Implementation Files

```
src/experiment/postprocess.py     # Core post-processing pipeline
src/experiment/events.py          # Event generation and manipulation
src/experiment/export.py          # CSV_BI and other format exports
src/experiment/streaming.py       # Real-time processing (future)

tests/test_postprocess.py         # Unit tests for each component
tests/test_events.py              # Event manipulation tests
tests/test_export.py              # Export format validation
tests/test_integration_post.py    # End-to-end post-processing
```

## 📐 Technical Specifications

### 1. Hysteresis Thresholding
State-machine based dual-threshold system preventing rapid oscillations:

```python
@dataclass
class HysteresisState:
    """Track state machine for hysteresis."""
    in_seizure: bool = False
    onset_buffer: int = 0
    offset_buffer: int = 0

def apply_hysteresis(
    probs: torch.Tensor,
    tau_on: float = 0.86,
    tau_off: float = 0.78,
    min_onset_samples: int = 128,  # 0.5s @ 256Hz
    min_offset_samples: int = 256,  # 1.0s @ 256Hz
) -> torch.Tensor:
    """
    Apply dual-threshold hysteresis with temporal stability.

    State transitions:
    - OFF → ON: probs > tau_on for min_onset_samples
    - ON → OFF: probs < tau_off for min_offset_samples

    Args:
        probs: (B, T) probabilities in [0, 1]
        tau_on: Upper threshold for seizure onset
        tau_off: Lower threshold for seizure offset
        min_onset_samples: Minimum samples above tau_on to trigger
        min_offset_samples: Minimum samples below tau_off to end

    Returns:
        Binary mask (B, T) with hysteresis applied
    """
    B, T = probs.shape
    output = torch.zeros_like(probs, dtype=torch.bool)

    for b in range(B):
        state = HysteresisState()

        for t in range(T):
            p = probs[b, t].item()

            if not state.in_seizure:
                # Accumulate onset evidence
                if p > tau_on:
                    state.onset_buffer += 1
                    if state.onset_buffer >= min_onset_samples:
                        state.in_seizure = True
                        # Retroactively mark onset
                        start_idx = max(0, t - min_onset_samples + 1)
                        output[b, start_idx:t+1] = True
                else:
                    state.onset_buffer = 0
            else:
                # In seizure: check for offset
                if p < tau_off:
                    state.offset_buffer += 1
                    if state.offset_buffer >= min_offset_samples:
                        state.in_seizure = False
                        state.offset_buffer = 0
                else:
                    state.offset_buffer = 0
                    output[b, t] = True

                # Continue marking if still in seizure
                if state.in_seizure:
                    output[b, t] = True

    return output
```

**Optimization Note**: For production, implement vectorized version using cumsum tricks or JIT compilation.

### 2. Morphological Operations
Remove noise and fill gaps using binary morphology:

```python
def apply_morphology(
    mask: torch.Tensor,
    opening_kernel: int = 5,
    closing_kernel: int = 5,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Apply binary opening then closing to clean mask.

    Opening: erosion → dilation (removes small noise)
    Closing: dilation → erosion (fills small gaps)

    Args:
        mask: (B, T) binary mask
        opening_kernel: Size for opening operation (samples)
        closing_kernel: Size for closing operation (samples)

    Returns:
        Cleaned binary mask (B, T)
    """
    from scipy.ndimage import binary_opening, binary_closing

    B, T = mask.shape
    output = torch.zeros_like(mask, dtype=torch.bool)

    for b in range(B):
        # Convert to numpy for scipy operations
        mask_np = mask[b].cpu().numpy()

        # Opening to remove noise
        if opening_kernel > 0:
            mask_np = binary_opening(
                mask_np,
                structure=np.ones(opening_kernel)
            )

        # Closing to fill gaps
        if closing_kernel > 0:
            mask_np = binary_closing(
                mask_np,
                structure=np.ones(closing_kernel)
            )

        output[b] = torch.from_numpy(mask_np).to(device)

    return output
```

**GPU Alternative**: Use `torch.nn.functional.max_pool1d` and `min_pool1d` for erosion/dilation.

### 3. Duration Filtering
Enforce clinical minimum/maximum durations:

```python
def filter_duration(
    mask: torch.Tensor,
    min_duration_s: float = 3.0,
    max_duration_s: float = 600.0,
    sampling_rate: int = 256,
) -> torch.Tensor:
    """
    Remove events shorter than min or longer than max duration.

    Clinical requirements:
    - Minimum 3s: Brief events are likely artifacts
    - Maximum 600s: Longer events need segmentation

    Args:
        mask: (B, T) binary mask
        min_duration_s: Minimum event duration in seconds
        max_duration_s: Maximum event duration in seconds
        sampling_rate: Hz

    Returns:
        Filtered mask (B, T)
    """
    min_samples = int(min_duration_s * sampling_rate)
    max_samples = int(max_duration_s * sampling_rate)

    B, T = mask.shape
    output = torch.zeros_like(mask, dtype=torch.bool)

    for b in range(B):
        events = mask_to_events(mask[b], sampling_rate)

        for start_s, end_s in events:
            duration_s = end_s - start_s

            if min_duration_s <= duration_s <= max_duration_s:
                # Keep this event
                start_idx = int(start_s * sampling_rate)
                end_idx = int(end_s * sampling_rate)
                output[b, start_idx:end_idx] = True
            elif duration_s > max_duration_s:
                # Segment long events (optional)
                # For now, truncate to max
                start_idx = int(start_s * sampling_rate)
                end_idx = start_idx + max_samples
                output[b, start_idx:end_idx] = True

    return output
```

### 4. Window Stitching
Reconstruct continuous timeline from overlapping windows:

```python
def stitch_windows(
    window_probs: List[torch.Tensor],
    window_starts: List[int],
    total_length: int,
    window_size: int = 15360,  # 60s @ 256Hz
    stride: int = 2560,         # 10s @ 256Hz
) -> torch.Tensor:
    """
    Overlap-add stitching with weighted averaging.

    Strategy:
    - Sum overlapping probabilities
    - Track contribution count per sample
    - Average by dividing sum by count

    Args:
        window_probs: List of (T_window,) probability tensors
        window_starts: Start indices for each window
        total_length: Total samples in recording
        window_size: Samples per window
        stride: Samples between window starts

    Returns:
        Continuous probability timeline (total_length,)
    """
    prob_sum = torch.zeros(total_length, dtype=torch.float32)
    count = torch.zeros(total_length, dtype=torch.float32)

    for probs, start_idx in zip(window_probs, window_starts):
        end_idx = min(start_idx + window_size, total_length)
        actual_size = end_idx - start_idx

        # Add probabilities
        prob_sum[start_idx:end_idx] += probs[:actual_size]
        count[start_idx:end_idx] += 1.0

    # Average where we have data
    continuous = torch.where(
        count > 0,
        prob_sum / count,
        torch.zeros_like(prob_sum)
    )

    return continuous
```

**Edge Cases**:
- Handle partial windows at recording boundaries
- Ensure no division by zero
- Consider weighted averaging based on distance from window center

### 5. Event Generation and Merging
Convert continuous masks to clinical event intervals:

```python
@dataclass
class SeizureEvent:
    """Clinical seizure event with metadata."""
    start_s: float
    end_s: float
    confidence: float
    merged_count: int = 1

def mask_to_events(
    mask: torch.Tensor,
    sampling_rate: int = 256,
    tau_merge: float = 2.0,
) -> List[SeizureEvent]:
    """
    Convert binary mask to event list with merging.

    Merging logic:
    - Events within tau_merge seconds are combined
    - Confidence = mean probability during event

    Args:
        mask: (T,) binary mask
        sampling_rate: Hz
        tau_merge: Merge events closer than this (seconds)

    Returns:
        List of SeizureEvent objects
    """
    # Find transitions
    mask_np = mask.cpu().numpy()
    diff = np.diff(np.concatenate([[0], mask_np, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    events = []
    for start_idx, end_idx in zip(starts, ends):
        start_s = start_idx / sampling_rate
        end_s = end_idx / sampling_rate

        # Calculate confidence (requires original probs)
        # For now, use 1.0 as placeholder
        confidence = 1.0

        events.append(SeizureEvent(
            start_s=start_s,
            end_s=end_s,
            confidence=confidence
        ))

    # Merge close events
    if tau_merge > 0 and len(events) > 1:
        merged = [events[0]]

        for event in events[1:]:
            last = merged[-1]
            gap = event.start_s - last.end_s

            if gap <= tau_merge:
                # Merge events
                last.end_s = event.end_s
                last.confidence = (
                    last.confidence * last.merged_count +
                    event.confidence
                ) / (last.merged_count + 1)
                last.merged_count += 1
            else:
                merged.append(event)

        events = merged

    return events
```

### 6. Confidence Scoring
Assign confidence to each detected event:

```python
def calculate_event_confidence(
    probs: torch.Tensor,
    event: SeizureEvent,
    sampling_rate: int = 256,
) -> float:
    """
    Calculate confidence score for an event.

    Strategies:
    1. Mean probability during event
    2. Peak probability during event
    3. Percentile (e.g., 75th) during event
    4. Weighted by distance from event center

    Args:
        probs: (T,) probability timeline
        event: SeizureEvent with start/end times
        sampling_rate: Hz

    Returns:
        Confidence score in [0, 1]
    """
    start_idx = int(event.start_s * sampling_rate)
    end_idx = int(event.end_s * sampling_rate)

    event_probs = probs[start_idx:end_idx]

    if len(event_probs) == 0:
        return 0.0

    # Strategy 1: Mean (most stable)
    confidence = float(event_probs.mean().item())

    # Strategy 2: Peak (most optimistic)
    # confidence = float(event_probs.max().item())

    # Strategy 3: 75th percentile (robust)
    # confidence = float(torch.quantile(event_probs, 0.75).item())

    return confidence
```

## 📊 Integration with Phase 3

### Pipeline Integration
```python
def postprocess_predictions(
    raw_probs: torch.Tensor,
    config: PostprocessingConfig,
    sampling_rate: int = 256,
) -> List[List[SeizureEvent]]:
    """
    Complete post-processing pipeline.

    Args:
        raw_probs: (B, T) model outputs
        config: PostprocessingConfig from schemas
        sampling_rate: Hz

    Returns:
        List of event lists, one per batch item
    """
    # 1. Hysteresis
    masks = apply_hysteresis(
        raw_probs,
        tau_on=config.hysteresis.tau_on,
        tau_off=config.hysteresis.tau_off,
    )

    # 2. Morphology
    masks = apply_morphology(
        masks,
        opening_kernel=config.morphology.opening_kernel,
        closing_kernel=config.morphology.closing_kernel,
    )

    # 3. Duration filtering
    masks = filter_duration(
        masks,
        min_duration_s=config.min_duration_s,
        max_duration_s=config.max_duration_s,
        sampling_rate=sampling_rate,
    )

    # 4. Generate events with confidence
    all_events = []
    for b in range(masks.shape[0]):
        events = mask_to_events(
            masks[b],
            sampling_rate=sampling_rate,
            tau_merge=config.tau_merge,
        )

        # Calculate confidence for each event
        for event in events:
            event.confidence = calculate_event_confidence(
                raw_probs[b], event, sampling_rate
            )

        all_events.append(events)

    return all_events
```

### Config Schema Extension
```python
@dataclass
class PostprocessingConfig:
    """Extended post-processing configuration."""

    # Hysteresis
    hysteresis: HysteresisConfig

    # Morphology
    morphology: MorphologyConfig

    # Duration constraints
    min_duration_s: float = 3.0
    max_duration_s: float = 600.0

    # Event merging
    tau_merge: float = 2.0

    # Confidence scoring
    confidence_method: str = "mean"  # mean, peak, percentile
    confidence_percentile: float = 0.75

    # Window stitching
    stitch_method: str = "overlap_add"  # overlap_add, max, weighted

    # Real-time settings (future)
    streaming_enabled: bool = False
    streaming_buffer_s: float = 10.0

@dataclass
class HysteresisConfig:
    tau_on: float = 0.86
    tau_off: float = 0.78
    min_onset_samples: int = 128
    min_offset_samples: int = 256

@dataclass
class MorphologyConfig:
    opening_kernel: int = 5
    closing_kernel: int = 5
    use_gpu: bool = False
```

## 🧪 Test-Driven Development

### Unit Tests (`tests/test_postprocess.py`)
```python
class TestHysteresis:
    def test_basic_transition(self):
        """Test OFF→ON→OFF state transition."""
        probs = torch.tensor([
            [0.0, 0.9, 0.9, 0.9, 0.7, 0.7, 0.0]
        ])
        mask = apply_hysteresis(probs, tau_on=0.86, tau_off=0.78)
        # Should trigger ON at index 1, OFF at index 6

    def test_no_rapid_oscillation(self):
        """Verify hysteresis prevents flickering."""
        probs = torch.tensor([
            [0.85, 0.87, 0.85, 0.87, 0.85, 0.87]
        ])
        mask = apply_hysteresis(probs, tau_on=0.86, tau_off=0.78)
        # Should stay mostly stable despite oscillation

    def test_minimum_duration_enforcement(self):
        """Check min onset/offset sample requirements."""
        # Brief spike shouldn't trigger
        probs = torch.tensor([[0.0, 0.95, 0.0]])
        mask = apply_hysteresis(
            probs, tau_on=0.86, tau_off=0.78,
            min_onset_samples=2
        )
        assert mask.sum() == 0

class TestMorphology:
    def test_noise_removal(self):
        """Opening should remove isolated spikes."""
        mask = torch.tensor([
            [0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0]
        ], dtype=torch.bool)
        cleaned = apply_morphology(mask, opening_kernel=3)
        # Isolated spikes at indices 1 and 9 should be removed

    def test_gap_filling(self):
        """Closing should fill small gaps."""
        mask = torch.tensor([
            [1, 1, 0, 1, 1]
        ], dtype=torch.bool)
        filled = apply_morphology(mask, closing_kernel=3)
        # Gap at index 2 should be filled

class TestDurationFilter:
    def test_minimum_duration(self):
        """Events < 3s should be removed."""
        # 2-second event (512 samples @ 256Hz)
        mask = torch.zeros(1, 1024, dtype=torch.bool)
        mask[0, 100:612] = True
        filtered = filter_duration(mask, min_duration_s=3.0)
        assert filtered.sum() == 0

    def test_maximum_duration(self):
        """Events > 600s should be truncated."""
        # Create 700s event
        mask = torch.ones(1, 179200, dtype=torch.bool)  # 700s @ 256Hz
        filtered = filter_duration(mask, max_duration_s=600.0)
        assert filtered.sum() == 153600  # 600s worth

class TestWindowStitching:
    def test_perfect_overlap(self):
        """Overlapping windows should average correctly."""
        # Two windows with 50% overlap
        window1 = torch.full((100,), 0.8)
        window2 = torch.full((100,), 0.6)

        stitched = stitch_windows(
            [window1, window2],
            [0, 50],
            total_length=150,
            window_size=100,
        )

        # Non-overlapping regions
        assert abs(stitched[0:50].mean() - 0.8) < 1e-6
        assert abs(stitched[100:150].mean() - 0.6) < 1e-6
        # Overlapping region should average
        assert abs(stitched[50:100].mean() - 0.7) < 1e-6

class TestEventGeneration:
    def test_event_merging(self):
        """Close events should merge."""
        events = [
            SeizureEvent(10.0, 15.0, 0.9),
            SeizureEvent(16.0, 20.0, 0.8),  # 1s gap
            SeizureEvent(25.0, 30.0, 0.85),  # 5s gap
        ]
        merged = merge_events(events, tau_merge=2.0)
        assert len(merged) == 2  # First two merge, third stays
        assert merged[0].end_s == 20.0

    def test_confidence_calculation(self):
        """Confidence should reflect probabilities during event."""
        probs = torch.tensor([0.1, 0.9, 0.95, 0.88, 0.1])
        event = SeizureEvent(
            start_s=1/256,
            end_s=4/256,
            confidence=0.0
        )
        conf = calculate_event_confidence(probs, event, sampling_rate=256)
        expected = (0.9 + 0.95 + 0.88) / 3
        assert abs(conf - expected) < 1e-6
```

### Integration Tests (`tests/test_integration_post.py`)
```python
def test_full_pipeline():
    """End-to-end post-processing validation."""
    # Generate synthetic data
    probs = generate_synthetic_probs_with_seizures()

    config = PostprocessingConfig(
        hysteresis=HysteresisConfig(tau_on=0.86, tau_off=0.78),
        morphology=MorphologyConfig(opening_kernel=5, closing_kernel=5),
        min_duration_s=3.0,
        max_duration_s=600.0,
        tau_merge=2.0,
    )

    events = postprocess_predictions(probs, config)

    # Verify clinical constraints
    for event_list in events:
        for event in event_list:
            duration = event.end_s - event.start_s
            assert duration >= 3.0
            assert duration <= 600.0
            assert 0.0 <= event.confidence <= 1.0

def test_real_eeg_scenario():
    """Test with realistic EEG patterns."""
    # Load sample predictions from Phase 3
    probs = torch.load("test_data/sample_predictions.pt")
    config = PostprocessingConfig.from_yaml("configs/postprocess.yaml")

    events = postprocess_predictions(probs, config)

    # Should detect known seizures
    assert len(events[0]) > 0  # At least one seizure detected
```

## 📈 Performance Optimizations

### GPU Acceleration
```python
def apply_morphology_gpu(
    mask: torch.Tensor,
    opening_kernel: int = 5,
    closing_kernel: int = 5,
) -> torch.Tensor:
    """GPU-accelerated morphology using pooling."""
    # Erosion via min pooling
    eroded = F.min_pool1d(
        mask.float().unsqueeze(1),
        kernel_size=opening_kernel,
        stride=1,
        padding=opening_kernel//2
    ).squeeze(1) > 0.5

    # Dilation via max pooling
    dilated = F.max_pool1d(
        eroded.float().unsqueeze(1),
        kernel_size=opening_kernel,
        stride=1,
        padding=opening_kernel//2
    ).squeeze(1) > 0.5

    return dilated
```

### Vectorized Hysteresis
```python
@torch.jit.script
def apply_hysteresis_vectorized(
    probs: torch.Tensor,
    tau_on: float,
    tau_off: float,
) -> torch.Tensor:
    """JIT-compiled vectorized hysteresis."""
    # Implementation using cumsum and torch operations
    # Avoid Python loops for 10x speedup
    pass
```

### Streaming Mode (Future)
```python
class StreamingPostProcessor:
    """Maintain state for real-time processing."""

    def __init__(self, config: PostprocessingConfig):
        self.config = config
        self.buffer = deque(maxlen=config.streaming_buffer_s * 256)
        self.state = HysteresisState()

    def process_chunk(self, chunk: torch.Tensor) -> List[SeizureEvent]:
        """Process incoming chunk maintaining state."""
        # Add to buffer
        # Apply hysteresis with state
        # Return any completed events
        pass
```

## 📊 Metrics and Evaluation

### Clinical Alignment Metrics
```python
def evaluate_postprocessing(
    raw_probs: torch.Tensor,
    processed_events: List[List[SeizureEvent]],
    reference_events: List[List[Tuple[float, float]]],
) -> Dict[str, float]:
    """
    Compare raw vs processed performance.

    Returns:
        - raw_taes: TAES without post-processing
        - processed_taes: TAES with post-processing
        - improvement: Relative improvement
        - fa_reduction: FA/24h reduction
        - merge_ratio: Events merged / total events
    """
    pass
```

## 🚦 CSV_BI Export

### Temple Format Compliance
```python
def export_csv_bi(
    events: List[SeizureEvent],
    filename: str,
    patient_id: str,
    recording_id: str,
    sampling_rate: int = 256,
):
    """
    Export to CSV_BI format for Temple evaluation.

    Format:
    file,start_time,end_time,label,confidence
    """
    rows = []
    for event in events:
        rows.append({
            'file': f"{patient_id}_{recording_id}.edf",
            'start_time': event.start_s,
            'end_time': event.end_s,
            'label': 'seiz',
            'confidence': event.confidence,
        })

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
```

## ⚙️ Configuration

### Default Configuration (`configs/postprocess.yaml`)
```yaml
postprocessing:
  hysteresis:
    tau_on: 0.86
    tau_off: 0.78
    min_onset_samples: 128
    min_offset_samples: 256

  morphology:
    opening_kernel: 5
    closing_kernel: 5
    use_gpu: true

  duration:
    min_duration_s: 3.0
    max_duration_s: 600.0

  events:
    tau_merge: 2.0
    confidence_method: "mean"
    confidence_percentile: 0.75

  stitching:
    method: "overlap_add"
    window_size: 15360
    stride: 2560

  export:
    formats: ["csv_bi", "json", "npy"]
    include_confidence: true
```

## ✅ Definition of Done (Phase 4)

1. [ ] All post-processing components implemented with type hints
2. [ ] Hysteresis prevents rapid state oscillations
3. [ ] Morphology removes noise and fills gaps correctly
4. [ ] Duration filtering enforces clinical constraints
5. [ ] Window stitching produces continuous timelines
6. [ ] Event merging reduces fragmentation
7. [ ] Confidence scores are calibrated and meaningful
8. [ ] CSV_BI export matches Temple format exactly
9. [ ] GPU optimizations available where beneficial
10. [ ] Integration with Phase 3 pipeline seamless
11. [ ] Unit tests achieve >95% coverage
12. [ ] `make q` passes (Ruff + mypy strict)
13. [ ] Performance: <100ms for 1-hour recording
14. [ ] Documentation complete with examples

## 🔗 Integration Points

### From Phase 3
- Receives: Raw probabilities (B, T) from model output
- Config: PostprocessingConfig from schemas.py
- Metrics: TAES, FA/24h calculated on processed events

### To Phase 5 (Deployment)
- Provides: Clinical events with confidence scores
- Exports: CSV_BI for evaluation, JSON for API
- Streaming: Real-time processing capability

## 🚀 Performance Targets

- **Latency**: <100ms for 1-hour recording
- **Memory**: <1GB for 24-hour recording
- **Accuracy**: Improve TAES by >10% vs raw
- **FA Reduction**: Reduce FA/24h by >30%
- **Clinical**: Maintain >95% sensitivity @ 10 FA/24h

## 📝 Implementation Notes

### SOLID Principles
- **Single Responsibility**: Each function does one thing
- **Open/Closed**: Extensible for new methods via config
- **Liskov**: All processors follow same interface
- **Interface Segregation**: Minimal dependencies
- **Dependency Inversion**: Config-driven behavior

### Critical Considerations
1. **Determinism**: Same input → same output always
2. **Numerical Stability**: No NaN/Inf propagation
3. **Edge Cases**: Handle empty events, single samples
4. **Memory**: Stream large recordings in chunks
5. **Compatibility**: Support both GPU and CPU paths

### Common Pitfalls to Avoid
- Don't modify probabilities in-place
- Ensure thread safety for parallel processing
- Handle timezone/timestamp conversions carefully
- Validate all indices before array access
- Test with adversarial inputs (all 0s, all 1s, random)

---

**Status**: Ready for TDD implementation 🚀
**Estimated Time**: 3-4 days
**Prerequisites**: Phase 3 complete, model outputs available
**Owner**: Post-processing specialist 🔬

**Next Steps**:
1. Write comprehensive unit tests first (TDD)
2. Implement core functions one by one
3. Integrate with Phase 3 pipeline
4. Benchmark performance on real data
5. Fine-tune thresholds on validation set