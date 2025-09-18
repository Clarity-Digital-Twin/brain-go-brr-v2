# PHASE 6 — Streaming & Real‐Time Inference (Iron‐Clad, TDD)

**Purpose**: Enhance existing streaming implementation to handle complete post-processing pipeline in real-time with stateful hysteresis, morphology, duration filtering, and event tracking across chunk boundaries.

## 🎯 Scope & Current State

### What We Have (Implemented)
- ✅ **HysteresisState** dataclass tracking state across chunks
- ✅ **StreamingPostProcessor** class with stateful hysteresis
- ✅ **Morphology support** via `apply_morphology()` (CPU/GPU paths)
- ✅ **State serialization** via `get_state_dict()`/`load_state_dict()`
- ✅ **Basic tests** covering state persistence, event spanning, reset
- ✅ **GPU processing** support for morphology

### What's Missing (Phase 6 Goals)
- ❌ **Duration filtering** across chunks (currently commented out as "needs buffering")
- ❌ **Event finalization** with proper confidence scoring
- ❌ **SeizureEvent** integration (returns raw masks/tuples, not SeizureEvent objects)
- ❌ **Buffer management** for morphology context at boundaries
- ❌ **Flush method** to finalize pending events at stream end
- ❌ **Performance benchmarking** against latency targets
- ❌ **Equivalence tests** comparing streaming vs offline outputs
- ❌ **CLI streaming command** for real-time inference

## 🏗️ Architecture (Current + Enhancements)

### Existing Classes (Keep)
```python
@dataclass
class HysteresisState:
    """Current state tracking (KEEP AS-IS)."""
    in_event: bool = False
    onset_counter: int = 0
    offset_counter: int = 0
    onset_start_global: int = -1
    total_samples_processed: int = 0
```

### Enhanced StreamingPostProcessor
```python
class StreamingPostProcessor:
    """Enhanced with duration filtering and event tracking."""

    def __init__(self, config, sampling_rate=256, chunk_size=2560):
        # Existing initialization
        self.state = HysteresisState()

        # NEW: Event tracking across chunks
        self.active_events: list[tuple[int, int]] = []  # (start_idx, current_end_idx)
        self.completed_events: list[SeizureEvent] = []

        # NEW: Duration filtering buffer
        self.event_buffer: deque[tuple[int, int, np.ndarray]] = deque()
        self.min_duration_samples = int(config.duration.min_duration_s * sampling_rate)
        self.max_duration_samples = int(config.duration.max_duration_s * sampling_rate)

        # NEW: Morphology context buffer
        self.morphology_buffer: np.ndarray | None = None
        self.buffer_overlap = max(
            config.morphology.opening_kernel,
            config.morphology.closing_kernel
        ) // 2
```

### New Additions
```python
class StreamingConfig(BaseModel):
    """Configuration for streaming inference."""
    chunk_size_samples: int = Field(default=2560, description="Chunk size (10s @ 256Hz)")
    buffer_overlap: int = Field(default=256, description="Overlap for morphology context")
    latency_budget_ms: int = Field(default=100, description="Max processing time per chunk")
    enable_duration_filtering: bool = Field(default=True)
    enable_morphology: bool = Field(default=True)
```

## 📐 Enhanced API (Building on Existing)

### Methods to Add to StreamingPostProcessor
```python
def process_chunk_with_events(
    self, probs: torch.Tensor, return_completed: bool = True
) -> tuple[torch.Tensor, list[SeizureEvent]]:
    """Enhanced process_chunk that returns SeizureEvent objects."""
    # 1. Apply hysteresis (existing)
    masks = self._apply_stateful_hysteresis(probs)

    # 2. Apply morphology with context buffer (enhance)
    if self.config.morphology.opening_kernel > 1:
        masks = self._apply_buffered_morphology(masks)

    # 3. Track events across chunks (new)
    self._update_event_tracking(masks)

    # 4. Apply duration filtering (new)
    completed_events = self._finalize_completed_events()

    return masks, completed_events if return_completed else []

def flush(self) -> list[SeizureEvent]:
    """Finalize any pending events at stream end."""
    final_events = []

    # Close any active hysteresis event
    if self.state.in_event and self.state.onset_start_global >= 0:
        end_idx = self.state.total_samples_processed
        duration = end_idx - self.state.onset_start_global

        if duration >= self.min_duration_samples:
            event = self._create_seizure_event(
                self.state.onset_start_global,
                end_idx
            )
            final_events.append(event)

    # Clear all buffers
    self.reset()
    return final_events

def _create_seizure_event(
    self, start_idx: int, end_idx: int, probs: np.ndarray | None = None
) -> SeizureEvent:
    """Create SeizureEvent with confidence scoring."""
    start_s = start_idx / self.sampling_rate
    end_s = end_idx / self.sampling_rate

    # Calculate confidence (mean/peak/percentile)
    confidence = 0.9  # Default if no probs available
    if probs is not None:
        method = self.config.events.confidence_method
        if method == "mean":
            confidence = float(np.mean(probs[start_idx:end_idx]))
        elif method == "peak":
            confidence = float(np.max(probs[start_idx:end_idx]))
        elif method == "percentile":
            pct = self.config.events.confidence_percentile
            confidence = float(np.percentile(probs[start_idx:end_idx], pct * 100))

    return SeizureEvent(start_s=start_s, end_s=end_s, confidence=confidence)
```

## 🧪 TDD Plan (Enhance Existing Tests)

### Existing Tests (Keep)
- ✅ `test_state_initialization()` - Initial state verification
- ✅ `test_chunk_processing()` - Basic chunk processing
- ✅ `test_state_persistence_across_chunks()` - Hysteresis state carryover
- ✅ `test_event_spanning_chunks()` - Events across boundaries
- ✅ `test_reset_state()` - State reset functionality
- ✅ `test_state_serialization()` - Save/load state
- ✅ `test_gpu_processing()` - GPU support

### New Tests to Add
```python
def test_duration_filtering_across_chunks():
    """Test that duration filtering works across chunk boundaries."""
    # Create event that spans 3 chunks
    # Chunk 1: event starts (1s duration)
    # Chunk 2: event continues (10s duration)
    # Chunk 3: event ends (1s duration)
    # Total: 12s (should be kept as > min_duration_s=3.0)

def test_long_event_segmentation():
    """Test segmentation of events > max_duration_s."""
    # Create event lasting 700s (> max_duration_s=600)
    # Should be segmented into 600s + 100s

def test_morphology_with_buffer_context():
    """Test morphology operations maintain continuity at boundaries."""
    # Process same signal in:
    # 1. Single pass (offline)
    # 2. Multiple chunks (streaming)
    # Compare outputs - should match within 1 sample at boundaries

def test_flush_pending_events():
    """Test flush() finalizes pending events correctly."""
    # Start event in chunk but don't complete
    # Call flush() - should finalize if duration sufficient

def test_confidence_scoring_methods():
    """Test mean/peak/percentile confidence calculations."""
    # Known probability sequence
    # Verify each method produces expected confidence

def test_streaming_vs_offline_equivalence():
    """Integration test: streaming ≈ offline processing."""
    # Generate 5-minute synthetic signal
    # Process offline vs streaming (30s chunks)
    # Compare:
    #   - Number of events (exact match)
    #   - Event boundaries (within 2 samples)
    #   - Confidence scores (within 1e-4)

def test_latency_budget():
    """Performance test: meet latency requirements."""
    # Process 10s chunk (2560 samples)
    # Measure time - should be < 100ms on CPU
    # Test with/without morphology
```

## 🖥️ CLI Integration

### Add to `src/cli.py`
```python
@cli.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("--checkpoint", required=True, type=click.Path(exists=True))
@click.option("--chunk-size", default=10.0, help="Chunk size in seconds")
@click.option("--output", type=click.Choice(["json", "csv_bi", "console"]), default="console")
@click.option("--gpu/--cpu", default=False, help="Use GPU for processing")
def stream(
    input_path: Path,
    checkpoint: Path,
    chunk_size: float,
    output: str,
    gpu: bool,
) -> None:
    """Stream inference on EDF file or directory."""
    # Load model
    model = load_checkpoint(checkpoint)
    device = "cuda" if gpu and torch.cuda.is_available() else "cpu"

    # Initialize streaming processor
    config = Config.from_checkpoint(checkpoint)
    processor = StreamingPostProcessor(
        config.postprocessing,
        sampling_rate=256,
        chunk_size=int(chunk_size * 256)
    )

    # Process EDF in chunks
    for chunk in edf_chunk_generator(input_path, chunk_size):
        probs = model(chunk)
        masks, events = processor.process_chunk_with_events(probs)

        # Output events
        if output == "console" and events:
            for event in events:
                console.print(f"[yellow]Seizure:[/yellow] {event.start_s:.1f}s - {event.end_s:.1f}s (confidence: {event.confidence:.2f})")

    # Flush pending events
    final_events = processor.flush()
    if final_events:
        console.print(f"[green]Flushed {len(final_events)} pending events[/green]")
```

## ⚡ Performance Requirements

| Metric | Target | Rationale |
|--------|--------|-----------|
| **Memory** | O(1) per channel | No growth with stream length |
| **Latency** | < 100ms per 10s chunk | Real-time requirement |
| **CPU Throughput** | > 100× realtime | 10s processed in < 100ms |
| **GPU Throughput** | > 1000× realtime | Optional acceleration |
| **Accuracy** | Within 1 sample of offline | Clinical equivalence |

## 🔧 Implementation Plan

### Phase 6.1: Core Enhancements (2 days)
1. Add duration filtering with event buffer
2. Implement morphology context buffer
3. Create SeizureEvent integration
4. Add flush() method
5. Enhance state serialization

### Phase 6.2: Testing (2 days)
1. Write duration filtering tests
2. Add morphology continuity tests
3. Create streaming vs offline equivalence tests
4. Benchmark performance
5. Test edge cases (partial events, long events)

### Phase 6.3: CLI & Integration (1 day)
1. Add streaming CLI command
2. Create EDF chunk generator
3. Add output formatters (JSON, CSV_BI)
4. Integration test with real EDF files
5. Documentation and examples

## ✅ Definition of Done (DOD)

### Must Have (Core Requirements)
- [ ] Enhanced `StreamingPostProcessor` with duration filtering
- [ ] `SeizureEvent` integration returning proper objects
- [ ] `flush()` method for stream termination
- [ ] Morphology buffer management for continuity
- [ ] State serialization includes all new fields
- [ ] Tests: duration filtering across chunks
- [ ] Tests: streaming vs offline equivalence (< 1 sample difference)
- [ ] Performance: < 100ms per 10s chunk on CPU
- [ ] `make q` passes (ruff + mypy + tests)

### Should Have (Completeness)
- [ ] CLI `stream` command with EDF support
- [ ] Output formatters (JSON, CSV_BI, console)
- [ ] Confidence scoring (mean/peak/percentile)
- [ ] Long event segmentation (> 600s)
- [ ] GPU morphology parity tests
- [ ] Benchmark suite with performance reports

### Nice to Have (Polish)
- [ ] WebSocket server for real-time streaming
- [ ] Grafana/Prometheus metrics export
- [ ] Docker container with streaming service
- [ ] Example Jupyter notebook
- [ ] Video demo of real-time detection

## ⚠️ Critical Considerations

### Algorithm Consistency
- **Hysteresis**: Exact same thresholds and stability windows as offline
- **Morphology**: Buffer overlap to prevent boundary artifacts
- **Duration**: Track events across chunks; don't drop at boundaries
- **Confidence**: Calculate on complete events, not partial chunks

### Performance Constraints
- **Memory**: Fixed buffers; no unbounded growth
- **Latency**: Benchmark on target hardware (RPi4, Jetson Nano)
- **Accuracy**: Clinical equivalence to offline processing

### Edge Cases
- Stream start: No prior context for morphology
- Stream end: Pending events need finalization
- Dropped chunks: Handle gracefully with state reset
- Variable chunk sizes: Support but maintain sample tracking

## 🔗 Cross-References

| Phase | Dependency | Details |
|-------|------------|---------|
| **Phase 4** | Core algorithms | Hysteresis, morphology, duration filtering |
| **Phase 5** | Evaluation metrics | TAES targets, confidence scoring |
| **Phase 3** | Model inference | Getting probabilities from model |
| **AGENTS.md** | Architecture | Device handling, performance targets |
| **schemas.py** | Configuration | PostprocessingConfig, EventsConfig |

## 📚 Key Implementation Files

```
src/experiment/
├── streaming.py          # Main implementation (enhance existing)
├── events.py            # SeizureEvent dataclass (use existing)
├── postprocess.py       # Reference algorithms (use existing)
└── schemas.py           # Add StreamingConfig

tests/
├── test_streaming.py    # Enhance with new tests
└── test_integration.py  # Add streaming vs offline tests

src/
└── cli.py              # Add stream command
```

---

**Status**: Ready for implementation with clear requirements and existing foundation 🚀
**Estimated Time**: 5 days (2 core + 2 test + 1 integration)
**Prerequisites**: Phase 4-5 complete ✅, streaming.py exists ✅
**Risk Level**: Low - building on working code
**Impact**: Enables real-time clinical deployment