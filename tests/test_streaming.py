"""Tests for streaming inference with stateful hysteresis."""

import pytest
import torch

from src.experiment.schemas import (
    HysteresisConfig,
    MorphologyConfig,
    PostprocessingConfig,
)
from src.experiment.streaming import StreamingPostProcessor


class TestStreamingPostProcessor:
    """Test streaming post-processor."""

    @pytest.fixture
    def config(self) -> PostprocessingConfig:
        """Create test config."""
        return PostprocessingConfig(
            hysteresis=HysteresisConfig(
                tau_on=0.86,
                tau_off=0.78,
                min_onset_samples=2,
                min_offset_samples=3,
            ),
            morphology=MorphologyConfig(
                opening_kernel=3,
                closing_kernel=5,
            ),
        )

    @pytest.fixture
    def processor(self, config: PostprocessingConfig) -> StreamingPostProcessor:
        """Create processor instance."""
        return StreamingPostProcessor(config, sampling_rate=256, chunk_size=256)

    def test_state_initialization(self, processor: StreamingPostProcessor) -> None:
        """Test initial state."""
        assert processor.state.in_event is False
        assert processor.state.onset_counter == 0
        assert processor.state.offset_counter == 0
        assert processor.state.total_samples_processed == 0

    def test_chunk_processing(self, processor: StreamingPostProcessor) -> None:
        """Test processing single chunk."""
        # Create chunk with rising/falling pattern
        probs = torch.tensor([0.5, 0.9, 0.95, 0.85, 0.7, 0.6, 0.5])
        masks, _events = processor.process_chunk(probs, return_events=True)

        assert masks.shape == probs.shape
        assert processor.state.total_samples_processed == 7

    def test_state_persistence_across_chunks(self, processor: StreamingPostProcessor) -> None:
        """Test hysteresis state maintained across chunk boundaries."""
        # Disable morphology for this test (it removes short events)
        processor.config.morphology.opening_kernel = 1
        processor.config.morphology.closing_kernel = 1

        # First chunk: start event but don't end it
        chunk1 = torch.tensor([0.5, 0.87, 0.88, 0.89])  # Triggers at sample 1
        masks1, _ = processor.process_chunk(chunk1)

        # Should be in event state (needs 2 samples above tau_on)
        assert processor.state.in_event is True
        assert torch.any(masks1[-2:])  # Last 2 samples should be marked

        # Second chunk: continue event
        chunk2 = torch.tensor([0.85, 0.84, 0.83, 0.82])  # Still above tau_off
        masks2, _ = processor.process_chunk(chunk2)

        # Should still be in event
        assert processor.state.in_event is True
        assert torch.all(masks2)  # All samples should be True (above tau_off)

        # Third chunk: end event
        chunk3 = torch.tensor([0.75, 0.70, 0.65, 0.60])  # Below tau_off
        _masks3, _ = processor.process_chunk(chunk3)

        # Should exit event after min_offset_samples (3)
        assert processor.state.in_event is False

    def test_event_spanning_chunks(self, processor: StreamingPostProcessor) -> None:
        """Test events that span multiple chunks."""
        # Set min samples to 1 for simpler testing
        processor.config.hysteresis.min_onset_samples = 1
        processor.config.hysteresis.min_offset_samples = 1
        # Disable morphology for this test
        processor.config.morphology.opening_kernel = 1
        processor.config.morphology.closing_kernel = 1

        # Chunk 1: Event starts (0.9 > 0.86)
        chunk1 = torch.tensor([0.5, 0.9, 0.88, 0.87])
        masks1, _events1 = processor.process_chunk(chunk1, return_events=True)

        # Chunk 2: Event continues and ends (0.77 < 0.78)
        chunk2 = torch.tensor([0.85, 0.77, 0.70, 0.65])
        masks2, _events2 = processor.process_chunk(chunk2, return_events=True)

        # Should have detected parts of the event in both chunks
        assert torch.any(masks1[1:])  # Event starts at sample 1
        assert masks2[0]  # First sample should still be in event

    def test_reset_state(self, processor: StreamingPostProcessor) -> None:
        """Test state reset."""
        # Process some data
        probs = torch.rand(10)
        processor.process_chunk(probs)

        # State should be modified
        assert processor.state.total_samples_processed == 10

        # Reset
        processor.reset()

        # State should be fresh
        assert processor.state.total_samples_processed == 0
        assert processor.state.in_event is False

    def test_state_serialization(self, processor: StreamingPostProcessor) -> None:
        """Test saving and loading state."""
        # Process some chunks to build state
        chunk1 = torch.tensor([0.5, 0.9, 0.88])
        processor.process_chunk(chunk1)

        # Save state
        state_dict = processor.get_state_dict()
        assert "in_event" in state_dict
        assert "total_samples_processed" in state_dict
        assert state_dict["total_samples_processed"] == 3

        # Create new processor and load state
        new_processor = StreamingPostProcessor(processor.config, sampling_rate=256)
        new_processor.load_state_dict(state_dict)

        # States should match
        assert new_processor.state.in_event == processor.state.in_event
        assert (
            new_processor.state.total_samples_processed == processor.state.total_samples_processed
        )

    def test_gpu_processing(self, processor: StreamingPostProcessor) -> None:
        """Test GPU processing if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        probs = torch.rand(10).cuda()
        masks, _ = processor.process_chunk(probs)

        assert masks.device == probs.device
        assert masks.shape == probs.shape
