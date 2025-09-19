"""Streaming inference with stateful hysteresis for real-time seizure detection.

This module provides stateful post-processing for continuous EEG streams,
maintaining hysteresis state across chunks for seamless real-time detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.brain_brr.config.schemas import PostprocessingConfig
from src.brain_brr.post.postprocess import apply_morphology


@dataclass
class HysteresisState:
    """Maintains hysteresis state across stream chunks."""

    in_event: bool = False
    onset_counter: int = 0
    offset_counter: int = 0
    onset_start_global: int = -1
    total_samples_processed: int = 0


class StreamingPostProcessor:
    """Stateful post-processor for real-time streaming inference.

    Maintains hysteresis state across chunks to handle events that
    span chunk boundaries correctly.
    """

    def __init__(
        self,
        config: PostprocessingConfig,
        sampling_rate: int = 256,
        chunk_size: int = 2560,  # 10 seconds at 256 Hz
    ):
        """Initialize streaming processor.

        Args:
            config: Post-processing configuration
            sampling_rate: Sampling rate in Hz
            chunk_size: Expected chunk size in samples
        """
        self.config = config
        self.sampling_rate = sampling_rate
        self.chunk_size = chunk_size

        # Hysteresis state
        self.state = HysteresisState()

        # Buffer for morphology operations (need context at boundaries)
        self.buffer_size = max(config.morphology.opening_kernel, config.morphology.closing_kernel)
        self.prob_buffer: list[float] = []
        self.mask_buffer: list[bool] = []

    def reset(self) -> None:
        """Reset all internal state for new stream."""
        self.state = HysteresisState()
        self.prob_buffer = []
        self.mask_buffer = []

    def process_chunk(
        self, probs: torch.Tensor | np.ndarray, return_events: bool = False
    ) -> tuple[torch.Tensor, list[tuple[float, float]] | None]:
        """Process a chunk of probabilities maintaining state.

        Args:
            probs: Probability chunk (T,) or (1, T)
            return_events: If True, also return events found in this chunk

        Returns:
            masks: Binary seizure mask for this chunk
            events: List of (start_s, end_s) if return_events=True, else None
        """
        # Convert to numpy for processing
        if isinstance(probs, torch.Tensor):
            prob_np = probs.squeeze().cpu().numpy()
            device = probs.device
        else:
            prob_np = np.asarray(probs).squeeze()
            device = torch.device("cpu")

        chunk_len = len(prob_np)

        # Apply stateful hysteresis
        mask = np.zeros(chunk_len, dtype=bool)
        tau_on = self.config.hysteresis.tau_on
        tau_off = self.config.hysteresis.tau_off
        min_onset = self.config.hysteresis.min_onset_samples
        min_offset = self.config.hysteresis.min_offset_samples

        events_in_chunk: list[tuple[float, float]] = []
        event_start_local = -1

        for i in range(chunk_len):
            global_idx = self.state.total_samples_processed + i

            if not self.state.in_event:
                # Check for onset
                if prob_np[i] >= tau_on:
                    self.state.onset_counter += 1
                    if self.state.onset_counter == 1:
                        self.state.onset_start_global = global_idx
                        event_start_local = i
                    if self.state.onset_counter >= min_onset:
                        # Retroactively mark onset in current chunk
                        if event_start_local >= 0:
                            mask[event_start_local : i + 1] = True
                        else:
                            # Onset started in previous chunk
                            mask[: i + 1] = True
                        self.state.in_event = True
                        self.state.onset_counter = 0
                else:
                    self.state.onset_counter = 0
                    self.state.onset_start_global = -1
                    event_start_local = -1
            else:
                # In event, check for offset
                if prob_np[i] < tau_off:
                    self.state.offset_counter += 1
                    if self.state.offset_counter < min_offset:
                        mask[i] = True
                    if self.state.offset_counter >= min_offset:
                        # Event ended
                        if return_events and event_start_local >= 0:
                            start_s = event_start_local / self.sampling_rate
                            end_s = (i - min_offset + 1) / self.sampling_rate
                            if end_s > start_s:
                                events_in_chunk.append((start_s, end_s))
                        self.state.in_event = False
                        self.state.offset_counter = 0
                        event_start_local = -1
                else:
                    mask[i] = True
                    self.state.offset_counter = 0

        # Handle ongoing event at chunk boundary
        if return_events and self.state.in_event and event_start_local >= 0:
            # Event continues into next chunk
            start_s = event_start_local / self.sampling_rate
            end_s = chunk_len / self.sampling_rate
            events_in_chunk.append((start_s, end_s))

        # Update total samples processed
        self.state.total_samples_processed += chunk_len

        # Convert mask to tensor
        mask_tensor = torch.from_numpy(mask).to(device)

        # Apply morphology if enabled (simplified for streaming)
        if self.config.morphology.opening_kernel > 1 or self.config.morphology.closing_kernel > 1:
            # Add batch dimension for morphology function
            mask_batch = mask_tensor.unsqueeze(0)
            mask_batch = apply_morphology(
                mask_batch,
                opening_kernel=self.config.morphology.opening_kernel,
                closing_kernel=self.config.morphology.closing_kernel,
                use_gpu=device.type == "cuda",
            )
            mask_tensor = mask_batch.squeeze(0)

        # Duration filtering would need buffering across chunks
        # For real-time, we skip it or apply it in post-processing

        return mask_tensor, events_in_chunk if return_events else None

    def get_state_dict(self) -> dict[str, Any]:
        """Get current state for checkpointing.

        Returns:
            State dictionary that can be saved/loaded
        """
        return {
            "in_event": self.state.in_event,
            "onset_counter": self.state.onset_counter,
            "offset_counter": self.state.offset_counter,
            "onset_start_global": self.state.onset_start_global,
            "total_samples_processed": self.state.total_samples_processed,
            "prob_buffer": self.prob_buffer,
            "mask_buffer": self.mask_buffer,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state from checkpoint.

        Args:
            state_dict: Previously saved state dictionary
        """
        self.state.in_event = state_dict["in_event"]
        self.state.onset_counter = state_dict["onset_counter"]
        self.state.offset_counter = state_dict["offset_counter"]
        self.state.onset_start_global = state_dict["onset_start_global"]
        self.state.total_samples_processed = state_dict["total_samples_processed"]
        self.prob_buffer = state_dict.get("prob_buffer", [])
        self.mask_buffer = state_dict.get("mask_buffer", [])
