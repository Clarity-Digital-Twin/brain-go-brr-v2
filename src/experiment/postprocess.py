"""Post-processing operations for seizure detection.

This module implements hysteresis thresholding, morphological operations,
duration filtering, and window stitching for converting raw probabilities
to clinical seizure events.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage  # type: ignore[import-untyped]
from torch.nn import functional

from src.experiment.schemas import (
    PostprocessingConfig,
)


def apply_hysteresis(
    probs: torch.Tensor,
    tau_on: float = 0.86,
    tau_off: float = 0.78,
    min_onset_samples: int = 1,
    min_offset_samples: int = 1,
) -> torch.Tensor:
    """Apply hysteresis thresholding with optional stability windows.

    Args:
        probs: Probabilities tensor (B, T) in [0, 1]
        tau_on: Upper threshold to enter seizure state
        tau_off: Lower threshold to exit seizure state
        min_onset_samples: Minimum samples above tau_on to enter (stability)
        min_offset_samples: Minimum samples below tau_off to exit (stability)

    Returns:
        Binary mask tensor (B, T) as bool
    """
    if tau_on <= tau_off:
        raise ValueError(f"tau_on ({tau_on}) must be > tau_off ({tau_off})")

    batch_size, seq_len = probs.shape
    device = probs.device

    # Vectorized implementation for better performance
    if min_onset_samples == 1 and min_offset_samples == 1:
        # Fast path: no stability windows needed
        masks = torch.zeros_like(probs, dtype=torch.bool, device=device)

        # Process all batches in parallel
        above_on = probs >= tau_on
        below_off = probs < tau_off

        for b in range(batch_size):
            mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
            state = False

            for i in range(seq_len):
                if not state:
                    if above_on[b, i]:
                        state = True
                        mask[i] = True
                else:
                    if below_off[b, i]:
                        state = False
                    else:
                        mask[i] = True

            masks[b] = mask
    else:
        # Full implementation with stability windows
        masks = torch.zeros_like(probs, dtype=torch.bool, device=device)

        for b in range(batch_size):
            prob_seq = probs[b].cpu().numpy()
            mask = np.zeros(seq_len, dtype=bool)

            in_event = False
            onset_counter = 0
            offset_counter = 0
            onset_start = -1

            for i in range(seq_len):
                if not in_event:
                    # Check for onset (>= to include equality)
                    if prob_seq[i] >= tau_on:
                        onset_counter += 1
                        if onset_counter == 1:
                            onset_start = i
                        if onset_counter >= min_onset_samples:
                            # Retroactively mark onset
                            mask[onset_start : i + 1] = True
                            in_event = True
                            onset_counter = 0
                    else:
                        onset_counter = 0
                        onset_start = -1
                else:
                    # In event, check for offset or continue
                    if prob_seq[i] < tau_off:
                        offset_counter += 1
                        # Still mark as True until we confirm offset
                        if offset_counter < min_offset_samples:
                            mask[i] = True
                        if offset_counter >= min_offset_samples:
                            # Exit event (don't mark the final offset samples)
                            in_event = False
                            offset_counter = 0
                    else:
                        offset_counter = 0
                        mask[i] = True

            masks[b] = torch.from_numpy(mask).to(device)

    return masks


def apply_morphology(
    masks: torch.Tensor,
    opening_kernel: int = 11,
    closing_kernel: int = 31,
    use_gpu: bool = False,
) -> torch.Tensor:
    """Apply morphological operations (opening then closing).

    Opening removes isolated spikes; closing fills gaps.

    Args:
        masks: Binary masks (B, T) as bool
        opening_kernel: Size of opening kernel (must be odd)
        closing_kernel: Size of closing kernel (must be odd)
        use_gpu: Whether to use GPU acceleration (not yet implemented)

    Returns:
        Cleaned binary masks (B, T) as bool
    """
    if opening_kernel % 2 == 0 or closing_kernel % 2 == 0:
        raise ValueError("Kernel sizes must be odd")

    if use_gpu and masks.is_cuda:
        # GPU path using pooling operations
        # Convert bool to float for pooling operations
        x = masks.float()

        # Opening: erosion (min pool) then dilation (max pool)
        if opening_kernel > 1:
            padding = opening_kernel // 2
            # Erosion via -max_pool1d(-x)
            x = x.unsqueeze(1)  # Add channel dim for pooling
            x = -functional.max_pool1d(-x, kernel_size=opening_kernel, stride=1, padding=padding)
            # Dilation via max_pool1d
            x = functional.max_pool1d(x, kernel_size=opening_kernel, stride=1, padding=padding)
            x = x.squeeze(1)

        # Closing: dilation (max pool) then erosion (min pool)
        if closing_kernel > 1:
            padding = closing_kernel // 2
            if x.dim() == 2:
                x = x.unsqueeze(1)
            # Dilation via max_pool1d
            x = functional.max_pool1d(x, kernel_size=closing_kernel, stride=1, padding=padding)
            # Erosion via -max_pool1d(-x)
            x = -functional.max_pool1d(-x, kernel_size=closing_kernel, stride=1, padding=padding)
            x = x.squeeze(1)

        return x > 0.5  # Back to bool

    # CPU path using scipy ndimage
    batch_size = masks.shape[0]
    device = masks.device
    cleaned_masks = torch.zeros_like(masks)

    for b in range(batch_size):
        mask_np = masks[b].cpu().numpy().astype(float)

        # Opening: erosion followed by dilation (removes spikes)
        if opening_kernel > 1:
            kernel = np.ones(opening_kernel)
            mask_np = ndimage.binary_erosion(mask_np, kernel).astype(float)
            mask_np = ndimage.binary_dilation(mask_np, kernel).astype(float)

        # Closing: dilation followed by erosion (fills gaps)
        if closing_kernel > 1:
            kernel = np.ones(closing_kernel)
            mask_np = ndimage.binary_dilation(mask_np, kernel).astype(float)
            mask_np = ndimage.binary_erosion(mask_np, kernel).astype(float)

        cleaned_masks[b] = torch.from_numpy(mask_np.astype(bool)).to(device)

    return cleaned_masks


def filter_duration(
    masks: torch.Tensor,
    min_duration_samples: int,
    max_duration_samples: int,
    sampling_rate: int = 256,
) -> torch.Tensor:
    """Filter events by duration and segment long events.

    Args:
        masks: Binary masks (B, T) as bool
        min_duration_samples: Minimum duration in samples
        max_duration_samples: Maximum duration in samples
        sampling_rate: Sampling rate for conversion (not used directly)

    Returns:
        Filtered binary masks (B, T) as bool
    """
    batch_size = masks.shape[0]
    filtered_masks = torch.zeros_like(masks)

    for b in range(batch_size):
        mask_np = masks[b].cpu().numpy()

        # Find connected components (events)
        labeled, num_features = ndimage.label(mask_np)

        for i in range(1, num_features + 1):
            indices = np.where(labeled == i)[0]
            if len(indices) == 0:
                continue

            duration = len(indices)
            start_idx = indices[0]

            # Filter by duration
            if duration < min_duration_samples:
                continue  # Too short, skip

            if duration <= max_duration_samples:
                # Keep as-is
                filtered_masks[b, indices] = True
            else:
                # Segment long event into chunks
                for chunk_start in range(start_idx, start_idx + duration, max_duration_samples):
                    chunk_end = min(chunk_start + max_duration_samples, start_idx + duration)
                    filtered_masks[b, chunk_start:chunk_end] = True

    return filtered_masks


def stitch_windows(
    window_probs: list[torch.Tensor],
    window_starts: list[int],
    total_length: int,
    method: str = "overlap_add",
) -> torch.Tensor:
    """Stitch overlapping windows into continuous probability.

    Args:
        window_probs: List of probability tensors, each shape (T_window,)
        window_starts: List of start indices for each window
        total_length: Total length of output sequence
        method: Stitching method ("overlap_add", "overlap_add_weighted", "max")

    Returns:
        Continuous probability tensor (total_length,)
    """
    if not window_probs:
        return torch.zeros(total_length)

    device = window_probs[0].device
    output = torch.zeros(total_length, device=device, dtype=torch.float32)

    if method == "max":
        # Simple max across overlapping windows
        for prob, start in zip(window_probs, window_starts, strict=False):
            end = min(start + len(prob), total_length)
            output[start:end] = torch.maximum(output[start:end], prob[: end - start])

    elif method == "overlap_add":
        # Average overlapping windows
        counts = torch.zeros(total_length, device=device, dtype=torch.float32)
        for prob, start in zip(window_probs, window_starts, strict=False):
            end = min(start + len(prob), total_length)
            output[start:end] += prob[: end - start]
            counts[start:end] += 1
        output = output / counts.clamp(min=1e-8)

    elif method == "overlap_add_weighted":
        # Weighted average using triangular window
        weights_sum = torch.zeros(total_length, device=device, dtype=torch.float32)
        for prob, start in zip(window_probs, window_starts, strict=False):
            window_len = len(prob)
            end = min(start + window_len, total_length)

            # Create triangular weight
            weight = torch.zeros(window_len, device=device)
            mid = window_len // 2
            weight[:mid] = torch.linspace(0.5, 1.0, mid, device=device)
            weight[mid:] = torch.linspace(1.0, 0.5, window_len - mid, device=device)

            output[start:end] += prob[: end - start] * weight[: end - start]
            weights_sum[start:end] += weight[: end - start]

        output = output / weights_sum.clamp(min=1e-8)

    else:
        raise ValueError(f"Unknown stitching method: {method}")

    return output.clamp(min=0, max=1)


def postprocess_predictions(
    probs: torch.Tensor,
    config: PostprocessingConfig,
    sampling_rate: int = 256,
) -> torch.Tensor:
    """Complete post-processing pipeline.

    Args:
        probs: Probability tensor (B, T) in [0, 1]
        config: Post-processing configuration
        sampling_rate: Sampling rate in Hz

    Returns:
        Binary masks (B, T) as bool tensor
    """
    # 1. Hysteresis thresholding with stability
    masks = apply_hysteresis(
        probs,
        tau_on=config.hysteresis.tau_on,
        tau_off=config.hysteresis.tau_off,
        min_onset_samples=config.hysteresis.min_onset_samples,
        min_offset_samples=config.hysteresis.min_offset_samples,
    )

    # 2. Morphological operations (opening then closing)
    masks = apply_morphology(
        masks,
        opening_kernel=config.morphology.opening_kernel,
        closing_kernel=config.morphology.closing_kernel,
        use_gpu=config.morphology.use_gpu,
    )

    # 3. Duration filtering with segmentation
    min_samples = int(config.duration.min_duration_s * sampling_rate)
    max_samples = int(config.duration.max_duration_s * sampling_rate)
    masks = filter_duration(masks, min_samples, max_samples, sampling_rate)

    return masks
