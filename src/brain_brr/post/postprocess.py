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

from src.brain_brr.config.schemas import (
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

    # Vectorized implementation using run-length processing
    masks = torch.zeros_like(probs, dtype=torch.bool, device=device)

    for b in range(batch_size):
        x = probs[b].detach().cpu().numpy()

        high = x >= tau_on
        low_ok = x >= tau_off
        below_off = ~low_ok

        if seq_len == 0:
            continue

        # Run-length encode boolean array: returns starts and lengths of True runs
        def rle_runs(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            if a.size == 0:
                return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
            pad = np.empty(a.size + 2, dtype=np.bool_)
            pad[0] = False
            pad[-1] = False
            pad[1:-1] = a
            diff = np.diff(pad.view(np.int8))
            starts = np.flatnonzero(diff == 1)
            ends = np.flatnonzero(diff == -1)
            lengths = ends - starts
            return starts.astype(np.int64), lengths.astype(np.int64)

        # Firm OFF runs: consecutive samples below tau_off with length >= min_offset_samples
        off_starts, off_lens = rle_runs(below_off)
        if off_starts.size:
            mask_firm = off_lens >= min_offset_samples
            off_starts = off_starts[mask_firm]
            off_lens = off_lens[mask_firm]
        else:
            off_lens = off_lens  # keep dtype

        # Candidate ON runs: consecutive samples >= tau_on with length >= min_onset_samples
        on_starts, on_lens = rle_runs(high)
        if on_starts.size:
            mask_on = on_lens >= min_onset_samples
            on_starts = on_starts[mask_on]
        # Zone boundaries are between firm OFF runs
        zones: list[tuple[int, int]] = []
        prev_end = 0
        for s, length in zip(off_starts, off_lens, strict=False):
            # Include the initial (min_offset_samples-1) below_off samples inside the zone
            ze = int(s + max(0, min_offset_samples - 1))
            if ze > prev_end:
                zones.append((prev_end, min(ze, seq_len)))
            prev_end = s + int(length)
        if prev_end < seq_len:
            zones.append((prev_end, seq_len))

        mask_np = np.zeros(seq_len, dtype=np.bool_)

        if on_starts.size == 0 or len(zones) == 0:
            masks[b] = torch.from_numpy(mask_np).to(device)
            continue

        # For each zone, find the first qualifying onset and fill until zone end
        # on_starts is sorted; use pointer to scan once
        on_ptr = 0
        n_on = int(on_starts.size)
        for zs, ze in zones:
            while on_ptr < n_on and on_starts[on_ptr] < zs:
                on_ptr += 1
            if on_ptr >= n_on:
                break
            onset = int(on_starts[on_ptr])
            if onset < ze:
                mask_np[onset:ze] = True
                on_ptr += 1  # advance to avoid reusing same onset in later zones

        masks[b] = torch.from_numpy(mask_np).to(device)

    return masks


def apply_morphology(
    masks: torch.Tensor,
    opening_kernel: int = 11,
    closing_kernel: int = 31,
    use_gpu: bool = False,
    kernel_size: int | None = None,
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
    if kernel_size is not None:
        opening_kernel = kernel_size
        closing_kernel = kernel_size

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

    # CPU path using scipy ndimage - optimized batch processing
    batch_size = masks.shape[0]
    device = masks.device

    # Process all masks at once for better memory locality
    masks_np = masks.cpu().numpy().astype(float)

    # Precompute kernels once
    opening_struct = np.ones(opening_kernel) if opening_kernel > 1 else None
    closing_struct = np.ones(closing_kernel) if closing_kernel > 1 else None

    # Process batch
    for b in range(batch_size):
        # Opening: erosion followed by dilation (removes spikes)
        if opening_kernel > 1 and opening_struct is not None:
            masks_np[b] = ndimage.binary_erosion(masks_np[b], opening_struct).astype(float)
            masks_np[b] = ndimage.binary_dilation(masks_np[b], opening_struct).astype(float)

        # Closing: dilation followed by erosion (fills gaps)
        if closing_kernel > 1 and closing_struct is not None:
            masks_np[b] = ndimage.binary_dilation(masks_np[b], closing_struct).astype(float)
            masks_np[b] = ndimage.binary_erosion(masks_np[b], closing_struct).astype(float)

    # Convert back to torch tensor efficiently
    cleaned_masks = torch.from_numpy(masks_np.astype(bool)).to(device)
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
        arr = masks[b].detach().cpu().numpy().astype(np.bool_)

        if arr.size == 0:
            continue

        pad = np.empty(arr.size + 2, dtype=np.bool_)
        pad[0] = False
        pad[-1] = False
        pad[1:-1] = arr
        diff = np.diff(pad.view(np.int8))
        starts = np.flatnonzero(diff == 1)
        ends = np.flatnonzero(diff == -1)

        for s, e in zip(starts, ends, strict=False):
            length = int(e - s)
            if length < min_duration_samples:
                continue
            if length <= max_duration_samples:
                filtered_masks[b, s:e] = True
            else:
                for chunk_start in range(s, s + length, max_duration_samples):
                    chunk_end = min(chunk_start + max_duration_samples, s + length)
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
