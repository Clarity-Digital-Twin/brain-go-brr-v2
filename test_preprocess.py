#!/usr/bin/env python3
"""Test if preprocessing actually normalizes data."""

import numpy as np
from src.brain_brr.data.preprocess import preprocess_recording

# Create raw data similar to what we see in cache (range -80 to 100)
n_channels = 19
n_samples = 15360
raw_data = np.random.randn(n_channels, n_samples) * 30 + 10  # Mean 10, std 30

print(f"Raw data range: [{raw_data.min():.1f}, {raw_data.max():.1f}]")
print(f"Raw data mean: {raw_data.mean():.1f}, std: {raw_data.std():.1f}")

# Apply preprocessing
processed = preprocess_recording(raw_data, fs_original=256)

print(f"\nProcessed data range: [{processed.min():.1f}, {processed.max():.1f}]")
print(f"Processed data mean: {processed.mean():.3f}, std: {processed.std():.3f}")
print(f"Processed dtype: {processed.dtype}")

# Check per-channel stats
for i in range(min(3, n_channels)):
    channel_mean = processed[i].mean()
    channel_std = processed[i].std()
    print(f"  Channel {i}: mean={channel_mean:.3f}, std={channel_std:.3f}")