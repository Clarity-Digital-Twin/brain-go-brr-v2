#!/usr/bin/env python3
"""Debug script to test with actual training data."""

import torch
from pathlib import Path
from src.brain_brr.models import SeizureDetector
from src.brain_brr.config.schemas import ModelConfig

# Load a bad batch saved during training
debug_files = list(Path("debug").glob("bad_batch_*.pt"))
if not debug_files:
    print("No bad batches found. Let me create synthetic data similar to training.")
    # Use training-like data
    batch_size = 4
    n_channels = 19
    window_samples = 15360

    # Training uses normalized data, typically in range [-5, 5]
    windows = torch.randn(batch_size, n_channels, window_samples) * 2.0
    labels = torch.ones(batch_size, window_samples)
else:
    print(f"Loading bad batch: {debug_files[0]}")
    data = torch.load(debug_files[0])
    windows = data['windows']
    labels = data['labels']

print(f"Windows shape: {windows.shape}, range: [{windows.min():.3f}, {windows.max():.3f}]")
print(f"Labels shape: {labels.shape}")

# Set up model
model = SeizureDetector.from_config(ModelConfig())

# Test in TRAINING mode (same as actual training)
model.train()
print("\n=== TRAINING MODE ===")
with torch.no_grad():
    output_train = model(windows)
print(f"Output range: [{output_train.min():.3f}, {output_train.max():.3f}]")
print(f"Has inf: {torch.isinf(output_train).any()}, count: {torch.isinf(output_train).sum()}")
print(f"Has nan: {torch.isnan(output_train).any()}, count: {torch.isnan(output_train).sum()}")

# Test in EVAL mode
model.eval()
print("\n=== EVAL MODE ===")
with torch.no_grad():
    output_eval = model(windows)
print(f"Output range: [{output_eval.min():.3f}, {output_eval.max():.3f}]")
print(f"Has inf: {torch.isinf(output_eval).any()}, count: {torch.isinf(output_eval).sum()}")
print(f"Has nan: {torch.isnan(output_eval).any()}, count: {torch.isnan(output_eval).sum()}")

# Check specific layers in training mode
print("\n=== LAYER ANALYSIS (TRAINING MODE) ===")
model.train()
x = windows

# TCN check
features = model.tcn_encoder(x)
print(f"TCN output: shape={features.shape}, range=[{features.min():.3f}, {features.max():.3f}]")
print(f"  TCN zeros: {(features == 0).sum()} / {features.numel()}")
print(f"  TCN near-zero: {(features.abs() < 1e-6).sum()} / {features.numel()}")

# Check if BatchNorm is the issue
if hasattr(model.tcn_encoder, 'blocks'):
    for i, block in enumerate(model.tcn_encoder.blocks[:2]):  # Check first 2 blocks
        if hasattr(block, 'bn') or hasattr(block, 'norm'):
            print(f"  Block {i} has normalization layer")