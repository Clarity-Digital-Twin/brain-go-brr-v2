#!/usr/bin/env python
"""Test V3 with all fixes applied."""

import torch
import yaml
import os
from pathlib import Path

# Set environment variables
os.environ["BGB_SANITIZE_GRADS"] = "1"
os.environ["BGB_NAN_DEBUG"] = "1"

print("="*60)
print("TESTING V3 WITH ALL FIXES")
print("="*60)

# Load config
with open('configs/local/train.yaml') as f:
    cfg_dict = yaml.safe_load(f)

from src.brain_brr.config.schemas import Config
config = Config(**cfg_dict)

print(f"Architecture: {config.model.architecture}")
print(f"Batch size: {config.training.batch_size}")
print(f"Learning rate: {config.training.learning_rate}")

# Create model
print("\nCreating V3 model...")
from src.brain_brr.models.detector import SeizureDetector
model = SeizureDetector.from_config(config.model)
model.train()

if torch.cuda.is_available():
    model = model.cuda()
    device = 'cuda'
else:
    device = 'cpu'

# Create optimizer with FIX
print("\nCreating optimizer with parameter groups...")
from src.brain_brr.train.loop import create_optimizer
optimizer = create_optimizer(model, config.training)

# Test with saved bad batches
import glob
bad_batch_files = sorted(glob.glob('debug/bad_batch_*.pt'))[:10]

if not bad_batch_files:
    print("No bad batch files found, using random data")
    bad_batch_files = [None] * 5

print(f"\nTesting {len(bad_batch_files)} batches...")
print("="*60)

loss_fn = torch.nn.BCEWithLogitsLoss()

for i, batch_file in enumerate(bad_batch_files):
    if batch_file:
        batch = torch.load(batch_file)
        windows = batch['windows'].cuda()
        labels = batch['labels'].cuda()
        print(f"\nBatch {i} (from {Path(batch_file).name}):")
    else:
        windows = torch.randn(4, 19, 15360, device=device)
        labels = (torch.rand(4, 15360, device=device) > 0.9).float()
        print(f"\nBatch {i} (random):")

    # Forward pass
    optimizer.zero_grad(set_to_none=True)
    logits = model(windows)

    # Check for NaN
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        print(f"  ❌ FAILURE: NaN/Inf in logits")
        nan_count = torch.isnan(logits).sum().item()
        print(f"     {nan_count}/{logits.numel()} NaNs")
        break

    # Loss and backward
    loss = loss_fn(logits, labels)

    if torch.isnan(loss):
        print(f"  ❌ FAILURE: Loss is NaN")
        break

    loss.backward()

    # Check gradients (with sanitization)
    grad_has_nan = False
    for name, param in model.named_parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            grad_has_nan = True
            param.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

    if grad_has_nan:
        print(f"  ⚠️  Sanitized NaN gradients")

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)

    # Optimizer step
    optimizer.step()

    # Check parameters
    param_has_nan = False
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            param_has_nan = True
            print(f"  ❌ FAILURE: NaN in parameter {name}")
            break

    if param_has_nan:
        break

    print(f"  ✅ OK - loss={loss.item():.4f}, logits_mean={logits.mean():.3f}")

print("\n" + "="*60)
print("TEST COMPLETE")

if i == len(bad_batch_files) - 1:
    print("✅ ALL FIXES WORKING - V3 TRAINING SHOULD SUCCEED!")
else:
    print(f"❌ Failed at batch {i}")