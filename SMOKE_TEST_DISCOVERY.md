# CRITICAL SMOKE TEST DISCOVERY

## What We Found

### The Problem
Our smoke tests are **fundamentally wrong** - we're trying to use REAL EDF data when we should use SYNTHETIC data!

### Evidence from Codebase

1. **Unit Tests Use Synthetic Data** (`tests/unit/train/test_loop.py:36-52`):
```python
# Create synthetic balanced dataset (10 windows)
windows = torch.randn(10, 19, 15360)
labels = torch.zeros(10, 15360)
labels[::2, 5000:10000] = 1  # Make 50% positive

# Direct TensorDataset - NO EDF loading!
train_dataset = TensorDataset(windows[:8], labels[:8])
```

2. **Current Smoke Test Problems**:
   - Loading real EDF files (slow I/O)
   - Sampler checking 80 windows by loading each one (`dataset[idx]`)
   - No seizures in first 2 files alphabetically
   - Hanging on "Checking 80 windows for seizures..."

3. **Our "Fixes" Were Band-Aids**:
   - Added `BGB_SMOKE_TEST=1` to bypass seizure checks
   - Skipped dataset sampling
   - Skipped sampler window checking
   - **BUT WE'RE STILL LOADING REAL DATA!**

## The Right Solution

### Option 1: Synthetic Data Mode (BEST)
Create a `--synthetic` flag that uses `TensorDataset` with random tensors:
- Instant loading (no I/O)
- Guaranteed balanced data
- Tests all pipeline mechanics
- Works identically for U-Net and TCN

### Option 2: Pre-cached Test Data
Have a small set of pre-processed `.npz` files ready:
- Skip EDF loading entirely
- Use known good data with seizures
- Still requires some I/O

### Option 3: Fix Current Approach (NOT RECOMMENDED)
Continue with real EDF but:
- Find files with seizures first
- Cache everything upfront
- Still slow and fragile

## Why This Matters

1. **Smoke tests should be FAST** (<10 seconds)
2. **Architecture agnostic** - TCN vs U-Net shouldn't matter
3. **No data dependencies** - shouldn't fail based on file ordering
4. **Test mechanics, not data** - we're validating shapes and gradients

## Current Status

- Real training runs in tmux/Modal are FINE (they use full data)
- Smoke test configs exist but are using wrong approach
- TCN integration is CORRECT - the smoke test approach is wrong
- Mamba fallback warning is EXPECTED without GPU package

## Immediate Action Items

1. **DON'T use configs/local/smoke*.yaml** for now
2. **Use unit tests** for validation: `pytest tests/unit/train/test_loop.py`
3. **For real training**: Use full configs with proper data

## The Real Test Should Be

```python
# Create synthetic data
data = torch.randn(batch_size, 19, 15360)
labels = torch.zeros(batch_size, 15360)
labels[:, 5000:10000] = 1  # Some positive labels

# Test forward pass
model = SeizureDetector.from_config(config)
output = model(data)
loss = F.binary_cross_entropy_with_logits(output, labels)

# Test backward pass
loss.backward()
optimizer.step()

# Verify shapes and no NaNs
assert output.shape == labels.shape
assert not torch.isnan(loss)
```

This tests EVERYTHING we need without touching a single EDF file!