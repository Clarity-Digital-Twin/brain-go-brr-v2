# Your First Training Run

**Last Updated**: October 1, 2025
**Codebase Version**: v3.4.1
**Time Required**: 30 minutes to start, 200-300 hours total (RTX 4090)

---

## What You'll Learn

This guide walks you through a **complete training run** from start to finish, including:
- Setting up your environment
- Running full 100-epoch training
- Monitoring progress
- Handling interruptions
- Evaluating results

---

## Prerequisites

- ✅ Quickstart completed successfully (`getting-started/quickstart.md`)
- ✅ Cache fully built: `cache/tusz/train/` (4667 files) and `cache/tusz/dev/` (1832 files)
- ✅ ~50GB free disk space for cache
- ✅ ~200-300 hours of GPU time available (RTX 4090) or ~100 hours (A100)

---

## Part 1: Preparation (5 minutes)

### Step 1: Verify Cache Integrity

```bash
# Check train cache
python -m src scan-cache --cache-dir cache/tusz/train
# Expected: "Found 4667 files, XX seizures"

# Check dev cache
python -m src scan-cache --cache-dir cache/tusz/dev
# Expected: "Found 1832 files, XX seizures"
```

**Important**: Ensure manifest.json exists in both directories. If missing, it will be auto-created on first run.

### Step 2: Validate Configuration

```bash
python -m src validate configs/local/train.yaml
```

Expected output:
```
✅ Config validation passed
✅ Data paths exist
✅ Model architecture valid
✅ Training settings compatible
```

### Step 3: Set Up tmux Session (Recommended)

Training takes 200-300 hours. Use tmux to keep it running:

```bash
tmux new -s train  # Create new session named "train"
```

**tmux cheat sheet**:
- Detach: `Ctrl+B`, then `D`
- Reattach: `tmux attach -t train`
- List sessions: `tmux ls`
- Kill session: `tmux kill-session -t train`

---

## Part 2: Start Training (2 minutes)

### Local Training (RTX 4090)

```bash
# Inside tmux session
export BGB_SANITIZE_GRADS=1  # CRITICAL
export BGB_NAN_DEBUG=1       # Recommended

make train-local
```

**What this does**:
```yaml
# Runs: python -m src train configs/local/train.yaml
# Using:
training:
  batch_size: 12
  epochs: 100
  mixed_precision: false  # Disabled for RTX 4090 stability
  use_balanced_sampling: true  # CRITICAL
```

### Modal Training (A100-80GB)

```bash
# Detached for long runs
modal run --detach deploy/modal/app.py \
  --action train \
  --config configs/modal/train.yaml

# Note the App ID from output
```

**What this does**:
```yaml
# Using:
training:
  batch_size: 64  # Larger for 80GB VRAM
  epochs: 100
  mixed_precision: true  # A100 tensor cores
```

Monitor:
```bash
modal app logs <app-id>  # Stream logs
```

---

## Part 3: Monitoring (Ongoing)

### What to Watch For

#### ✅ Good Signs (Normal Training)

```
[INFO] Epoch 1/100 | Batch 50/389
[INFO] Loss: 0.623 (↓ from 0.693)
[INFO] LR: 1.00e-04
[DEBUG] Large grad norm: 5.72e+00 (clipped to 0.5)  ← NORMAL!
[INFO] [GRADIENTS] P95=9.74, P99=15.23 (decreasing trend)
```

**Key metrics**:
- **Loss**: Should decrease over time (0.69 → 0.3-0.4 by epoch 50)
- **Gradient P95**: ~5-20 is normal for BiMamba+GNN (NOT < 1.0 like transformers!)
- **LR**: Should follow warmup schedule (6.62e-07 → 1e-04 over first 1000 steps)

#### ⚠️ Warning Signs (Action Needed)

```
[WARNING] Sanitized NaN gradients at batch 42
[WARNING] [GRADIENTS] P95=45.23 (sudden spike from 9.74)
```

**What to do**:
1. Check if this is a one-time spike (continue monitoring)
2. If frequent (>5% of batches), see troubleshooting below

#### 🚨 Critical Issues (Stop Training)

```
[ERROR] NaN loss detected at batch 42
[ERROR] Non-finite logits detected
```

**Immediate actions**:
1. Stop training (`Ctrl+C` or `modal app stop <app-id>`)
2. Check cache was built after Sept 26, 2025 (preprocessing fix)
3. If cache is old: `rm -rf cache/tusz && make setup-cache`
4. See `../08-operations/nan-prevention-complete.md`

### Monitoring Commands

```bash
# Attach to tmux session
tmux attach -t train

# Check GPU usage
nvidia-smi

# Check last 50 log lines
tail -50 logs/train.log  # If logging to file

# Search for NaN issues
grep -i "nan" logs/train.log
```

---

## Part 4: Checkpoints & Resume (Optional)

### Checkpoint Strategy

By default, checkpoints save every epoch:

```
checkpoints/
├── epoch_001.pt  (~125MB each)
├── epoch_002.pt
├── ...
└── best_model.pt  (lowest validation loss)
```

### Resume from Checkpoint

If training stops (power outage, manual stop):

**Local**:
```bash
export BGB_SANITIZE_GRADS=1 BGB_NAN_DEBUG=1
python -m src train configs/local/train.yaml --resume checkpoints/epoch_042.pt
```

**Modal**:
```bash
modal run --detach deploy/modal/app.py \
  --action train \
  --config configs/modal/train.yaml \
  --resume true  # Auto-finds latest checkpoint
```

---

## Part 5: After Training (Evaluation)

### Check Final Results

```bash
# Validation metrics saved in:
cat checkpoints/validation_results.json
```

Expected metrics (v3.4.1 target):
```json
{
  "epoch": 100,
  "val_loss": 0.32,
  "sensitivity": 0.92,
  "false_alarm_rate_per_24h": 5.3
}
```

### Run Full Evaluation

```bash
python -m src evaluate \
  --checkpoint checkpoints/best_model.pt \
  --data-dir data_ext4/tusz/edf/dev \
  --cache-dir cache/tusz/dev
```

Output:
```
Sensitivity: 92.3% (target: >90%)
FA Rate (5 FA/24h): 5.1 false alarms/24h (target: <5)
TAES Score: 0.87
```

See `../06-evaluation/metrics-and-taes.md` for details on interpreting these metrics.

---

## Troubleshooting

### Training Stops with NaN Loss

**Root cause**: Old cache (before Sept 26 preprocessing fix)

**Solution**:
```bash
rm -rf cache/tusz
python -m src build-cache --data-dir data_ext4/tusz/edf/train --cache-dir cache/tusz/train
python -m src build-cache --data-dir data_ext4/tusz/edf/dev --cache-dir cache/tusz/dev
python -m src scan-cache --cache-dir cache/tusz/train
python -m src scan-cache --cache-dir cache/tusz/dev
```

### GPU OOM During Training

**Solution 1**: Reduce batch size in `configs/local/train.yaml`:
```yaml
training:
  batch_size: 8  # Down from 12
```

**Solution 2**: Enable semi-dynamic PE (slightly less memory):
```yaml
model:
  graph:
    semi_dynamic_interval: 5  # Update every 5 timesteps
```

### WSL2: Dataloader Hangs After First Epoch

**Solution**: Disable multiprocessing in `configs/local/train.yaml`:
```yaml
data:
  num_workers: 0  # WSL2 fix
```

### Loss Not Decreasing After 20 Epochs

**Check**:
1. Balanced sampling enabled? (`use_balanced_sampling: true`)
2. Learning rate too low? (should reach 1e-4 after warmup)
3. Batch size too small? (min 8 for stable gradients)

See `../08-operations/troubleshooting.md` for comprehensive troubleshooting.

---

## Expected Timeline

### Local (RTX 4090)

- **Epoch 1-10**: ~2-3 hours/epoch, loss 0.69 → 0.55
- **Epoch 11-50**: ~2 hours/epoch, loss 0.55 → 0.35
- **Epoch 51-100**: ~2 hours/epoch, loss 0.35 → 0.30 (plateau)
- **Total**: ~200-300 hours (~8-12 days continuous)

### Modal (A100-80GB)

- **Epoch 1-100**: ~1 hour/epoch
- **Total**: ~100 hours (~4 days continuous)
- **Cost**: ~$319 total (100 epochs × $3.19/hour)

---

## What's Next?

### ✅ Training Complete

- **Deploy model**: `../07-cli-tools/cli-usage.md`
- **Hyperparameter tuning**: Adjust learning rate, warmup, batch size
- **Experiment tracking**: Set up Weights & Biases integration

### 📚 Learn More

- **Gradient monitoring**: `../08-operations/gradient-monitoring.md`
- **Warmup schedules**: `../05-training/warmup-schedules.md`
- **Architecture details**: `../04-model/v3-architecture.md`
- **Performance optimization**: `../08-operations/performance-optimization.md`

---

## Quick Reference

| Stage | Command | Time |
|-------|---------|------|
| Validate | `python -m src validate configs/local/train.yaml` | 1 min |
| Train local | `make train-local` | 200-300h |
| Train Modal | `modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml` | 100h |
| Resume | `python -m src train configs/local/train.yaml --resume checkpoints/epoch_X.pt` | - |
| Evaluate | `python -m src evaluate --checkpoint checkpoints/best_model.pt` | 30 min |

**Critical environment variables**:
- `BGB_SANITIZE_GRADS=1` - **REQUIRED** for PyTorch 2.5.0
- `BGB_NAN_DEBUG=1` - Show NaN warnings (recommended)

---

**Status**: You now know how to run full 100-epoch training from start to finish! 🚀
