# Exp4 Monitoring Guide

## Current Status
- **Session**: exp4 (tmux)
- **Experiment**: FLA + SGDR (Cyclic LR)
- **Resuming from**: Epoch 33 → Target: Epoch 100
- **Current**: Epoch 34, Batch ~2956/7702
- **Best so far**: 0.2633 @ 10 FA/24h
- **Fix active**: ✅ Eigendecomposition jitter + CPU fallback

## Quick Checks

### 1. Training Progress
```bash
tmux attach -t exp4
# Press Ctrl+B, then D to detach
```

### 2. GPU Status
```bash
watch -n 2 nvidia-smi
# Should see: 98% GPU util, ~11-18GB VRAM, ~250-300W
```

### 3. Check for Eigendecomp Issues
```bash
tmux capture-pane -t exp4 -p -S -1000 | grep -i "eigendecomp\|CPU fallback"
# Should be EMPTY (no warnings = fix working)
```

### 4. Check Latest Metrics
```bash
tail -f results/local_fla_exp4_cyclic/training.log | grep -E "PROGRESS|Best"
```

## What to Watch For

### ✅ GOOD Signs:
- GPU util: 95-100%
- Loss: Decreasing slowly
- No "eigendecomp failed" warnings
- No "NaN/Inf detected" warnings
- Batch progress: ~2-3s/it

### 🚨 BAD Signs (STOP TRAINING):
- "GPU eigendecomp failed" → Appears repeatedly (>10x)
- Loss = NaN or Inf
- GPU util drops to 0%
- OOM errors

## Performance Expectations

- **Time per epoch**: ~9.6 hours
- **Epochs remaining**: 66 (epoch 34 → 100)
- **Total time**: ~26 days
- **Checkpoints**: Every 30 min + every epoch

## Resume After Crash

```bash
tmux new -s exp4
export BGB_NAN_DEBUG=1
make resume
# Ctrl+B D to detach
```

## Stop Training

```bash
tmux kill-session -t exp4
```

---
Generated: 2025-11-23 21:21 EST
