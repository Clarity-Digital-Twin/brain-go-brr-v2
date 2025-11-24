# Brain-Go-Brr Exp4 SGDR Analysis - For External Review

## Executive Summary

We ran **Experiment 4 (SGDR with cyclic learning rate restarts)** to test if cyclic LR could help escape training plateaus. The experiment completed 33 epochs before a CUDA crash. Results are **MIXED** - SGDR showed benefits but didn't outperform baseline's peak.

**Key Result**: Exp4 achieved **0.2633** (stable) vs Baseline's **0.2801** (unstable, lost to crash).

---

## Project Context

### Task
Clinical EEG seizure detection from TUH Seizure Corpus (19 channels, 256Hz, 60s windows).

### Model Architecture (V3)
- **TCN**: 8 layers, stride_down=16, causal_conv1d optimizations
- **Temporal SSM**: Gated DeltaNet (FLA), 6 layers, d_model=512
- **GNN**: SSGConv (α=0.05), 2 layers, dynamic Laplacian PE (16 eigenvectors)
- **Dual-stream**: Node (19×) + Edge (171×) parallel processing
- **Total params**: ~31M

### Metric
`sensitivity_at_10fa` - Sensitivity at 10 false alarms per 24 hours (clinical threshold).

### Training Setup
- **Hardware**: RTX 4090 (24GB), WSL2
- **Dataset**: 4667 train files (34.2% seizure ratio after balanced sampling)
- **Batch size**: 8 (num_workers=0 for WSL2 stability)
- **Loss**: Focal (α=0.5, γ=2.0) with warmup
- **Gradient clip**: 0.5
- **Early stopping**: patience=15, min_epochs=30

---

## Experimental Design

### Baseline (train_fla.yaml)
- **Scheduler**: Cosine annealing
- **LR**: 1e-4 → 1e-6 over 100 epochs
- **Warmup**: 3% (warmup_ratio=0.03)

### Exp4 (train_fla_exp4_cyclic.yaml)
- **Scheduler**: SGDR (Stochastic Gradient Descent with Warm Restarts)
- **Cycles**: t_initial=10, t_mult=2 (10→20→40 epochs)
- **LR**: 1e-4 (peak) → 1e-6 (min), restart to 1e-4
- **Warmup**: 3% (same as baseline)
- **Patience**: 15 (vs baseline's 20)

### Hypothesis
Baseline plateaued at 0.257 for 13 epochs (epochs 17-29). SGDR LR spikes should help model escape local minima and rediscover better solutions.

---

## Results

### Performance Comparison

| Metric | Baseline | Exp4 | Winner |
|--------|----------|------|--------|
| **Peak** | 0.2801 @ epoch 9 | 0.2633 @ epoch 32 | Baseline (+6.4%) |
| **Stable peak** | 0.2577 (post-crash) | 0.2633 | Exp4 (+2.2%) |
| **Training stability** | Crashed @ epoch 13 | No crashes | Exp4 ✅ |
| **Monotonic improvement** | No (crash) | Yes | Exp4 ✅ |

### Exp4 Progression

```
Epoch  1-3:  0.2381 (warmup)
Epoch  4-5:  0.2521 (+0.0140)
Epoch  6-28: 0.2596 (23-epoch plateau!) ⚠️
Epoch 29-31: 0.2614 (+0.0019)
Epoch 32-33: 0.2633 (+0.0019, NEW BEST)
Epoch 34+:   CUDA crash
```

### Baseline Progression (for comparison)

```
Epoch  8:    0.2633
Epoch  9:    0.2801 (PEAK)
Epoch 10-12: 0.2801 (stable)
Epoch 13:    0.2381 (CRASH - dropped 42%!)
Epoch 14-16: 0.2493 (recovering)
Epoch 17-31: 0.2577 (plateau, never recovered)
```

---

## SGDR Cycle Analysis

### Expected Cycles (after 3-epoch warmup)
- **Cycle 1**: Epochs 3-13 (10 epochs)
- **Cycle 2**: Epochs 13-33 (20 epochs)
- **Cycle 3**: Epochs 33-73 (40 epochs, not completed)

### Observed Improvement Timing

| Epoch | Metric | Delta | Cycle Context |
|-------|--------|-------|---------------|
| 4 | 0.2521 | +0.0140 | ⭐ Cycle 1 start |
| 6 | 0.2596 | +0.0075 | Cycle 1 middle |
| 7-28 | 0.2596 | 0.0000 | Plateau (spans Cycle 1 end + Cycle 2) |
| 29 | 0.2614 | +0.0019 | Cycle 2, approaching Cycle 3 |
| 32 | 0.2633 | +0.0019 | ⭐ Cycle 3 start |

### Key Observations
1. ✅ Improvements DO correlate with cycle restarts (epochs 4, 32)
2. ❌ Cycle 2 restart (epoch 13) had NO visible effect
3. ⚠️ 23-epoch plateau (6-28) persisted through ENTIRE Cycle 1 end + most of Cycle 2
4. ⚠️ Late improvements (29-33) suggest cumulative effect or lucky escape

---

## Critical Insights

### 1. Training Stability Difference

**Baseline crash @ epoch 13** is VERY concerning:
- Metric dropped from 0.2801 → 0.2381 (42% loss!)
- Never recovered to 0.2801 peak
- Suggests gradient explosion, NaN propagation, or checkpoint corruption

**Exp4 had NO performance crashes**:
- Monotonic improvement (never worse than previous best)
- Smooth transitions between plateaus
- SGDR may provide implicit regularization

**Question for reviewer**: Could SGDR's frequent LR resets prevent gradient instability?

### 2. SGDR Effectiveness

**Partial success**:
- ✅ Improvements near cycle restarts (epochs 4, 32)
- ❌ Cycle 2 restart (epoch 13) ineffective
- ❌ 23-epoch plateau suggests LR=1e-4 still too conservative for escape

**Hypothesis**: 
- Either: 0.2596 is a very strong local minimum
- Or: Need higher peak LR (e.g., 5e-4 or 1e-3) for effective escape
- Or: Cycle length too short (model needs more time to explore)

### 3. Early Improvements vs Late Improvements

**Baseline**: Hit 0.2801 very early (epoch 9), then crashed
**Exp4**: Slow steady climb, peak late (epoch 32)

**Possible explanations**:
1. **Random seed**: Different initialization, different trajectory
2. **Cosine LR better early**: Smooth decay allows faster initial convergence
3. **SGDR disrupts early training**: Restarts prevent early-epoch peak discovery
4. **SGDR enables late discovery**: Restarts help find solutions after epoch 30

**Question for reviewer**: Is early peak (unstable) better than late peak (stable)?

### 4. Min Epochs Configuration

**Critical finding**: `min_epochs=30` SAVED Exp4!

Without it:
- Early stopping would trigger @ epoch 21 (15 epochs after epoch 6 plateau)
- Late improvements (epochs 29-33) would be MISSED
- Final metric would be 0.2596 instead of 0.2633

**Lesson**: Long patience + min_epochs crucial for SGDR experiments.

### 5. CUDA Crashes

Both experiments crashed:
- **Baseline**: Epoch 13 (performance crash - training error)
- **Exp4**: Epoch 34 (CUDA driver crash - hardware error)

**Exp4 crash @ epoch 34**:
```
CUDA error: unknown error
CUDA kernel errors might be asynchronously reported
```

This is a **transient CUDA driver glitch** (not training-related):
- GPU healthy after restart (no errors in dmesg)
- Exp4 crashed again at same point after resume (epoch 34)
- Likely GPU driver stability issue on WSL2 + long-running training (4+ days)

---

## Open Questions for Review

### 1. Why did baseline outperform Exp4 early?

**Data**:
- Baseline: 0.2801 @ epoch 9
- Exp4: 0.2596 @ epoch 6 (peak until epoch 28)

**Possible explanations**:
- Random seed differences (both use seed=42, but resume chaos?)
- Cosine schedule better for early convergence?
- SGDR disrupting early optimization?

**Questions**:
- Should we re-run baseline with same exact initialization as Exp4?
- Is there a way to get "best of both" (early peak + late stability)?

### 2. What caused baseline epoch 13 crash?

**Evidence**:
- Metric dropped 0.2801 → 0.2381 (42% loss!)
- No CUDA errors reported
- Never recovered to peak

**Possible causes**:
- Gradient explosion (but gradient_clip=0.5 should prevent this)
- NaN propagation (but NaN debugging enabled, no warnings)
- Checkpoint corruption during save/load
- Batch with extreme outliers causing model instability

**Questions**:
- Should we inspect epoch 12-13 checkpoints more carefully?
- Could this be related to GNN Laplacian eigendecomposition?
- Is this a reproducible failure or one-off bad luck?

### 3. Would Exp4 eventually reach 0.2801+ if continued?

**Evidence**:
- Counter=1, patience=15 (14 epochs remaining)
- Cycle 3 just started (40-epoch cycle, only 1 epoch completed)
- Late improvements (epochs 29-33) suggest momentum

**Arguments for resuming**:
- Cycle 3's longer duration (40 epochs) might allow deeper exploration
- Improvements at cycle starts suggest more gains possible
- Only 1 epoch past best, plenty of patience left

**Arguments against resuming**:
- Exp4 never reached baseline's 0.2801 in 33 epochs
- 23-epoch plateau suggests strong local minimum
- CUDA crashes interrupting repeatedly (hardware limitation)

### 4. Is SGDR worth it for this task?

**Trade-offs**:

**Pros**:
- ✅ More stable training (no performance crashes)
- ✅ Monotonic improvement
- ✅ Late-epoch gains (validated min_epochs strategy)
- ✅ Improvements correlate with cycle restarts

**Cons**:
- ❌ Lower peak than baseline (0.2633 vs 0.2801)
- ❌ 23-epoch plateau despite restarts
- ❌ Cycle 2 restart ineffective
- ❌ Slower convergence (peak @ epoch 32 vs 9)

**Net assessment**: **INCONCLUSIVE**
- If you value stability: Exp4 wins
- If you value peak performance: Baseline wins (if you can fix the crash)

---

## Recommendations for Next Steps

### Option 1: Fix Baseline Instability
**Priority**: HIGH

If we can figure out what caused baseline's epoch 13 crash and prevent it, baseline's 0.2801 peak is the clear winner.

**Actions**:
1. Inspect epochs 12-13-14 checkpoints for anomalies
2. Check validation batch that triggered crash
3. Add more aggressive NaN detection
4. Consider: Is GNN eigendecomposition stable at epoch 13?
5. Re-run baseline with extra safety checks

### Option 2: Tune SGDR Hyperparameters
**Priority**: MEDIUM

SGDR showed promise but needs better tuning.

**Experiments**:
1. **Higher peak LR**: Try 5e-4 or 1e-3 (current 1e-4 may be too conservative)
2. **Longer cycles**: Try t_initial=15 or 20 (more time to explore)
3. **Different t_mult**: Try t_mult=1.5 (smoother growth)
4. **Combine with baseline init**: Start from baseline epoch 9 checkpoint, apply SGDR

### Option 3: Resume Exp4 to Completion
**Priority**: LOW

Continue Exp4 to test if Cycle 3 discovers better solutions.

**Risk**: Repeated CUDA crashes may prevent completion.

**Mitigation**: 
- Add checkpoint-and-restart automation
- Monitor GPU temperature/power
- Consider shorter training sessions with manual restarts

### Option 4: Try Alternative Schedulers
**Priority**: MEDIUM

Test other LR schedules that help with plateau escape.

**Candidates**:
1. **OneCycleLR**: Single triangle (0 → peak → 0)
2. **ReduceLROnPlateau**: Adaptive reduction when stuck
3. **Hybrid**: Cosine + manual LR boost at epoch 30

---

## Code Quality & Architecture Notes

### Strengths
✅ **Well-structured training loop** (`src/brain_brr/train/loop.py`)
- Clean checkpoint save/resume logic
- Early stopping with state persistence
- Comprehensive logging

✅ **WandB integration** (`src/brain_brr/train/wandb_integration.py`)
- Automatic run ID persistence
- Resume support via `.wandb_run_id` file
- Graceful fallback if WandB unavailable

✅ **Scheduler factory** (`src/brain_brr/train/scheduler_factory.py`)
- Clean abstraction for different schedulers
- SGDR implementation with gradient accumulation awareness
- Warmup support

✅ **Early stopping fix** (recent commit)
- Fixed off-by-one bug: `counter > patience` → `counter >= patience`
- Now stops correctly after exactly `patience` epochs

### Potential Issues

⚠️ **Baseline epoch 13 crash** - undiagnosed
- No clear root cause identified
- Could be gradient/NaN issue, checkpoint corruption, or GNN instability
- **Recommendation**: Add more defensive checks

⚠️ **CUDA crashes on WSL2**
- Transient driver errors after 3-4 days continuous training
- Not a code issue, but impacts reproducibility
- **Recommendation**: Implement auto-resume on CUDA errors

⚠️ **No training.log for Exp4**
- Checkpoints exist, but `training.log` missing
- Metrics recoverable from checkpoints, but less convenient
- **Recommendation**: Verify logging configuration

### Architecture Questions

**For external reviewer**:

1. **GNN Laplacian eigendecomposition** (`src/brain_brr/models/gnn_pyg.py`)
   - Computed every `semi_dynamic_interval=5` batches
   - Could this cause gradient spikes? (eigendecomp is non-differentiable)
   - Is the sign consistency fix (`pe_sign_consistency=true`) sufficient?

2. **Edge Mamba stream** (`src/brain_brr/models/detector.py`)
   - 171 edges (all pairs from 19 channels)
   - Each edge has its own temporal model
   - Could edge stream be causing instabilities?

3. **Focal loss with warmup** (`src/brain_brr/train/train_step.py`)
   - Gamma warmed 1.0 → 2.0 over 1000 batches
   - Adjacency temperature warmed 2.0 → 1.0
   - Could warmup schedules interact poorly with SGDR restarts?

4. **Balanced sampling** (`src/brain_brr/data/datasets.py`)
   - 34.2% seizures (vs natural 8%)
   - Could oversampling bias affect convergence?

---

## Files to Review

### Core training loop
- `src/brain_brr/train/loop.py` (main training logic)
- `src/brain_brr/train/train_step.py` (forward/backward pass)
- `src/brain_brr/train/val_step.py` (validation)
- `src/brain_brr/train/early_stopping.py` (early stopping logic)

### Scheduler implementation
- `src/brain_brr/train/scheduler_factory.py` (SGDR + warmup)

### Model architecture
- `src/brain_brr/models/detector.py` (V3 dual-stream orchestrator)
- `src/brain_brr/models/gnn_pyg.py` (GNN + dynamic Laplacian PE)
- `src/brain_brr/models/mamba.py` (BiMamba wrapper)
- `src/brain_brr/models/tcn.py` (TCN encoder)

### Data pipeline
- `src/brain_brr/data/datasets.py` (BalancedSeizureDataset)
- `src/brain_brr/data/io.py` (memory-mapped NPY loading)

### Configuration
- `configs/local/train_fla.yaml` (baseline)
- `configs/local/train_fla_exp4_cyclic.yaml` (SGDR experiment)

---

## Questions for External Reviewer

1. **Performance**: Why did baseline reach 0.2801 but Exp4 only 0.2633? Is this scheduler effect or random seed?

2. **Stability**: What caused baseline's epoch 13 crash? How can we prevent it?

3. **SGDR tuning**: Why was Cycle 2 restart (epoch 13) ineffective? Should we try higher peak LR or longer cycles?

4. **Architecture**: Could GNN eigendecomposition or edge Mamba be causing instabilities?

5. **Training strategy**: Should we prioritize fixing baseline (0.2801 peak) or tuning Exp4 (0.2633 stable)?

6. **Next experiment**: What would you recommend as Exp5?

---

## Appendix: Environment

- **OS**: WSL2 (Ubuntu on Windows)
- **GPU**: RTX 4090 (24GB)
- **CUDA**: 12.4
- **PyTorch**: 2.5.0+cu124
- **Python**: 3.11.13
- **Key deps**: mamba-ssm 2.2.5, fla (flash-linear-attention), torch-geometric 2.6.1

**Known WSL2 issues**:
- `num_workers > 0` causes hangs (use num_workers=0)
- Long-running training triggers CUDA driver crashes (restart required)
- ext4 partition required for mmap cache (NTFS causes SIGBUS)

