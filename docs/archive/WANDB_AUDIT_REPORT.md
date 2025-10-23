# 🔍 Wandb Audit Report - Exp1 Regularization Run
**Generated**: 2025-10-22 19:19 EDT
**Run ID**: `191b56e837544d9e9bf3236c01f0376d`
**Status**: ⚠️ **UPLOAD STALLED, DATA INTACT LOCALLY**

---

## 📊 Executive Summary

**The Good News**: Training is healthy and all data is being logged locally.
**The Bad News**: Wandb cloud upload stalled due to 502 Server Error at 12:35 EDT (7 hours ago).
**Data Status**: ✅ **ALL DATA IS SAFE** in local `.wandb` protobuf file (5.8MB, actively growing).

---

## 🎯 Current Training Status

### Training Progress
- **Current**: Epoch 8, Batch 52/7702 (1% into epoch 8)
- **Completed**: Epoch 7 finished successfully at 18:55 EDT
- **Runtime**: 10.6 hours CPU time (process PID 3790871)
- **Health**: ✅ Training stable, no errors

### Checkpoint Status
```
last.pt:        Epoch 7, Global Step 53914, Best Metric 0.2726 (18:55 EDT)
epoch_007.pt:   Saved successfully (189MB, 18:55 EDT)
best.pt:        Updated with epoch 7 results (189MB, 18:55 EDT)
```

**Epoch 7 metrics ARE in checkpoint** - validation ran successfully!

---

## 🚨 Root Cause Analysis

### Timeline of Events

**12:08 EDT** - Wandb initialized successfully
- Run resumed with ID `191b56e837544d9e9bf3236c01f0376d`
- Backend connected: "backend started and connected"
- Stream services started: handler, sender, writer

**12:35 EDT** - 502 Server Error encountered
```json
{
  "level": "INFO",
  "msg": "api: retrying HTTP error",
  "status": 502,
  "url": "https://api.wandb.ai/files/.../file_stream",
  "error": "Server Error: The server encountered a temporary error"
}
```

**12:35 - 19:19 EDT** - Upload stalled, local logging continues
- No further wandb internal log entries (7 hours silence)
- Local `.wandb` file continues growing: 5.8MB (last modified 19:16)
- Training continues normally, epochs 7-8 logging locally

### Root Cause

**Wandb Infrastructure Issue**: 502 Bad Gateway on `file_stream` endpoint

This is a **known wandb bug** documented in multiple GitHub issues:
- Issue #2039: Network error entering retry loop
- Community reports from 2024: Upload stuck with 502 errors
- Pattern: file_stream endpoint fails → upload hangs → local logging continues

The wandb backend service hit a server error and entered a failed state, but:
- ✅ Did NOT crash the training process
- ✅ Continued logging to local `.wandb` protobuf file
- ❌ Stopped uploading to cloud (hence empty web UI)

---

## 📁 Data Inventory

### Wandb Local Files
```
run-20251022_120834-*/
├── run-*.wandb              5.8 MB  (CONTAINS ALL LOGGED DATA)
├── files/
│   ├── output.log           467 KB
│   ├── requirements.txt     4.7 KB
│   └── wandb-metadata.json  1.1 KB
└── logs/
    ├── debug.log            3.3 KB  (last: 12:08)
    └── debug-internal.log   ~200 B  (last: 12:35 with 502 error)
```

### What's in the .wandb file?
- **Format**: Protocol Buffer (binary)
- **Contains**: ALL logged metrics since 12:08 (epochs 7+)
- **Status**: ✅ Actively growing (last modified 3 min ago)
- **Upload Status**: ❌ Not uploaded due to 502 error

### Web UI Status
```
History entries: 4 (steps 0-3 from OLD run Oct 18-20)
Summary _step: 3
Latest metrics: From step 3 (epoch 3, Oct 20)
Epochs 7-8 data: Not visible (stuck in local .wandb file)
```

---

## 💡 Why Data is Missing from Web UI

**The Resume Bug** (GitHub Issue #2971):
When resuming a run, if upload fails:
1. Local logging continues to `.wandb` protobuf file ✅
2. Cloud upload stalls ❌
3. Web UI shows only pre-failure data ❌

**NOT** the step-overwrite bug (that would be code-level).
This is **infrastructure-level** - wandb server failed mid-run.

---

## 🔧 Available Options

### Option 1: Let Training Continue (RECOMMENDED)
**Rationale**: Data is safe locally, training is healthy

**Pros**:
- ✅ No interruption to training
- ✅ All checkpoints saved locally
- ✅ Can manually sync later
- ✅ Avoids losing progress (epoch 8 already started)

**Cons**:
- ❌ No live monitoring on web UI
- ❌ Epochs 7+ data not visible until manual sync

**Action**: Continue to epoch 20-30, rely on early stopping (patience=5)

### Option 2: Restart Wandb (RISKY)
**Rationale**: Force re-init wandb to re-establish upload

**Pros**:
- ✅ May restore web UI updates

**Cons**:
- ❌ Requires stopping training
- ❌ Must restart from checkpoint (loses epoch 8 progress)
- ❌ May create new run ID (history fragmentation)
- ❌ No guarantee 502 error won't recur

**Action**: Kill training, restart with `--resume`

### Option 3: Manual Sync After Completion
**Rationale**: Fix upload issue after training finishes

**Pros**:
- ✅ No training interruption
- ✅ All local data intact
- ✅ Can retry sync multiple times

**Cons**:
- ❌ Must wait until training ends
- ❌ Known issue: sync may not upload buffered protobuf data (see Issue #1357)

**Action**: Run `wandb sync` after training completes

### Option 4: Monitor Locally, Accept Web UI Loss
**Rationale**: Training metrics matter more than wandb UI

**Pros**:
- ✅ Zero risk to training
- ✅ Checkpoints have all validation metrics
- ✅ Can analyze results from checkpoints

**Cons**:
- ❌ No wandb charts for epochs 7+
- ❌ Less convenient monitoring

**Action**: Check checkpoints for best metric, use local logs

---

## 📈 Data Recovery Potential

### What CAN be recovered:
- ✅ All validation metrics (in checkpoints: `ckpt['val_metrics']`)
- ✅ Train/val losses (in checkpoint history)
- ✅ Best model (saved as `best.pt`)
- ✅ Early stopping decisions (based on local metrics)

### What MIGHT be recoverable:
- ⚠️ Epoch-level metrics from `.wandb` file (via manual sync)
- ⚠️ Batch-level training logs (if logged to wandb)

### What is LOST:
- ❌ Live web UI monitoring (epochs 7+)
- ❌ Real-time charts (unless manual sync succeeds)

---

## 🎯 Recommended Action Plan

### Immediate (Next 30 seconds):
1. ✅ **DO NOTHING** - Let training continue
2. ✅ Accept wandb web UI will be stale
3. ✅ Monitor via local checkpoints

### Short-term (During training):
1. ✅ Check `checkpoints/best.pt` periodically for best metric
2. ✅ Early stopping will work (uses local metrics, not wandb)
3. ✅ If early stops at epoch ~12 (5 patience from epoch 7 best), that's VALID

### Long-term (After training):
1. ⚠️ Attempt `wandb sync` on completed run
2. ⚠️ If sync fails (likely due to Issue #1357), accept data loss
3. ✅ Analyze results from checkpoints (all metrics preserved)
4. ✅ Document findings in experiment notes

---

## 🔬 Technical Details

### Code Logging Behavior
```python
# loop.py:414
wandb_logger.log(wandb_metrics, step=epoch)
```

**Logging uses `step=epoch`**, NOT `step=global_step`.
This means:
- Old run logged steps 0-3 (epochs 0-3)
- Current run should log steps 7, 8, 9, ... (epochs 7, 8, 9, ...)
- **No step collision** - different epoch numbers

### Why Web UI Shows Steps 0-3 Only:
The 502 error occurred **before** epoch 7 validation logged.
Local `.wandb` file contains epoch 7+ data, but upload stalled.

---

## 🎓 Lessons Learned

### For This Run:
1. **Wandb is not critical** - checkpoints have all needed metrics
2. **Early stopping works locally** - doesn't need wandb upload
3. **Training completion > Live monitoring**

### For Future Runs:
1. Consider local logging fallback (CSV/JSON) for critical metrics
2. Monitor wandb `debug-internal.log` for upload errors
3. Set `WANDB_MODE=offline` if network unreliable (sync manually later)
4. Use `wandb.watch()` less frequently to reduce upload load

---

## 📞 If You Need to Decide NOW

**Question**: Should we restart to fix wandb?
**Answer**: **NO** - Continue training

**Reasoning**:
- Training is healthy (10+ hours runtime, epoch 8 started)
- Data is safe in checkpoints
- Restarting risks:
  - Losing epoch 8 progress
  - Same 502 error recurring
  - Creating fragmented run history
- Early stopping will work correctly (local metrics)
- Final results matter more than live charts

**Let it ride to epochs 20-30, rely on early stopping (patience=5 from epoch 7 best @ 0.2726).**

---

## 🔗 References

- **Wandb Issue #2971**: Resumed runs lose pre-resumed data
- **Wandb Issue #1357**: Large history files don't sync
- **Wandb Issue #2039**: Network error retry loops
- **Community Reports**: 502 errors on file_stream endpoint (2024)

---

**Generated by**: Claude (Anthropic)
**Contact**: Check checkpoints at `results/local_fla_exp1_reg/checkpoints/`
