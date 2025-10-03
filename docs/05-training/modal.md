# Modal Training (A100-80GB)

**Last Updated**: October 1, 2025 (v3.4.1)

## CRITICAL: Always Use --detach
Modal functions will stop after ~8 minutes when terminal disconnects unless you use `--detach`.
**All long-running commands MUST use --detach or run in tmux!**

## Quick Commands

- Populate cache (one-time, from S3): `modal run --detach deploy/modal/app.py --action populate-cache`
  - **IMPORTANT**: This command removes existing `/results/cache/tusz/{train,dev}` before copying fresh data from S3
  - Training uses the SSD cache and will NOT clear it unless you re-run populate-cache or clean-cache
- Test Mamba CUDA: `modal run deploy/modal/app.py --action test-mamba`
- Smoke: `modal run --detach deploy/modal/app.py --action train --config configs/modal/smoke.yaml`
- Full (detached): `modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml`
- Clean old cache (if needed): `modal run deploy/modal/app.py --action clean-cache`

## Monitoring

- List running apps: `modal app list`
- Stream logs: `modal app logs <app-id>`
- Stop training: `modal app stop <app-id>`

## Configuration (v3.4.1)

### Resources (deploy/modal/app.py:501-503)
```python
gpu: modal.gpu.A100(count=1, size="80GB")
memory: 98304  # 96GB RAM (3x safety margin)
cpu: 24        # 24 cores (avoids CPU bottleneck during data loading)
```

### Model Parameters (configs/modal/train.yaml)
```yaml
training:
  batch_size: 32                # v3.4.1: Reduced from 64 (OOM fix)
  gradient_accumulation_steps: 2  # v3.4.1: Maintain effective batch=64
  mixed_precision: true         # CRITICAL: 3.8x faster on A100
  learning_rate: 8.0e-5         # Batch-scaled
  gradient_clip: 0.5            # Gradient protection
  loss: focal                   # REQUIRED for 12:1 imbalance
  focal_gamma: 2.0             # Warmup from 1.0 → 2.0 (v3.4.1)

model:
  architecture: v3              # Dual-stream (node + edge Mamba)
  warmup_schedule:              # v3.4.1: Gradient stabilization
    enabled: true
    warmup_steps: 1000
    adj_temperature_enabled: true
    focal_gamma_enabled: true

  graph:
    edge_similarity_margin: 0.01  # v3.3.0: Boundary safety
    use_dynamic_pe: true          # Always enabled with safeguards
    semi_dynamic_interval: 5      # Optimal update rate
```

## Critical Fixes (v3.4.1)

### A100 OOM Fix (October 2025)
**Problem**: Training crashed at batch 0 backward pass with OOM
**Error**: `CUDA out of memory. Tried to allocate 10.69 GiB. GPU 0 has total capacity of 79.25 GiB of which 2.04 GiB is free. Process 1 has 77.20 GiB memory in use.`

**Root Cause**:
- `batch_size=64` + `gradient_accumulation_steps=1` processes **64 samples in forward+backward**
- Peak memory during backward: **~77GB** (exceeds A100-80GB capacity)

**Memory Profile**:
```
batch_size=64, grad_accum=1:  Peak = 77GB (CRASH ❌)
batch_size=32, grad_accum=2:  Peak = 50GB (SAFE ✅)
```

**Fix Applied**:
```yaml
# configs/modal/train.yaml
training:
  batch_size: 32                # Reduced from 64
  gradient_accumulation_steps: 2  # Increased from 1
  # Effective batch still 64, peak memory reduced by ~35%
```

**Key Insight**:
- **batch_size** controls **peak memory** (forward+backward activations stored simultaneously)
- **gradient_accumulation** splits backward into smaller chunks, reducing peak
- Both configs have **identical effective batch** (64) and **same learning dynamics**
- Only difference: memory footprint

See `configs/README.md` for full OOM analysis.

### Hang Detection & Logging (deploy/modal/app.py:722-723, loop.py:440)
**Problem**: Training appeared to hang for 60+ minutes during initialization
**Fix**: Enhanced logging and faster heartbeats
```python
# Modal auto-sets environment variables:
BGB_LOG_EVERY_N_STEPS=10     # Log every 10 batches (vs default 50)
heartbeat_interval=120        # 2-minute heartbeats (vs 5 minutes)
```

### XID 31 GPU Crash Prevention (deploy/modal/app.py:541-551)
**Problem**: A100 crashes with "XID 31 MMU Fault" due to memory fragmentation
**Fix**: Memory allocator optimization + unique Triton cache per run
```python
# deploy/modal/app.py:541-542
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

# deploy/modal/app.py:549-550 - Prevent stale kernel cache
TRITON_CACHE_DIR=f"/tmp/triton_cache_run_{run_id}"
TORCHINDUCTOR_CACHE_DIR=f"/tmp/tii_cache_run_{run_id}"
```

See `docs_v2/reference/incidents/modal-xid31-recurrence.md` for detailed investigation.

### NaN Protection (deploy/modal/app.py:720-723)
Modal automatically enables gradient sanitization:
```python
BGB_SANITIZE_GRADS=1     # Sanitize NaN gradients
BGB_NAN_DEBUG=1          # Debug NaN losses
```

## Initialization Timeline (v3.4.1)

**Total initialization: ~10-15 minutes** before first epoch starts

| Phase | Duration | What's Happening |
|-------|----------|------------------|
| Startup | ~1 min | Container launch, env setup |
| Train manifest load | ~2-3 min | Load 61,616 windows from manifest.json (cached) |
| Dev manifest load | ~2-3 min | Load 148,224 windows from manifest.json (cached) |
| Model creation | ~10 sec | Initialize 31M parameters |
| W&B initialization | ~3 sec | Connect to Weights & Biases |
| Worker spawn | **~3-5 min** | DataLoader workers (v3.4.1: fixed from 1h+) |
| Preflight batch | ~2 min | Test forward/backward pass |
| **Total to epoch start** | **~10-15 min** | When training actually begins |

**v3.4.1 Fix**: Worker spawn reduced from 1h+ to <5 min by setting `persistent_workers: false` and `num_workers: 4` (see MODAL_TRAINING_HANG_INVESTIGATION.md). See **DataLoader profiles** below for throughput-oriented variants once the baseline is stable.

## DataLoader profiles

Baseline (safe)

```yaml
data:
  num_workers: 4
  persistent_workers: false
  prefetch_factor: 2
training:
  batch_size: 32
  gradient_accumulation_steps: 2  # effective 64
```

- Crash-proof configuration used for day-to-day runs.
- Keeps first epoch under ~15 minutes and peak VRAM comfortably below 50 GB.

Throughput profile (after smoke verification)

```yaml
data:
  num_workers: 8
  persistent_workers: false
  prefetch_factor: 4
```

- Doubles data-loading throughput while keeping startup predictable.

Aggressive profile (benchmarking only)

```yaml
data:
  num_workers: 8
  persistent_workers: true
  prefetch_factor: 4
training:
  batch_size: 64
  gradient_accumulation_steps: 1
```

- Re-enables persistent workers after capping prefetch. Only keep if the first epoch remains <15 minutes and VRAM stays <70 GB; otherwise drop back to the safe profile.

## Cache and Volumes

- Raw data mounted at `/data/edf/` (read‑only dataset mount)
- Cache on persistent SSD volume at `/results/cache/tusz` (patient‑disjoint subdirs: `{train,dev}`)
- Results saved to `/results/` (same persistent volume)
- Ensure `data.data_dir: /data/edf`, `data.split_policy: official_tusz`
- Ensure `data.cache_dir: /results/cache/tusz` in configs
- Do not use S3 for cache on Modal; prebuilt caches should be synced into the Modal volume

## Patient Disjointness

On startup, the app verifies that patient sets in `/data/edf/train` and `/data/edf/dev` are disjoint and aborts if not (deploy/modal/app.py:569-581).

## Resuming

- Use `--resume` flag or set `training.resume: true`.
- Training prioritizes `mid_epoch_*.pt` when resuming; falls back to `last.pt`.

## Verification Checklist

### After populate-cache
Check logs for:
```
[COPY] ✅ Copied 4667 train files
[COPY] ✅ Copied 1832 dev files
[COPY] ✅ Copied metadata file
✅ Cache population complete! 4667 train, 1832 dev files
```

### During training startup
Expect to see:
```
[CACHE] ✅ Cache built with official_tusz policy
[DATASET] BalancedSeizureDataset: XXXX windows from manifest
[DATASET] Seizure ratio: XX% (from manifest)
[DATASET] Using pos_weight: X.XX (sqrt scaling)
```

### Warning Signs
- "Seizure ratio: 0%" → Missing manifest, rebuild and re-upload
- "Falling back to EEGWindowDataset" → Manifest creation failed
- Training hangs at epoch boundaries → Increase CPU/RAM allocation
- No W&B logs after 20+ minutes → Check Modal logs for actual errors (baseline emits W&B within the first 10 minutes)

## Troubleshooting

### Training appears stuck during initialization
**Expected**: 10-15 minutes with the safe dataloader profile
- Check logs: `modal app logs <app-id>`
- Look for the cache validation lines and the first `[BATCH START]` message.
- If startup exceeds ~15 minutes, inspect worker spawn counts; revert to the baseline profile (4 workers, `persistent_workers: false`) before investigating further.

### XID 31 GPU crashes
**Cause**: A100 memory fragmentation or stale Triton cache
**Fix**: Already implemented in v3.4.1 (deploy/modal/app.py:541-551)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- Unique Triton cache per run
- If still occurring: Check Modal logs for kernel compilation errors

### Zero seizures in batches
**Cause**: Missing manifest or balanced sampling disabled
**Fix**:
1. Verify manifest exists: `modal volume ls seizure-detection-data results/cache/tusz/train/manifest.json`
2. Ensure `use_balanced_sampling: true` in config
3. Rebuild cache if needed: `modal run --detach deploy/modal/app.py --action populate-cache`

### PyG/Mamba import issues
**Stack versions** (deploy/modal/app.py:16-18,36-38):
- CUDA 12.4.0
- PyTorch 2.5.0+cu124
- mamba-ssm 2.2.5 (with PR #708 patch applied)
- causal-conv1d 1.5.2
- PyTorch Geometric 2.6.1

If imports fail: Check Modal image build logs for compilation errors.

### Slow epoch boundaries
**Fix**: Already implemented - 24 CPU + 96GB RAM (deploy/modal/app.py:501-502)

### Commands
- View logs: `modal app logs <app-id>`
- Stop training: `modal app stop <app-id>`
- List volumes: `modal volume ls seizure-detection-data`
