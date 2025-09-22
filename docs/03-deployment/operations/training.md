Training (Loop, Dataloaders, Guards)

Scope
- Train/val split handling, dataset choice, samplers, logging, guardrails.

LR scheduling
- The training loop steps the LR scheduler once per optimizer update and does so after the optimizer step. See `src/brain_brr/train/loop.py:555`.
- A PyTorch warning may appear on the first batch: `Detected call of lr_scheduler.step() before optimizer.step()`.
  - In this codebase, the order is correct: `optimizer.step()` (or `scaler.step(optimizer)` when AMP is enabled) happens before `scheduler.step()`.
  - Why the warning can still show:
    - LambdaLR with `last_epoch=-1` initializes to base LR and the first call sets step=0. Some torch versions emit the warning even when the optimizer has stepped within the same iteration.
    - With AMP, if the very first update overflows, `scaler.step(optimizer)` skips the underlying `optimizer.step()`. The scheduler still advances one step by design; this can trigger the warning once. The schedule remains well‑defined and training proceeds normally.
  - Impact: benign for our per‑step warmup+cosine schedule. It can shift the schedule by at most one step in the rare first‑batch overflow case; smoke runs and typical training are unaffected.
  - Verify locally by logging `optimizer.param_groups[0]['lr']` across the first few steps or comparing `scheduler.get_last_lr()` before/after `optimizer.step()`.

Dataset selection
- If manifest exists and use_balanced is enabled → BalancedSeizureDataset for train.
- Otherwise → EEGWindowDataset (classic) for train.
- Validation/test always use EEGWindowDataset (no balancing).
- WeightedRandomSampler is bypassed when using BalancedSeizureDataset.

Guardrails
- On scan‑cache/build‑cache: exit non‑zero if partial==0 and full==0.
- On train startup: abort if BalancedSeizureDataset length is 0.
- Logs dataset composition (partial/full/background counts and ratios).

Checkpoints & early stopping
- Checkpoints live under `<config.experiment.output_dir>/checkpoints/`:
  - `best.pt` (best by early_stopping.metric)
  - `last.pt` (last epoch)
- Early stopping config: `training.early_stopping.{metric,mode,patience}`
- Resume restores model/optimizer/scheduler and saved config snapshot.

Config notes
- Local (WSL2): prefer `num_workers: 4`, `pin_memory: true`, `persistent_workers: true`, `prefetch_factor: 2`. If you encounter hangs, fall back to `num_workers: 0`, `pin_memory: false`.
- Keep configs/local/smoke.yaml for quick verification; use local/train.yaml vs modal/train.yaml for full runs.
- Balanced sampling: training validates the manifest and rebuilds if empty/stale; set `BGB_FORCE_MANIFEST_REBUILD=1` to force a fresh manifest.

Code anchors
- src/brain_brr/train/loop.py (dataset selection, fail‑fast, sampler bypass, checkpoints)
- src/brain_brr/cli/cli.py (train entry)
- configs/local/{train.yaml,smoke.yaml}; configs/modal/{train.yaml,smoke.yaml}

Cross-references
- Architecture: ../02-architecture/canonical-spec.md (training configuration)
- Deployment: ./deploy-preflight.md, ./deploy-modal.md
