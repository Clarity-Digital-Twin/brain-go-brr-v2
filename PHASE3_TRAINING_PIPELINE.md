# PHASE3_TRAINING_PIPELINE.md — Training & Evaluation Orchestration

## 🎯 Phase 3 Goal
Build a robust, reproducible training and evaluation pipeline that trains the Phase 2 model on standardized EEG windows and reports clinically relevant metrics (TAES, FA/24h curves) with clean logging, checkpointing, and early stopping.

## 📋 Phase 3 Checklist
- [ ] Data loaders (balanced sampling, reproducible splits)
- [ ] Loss/optimizer/scheduler wired to config
- [ ] AMP + gradient clipping + sanity protections
- [ ] Validation loop with TAES, FA@{10,5,2.5,1}
- [ ] Checkpointing + early stopping (metric from config)
- [ ] Logging (TensorBoard; optional Weights & Biases)
- [ ] CLI integration (`python -m src.experiment.pipeline --config ...`)
- [ ] TDD: unit + smoke tests, quality gates green

## 🔧 Implementation Files
```
src/experiment/pipeline.py        # Orchestration: train/validate loops
src/experiment/evaluate.py        # Metrics: TAES, FA/24h, AUROC (stub -> full)
src/experiment/data.py            # (Phase 1) EEG I/O + preprocessing + windows
src/experiment/models.py          # (Phase 2) Full model

tests/test_training.py            # Training smoke/TDD
tests/test_evaluate.py            # Metrics unit tests (TAES, FA curves)
```

## 📐 Data & Splits
- Input windows: `(B, 19, 15360)` at 256 Hz (60s, 10s stride) — consistent with Phase 1.
- **Canonical training labels**: `(B, 15360)` binary masks (any-channel seizure). If a dataset provides per-channel labels `(B, 19, 15360)`, aggregate once via `labels.max(dim=1)[0]`.
- Train/val split: `data.validation_split` from config (default 0.2).
- **Balanced sampling**: Use `WeightedRandomSampler` at dataset scope to oversample positive windows (compute once, not per batch).
  ```python
  # window_labels: (N, T) aggregated binary labels per window
  window_has_seizure = (window_labels.max(dim=1).values > 0).float()  # (N,)
  pos_ratio = float(window_has_seizure.mean().item())
  pos_weight = (1 - pos_ratio) / max(pos_ratio, 1e-8)

  # Per-sample weights: positives get pos_weight, negatives 1.0
  sample_weights = torch.where(window_has_seizure > 0, pos_weight, 1.0)
  sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
  ```
- DataLoader:
  - `batch_size`: from config
  - `num_workers`: from config
  - `pin_memory=True` when CUDA available
  - `sampler=sampler` for balanced training
  - Deterministic seeding from `experiment.seed` (set a `torch.Generator` with `manual_seed(seed)` and a `worker_init_fn` to seed NumPy/Python per worker).

## 🔨 Loss, Optimizer, Scheduler
- **Loss**: Binary cross-entropy over per-timestep probabilities with element-wise weighting.
  - Model outputs `(B, 15360)` (sigmoid already applied in Phase 2.5)
  - Labels: `(B, 15360)` (aggregate once if necessary)
  - Class imbalance handling (99% neg / 1% pos typical): compute `pos_ratio` over the training set, then do element-wise weighting with reduction='none'.
  ```python
  bce = nn.BCELoss(reduction='none')
  pos_weight = (1 - pos_ratio) / max(pos_ratio, 1e-8)
  weights = torch.where(labels_agg > 0, pos_weight, 1.0)  # (B, 15360)
  per_elem = bce(outputs, labels_agg)
  loss = (per_elem * weights).sum() / (weights.sum() + 1e-8)
  ```
  - Note: If you prefer logits + `nn.BCEWithLogitsLoss(pos_weight=...)`, remove the Sigmoid in the model head and adjust accordingly.
- **Optimizer**: AdamW with `learning_rate`, `weight_decay` from `training` config.
- **Scheduler**: Cosine with warmup (`scheduler.type=cosine`, `warmup_ratio` from config). Step per-iteration if using a step-based schedule (recommended for windows); otherwise step per-epoch. Document and keep consistent with the chosen scheduler.
- **Gradient clipping**: `training.gradient_clip` (global-norm) each step.
- **Mixed precision**: `training.mixed_precision` → use `torch.cuda.amp.autocast` + `GradScaler` when CUDA.

## 🔁 Training Loop (Epoch)
1) **Train step**:
   ```python
   for batch_idx, (windows, labels) in enumerate(train_loader):
       # windows: (B, 19, 15360), labels: (B, 19, 15360)
       labels_agg = labels.max(dim=1)[0]  # (B, 15360) any-seizure

       with autocast(enabled=use_amp):
           outputs = model(windows)  # (B, 15360)
           per_elem = bce(outputs, labels_agg)           # reduction='none'
           weights = torch.where(labels_agg > 0, pos_weight, 1.0)
           loss = (per_elem * weights).sum() / (weights.sum() + 1e-8)

       scaler.scale(loss).backward()
       scaler.unscale_(optimizer)
       torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
       scaler.step(optimizer)
       scaler.update()
       scheduler.step()
   ```

2) **Validation step** (end of epoch):
   - Forward on val set → collect probabilities over time
   - Post-processing → events: apply hysteresis from `postprocessing.hysteresis` (τ_on=0.86, τ_off=0.78), then morphology (`postprocessing.morphology.kernel_size`), then convert to event intervals [(start_s, end_s), ...]
   - Metrics (event-level): TAES, AUROC (prob‑wise), sensitivity@FA rates and FA/24h computed from event lists

3) **Early stopping + checkpointing**:
   - Metric: `training.early_stopping.metric` (default: `sensitivity_at_10fa`)
   - Patience from config
   - Checkpoint format:
     ```python
     checkpoint = {
         'epoch': epoch,
         'model_state_dict': model.state_dict(),
         'optimizer_state_dict': optimizer.state_dict(),
         'scheduler_state_dict': scheduler.state_dict(),
         'config': config.to_dict(),
         'best_metric': best_metric,
         'metric_name': early_stopping.metric
     }
     torch.save(checkpoint, 'results/checkpoints/best.pt')
     ```

## 🧪 Evaluation & Metrics

### TAES Implementation (Time-Aligned Event Scoring)
Based on Picone et al. 2021, TAES weights errors by temporal overlap:
```python
def calculate_taes(
    pred_events: List[Tuple[float, float]],  # [(start, end), ...]
    ref_events: List[Tuple[float, float]],
) -> float:
    """
    Calculate TAES metric based on temporal overlap.

    For each reference event:
    - Find overlapping predictions
    - Weight by overlap percentage
    - Penalize false alarms by duration

    Returns: TAES score in [0, 1], higher is better
    """
    # Implementation per literature:
    # 1. Calculate overlap ratios for each ref event
    # 2. Weight TPs by overlap percentage
    # 3. Penalize FPs by their duration
    # 4. Normalize to [0, 1]
```

### FA/24h Calculation (event-level)
```python
def fa_per_24h(
    pred_events: List[List[Tuple[float, float]]],  # per-record predicted events (s)
    ref_events: List[List[Tuple[float, float]]],   # per-record reference events (s)
    total_hours: float,
) -> float:
    """False alarms per 24h: count predicted events with no overlap to any ref event, normalized by duration."""
    fa = 0
    for preds, refs in zip(pred_events, ref_events):
        for p in preds:
            if not any(overlap(p, r) > 0 for r in refs):
                fa += 1
    return (fa / max(total_hours, 1e-8)) * 24.0
```

### Sensitivity at FA Rates (event-level)
```python
def sensitivity_at_fa_rates(
    probs: torch.Tensor,                   # (N, T) probabilities
    labels: torch.Tensor,                  # (N, T) binary
    fa_targets: List[float],               # [10, 5, 2.5, 1]
    post_cfg: PostprocessingConfig,
    sampling_rate: int = 256,
) -> Dict[str, float]:
    """Find sensitivity at specific FA/24h operating points using eventization + threshold search."""
    results = {}
    # Convert reference labels to event lists once
    ref_events = batch_masks_to_events(labels, sampling_rate)
    total_hours = labels.numel() / (sampling_rate * 3600)

    for fa_target in fa_targets:
        thr = find_threshold_for_fa_eventized(
            probs, post_cfg, ref_events, fa_target, total_hours, sampling_rate
        )
        pred_events = batch_probs_to_events(probs, post_cfg, sampling_rate, threshold=thr)
        # Sensitivity = fraction of ref events overlapped by any prediction
        tp = 0
        n_ref = 0
        for refs, preds in zip(ref_events, pred_events):
            n_ref += len(refs)
            for r in refs:
                if any(overlap(r, p) > 0 for p in preds):
                    tp += 1
        results[f'sensitivity_at_{fa_target}fa'] = float(tp / max(n_ref, 1))
    return results
```

Expected evaluation API:
```python
from src.experiment.schemas import PostprocessingConfig

def evaluate_predictions(
    probs: torch.Tensor,             # (N, T)
    labels: torch.Tensor,            # (N, T)
    fa_rates: List[float],           # [10, 5, 2.5, 1]
    post_cfg: PostprocessingConfig,  # Pydantic config
) -> Dict[str, Any]:
    """
    Return dict of metrics including TAES, AUROC, and sensitivity@FA.

    Keys: {
        'taes': float,
        'auroc': float,
        'sensitivity_at_10fa': float,
        'sensitivity_at_5fa': float,
        'sensitivity_at_2.5fa': float,
        'sensitivity_at_1fa': float,
        'fa_curve': List[Tuple[float, float]],  # (FA/24h, sensitivity)
    }
    """
```

## 🧾 Logging & Checkpointing
- **TensorBoard**: scalar logs (train/val loss, metrics), learning-rate, gradients optional
- **Weights & Biases** (optional): use `experiment.wandb.enabled`
- **Checkpoints**:
  - Best by metric: `results/checkpoints/best.pt`
  - Last epoch: `results/checkpoints/last.pt`
  - Resume capability with full state restoration

## 🖥️ CLI & Commands
- Validate config: `python -m src.cli validate configs/local.yaml`
- Train (local/dev): `python -m src.experiment.pipeline --config configs/local.yaml`
- Train (production): `python -m src.experiment.pipeline --config configs/production.yaml`
- Resume training: `python -m src.experiment.pipeline --config configs/local.yaml --resume results/checkpoints/last.pt`
- Makefile shortcuts:
  - `make train-local` → local config
  - `make train-prod` → production config

## 🧪 TDD Plan (Tests First)

### `tests/test_training.py` (smoke)
```python
def test_training_smoke():
    """Single epoch with tiny synthetic data."""
    # Create balanced synthetic dataset (10 windows)
    windows = torch.randn(10, 19, 15360)
    labels = torch.zeros(10, 19, 15360)
    labels[::2, :, 5000:10000] = 1  # 50% positive

    # Mini config
    config = Config.from_yaml('configs/local.yaml')
    config.data.max_samples = 10
    config.training.epochs = 1

    # Train one epoch
    model = SeizureDetectorV2()
    train_epoch(model, dataloader, config)

    # Verify: loss finite, checkpoint written, no NaN
```

### `tests/test_evaluate.py` (unit)
```python
def test_taes_perfect_overlap():
    """TAES should be 1.0 for perfect overlap."""
    ref_events = [(10, 20), (30, 40)]
    pred_events = [(10, 20), (30, 40)]
    assert calculate_taes(pred_events, ref_events) == 1.0

def test_taes_no_overlap():
    """TAES should be 0.0 for no overlap."""
    ref_events = [(10, 20)]
    pred_events = [(30, 40)]
    assert calculate_taes(pred_events, ref_events) == 0.0

def test_fa_per_24h_event_level():
    """Verify FA/24h counts predicted events without overlap to refs."""
    # 1 hour total, 10 predicted events, 0 reference events → 10 FA/hr → 240 FA/24h
    pred_events = [[(i*300.0, i*300.0 + 5.0) for i in range(10)]]  # 10 events in 3600s
    ref_events = [[]]
    fa_rate = fa_per_24h(pred_events, ref_events, total_hours=1.0)
    assert abs(fa_rate - 240.0) < 1e-3

def test_sensitivity_at_fa_event_level(post_cfg):
    """Monotonicity: sensitivity decreases as target FA/24h decreases."""
    # Construct simple probabilities/labels where higher thresholds reduce FA and sensitivity
    probs = torch.linspace(0, 1, steps=2048).unsqueeze(0)        # (1, T)
    labels = (probs > 0.6).float()                               # (1, T)
    metrics = sensitivity_at_fa_rates(probs, labels, [10, 5, 1], post_cfg, sampling_rate=256)
    assert metrics['sensitivity_at_10fa'] >= metrics['sensitivity_at_5fa']
    assert metrics['sensitivity_at_5fa'] >= metrics['sensitivity_at_1fa']
```

Quality gates:
- `make q` (ruff + format + mypy) → all files green
- `pytest -m unit -v` and `pytest -m integration -v` on CI

## ⚙️ Implementation Notes
- **Seeding**: set `torch`, `numpy`, `random`, and DataLoader seeds from `experiment.seed`
- **DataLoader determinism**: pass a `torch.Generator().manual_seed(seed)` to DataLoader and a `worker_init_fn` that seeds NumPy/Python per worker.
- **Device**: `experiment.device=auto` → pick CUDA if available; otherwise CPU
- **Class imbalance**: typical 99% negative, 1% positive
  - For probabilities output: use element-wise weighting with `BCELoss(reduction='none')` as shown above
  - For logits output: use `BCEWithLogitsLoss(pos_weight=...)` and remove Sigmoid from the model head
  - WeightedRandomSampler at dataset scope for balanced mini-batches
- **Performance**:
  - Use `pin_memory=True` with CUDA
  - Consider `torch.compile` for model when PyTorch 2.5+
  - Pre-allocate tensors; avoid `.cpu()` copies in loops

## ✅ Definition of Done (Phase 3)
1) Training and validation loops implemented and configurable from YAML
2) Metrics computed per spec (TAES, AUROC, FA curves; sensitivity@{10,5,2.5,1})
3) Reproducible results with fixed seed
4) Checkpointing + early stopping honoring config
5) Logging via TensorBoard; optional W&B
6) Quality gates green; tests passing (unit + smoke)

## 🔗 Config Alignment (matches schemas.py exactly)
- Optimizer: `training.optimizer=adamw`
- Scheduler: `training.scheduler.type=cosine`, `warmup_ratio`
- AMP: `training.mixed_precision=true|false`
- Clip: `training.gradient_clip` (float)
- Early stopping: `training.early_stopping.{patience, metric}`
- Post-processing: `postprocessing.hysteresis.{tau_on,tau_off}`, `morphology.kernel_size`
- Evaluation: `evaluation.{metrics,fa_rates}`
- Mamba conv: `model.mamba.conv_kernel` (NOT d_conv or mamba_d_conv)

---
Status: Ready for implementation (TDD-first) ✅
Estimated Time: 2–3 days
Owners: Training duo (eng + reviewer) 🔬🚀
