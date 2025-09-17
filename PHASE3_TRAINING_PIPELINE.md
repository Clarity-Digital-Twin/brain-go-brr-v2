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
- **Labels**: `(B, 19, 15360)` per-channel binary masks from Phase 1 data.py
- **Label Aggregation**: For training loss, reduce to `(B, 15360)` by taking max across channels (any-seizure).
- Train/val split: `data.validation_split` from config (default 0.2).
- **Balanced sampling**: Use `WeightedRandomSampler` to oversample positive windows.
  ```python
  # Calculate sample weights
  seizure_frames = (labels.max(dim=1)[0] > 0).sum(dim=1)  # per window
  weights = torch.where(seizure_frames > 0, pos_weight, 1.0)
  sampler = WeightedRandomSampler(weights, len(weights))
  ```
- DataLoader:
  - `batch_size`: from config
  - `num_workers`: from config
  - `pin_memory=True` when CUDA available
  - `sampler=sampler` for balanced training
  - Deterministic seeding from `experiment.seed`.

## 🔨 Loss, Optimizer, Scheduler
- **Loss**: Binary cross-entropy (BCE) over per-timestep probabilities.
  - Model outputs `(B, 15360)` per Phase 2.5 (sigmoid applied)
  - Labels: aggregate from `(B, 19, 15360)` → `(B, 15360)` via `labels.max(dim=1)[0]`
  - Use `torch.nn.BCELoss()` with optional `pos_weight` for class imbalance (99% negative, 1% positive typical)
  - `pos_weight = (1 - pos_ratio) / pos_ratio` where pos_ratio is fraction of positive samples
- **Optimizer**: AdamW with `learning_rate`, `weight_decay` from `training` config.
- **Scheduler**: Cosine with warmup (`scheduler.type=cosine`, `warmup_ratio` from config).
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
           loss = criterion(outputs, labels_agg)

       scaler.scale(loss).backward()
       scaler.unscale_(optimizer)
       torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
       scaler.step(optimizer)
       scaler.update()
       scheduler.step()
   ```

2) **Validation step** (end of epoch):
   - Forward on val set → gather predicted probabilities
   - Post-processing: hysteresis from `postprocessing.hysteresis` (τ_on=0.86, τ_off=0.78)
   - Morphological ops: `postprocessing.morphology.kernel_size`
   - Metrics: TAES, sensitivity, specificity, AUROC, FA@{10,5,2.5,1}

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

### FA/24h Calculation
```python
def calculate_fa_per_24h(
    predictions: torch.Tensor,  # (N, T) binary
    labels: torch.Tensor,       # (N, T) binary
    sampling_rate: int = 256,
) -> float:
    """Calculate false alarms normalized to 24 hours."""
    false_alarms = (predictions & ~labels).sum()
    total_hours = predictions.numel() / (sampling_rate * 3600)
    fa_per_24h = (false_alarms / total_hours) * 24
    return fa_per_24h
```

### Sensitivity at FA Rates
```python
def sensitivity_at_fa_rates(
    probs: torch.Tensor,        # (N, T) probabilities
    labels: torch.Tensor,       # (N, T) binary
    fa_targets: List[float],    # [10, 5, 2.5, 1]
) -> Dict[str, float]:
    """Find sensitivity at specific FA/24h operating points."""
    results = {}
    for fa_target in fa_targets:
        # Binary search for threshold that gives fa_target FA/24h
        threshold = find_threshold_for_fa(probs, labels, fa_target)
        preds = (probs >= threshold).float()
        sensitivity = (preds & labels).sum() / labels.sum()
        results[f'sensitivity_at_{fa_target}fa'] = sensitivity.item()
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

def test_fa_per_24h():
    """Verify FA/24h calculation."""
    # 256 Hz, 1 hour of data, 10 false alarms
    preds = torch.zeros(256 * 3600)
    labels = torch.zeros(256 * 3600)
    preds[:10] = 1  # 10 FAs

    fa_rate = calculate_fa_per_24h(preds, labels, 256)
    assert abs(fa_rate - 240.0) < 0.1  # 10 FA/hr * 24 hr

def test_sensitivity_at_fa():
    """Test sensitivity calculation at FA rates."""
    # Create synthetic data with known characteristics
    probs = torch.rand(1000)
    labels = (probs > 0.7).float()  # 30% positive

    metrics = sensitivity_at_fa_rates(probs, labels, [10, 5, 1])
    # Verify sensitivity decreases as FA rate decreases
    assert metrics['sensitivity_at_10fa'] >= metrics['sensitivity_at_5fa']
    assert metrics['sensitivity_at_5fa'] >= metrics['sensitivity_at_1fa']
```

Quality gates:
- `make q` (ruff + format + mypy) → all files green
- `pytest -m unit -v` and `pytest -m integration -v` on CI

## ⚙️ Implementation Notes
- **Seeding**: set `torch`, `numpy`, `random`, and DataLoader seeds from `experiment.seed`
- **Device**: `experiment.device=auto` → pick CUDA if available; otherwise CPU
- **Class imbalance**: typical 99% negative, 1% positive
  - Use `pos_weight` in BCELoss: `pos_weight = (1 - pos_ratio) / pos_ratio`
  - WeightedRandomSampler for balanced mini-batches
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

