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
- Train/val split: `data.validation_split` from config (default 0.2).
- Balanced sampling: aim for ~50% seizure windows during training (oversample positives if dataset is imbalanced).
- DataLoader:
  - `batch_size`: from config
  - `num_workers`: from config
  - `pin_memory=True` when CUDA available
  - Deterministic seeding from `experiment.seed` (set before DataLoader init).

## 🔨 Loss, Optimizer, Scheduler
- Loss: Binary cross-entropy (BCE) over per-timestep probabilities.
  - Model outputs `sigmoid` per Phase 2 → use `torch.nn.BCELoss` on `(B, 15360)` vs `(B, 15360)` label masks.
  - Optional weighting: per-batch positive/negative ratio (simple scalar weights) if needed for heavy class imbalance.
- Optimizer: AdamW with `learning_rate`, `weight_decay` from `training` config.
- Scheduler: Cosine with warmup (`scheduler.type=cosine`, `warmup_ratio` from config).
- Gradient clipping: `training.gradient_clip` (global-norm) each step.
- Mixed precision: `training.mixed_precision` → use `torch.autocast` + `GradScaler` when CUDA.

## 🔁 Training Loop (Epoch)
1) Train step:
   - forward → loss (BCE)
   - backward (AMP scaler if enabled)
   - clip gradients → step optimizer → step scheduler
   - log running loss every `logging.log_every_n_steps` steps
2) Validation step (end of epoch):
   - forward on val set → gather predicted probabilities
   - post-processing: hysteresis and morphology from `postprocessing.*` (τ_on/τ_off, kernel_size)
   - metrics: TAES, sensitivity, specificity, AUROC, FA@{10,5,2.5,1}
3) Early stopping + checkpointing:
   - metric: `training.early_stopping.metric` (local default: `sensitivity_at_10fa`; production: `sensitivity_at_5fa`)
   - patience from config; save best model under `results/checkpoints/best.pt`

## 🧪 Evaluation & Metrics
- Hysteresis (τ_on=0.86, τ_off=0.78) applied to per-timestep probabilities → binary timeline.
- FA/24h: count false alarms normalized to 24h; compute sensitivity at FA targets [10,5,2.5,1].
- TAES: Time-Aligned Event Scoring for alignment quality (see NEDC/TAES references). Implementation lives in `src/experiment/evaluate.py`.
- AUROC: sanity check metric; not the clinical primary.

Expected evaluation API (to implement in Phase 3):
```python
def evaluate_predictions(
    probs: torch.Tensor,        # (N, T)
    labels: torch.Tensor,       # (N, T)
    fa_rates: list[float],      # [10, 5, 2.5, 1]
    post_cfg: dict,             # {"tau_on": 0.86, "tau_off": 0.78, "kernel_size": 5}
) -> dict:
    """Return dict of metrics including TAES, AUROC, and sensitivity@FA.
    Keys suggested: {
        'taes': float,
        'auroc': float,
        'sensitivity_at_10fa': float,
        'sensitivity_at_5fa': float,
        'sensitivity_at_2_5fa': float,
        'sensitivity_at_1fa': float,
        'fa_curve': list[tuple[float, float]],  # (FA/24h, sensitivity)
    }
    """
```

## 🧾 Logging & Checkpointing
- TensorBoard: scalar logs (train/val loss, metrics), learning-rate, and histograms optional.
- Weights & Biases (optional): use `experiment.wandb.enabled`; replicate the same keys as TensorBoard.
- Checkpoints:
  - Save best by early-stopping metric under `results/checkpoints/best.pt` (model + config + epoch + metric).
  - Save last epoch under `results/checkpoints/last.pt` for resume.

## 🖥️ CLI & Commands
- Validate config: `python -m src.cli validate configs/local.yaml`
- Train (local/dev): `python -m src.experiment.pipeline --config configs/local.yaml`
- Train (production): `python -m src.experiment.pipeline --config configs/production.yaml`
- Makefile shortcuts:
  - `make train-local` → local config
  - `make train-prod` → production config

## 🧪 TDD Plan (Tests First)
Place tests in `tests/` with markers and small synthetic data to avoid heavy deps.

- `tests/test_training.py` (smoke)
  - Builds a tiny synthetic dataset (balanced windows) → 1–2 mini-batches
  - Runs 1 epoch with AMP disabled (CPU), verifies: forward/back succeeds; loss finite; checkpoint written
  - Verifies gradient clipping applied (no exploding gradients)

- `tests/test_evaluate.py` (unit)
  - Constructs simple prediction/label timelines to verify hysteresis and FA counting
  - Asserts sensitivity increases monotonically when FA targets increase
  - Asserts TAES is within [0,1] and improves with better alignment

Quality gates:
- `make q` (ruff + format + mypy) → all files green
- `pytest -m unit -v` and `pytest -m integration -v` on CI (CPU-only)

## ⚙️ Implementation Notes
- Seeding: set `torch`, `numpy`, and DataLoader seeds from `experiment.seed`.
- Device: `experiment.device=auto` → pick CUDA if available; otherwise CPU. Support AMP only when CUDA.
- Performance:
  - Use `pin_memory=True` with CUDA
  - Pre-allocate tensors where possible; avoid unnecessary `.cpu()` copies
  - Consider `torch.compile` for model when PyTorch 2.5+ and CUDA available
- Caching (later/infra): structure stages to return picklable outputs (e.g., predictions, metrics) so we can persist and reload.

## ✅ Definition of Done (Phase 3)
1) Training and validation loops implemented and configurable from YAML
2) Metrics computed per spec (TAES, AUROC, FA curves; sensitivity@{10,5,2.5,1})
3) Reproducible results with fixed seed
4) Checkpointing + early stopping honoring config
5) Logging via TensorBoard; optional W&B
6) Quality gates green; tests passing (unit + smoke)

## 🔗 Config Alignment (must match schemas)
- Optimizer: `training.optimizer=adamw`
- Scheduler: `training.scheduler.type=cosine`, `warmup_ratio` used
- AMP: `training.mixed_precision=true|false`
- Clip: `training.gradient_clip` (float)
- Early stopping: `training.early_stopping.{patience, metric}`
- Post-processing: `postprocessing.hysteresis.{tau_on,tau_off}`, `morphology.kernel_size`
- Evaluation: `evaluation.{metrics,fa_rates}`

---
Status: Ready for implementation (TDD-first) ✅
Estimated Time: 2–3 days
Owners: Training duo (eng + reviewer) 🔬🚀

