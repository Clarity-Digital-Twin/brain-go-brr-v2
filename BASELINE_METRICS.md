# Baseline FLA Training Metrics

**Experiment**: Full FLA Gated DeltaNet baseline (31M params)
**Config**: `configs/local/train_fla.yaml`
**Status**: 🟢 RUNNING (Epoch 20 validating - 9% complete)
**Started**: Oct 16, 2025 10:55 AM EDT
**Hardware**: RTX 4090 (24GB VRAM)
**Wandb Run (Old)**: https://wandb.ai/jj-vcmcswaggins-novamindnyc/seizure-v3-rtx4090/runs/5ee302c0a01d4e43b8e782fa2ffb0e90 (Epochs 0-15)
**Wandb Run (Current)**: https://wandb.ai/jj-vcmcswaggins-novamindnyc/seizure-v3-rtx4090/runs/c7eb044ceee34392ad1c793e783f4bc4 (Epoch 17+, partial sync)

---

## Training Configuration

```yaml
learning_rate: 0.0001
batch_size: 8
epochs: 100
optimizer: adamw
scheduler: cosine (warmup_ratio: 0.03)
weight_decay: 0.01
dropout: 0.1
gradient_clip: 0.5
mixed_precision: false

early_stopping:
  patience: 20
  min_epochs: 30
  metric: sensitivity_at_10fa
```

---

## Epoch Results

| Epoch | Sensitivity@10FA | Global Step | Date | Notes |
|-------|-----------------|-------------|------|-------|
| 8 | 0.2633 | 61,616 | Oct 16 | Early baseline |
| 9 | 0.2801 | 69,318 | Oct 16 | 🏆 Peak performance (so far) |
| 10 | 0.2801 | 77,020 | Oct 17 | Plateau start |
| 11 | 0.2801 | 84,722 | Oct 17 | Plateau continues |
| 12 | 0.2801 | 92,424 | Oct 18 | Plateau (pre-crash) |
| 13 | 0.2493 | 100,126 | Oct 23 | Post-resume, performance drop |
| 14 | 0.2493 | 107,828 | Oct 24 | Matches epoch 13, plateau low |
| 15 | 0.2493 | 115,530 | Oct 24 | Plateau continues |
| 16 | 0.2493 | 123,232 | Oct 25 | Plateau (3 epochs at 0.2493) |
| 17 | 0.2577 | 130,934 | Oct 25 | 🎉 Recovery! +3.4% improvement |
| 18 | 0.2577 | 138,636 | Oct 27 | Plateau at recovery level |
| 19 | 0.2577 | 146,338 | Oct 27 | Plateau continues (3 epochs) |
| 20 | Validating (9%) | TBD | Oct 27 | Will restart with WandB fix after validation |

**Best Overall**: 0.2801 @ Epoch 9 (Oct 16)
**Post-Resume Best**: 0.2577 @ Epochs 17-19 (Oct 25-27) - Plateaued after recovery

---

## Training Timeline

- **Oct 16, 10:55**: Training started
- **Oct 18, 13:47**: WSL2 suspend (mid-epoch 13, batch 7355)
- **Oct 23, 17:31**: Training resumed from mid-epoch checkpoint
- **Oct 23, 23:15**: Epoch 13 completed
- **Oct 24, 09:24**: Epoch 14 complete (0.2493)
- **Oct 24, 19:35**: Epoch 15 complete (0.2493)
- **Oct 25, 05:46**: Epoch 16 complete (0.2493)
- **Oct 25, 15:55**: Epoch 17 complete (0.2577 - recovery!)
- **Oct 25, 18:39**: Epoch 18 crash (glibc malloc corruption, WSL2 issue)
- **Oct 26, 09:53**: Training resumed from epoch 18, batch 4165
- **Oct 26, 16:52**: New WandB run started (c7eb044c) - resumed training
- **Oct 27, 03:04**: Epoch 18 complete (0.2577 - plateau continues)
- **Oct 27, 13:16**: Epoch 19 complete (0.2577 - 3rd epoch at recovery level)
- **Oct 27, ~18:00**: Epoch 20 validation started (currently 9% complete)
- **Oct 27, ~23:00** (est): Will stop training and restart with .env WandB fix

---

## Key Observations

### Plateau Pattern (Epochs 9-12)
- Metric stuck at 0.2801 for 4 consecutive epochs
- Prompted investigation into overfitting hypothesis
- Led to Exp1 (regularization) experiment

### Post-Resume Performance Pattern (Epochs 13-19)
- **Initial Drop (Epochs 13-16)**: Sensitivity dropped to 0.2493 after resume (from 0.2801 peak)
  - **-11% decline** from best (epoch 9)
  - Plateaued for 4 epochs (13-16) at 0.2493
  - Possible causes: resume disruption, scheduler state, natural variance

- **Recovery (Epoch 17)**: 🎉 **0.2577 (+3.4% improvement!)**
  - First improvement in 5 epochs
  - Still **-8%** below peak (0.2801) but recovery confirmed

- **New Plateau (Epochs 17-19)**: 📊 **0.2577 (stable for 3 epochs)**
  - Model settled at new plateau after recovery
  - Consistent performance at recovery level
  - May need intervention to break plateau (patience counter: 13/20)

### Experiment Context
- **Exp1 (Regularization)**: Cancelled @ epoch 9, peaked at 0.2726 (worse than baseline)
- **Conclusion**: Model NOT overfitting, regularization harmful
- **Current Strategy**: Continue baseline to epoch 20-30, evaluate Exp2 (lower LR) at epoch 17

---

## Next Milestones

- ✅ **Epoch 14** (Oct 24): Plateau continued at 0.2493
- ✅ **Epoch 15** (Oct 24): Plateau continued at 0.2493
- ✅ **Epoch 17** (Oct 25): Recovery to 0.2577 confirmed
- 🔄 **Epoch 20** (Oct 27, ~23:00): Will restart training with WandB .env fix after validation
- **Epoch 21+** (~Oct 28+): Training resumes with proper WandB syncing
- **Epoch 30** (~Nov 3): Min epochs reached, patience countdown begins (if no improvement)

### WandB Data Status
- ✅ **Epochs 0-15**: Synced to old run (5ee302c0)
- ❌ **Epochs 16, 18, 19**: Lost (not synced to WandB)
- ⚠️ **Epoch 17**: Partially synced to new run (c7eb044c)
- 🔄 **Epoch 20**: Will be lost unless manually synced
- ✅ **Epoch 21+**: Will sync automatically with .env fix

---

## Update Instructions

After each epoch completes, extract metrics:

```bash
python3 << 'EOF'
import torch
epoch = 14  # Update this
ckpt = torch.load(f'results/local_fla_training/checkpoints/epoch_{epoch:03d}.pt',
                 map_location='cpu', weights_only=False)
print(f"Epoch {epoch}: {ckpt.get('best_metric', 0):.4f} @ step {ckpt.get('global_step', 0):,}")
EOF
```

Add the result to the table above.

---

**Last Updated**: Oct 27, 2025 18:55 EDT (Epoch 20 validating 9%, will restart with WandB .env fix after completion. **New plateau at 0.2577 for 3 epochs.**)
