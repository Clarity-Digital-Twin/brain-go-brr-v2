# Baseline FLA Training Metrics

**Experiment**: Full FLA Gated DeltaNet baseline (31M params)
**Config**: `configs/local/train_fla.yaml`
**Status**: ⚠️ HISTORICAL SNAPSHOT (baseline run notes from Oct–Nov 2025; superseded by Exp4)
**Started**: Oct 16, 2025 10:55 AM EDT
**Hardware**: RTX 4090 (24GB VRAM)
**Wandb Run (Old)**: https://wandb.ai/jj-vcmcswaggins-novamindnyc/seizure-v3-rtx4090/runs/5ee302c0a01d4e43b8e782fa2ffb0e90 (Epochs 0-15)
**Wandb Run (Current)**: https://wandb.ai/jj-vcmcswaggins-novamindnyc/seizure-v3-rtx4090/runs/c7eb044ceee34392ad1c793e783f4bc4 (Epoch 17+, partial sync)

---

## Current Best (Held-Out TUSZ Eval)

FLA Exp4 (Gated DeltaNet) is the current best held-out benchmark:
- **35.9% sensitivity @ 10 FA/24h** (AUROC 0.8654) on TUSZ eval
- SSOT: `results/local_fla_exp4_cyclic/eval_results_v2.json` (checkpoint: `results/local_fla_exp4_cyclic/checkpoints/best.pt`)

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
| 20 | 0.2577 | 154,040 | Oct 27 | Plateau holds (4th epoch) |
| 21 | 0.2577 | 161,742 | Oct 28 | No change; patience 9/20 |
| 22 | 0.2577 | 169,444 | Oct 29 | Plateau continues (6th epoch) |
| 23 | 0.2577 | 177,146 | Oct 29 | Plateau continues (7th epoch) |
| 24 | 0.2577 | 184,848 | Oct 30 | Plateau continues (8th epoch) |
| 25 | 0.2577 | 192,550 | Oct 30 | Plateau continues (9th epoch) |
| 26 | 0.2577 | 200,252 | Oct 31 | Plateau continues (10th epoch) |
| 27 | 0.2577 | 207,954 | Oct 31 | Plateau continues (11th epoch) |
| 28 | 0.2577 | 215,656 | Nov 01 | Plateau continues (12th epoch) |
| 29 | 0.2577 | 223,358 | Nov 01 | Plateau continues (13th epoch); patience 13/20 |

**Best Overall**: 0.2801 @ Epoch 9 (Oct 16)
**Post-Resume Best**: 0.2577 @ Epochs 17-29 (Oct 25 - Nov 01) - Stuck in plateau for 13 epochs

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

- **Extended Plateau (Epochs 17-29)**: 📊 **0.2577 (stuck for 13 epochs!)**
  - Model settled at new plateau after recovery
  - No improvement for 13 consecutive epochs (Oct 25 → Nov 01)
  - Patience counter: 13/20 (will early stop at epoch 36 if no improvement)
  - LR decayed to ~8.3e-05 (30% through cosine schedule)

### Experiment Context
- **Exp1 (Regularization)**: Cancelled @ epoch 9, peaked at 0.2726 (worse than baseline)
- **Conclusion**: Model NOT overfitting, regularization harmful
- **Current Strategy**: Continue baseline to epoch 20-30, evaluate Exp2 (lower LR) at epoch 17

---

## Next Milestones

- ✅ **Epoch 14** (Oct 24): Plateau continued at 0.2493
- ✅ **Epoch 15** (Oct 24): Plateau continued at 0.2493
- ✅ **Epoch 17** (Oct 25): Recovery to 0.2577 confirmed
- ✅ **Epochs 20-29** (Oct 27 - Nov 01): Plateau extended to 13 epochs at 0.2577
- 🔄 **Epoch 30** (Nov 01, in progress): Min epochs reached, patience 13/20
- **Epoch 36** (~Nov 3-4, projected at the time): Expected early stop (patience exhausted)

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

**Snapshot Last Updated**: Nov 01, 2025 20:45 EDT (Epoch 30 in progress at the time; plateau at 0.2577 for 13 consecutive epochs (17-29))
