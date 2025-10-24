# Baseline FLA Training Metrics

**Experiment**: Full FLA Gated DeltaNet baseline (31M params)
**Config**: `configs/local/train_fla.yaml`
**Status**: 🟢 RUNNING (Epoch 14 in progress)
**Started**: Oct 16, 2025 10:55 AM EDT
**Hardware**: RTX 4090 (24GB VRAM)
**Wandb Run**: https://wandb.ai/jj-vcmcswaggins-novamindnyc/seizure-v3-rtx4090/runs/5ee302c0a01d4e43b8e782fa2ffb0e90

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
| 15 | TBD | TBD | Oct 24 | Training started 09:24 |

**Best Overall**: 0.2801 @ Epoch 9 (Oct 16)

---

## Training Timeline

- **Oct 16, 10:55**: Training started
- **Oct 18, 13:47**: WSL2 suspend (mid-epoch 13, batch 7355)
- **Oct 23, 17:31**: Training resumed from mid-epoch checkpoint
- **Oct 23, 23:15**: Epoch 13 completed
- **Oct 24, 09:15**: Epoch 14 validation completing

---

## Key Observations

### Plateau Pattern (Epochs 9-12)
- Metric stuck at 0.2801 for 4 consecutive epochs
- Prompted investigation into overfitting hypothesis
- Led to Exp1 (regularization) experiment

### Post-Resume Performance Drop (Epochs 13-14)
- Sensitivity dropped to 0.2493 after resume (from 0.2801 peak)
- **-11% decline** from best (epoch 9)
- Epoch 14 confirms this is real, not just variance
- Both epochs 13-14 plateau at 0.2493
- Possible causes: resume disruption, scheduler state, natural variance
- Need to monitor epochs 15-17 to see if recovery occurs

### Experiment Context
- **Exp1 (Regularization)**: Cancelled @ epoch 9, peaked at 0.2726 (worse than baseline)
- **Conclusion**: Model NOT overfitting, regularization harmful
- **Current Strategy**: Continue baseline to epoch 20-30, evaluate Exp2 (lower LR) at epoch 17

---

## Next Milestones

- **Epoch 14** (~Oct 24, 09:15): Check if metric recovers from epoch 13 drop
- **Epoch 15** (~Oct 24, 19:00): Early improvement signal
- **Epoch 17** (~Oct 25, 17:00): **EXP2 DECISION POINT** - Continue baseline or start lower LR experiment?
- **Epoch 20** (~Oct 26, 15:00): Clear pattern confirmation
- **Epoch 30** (~Nov 2, 09:00): Min epochs reached, patience countdown begins

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

**Last Updated**: Oct 24, 2025 09:25 EDT (Epoch 14 complete, Epoch 15 training started)
