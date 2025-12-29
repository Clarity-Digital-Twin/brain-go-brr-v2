# Exp4 Monitoring Guide

## Current Status

- ✅ **Training COMPLETE**: 78 epochs (early stopped), best epoch 63
- Output dir: `results/local_fla_exp4_cyclic/`
- Checkpoint: `results/local_fla_exp4_cyclic/checkpoints/best.pt`
- Eval SSOT: `results/local_fla_exp4_cyclic/eval_results_v2.json`

## Quick Checks

### 1. Confirm Best Checkpoint Exists
```bash
ls -la results/local_fla_exp4_cyclic/checkpoints/best.pt
```

### 2. Inspect Training Summary (Early Stop + Best Epoch)
```bash
rg -n \"Early stopping at epoch|Training complete\\. Best epoch\" results/local_fla_exp4_cyclic/training_resumed.log
```

### 3. Inspect Eval Summary (AUROC + Sensitivity@FA)
```bash
rg -n \"AUROC:|\\[FA\\]\" results/local_fla_exp4_cyclic/eval_v2.log
```

---
Last Updated: 2025-12-20
