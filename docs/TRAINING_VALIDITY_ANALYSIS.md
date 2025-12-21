# Training Validity Analysis: Is Our Training Valid?

## TL;DR: **YES - Exp4 Training is VALID (and COMPLETE) ✅**

---

## The Bug

**Root Cause**: Eigenvalue degeneracy in `torch.linalg.eigh` → cuSOLVER crash
**Nature**: **Crash condition** (not a correctness bug)
- When eigenvalues degenerate → CUDA crashes (hard failure)
- When eigenvalues NOT degenerate → Code runs correctly
- **The jitter PREVENTS the crash, doesn't fix incorrect behavior**

---

## What Actually Happened

### Baseline FLA (crashed)
- **Config**: `configs/local/train_fla.yaml` (cosine scheduler)
- **Crashed**: October 16, 2025 during **validation** at epoch ~13
- **Location**: Validation batch 8886/18528
- **Error**: "CUDA error: unknown error" (cuSOLVER degeneracy)
- **Epochs 0-13**: ✅ Valid training data
- **Checkpoints**: Exist for epoch 13 (Oct 18)

### Exp4 SGDR ✅ (completed)
- **Config**: `configs/local/train_fla_exp4_cyclic.yaml` (SGDR)
- **Completed**: 78 epochs (early stopped), best epoch 63
- **Best dev metric**: 0.2904 @ 10 FA/24h (during training validation)
- **Held-out TUSZ eval**: 35.9% sensitivity @ 10 FA/24h (AUROC 0.8654)
- **SSOT**: `results/local_fla_exp4_cyclic/eval_results_v2.json`
- **Eigendecomp warnings**: ZERO ❌ (searched all logs)
- **Status**: **FULLY VALID** - training + evaluation completed cleanly

---

## DeepMind Decision Matrix

| Question | Answer | Implication |
|----------|--------|-------------|
| Did Exp4 hit the crash condition? | ❌ No | Training ran correctly |
| Does Exp4 have eigendecomp warnings? | ❌ None | No numerical instability detected |
| Are Exp4's metrics stable? | ✅ Yes | Training ran to early stop without crash |
| Can we trust Exp4's checkpoints? | ✅ Yes | No code bugs, only avoided crash |
| Should we restart Exp4 from scratch? | ❌ No | Already complete; reruns only for ablations |

---

## What Would DeepMind Do?

Based on AlphaFold/Gemini development practices:

### ✅ **Proceed with Post-Training Validation**
1. Keep the jitter protection enabled for future runs
2. Run official NEDC scoring for Exp4 outputs (OVERLAP/TAES/SzCORE) for publication-ready reporting
3. Treat the baseline crash as an operational stability issue (not a correctness invalidation of Exp4)

---

## Validation Checklist

Post-completion checklist:
- [x] Fix merged to `development` ✅
- [x] Tests pass (595/595) ✅
- [x] Exp4 logs clean (no eigendecomp warnings) ✅
- [x] Exp4 checkpoint valid (`results/local_fla_exp4_cyclic/checkpoints/best.pt`) ✅
- [x] Exp4 training complete (early stopped at epoch 78) ✅
- [x] Held-out eval complete (SSOT JSON written) ✅

---

## Bottom Line

Exp4 training is valid and complete; the eigendecomp issue was a crash condition, not a correctness bug. The next validation step is producing publication-ready NEDC scorer outputs for the Exp4 checkpoint.
