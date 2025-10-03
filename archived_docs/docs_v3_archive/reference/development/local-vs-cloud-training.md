# Local vs Cloud Training Configuration Differences

**Date**: 2025-09-30
**Purpose**: Document intentional differences between RTX 4090 and A100 training configs
**Status**: These differences are CORRECT and INTENTIONAL

## Key Differences

| Setting | Local (RTX 4090) | Cloud (A100) | Why Different? |
|---------|------------------|--------------|----------------|
| **Gradient Clip** | 0.1 | 0.5 | RTX 4090 needs tighter clip for stability |
| **Batch Size** | 12 | 64 | Limited by VRAM (24GB vs 80GB) |
| **Mixed Precision** | false | true | RTX 4090 has AMP NaN issues, A100 is stable |
| **Learning Rate** | 1e-4 | 3e-5 | Adjusted for batch size difference |
| **Training Time** | ~200 hours | ~100 hours | Different hardware speed + batch size |
| **Cost** | Free (local GPU) | ~$319 (Modal) | Resource optimization |

## Expected Outcomes

### ✅ What Should Be Similar
- **TAES**: Within ±0.05 (e.g., 0.85 vs 0.84)
- **AUROC**: Within ±0.02 (e.g., 0.92 vs 0.91)
- **Sensitivity @ 1FA/24h**: Within ±5% (e.g., 75% vs 71%)
- **Convergence**: Both should reach clinical targets (<1 FA/24h)

### ❌ What Will Be Different
- **Exact weight values**: Completely different (this is normal!)
- **Training trajectory**: Different loss curves (local more conservative)
- **Convergence speed**: A100 ~2x faster due to larger batches + looser clip
- **Checkpoint sizes**: Same (~125MB), but non-interchangeable

## Why Weights Are Different (And Why That's OK)

**Neural networks have infinite equivalent solutions:**

```
Think of it like cooking:
- Local training: "Stir gently, low heat" → Final dish A
- Cloud training: "Stir vigorously, high heat" → Final dish B
- Both dishes taste great, but ingredients mixed differently
```

**Mathematical explanation:**
- Loss landscape has many local minima with similar performance
- Different optimization paths (due to clipping) reach different minima
- Both minima can have excellent clinical performance

**Proof that this is normal:**
- Even same config with different random seeds produces different weights
- Published papers report mean±std across 3-5 runs (acknowledging variance)
- What matters is generalization, not exact weight values

## Gradient Clipping Impact

**What gradient clipping does:**
```python
# Before clipping (can explode)
grad = [100.0, -250.0, 89.0]  # Huge values!
optimizer.step()  # Model explodes → NaN

# After clipping to 0.1
grad = [0.08, -0.09, 0.07]  # Scaled down
optimizer.step()  # Model stable ✅
```

**Effect on training:**
- **0.1 clip**: Smaller steps, more stable, slower convergence
- **0.5 clip**: Larger steps, faster convergence, needs monitoring
- Both prevent explosion, just different "speed limits"

## Monitoring for Equivalence

**After both trainings complete, compare:**

### Quantitative Metrics
```bash
# Extract final metrics from both runs
Local:  TAES=?, AUROC=?, Sens@1FA=?
Cloud:  TAES=?, AUROC=?, Sens@1FA=?

# Acceptable if all within tolerance:
|local_taes - cloud_taes| < 0.05 ✅
|local_auroc - cloud_auroc| < 0.02 ✅
|local_sens - cloud_sens| < 0.05 ✅
```

### Qualitative Checks
1. **Loss curves**: Both should decrease smoothly (different rates OK)
2. **Validation trends**: Both should show improving AUROC
3. **Final performance**: Both should meet clinical targets
4. **Gradient stability**: Both should show decreasing grad norm messages

### Red Flags (Investigate If Seen)
- 🚨 One training collapses (all-negative predictions)
- 🚨 Metric difference >10% (e.g., TAES 0.85 vs 0.70)
- 🚨 One training shows increasing grad norms after batch 500
- 🚨 Persistent NaN warnings (not just early training clips)

## Unifying Configs (Not Recommended)

**If you wanted identical configs (for science, not production):**

### Option 1: Match to Local (Conservative)
```yaml
# Both use RTX 4090 settings
gradient_clip: 0.1
batch_size: 12
mixed_precision: false

Impact:
- A100 trains slower (~2x time)
- Wastes Modal money ($600 instead of $300)
- Metrics should be closer (within ±0.01)
```

### Option 2: Match to Cloud (Risky)
```yaml
# Both use A100 settings
gradient_clip: 0.5
batch_size: 64  # OOM on RTX 4090!
mixed_precision: true  # NaN on RTX 4090!

Impact:
- Local training crashes (VRAM overflow or NaN)
- Not viable
```

**Conclusion**: Current configs are optimal for each hardware. Don't change.

## Post-Training Comparison Plan

**After both complete (~7 days):**

1. **Extract final metrics**:
   ```bash
   # Local
   grep "best_taes\|best_auroc\|best_sensitivity" /tmp/local-train-*.log

   # Cloud
   modal app logs <app-id> | grep "best_taes\|best_auroc\|best_sensitivity"
   ```

2. **Compare checkpoints**:
   ```bash
   # Checkpoint sizes should be similar
   ls -lh checkpoints/local/best.pt
   ls -lh checkpoints/cloud/best.pt

   # Weight distributions (should be different!)
   python scripts/compare_weights.py local/best.pt cloud/best.pt
   ```

3. **Inference comparison**:
   ```bash
   # Run both models on dev set
   python -m src evaluate --checkpoint local/best.pt --data dev
   python -m src evaluate --checkpoint cloud/best.pt --data dev

   # Compare predictions (should agree ~95% of time)
   ```

4. **Document findings**:
   - Add results to this file
   - Update CHANGELOG.md
   - Decide which checkpoint to use for production (likely cloud due to cost)

## References

- Gradient behavior: `docs/10-final-refactor-NAN/GRADIENT_BEHAVIOR_GUIDE.md`
- NaN protection: `docs/10-final-refactor-NAN/NAN_CANONICAL.md`
- Local config: `configs/local/train.yaml`
- Cloud config: `configs/modal/train.yaml`

## Decision Log

**2025-09-30**: Confirmed different clipping values are correct
- Local: 0.1 (RTX 4090 stability requirement)
- Cloud: 0.5 (A100 speed optimization)
- Both configs validated via smoke tests
- Expect similar performance, different weights
- Will compare after both trainings complete

---

**TL;DR**: Different configs are intentional and correct. Weights will differ, but performance should be similar. Don't unify unless doing controlled experiment (wastes money/time).