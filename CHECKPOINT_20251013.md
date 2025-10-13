# CHECKPOINT: October 13, 2025 19:47 EDT

## Priority: Research Realistic Performance Targets

### Current Issue

**README.md claims** (lines 67-79):
```
| False Alarm Rate | Sensitivity Target | Clinical Reality |
|------------------|-------------------|-----------------|
| 10 FA/24h | >95% | Initial deployment - high alarm fatigue |
| 5 FA/24h  | >90% | Standard care - manageable workload |
| 1 FA/24h  | >75% | Gold standard - sustained clinical use |
```

**Problem**: These targets appear unrealistic based on published competition results.

### Evidence from Literature

**Neureka 2020 Competition Results** ([picone-model-benchmarks.md](literature/markdown/picone-model-benchmarks/picone-model-benchmarks.md)):

**Best performing system (`sia`)** using TAES metric:
- ~1 FA/24h @ **11.37% sensitivity** ❌ (NOT 75%)

**Baseline system (`nedc`)** using TAES metric:
- ~17 FA/24h @ **35.54% sensitivity**

**Other competition models** (dev set):
- `pnc98`: ~14 FA/24h @ 20-26% sensitivity
- `yff`: ~14 FA/24h @ 14-20% sensitivity
- `lzk`: Various performance levels

**Reality Check**:
- State-of-the-art (2020): **1 FA/24h @ ~11% sensitivity**
- Our claimed target: **1 FA/24h @ >75% sensitivity**
- Gap: **~64 percentage points** 🚨

### SeizureTransformer Results

Check: [literature/markdown/seizure_transformer/SeizureTransformer.md](literature/markdown/seizure_transformer/SeizureTransformer.md)

Review Figure 4 images to understand:
- Actual sensitivity/FA tradeoff curves
- Where models operate on the Pareto frontier
- What "good" performance actually looks like

### Questions to Answer

1. **What are realistic targets for TUSZ?**
   - Based on competition results (Neureka 2020)
   - Based on published papers (SeizureTransformer, EvoBrain, EEG-Mamba)
   - What's the current SOTA (state of the art)?

2. **What metric should we use?**
   - TAES (Time-Aligned Event Scoring) - used in Neureka competition
   - OVLP (Any-Overlap) - popular in EEG community
   - EPCH (Epoch-based) - biased toward long events
   - Our README mentions TAES but doesn't specify for targets

3. **What's achievable with our architecture?**
   - Can we beat Neureka 2020 best (1 FA/24h @ 11% sensitivity)?
   - What would be a "win"?
     - Option A: Match SOTA (1 FA/24h @ 15-20% sensitivity)?
     - Option B: Push Pareto frontier (e.g., 5 FA/24h @ 40-50% sensitivity)?
     - Option C: Something else entirely?

4. **Should we adjust clinical deployment claims?**
   - Current: "gold standard - sustained clinical use" at 1 FA/24h @ 75%
   - Reality: No published model achieves this
   - Alternative: Frame as research milestone? "Towards clinical deployment"?

### Action Items

Before training completes (Epoch 20 early stop or Epoch 100 full):

1. **Research Phase** (2-4 hours):
   - [ ] Read SeizureTransformer paper + analyze Figure 4 DET curves
   - [ ] Review EvoBrain paper performance claims (95% AUROC ≠ 95% sensitivity!)
   - [ ] Survey EEG-Mamba paper results
   - [ ] Check latest TUSZ leaderboard (if exists)
   - [ ] Document realistic targets in dedicated file

2. **Update Documentation**:
   - [ ] Create `PERFORMANCE_TARGETS.md` with realistic expectations
   - [ ] Update README.md performance table with achievable targets
   - [ ] Add "stretch goal" vs "realistic goal" framing
   - [ ] Clarify metric (TAES vs OVLP vs EPCH)

3. **Set Evaluation Criteria**:
   - [ ] Define "success" for FLA training
   - [ ] Define when to resume BiMamba2 training
   - [ ] Decide early stopping criteria (Epoch 20 vs 100)

### Why This Matters

**Current state**:
- README overpromises (1 FA/24h @ 75% sensitivity)
- No model has achieved this on TUSZ
- Sets unrealistic expectations

**Risk**:
- Train for 60 days (Epoch 100)
- Achieve 1 FA/24h @ 20% sensitivity
- Think we "failed" when actually we matched/beat SOTA

**Fix**:
- Research realistic targets NOW
- Set achievable goals
- Celebrate real wins (beating Neureka 2020 competition)

### Current Training Status

- **FLA (Local RTX 4090)**: Epoch 2, batch 7323/7702 (95% through), progressing normally
- **BiMamba2 (Modal)**: PAUSED at Epoch 6, $1.1k spent, backed up
- **Plan**: Early stop FLA at Epoch 20 (~20 days), evaluate, decide next steps

### Next Steps

1. **Immediate** (while training continues):
   - Research realistic performance targets
   - Read SeizureTransformer + competition papers
   - Draft `PERFORMANCE_TARGETS.md`

2. **After Epoch 20** (~20 days from now):
   - Evaluate FLA performance vs realistic benchmarks
   - Decide: Continue to Epoch 100? Stop? Resume BiMamba2?

3. **Documentation**:
   - Update README.md with honest, achievable targets
   - Frame research contribution accurately

---

**Created**: 2025-10-13 19:47 EDT
**Author**: Independent research, budget-conscious decision-making
**Status**: ACTIVE - research needed before training completes
