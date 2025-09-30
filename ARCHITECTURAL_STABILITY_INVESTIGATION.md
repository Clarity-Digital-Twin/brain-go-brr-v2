# V3 Architectural Stability: Deep Investigation & Decision Framework

**Date**: September 30, 2025
**Status**: 🔬 **ACTIVE INVESTIGATION** - Batch 88+ monitoring
**Context**: PR1+2+3 enabled, gradient spikes observed, competing hypotheses

---

## Executive Summary

Two competing analyses of current gradient behavior with PR1+2+3 enabled:

**Hypothesis A (Premature Concern)**: "PR1+2+3 insufficient, need PR-6 through PR-10 immediately"
**Hypothesis B (Normal Early Training)**: "Protection stack working, wait for 100-200 batches before intervening"

**Current Consensus**: **Hypothesis B is more likely correct**, but we document both perspectives and define clear decision criteria.

---

## Part 1: Current State (Batch 88, 15:41 UTC)

### Configuration Verified ✅
```yaml
# configs/local/train.yaml confirmed active
model:
  norms:
    boundary_norm: layernorm         # PR-1 ✅
    after_tcn_proj: true
    after_node_mamba: true
    after_edge_mamba: true
    after_gnn: true
    before_decoder: true
  graph:
    edge_lift_activation: tanh       # PR-2 ✅
    edge_lift_norm: layernorm
    adj_row_softmax: true            # PR-3 ✅
    adj_softmax_tau: 1.0
    adj_ema_beta: 0.95
    adj_force_symmetric: true
```

### Observed Behavior
```
Batch 7:  grad_norm = 1.82  (clipped to 0.1)
Batch 24: grad_norm = 5.31  (clipped to 0.1) ← Spike
Batch 52: grad_norm = 2.00  (clipped to 0.1)
Batch 86: grad_norm = 4.16  (clipped to 0.1) ← Spike
Batch 88: grad_norm = 1.06  (clipped to 0.1)

Range: 1.0 - 5.31
Frequency: ~60% of batches show "Large grad norm" message
No NaN losses: ✅ Zero occurrences
```

### Parameter Count Confirms PR-1 Active
```
Baseline: 31,473,802 parameters
Current:  31,475,754 parameters (+1,952)
Delta matches LayerNorm additions from PR-1
```

---

## Part 2: Hypothesis A - "PR1+2+3 Insufficient"

### Core Claims
1. **TCN lacks weight normalization** (2025 literature: "required for large receptive fields")
2. **BiMamba lacks pre-norm pattern** (Reference Mamba uses pre-norm, not post-norm)
3. **Gradient clip too aggressive** (0.1 is very tight, 90%+ batches trigger)
4. **LR warmup too short** (154 batches = 10 minutes, SSMs need 1000+ batches)
5. **Dynamic PE too noisy** (eigendecomposition 960x per forward pass)

### Proposed Fixes
- **PR-6**: Add `torch.nn.utils.weight_norm` to TCN conv layers
- **PR-7**: Add RMSNorm BEFORE Mamba2 blocks (pre-norm pattern)
- **PR-8**: Adaptive gradient clipping (1.0 → 0.3 over 5000 batches)
- **PR-9**: Extend warmup to 1000 batches
- **~~PR-10~~**: ❌ **REJECTED** - Semi-dynamic PE compromises ML stack (we're EvoBrain-level BiMamba, not hacky workarounds)

### Supporting Evidence
- **2025 Literature**: TTSNet (Jan 2025) explicitly requires weight norm for TCN
- **Reference Mamba**: Official implementation uses pre-norm (norm before Mamba)
- **Grad norm trend**: Batch 10 (1.11-2.05) vs Batch 82 (1.15-2.23) - no improvement
- **Spike at batch 86**: 4.16 is HIGHER than earlier spikes

### Weaknesses of Hypothesis A
- ⚠️ **Sample size too small**: 88 batches << 15,404 batches/epoch
- ⚠️ **Ignores gradient guide**: Official docs say this is normal for early training
- ⚠️ **No NaN losses**: System IS working as designed
- ⚠️ **Tight clipping**: 0.1 threshold means anything >1.0 gets flagged

---

## Part 3: Hypothesis B - "Normal Early Training"

### Core Claims
1. **Gradient guide explicitly says this is normal** (docs/10-major-NAN-refactor/GRADIENT_BEHAVIOR_GUIDE.md:9-31)
2. **Protection stack is working** (sanitization + clipping, zero NaN losses)
3. **50-100 batches too early to judge** (need 100-200 batches minimum)
4. **Dynamic PE noisiest component** (eigendecomp gradients swing during adjacency learning)
5. **BGB_SANITIZE_GRADS=1 is REQUIRED** (not a workaround, but operating posture)

### From Official Documentation
```markdown
## Understanding "Large Grad Norm" Messages

### What It Means

**This is EXPECTED in early training!**

1. **Detected**: Gradient norm (2.52) exceeds threshold (0.1)
2. **Clipped**: Automatically scaled down to 0.1
3. **Safe**: Training continues without NaN
4. **Temporary**: Frequency decreases as model learns

### Why It Happens

**Early Training (batches 0-100):**
- Random weight initialization → large errors
- Large errors → large gradients
- Gradient clipping prevents explosion
- **Frequency**: Every few batches (normal)
```

### Supporting Evidence
- **Zero NaN losses**: After 88 batches, no single NaN loss reported
- **Loss decreasing**: 0.0862 (batch 18) → 0.0839 (batch 73) - training progressing
- **Clipping working**: Every large grad norm gets clipped safely
- **Expected behavior**: Architectural instability doc predicted "unbounded flow" during early training

### Diagnostic Recommendations
1. **Monitor through batch 200**: Trend should become clear
2. **Log per-module norms**: Identify which component is spiking
3. **Test without dynamic PE**: `use_dynamic_pe: false` to isolate eigendecomp
4. **Watch for NaN gradient sanitization**: If this starts appearing, it's a problem

### Weaknesses of Hypothesis B
- ⚠️ **Batch 86 spike to 4.16**: Higher than earlier spikes, concerning trend
- ⚠️ **No obvious downward trend**: Range still 1.0-4.0+ after 88 batches
- ⚠️ **High clipping frequency**: 60% of batches is aggressive

---

## Part 4: Literature Review Summary

### TCN Gradient Stability (2025 Research)
**Source**: "TTSNet: Transformer–Temporal Convolutional Network" (Jan 2025)

> "The major issue for very deep TCN networks with large receptive fields is exploding and/or vanishing gradients... Weight normalization is applied to every convolutional layer to normalize the input of hidden layers, which counteracts the exploding gradient problem."

**Our TCN**:
- 8 layers, exponential dilation (1, 2, 4, ..., 128)
- Receptive field: 15,360 samples (60 seconds)
- Residual connections: ✅ Yes
- Weight normalization: ❌ No
- Current approach: Conservative init (gain=0.2) + gradient clipping

### Mamba SSM Stability (2025 Research)
**Source**: "State Space Models Stabilization Techniques" (2025)

> "SSMs suffer from gradient explosion or vanishing when handling long sequences. Log(A) is tracked instead of directly optimizing matrix A for numerical stability. Mamba-2 includes an additional normalization layer for improved stability... The Mamba architecture adds the output from Selective SSM to the original input and then applies a normalization layer, which can be either Layer normalization or RMS normalization."

**Our BiMamba**:
- Current: POST-norm (norm after Mamba output via PR-1)
- Reference: PRE-norm (norm before Mamba input)
- Residual connections: ✅ Yes (with LayerScale)
- Internal RMSNorm: ❌ No (only boundary norms)

### GNN Laplacian PE Stability (2025 Research)
**Source**: "Understanding and Improving Laplacian Positional Encodings For Temporal GNNs" (June 2025)

> "Extending static Laplacian eigenvector approaches to temporal graphs through the supra-Laplacian poses key challenges including high eigendecomposition costs, limited theoretical understanding, and ambiguity... Minor perturbations to the Laplacian can produce substantially different eigenspaces."

**Our Dynamic PE**:
- Eigendecomposition: Every timestep (960x per forward pass)
- Conditioning: Row-softmax + EMA + symmetry (PR-3) ✅
- Regularization: ε=1e-3 on Laplacian ✅
- Issue: High compute + gradient noise during adjacency learning

---

## Part 5: Decision Framework

### Phase 1: Monitor (NOW - Batch 200)

**Action**: Continue current training, collect data

**Success Criteria** (by batch 200):
- ✅ Grad norm P95 < 2.5 (down from current ~4.0)
- ✅ Grad norm P50 < 1.5
- ✅ Clipping frequency < 40% (down from 60%)
- ✅ No NaN losses
- ✅ Loss continuing to decrease

**Failure Criteria** (triggers Phase 2):
- 🚨 Grad norm P95 > 3.0 or increasing trend
- 🚨 "Sanitized NaN gradients" messages appearing
- 🚨 Clipping frequency > 70%
- 🚨 Loss plateau or NaN

**Diagnostic Commands**:
```bash
# Extract grad norms and plot trend
grep "Large grad norm at batch" full_training.log | \
  awk '{print $9, $11}' | sed 's/://g' > grad_norms.txt

# Count clipping frequency
total=$(grep "PROGRESS\|HEARTBEAT" full_training.log | tail -1 | awk '{print $5}' | cut -d'/' -f1)
clipped=$(grep "Large grad norm" full_training.log | wc -l)
echo "Clipping rate: $clipped / $total = $(echo "scale=2; 100*$clipped/$total" | bc)%"

# Check for NaN gradient sanitization (BAD if appears)
grep "Sanitized NaN gradients" full_training.log
```

### Phase 2: Isolate Dynamic PE (IF Phase 1 fails)

**Action**: Test with dynamic PE disabled

**Commands**:
```bash
# Create test config
cp configs/local/train.yaml configs/local/train_no_dyn_pe.yaml
# Edit: set graph.use_dynamic_pe: false

# Run 100-batch test in tmux
tmux new -s no-dyn-pe
export BGB_SANITIZE_GRADS=1 BGB_NAN_DEBUG=1 BGB_LIMIT_FILES=50
.venv/bin/python -m src train configs/local/train_no_dyn_pe.yaml
# Detach: Ctrl+B D
```

**Interpretation**:
- If grad norms drop to <2.0 consistently → Dynamic PE is the culprit → Apply PR-10
- If grad norms still high → TCN or Mamba issue → Proceed to Phase 3

### Phase 3: Apply Surgical Fixes (IF Phase 2 isolates TCN/Mamba)

**CRITICAL CONSTRAINT**: ❌ **NO COMPROMISES TO ML STACK** - We are EvoBrain-level BiMamba, not quick hacks!

**PR-6: Weight Normalization on TCN**
```python
# In TCNEncoder.__init__ or _initialize_weights
if config.get("tcn_use_weight_norm", False):
    for m in self.tcn.modules():
        if isinstance(m, nn.Conv1d) and m.weight.requires_grad:
            m = torch.nn.utils.weight_norm(m)
```

**PR-7: Pre-Norm Pattern for BiMamba**
```python
# In BiMamba2Layer.__init__
if config.get("mamba_use_pre_norm", False):
    self.pre_norm_fwd = RMSNorm(d_model)
    self.pre_norm_bwd = RMSNorm(d_model)

# In forward()
if self.pre_norm_fwd:
    x_fwd = self.pre_norm_fwd(x)
    x_bwd = self.pre_norm_bwd(x)
else:
    x_fwd = x_bwd = x
```

**PR-8: Adaptive Gradient Clipping**
```python
# In train/loop.py
def get_grad_clip_value(global_step: int, config: dict) -> float:
    if not config.get("gradient_clip_schedule", {}).get("enabled", False):
        return config["gradient_clip"]  # Fixed value

    warmup_steps = config["gradient_clip_schedule"]["warmup_steps"]
    warmup_value = config["gradient_clip_schedule"]["warmup_value"]
    final_value = config["gradient_clip_schedule"]["final_value"]
    rampdown_steps = config["gradient_clip_schedule"]["rampdown_steps"]

    if global_step < warmup_steps:
        return warmup_value
    elif global_step < warmup_steps + rampdown_steps:
        t = (global_step - warmup_steps) / rampdown_steps
        return warmup_value - t * (warmup_value - final_value)
    else:
        return final_value
```

**PR-9: Extended Warmup**
```yaml
# configs/local/train.yaml
training:
  warmup_steps: 1000  # Up from 154 (warmup_ratio: 0.01)
```

**~~PR-10~~: Semi-Dynamic PE** ❌ **REJECTED**
- **Why rejected**: Compromises the ML architecture - Dynamic Laplacian PE is a core feature, not optional
- **The real solution**: Fix the eigendecomposition stability properly with PR-3 adjacency conditioning (already enabled!)
- **No hacky workarounds**: We get it working 100% robustly, not 80% with shortcuts

### Phase 4: Validate Fixes

**After each fix**:
- Run 100-batch test
- Compare grad norm distribution
- Proceed to next fix only if improvement confirmed

---

## Part 6: Reconciliation of Hypotheses

### What Both Agree On
1. ✅ Dynamic PE eigendecomposition is noisy
2. ✅ Current stack (PR1+2+3) is fundamentally sound
3. ✅ Gradient clipping + sanitization is working as designed
4. ✅ No NaN losses = system is stable
5. ✅ 2025 literature provides valid improvement paths

### Key Disagreement
- **Hypothesis A**: "Need fixes NOW, 88 batches shows pattern"
- **Hypothesis B**: "Wait until batch 200, too early to judge"

### Resolution: Phased Approach
1. **Monitor first** (Hypothesis B) - collect more data
2. **Apply fixes if needed** (Hypothesis A) - have contingency ready
3. **Follow literature** (Both) - use proven techniques

---

## Part 7: Expected Behavior Comparison

### Baseline (Before PR1+2+3)
```
Grad norms: 1.5 - 3.0 range
NaN losses: Every 10-20 batches
Training: Unstable, required multiple interventions
Status: UNACCEPTABLE
```

### Current (PR1+2+3 Active)
```
Grad norms: 1.0 - 5.3 range (spikes to 5.3)
NaN losses: ZERO in 88 batches
Training: Stable, loss decreasing
Status: ACCEPTABLE but monitoring
```

### Target (After Stabilization)
```
Grad norms: < 1.0 P95, < 0.5 P50
NaN losses: Zero
Clipping: < 10% of batches
Status: OPTIMAL
```

### With PR-6 through PR-10 (If Needed)
```
Grad norms: < 1.0 P95, < 0.3 P50 (matches reference implementations)
NaN losses: Zero
Clipping: < 5% of batches, could use loose clip (1.0)
Status: BEST PRACTICE 2025
```

---

## Part 8: Critical Questions

### Q1: Is current behavior acceptable for production?
**Answer**: YES, with caveats
- Zero NaN losses after 88 batches ✅
- Loss decreasing steadily ✅
- Protection stack (clipping + sanitization) working ✅
- BUT: High clipping frequency (60%) indicates room for improvement

### Q2: Should we intervene now or wait?
**Answer**: WAIT until batch 200, but prepare fixes
- 88 batches is too small sample (0.6% of one epoch)
- Gradient guide explicitly says 50-100 batches is normal for spikes
- Have PR-6 through PR-10 ready as contingency

### Q3: Which fix should we apply first if needed?
**Answer**: PR-6 (TCN weight norm) - most literature-supported
1. ❌ **NOT semi-dynamic PE** - that compromises the ML stack
2. Test with `use_dynamic_pe: false` ONLY for isolation/diagnosis
3. If Dynamic PE is noisy, the fix is BETTER conditioning (PR-3 already enabled!), NOT disabling it
4. If TCN is culprit, apply PR-6 (weight norm) - proven technique from 2025 literature

### Q4: Is the external advice (Hypothesis B) correct?
**Answer**: MOSTLY correct, with nuances
- ✅ Correct: Current behavior is within expected early training range
- ✅ Correct: Protection stack is working as designed
- ✅ Correct: 50-100 batches too early to judge
- ⚠️ Nuance: Batch 86 spike to 4.16 is concerning if trend continues
- ⚠️ Nuance: Literature supports additional improvements

### Q5: Is my analysis (Hypothesis A) wrong?
**Answer**: TOO HASTY, but technically sound
- ❌ Wrong: "Pattern isn't changing" - 88 batches is too few
- ❌ Wrong: "Act now" - premature intervention
- ✅ Correct: Literature review and fix proposals are valid
- ✅ Correct: TCN weight norm and Mamba pre-norm are best practices
- ✅ Correct: Dynamic PE is noisy (confirmed by both hypotheses)

---

## Part 9: Monitoring Checklist

### Every 50 Batches
- [ ] Check latest grad norm (tail -50 full_training.log | grep "grad norm")
- [ ] Verify loss decreasing (grep "PROGRESS" full_training.log | tail -5)
- [ ] Count NaN sanitization events (grep "Sanitized NaN" full_training.log | wc -l)
- [ ] Check for non-finite logits (grep "Non-finite logits" full_training.log)

### At Batch 100
- [ ] Calculate grad norm statistics (P50, P95, max)
- [ ] Plot grad norm trend (is it decreasing?)
- [ ] Decide: Continue monitoring OR proceed to Phase 2

### At Batch 200
- [ ] Final decision: Is current stack sufficient?
- [ ] If yes: Continue training, document success
- [ ] If no: Execute Phase 2 (isolate Dynamic PE)

### At Batch 500
- [ ] Validate long-term stability
- [ ] Compare with baseline (pre-PR1+2+3)
- [ ] Update architectural docs with findings

---

## Part 10: Success Metrics by Phase

### Phase 1 Success (By Batch 200)
- Grad norm P95 < 2.5
- Grad norm showing downward trend
- Clipping frequency < 50%
- Zero NaN losses

→ **Action**: Continue current training, document success

### Phase 2 Success (After PE isolation test)
- With `use_dynamic_pe: false`, grad norms drop to <2.0
- Identifies Dynamic PE as primary culprit

→ **Action**: Apply PR-10 (semi-dynamic PE), test 100 batches

### Phase 3 Success (After surgical fixes)
- PR-6 (TCN weight norm): Grad norm P95 drops by 30%
- PR-7 (Mamba pre-norm): Grad norm P95 drops by another 20%
- Combined: Grad norm P95 < 1.0

→ **Action**: Deploy to full training, update docs

### Phase 4 Success (Long-term validation)
- 5000+ batches with stable grad norms
- Model converges to target metrics
- No NaN losses throughout training

→ **Action**: Merge improvements, update CLAUDE.md, publish findings

---

## Part 11: Lessons Learned

### What Worked
1. ✅ **Systematic literature review** - Found 2025 papers on exact issues
2. ✅ **Reference implementation comparison** - Identified pre-norm gap
3. ✅ **External feedback integration** - Balanced premature intervention
4. ✅ **Phased approach** - Monitor → Isolate → Fix → Validate

### What Didn't Work
1. ❌ **Jumping to conclusions** - 88 batches too early to judge
2. ❌ **Ignoring official docs** - Gradient guide explicitly covered this
3. ❌ **Confirmation bias** - Looked for problems, found "problems"

### Critical Realizations
1. 💡 **Gradient clipping IS the design** - Not a band-aid, but required posture
2. 💡 **Early training is noisy** - Random init → large errors → large grads
3. 💡 **Dynamic PE is fundamentally noisy** - Eigendecomp during learning is hard
4. 💡 **Sample size matters** - 88/15404 = 0.6% of one epoch

---

## Part 12: Final Recommendations

### Immediate (Next 2 Hours)
1. **Continue monitoring** - Let training reach batch 200
2. **Collect data** - Save grad norm logs every 50 batches
3. **No interventions** - Do not stop training or change config

### Short-Term (Batch 200 Decision Point)
1. **If stable**: Document success, continue training
2. **If unstable**: Execute Phase 2 (isolate Dynamic PE)
3. **Update this doc**: Add batch 200 findings

### Medium-Term (After Isolation)
1. **If Dynamic PE culprit**: Apply PR-10 (semi-dynamic)
2. **If TCN/Mamba culprit**: Apply PR-6 (weight norm) then PR-7 (pre-norm)
3. **Validate each fix**: 100-batch test before next fix

### Long-Term (After Full Stack Stable)
1. **Document learnings** - Update architecture docs
2. **Compare with Modal** - A100 behavior may differ
3. **Publish findings** - Contribute to 2025 literature

---

## Conclusion

**Current Status**: System is STABLE but not OPTIMAL

**Hypothesis Resolution**: Hypothesis B (wait and monitor) is MORE CORRECT than Hypothesis A (intervene now), but Hypothesis A provides valid contingency plans.

**Next Action**: Monitor through batch 200, then decide based on data, not speculation.

**Key Insight**: "Large grad norm" messages are INFORMATION, not ERRORS. The protection stack (clipping + sanitization) is working exactly as designed for PyTorch 2.5.0 + mamba-ssm 2.2.5 stack.

**Bottom Line**: We have both the patience (wait 200 batches) AND the tools (PR-6 through PR-10) to achieve optimal stability. No need to rush.

---

**This document is the SINGLE SOURCE OF TRUTH for V3 architectural stability investigation as of September 30, 2025.**

**Next Update**: After batch 200 (ETA: ~3 hours on RTX 4090)
