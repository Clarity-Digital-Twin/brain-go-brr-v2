# FLA Research Roadmap - Pragmatic Independent Researcher Strategy

**Date**: October 9, 2025
**Branch**: `feature/flash-linear-attention`
**Status**: 🎯 BiMamba2 baseline training LIVE, FLA infrastructure complete

---

## 🎯 **The Big Picture: Research Constraints & Strategy**

### **Reality Check: Why We're NOT Google DeepMind**

**Academic/Industry Approach** (ideal but impractical for us):
- ❌ Run ablation studies for each component (edge-only, node-only, GNN-only, LPE-only)
- ❌ Train 10+ model variants to isolate incremental improvements
- ❌ Use medium-scale validation (50-200 files) to compare each phase
- ❌ Statistical significance testing with multiple runs per config
- ❌ Cost: ~$5,000-10,000 in compute + 6-12 months

**Our Approach** (pragmatic, resource-constrained):
- ✅ Train **TWO** full stacks end-to-end: BiMamba2 baseline → FLA candidate
- ✅ Compare final performance on full dataset (100 epochs, ~4667 train files)
- ✅ Make deployment decision based on A/B comparison
- ✅ Cost: ~$650 in compute + 2-3 weeks
- ✅ Accept: We won't know which component contributes what, but we'll know if FLA is better overall

### **Why This Makes Sense**

1. **Historical Precedent**: V3 architecture evolved incrementally (TCN → BiMamba → GNN → LPE) without individual ablations
   - We didn't train "TCN-only" vs "TCN+BiMamba" vs "TCN+BiMamba+GNN"
   - We trained the **full stack** each time and iterated based on results
   - This worked because we had limited compute/time

2. **Seizure Detection is Hard**:
   - TUSZ benchmark: Top models achieve ~40-60% sensitivity@10FA (see literature/markdown/EEG-BIMAMBA)
   - 12:1 class imbalance (seizures are rare)
   - Long training times (100 epochs × 2-3 hours = 200-300 hours)
   - Incremental validation (50 files) is unreliable due to high variance

3. **Resource Constraints**:
   - No academic compute cluster
   - Limited budget (~$1,000 for this research phase)
   - Single researcher, not a team
   - Need results in weeks, not months

4. **Scientific Validity**:
   - A/B comparison on full dataset is still **scientifically valid**
   - Many published papers compare full architectures, not ablations
   - We can publish: "We compared BiMamba2 vs Gated DeltaNet on TUSZ seizure detection"
   - Missing ablations are a limitation we acknowledge, not a fatal flaw

---

## 📊 **Current Status (October 9, 2025)**

### **BiMamba2 Baseline (v3.9.0)** - TRAINING LIVE
```
Architecture: TCN + BiMamba2 + GNN + Dynamic LPE
Status: Modal A100-80GB, 100 epochs, ~4667 train files
Progress: ~10-20% complete (exact epoch TBD - check Modal logs)
ETA: 4-5 days (with resume cycles)
Expected baseline: sensitivity@10FA = 40-60% (TUSZ benchmark range)
```

### **FLA Research Stack** - INFRASTRUCTURE COMPLETE
```
Architecture: TCN + Gated DeltaNet + GNN + Dynamic LPE
Code: ✅ BiGatedDeltaNet wrapper complete
Config: ✅ All local configs exist
Tests: ✅ Smoke tests passed (Phase 0, 1a, 1b, 2)
Medium validation: ⚠️ Technical success (no crashes, no OOM), performance unstable (2.73% seizures)
Status: READY for full training after BiMamba2 baseline completes
```

---

## 🚀 **The Two-Stack Strategy**

### **Stack 1: BiMamba2 Baseline** (CURRENT)
```yaml
Temporal (Node): BiMamba2 (6 layers, d_model=512, d_state=16)
Temporal (Edge): BiMamba2 (2 layers, d_model=32, d_state=8)
Graph: GNN with Dynamic LPE (SSGConv, 2 layers, k=16 eigenvectors)
Frontend: TCN (8 layers, stride_down=16)
Backend: Gated fusion + decoder
```

**Training**:
- Platform: Modal A100-80GB
- Config: `configs/modal/train.yaml`
- Status: **RUNNING NOW**
- Cost: ~$319 (100 hours)
- Expected: sensitivity@10FA = 40-60%

---

### **Stack 2: FLA Candidate** (NEXT)
```yaml
Temporal (Node): Gated DeltaNet (6 layers, d_model=512, num_heads=6, headdim=8)
Temporal (Edge): Gated DeltaNet (2 layers, d_model=32, num_heads=3, headdim=8)
Graph: GNN with Dynamic LPE (SAME as BiMamba2)
Frontend: TCN (SAME as BiMamba2)
Backend: Gated fusion + decoder (SAME as BiMamba2)
```

**Training**:
- Platform: Modal A100-80GB (after BiMamba2 completes)
- Config: `configs/modal/phase2_both_gdn.yaml` (TO BE CREATED)
- Status: **BLOCKED** - waiting for BiMamba2 baseline
- Cost: ~$319 (100 hours)
- Expected: sensitivity@10FA = ? (hypothesis: ≥ BiMamba2 + 3%)

---

## 📋 **Execution Roadmap**

### ✅ **Phase 0: Infrastructure** (COMPLETE - Oct 7-8, 2025)
- [x] BiGatedDeltaNet wrapper (bf16-safe, bidirectional)
- [x] Config schema (`temporal_type_{edge,node}`, `gdn_*` params)
- [x] Builder factory pattern (edge_stream.py, node_stream.py)
- [x] FLA dependency (`flash-linear-attention==1.0.5`)
- [x] Constraint validation (0.75× product rule for heads)

**Evidence**: All code complete, quality checks pass

---

### ✅ **Phase 1a: Edge GDN Smoke Test** (COMPLETE - Oct 8, 2025)
- [x] Config: `configs/local/phase1a_edge_gdn.yaml`
- [x] Smoke test: 3 files, 10 epochs → PASSED (no crashes, no NaNs)
- [x] Bugfixes applied:
  - Selective bf16 conversion (FLA layers only)
  - Wired node override fields
  - Updated Doc 1 template (edge_d_model=32)

**Evidence**: `/tmp/phase1a_smoke_v4.log`

---

### ✅ **Phase 1b: Node GDN Smoke Test** (COMPLETE - Oct 8, 2025)
- [x] Config: `configs/local/phase1b_node_gdn.yaml`
- [x] Smoke test: 3 files, 10 epochs → PASSED (early stop epoch 7)
- [x] Result: No crashes, no NaNs, loss converged

**Evidence**: `/tmp/phase1b_smoke.log`

---

### ✅ **Phase 2: Both GDN Smoke + Medium Validation** (COMPLETE - Oct 8, 2025)

**Smoke Test** (3 files):
- [x] Config: `configs/local/phase2_both_gdn.yaml`
- [x] Result: PASSED (sensitivity@10FA = 1.0, early stop epoch 7)

**Medium Validation** (50 files, 6 epochs):
- [x] Config: `configs/local/phase2_medium_gdn.yaml`
- [x] Technical success: ✅ No crashes, no NaNs, no OOM (GPU: 17.5GB, RAM: 23.5GB)
- [x] Performance issue: ⚠️ Model collapsed at epoch 4 (sensitivity dropped to 10.53%)
- [x] Root cause: Only 99/3626 windows (2.73%) had seizures with `BGB_LIMIT_FILES=50`

**Evidence**: `/tmp/phase2_smoke.log`, `/tmp/phase2_medium.log`

**Conclusion**: Infrastructure validated ✅, but limited data caused instability

---

### 🔄 **Phase 3: BiMamba2 Baseline Training** (IN PROGRESS)
- [x] Config: `configs/modal/train.yaml`
- [x] Platform: Modal A100-80GB
- [x] Validation OOM fix deployed (disk-backed storage)
- [ ] **WAITING**: Training completion (~4-5 days)
- [ ] **DELIVERABLE**: Baseline sensitivity@10FA metric

**Evidence**: Modal app running, logs show disk-backed validation

---

### ⏳ **Phase 4: FLA Full Training** (NEXT - After BiMamba2)

**Step 1: Create Modal FLA Config** (~30 minutes)
```bash
# Copy local config to Modal
cp configs/local/phase2_both_gdn.yaml configs/modal/phase2_both_gdn.yaml

# Adjust Modal-specific params:
# - batch_size: 48 (A100-80GB can handle larger batches)
# - mixed_precision: true (A100 tensor cores)
# - num_workers: 4 (avoid overhead)
# - cache_dir: /results/cache/tusz_mmap (Modal persistent volume)

# Keep FLA params unchanged:
# - temporal_type: "gated_deltanet"
# - gdn_edge_num_heads: 3, gdn_edge_headdim: 8
# - edge_mamba_d_model: 32
```

**Step 2: Launch Modal Training**
```bash
# Deploy FLA stack (DETACHED - long run)
modal run --detach deploy/modal/app.py \
  --action train \
  --config configs/modal/phase2_both_gdn.yaml

# Monitor
modal app list
modal app logs <app-id>
```

**Step 3: Wait for Completion** (~4-5 days)

---

### 📊 **Phase 5: A/B Comparison & Decision** (FINAL)

**Compare Metrics**:
```
                        BiMamba2 Baseline    FLA Candidate    Delta
sensitivity@10FA        [TBD from Modal]     [TBD from Modal]  [?]
sensitivity@5FA         [TBD]                [TBD]             [?]
sensitivity@1FA         [TBD]                [TBD]             [?]
AUROC                   [TBD]                [TBD]             [?]
TAES                    [TBD]                [TBD]             [?]
Training time/epoch     ~1 hour              ~1 hour           [?]
```

**Decision Criteria**:
1. 🎯 **DEPLOY FLA**: sensitivity@10FA ≥ BiMamba2 + 3% (meaningful improvement)
2. ⚠️ **CONSIDER FLA**: sensitivity@10FA ≥ BiMamba2 + 1% (marginal improvement)
3. ✅ **KEEP BiMamba2**: sensitivity@10FA < BiMamba2 (FLA underperforms)

**Publication Strategy**:
- **If FLA wins**: "Gated DeltaNet improves EEG seizure detection by X% over BiMamba2 on TUSZ"
- **If BiMamba2 wins**: "BiMamba2 remains competitive with Gated DeltaNet for EEG tasks"
- **Either way**: We have a full-stack comparison on a real clinical benchmark

---

## 🚨 **Constraints & Limitations (Acknowledged)**

### **What We're NOT Doing** (and why it's okay)

1. ❌ **Individual component ablations** (edge-only, node-only, GNN-only)
   - Reason: Limited compute budget (~$1,000), need results in weeks not months
   - Impact: We won't know which component contributes what
   - Mitigation: Full-stack A/B comparison is still scientifically valid

2. ❌ **Medium-scale validation for each phase**
   - Reason: High variance with limited data (2.73% seizures at 50 files)
   - Impact: Can't validate incremental improvements reliably
   - Mitigation: Full dataset (4667 files) provides stable metrics

3. ❌ **Multiple training runs for statistical significance**
   - Reason: Each run costs ~$319 and takes 4-5 days
   - Impact: Can't measure variance across runs
   - Mitigation: Single-run comparison is standard in many papers

4. ❌ **Local full training**
   - Reason: ~200-300 hours (12-15 days) per run on RTX 4090
   - Impact: Slower iteration
   - Mitigation: Modal A100 is faster (~100 hours per run)

### **What We're Accepting**

1. ✅ **Limited data validation was unstable** (medium run collapsed)
   - This is expected: 2.73% seizure ratio is too sparse
   - Solution: Full dataset training only

2. ✅ **Seizure detection is hard** (TUSZ benchmark: 40-60% sensitivity@10FA)
   - Our expectations align with literature (see EEG-BIMAMBA paper)
   - We're comparing against ourselves (BiMamba2), not SOTA claims

3. ✅ **We evolved V3 without individual ablations**
   - Historical precedent: We trained full stacks (TCN → BiMamba → GNN → LPE)
   - This worked for us before, it will work again

---

## 📚 **Literature Context**

### **EEG-BIMAMBA Paper** (Anonymous, under review)
- Uses BiMamba for multi-task EEG (seizure, emotion, sleep, motor imagery)
- Achieves SOTA on multiple datasets
- Key insight: BiMamba balances accuracy, speed, and memory for long EEG
- Relevance: Validates our choice of BiMamba2 as baseline

### **Gated DeltaNet Paper** (Yang et al., ICLR 2025)
- Published at ICLR 2025 (peer-reviewed, not anonymous)
- Beats Mamba2 on language modeling, retrieval, long-context understanding
- Key insight: Gating (memory erasure) + delta rule (targeted updates) = complementary
- Relevance: FLA may improve over BiMamba2 for EEG sequences

### **TUSZ Benchmark** (Picone et al.)
- 12:1 class imbalance (seizures are rare)
- Top models: 40-60% sensitivity@10FA
- Training time: 100 epochs standard
- Relevance: Sets realistic expectations for our results

---

## 🎯 **Next Actions (Priority Order)**

### **IMMEDIATE** (Now)
1. ✅ Keep BiMamba2 Modal training running (don't interrupt)
2. ✅ Monitor validation logs for disk-backed storage (user is doing this)
3. ✅ Unified roadmap created (this document)

### **AFTER BiMamba2 COMPLETES** (~4-5 days from now)
1. 📊 Analyze BiMamba2 results:
   ```bash
   # Check W&B dashboard
   # Look for: sensitivity@10FA, @5FA, @1FA, AUROC, TAES
   # Document: Best epoch, final metrics, training stability
   ```

2. 📝 Create Modal FLA config:
   ```bash
   # Copy + adjust for Modal (batch_size, mixed_precision, etc.)
   # Validate with: make q
   # Git commit: "feat: Add Modal FLA config for A/B comparison"
   ```

3. 🚀 Launch FLA Modal training:
   ```bash
   modal run --detach deploy/modal/app.py \
     --action train \
     --config configs/modal/phase2_both_gdn.yaml
   ```

4. ⏳ Wait for FLA completion (~4-5 days)

5. 📊 A/B comparison & decision (deploy or rollback)

### **OPTIONAL** (If time/budget allows)
- 🧪 Local FLA experiment (full dataset, no `BGB_LIMIT_FILES`)
  - Pro: Validates on local hardware
  - Con: ~200-300 hours
  - Verdict: **SKIP** - Modal is faster

---

## 💡 **Key Insights**

### **Why This Roadmap Works**

1. **Pragmatic over perfect**: We're resource-constrained, not Google DeepMind
2. **Full-stack comparison**: Valid scientific approach, publishable results
3. **Historical success**: V3 evolved this way (incremental full-stack training)
4. **Clear decision criteria**: ≥3% improvement → deploy, <0% → rollback
5. **Acknowledged limitations**: No ablations, single run, limited validation

### **What Success Looks Like**

**Best case**: FLA beats BiMamba2 by ≥3% sensitivity@10FA
- Deploy FLA to production
- Publish: "Gated DeltaNet improves TUSZ seizure detection by X%"
- Tag: `v4.0.0-fla-production`

**Good case**: FLA beats BiMamba2 by 1-3%
- Consider deployment (marginal improvement)
- Document findings, iterate on hybrid architectures

**Acceptable case**: FLA performs similar to BiMamba2 (±1%)
- Keep BiMamba2 baseline (simpler, proven)
- Document: "Gated DeltaNet is competitive but not superior"

**Worst case**: FLA underperforms BiMamba2 by >1%
- Rollback to BiMamba2
- Document learnings, investigate why
- Possible next steps: GLA, HGRN2, or hybrid architectures

---

## 📊 **Timeline**

```
Oct 9, 2025:    BiMamba2 training LIVE (day 1 of 4-5)
Oct 12-13:      BiMamba2 completes, analyze results
Oct 13:         Create Modal FLA config (~30 min)
Oct 13:         Launch FLA Modal training (day 1 of 4-5)
Oct 17-18:      FLA completes, A/B comparison
Oct 18:         Decision: Deploy FLA or keep BiMamba2
Oct 19:         Merge to main, tag release, write postmortem
```

**Total time**: ~2-3 weeks from start to deployment decision
**Total cost**: ~$650 ($319 × 2 runs)

---

## 🎉 **Bottom Line**

**This roadmap is realistic, achievable, and scientifically valid.**

We're not doing ablations because we can't afford them. We're training two full stacks and comparing results. This is how many published papers work, and it's how we've successfully evolved V3 so far.

**The goal**: Know whether Gated DeltaNet is better than BiMamba2 for TUSZ seizure detection.
**The method**: A/B comparison on full dataset (4667 train files, 100 epochs).
**The decision**: Deploy if ≥3% improvement, rollback otherwise.
**The timeline**: 2-3 weeks.
**The cost**: ~$650.

**Let's ship it.** 🚀

---

**Document Status**: 🎯 ACTIVE ROADMAP
**Last Updated**: October 9, 2025
**Next Milestone**: BiMamba2 baseline completion (~4-5 days)
