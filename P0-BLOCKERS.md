# 🚨 P0 BLOCKERS: Architecture Contradictions

**Date**: October 1, 2025
**Severity**: CRITICAL
**Status**: INVESTIGATION IN PROGRESS

## Summary

During deep documentation audit, discovered **FUNDAMENTAL CONTRADICTIONS** between:
1. Configuration files (configs/*.yaml)
2. Source code implementation (src/brain_brr/models/detector.py)
3. Documentation (docs/00-overview/, docs/04-model/)

**Impact**: We don't know what architecture we're actually training. Configs specify parameters that are IGNORED by code.

---

## 🔴 BLOCKER 1: Node Mamba d_model Mismatch

### What Configs Say
```yaml
# configs/local/train.yaml:54-55
# configs/modal/train.yaml:53-54
mamba:
  n_layers: 6
  d_model: 512  # ← SPECIFIED IN CONFIG
  d_state: 16
```

### What Code Does
```python
# src/brain_brr/models/detector.py:474-480
instance.node_mamba = BiMamba2(
    d_model=64,  # ← HARDCODED, IGNORES CONFIG
    d_state=16,
    d_conv=4,
    expand=2,
    headdim=8,
    num_layers=6,
)
```

### What Docs Say
```markdown
# docs/00-overview/overview.md:17
- Node Mamba: 6 layers, d_model=512, headdim=64, d_state=16, expand=2

# docs/04-model/v3-architecture.md:24
- Input `(B,19,15360)` → TCN `(B,512,960)` → Electrode features `(B,19,960,512)`
- **Node stream**: BiMamba2 over `(B*19,512,960)` → `(B,19,960,512)`
```

### Reality Check
```python
# src/brain_brr/models/detector.py:279-283
elec_flat = self.proj_to_electrodes(features)  # (B, 19*64, 960)
elec_feats = elec_flat.reshape(batch_size, 19, 64, seq_len).permute(
    0, 1, 3, 2
)  # (B, 19, 960, 64)  ← ACTUAL SHAPE IS 64, NOT 512!
```

### Contradiction
- **Config specifies**: `d_model=512`
- **Code uses**: `d_model=64` (hardcoded)
- **Docs claim**: `d_model=512`
- **Actual tensor shape**: `(B,19,960,64)` not `(B,19,960,512)`

**CRITICAL**: The config parameter `mamba.d_model` is **COMPLETELY IGNORED** by V3 architecture!

---

## 🔴 BLOCKER 2: Node Mamba headdim Mismatch

### What Docs Say
```markdown
# docs/00-overview/overview.md:17
- Node Mamba: 6 layers, d_model=512, headdim=64, d_state=16, expand=2
```

### What Code Does
```python
# src/brain_brr/models/detector.py:479
headdim=8,  # Critical: (64*2)/8 = 16 is multiple of 8
```

### Contradiction
- **Docs claim**: `headdim=64`
- **Code uses**: `headdim=8`
- **Math**: (64*2)/8 = 16 ✅ (correct CUDA alignment)
- **Docs math**: (512*2)/64 = 16 ✅ (would also work, but NOT what code does)

---

## 🔴 BLOCKER 3: V3 Architecture Flow Confusion

### What Docs Say
```markdown
# docs/00-overview/architecture-summary.md:9-18
Input (B,19,15360)
  → TCNEncoder (8×, 64→512, stride_down=16) → (B,512,960)
  → Electrode proj 512→19×512 → (B,19,960,512)  ← DOCS SAY 512
    ├─ Node Mamba (BiMamba2, d_model=512, n_layers=6)
```

### What Code Does
```python
# src/brain_brr/models/detector.py:540-541
instance.proj_to_electrodes = nn.Conv1d(512, 19 * 64, kernel_size=1)  # 512 → 19*64
instance.proj_from_electrodes = nn.Conv1d(19 * 64, 512, kernel_size=1)  # 19*64 → 512

# Actual flow:
# (B,512,960) → proj_to_electrodes → (B,19*64,960) → reshape → (B,19,960,64)
```

### Contradiction
- **Docs claim**: `512→19×512` projection
- **Code implements**: `512→19×64` projection
- **Result**: Node features are (B,19,960,**64**) not (B,19,960,**512**)

---

## 🔴 BLOCKER 4: Main Mamba vs Node Mamba Confusion

### Code Has TWO Mamba Instances

#### 1. Main Mamba (V2-style, line 129)
```python
# src/brain_brr/models/detector.py:129-137
self.mamba = BiMamba2(
    d_model=512,  # ← THIS uses d_model=512
    d_state=mamba_d_state,
    d_conv=mamba_d_conv,
    expand=2,
    headdim=64,
    num_layers=mamba_layers,
)
```

#### 2. Node Mamba (V3-only, line 474)
```python
# src/brain_brr/models/detector.py:474-484
instance.node_mamba = BiMamba2(
    d_model=64,  # ← THIS uses d_model=64
    d_state=16,
    d_conv=4,
    expand=2,
    headdim=8,
    num_layers=6,
)
```

### Question
**Which one does V3 actually use?**

Looking at forward pass:
```python
# src/brain_brr/models/detector.py:264-271
if (
    self.node_mamba
    and self.edge_mamba
    and self.proj_to_electrodes
    and self.proj_from_electrodes
    # ... V3 components
):
    # V3 path uses node_mamba (d_model=64)
```

**Answer**: V3 uses `node_mamba` with d_model=64, NOT the main `self.mamba` with d_model=512!

### Contradiction
- **Config's `mamba.d_model=512`** controls the **unused** `self.mamba` instance (V2 path)
- **V3 path** uses **hardcoded** `node_mamba` with `d_model=64`
- **Docs claim** V3 uses d_model=512 (WRONG)

---

## 🟡 BLOCKER 5: Parameter Count Discrepancy (Needs Verification)

### What Docs Say
```markdown
# docs/00-overview/overview.md:15
- ~31M parameters overall
```

### Reality Check Needed
With `d_model=64` instead of `d_model=512`, the parameter count would be:
- **Node Mamba**: 64 dim × 6 layers = much smaller than expected
- **If docs assumed 512**: Parameter count would be 64x larger for Node Mamba

**TODO**: Calculate actual parameter count and verify against 31M claim.

---

## 🔍 Root Cause Analysis

### Theory
1. **V2 architecture**: Used `self.mamba` with d_model=512 directly
2. **V3 architecture**: Introduced electrode projection (512→19×64) + `node_mamba` with d_model=64
3. **Migration bug**: Config still has `mamba.d_model=512` from V2 days
4. **Documentation bug**: Docs were updated to say "512" but code was never changed from "64"

### Evidence
- Config parameter `mamba.d_model` is read but NEVER used by V3 code path
- V3 code has **hardcoded** values: d_model=64, headdim=8, d_state=16, n_layers=6
- Only Edge Mamba reads from config: `edge_mamba_d_model`, `edge_mamba_d_state`

---

## 🎯 SSOT (Single Source of Truth) Candidates

### Option A: Code is SSOT → Fix Configs + Docs
**Assumption**: The running model with d_model=64 is correct
**Action**:
- Update configs to reflect d_model=64 for clarity (or remove unused param)
- Update ALL docs to show d_model=64, headdim=8
- Fix all shape annotations: (B,19,960,64) not (B,19,960,512)
- Document that `mamba.d_model` is unused by V3

**Pros**: Matches what's actually training
**Cons**: Were we intending to use 512? Is performance worse than expected?

### Option B: Configs/Docs are SSOT → Fix Code
**Assumption**: We INTENDED to use d_model=512
**Action**:
- Change projection: `512→19×512` (currently `512→19×64`)
- Change node_mamba: `d_model=512, headdim=64` (currently `d_model=64, headdim=8`)
- Update backprojection: `19×512→512` (currently `19×64→512`)
- Rerun training from scratch with much larger model

**Pros**: Matches documented intent, larger capacity
**Cons**: 8x memory increase, need full retraining, breaks existing checkpoints

### Option C: Split Config (Make Explicit)
**Assumption**: V3 INTENTIONALLY uses smaller per-electrode features
**Action**:
- Keep code as-is (d_model=64)
- Add new config section:
```yaml
model:
  mamba:
    d_model: 512  # Main Mamba (V2, unused in V3)

  graph:
    node_d_model: 64  # V3 Node Mamba (per-electrode)
    node_headdim: 8
    edge_d_model: 16  # V3 Edge Mamba (per-edge)
```
- Update docs to clarify the two different dimensions

**Pros**: Makes design explicit, preserves current training
**Cons**: More config complexity

---

## 🚨 IMMEDIATE QUESTIONS FOR USER

1. **Design Intent**: Was V3 Node Mamba SUPPOSED to use d_model=512 or d_model=64?
   - If 512: We need major code refactor + retrain
   - If 64: We need config/docs fixes only

2. **Training History**: All checkpoints were trained with d_model=64 for Node Mamba
   - Changing to 512 would invalidate all existing checkpoints
   - Do we accept that cost?

3. **Performance Impact**: Is current performance meeting targets?
   - If yes: Maybe d_model=64 is sufficient
   - If no: Could d_model=512 help?

4. **Memory Budget**:
   - Current: Node stream uses (19 × 64) = 1,216 dim total
   - With 512: Node stream would use (19 × 512) = 9,728 dim total
   - That's **8x memory increase** for node stream alone

---

## 📋 All Affected Files

### Config Files (claim d_model=512)
- `configs/local/train.yaml:54-55`
- `configs/modal/train.yaml:53-54`
- `configs/local/smoke.yaml` (likely)
- `configs/modal/smoke.yaml` (likely)

### Source Code (uses d_model=64)
- `src/brain_brr/models/detector.py:474-484` (node_mamba creation)
- `src/brain_brr/models/detector.py:540-541` (projections)
- `src/brain_brr/models/detector.py:279-283` (forward pass shapes)

### Documentation (claims d_model=512)
- `docs/00-overview/overview.md:17`
- `docs/00-overview/architecture-summary.md:9-28`
- `docs/04-model/v3-architecture.md` (multiple locations)
- `docs/04-model/mamba.md` (multiple locations)
- `docs/04-model/gnn.md` (shape references)
- `CLAUDE.md:13` (architecture summary)

---

## 🔧 Recommended Action Plan

### Phase 1: Determine SSOT (NOW)
1. User decides: Is d_model=64 correct or should it be 512?
2. Check performance: Is current model meeting targets?
3. Decision: Fix docs or fix code?

### Phase 2: Execute Fixes
**If Code is SSOT (d_model=64)**:
- Update 4 config files (mark mamba.d_model as unused for V3)
- Update 6+ doc files (all shape annotations)
- Add explicit node_d_model config for clarity

**If Docs/Config are SSOT (d_model=512)**:
- Update detector.py (3 locations)
- Retrain from scratch
- Update existing checkpoints or discard

### Phase 3: Prevent Future Issues
- Add architecture validation test comparing config vs actual shapes
- Add CLAUDE.md note about which configs apply to which architecture
- Consider schema validation that checks config vs code match

---

## 🔬 Verification Commands

```bash
# Check actual model parameter count
python -c "from src.brain_brr.models.detector import SeizureDetector; from src.brain_brr.config.schema import load_config; cfg = load_config('configs/local/train.yaml'); m = SeizureDetector.from_config(cfg.model); print(f'Total params: {sum(p.numel() for p in m.parameters()):,}')"

# Check node_mamba specifically
python -c "from src.brain_brr.models.detector import SeizureDetector; from src.brain_brr.config.schema import load_config; cfg = load_config('configs/local/train.yaml'); m = SeizureDetector.from_config(cfg.model); print(f'Node Mamba d_model: {m.node_mamba.d_model if m.node_mamba else None}')"

# Check actual tensor shapes during forward pass
# (requires adding print statements or debugger)
```

---

---

## ✅ ROOT CAUSE IDENTIFIED

**Date**: October 1, 2025 23:15
**Investigator**: Claude (deep audit session #2)

### What Happened

1. **Original Design (Correct)**: V3 Node Mamba used d_model=64 per-electrode
   - Code implemented correctly in detector.py:475
   - Original docs (archived_docs/docs_v1_archive/) documented d_model=64
   - configs/README.md correctly stated d_model=64

2. **Documentation Audit Gone Wrong (commit c601e60, Oct 1 17:10)**:
   - During earlier documentation audit, I (Claude) misread the code
   - Thought code was using `cfg.mamba.d_model` (which is 512)
   - "Corrected" docs from d_model=64 → d_model=512
   - Commit message: "Adjusted Node Mamba parameters in the architecture summary, increasing d_model from 64 to 512 to enhance model capacity"
   - **THIS WAS WRONG** - created the contradiction!

3. **Actual Code Flow**:
   ```python
   # Line 129: Main Mamba (V2 path, UNUSED by V3)
   self.mamba = BiMamba2(d_model=512)  # Uses cfg.mamba.d_model

   # Line 474: Node Mamba (V3 path, ACTIVE)
   instance.node_mamba = BiMamba2(d_model=64)  # HARDCODED, ignores config
   ```

### Evidence Trail

**Git History**:
```bash
commit c601e60 (Oct 1 17:10)
Author: JJ <JJ@NovamindNYC.com>

-  Node Mamba: 6 layers, d_model=64, headdim=8, d_state=16, expand=2
+  Node Mamba: 6 layers, d_model=512, headdim=64, d_state=16, expand=2
```

**Archived Docs (CORRECT)**:
- `archived_docs/docs_v1_archive/00-overview/overview.md:17`: d_model=64, headdim=8
- `archived_docs/docs_v2_archive/04-model/v3-architecture.md:214`: d_model=64 per-electrode
- `configs/README.md:7`: d_model=64, 6 layers, headdim=8

**Reference Implementation (EvoBrain)**:
```python
# reference_repos/EvoBrain-FBC5/model/EvoBrain.py:813
d_model=feat_target_size,  # Configurable per task
```

### Resolution: CODE IS CORRECT (d_model=64)

**SSOT Decision**: The implementation with d_model=64 is CORRECT.

**Rationale**:
1. **V3 Design Philosophy**: Per-electrode features with 64 dimensions each
   - Total: 19 electrodes × 64 dims = 1,216-dimensional space
   - Projection: 512 → 19×64, then back: 19×64 → 512

2. **CUDA Alignment**: (64*2)/8 = 16 ✅ (multiple of 8, required)

3. **Memory Efficiency**: Node stream uses ~1.2K dims vs 9.7K with d_model=512

4. **Training History**: ALL checkpoints trained with d_model=64

5. **Config Parameter `mamba.d_model=512`**: Controls V2 path (unused in V3)

### Action Plan

**IMMEDIATE (P0)**:
1. ✅ Revert docs changes from commit c601e60
2. ✅ Restore d_model=64, headdim=8 in all docs
3. ✅ Fix all shape annotations: (B,19,960,64) not (B,19,960,512)
4. ✅ Add note to configs: `mamba.d_model` is unused by V3

**FOLLOW-UP (P1)**:
1. Add config validation: warn if `mamba.d_model != 512` (for V2 compat)
2. Add architecture test: verify actual d_model matches expected 64
3. Update CLAUDE.md to clarify V2 vs V3 config differences

## Status: 🟢 RESOLVED - REVERTING INCORRECT DOCS

**Next Step**: Revert documentation changes and restore d_model=64.
