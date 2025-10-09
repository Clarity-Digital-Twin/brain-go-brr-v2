# Config Architecture Separation Strategy

**Date**: October 9, 2025
**Status**: 🟡 **READY TO EXECUTE** - Plan documented, awaiting implementation
**Purpose**: Separate BiMamba2 and FLA (Gated DeltaNet) configs for clean research comparison

---

## Philosophy: Separate Configs = Separate Research Subjects

**Core Principle**: BiMamba2 and FLA are **independent research architectures** being compared empirically. Each should have dedicated configs with zero architectural entanglement.

**Why NOT use config flags?**
```yaml
# ❌ BAD: Flag-based switching
model:
  use_fla: false  # or true
  edge_arch: "bimamba"  # or "gated_deltanet"
  node_arch: "bimamba"  # or "gated_deltanet"
```

**Problems with flags**:
- Code must check flags and branch logic everywhere
- Easy to create hybrid states accidentally (edge=FLA, node=BiMamba)
- Harder to diff architectures (flags scattered across config)
- Confusing commands (`--use_fla=true` less clear than `--config train_fla.yaml`)

**Why separate files?**
```yaml
# ✅ GOOD: Dedicated files
configs/local/smoke_bimamba.yaml  # BiMamba2 stack
configs/local/smoke_fla.yaml      # FLA stack
```

**Benefits**:
- **Crystal clear intent** - Filename tells you exactly what architecture runs
- **Clean comparison** - `diff smoke_bimamba.yaml smoke_fla.yaml` shows ONLY architectural differences
- **No code branching** - Code doesn't need to check architecture type
- **Future-proof** - Add new architectures by adding new files (smoke_arch3.yaml)
- **Professional standard** - How research labs manage multi-model experiments

---

## Current State (October 9, 2025)

### Production Configs (BiMamba2 Stack)
```
configs/local/
├── smoke.yaml    # 3 files, 1 epoch, RTX 4090
└── train.yaml    # Full dataset, 100 epochs, RTX 4090

configs/modal/
├── smoke.yaml    # 50 files, 1 epoch, A100-80GB
└── train.yaml    # Full dataset, 100 epochs, A100-80GB (CURRENTLY RUNNING)
```

**Status**: These are production-ready BiMamba2 configs. Modal training LIVE (W&B run: 983c1fbf706b4d0f8870cc0331dc6201).

### Ablation Configs (Phase Experiments - TO BE DELETED)
```
configs/local/
├── phase1a_edge_gdn.yaml     # Edge stream only (GatedDeltaNet)
├── phase1b_node_gdn.yaml     # Node stream only (GatedDeltaNet)
├── phase2_both_gdn.yaml      # Full dual-stream (small dataset)
└── phase2_medium_gdn.yaml    # Full dual-stream (medium dataset)
```

**Purpose**: These were incremental validation during FLA implementation (Phase 0 → Phase 1a → Phase 1b → Phase 2).

**Status**: ✅ All phases passed smoke tests. Configs served their purpose but are now technical debt.

**Why delete?**
- Not useful for final comparison (partial architectures, non-standard data splits)
- Confusing to keep (users might think these are "official" FLA configs)
- Redundant with full-stack `smoke_fla.yaml` / `train_fla.yaml`

---

## Target State (Post-Migration)

### Separated Architecture Configs
```
configs/local/
├── smoke_bimamba.yaml    # BiMamba2: 3 files, 1 epoch, RTX 4090
├── train_bimamba.yaml    # BiMamba2: Full dataset, 100 epochs, RTX 4090
├── smoke_fla.yaml        # FLA: 3 files, 1 epoch, RTX 4090
└── train_fla.yaml        # FLA: Full dataset, 100 epochs, RTX 4090

configs/modal/
├── smoke_bimamba.yaml    # BiMamba2: 50 files, 1 epoch, A100-80GB
├── train_bimamba.yaml    # BiMamba2: Full dataset, 100 epochs, A100-80GB
├── smoke_fla.yaml        # FLA: 50 files, 1 epoch, A100-80GB
└── train_fla.yaml        # FLA: Full dataset, 100 epochs, A100-80GB
```

**Key differences between BiMamba2 and FLA configs:**
```yaml
# smoke_bimamba.yaml
model:
  edge_arch: "bimamba"
  node_arch: "bimamba"
  edge_model:
    d_model: 64
    n_layers: 6
    # ... BiMamba-specific params
  node_model:
    d_model: 64
    n_layers: 6
    # ... BiMamba-specific params

# smoke_fla.yaml
model:
  edge_arch: "gated_deltanet"
  node_arch: "gated_deltanet"
  edge_model:
    d_model: 64
    n_layer: 6
    expand_k: 1.0
    expand_v: 2.0
    # ... GatedDeltaNet-specific params
  node_model:
    d_model: 64
    n_layer: 6
    expand_k: 1.0
    expand_v: 2.0
    # ... GatedDeltaNet-specific params
```

**All other settings IDENTICAL** (batch size, learning rate, loss, data paths, etc.).

---

## Migration Steps

### Phase 1: Rename Existing BiMamba2 Configs ✅ SAFE
```bash
# Local configs
cd /home/jj/proj/brain-go-brr-v2/configs/local
git mv smoke.yaml smoke_bimamba.yaml
git mv train.yaml train_bimamba.yaml

# Modal configs
cd /home/jj/proj/brain-go-brr-v2/configs/modal
git mv smoke.yaml smoke_bimamba.yaml
git mv train.yaml train_bimamba.yaml
```

**Why safe?** No code changes needed—just updating paths in commands/docs.

### Phase 2: Create FLA Configs (Copy + Edit Architecture Section)
```bash
# Local FLA configs
cp configs/local/smoke_bimamba.yaml configs/local/smoke_fla.yaml
cp configs/local/train_bimamba.yaml configs/local/train_fla.yaml

# Modal FLA configs
cp configs/modal/smoke_bimamba.yaml configs/modal/smoke_fla.yaml
cp configs/modal/train_bimamba.yaml configs/modal/train_fla.yaml
```

Then edit each `*_fla.yaml` file:
1. Change `edge_arch: "bimamba"` → `edge_arch: "gated_deltanet"`
2. Change `node_arch: "bimamba"` → `node_arch: "gated_deltanet"`
3. Update `edge_model` section to GatedDeltaNet params (d_model=64, n_layer=6, expand_k=1.0, expand_v=2.0, etc.)
4. Update `node_model` section to GatedDeltaNet params (same as edge)
5. Keep ALL other settings identical (batch_size, learning_rate, loss, data paths, etc.)

**Reference for FLA params**: Use `configs/local/phase2_both_gdn.yaml` as template (it has correct GatedDeltaNet structure).

### Phase 3: Delete Phase Configs
```bash
cd /home/jj/proj/brain-go-brr-v2/configs/local
git rm phase1a_edge_gdn.yaml
git rm phase1b_node_gdn.yaml
git rm phase2_both_gdn.yaml
git rm phase2_medium_gdn.yaml
```

**Why delete?** They served their purpose (incremental validation) but are now technical debt. Full-stack FLA configs replace them.

### Phase 4: Update Makefile Targets
```makefile
# Current (single architecture assumed)
.PHONY: s
s:  ## Local smoke test (3 files)
    $(PYTHON) -m src train configs/local/smoke.yaml

# New (explicit architecture selection)
.PHONY: smoke-bimamba
smoke-bimamba:  ## Local BiMamba2 smoke test (3 files)
    $(PYTHON) -m src train configs/local/smoke_bimamba.yaml

.PHONY: smoke-fla
smoke-fla:  ## Local FLA smoke test (3 files)
    $(PYTHON) -m src train configs/local/smoke_fla.yaml

.PHONY: train-bimamba
train-bimamba:  ## Local BiMamba2 full training (100 epochs)
    $(PYTHON) -m src train configs/local/train_bimamba.yaml

.PHONY: train-fla
train-fla:  ## Local FLA full training (100 epochs)
    $(PYTHON) -m src train configs/local/train_fla.yaml
```

**Keep legacy aliases** for backward compatibility:
```makefile
.PHONY: s
s: smoke-bimamba  ## Alias: Local smoke test (defaults to BiMamba2)

.PHONY: train-local
train-local: train-bimamba  ## Alias: Local training (defaults to BiMamba2)
```

### Phase 5: Update CLAUDE.md References
```markdown
# OLD
make s                  # Smoke test
make train-local        # Full training
modal run ... --config configs/modal/smoke.yaml

# NEW
make smoke-bimamba      # BiMamba2 smoke test
make smoke-fla          # FLA smoke test
make train-bimamba      # BiMamba2 full training
make train-fla          # FLA full training
modal run ... --config configs/modal/smoke_bimamba.yaml
modal run ... --config configs/modal/smoke_fla.yaml
```

Also update:
- Quick Commands table (lines ~20-35)
- Local Training section (lines ~45-65)
- Modal Cloud Deployment section (lines ~100-125)

### Phase 6: Update STATUS.md / README.md / Docs
Search for references to:
- `configs/local/smoke.yaml` → `configs/local/smoke_bimamba.yaml` (or context-specific)
- `configs/modal/train.yaml` → `configs/modal/train_bimamba.yaml` (or context-specific)
- Phase config paths → Remove or update to new FLA configs

---

## Validation Checklist

After migration, verify:

**Config Integrity**:
- [ ] All 8 configs exist (4 local + 4 modal)
- [ ] BiMamba2 configs unchanged except architecture naming
- [ ] FLA configs differ ONLY in architecture section
- [ ] No phase configs remain (`ls configs/local/phase*.yaml` returns nothing)

**Code Quality**:
- [ ] `make q` passes (lint + format + mypy)
- [ ] Config validation passes: `python -m src validate-config configs/local/smoke_bimamba.yaml`
- [ ] Config validation passes: `python -m src validate-config configs/local/smoke_fla.yaml`

**Smoke Tests**:
- [ ] BiMamba2 local smoke: `make smoke-bimamba` (or `python -m src train configs/local/smoke_bimamba.yaml`)
- [ ] FLA local smoke: `make smoke-fla` (or `python -m src train configs/local/smoke_fla.yaml`)
- [ ] Both complete without errors, produce valid checkpoints

**Documentation**:
- [ ] CLAUDE.md updated with new config paths
- [ ] STATUS.md updated with new config paths
- [ ] README.md updated with new config paths
- [ ] FLA docs reference correct config paths

---

## Usage Examples

### Local Development
```bash
# BiMamba2 smoke test (3 files, ~5 min)
make smoke-bimamba
# or: python -m src train configs/local/smoke_bimamba.yaml

# FLA smoke test (3 files, ~5 min)
make smoke-fla
# or: python -m src train configs/local/smoke_fla.yaml

# BiMamba2 full training (100 epochs, ~200-300 hours)
make train-bimamba
# or: python -m src train configs/local/train_bimamba.yaml

# FLA full training (100 epochs, ~200-300 hours)
make train-fla
# or: python -m src train configs/local/train_fla.yaml
```

### Modal Cloud Training
```bash
# BiMamba2 smoke test (50 files, ~10 min)
modal run --detach deploy/modal/app.py --action train \
  --config configs/modal/smoke_bimamba.yaml

# FLA smoke test (50 files, ~10 min)
modal run --detach deploy/modal/app.py --action train \
  --config configs/modal/smoke_fla.yaml

# BiMamba2 full training (100 epochs, ~100 hours, ~$319)
modal run --detach deploy/modal/app.py --action train \
  --config configs/modal/train_bimamba.yaml

# FLA full training (100 epochs, ~100 hours, ~$319)
modal run --detach deploy/modal/app.py --action train \
  --config configs/modal/train_fla.yaml
```

### Comparing Configs
```bash
# See architectural differences
diff configs/local/smoke_bimamba.yaml configs/local/smoke_fla.yaml

# Verify non-architectural settings identical
diff configs/local/smoke_bimamba.yaml configs/local/smoke_fla.yaml \
  | grep -v "arch\|model:"
# Should show ONLY architecture-related lines
```

---

## Rollback Plan (If Something Goes Wrong)

**Phase 1 rollback** (rename only):
```bash
git mv configs/local/smoke_bimamba.yaml configs/local/smoke.yaml
git mv configs/local/train_bimamba.yaml configs/local/train.yaml
git mv configs/modal/smoke_bimamba.yaml configs/modal/smoke.yaml
git mv configs/modal/train_bimamba.yaml configs/modal/train.yaml
```

**Phase 2-3 rollback** (new files + deletions):
```bash
git checkout HEAD -- configs/local/phase*.yaml  # Restore phase configs
rm configs/local/*_fla.yaml                     # Remove FLA configs
rm configs/modal/*_fla.yaml
```

**Full rollback** (if migration fails):
```bash
git reset --hard HEAD  # Revert all config changes
git clean -fd configs/ # Remove untracked files
```

**Why safe?** All changes are in `configs/` directory. No code logic changes required (detector.py already supports both architectures via `edge_arch`/`node_arch` params).

---

## Why This Matters for Research

**Scenario 1: BiMamba2 completes training**
```bash
# Immediately ready to launch FLA comparison
modal run --detach deploy/modal/app.py --action train \
  --config configs/modal/train_fla.yaml

# No config editing needed
# No risk of accidentally changing hyperparameters
# Perfect reproducibility
```

**Scenario 2: Future architecture (e.g., RWKV-7)**
```bash
# Add new configs without touching existing ones
cp configs/modal/train_bimamba.yaml configs/modal/train_rwkv7.yaml
# Edit architecture section only
# Launch immediately: modal run ... --config configs/modal/train_rwkv7.yaml
```

**Scenario 3: Ablation study (e.g., FLA edge + BiMamba2 node)**
```bash
# Create hybrid config explicitly
cp configs/local/train_fla.yaml configs/local/train_hybrid_fla_edge.yaml
# Edit: edge_arch="gated_deltanet", node_arch="bimamba"
# Clear naming prevents confusion
```

---

## Current Status

**Documentation**: ✅ COMPLETE - This document captures migration plan
**Execution**: 🟡 **PENDING** - Awaiting user approval to execute

**Next Steps**:
1. Review this plan with user
2. Execute Phase 1-6 systematically
3. Run validation checklist
4. Commit with message: `refactor: Separate BiMamba2 and FLA configs for clean research comparison`
5. Update all branches (feature → development → main)

**Timeline**: ~30-45 minutes for complete migration + validation

---

**Philosophy**: Clean separation now = zero friction later. When BiMamba2 finishes, you want `modal run ... --config train_fla.yaml` to Just Work™ with ZERO configuration risk. This migration ensures that. 🚀
