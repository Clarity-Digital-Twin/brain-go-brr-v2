# FLA Quick Reference - October 9, 2025

**TL;DR**: BiMamba2 baseline training LIVE on Modal. FLA infrastructure complete. Wait for baseline → Create Modal config → Train FLA → A/B comparison. Architecture details now live in `docs/04-model/mamba.md`; this file tracks operational status.

---

## 🎯 **Current Status (v4.0.0 - October 16, 2025)**

| Component | Status | Evidence |
|-----------|--------|----------|
| **BiMamba2 Baseline** | ⏸️ PAUSED at Epoch 6 (Modal A100) | High costs ($18,600 projected for 100 epochs) |
| **FLA Training** | 🔄 ACTIVE - Epoch 7/100 (Local RTX 4090) | Progressing normally, $0 cost (local hardware) |
| **WSL2 SIGBUS Fix** | ✅ RESOLVED | Cache on ext4 filesystem (see docs/08-operations/wsl2-sigbus-fix.md) |
| **Dual-Stack Milestone** | ✅ ACHIEVED (v4.0.0) | Both architectures production-ready |

---

## 📊 **Full Roadmap**

**See**: `FLA_ROADMAP.md` (unified document with complete strategy)

**Archived** (redundant):
- `docs/archive_v1/FLA_IMPLEMENTATION_STATUS_OLD.md`
- `docs/archive_v1/PHASE2_EXECUTION_ROADMAP_OLD.md`

---

## 🚀 **Next Actions**

### **NOW** (October 2025)
```bash
# Monitor FLA local training (Epoch 7/100)
tmux attach -t train-fla  # If in tmux
tail -f results/local_fla_training/training.log

# FLA training progressing normally on local RTX 4090
# BiMamba2 Modal training PAUSED due to costs ($186/epoch measured)
```

### **STRATEGY UPDATE**
**Decision**: FLA is PRIMARY training stack (local, $0 cost) while BiMamba2 remains PAUSED (Modal, expensive). Both are novel architectures; results publishable regardless of outcome.

### **When FLA Completes** (~40 days remaining)
```bash
# 1. Document FLA results
# - Record final sensitivity@10FA, AUROC, TAES metrics
# - Analyze loss curves, gradient stability
# - Document training time per epoch

# 2. Compare vs BiMamba2 baseline
# - BiMamba2: 6 epochs of data available (Modal)
# - FLA: Full 100-epoch training (Local)
# - Calculate performance delta

# 3. Both results are publishable
# - Novel dual-stack architecture comparison
# - First FLA application to clinical EEG
# - Cost-effective local training methodology
```

---

## 💡 **Key Insights**

### **Why No Ablations?**

**Academic approach** (ideal):
- Test edge-only, node-only, GNN-only, LPE-only
- Cost: ~$5,000-10,000 + 6-12 months

**Our approach** (pragmatic):
- Test TWO full stacks: BiMamba2 vs FLA
- Cost: ~$650 + 2-3 weeks
- **This is valid**: Many papers compare full architectures

### **Why Full Dataset Only?**

Medium validation (50 files) showed:
- ✅ No crashes, no NaNs, no OOM
- ⚠️ Model collapsed (only 2.73% seizures)
- **Conclusion**: Need full 4667 files for stable training

### **Expected Performance?**

TUSZ benchmark (see `literature/markdown/EEG-BIMAMBA`):
- Top models: 40-60% sensitivity@10FA
- 12:1 class imbalance (seizures are rare)
- Seizure detection is HARD

**Our goal**: Empirically compare BiMamba2 vs FLA on TUSZ. Both stacks are novel - no prior work exists with these architectures. Results are publishable regardless of which performs better.

---

## 📚 **Documentation**

| File | Purpose |
|------|---------|
| `FLA_ROADMAP.md` | **MAIN** - Complete strategy, constraints, timeline |
| `FLA_QUICK_REFERENCE.md` | This file - Quick status check |
| `CLAUDE.md` | Updated with FLA status (lines 368-378) |
| `configs/local/phase2_both_gdn.yaml` | FLA config (smoke + medium validated) |
| `configs/modal/phase2_both_gdn.yaml` | **TODO** - Create after BiMamba2 completes |

---

## 🎯 **Bottom Line: Research Exploration**

**Is FLA ready for local training?** YES (technically), but NOT RECOMMENDED (use Modal instead)

**Do we need Modal configs now?** NO - create after BiMamba2 baseline completes

**What's the strategy?** Two-stack research comparison - train both, document both, compare results

**When will we know?** ~2-3 weeks from now (BiMamba2 done + FLA done + comparison)

**Cost?**
- **BiMamba2 (Modal)**: $186/epoch measured → $18,600 projected for 100 epochs (PAUSED)
- **FLA (Local RTX 4090)**: $0 (only electricity, negligible)
- **Strategy**: FLA is cost-effective primary training; BiMamba2 provides high-end comparison baseline

**Key insight**: Both stacks are novel. Both results are publishable regardless of outcome. Local training proved viable alternative to expensive cloud compute.

---

**Last Updated**: October 16, 2025 (v4.0.0 training status)
**Next Milestone**: FLA training completion (~40 days, Epoch 7/100 current)
