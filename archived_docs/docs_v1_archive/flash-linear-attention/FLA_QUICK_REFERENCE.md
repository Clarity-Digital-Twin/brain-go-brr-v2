# FLA Quick Reference - October 9, 2025

**TL;DR**: BiMamba2 baseline training LIVE on Modal. FLA infrastructure complete. Wait for baseline → Create Modal config → Train FLA → A/B comparison.

---

## 🎯 **Current Status**

| Component | Status | Evidence |
|-----------|--------|----------|
| **BiMamba2 Baseline** | 🔄 Training LIVE (Modal A100) | Check Modal logs |
| **FLA Infrastructure** | ✅ Complete | All smoke tests passed |
| **Medium Validation** | ⚠️ Technical success, performance unstable | `/tmp/phase2_medium.log` |
| **Modal FLA Config** | ❌ Not created yet | Create after BiMamba2 completes |

---

## 📊 **Full Roadmap**

**See**: `FLA_ROADMAP.md` (unified document with complete strategy)

**Archived** (redundant):
- `docs/archive_v1/FLA_IMPLEMENTATION_STATUS_OLD.md`
- `docs/archive_v1/PHASE2_EXECUTION_ROADMAP_OLD.md`

---

## 🚀 **Next Actions**

### **NOW** (Waiting)
```bash
# Monitor BiMamba2 Modal training
modal app list
modal app logs <app-id>

# Look for: "[VALIDATION] Starting disk-backed validation"
```

### **AFTER BiMamba2 COMPLETES** (~4-5 days)
```bash
# 1. Create Modal FLA config (~30 min)
cp configs/local/phase2_both_gdn.yaml configs/modal/phase2_both_gdn.yaml
# Edit: batch_size=48, mixed_precision=true, num_workers=4
make q  # validate

# 2. Launch FLA training
modal run --detach deploy/modal/app.py \
  --action train \
  --config configs/modal/phase2_both_gdn.yaml

# 3. Wait ~4-5 days

# 4. A/B comparison
# - If FLA ≥ BiMamba2 + 3%: Deploy FLA
# - If FLA < BiMamba2: Keep BiMamba2
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

**Our goal**: Know if FLA beats BiMamba2, not claim SOTA

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

## 🎉 **Bottom Line**

**Is FLA ready for local training?** YES (technically), but NOT RECOMMENDED (use Modal instead)

**Do we need Modal configs now?** NO - create after BiMamba2 baseline completes

**What's the strategy?** Two-stack A/B comparison (pragmatic, valid, achievable)

**When will we know?** ~2-3 weeks from now (BiMamba2 done + FLA done + comparison)

**Cost?** ~$650 total ($319 × 2 runs)

---

**Last Updated**: October 9, 2025
**Next Milestone**: BiMamba2 baseline completion (~4-5 days)
