# Archived: False Protection Claims (Oct 2025)

**Date Archived**: October 4, 2025
**Reason**: These documents claimed protection features that were never implemented

## What Was Wrong

These documents claimed that 4 protection flags (`BGB_SANITIZE_GRADS`, `BGB_SANITIZE_INPUTS`, `BGB_SKIP_OPT_STEP_ON_NAN`, `BGB_SAFE_CLAMP`) were "REQUIRED", "CRITICAL", or "MANDATORY" for training, but:

- The flags were defined in `env.py` but **never used** in training code
- Grep searches showed 0 references in `src/` (except definitions)
- Training worked fine without them because gradient clipping was always applied

## What Was Actually Working

- **Gradient clipping**: `torch.nn.utils.clip_grad_norm_()` from config (always applied)
- **LayerNorm**: At 5 component boundaries (always enabled via config)
- **Preprocessing**: Outlier clipping to ±10σ in `preprocess.py`
- **Detached eigenvectors**: v3.3.1 fix preventing gradient explosion

## What Changed (Oct 4, 2025)

1. **Implemented `BGB_SANITIZE_GRADS`** as optional debugging tool (default: OFF)
2. **Removed `BGB_SANITIZE_INPUTS`** - preprocessing handles this
3. **Removed `BGB_SKIP_OPT_STEP_ON_NAN`** - breaks LR schedules
4. **Removed `BGB_SAFE_CLAMP`** - LayerNorm is better
5. **Updated all active documentation** to reflect truth

## Replacement Documentation

- **Authoritative guide**: `docs/08-operations/gradient-protection-guide.md`
- **Environment variables**: `CLAUDE.md` (updated)
- **Config comments**: `configs/local/*.yaml` (updated)

## Files in This Archive

- `nan-prevention-complete.md` - Claimed "3-tier protection system" that didn't exist

---

**Key Takeaway**: Training was always protected by gradient clipping. The flags were documentation-only features that created a false sense of security.
