# Release Notes

## v0.2.0 - Critical Bug Fixes (2025-09-21)

### 🚨 Critical Fixes Required

This release fixes **P0 blockers** that prevented seizure detection in training. If you're using v0.1.0, **upgrade immediately**.

### What's Fixed

#### CSV Parser (CRITICAL)
- **Before**: Training detected 0% seizures due to broken TUSZ CSV_BI parser
- **After**: Parser correctly reads all seizure annotations
- **Impact**: Training now finds 313 partial and 55 full seizure windows in test cache

#### Seizure Type Detection
- **Before**: Only looked for "seiz" label (doesn't exist in TUSZ)
- **After**: Detects all TUSZ types: gnsz, fnsz, cpsz, absz, spsz, tcsz, tnsz, mysz
- **Impact**: Complete seizure coverage in training data

#### Training Stability
- Implemented BalancedSeizureDataset with SeizureTransformer's formula
- Added hard guards to prevent training with 0 seizures
- Fixed Modal pipeline limiting to 50 files instead of 3734

#### Configuration Cleanup
- Reorganized configs into clean `local/` and `modal/` structure
- Fixed WSL2 compatibility issues
- Verified A100 optimizations for cloud training

### Quick Upgrade

```bash
git pull
git checkout v0.2.0

# For local training
python -m src train configs/local/train.yaml

# For Modal cloud
modal run --detach deploy/modal/app.py::train
```

### Verification

After cache build, you should see:
```
✅ Cache build complete + manifest: partial=XXX, full=XX, none=XXXX
```

If `partial > 0`, the fixes are working correctly.

### Documentation

- See `configs/README.md` for new config structure
- Check `CHANGELOG.md` for complete fix details
- Review `FIX_SUMMARY_20250921.md` for technical details

---

**Full Changelog**: https://github.com/yourusername/brain-go-brr-v2/compare/v0.1.0...v0.2.0
