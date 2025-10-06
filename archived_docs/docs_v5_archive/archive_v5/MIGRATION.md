# Migration Guide: V3 → V4

**Breaking Changes Summary**

## Removed: Custom Split Policy Support

**Effective Version:** V4.0+

### What Changed

The deprecated `split_policy`, `validation_split`, and `split_seed` configuration fields have been **removed**. The system now **always** uses the official TUSZ patient-disjoint splits.

### Why This Change

1. **Patient Leakage Prevention**: The old custom split policy could cause the same patient to appear in both train and validation sets, leading to overly optimistic metrics
2. **Reproducibility**: Official TUSZ splits ensure consistent results across research groups
3. **Code Simplification**: Removes ~60 lines of deprecated code paths

### Migration Steps

#### Before (V3.x):

```yaml
# configs/local/train.yaml
data:
  data_dir: data_ext4/tusz/edf
  cache_dir: cache/tusz
  split_policy: official_tusz          # ← REMOVE THIS LINE
  validation_split: 0.2                # ← REMOVE THIS LINE (if present)
  split_seed: 42                       # ← REMOVE THIS LINE (if present)
  sampling_rate: 256
  # ... other fields
```

#### After (V4.0):

```yaml
# configs/local/train.yaml
data:
  data_dir: data_ext4/tusz/edf        # Parent dir containing train/dev/eval
  cache_dir: cache/tusz                # Will create train/ and dev/ subdirs
  sampling_rate: 256
  # ... other fields
```

### Directory Structure Requirements

Your data directory must follow the official TUSZ structure:

```
data_ext4/tusz/edf/
├── train/               # Training split (patient-disjoint)
│   ├── 01_tcp_ar/
│   ├── 02_tcp_le/
│   └── 03_tcp_ar_a/
├── dev/                 # Validation split (patient-disjoint)
│   ├── 01_tcp_ar/
│   ├── 02_tcp_le/
│   └── 03_tcp_ar_a/
└── eval/                # Test split (never used during training)
    ├── 01_tcp_ar/
    ├── 02_tcp_le/
    └── 03_tcp_ar_a/
```

### Cache Directory Naming

The validation cache directory is now **always** named `dev/` (not `val/`):

```
cache/tusz/
├── train/              # Training cache
│   ├── manifest.json
│   └── *.npz
└── dev/                # Validation cache (changed from "val")
    └── *.npz
```

**Note**: If you have existing `cache/tusz/val/` from V3, rename it to `cache/tusz/dev/` or rebuild the cache.

### Error Messages

If your config still contains deprecated fields, you will see:

```
ValidationError: Extra inputs are not permitted
  split_policy: Extra inputs are not permitted
  validation_split: Extra inputs are not permitted
```

**Solution**: Remove the deprecated fields from your YAML config.

### Automated Migration

Use this script to automatically update your configs:

```bash
# Remove split_policy lines from all configs
find configs/ -name "*.yaml" -type f -exec sed -i '/split_policy:/d' {} \;
find configs/ -name "*.yaml" -type f -exec sed -i '/validation_split:/d' {} \;
find configs/ -name "*.yaml" -type f -exec sed -i '/split_seed:/d' {} \;
```

### Testing After Migration

```bash
# 1. Verify config is valid
.venv/bin/python -c "
from pathlib import Path
from src.brain_brr.config import load_config
cfg = load_config(Path('configs/local/train.yaml'))
print('✅ Config valid')
"

# 2. Run smoke test
export BGB_LIMIT_FILES=3
make s

# 3. Check that splits are patient-disjoint
# Look for this log message:
# "✅ PATIENT DISJOINTNESS VERIFIED - No leakage!"
```

### Rollback (if needed)

If you need to temporarily use V3 with custom splits:

```bash
git checkout v3.x.x  # Replace with your last V3 version
```

Then update to V4 when ready.

---

## Summary

| Field | Status | Action |
|-------|--------|--------|
| `split_policy` | ❌ Removed | Delete from configs |
| `validation_split` | ❌ Removed | Delete from configs |
| `split_seed` | ❌ Removed | Delete from configs |
| Official TUSZ splits | ✅ Required | Ensure data dir follows TUSZ structure |

**Questions?** See `docs/` or open an issue.
