# CRITICAL: TUSZ Naming Convention - USE 'DEV' NOT 'VAL'

## 🚨 ULTRA IMPORTANT - READ THIS FIRST

**WE USE `dev` EVERYWHERE, NOT `val`!**

This is DEEPLY needed for clarity and consistency with TUSZ official documentation.

## Why This Matters

TUSZ (Temple University Hospital EEG Seizure Corpus) provides THREE official splits:
- `train/` - Training data (4,667 files)
- `dev/` - Development/validation data (1,832 files)
- `eval/` - Held-out evaluation data (865 files)

**WE DO NOT RENAME THESE!** Using `val` instead of `dev` causes massive confusion when:
- Reading TUSZ documentation
- Discussing with collaborators
- Publishing results
- Debugging data issues

## Where This Applies

### 1. File System Paths
```bash
# CORRECT ✅
cache/tusz/dev/
/results/cache/tusz/dev/

# WRONG ❌
cache/tusz/val/
/results/cache/tusz/val/
```

### 2. S3 Bucket Structure
```bash
# CORRECT ✅
s3://brain-go-brr-eeg-data-20250919/cache/tusz/dev/

# WRONG ❌
s3://brain-go-brr-eeg-data-20250919/cache/tusz/val/
```

### 3. Modal SSD Volume
```bash
# CORRECT ✅
/results/cache/tusz/dev/

# WRONG ❌
/results/cache/tusz/val/
```

### 4. CLI Commands
```bash
# CORRECT ✅
python -m src build-cache --cache-dir cache/tusz/dev --split dev

# WRONG (but accepted for backward compat) ⚠️
python -m src build-cache --cache-dir cache/tusz/val --split val
```

### 5. Configuration Files
```yaml
# CORRECT ✅
data:
  cache_dir: cache/tusz  # Contains train/ and dev/

# The system will look for:
# - cache/tusz/train/
# - cache/tusz/dev/
```

## Code Implementation

### Training Loop (src/brain_brr/train/loop.py)
```python
val_cache = data_cache_root / "dev"  # NOT "val"!
```

### CLI (src/brain_brr/cli/cli.py)
```python
# 'val' is accepted as an alias for backward compatibility
if split == "val":
    split = "dev"
```

### Modal App (deploy/modal/app.py)
```python
# Copy dev split (TUSZ 'dev' → cache 'dev')
# CRITICAL: We use 'dev' naming to match TUSZ's official split naming!
# DO NOT rename to 'val' - this causes confusion with TUSZ documentation.
dev_src = src / "dev"
dev_dst = dst / "dev"
```

## Documentation Consistency

Every document in this repo uses `dev`:
- README.md
- docs/02-data/cache-layout.md
- docs/08-operations/modal-volume-architecture.md
- configs/README.md
- CLAUDE.md (AI agent instructions)

## The Rule

**If you see `val` anywhere in a path, it's WRONG and needs fixing!**

Exception: Python variable names like `val_files` or `val_metrics` are fine - they're just variables, not filesystem paths.

## Why We're So Emphatic About This

This naming inconsistency caused DAYS of confusion and debugging. Using TUSZ's official naming (`dev`) everywhere:
- Eliminates confusion
- Matches official documentation
- Prevents future bugs
- Makes collaboration easier
- Ensures reproducibility

## Summary

**USE `dev` EVERYWHERE - NO EXCEPTIONS!**

This is not a preference - it's a REQUIREMENT for clarity and correctness.

---

*Last updated: September 26, 2025*
*Status: ENFORCED EVERYWHERE IN CODEBASE ✅*