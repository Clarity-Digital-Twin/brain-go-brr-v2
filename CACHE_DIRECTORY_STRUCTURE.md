# CACHE DIRECTORY STRUCTURE - EXACTLY WHERE SHIT GOES

## PROBLEM: Smoke test config has CONFLICTING cache directories!

```yaml
# configs/smoke_test.yaml
data:
  cache_dir: cache/data    # NOT SPECIFIED - uses default
experiment:
  cache_dir: cache/smoke   # Line 89 - DIFFERENT!
```

**Training uses `data.cache_dir` (or default `cache/data` if missing)**

## THE ACTUAL TRUTH:

### 1. LOCAL SMOKE TEST (`configs/smoke_test.yaml`)
- **BUILDS CACHE IN**: `cache/data/train/` and `cache/data/val/`
- **Size**: ~50 files (if using BGB_LIMIT_FILES=50)
- **Problem**: This is THROWAWAY cache, not reused

### 2. LOCAL FULL TRAINING (`configs/tusz_train_wsl2.yaml`)
- **BUILDS CACHE IN**: `cache/tusz/train/` and `cache/tusz/val/`
- **Size**: 3734 train files, 933 val files
- **THIS IS THE REAL CACHE**

### 3. MODAL TRAINING (`configs/tusz_train_a100.yaml`)
- **BUILDS CACHE IN**: `/tmp/cache/train/` and `/tmp/cache/val/`
- **Remote ephemeral storage**

## WHY THEY DON'T SHARE:

Each config specifies its own `data.cache_dir`:
- smoke_test.yaml → `cache/data` (default)
- tusz_train_wsl2.yaml → `cache/tusz`
- tusz_train_a100.yaml → `/tmp/cache`

**THEY NEVER SHARE CACHES!**

## THE FIX (if you want to share):

Edit `configs/smoke_test.yaml`:
```yaml
data:
  cache_dir: cache/tusz  # <-- ADD THIS LINE
```

Then smoke test would use/build same cache as full training.

## CURRENT REALITY:

```
cache/
├── test/          # Our manual test (100 files, has seizures!)
│   ├── *.npz
│   └── manifest.json
├── data/          # Smoke test builds here (50 files)
│   ├── train/
│   └── val/
└── tusz/          # Full training builds here (3734 files) - CURRENTLY BUILDING
    ├── train/
    └── val/
```

## RECOMMENDATION:

**LET THEM BE SEPARATE!**
- Smoke test = quick throwaway test
- Full training = real cache that matters
- Don't mix test garbage with production cache

## WHAT'S HAPPENING RIGHT NOW:

Full training is building in `cache/tusz/train/`:
- Processing file 61/3734 (1.6% done)
- Will take ~2-3 hours
- Will create manifest with seizure counts
- Will use BalancedSeizureDataset on next run

---

**TLDR: Smoke builds in `cache/data/`, Full builds in `cache/tusz/`. They don't share. That's fine.**