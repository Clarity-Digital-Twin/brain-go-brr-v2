# P0: CRITICAL TRAINING PIPELINE INVESTIGATION
**Severity**: P0 - CRITICAL SYSTEM FAILURE
**Date**: 2025-09-20
**Status**: ACTIVE TRAINING COMPROMISED

## EXECUTIVE SUMMARY
The entire training pipeline is fundamentally broken due to inconsistent cache handling between local and Modal environments. This is causing 30-60x slower training, wasting hundreds of dollars in compute, and making the system unprofessional and unreliable.

## THE FUNDAMENTAL PROBLEM

### What Should Happen (Professional Pipeline)
1. **First Run**: Build complete cache once (~1-2 hours)
2. **All Future Runs**: Load from cache (<1 minute)
3. **Consistent Behavior**: Same code path for local and cloud
4. **Clear Cache Management**: One source of truth for cache location

### What Actually Happens (Current Disaster)
1. **Local**: Partial cache (246/3734 files), builds rest on-demand during training
2. **Modal**: No cache exists, builds all 3734 files upfront before training
3. **Cache Directory Mismatch**: Config says `cache/tusz`, code uses `cache/train`
4. **Performance**: 27s/batch instead of 2-3s/batch

## DEEP DIVE: LOCAL TRAINING PIPELINE

### Current Behavior (WSL2)
```
CONFIG: cache_dir: cache/tusz
REALITY: cache/train has 246 files from smoke test
RESULT: Mixed cache/compute during training
```

#### The Flow:
1. **Dataset Init** (`src/brain_brr/data/datasets.py:45-84`)
   - Iterates through all 3734 EDF files
   - For each file, checks cache at `cache/tusz/{file}_windows.npz`
   - **PROBLEM**: Cache doesn't exist at `cache/tusz`, it's at `cache/train`!
   - Falls back to `_process_file()` for missing cache
   - Builds cache on-demand during training

2. **During Training** (batch iteration)
   - `__getitem__()` called for each batch
   - Line 156: Checks if cache exists
   - Line 165: If no cache, calls `_process_file()` DURING TRAINING
   - This triggers full EDF loading, preprocessing, windowing PER BATCH
   - Result: 27 seconds per batch instead of 2-3 seconds

3. **Evidence from Logs**:
```
[Batch 123] Interpolated channels ['Fz', 'Pz'] for file data_ext4/tusz/edf/train/aaaaaimu/s005_2014/01_tcp_ar/aaaaaimu_s005_t009.edf
```
This message appears DURING TRAINING at batch 123! It should NEVER appear during training.

## DEEP DIVE: MODAL TRAINING PIPELINE

### Current Behavior (A100)
```
CONFIG: cache_dir: /results/cache/tusz
REALITY: No cache exists
RESULT: Builds all 3734 files upfront
```

#### The Flow:
1. **Dataset Init**
   - Same code, but `/results/cache/tusz` is empty
   - Processes ALL 3734 files in `__init__()`
   - Takes ~6 hours on Modal due to network I/O
   - Creates complete cache before training starts

2. **During Training**
   - All data cached, fast batch loading
   - ~2-3s/batch once cache is built
   - But wasted 6 HOURS building cache that should exist!

3. **Evidence from Logs**:
```
[DATA] Processing file 3121/3734: aaaaaajz_s004_t001.edf
[DATA] Building cache for aaaaaajz_s004_t001.edf...
```
All 3734 files processed BEFORE epoch 1 starts.

## THE IDEAL SOLUTION (WHAT GOOGLE/DEEPMIND WOULD DO)

### 1. Unified Cache Strategy
```yaml
# BOTH configs should use same relative path structure
experiment:
  cache_dir: cache/tusz  # Consistent across environments
```

### 2. Cache Build Pipeline
```python
# Separate cache building from training
python -m src.brain_brr.data.build_cache \
    --data-dir data_ext4/tusz/edf/train \
    --cache-dir cache/tusz \
    --num-workers 8
```

### 3. Cache Verification
```python
def verify_cache_completeness(cache_dir: Path, data_files: list[Path]) -> bool:
    """Check if cache is complete before training."""
    missing = []
    for edf_path in data_files:
        cache_path = cache_dir / f"{edf_path.stem}_windows.npz"
        if not cache_path.exists():
            missing.append(edf_path)

    if missing:
        print(f"[CACHE] WARNING: {len(missing)} files not cached!")
        return False
    return True
```

### 4. Pre-Training Checklist
```python
def pre_training_checks(config):
    # 1. Verify cache exists and is complete
    if not verify_cache_completeness(config.cache_dir, data_files):
        if config.auto_build_cache:
            build_complete_cache(data_files, config.cache_dir)
        else:
            raise ValueError("Incomplete cache! Run build_cache first.")

    # 2. Log cache statistics
    print(f"[CACHE] Using {cache_dir} with {n_files} cached files")
    print(f"[CACHE] Total size: {get_dir_size(cache_dir) / 1e9:.1f} GB")
```

## ACTIONABLE FIXES

### IMMEDIATE (Stop the Bleeding)

#### Fix 1: Symlink Cache Directories
```bash
# After current training completes
ln -s $(pwd)/cache/train $(pwd)/cache/tusz
```

#### Fix 2: Update Dataset to Check Multiple Locations
```python
# In datasets.py __init__
cache_paths = [
    self.cache_dir / f"{edf_path.stem}_windows.npz",
    Path("cache/train") / f"{edf_path.stem}_windows.npz",  # Fallback
]
cache_path = next((p for p in cache_paths if p.exists()), None)
```

### SHORT TERM (Next 24 Hours)

#### Fix 3: Build Complete Cache
```bash
# Build full cache properly
python -c "
from src.brain_brr.data.datasets import EEGWindowDataset
from pathlib import Path

dataset = EEGWindowDataset(
    edf_files=list(Path('data_ext4/tusz/edf/train').glob('**/*.edf')),
    cache_dir=Path('cache/tusz_complete'),
)
print('Cache build complete!')
"
```

#### Fix 4: Create Cache Archive for Modal
```bash
# After cache is complete
tar -czf tusz_cache_complete_v2.tar.gz cache/tusz_complete/
# Upload to Modal volume or S3
```

### LONG TERM (Proper Fix)

#### Fix 5: Implement Cache Manager
```python
class CacheManager:
    """Centralized cache management."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.manifest_path = cache_dir / "manifest.json"

    def build_cache(self, data_files: list[Path], num_workers: int = 8):
        """Build complete cache with progress tracking."""
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for file in data_files:
                future = executor.submit(self._process_and_cache_file, file)
                futures.append(future)

            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()

    def verify_cache(self) -> CacheStatus:
        """Check cache completeness and integrity."""
        # Return detailed status
        pass

    def sync_to_cloud(self, remote_path: str):
        """Sync cache to S3/Modal volume."""
        pass
```

## COST ANALYSIS

### Current Waste
- **Modal A100**: $3/hour × 6 hours cache building = $18 per run
- **Local RTX 4090**: 27s/batch × 31,512 batches = 236 hours instead of 15 hours
- **Developer Time**: Countless hours debugging this mess

### After Fix
- **Modal A100**: <5 minutes to extract pre-built cache
- **Local RTX 4090**: 2-3s/batch = 15-20 hours total
- **Developer Time**: Zero debugging, it just works

## WHY THIS HAPPENED

1. **Config/Code Mismatch**: Config says `cache/tusz`, smoke test used `cache/train`
2. **No Cache Validation**: System doesn't verify cache before training
3. **Silent Fallback**: Dataset silently processes files on-demand instead of failing
4. **No Cache Management Tools**: No way to build, verify, or sync cache
5. **Inconsistent Environments**: Local and Modal behave differently

## LESSONS FOR GOOGLE/DEEPMIND STANDARDS

1. **Explicit is Better**: Fail loudly on missing cache, don't silently degrade
2. **Single Source of Truth**: One cache directory, one config
3. **Pre-Flight Checks**: Validate everything before expensive operations
4. **Tooling First**: Build cache management tools before training
5. **Reproducibility**: Same behavior local and cloud

## IMMEDIATE ACTION ITEMS

### DO NOW (While Training Runs)
1. ✅ Document the issue (this document)
2. ⬜ Monitor current training progress
3. ⬜ Prepare symlink command for after training

### DO AFTER TRAINING COMPLETES
1. ⬜ Create symlink: `ln -s $(pwd)/cache/train $(pwd)/cache/tusz`
2. ⬜ Test with small batch to verify cache usage
3. ⬜ Build complete cache if needed
4. ⬜ Create proper archive for Modal

### DO THIS WEEK
1. ⬜ Implement CacheManager class
2. ⬜ Add pre-training validation
3. ⬜ Create cache build script
4. ⬜ Update documentation

## MONITORING COMMANDS

```bash
# Check local training progress
tmux attach -t train

# Check Modal training
modal app logs

# Monitor cache directory growth
watch -n 10 'du -sh cache/* | sort -h'

# Check for on-demand processing (BAD!)
grep "Interpolated channels" training.log | tail
```

## FINAL VERDICT

This is a **CRITICAL SYSTEM FAILURE** that makes the entire training pipeline unprofessional and unreliable. The fix is straightforward but requires immediate action to prevent further waste of compute resources.

**The system is building cache DURING TRAINING instead of BEFORE TRAINING.**

This is the difference between a professional ML pipeline and a amateur hack job.

---
**Status**: ACTIVE - Both training runs compromised but continuing
**Next Update**: After current training completes (~15 hours)