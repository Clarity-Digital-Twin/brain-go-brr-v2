# P1: CACHE STRATEGY IS A DISASTER FOR OSS CONTRIBUTORS

**Severity**: P1 - Critical for Reproducibility & OSS Contribution
**Date**: 2025-09-20
**Status**: ACTIVE - Not blocking us, but will destroy anyone trying to reproduce
**Impact**: Every OSS contributor will waste days debugging cache issues

## THE DEEP, FUNDAMENTAL PROBLEM

We have created a cache system that is:
1. **Inconsistent** across environments
2. **Implicit** in its behavior
3. **Undocumented** in its requirements
4. **Conflicting** between configs
5. **Wasteful** of compute resources

This isn't just bad - it's **embarrassingly unprofessional** for an OSS project claiming to "shock the world with O(N) clinical seizure detection."

## PART 1: THE CACHE CONFUSION MATRIX

### What We Have (The Mess)

```
┌─────────────────────────────────────────────────────────────────────┐
│                     CURRENT CACHE CHAOS                              │
├─────────────────┬──────────────────┬─────────────────────────────────┤
│ Config          │ Cache Location    │ What Gets Cached                │
├─────────────────┼──────────────────┼─────────────────────────────────┤
│ smoke_test.yaml │ cache/smoke       │ First 50 files (BGB_LIMIT_FILES)│
│                 │ (experiment only) │ BUT NO data.cache_dir SET!      │
├─────────────────┼──────────────────┼─────────────────────────────────┤
│ tusz_train_wsl2 │ cache/tusz        │ All 3734 train + 933 val files  │
│                 │ (both data+exp)   │ ~300GB when complete             │
├─────────────────┼──────────────────┼─────────────────────────────────┤
│ tusz_train_a100 │ /results/cache/   │ All 3734 train + 933 val files  │
│                 │ tusz              │ ~300GB when complete             │
├─────────────────┼──────────────────┼─────────────────────────────────┤
│ local.yaml      │ cache             │ Whatever it processes            │
│                 │ (no subdirs!)     │ CONFLICTS WITH EVERYTHING       │
├─────────────────┼──────────────────┼─────────────────────────────────┤
│ production.yaml │ cache             │ Whatever it processes            │
│                 │ (no subdirs!)     │ CONFLICTS WITH EVERYTHING       │
└─────────────────┴──────────────────┴─────────────────────────────────┘
```

### The Disaster Scenarios

#### Scenario 1: OSS Contributor Runs Smoke Test
```bash
# Contributor thinks: "Let me test this quickly"
export BGB_LIMIT_FILES=50
python -m src train configs/smoke_test.yaml

# What happens:
# 1. NO data.cache_dir set, defaults to... what? cache/data?
# 2. Builds cache for 50 files to cache/smoke... maybe?
# 3. Takes 30 minutes for "smoke test"
# 4. Cache is USELESS for full training
```

#### Scenario 2: Contributor Switches to Full Training
```bash
# After smoke test works, they try full training
python -m src train configs/tusz_train_wsl2.yaml

# What happens:
# 1. Expects cache at cache/tusz - NOT THERE
# 2. Starts building 3734 files from scratch
# 3. Even though 50 files already cached at cache/smoke!
# 4. Wastes 3+ hours rebuilding
```

#### Scenario 3: Multiple Configs Fight Over Cache
```bash
# User tries different configs
python -m src train configs/local.yaml      # Writes to cache/
python -m src train configs/smoke_test.yaml # Writes to cache/smoke
python -m src train configs/tusz_train.yaml # Writes to cache/tusz

# Result:
# - 3 different caches for SAME DATA
# - 900GB wasted disk space
# - Complete confusion about which cache is "correct"
```

## PART 2: WHY THIS HAPPENED (ROOT CAUSE ANALYSIS)

### 1. Two Cache Config Keys Fighting
```yaml
data:
  cache_dir: ???  # Used by actual dataset/training code

experiment:
  cache_dir: ???  # Used by... logging? Nothing? WHO KNOWS?
```

**THE TRUTH**: Training uses `data.cache_dir`, not `experiment.cache_dir`!
But half our configs only set `experiment.cache_dir`!

### 2. No Separation of Concerns
- Smoke tests process FULL dataset path (`data_ext4/tusz/edf/train`)
- Only limited by environment variable (`BGB_LIMIT_FILES`)
- But cache to different location
- No way to share cache between smoke and full!

### 3. No Cache Validation
- System doesn't check if cache is complete
- Doesn't verify cache matches config
- Silently builds on-demand (SLOW)
- No warnings about cache mismatches

### 4. Modal vs Local Inconsistency
- Local: `cache/tusz`
- Modal: `/results/cache/tusz`
- No abstraction layer to handle this
- Hardcoded paths everywhere

## PART 3: WHAT PROFESSIONALS (GOOGLE/DEEPMIND) WOULD DO

### A. Proper Cache Hierarchy
```
cache/
├── smoke/                   # Quick tests (1-10 files)
│   ├── v1_256hz_10-20/     # Versioned by preprocessing
│   │   ├── manifest.json    # What's in this cache
│   │   └── *.npz           # Actual cache files
│   └── latest -> v1_256hz_10-20
├── dev/                     # Development (100-500 files)
│   ├── v1_256hz_10-20/
│   └── latest -> v1_256hz_10-20
└── full/                    # Complete dataset
    ├── v1_256hz_10-20/
    │   ├── train/           # 3734 files
    │   ├── val/             # 933 files
    │   └── manifest.json
    └── latest -> v1_256hz_10-20
```

### B. Single Source of Truth Config
```yaml
# config/cache/base.yaml
cache:
  version: v1_256hz_10-20
  root: ${CACHE_ROOT:-cache}  # Override via env

  profiles:
    smoke:
      subset: first_50_files
      size: ~5GB
      build_time: ~10min

    dev:
      subset: first_500_files
      size: ~50GB
      build_time: ~1hr

    full:
      subset: all_files
      size: ~300GB
      build_time: ~3hr

# configs/smoke_test.yaml
cache:
  profile: smoke
  reuse_from: [full, dev]  # Can use larger caches if available!
```

### C. Cache Validation & Reuse
```python
class CacheManager:
    def find_compatible_cache(self, required_files: list[Path]) -> Path | None:
        """Find existing cache that contains all required files."""
        # Check full cache first
        if self.full_cache_exists():
            if self.full_cache.contains_all(required_files):
                return self.full_cache.path

        # Check dev cache
        if self.dev_cache_exists():
            if self.dev_cache.contains_all(required_files):
                return self.dev_cache.path

        return None

    def validate_cache(self, cache_dir: Path) -> CacheStatus:
        """Validate cache completeness and version."""
        manifest = self.load_manifest(cache_dir)

        # Check version matches
        if manifest.version != self.config.version:
            return CacheStatus.VERSION_MISMATCH

        # Check files present
        missing = self.find_missing_files(manifest.files)
        if missing:
            return CacheStatus.INCOMPLETE

        return CacheStatus.VALID
```

### D. Smart Cache Building
```python
def build_cache_incrementally(files: list[Path], cache_dir: Path):
    """Build only missing files, reuse existing."""
    existing = find_existing_cache_files(cache_dir)
    needed = set(files) - set(existing)

    if needed:
        print(f"Building {len(needed)} missing files...")
        print(f"Reusing {len(existing)} existing files...")

    for file in tqdm(needed):
        process_and_cache(file, cache_dir)
```

### E. Clear Documentation
```markdown
# Cache Management Guide

## Quick Start
- Smoke test: Runs automatically with 50 files
- Full training: Run `make cache-full` first (3 hours)
- Cache location: `cache/` (override with CACHE_ROOT env)

## Cache Profiles
| Profile | Files | Size | Build Time | Use Case |
|---------|-------|------|------------|----------|
| smoke   | 50    | 5GB  | 10 min     | CI/CD    |
| dev     | 500   | 50GB | 1 hour     | Development |
| full    | 4667  | 300GB| 3 hours    | Training |

## Reusing Cache
Smaller profiles automatically use larger caches if available:
- smoke test → uses full cache if present
- dev → uses full cache if present
```

## PART 4: THE IMPACT ON CONTRIBUTORS

### Current Experience (HELL)
1. Clone repo
2. Run smoke test → 30 minutes building cache
3. Run full training → 3 hours building DIFFERENT cache
4. Disk full from duplicate caches
5. Find cache mismatch issue
6. Delete everything, start over
7. Give up, close PR

### What It Should Be
1. Clone repo
2. Run `make cache-smoke` → 10 minutes
3. Run smoke test → uses cache, instant
4. Run `make cache-full` → builds on top of smoke cache
5. Run full training → uses cache, fast
6. Submit successful PR

## PART 5: IMMEDIATE FIXES NEEDED

### Fix 1: Unify Cache Config
```python
# src/brain_brr/config/schemas.py
@dataclass
class DataConfig:
    cache_dir: str = "cache/full"  # DEFAULT THAT MAKES SENSE

@dataclass
class ExperimentConfig:
    # REMOVE cache_dir from here - it doesn't belong!
    output_dir: str = "results"
```

### Fix 2: Add Cache Profile System
```python
# src/brain_brr/data/cache_profiles.py
CACHE_PROFILES = {
    "smoke": {
        "dir": "cache/smoke",
        "max_files": 50,
        "reuse_from": ["cache/dev", "cache/full"]
    },
    "dev": {
        "dir": "cache/dev",
        "max_files": 500,
        "reuse_from": ["cache/full"]
    },
    "full": {
        "dir": "cache/full",
        "max_files": None,
        "reuse_from": []
    }
}
```

### Fix 3: Add Cache Commands
```makefile
# Makefile additions
cache-smoke:
	python -m src.brain_brr.data.build_cache --profile smoke

cache-dev:
	python -m src.brain_brr.data.build_cache --profile dev

cache-full:
	python -m src.brain_brr.data.build_cache --profile full

cache-status:
	python -m src.brain_brr.data.cache_status
```

### Fix 4: Add Pre-Flight Check
```python
def pre_training_check(config):
    cache_status = validate_cache(config.data.cache_dir)

    if cache_status == CacheStatus.NOT_FOUND:
        print("❌ No cache found!")
        print(f"Run: make cache-{config.cache_profile}")
        sys.exit(1)

    elif cache_status == CacheStatus.INCOMPLETE:
        print("⚠️ Cache incomplete!")
        compatible = find_compatible_cache(config.required_files)
        if compatible:
            print(f"Using compatible cache: {compatible}")
            config.data.cache_dir = compatible
        else:
            print("Run: make cache-full")
            sys.exit(1)
```

## PART 6: LONG TERM VISION

### The Dream Setup
```bash
# First time user
git clone https://github.com/brain-go-brr/v2
cd v2
make setup

# Automatically detects no cache
> No cache detected. Choose:
> 1. Download pre-built cache (5GB smoke / 300GB full)
> 2. Build cache locally (10min smoke / 3hr full)
> 3. Continue without cache (not recommended)

# User selects 1
> Downloading smoke cache (5GB)...
> ✅ Ready for smoke tests!

# Run smoke test - INSTANT
make test-smoke  # Uses downloaded cache

# Upgrade to full
make cache-upgrade
> Downloading remaining cache files (295GB)...
> ✅ Ready for full training!
```

## PART 7: WHY THIS MATTERS

### For OSS Success
- **First Impression**: Smoke test should work in <5 minutes
- **Reproducibility**: Anyone should get same results
- **Contributor Experience**: No hidden gotchas
- **Professional Standards**: Google/Meta level quality

### For Scientific Credibility
- **Reproducible Results**: Same cache = same results
- **Version Control**: Cache versioned with preprocessing
- **Audit Trail**: Manifest tracks what was cached when
- **No Silent Failures**: Explicit about cache state

## THE BOTTOM LINE

**We built a Ferrari engine (Bi-Mamba) but forgot the keys (cache management).**

Every single OSS contributor will:
1. Hit cache confusion
2. Waste hours debugging
3. Get frustrated
4. Give up

This is not a P0 for us (we know the workarounds), but it's a **P0 for OSS adoption**.

## ACTION ITEMS

### Immediate (This Week)
1. [ ] Add `data.cache_dir` to ALL configs
2. [ ] Remove `experiment.cache_dir` (it does nothing!)
3. [ ] Document cache behavior in README
4. [ ] Add cache-status command

### Short Term (Next Sprint)
1. [ ] Implement cache profiles
2. [ ] Add cache reuse logic
3. [ ] Create pre-built cache archives
4. [ ] Add cache validation

### Long Term (v2.1)
1. [ ] Cache versioning system
2. [ ] Automatic cache downloads
3. [ ] Cache sharing between users
4. [ ] Cloud cache sync

---

**Remember**: Every hour we don't fix this is an hour multiplied by every future contributor.

**This is technical debt at 29% APR compounding daily.**