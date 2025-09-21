# MODAL PIPELINE ROOT CAUSE ANALYSIS

## THE CONFUSION: Why is Modal using 50 files?

### First Principles: What SHOULD happen

1. **Smoke Test Pipeline** (`modal run --detach deploy/modal/app.py::train --config-path configs/smoke_test.yaml`)
   - Should set BGB_LIMIT_FILES=50
   - Should use cache at /results/cache/smoke/
   - Should process 50 train + 10 val files

2. **Full Training Pipeline** (`modal run --detach deploy/modal/app.py::train`)
   - Should NOT set BGB_LIMIT_FILES (or unset it)
   - Should use cache at /results/cache/tusz/
   - Should process ALL 3734 files
   - Default config is tusz_train_a100.yaml (line 91 of app.py)

### What ACTUALLY Happened

Looking at your Modal output:
```
[DEBUG] BGB_LIMIT_FILES=50: using 50 train, 10 val files
[DATA] Processing file 1/50: aaaaagxr_s018_t000.edf
```

This means Modal IS seeing BGB_LIMIT_FILES=50 even though we're running full training!

## ROOT CAUSE INVESTIGATION

### Hypothesis 1: Environment Variable Inheritance
Modal might be inheriting BGB_LIMIT_FILES from:
- Your local shell environment
- Previous Modal runs
- The Modal Secret/environment configuration

### Hypothesis 2: Config Path Detection Bug
Line 121 in app.py checks: `if "smoke" in config_path.lower()`
But what if config_path is being set incorrectly?

### Hypothesis 3: Default Config Not Working
The default `config_path="configs/tusz_train_a100.yaml"` might not be applying

## THE FIX WE IMPLEMENTED

```python
# Line 121-125 in deploy/modal/app.py
if "smoke" in config_path.lower():
    env["BGB_LIMIT_FILES"] = "50"
else:
    # EXPLICITLY UNSET for full training to avoid inheritance
    env.pop("BGB_LIMIT_FILES", None)
```

This SHOULD fix it by:
1. Setting BGB_LIMIT_FILES=50 ONLY for smoke tests
2. EXPLICITLY REMOVING it for all other configs

## VERIFICATION NEEDED

We need to see if the latest Modal run (after the fix) shows:
- NO "BGB_LIMIT_FILES=50" debug message
- Processing 3734 files, not 50
- Building cache at /results/cache/tusz/

## COMPLETE PIPELINE STATUS

### Local Pipeline ✅
- **Smoke**: cache/smoke/ (50 files) - Config fixed
- **Full**: cache/tusz/ (3734 files) - CURRENTLY RUNNING, 245+ files cached

### Modal Pipeline ❓
- **Smoke**: Should work (BGB_LIMIT_FILES=50 when "smoke" in path)
- **Full**: TESTING NOW with explicit unset fix

## THE DEEPER ISSUE

The real problem might be that Modal's environment is "sticky" - environment variables from previous runs or from Modal Secrets might persist. Our fix explicitly clears BGB_LIMIT_FILES for non-smoke runs, which should solve this.

## WHAT TO WATCH FOR

In the Modal dashboard for the latest run:
1. Check if it says "Loading 3734 train, 933 val files" (not 50/10)
2. Check if there's NO "BGB_LIMIT_FILES=50" debug message
3. Check if cache builds to /results/cache/tusz/

If it STILL shows 50 files, then:
- Modal might have BGB_LIMIT_FILES set as a Secret
- The config detection might be failing
- The subprocess might be inheriting from somewhere else

## NUCLEAR OPTION

If the fix doesn't work, we can be MORE aggressive:

```python
# Force unset at the START of the function
if "BGB_LIMIT_FILES" in os.environ:
    del os.environ["BGB_LIMIT_FILES"]

# Then in subprocess env:
if "smoke" in config_path.lower():
    env["BGB_LIMIT_FILES"] = "50"
else:
    # Triple-ensure it's not set
    env.pop("BGB_LIMIT_FILES", None)
    env["BGB_LIMIT_FILES"] = ""  # Set to empty string to override any inheritance
```