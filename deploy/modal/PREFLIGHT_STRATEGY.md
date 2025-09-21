# PREFLIGHT STRATEGY - VERIFY FIXES BEFORE BURNING RESOURCES

**Goal**: Verify CSV parser fixes work BEFORE full cache build or training
**Principle**: Test small → Verify → Scale up

## 1. PARSER VERIFICATION (10 min)

### Build tiny test cache to verify parser works
```bash
# Kill any existing builds first
tmux kill-session -t test-cache 2>/dev/null

# Start test cache build in tmux
tmux new -s test-cache -d "
source .venv/bin/activate
find data_ext4/tusz/edf/train -name '*.edf' | head -100 > test_files.txt
python -c '
from pathlib import Path
from src.brain_brr.data import EEGWindowDataset
files = [Path(f.strip()) for f in open(\"test_files.txt\")]
labels = [f.with_suffix(\".csv\") for f in files]
print(f\"Building test cache for {len(files)} files...\")
_ = EEGWindowDataset(files, label_files=labels, cache_dir=Path(\"cache/test\"))
print(\"Test cache built!\")
'
python -m src scan-cache --cache-dir cache/test
"

# Monitor
tmux attach -t test-cache
```

### SUCCESS CRITERIA:
- Manifest shows: `partial > 0` OR `full > 0`
- If still 0 seizures → PARSER STILL BROKEN, DO NOT PROCEED

## 2. PIPELINE STRUCTURE (How it SHOULD work)

### Local Pipeline (`configs/smoke_test.yaml` → `configs/tusz_train_wsl2.yaml`)
```yaml
# smoke_test.yaml
training:
  epochs: 1
  batch_size: 16
data:
  cache_dir: cache/tusz  # <-- Cache location

# tusz_train_wsl2.yaml
training:
  epochs: 100
  batch_size: 32
data:
  cache_dir: cache/tusz  # <-- Same cache
```

**The training loop (`src/brain_brr/train/loop.py`) AUTOMATICALLY:**
1. Checks if cache exists
2. If missing, builds it from `data.data_dir`
3. If exists, uses it
4. Auto-builds manifest if missing
5. Uses BalancedSeizureDataset if manifest exists

### Modal Pipeline (`scripts/modal_train.py`)
Similar but runs on cloud GPUs

## 3. EXECUTION PLAN

### Phase 1: Verify Parser (10 min)
```bash
# Run test cache build above
# CHECK: Manifest shows seizures > 0
```

### Phase 2: Smoke Test (30 min)
```bash
# The training pipeline will AUTO-BUILD full cache if needed
python -m src train configs/smoke_test.yaml
```

**What happens:**
- Checks `cache/tusz/train`
- If missing, builds from `data_ext4/tusz/edf/train`
- Creates manifest
- Runs 1 epoch
- **VERIFY**: Training logs show `Windows with seizures: X/Y (Z%)`

### Phase 3: Full Local Training (6-8 hours)
```bash
# Use existing cache from smoke test
tmux new -s train
source .venv/bin/activate
python -m src train configs/tusz_train_wsl2.yaml
# Ctrl-B D to detach
```

### Phase 4: Modal Training (Optional, after local works)
```bash
modal run scripts/modal_train.py
```

## 4. WHAT TO WATCH FOR

### During Cache Build:
```
[DATA] Processing file X/Y: filename.edf
[DATA] Building cache for filename.edf...
```

### After Cache Build:
```
✅ Cache build complete + manifest: partial=X, full=Y, none=Z
```
**CRITICAL**: If partial=0 and full=0 → STOP, parser still broken

### During Training:
```
[DATASET] BalancedSeizureDataset: N windows from manifest
[TRAIN] Epoch 1/1
[DATASET] Windows with seizures: X/Y (Z%)  <-- MUST BE > 0%
```

## 5. REPLICATABLE OSS COMMANDS

For someone cloning the repo fresh:

```bash
# Setup
git clone https://github.com/yourrepo/brain-go-brr-v2
cd brain-go-brr-v2
make setup

# Get data (they need to download TUSZ themselves)
# Place in data_ext4/tusz/edf/{train,eval}

# Run training (auto-builds cache)
python -m src train configs/smoke_test.yaml  # Quick test
python -m src train configs/tusz_train_wsl2.yaml  # Full training
```

## KEY INSIGHT: The pipelines are SELF-CONTAINED

- **No manual cache building needed** - training pipeline handles it
- **No manual manifest creation** - auto-built when needed
- **Automatic balancing** - uses BalancedSeizureDataset when manifest exists

## CURRENT STATUS

1. CSV parser FIXED (handles CSV_BI format)
2. Seizure types FIXED (all TUSZ types)
3. Hard guards ADDED (exits on 0 seizures)
4. Need to verify with test cache
5. Then use normal pipeline for everything else

---

**TLDR: Test small cache → Run smoke_test.yaml → Run full training**