# P0 CRITICAL ISSUES - MUST FIX

**Date:** 2025-09-19
**Status:** 🔥 CRITICAL - Training blocked

## 🚨 P0: Training Blockers (FIX NOW)

### 1. TUSZ Channel Naming Mismatch [BLOCKER]
**File:** `src/brain_brr/data/io.py:169`
**Issue:** TUSZ files have channels like `'EEG FP1-LE'` but code expects `'Fp1'`
**Impact:** Training crashes immediately with "Missing required channels"
**Evidence:**
```
ValueError: Missing required channels: {'F3', 'P4', 'T4', 'Fp2', 'O1', 'Pz', 'T6', 'Cz', 'F8', 'P3', 'C4', 'C3', 'O2', 'F7', 'Fp1', 'Fz', 'T5', 'T3', 'F4'}
```
**Fix:** Update channel normalization to strip 'EEG ' prefix and '-LE'/'-REF' suffixes

### 2. Dataset Label Duration Bug [DATA CORRUPTION]
**File:** `src/brain_brr/data/datasets.py:108`
**Issue:** Passing `n_samples` instead of `duration_sec` to `events_to_binary_mask`
**Impact:** Labels are wrong length (15360 × 256 = 3,932,160 instead of 15360)
**Current:**
```python
binary_mask = events_to_binary_mask(events, n_samples, 256)
```
**Should be:**
```python
binary_mask = events_to_binary_mask(events, n_samples / 256, 256)  # duration in seconds
```

### 3. Training Hangs After Validation
**File:** `src/brain_brr/train/loop.py`
**Issue:** Training shows "0/38" progress but never advances
**Symptoms:** Process alive but stuck, validation ran but training won't start
**Possible cause:** DataLoader hanging with multiprocessing issues on WSL2

## 🔧 P1: Broken But Workaroundable

### 4. Pyproject Entrypoints Point to Dead Code
**File:** `pyproject.toml:109,115`
**Issue:** Still references removed `src.experiment` module
```toml
[project.scripts]
train = "src.experiment.pipeline:main"  # BROKEN - module doesn't exist

[tool.hatch.version]
path = "src/experiment/__init__.py"  # BROKEN - file doesn't exist
```
**Fix:**
```toml
train = "src.brain_brr.train.loop:main"
path = "src/brain_brr/__init__.py"
```

### 5. LR Scheduler Warning
**File:** `src/brain_brr/train/loop.py:237`
**Issue:** PyTorch warns scheduler.step() called before optimizer.step()
**Impact:** First LR value skipped (minor but annoying)
**Note:** Code LOOKS correct, might be initialization issue

## ✅ P2: Documentation/Cosmetic

### 6. Old Documentation References
- AGENTS.md still mentions preserving `src/experiment/` (we removed it intentionally!)
- Phase docs reference old paths
- These are just outdated docs, not actual bugs

### 7. Model Output Semantics
**File:** `src/brain_brr/models/detector.py`
**Issue:** Docstring says "Sigmoid" but returns logits (which is correct for BCEWithLogitsLoss)
**Impact:** None - code is right, docs are wrong

## 🎯 First Principles Analysis

### What's ACTUALLY broken:
1. **TUSZ channel names** - Real blocker, prevents any training
2. **Label duration bug** - Real bug, corrupts training data
3. **Training hang** - Real issue, might be WSL2/multiprocessing related
4. **Pyproject paths** - Real but easy fix

### What's NOT actually broken:
- The `/experiment/` removal was INTENTIONAL (successful refactor!)
- Model returns logits (correct design for numerical stability)
- Torch version constraint (needed for mamba-ssm compatibility)
- Mamba kernel coercion (expected behavior, has fallback)

## 🚀 Action Plan

1. **IMMEDIATE (to unblock training):**
   - Fix TUSZ channel normalization
   - Fix label duration calculation
   - Debug/fix training hang (try num_workers=0)

2. **QUICK FIXES:**
   - Update pyproject.toml entrypoints
   - Fix LR scheduler warning if possible

3. **CLEANUP (later):**
   - Update docs to reflect new structure
   - Remove references to old `/experiment/` module

## 💡 Key Insight

The audit caught some real bugs but also flagged intentional refactor changes as "issues". The REAL blockers are:
- TUSZ data format incompatibility
- Label duration math error
- Training process hanging

Everything else is either intentional (refactor) or cosmetic (docs).