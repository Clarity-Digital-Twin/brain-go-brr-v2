# CRITICAL CACHE FUCKUP POSTMORTEM

**Date**: 2025-09-21
**Severity**: P0 - CATASTROPHIC DATA PIPELINE FAILURE
**Impact**: 254GB useless cache, $60+ Modal credits wasted, 0% seizures in training

## THE CATASTROPHIC FAILURE

We built 252,089 windows of training data with **ZERO FUCKING SEIZURES**.

### What Happened
1. Built 254GB cache locally (3,734 files, 252k windows)
2. Built similar cache on Modal.com (burning credits)
3. Training collapsed to 100% negative predictions
4. Model learned NOTHING - just predicted "no seizure" always
5. Complete waste of compute and storage

### Root Cause: CSV Parser Was Reading Wrong Fucking Columns

TUSZ uses CSV_BI format (bipolar montage with channel-specific annotations):
```csv
# version = csv_v1.0.0
# duration = 300.00 secs
channel,start_time,stop_time,label,confidence
FP1-F7,0.0000,36.8868,bckg,1.0000
FP1-F7,36.8868,183.3055,cpsz,1.0000  # <-- Complex partial seizure
```

Our parser expected simple format:
```csv
0.0000,16.0000,bckg
16.0000,256.0000,seiz
```

**THE PARSER WAS TRYING TO PARSE "FP1-F7" AS A FLOAT FOR START TIME**

## THE BROKEN CODE (Before Fix)

```python
# src/brain_brr/data/io.py - THE BROKEN SHIT
def parse_tusz_csv(csv_path: Path):
    # ...
    parts = line.split(",")
    if len(parts) >= 3:
        try:
            start = float(parts[0])  # TRIES TO PARSE "FP1-F7" AS FLOAT
            end = float(parts[1])     # FAILS SILENTLY
            label = parts[2].strip()  # NEVER REACHES HERE
            events.append((start, end, label))
        except ValueError:
            continue  # SILENTLY SKIPS ALL SEIZURE LINES
```

Result:
- Parser returned EMPTY events list for files WITH seizures
- Binary mask created was ALL ZEROS
- Cache NPZ files had labels arrays but they were ALL ZEROS
- Training saw 0% seizures across 252k windows

## EVIDENCE OF THE FUCKUP

### 1. Cache Scan Results
```
WARNING: No partial seizure windows found in 3734 files!
  Full seizure: 0, No seizure: 252089
```

### 2. Known Seizure File Test
```bash
# File we KNOW has seizures (66% of recording)
csv: aaaaaaac_s001_t000.csv
grep -c "cpsz" → 22 seizure events

# Same file in cache
cache/tusz/train/aaaaaaac_s001_t000_windows.npz
Max value: 0.0
Has any seizures: False
```

### 3. Statistical Impossibility
- TUSZ dataset has ~3% seizure prevalence at file level
- We cached 3,734 files with ZERO seizures
- Probability of this happening randomly: 0.97^3734 ≈ 0%

## THE FIXES APPLIED

### 1. CSV_BI Parser Fixed (`src/brain_brr/data/io.py:221-279`)

```python
# FIXED VERSION
def parse_tusz_csv(csv_path: Path):
    # Parse duration from header comments
    if line.startswith("#"):
        if "duration" in line.lower():
            # Extract from "# duration = 300.00 secs"

    # Skip CSV header row
    if parts[0] == "channel":
        continue

    # TUSZ CSV_BI: channel,start_time,stop_time,label,confidence
    if len(parts) >= 4:
        # Skip channel (parts[0]), use correct columns
        start = float(parts[1])  # Now reads actual start time
        end = float(parts[2])    # Now reads actual stop time
        label = parts[3].strip()  # Now reads actual label
```

### 2. Added ALL TUSZ Seizure Types (`src/brain_brr/data/io.py:301`)

```python
# WAS: Only looking for "seiz"
seizure_labels = {"seiz"}

# NOW: All TUSZ seizure types
seizure_labels = {"seiz", "gnsz", "fnsz", "spsz", "cpsz",
                  "absz", "tnsz", "tcsz", "spkz"}
```

### 3. Test Confirming Fix Works

```python
# Same file after fix
Duration: 301.0s
Seizure events: 22
Background events: 44
First seizure: (36.8868, 183.3055, 'cpsz')
Mask shape: (77056,)
Seizure ratio: 66.6%  # <-- NOW DETECTS SEIZURES!
```

## SECONDARY PROBLEMS DISCOVERED

### 1. WeightedRandomSampler Was Doomed to Fail
Even if CSV parsing worked, randomly sampling 20k windows from 250k+ would likely miss seizures due to <1% prevalence at window level.

### 2. No Validation of Cache Contents
Cache building never checked if seizures were present. Could have caught this immediately.

### 3. Silent Failures
Parser silently skipped malformed lines with try/except, hiding the problem.

## WHAT WE LOST

1. **Time**: ~6 hours building useless caches
2. **Money**: $60+ Modal credits
3. **Storage**: 254GB of garbage data
4. **Progress**: Days of failed training attempts

## LESSONS LEARNED

1. **ALWAYS TEST PARSERS ON ACTUAL DATA FORMAT**
2. **NEVER TRUST SILENT TRY/EXCEPT BLOCKS**
3. **VALIDATE DATA PIPELINE OUTPUT HAS EXPECTED DISTRIBUTION**
4. **CHECK CACHE CONTENTS BEFORE TRAINING**

## CURRENT STATUS

- All caches DELETED
- All training STOPPED
- CSV parser FIXED and TESTED
- BalancedSeizureDataset ready (implements SeizureTransformer approach)
- AWAITING SENIOR REVIEW before rebuilding

## NEXT STEPS (AFTER REVIEW)

1. Rebuild cache with fixed parser
2. Verify manifest shows partial/full/no-seizure windows
3. Confirm >0% seizures in cache
4. Only then start training

---

**THIS FUCKUP COULD HAVE BEEN AVOIDED WITH ONE SIMPLE TEST OF THE CSV PARSER ON ACTUAL TUSZ DATA**