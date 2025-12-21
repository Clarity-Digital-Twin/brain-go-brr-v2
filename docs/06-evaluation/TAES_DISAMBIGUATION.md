# TAES Disambiguation - CRITICAL NAMING COLLISION

**Date Created**: 2025-10-15
**Status**: Permanent Reference
**Priority**: P0 - READ THIS BEFORE COMPARING RESULTS TO LITERATURE

---

## 🚨 THE PROBLEM: "TAES" MEANS TWO COMPLETELY DIFFERENT THINGS

The term "TAES" (Time-Aligned Event Scoring) is used in TWO COMPLETELY DIFFERENT WAYS in seizure detection literature and our codebase. **This naming collision is coincidental but EXTREMELY CONFUSING.**

---

## 1️⃣ TAES as a METRIC/SCORE (What WE Compute)

### What It Is
A **single aggregate quality score** between 0 and 1 that measures how well predicted events align with reference events.

### Implementation
**Location**: `src/brain_brr/eval/metrics.py:79-146`

```python
def calculate_taes(pred_events, ref_events, alpha=0.15) -> float:
    """Calculate Time-Aligned Event Scoring (TAES).

    For each reference event r:
    - Compute overlap with all predicted events
    - Score = overlap_duration / ref_duration (capped at 1)

    False alarm penalty:
    - For predicted events with no overlap, accumulate duration
    - Penalty = alpha * (fp_duration / total_pred_duration)

    Returns:
        TAES score in [0, 1]
    """
```

### How It Works
1. For each reference seizure event, compute overlap with all predictions
2. Score = min(1.0, total_overlap / ref_duration) — **partial credit!**
3. Apply false alarm penalty: α × (FP_duration / total_pred_duration)
4. Return single score: base_score - penalty

### Example Output
```
TAES: 0.9658  ← This is a QUALITY METRIC, NOT a sensitivity!
```

### Key Properties
- **Range**: [0, 1] where 1.0 = perfect alignment
- **Partial credit**: 50% overlap of a seizure = 0.5 score
- **Penalty weight**: α = 0.15 (configurable)
- **Output**: Single scalar value

### What It Measures
- Temporal alignment quality
- How well events overlap (not just binary detection)
- Balance between detection and false alarms

---

## 2️⃣ NEDC TAES as a SCORING SYSTEM (What Literature Refers To)

### What It Is
A **method for computing sensitivity/FA rates** used by Temple NEDC and competitions like Neureka 2020.

### Source
- Temple University's Neural Engineering Data Consortium (NEDC)
- Official TUSZ evaluation scorer option
- Used in academic papers and competitions

### How It Works
When calculating **sensitivity** (not a single score!):
1. For each reference event, compute overlap with all predictions
2. Count as TP with **partial credit**: overlap_fraction counts as fractional TP
3. Sum fractional TPs across all reference events
4. Sensitivity = total_fractional_TPs / total_reference_events

### Example (SeizureTransformer on TUSZ eval set)

**Same predictions, different scoring systems:**

| Scoring System | Sensitivity | FA/24h | Notes |
|----------------|-------------|--------|-------|
| **NEDC TAES** | 65.21% | 136.73 | Partial credit for overlap |
| **NEDC OVERLAP** | 45.63% | 26.89 | Binary (any overlap = TP) |
| **SzCORE Event** | 52.35% | 8.59 | ±30s/60s tolerances |

**CRITICAL**: Same predictions → **3.1× difference in FA/24h** depending on scorer!

### Philosophy
- **NEDC TAES**: Strictest, rewards exact timing
- **NEDC OVERLAP**: Standard, any overlap counts
- **SzCORE Event**: Most permissive, allows early warnings

---

## 🔥 WHAT OUR CODE ACTUALLY IMPLEMENTS

### Our Metrics (FLA Exp4 Eval Example)
```
TAES: 0.9658                      ← METRIC #1 (quality score)
AUROC: 0.8654                     ← Discrimination ability
Sensitivity @ 10 FA/24h: 35.9%    ← Uses OVERLAP SCORING (#2 variant)
Sensitivity @ 5 FA/24h:  27.1%    ← Uses OVERLAP SCORING (#2 variant)
Sensitivity @ 2.5 FA/24h: 18.6%   ← Uses OVERLAP SCORING (#2 variant)
Sensitivity @ 1 FA/24h:   5.8%    ← Uses OVERLAP SCORING (#2 variant)
```

### Sensitivity Calculation (false_alarm.py:144-146)
```python
for ref_start, ref_end in refs:
    if any(_overlap((ref_start, ref_end), (ps, pe)) > 0 for (ps, pe) in preds):
        tp_count += 1  # ANY overlap > 0 = TP (BINARY!)
```

**This is NEDC OVERLAP logic, NOT NEDC TAES scoring!**
- Any overlap > 0 → counts as 1 TP (binary)
- NEDC TAES scoring would give 0.5 TP for 50% overlap

---

## ✅ IMPLEMENTATION SUMMARY

| What | Implemented? | What We Use |
|------|--------------|-------------|
| **TAES metric (#1)** | ✅ YES | `calculate_taes()` → single score [0,1] |
| **Sensitivity @ FA rates** | ✅ YES | **NEDC OVERLAP scoring** (binary TP counting) |
| **NEDC TAES scoring (#2)** | ❌ NO | Not implemented (would need fractional TP) |

---

## 🎯 HOW TO COMPARE TO LITERATURE

### ✅ CORRECT Comparisons

**Our TAES metric (0.9658)**:
- Compare to: Other papers reporting "TAES score" as a single quality metric
- Do NOT compare to: Sensitivity values calculated with NEDC TAES scoring

**Our Sensitivity @ FA rates (35.9% @ 10 FA/24h)**:
- ✅ Compare to: **NEDC OVERLAP** results (e.g., SeizureTransformer: 45.63% @ 26.89 FA)
- ✅ Compare to: **SzCORE Event** results (more permissive)
- ❌ DO NOT compare to: **NEDC TAES** sensitivity results (different scoring!)

### ❌ INCORRECT Comparisons

**WRONG**:
```
Our TAES: 0.9658
vs
SeizureTransformer TAES sensitivity: 65.21%
```
These are COMPLETELY DIFFERENT metrics!

**CORRECT**:
```
Our Sensitivity @ 10 FA/24h: 35.9% (OVERLAP scoring)
vs
SeizureTransformer @ 26.89 FA/24h: 45.63% (OVERLAP scoring)
```

---

## 📊 Quick Reference Table

### When You See "TAES" in Literature

| Context | Meaning | Implementation |
|---------|---------|----------------|
| "TAES score: 0.85" | **Metric #1** | Our `calculate_taes()` |
| "Evaluated with NEDC TAES" | **Scoring system #2** | NOT implemented |
| "TAES sensitivity: 65%" | **Scoring system #2** | NOT implemented |
| "Using TAES metric" | Ambiguous! | Check paper carefully |

### Our Eval Logs (Example)

| Log Line | Meaning | Type |
|----------|---------|------|
| `TAES: 0.9658` | Quality score [0,1] | Metric #1 |
| `Sensitivity@10.0FA/24h: 0.3590` | Event detection @ 10 FA | OVERLAP scoring |
| `AUROC: 0.8654` | Discrimination ability | Standard metric |

---

## 🚨 WARNING FOR FUTURE DEVELOPMENT

### If We Ever Implement NEDC TAES Scoring

**Current sensitivity code** (false_alarm.py:144-146):
```python
# OVERLAP scoring (binary)
if any(_overlap((ref_start, ref_end), (ps, pe)) > 0 for (ps, pe) in preds):
    tp_count += 1  # Binary TP
```

**Would need to change to**:
```python
# TAES scoring (fractional)
for ref_start, ref_end in refs:
    ref_dur = ref_end - ref_start
    total_overlap = sum(_overlap((ref_start, ref_end), (ps, pe))
                       for (ps, pe) in preds)
    fractional_tp = min(1.0, total_overlap / ref_dur)  # Partial credit!
    tp_count += fractional_tp
```

**CRITICAL**: This would change ALL sensitivity values!

---

## 📚 References

### TAES Metric (#1)
- Shah et al. 2018: "The TUH EEG Seizure Corpus" (defines TAES score metric)
- Our implementation: `src/brain_brr/eval/metrics.py:79-146`

### NEDC TAES Scoring System (#2)
- Shah et al. 2021: "Validation of Temporal Scoring Metrics for Automatic Seizure Detection"
- Neureka 2020 Competition: Used TAES scoring as primary evaluation
- Temple NEDC Scorer v6.0.0: Implements TAES/OVERLAP/EPOCH modes

### Why This Matters
- REALISTIC_PERFORMANCE_TARGETS.md: Documents scoring system differences
- SeizureTransformer results vary 3.1× depending on scorer choice
- Critical for fair comparison to literature

---

## 💡 TL;DR FOR QUICK REFERENCE

**"TAES" can mean:**

1. **TAES Metric** (what we compute):
   - Single quality score [0,1]
   - Measures event alignment with partial credit
   - Our output: `TAES: 0.9658`

2. **NEDC TAES Scoring** (what literature uses):
   - Way to calculate sensitivity with partial credit
   - Gives different sensitivity values than OVERLAP scoring
   - We DO NOT implement this

**Our sensitivity uses NEDC OVERLAP scoring** (binary TP counting, NOT TAES scoring)

**Bottom Line**:
- ✅ Compare our sensitivity to **NEDC OVERLAP** results in literature
- ❌ Never compare our TAES score (0.9658) to TAES-scored sensitivity values
- 🚨 Always check which scorer papers use before comparing!

---

**Last Updated**: 2025-12-20
**See Also**: `docs/06-evaluation/REALISTIC_PERFORMANCE_TARGETS.md` (section on scoring systems)
