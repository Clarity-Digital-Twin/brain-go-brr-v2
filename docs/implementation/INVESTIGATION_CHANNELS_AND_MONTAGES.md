# [ARCHIVED] Investigation: TUSZ Channels and Montages

**Date**: 2025-09-18
**Status**: Archived — see docs/references/TUSZ_CHANNELS.md and docs/implementation/PREPROCESSING_STRATEGY.md
**Goal**: Understand the ENTIRE TUSZ dataset before implementing any solution

## Why This Investigation?

We keep hitting channel issues reactively (Fz/Pz missing in some files). Instead of patching individual files, we need to understand:
1. What channels are ACTUALLY in each montage type
2. How consistent the channel sets are
3. Whether ALL files can provide our required 19 channels
4. What transformations are needed for each montage

## Key Questions to Answer

### 1. Channel Availability
- [ ] Do ALL files in TUSZ have our required 19 channels?
- [ ] What variations exist across montage types (01_tcp_ar, 02_tcp_le, 03_tcp_ar_a)?
- [ ] Are there files genuinely missing channels, or is it a naming/reference issue?

### 2. Montage Specifics
- [ ] What does each montage type mean?
  - `01_tcp_ar`: Average Reference
  - `02_tcp_le`: Linked Ears
  - `03_tcp_ar_a`: Average Reference Alternative
- [ ] How do references affect channel names (-REF, -LE, -AR)?
- [ ] Can we reliably convert between montages?

### 3. Dataset Coverage
- [ ] How many files in dev/train/eval?
- [ ] Distribution of montage types?
- [ ] Any systematic patterns in missing channels?

### 4. Production Implications
- [ ] Will inference crash on eval set with different channels?
- [ ] Should we handle montages differently?
- [ ] Do we need montage-specific preprocessing?

## Comprehensive Analysis Plan

### Phase 1: Full Dataset Scan
```python
# Analyze EVERY file in TUSZ to collect:
- File path
- Montage type
- Total channels
- Channel names (raw)
- Which of our 19 are present/missing
- Sample rate
- Duration
- Any errors
```

### Phase 2: Statistical Analysis
- Channel frequency across files
- Montage distribution
- Missing channel patterns
- Correlation between montage and missing channels

### Phase 3: Deep Dive on Problem Files
- Investigate files missing Fz/Pz specifically
- Check if they have alternative names
- Verify if data can be recovered/interpolated

## Data Collection Script Requirements

1. **Comprehensive**: Scan ALL files (dev, train, eval)
2. **Robust**: Don't crash on bad files, log errors
3. **Detailed**: Capture all channel variations
4. **Efficient**: Use multiprocessing for speed
5. **Persistent**: Save results to CSV/JSON for analysis

## Expected Outputs

1. **tusz_analysis.csv**: Full dataset inventory
2. **channel_statistics.json**: Aggregated statistics
3. **problem_files.txt**: List of files with issues
4. **montage_report.md**: Human-readable findings

## Implementation Strategy

Based on findings, we'll decide:

### Option A: Universal Channel Handling
- If 95%+ files have all channels → Fix canonicalization
- Handle all name variations (-REF, -LE, -AR)

### Option B: Montage-Specific Processing
- If channels vary by montage → Create montage adapters
- Transform to common space before processing

### Option C: Intelligent Fallback
- If some files genuinely lack channels → Interpolation strategy
- Use neighboring channels to estimate missing ones

## Critical Success Criteria

1. **100% file coverage**: Process ALL TUSZ files without skipping
2. **No crashes**: Robust handling of all variations
3. **Clear documentation**: Understand WHY each decision was made
4. **Production ready**: Solution works for train AND inference

## 🔬 ANALYSIS COMPLETE: CRITICAL FINDINGS

### Dataset Statistics
- **Total Files Analyzed**: 7,364 files
- **Files with ALL 19 channels**: 7,115 (96.6%)
- **Files MISSING Fz/Pz**: 249 (3.4%)
- **Files with errors**: 0 (robust analysis!)

### Montage Breakdown

| Montage | Total Files | Has All 19 | Success Rate | Missing Pattern |
|---------|------------|------------|--------------|-----------------|
| 02_tcp_le (Linked Ears) | 547 | 547 | **100.0%** | NONE! Perfect! |
| 01_tcp_ar (Average Ref) | 5,508 | 5,295 | 96.1% | Fz: 213, Pz: 213 |
| 03_tcp_ar_a (Avg Ref Alt) | 1,309 | 1,273 | 97.2% | Fz: 36, Pz: 36 |

### The Missing Channel Pattern

**CRITICAL DISCOVERY**: Files missing Fz/Pz have a specific pattern:
- They have `EEG CZ-REF` (center midline)
- They DON'T have `EEG FZ-REF` (frontal midline)
- They DON'T have `EEG PZ-REF` (parietal midline)
- **This is NOT a naming issue - these channels were NOT RECORDED**

### Why This Happened (Hypothesis)

Some EEG recordings in TUSZ used a modified 10-20 setup where:
1. **Cz was used as the reference electrode** (common in clinical EEG)
2. When Cz is the reference, you can't record FROM Cz
3. But TUSZ re-referenced the data, recovering Cz but NOT Fz/Pz
4. This affects ~3.4% of recordings, primarily in Average Reference montages

### Distribution Across Datasets

```
Dev set:   1,832 files → ~177 missing Fz/Pz (estimated)
Train set: 4,667 files → ~ 35 missing Fz/Pz (estimated)
Eval set:    865 files → ~ 37 missing Fz/Pz (estimated)
```

### Patient/Session Clustering

Many problem files cluster by patient/session:
- `aaaaahie` patient has MANY sessions missing Fz/Pz
- `aaaaamnk`, `aaaaampk`, `aaaaamqq`, `aaaaamva` also affected
- Suggests equipment/protocol differences at recording sites

## 🧠 Deep Technical Analysis

### The Midline Electrode Problem

The midline electrodes (Fz, Cz, Pz) form the sagittal line of the 10-20 system:
- **Fz (Frontal Zero)**: Frontal lobe, executive function
- **Cz (Central Zero)**: Motor cortex, central processing
- **Pz (Parietal Zero)**: Parietal lobe, sensory integration

When 249 files are missing Fz/Pz but HAVE Cz, this tells us:
1. These aren't random missing channels
2. It's a systematic recording setup difference
3. The data is REAL - just incomplete

### Spatial Implications for Seizure Detection

**Why this matters for our model:**
- Seizures often have **frontal** (Fz) or **parietal** (Pz) onset
- Missing these means missing potential seizure focus regions
- BUT: Seizures propagate - neighboring channels see activity
- F3/F4 can partially compensate for missing Fz
- P3/P4 can partially compensate for missing Pz

### Signal Processing Implications

Average Reference (AR) montage math:
```
V_channel = V_raw - (1/N) * Σ(all_channels)
```

When Fz/Pz are missing:
- The average reference is computed WITHOUT them
- This slightly biases the reference
- May affect amplitude calibration
- Could impact model predictions if not handled

## 🎯 Solution Analysis (DEEP DIVE)

### Option 1: SKIP Strategy (Conservative)
**Approach**: Exclude 249 files from training

**PROS:**
- ✅ Simplest implementation
- ✅ Clean, consistent data
- ✅ No artificial signals
- ✅ 7,115 files is still massive

**CONS:**
- ❌ Loses 3.4% of data
- ❌ Eval set WILL have these files
- ❌ Model never learns to handle missing channels
- ❌ **FATAL**: Production inference will crash on these files!

**Verdict**: ⚠️ **Not production-ready**

---

### Option 2: INTERPOLATION Strategy (Sophisticated)
**Approach**: Estimate Fz/Pz from spatial neighbors

**Mathematical Basis:**
```python
# Spherical spline interpolation
Fz_estimated = w1*F3 + w2*F4 + w3*F7 + w4*F8 + w5*Cz
Pz_estimated = w1*P3 + w2*P4 + w3*O1 + w4*O2 + w5*Cz

# Weights from spatial proximity (10-20 distances)
```

**PROS:**
- ✅ Keeps ALL 7,364 files
- ✅ Physically meaningful
- ✅ MNE has built-in interpolation
- ✅ Clinically accepted practice

**CONS:**
- ❌ Interpolated != real signal
- ❌ Adds computational overhead
- ❌ May introduce artifacts
- ❌ Model might overfit to interpolation patterns

**Implementation:**
```python
# Using MNE's spherical spline interpolation
raw.set_montage('standard_1020')
raw.interpolate_bads(reset_bads=True)
```

**Verdict**: ✅ **Best for accuracy**

---

### Option 3: ZERO-FILL Strategy (Pragmatic)
**Approach**: Add Fz=0, Pz=0 for missing channels

**PROS:**
- ✅ Simple to implement
- ✅ Model learns these are uninformative
- ✅ No false signal introduction
- ✅ Consistent tensor shapes

**CONS:**
- ❌ Loses spatial information
- ❌ Zero is not physiologically meaningful
- ❌ Might confuse model (zero ≠ missing)
- ❌ Breaks some preprocessing (z-score of 0?)

**Verdict**: ⚠️ **Quick hack, not recommended**

---

### Option 4: MASKING Strategy (Advanced)
**Approach**: Use attention masking for missing channels

**PROS:**
- ✅ Explicitly tells model which channels are missing
- ✅ Transformer/Mamba can handle masked inputs
- ✅ Most "honest" approach
- ✅ Generalizes to any missing pattern

**CONS:**
- ❌ Requires architecture changes
- ❌ Increases model complexity
- ❌ More training overhead
- ❌ May not work with CNN encoders

**Implementation sketch:**
```python
# Add channel mask to model
channel_mask = torch.ones(19)
channel_mask[[5, 18]] = 0  # Fz is index 5, Pz is index 18
x = x * channel_mask.unsqueeze(0).unsqueeze(-1)
```

**Verdict**: 🎯 **Best for research, complex for production**

---

### Option 5: HYBRID Strategy (Recommended)
**Approach**: Interpolate + confidence weighting

1. Interpolate missing channels
2. Add metadata flag for interpolated channels
3. Model learns to weight them less

**PROS:**
- ✅ Best of both worlds
- ✅ Model aware of interpolation
- ✅ Graceful degradation
- ✅ Production ready

**CONS:**
- ❌ Most complex implementation
- ❌ Requires careful validation

## 📊 Quantitative Impact Analysis

### Training Set Impact
```
Total training files: 5,465 (train + dev)
Files missing Fz/Pz: ~212 (3.9%)
Windows affected: ~212 * 50 = 10,600 windows
Total windows: ~273,250
Impact: 3.9% of training data
```

### Model Architecture Considerations

**U-Net Encoder**:
- Spatial convolutions might struggle with missing channels
- Interpolation helps maintain spatial structure

**Bi-Mamba**:
- Sequential processing can handle missing data
- Masking would work well here

**ResCNN**:
- Residual connections might propagate missing info
- Zero-filling would break residual math

## 🚨 Production Implications

### Inference Scenarios

1. **Hospital A**: Has all 19 channels ✅
2. **Hospital B**: Uses same setup as problem files (missing Fz/Pz) 🚨
3. **Hospital C**: Has DIFFERENT missing channels 💥

### We MUST handle:
- Missing Fz/Pz (we know about these)
- Potentially OTHER missing channels
- Different montages in production
- Real-time streaming with dropouts

## 📝 Recommendations

### Immediate (for training):
1. **Implement interpolation** for the 249 files
2. **Log which files** were interpolated
3. **Validate** interpolation quality
4. **Test** model performance on interpolated vs real

### Long-term (for production):
1. **Build robust channel handler** that can:
   - Detect missing channels
   - Interpolate if possible
   - Gracefully degrade if not
   - Alert on quality issues
2. **Train multiple models**:
   - One on perfect data
   - One on interpolated data
   - Ensemble them

### Testing Strategy:
1. **Holdout the 249 files** as special test set
2. Train on good files only
3. Test if model can handle interpolated files
4. Measure performance drop

## 🔬 Next Experiments

1. **Load one problem file** and visualize the channels
2. **Test MNE interpolation** quality
3. **Compare** model performance with/without problem files
4. **Analyze** if seizures occur in problem files

## 💭 Deep Questions to Consider

1. **Why do Average Reference montages have this issue more?**
   - TCP_AR: 3.9% missing
   - TCP_AR_A: 2.8% missing
   - TCP_LE: 0% missing

2. **Is there a clinical reason some recordings skip Fz/Pz?**
   - Surgical sites?
   - Bandaging/wounds?
   - Equipment limitations?

3. **Do the problem files have seizures?**
   - If yes → MUST include them
   - If no → Could skip for training

4. **Should we contact TUSZ maintainers?**
   - Report the issue
   - Ask for guidance
   - Check if it's documented

## 🎬 Action Plan

### Phase 1: Understand (TODAY)
- [x] Run comprehensive analysis
- [x] Document findings
- [ ] Visualize problem files
- [ ] Test interpolation quality

### Phase 2: Decide (TOMORROW)
- [ ] Choose strategy (likely interpolation)
- [ ] Implement solution
- [ ] Validate on sample files
- [ ] Test training pipeline

### Phase 3: Execute (THIS WEEK)
- [ ] Full training run
- [ ] Monitor for issues
- [ ] Compare metrics
- [ ] Document solution

## 🎯 CRITICAL DECISION POINT

### The Reality Check

We discovered that **96.6% of TUSZ files are perfect** and only **3.4% have missing Fz/Pz**.

**The Core Question**: Do we optimize for the 96.6% or accommodate the 3.4%?

### Option Comparison Matrix

| Strategy | Dev Time | Risk | Data Coverage | Production Ready | Recommendation |
|----------|----------|------|---------------|------------------|----------------|
| Skip files | 1 hour | HIGH (crashes on bad files) | 96.6% | ❌ No | Don't do this |
| Interpolate | 4 hours | LOW | 100% | ✅ Yes | **DO THIS** |
| Zero-fill | 2 hours | MEDIUM | 100% | ⚠️ Maybe | Backup option |
| Masking | 8+ hours | MEDIUM | 100% | ⚠️ Complex | Future work |
| Hybrid | 6 hours | LOW | 100% | ✅ Yes | Best but complex |

### The Winning Strategy: INTERPOLATION

**Why Interpolation Wins:**
1. **MNE already has it**: `raw.interpolate_bads()`
2. **Clinically accepted**: Standard practice in EEG analysis
3. **Maintains spatial structure**: CNNs can still work
4. **No architecture changes**: Drop-in solution
5. **Production ready**: Handles any future bad channels

### Implementation Pseudocode

```python
def load_edf_file_with_interpolation(edf_path: Path) -> tuple[np.ndarray, float]:
    # Load file
    raw = mne.io.read_raw_edf(edf_path, preload=True)

    # Canonicalize channels
    raw = canonicalize_channel_names(raw)

    # Check for missing required channels
    missing = set(REQUIRED_CHANNELS) - set(raw.ch_names)

    if missing:
        # Mark missing channels as bad
        raw.info['bads'] = list(missing)

        # Set montage for spatial info
        raw.set_montage('standard_1020', on_missing='ignore')

        # Interpolate missing channels
        raw.interpolate_bads(reset_bads=True)

        # Log this file as interpolated
        logger.warning(f"Interpolated {missing} for {edf_path}")

    # Continue with normal preprocessing
    return preprocess(raw)
```

### Validation Requirements

Before we proceed, we MUST:
1. **Test interpolation** on one problem file
2. **Visualize** the interpolated vs real channels
3. **Verify** the signal looks reasonable
4. **Check** if model can handle it

### The 249 Files List

We now have a CSV with all 249 problem files. We should:
1. Create a `problem_files.txt` for tracking
2. Test our solution on these specifically
3. Monitor their performance separately

### Final Architecture Decision

Our current architecture (U-Net + ResCNN + Bi-Mamba) will work with interpolation:
- **U-Net**: Handles spatial patterns, interpolation preserves these
- **ResCNN**: Residual connections work with interpolated data
- **Bi-Mamba**: Sequential processing is robust to interpolation

### What We Do NOW

1. **Clean up** analysis scripts from root ✅
2. **Document** deeply (THIS FILE) ✅
3. **Implement** interpolation in `load_edf_file()`
4. **Test** on the 249 problem files
5. **Train** with confidence

## 🏁 CONCLUSION

**Finding**: 3.4% of TUSZ files are missing Fz/Pz channels (249 files)

**Root Cause**: Different recording setups, likely Cz as reference

**Solution**: Spherical spline interpolation via MNE

**Impact**: Minimal - interpolation is standard practice

**Timeline**: 4 hours to implement and test

**Confidence**: HIGH - this will work

---

*This investigation revealed a manageable issue with a clear solution. The TUSZ dataset is 96.6% perfect, and we can handle the remaining 3.4% with standard EEG interpolation techniques.*
- How many unique channel configurations?
- What % have all 19 channels?
- What are the most common missing channels?
- Is there a pattern by montage type?
- Can we derive missing channels from available ones?

## Principles

1. **Data-driven**: Let the data tell us what to do
2. **No assumptions**: Verify everything
3. **Complete solution**: Handle ALL cases, not just common ones
4. **Future-proof**: Solution should work for new TUSZ data

---

This investigation will give us the complete picture before we implement ANYTHING.
