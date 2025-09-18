# Investigation: TUSZ Channels and Montages

**Date**: 2025-09-18
**Status**: 🔍 INVESTIGATING
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

## Current Known Issues

1. **File #47**: `aaaaahie_s030_t001.edf` missing Fz/Pz (01_tcp_ar montage)
2. **Canonicalization**: May not handle -REF suffix properly
3. **Unknown**: Full extent of channel variations

## Next Steps

1. Write comprehensive analysis script
2. Run on ALL TUSZ data (dev, train, eval)
3. Generate reports and statistics
4. Design solution based on data, not assumptions
5. Implement with confidence

## Questions for Analysis

- How many files total?
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