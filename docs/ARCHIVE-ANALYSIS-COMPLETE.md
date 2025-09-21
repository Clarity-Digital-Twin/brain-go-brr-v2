# Complete Archive Analysis - What to Preserve

After exhaustive review of `/docs_archive/`, here's what's unique and worth preserving:

## Critical Content to ADD to Current Docs

### 1. **Seizure Type Frequency Data** (TUSZ_SEIZURE_TYPES_CRITICAL.md)
**Unique Value:** Empirical occurrence counts for all 8 seizure types
```
gnsz: 23,804 (46.5%)
fnsz: 19,000 (37.1%)
cpsz: 3,597 (7.0%)
absz: 2,507 (4.9%)
spsz: 942 (1.8%)
tcsz: 857 (1.7%)
tnsz: 410 (0.8%)
mysz: 44 (0.1%)
```
**Action:** Add to `01-data-pipeline/tusz-csv-parser.md`

### 2. **Cache Strategy OSS Warning** (P1_CACHE_STRATEGY_OSS_BLOCKER.md)
**Unique Value:** Detailed cache confusion matrix showing config conflicts
- smoke_test.yaml → cache/smoke (BUT NO data.cache_dir!)
- local.yaml → cache (CONFLICTS WITH EVERYTHING)
- OSS contributors will waste days debugging
**Action:** Add warning section to `04-reference/configs.md`

### 3. **SeizureTransformer Sampling Formula** (TUSZ_SAMPLING_STRATEGY.md)
**Unique Value:** Exact mathematical formula for balanced sampling
```
D = Dps ∪ D*fs ∪ D*ns
D*fs = 0.3 × |Dps| (full seizure)
D*ns = 2.5 × |Dps| (no seizure)
```
**Action:** Already in `01-data-pipeline/tusz-cache-sampling.md` but verify formula is explicit

### 4. **Architecture Decision Rationale** (ARCHITECTURE_CLARIFICATION.md)
**Unique Value:** Clear explanation why we CAN'T use SeizureTransformer weights
- Different architecture (Transformer → Bi-Mamba-2)
- Must train from scratch
- Can copy U-Net/ResCNN structure but not weights
**Action:** Add to `02-architecture/canonical-spec.md` introduction

### 5. **MNE-Scipy Hybrid Strategy** (PREPROCESSING_STRATEGY.md)
**Unique Value:** Specific implementation approach
- MNE for robust EDF I/O only
- scipy for all signal processing (exact SeizureTransformer parity)
- Code examples showing the hybrid approach
**Action:** Verify this is in `01-data-pipeline/data-io.md`

### 6. **Tesseract OCR Setup** (SETUP_NOTES.md)
**Unique Value:** System dependency for PDF processing
```bash
sudo apt-get install tesseract-ocr
python literature/pdf_to_markdown.py --force-ocr
```
**Action:** Add to `03-operations/deploy-local-wsl2.md` if PDF processing needed

### 7. **Channel Interpolation Plan** (INTERPOLATION_IMPLEMENTATION_PLAN.md)
**Unique Value:** Surgical fix for 249 files missing Fz/Pz
- MNE interpolation strategy
- Exact code changes needed
**Action:** Consider adding to troubleshooting if still relevant

## Content Already Preserved

✅ Zero-seizure cache catastrophe → `CRITICAL-ISSUES-RESOLVED.md`
✅ mysz seizure type crisis → `tusz-mysz-crisis.md`
✅ Mamba CUDA fallback → Multiple docs mention SEIZURE_MAMBA_FORCE_FALLBACK
✅ EDF header repair → `tusz-edf-repair.md`
✅ Channel canonicalization → `tusz-channels.md`
✅ WSL2 issues → `wsl2-troubleshooting.md`
✅ Modal deployment → `deploy-modal.md`

## Content to DISCARD

- All phase docs (replaced by components/)
- Duplicate bug reports (15+ versions of same issues)
- RESOLUTION_STATUS.md (captured in CRITICAL-ISSUES-RESOLVED.md)
- Test logs and temporary notes
- Archived READMEs and index files

## Final Recommendation

1. **Extract the 7 unique items above** into appropriate current docs
2. **Delete /docs_archive/** completely
3. **Keep only /docs/** as single source of truth

The archives served their purpose for historical context, but maintaining them creates confusion. The new structure with selective preservation is superior.