# NEDC Evaluation Pipeline - Overview

**Status**: Ready for implementation (October 19, 2025)

**Purpose**: Get **official, publication-ready metrics** on TUSZ eval set using Temple University's NEDC v6.0.0 scorer

---

## The Problem

We have baseline.pt trained (28.01% dev sensitivity@10FA) but **NO official test metrics**.

**Without NEDC scoring**:
- ❌ Can't compare to literature (every seizure detection paper uses NEDC)
- ❌ Non-standard metrics (reviewers won't trust)
- ❌ Can't publish (journals require NEDC v6.0.0 scores)

**With NEDC scoring**:
- ✅ Apples-to-apples comparison with ALL papers
- ✅ Trusted, validated metrics (NEDC v6.0.0 is gold standard)
- ✅ Publication-ready results
- ✅ Clinical utility assessment (FA/24h is what clinicians care about)

---

## The Solution

**ONE new component** + extend existing CLI (~150 lines total)

```
┌────────────────────────────────────────────────────────────┐
│                  NEDC Evaluation Pipeline                  │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Step 1: CSV_BI Export (✅ ALREADY EXISTS!)               │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ export_csv_bi() - events/export.py:15-52          │ │
│  │ Predictions → CSV_BI with # headers               │ │
│  └──────────────────────────────────────────────────────┘ │
│                          ↓                                 │
│  Step 2: NEDC Scoring (🆕 NEW - nedc_wrapper.py)          │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ NEDCScorer - File-level loop over CSV_BI pairs    │ │
│  │ BetaPipeline.evaluate_overlap() per pair           │ │
│  │ Accumulate counts → compute metrics                │ │
│  │ ~100 lines, 9 tests                                │ │
│  └──────────────────────────────────────────────────────┘ │
│                          ↓                                 │
│  Step 3: Evaluation Service (🔧 EXTEND existing!)         │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ Add NEDC scoring to run_evaluation() service      │ │
│  │ Integrates NEDCScorer with EvaluationRequest       │ │
│  │ ~50 lines extension                                │ │
│  └──────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────┘
```

**Total new code**: ~150 lines (NOT 1,450 lines - we reuse existing infrastructure!)

---

## What We'll Get

### Publication-Ready Metrics (JSON)

```json
{
  "experiment": "baseline",
  "checkpoint": "best.pt (epoch 9)",
  "split": "eval",
  "algorithm": "NEDC v6.0.0 Overlap",
  "metrics": {
    "sensitivity_at_10FA_24h": 24.3,
    "sensitivity_at_5FA_24h": 21.8,
    "sensitivity_at_1FA_24h": 15.2,
    "f1": 0.31,
    "precision": 0.42,
    "recall": 0.24,
    "tp": 145,
    "fp": 198,
    "fn": 456
  },
  "comparison_to_dev": {
    "dev_sensitivity_10FA": 28.01,
    "test_sensitivity_10FA": 24.3,
    "gap": 3.71,
    "interpretation": "Overfitting hurts generalization by 3.7%"
  }
}
```

### Comparison Table (Markdown)

```markdown
| Model | Dev Sens@10FA | Test Sens@10FA | Dev-Test Gap | F1 | Notes |
|-------|---------------|----------------|--------------|----|----|
| Baseline | 28.01% | 24.3% | 3.7% | 0.31 | Overfitting |
| Exp1 (reg) | 26.5% | 25.8% | 0.7% | 0.33 | Better generalization |
| Literature [Shah] | - | 89% | - | - | SOTA benchmark |
```

---

## Implementation Timeline (2 Weeks)

### Week 1: Preprocessing + Tests (TDD)
- **Day 1**: Extend build-cache to support `--split eval` (change line 207 of cli.py)
- **Day 2-3**: Preprocess TUSZ eval set to cache (2-4 hours + validation)
- **Day 4-5**: Write 8 unit tests + 1 integration test for NEDCScorer (TDD)

**Deliverable**: Eval cache built, 9 NEDCScorer tests written (all failing)

### Week 2: NEDCScorer Implementation + Integration
- **Day 1-2**: Implement NEDCScorer (~100 lines) - make tests pass
- **Day 3**: Extend run_evaluation() service with NEDC scoring (~50 lines)
- **Day 4**: Integration test with nedc-bench on dev set
- **Day 5**: Run baseline.pt on eval set, get official NEDC metrics

**Deliverable**: Official test metrics obtained, baseline documented

---

## Quick Start (After Implementation)

### Step 1: Preprocess eval set (Week 1, Day 2-3)

```bash
# FIRST: Extend CLI to support eval split (change cli.py line 207)
# FROM: type=click.Choice(["train", "dev"])
# TO:   type=click.Choice(["train", "dev", "eval"])

# THEN: Preprocess eval set
python -m src build-cache \
  --data-dir data_ext4/tusz/edf/eval/ \
  --cache-dir cache/tusz_mmap/eval/ \
  --split eval
```

### Step 2: Run evaluation with NEDC scoring (Week 2, Day 5)

```bash
# After implementing NEDCScorer + extending run_evaluation()
python -m src evaluate \
  results/local_fla_training/checkpoints/best.pt \
  data_ext4/tusz/edf/eval/ \
  --output-json results/eval_baseline/metrics.json

# NEDCScorer will be called internally from run_evaluation() service
# Results saved to JSON with NEDC metrics
```

**NOTE**: Uses EXISTING Click-based CLI (positional args `checkpoint_path` and `data_path`)

---

## Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `src/brain_brr/eval/nedc_wrapper.py` | ~100 | NEDCScorer class - File-level NEDC integration |
| `tests/unit/eval/test_nedc_wrapper.py` | ~350 | 8 unit tests for NEDCScorer |
| `tests/integration/eval/test_nedc_integration.py` | ~150 | Integration test with nedc-bench |

**Total**: ~600 lines (implementation + tests)

---

## Success Criteria

**Phase 1 Complete** (NEDCScorer + Service Integration):
- [ ] 8/8 NEDCScorer unit tests pass
- [ ] 1/1 NEDC integration test passes
- [ ] Coverage ≥ 90% for nedc_wrapper.py
- [ ] run_evaluation() service extended with NEDC scoring
- [ ] `python -m src build-cache --split eval` works

**Phase 2 Complete** (Production Evaluation):
- [ ] Eval cache built at `cache/tusz_mmap/eval/`
- [ ] Baseline evaluated on eval set with NEDC scoring
- [ ] Official NEDC metrics saved to JSON
- [ ] Dev-test gap analysis documented

---

## Output Directory Structure

```
results/eval_baseline/
├── metrics.json              # Main evaluation metrics (includes NEDC scores)
└── predictions/
    └── eval/
        ├── aaaaaaaq_s006_t000_probs.npy
        ├── aaaaaaaq_s006_t000_labels.npy
        └── ... (~2000 files)
```

**NOTE**: CSV_BI files will be generated in memory during NEDC scoring, not saved to disk (unless needed for debugging)

---

## Related Documentation

- **`NEDC_IMPLEMENTATION_GUIDE.md`** - Step-by-step TDD implementation (START HERE for coding)
- **`NEDC_REFERENCE.md`** - CSV_BI format, dataclasses, error tables (reference during coding)

---

**Next Step**: Read `NEDC_IMPLEMENTATION_GUIDE.md` for TDD implementation phases with ACTUAL codebase patterns
