# NEDC Evaluation Pipeline - Implementation Overview

**Status**: DOCUMENTATION PHASE - Ready for AI audit before implementation

**Purpose**: High-level overview of the NEDC evaluation pipeline for official TUSZ test set scoring

**Date**: October 19, 2025

---

## 🚨 CRITICAL: Extend Existing Code, Don't Rewrite!

**EXISTING EVALUATION INFRASTRUCTURE**:
- ✅ `python -m src evaluate` CLI command (cli.py:305-413)
- ✅ `run_evaluation()` service (cli/services/evaluation.py:143-201)
- ✅ `export_csv_bi()` function (events/export.py:15-52)
- ✅ `validate_epoch()` inference (train/val_step.py:375-584)

**WHAT WE NEED TO ADD**:
- NEDC integration (Component 2: NEDCScorer)
- Extend eval CLI to support NEDC scoring
- NOT rewrite existing converters/evaluation!

---

## What This Is

NEDC evaluation pipeline to get **official, publication-ready metrics** on TUSZ eval/test set using Temple University's NEDC v6.0.0 scorer.

**The Problem**: We have baseline.pt (28.01% dev sensitivity@10FA) but NO official test metrics yet!

**The Solution**: 3-component pipeline (2 NEW components + extend 1 existing)

---

## Architecture (2 NEW + 1 EXTEND)

```
┌────────────────────────────────────────────────────────────┐
│                  NEDC Evaluation Pipeline                  │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Component 1: CSV_BI Export (✅ ALREADY EXISTS!)          │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ export_csv_bi() - events/export.py:15-52           │ │
│  │ Predictions → CSV_BI with header (# version...)    │ │
│  └──────────────────────────────────────────────────────┘ │
│                          ↓                                 │
│  Component 2: NEDCScorer (🆕 NEW - nedc_wrapper.py)       │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ Direct Python import of nedc-bench                  │ │
│  │ BetaPipeline.evaluate() + FA-sensitivity compute    │ │
│  └──────────────────────────────────────────────────────┘ │
│                          ↓                                 │
│  Component 3: Evaluation CLI (🔧 EXTEND existing!)        │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ Add --nedc-score flag to existing evaluate command │ │
│  │ Wraps run_evaluation() + NEDCScorer                 │ │
│  └──────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────┘
```

---

## Files to Create

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| ~~`src/brain_brr/eval/format_converter.py`~~ | ~~150~~ | ~~Convert .npy → CSV_BI~~ | ✅ EXISTS (events/export.py) |
| `src/brain_brr/eval/nedc_wrapper.py` | ~100 | Direct Python NEDC integration | 🆕 NEW |
| ~~`src/brain_brr/eval/evaluator.py`~~ | ~~200~~ | ~~End-to-end orchestrator + CLI~~ | ✅ EXISTS (cli/services/evaluation.py) |
| ~~`tests/unit/eval/test_format_converter.py`~~ | ~~400~~ | ~~10 unit tests~~ | ✅ EXISTS |
| `tests/unit/eval/test_nedc_wrapper.py` | ~350 | 8 unit tests | 🆕 NEW |
| `tests/integration/eval/test_nedc_integration.py` | ~150 | NEDC integration test | 🆕 NEW |

**Total new code**: ~600 lines (MUCH less than originally planned!)

---

## Implementation Timeline (2 Weeks - REVISED)

### Week 1: NEDCScorer + Preprocessing
- **Day 1**: Extend build-cache CLI to support eval split (cli.py:207)
- **Day 2-3**: Preprocess TUSZ eval set to cache (2-4 hours + validation)
- **Day 4-5**: Write 8 unit tests for NEDCScorer (TDD)

**Success**: Eval cache built, NEDCScorer tests passing

### Week 2: NEDCScorer Implementation + Integration
- **Day 1-2**: Implement NEDCScorer with BetaPipeline import (~100 lines)
- **Day 3**: Extend evaluate CLI with --nedc-score flag
- **Day 4**: Integration test with nedc-bench on dev set
- **Day 5**: Run baseline.pt on eval set, get official NEDC metrics

**Success**: Official test metrics obtained, baseline documented

**MAJOR CHANGE**: Reduced from 4 weeks (~1,450 lines) to 2 weeks (~600 lines) by reusing existing infrastructure!

---

## What We Get

### Publication-Ready Results
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
    "taes_score": 0.67,
    "f1": 0.31,
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

### Comparison Table
```markdown
| Model | Dev Sens@10FA | Test Sens@10FA | Dev-Test Gap | F1 | Notes |
|-------|---------------|----------------|--------------|----|----|
| Baseline | 28.01% | 24.3% | 3.7% | 0.31 | Overfitting |
| Exp1 (reg) | TBD | TBD | TBD | TBD | Stronger regularization |
| Literature [Shah] | - | 89% | - | - | SOTA benchmark |
```

---

## Why This Matters

**Without NEDC scoring**:
- ❌ Can't compare to literature (every paper uses NEDC)
- ❌ Non-standard metrics (people won't trust)
- ❌ Can't publish (journals require NEDC scores)

**With NEDC scoring**:
- ✅ Apples-to-apples comparison with ALL papers
- ✅ Trusted, validated metrics (NEDC v6.0.0 is gold standard)
- ✅ Publication-ready results
- ✅ Clinical utility assessment (FA/24h is what clinicians care about!)

---

## Related Documentation

**Detailed specs** (for implementation):
1. `EVALUATION_01_QUESTIONS_RESOLVED.md` - All answered questions
2. `EVALUATION_02_COMPONENT_CSVBI_CONVERTER.md` - Format converter specs
3. `EVALUATION_03_COMPONENT_NEDC_SCORER.md` - NEDC wrapper specs
4. `EVALUATION_04_COMPONENT_MODEL_EVALUATOR.md` - Orchestrator specs
5. `EVALUATION_05_TESTING_REQUIREMENTS.md` - All test specifications

**Original planning doc**: `NEXT_STEPS.md` (Section: "NEDC Evaluation Pipeline")

---

## Quick Start (After Implementation)

```bash
# Evaluate baseline on eval set with NEDC scoring
python -m src evaluate \
  --checkpoint results/local_fla_training/checkpoints/best.pt \
  --split eval \
  --nedc-score \
  --output results/eval_baseline/

# View results
cat results/eval_baseline/metrics/eval_overlap_metrics.json
```

**NOTE**: Uses EXISTING `python -m src evaluate` CLI (cli.py:305) with new `--nedc-score` flag, NOT a new evaluator.py module!

---

## Success Criteria

**Phase 1 Complete** (NEDCScorer + CLI Extension):
- [ ] NEDCScorer: 8/8 unit tests pass, 90% coverage
- [ ] NEDC integration test passes with nedc-bench
- [ ] Extend `python -m src evaluate` with `--nedc-score` flag
- [ ] Extend `python -m src build-cache` with `--split eval`

**Phase 2 Complete** (Production Evaluation):
- [ ] Eval cache built (cache/tusz_mmap/eval/)
- [ ] Baseline evaluated on eval set with NEDC scoring
- [ ] Official NEDC metrics obtained and documented
- [ ] Dev-test gap analysis complete

---

**Next Step**: AI agent audits all 5 detailed docs for accuracy before implementation begins.
