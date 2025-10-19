# NEDC Evaluation Pipeline - Implementation Overview

**Status**: DOCUMENTATION PHASE - Ready for AI audit before implementation

**Purpose**: High-level overview of the NEDC evaluation pipeline for official TUSZ test set scoring

**Date**: October 19, 2025

---

## What This Is

NEDC evaluation pipeline to get **official, publication-ready metrics** on TUSZ eval/test set using Temple University's NEDC v6.0.0 scorer.

**The Problem**: We have baseline.pt (28.01% dev sensitivity@10FA) but NO official test metrics yet!

**The Solution**: 3-component pipeline to convert our predictions → CSV_BI format → NEDC scorer → official metrics

---

## Architecture (3 Components)

```
┌────────────────────────────────────────────────────────────┐
│                  NEDC Evaluation Pipeline                  │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Component 1: CSVBIConverter (format_converter.py)        │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ .npy predictions → CSV_BI text files                │ │
│  │ Reuses existing batch_probs_to_events()             │ │
│  └──────────────────────────────────────────────────────┘ │
│                          ↓                                 │
│  Component 2: NEDCScorer (nedc_wrapper.py)                │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ Direct Python import of nedc-bench                  │ │
│  │ NO Docker! Just sys.path.insert()                   │ │
│  └──────────────────────────────────────────────────────┘ │
│                          ↓                                 │
│  Component 3: ModelEvaluator (evaluator.py)               │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ End-to-end: load checkpoint → inference → score     │ │
│  │ Publication-ready JSON + markdown tables            │ │
│  └──────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────┘
```

---

## Files to Create

| File | Lines | Purpose |
|------|-------|---------|
| `src/brain_brr/eval/format_converter.py` | ~150 | Convert .npy → CSV_BI |
| `src/brain_brr/eval/nedc_wrapper.py` | ~100 | Direct Python NEDC integration |
| `src/brain_brr/eval/evaluator.py` | ~200 | End-to-end orchestrator + CLI |
| `tests/unit/eval/test_format_converter.py` | ~400 | 10 unit tests |
| `tests/unit/eval/test_nedc_wrapper.py` | ~350 | 8 unit tests |
| `tests/integration/eval/test_evaluator.py` | ~250 | 4 integration tests |

**Total new code**: ~1,450 lines

---

## Implementation Timeline (4 Weeks)

### Week 1: CSVBIConverter
- **Day 1-2**: Write all 10 unit tests (TDD)
- **Day 3-4**: Implement CSVBIConverter
- **Day 5**: Test on real dev predictions, validate CSV_BI format

**Success**: 10/10 tests pass, dev predictions convert successfully

### Week 2: NEDCScorer
- **Day 1-2**: Write all 8 unit tests
- **Day 3-4**: Implement NEDCScorer with direct Python import
- **Day 5**: Integration test with nedc-bench sample data

**Success**: 8/8 tests pass, integration with nedc-bench works

### Week 3: ModelEvaluator
- **Day 1-2**: Write 4 integration tests (with mocking)
- **Day 3-5**: Implement ModelEvaluator + CLI
- **Day 6-7**: Full end-to-end test on dev set

**Success**: Full pipeline works, CLI functional

### Week 4: Production Evaluation
- **Baseline**: Evaluate baseline.pt on eval set, get official metrics
- **Exp1**: (When ready) Evaluate Exp1 on eval set, compare to baseline
- **Documentation**: Update experiment tracking, generate publication tables

**Success**: Official test metrics obtained, baseline documented

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
# Evaluate baseline on eval set
python -m src.brain_brr.eval.evaluator \
  --checkpoint results/local_fla_training/checkpoints/best.pt \
  --split eval \
  --algorithm overlap \
  --output results/eval_baseline/

# View results
cat results/eval_baseline/metrics/eval_overlap_metrics.json
```

---

## Success Criteria

**Phase 1 Complete**:
- [ ] CSVBIConverter: 10/10 tests pass, 95% coverage
- [ ] NEDCScorer: 8/8 tests pass, 90% coverage
- [ ] ModelEvaluator: 4/4 tests pass, end-to-end works

**Phase 2 Complete**:
- [ ] Baseline evaluated on eval set
- [ ] Official NEDC metrics obtained
- [ ] Results documented and reproducible
- [ ] Publication tables generated

---

**Next Step**: AI agent audits all 5 detailed docs for accuracy before implementation begins.
