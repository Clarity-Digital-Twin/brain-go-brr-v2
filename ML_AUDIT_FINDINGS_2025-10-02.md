# Brain-Go-Brr V3 — ML Audit Findings (2025-10-02)

Senior auditor review of the current training stack. No code changes were made; this document records issues and follow-ups for coordinating parallel agents.

## ✅ Context
- Architecture: TCN + BiMamba + GNN (V3 only)
- Training runs in progress (Modal smoke + local full run); avoid destabilising edits.
- Audit scope: numerical stability, data loading, evaluation correctness, and pipeline technical debt.

## 🔴 P0 — Blocking Issues
- **Event timeline corruption** (`src/brain_brr/eval/metrics.py:427-498`, `src/brain_brr/data/datasets.py:200-233`)
  - Window-level datasets do not expose file IDs or absolute start indices. Evaluation flattens all window masks to a shared 0–60 s clock, so TAES, FA/24h thresholds, and sensitivity scores are computed on overlapping, misaligned time axes.
  - Consequence: validation metrics, early-stopping decisions, and clinical target tracking are currently unreliable.
  - Needed fix: propagate `record_id` and `window_start` from datasets through loaders; stitch timelines before calling `batch_*_to_events`; recompute metrics on true patient chronology.

## 🟠 P1 — High-Severity Risks
- **Cache I/O bottleneck** (`src/brain_brr/data/datasets.py:204-233`, `src/brain_brr/data/datasets.py:256-320`)
  - Every minibatch loads compressed NPZ files and decompresses whole window tensors to retrieve a single slice. This creates tens of MB of transient allocations per sample and is the dominant factor in 20 h Modal epochs.
  - Recommendation: migrate caches to shard-per-window or memory-mapped arrays; ensure loaders fetch only the requested window without rehydrating entire files.
- **Validation memory blow-up** (`src/brain_brr/train/loop.py:1024-1078`)
  - Validation collects every probability/label tensor in RAM before scoring. The dev split exceeds capacity on long runs, leading to process OOMs or swap thrash.
  - Recommendation: stream metrics per batch or per record; avoid concatenating the entire dataset.

## 🟡 P2 — Medium Priority
- **Sampler bootstrap load** (`src/brain_brr/train/loop.py:1778-1810`)
  - When the balanced manifest is absent, the fallback sampler probes 20 000 windows, triggering the same heavy NPZ decompressions before epoch 1. Once manifest generation is fixed, reduce this probe or rely on manifest stats to recover startup time.

## 📌 Observations & Follow-Ups
- Mixed-precision guardrails, gradient sanitisation, and warm-up scheduling look robust (no immediate action).
- GNN/adjacency stack already clamps eigenvectors and enforces LayerScale; no new stability risks spotted.
- Keep refactoring plan targeted: do not attempt loop.py split or cache rebuild until running jobs finish.

## 📋 Recommended Sequence (post-training window)
1. Fix evaluation timeline (P0).
2. Redesign cache format/loader access (P1) and introduce streaming validation (P1).
3. Trim sampler bootstrap + follow-up refactors once above issues land.

Document owner: Codex senior auditor (2025-10-02).
