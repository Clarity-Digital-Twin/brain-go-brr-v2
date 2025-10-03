# Brain-Go-Brr V3 — ML Audit Findings (2025-10-02)

Senior auditor review of the current training stack. No code changes were made; this document records issues and follow-ups so parallel agents stay aligned.

## ✅ Context
- Architecture: TCN + BiMamba + GNN (V3 only)
- Long-running Modal + local trainings are in flight; avoid destabilising edits.
- Audit scope: numerical stability, data loading, evaluation correctness, and pipeline technical debt.

## 🔴 P0 — Blocking Issues
- **Event timeline corruption** (`src/brain_brr/eval/metrics.py:427-498`, `src/brain_brr/data/datasets.py:200-233`)
  - Datasets return `(window, label)` only; evaluation assumes all windows belong to a single continuous recording: `total_duration_s = (n_windows - 1) * 10.0 + 60.0` regardless of file boundaries.
  - Result: TAES, FA/24 h, and sensitivity collapse windows from different patients onto the same 0–60 s axis. Clinical metrics, early-stopping signals, and threshold sweeps are therefore invalid even though gradient updates remain correct.
  - Fix: propagate `record_id`/`window_start` (or equivalent metadata) through the dataloaders, stitch per-record timelines prior to calling `batch_*_to_events`, and recompute metrics on the true chronology. Retraining will be required once corrected.

## 🟠 P1 — High-Severity Risks (Nuanced)
- **Cache I/O inefficiency** (`src/brain_brr/data/datasets.py:204-233`, `src/brain_brr/data/datasets.py:256-320`)
  - `np.load(..., allow_pickle=False)` is invoked for every sample; the compressed NPZ is fully decompressed to pull a single window, so each minibatch repeatedly hydrates ~50–60 MB files. OS page cache and DataLoader prefetching soften the hit, but we still pay unnecessary CPU time and transient allocations.
  - This is *a* contributor to long epochs (Modal ≈18–22 h) but not the sole bottleneck—forward/backward compute dominates once data is resident. Plan a cache-format update (memory-mapped arrays or per-window shards) after the P0 fix.
- **Validation memory pressure** (`src/brain_brr/train/loop.py:1024-1078`)
  - Validation accumulates all logits/labels before scoring: the full dev split (~183 k windows × 15 360 timesteps) occupies ~22 GB. Modal (96 GB RAM) is safe; local 64 GB hosts can OOM or swap.
  - Mitigation: stream evaluation (per-record aggregation) after the timeline fix, or temporarily limit dev set locally.

## 🟡 P2 — Medium Priority
- **Sampler bootstrap probe** (`src/brain_brr/train/loop.py:1778-1810`) — *currently dormant*
  - The expensive 20 k-window bootstrap only runs when the balanced manifest is missing. With the existing manifests (train ≈27 MB, dev ≈13 MB) the fallback path does **not** execute. Keep the safeguard but no action required unless manifest generation fails.

## 📌 Observations & Notes
- Mixed-precision, gradient sanitisation, and warm-up schedules look solid; no new numerical hazards identified.
- GNN/adjacency stack already clamps eigenvectors, enforces LayerScale, and caches valid positional encodings.
- Continue to defer high-risk refactors (loop.py split, cache rebuild) until current training jobs finish and P0 is addressed.

## 📋 Recommended Sequence (post-training window)
1. Ship the evaluation timeline fix (P0), add metadata plumbing, and validate metrics on a small subset. 
2. Introduce streaming/record-aware validation to remove the 22 GB accumulation (P1) while keeping Modal runs safe.
3. Redesign cache access (P1) once profiling confirms the new format yields worthwhile gains.
4. Revisit remaining refactors and legacy cleanup afterwards.

Document owner: Codex senior auditor (2025-10-02). Latest update incorporates cross-agent verification of claims.
