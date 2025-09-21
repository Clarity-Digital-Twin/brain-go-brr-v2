Caching & Sampling (Manifest + Balanced)

Scope
- Cache NPZ layout, manifest building, BalancedSeizureDataset usage.

Code anchors
- src/brain_brr/data/cache_utils.py (scan_existing_cache)
- src/brain_brr/data/datasets.py (BalancedSeizureDataset)
- src/brain_brr/train/loop.py (dataset selection + guards)
- src/brain_brr/cli/cli.py (build-cache, scan-cache)

Docs
- TUSZ/CACHE_AND_SAMPLING.md
- TUSZ/PREFLIGHT_AND_TROUBLESHOOTING.md

Related Phase
- phases/PHASE1_DATA_PIPELINE.md (sampling section)

Notes
- SeizureTransformer ratios: ALL partial + 0.3× full + 2.5× none.
- Manifest must yield partial>0 or full>0 before training.
