TUSZ Docs (Index)

Recommended reading order
- OVERVIEW.md — What TUSZ is, what we use
- DATA_FLOW.md — Ingestion pipeline and order of operations
- CSV_BI_PARSER.md — Parsing channel,start,stop,label,confidence
- CHANNELS_AND_MONTAGE.md — 19‑ch canonical order and synonyms
- CACHE_AND_SAMPLING.md — Manifest + BalancedSeizureDataset ratios
- PREFLIGHT_AND_TROUBLESHOOTING.md — Must‑pass checks and fixes
- EDF_HEADER_REPAIR.md — Rare header fixes and integration

Code anchors
- src/brain_brr/data/io.py — CSV parsing, masks
- src/brain_brr/data/cache_utils.py — scan_existing_cache
- src/brain_brr/data/datasets.py — BalancedSeizureDataset
- src/brain_brr/constants.py — channel order

Notes
- Rebuild caches after parser changes; scan manifest must report partial>0 or full>0 before any training.
