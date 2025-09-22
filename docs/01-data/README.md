# Data Pipeline Documentation

## Overview

Complete documentation of the EDF → Training data pipeline, including TUSZ-specific handling, caching, and optimizations.

## Directory Structure

### `/pipeline` - Core Data Flow
- **[architecture.md](pipeline/architecture.md)** - Complete pipeline stages
- **[balanced-sampling.md](pipeline/balanced-sampling.md)** - 7200x optimization for Modal
- **[cache-rebuild.md](pipeline/cache-rebuild.md)** - Cache management procedures
- **[flow-diagram.md](pipeline/flow-diagram.md)** - Visual pipeline representation

### `/tusz` - TUSZ Dataset Specifics
- **[tusz-overview.md](tusz/tusz-overview.md)** - Dataset structure and stats
- **[tusz-csv-parser.md](tusz/tusz-csv-parser.md)** - CSV_BI format parsing
- **[tusz-channels.md](tusz/tusz-channels.md)** - Channel mapping and synonyms
- **[tusz-edf-repair.md](tusz/tusz-edf-repair.md)** - Malformed header handling
- **[tusz-cache-sampling.md](tusz/tusz-cache-sampling.md)** - Balanced sampling strategy
- **[tusz-data-flow.md](tusz/tusz-data-flow.md)** - TUSZ-specific pipeline
- **[tusz-preflight.md](tusz/tusz-preflight.md)** - Pre-training validation

### `/issues` - Historical Context
- **[critical-resolved.md](issues/critical-resolved.md)** - Major bugs and resolutions
- **[data-io.md](issues/data-io.md)** - I/O implementation details

## Quick Reference

### Key Statistics
- **Dataset**: TUH EEG Seizure Corpus v2.0.3
- **Train/Val Split**: 80/20
- **Cache Files**: 3734 train, 933 validation
- **Window Size**: 60s with 10s stride
- **Seizure Ratio**: ~34.2% of windows contain seizures
- **File Sizes**: 26-152MB per NPZ cache file

### Critical Optimizations

1. **Balanced Sampling** - Eliminates 2+ hour Modal delays
   - Uses pre-computed manifest instead of runtime sampling
   - 7200x speedup on network storage

2. **Channel Mapping** - Handles TUSZ inconsistencies
   - Maps `'EEG FP1-LE'` → `'Fp1'`
   - Handles synonyms: T7→T3, T8→T4, P7→T5, P8→T6

3. **CSV_BI Parser** - Correct seizure annotation parsing
   - Reads columns 2/3 for times, column 4 for labels
   - Supports all 9 seizure types including `mysz`

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| "0 windows in dataset" | Rebuild cache with correct CSV parser |
| "Missing required channels" | Check channel mapping in `constants.py` |
| "2+ hour dataset stats" | Ensure using BalancedSeizureDataset |
| "Manifest stale" | Set `BGB_FORCE_MANIFEST_REBUILD=1` |

## Code References

- **Data Loading**: `src/brain_brr/data/loader.py`
- **Datasets**: `src/brain_brr/data/dataset.py`
- **CSV Parser**: `src/brain_brr/data/io.py::parse_tusz_csv()`
- **Cache Utils**: `src/brain_brr/data/cache_utils.py`
- **Constants**: `src/brain_brr/constants.py`