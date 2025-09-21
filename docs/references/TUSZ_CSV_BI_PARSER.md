# TUSZ CSV_BI Parser — Canonical Guide

Last updated: 2025-09-21

Purpose: Document the correct parsing of TUSZ CSV_BI annotations and where it lives in code.

## Format

- CSV_BI columns: `channel,start_time,stop_time,label,confidence`
- Example:
  - `FP1-F7,36.8868,183.3055,cpsz,1.0000`

## Seizure labels

We treat the following labels as seizure: `{seiz, gnsz, fnsz, spsz, cpsz, absz, tnsz, tcsz, spkz}`

## Implementation

- Parser: `src/brain_brr/data/io.py` (parse_tusz_csv)
  - Parses duration from header comments `# duration = XXX secs`
  - Skips the header row (`channel,...`)
  - Uses columns 1..3 as `start, stop, label` (ignores channel name)
- Mask building: `src/brain_brr/data/io.py` (events_to_binary_mask)
  - fs=256
  - 1.0 during seizures, 0.0 otherwise

## Guardrails

- scan-cache/build-cache command fails if no seizures found (to prevent wasted training)

## Quick check

```
python - << 'PY'
from pathlib import Path
from src.brain_brr.data.io import parse_tusz_csv, events_to_binary_mask

csv = Path('data_ext4/tusz/edf/train/aaaaaaac/s001_2002/02_tcp_le/aaaaaaac_s001_t000.csv')
dur, events = parse_tusz_csv(csv)
mask = events_to_binary_mask(dur, events, fs=256)
print(dur, len(events), mask.shape, float(mask.mean()))
PY
```

