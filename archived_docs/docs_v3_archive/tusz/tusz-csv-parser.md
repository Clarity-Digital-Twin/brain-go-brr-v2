TUSZ CSV_BI Parser (Canonical)

Goal: turn TUSZ CSV_BI annotations into a per-sample binary mask robustly.

File format (CSV_BI)

- Header comments with metadata, e.g. `# duration = 300.00 secs`
- Header row: `channel,start_time,stop_time,label,confidence`
- Rows: `FP1-F7,36.8868,183.3055,cpsz,1.0000`

Parsing rules (implemented)

- Duration: parse from comment header; tolerate minor formatting variations
- Skip header row; split by comma
- Use start_time and stop_time as seconds (float); label is string
- Map label→binary: seizure if in the seizure set, else background

Seizure label set (CRITICAL - v2.0.3 verified)

- **CRITICAL**: TUSZ v2.0.3 actual seizure types: {seiz, gnsz, fnsz, cpsz, absz, spsz, tcsz, tnsz, mysz}
- **WARNING**: `spkz` does NOT exist in TUSZ v2.0.3 (empirically verified across the train split)
- **mysz discovery**: Found 44 occurrences (0.1% of corpus) - was missing until Sept 2025 fix
- Optional extension: treat any non-"bckg" as seizure (configurable knob — see TODO in code)

Empirical seizure type frequencies (v2.0.3 analysis):
| Type | Name | Count | Frequency |
|------|------|-------|-----------|
| gnsz | Generalized Non-Specific | 23,804 | 46.5% |
| fnsz | Focal Non-Specific | 19,000 | 37.1% |
| cpsz | Complex Partial | 3,597 | 7.0% |
| absz | Absence | 2,507 | 4.9% |
| spsz | Simple Partial | 942 | 1.8% |
| tcsz | Tonic-Clonic | 857 | 1.7% |
| tnsz | Tonic | 410 | 0.8% |
| mysz | Myoclonic | 44 | 0.1% |
| **Total** | | **51,161** | **100%** |

Mask generation

- Sampling rate: 256 Hz (fixed in our pipeline)
- For each event in seizure set: set mask[start:end] = 1
- Boundary indexing: deterministic integer indices for both ends
  - start_idx = int(start_sec * fs)  (floor)
  - end_idx   = int(end_sec * fs)    (floor)

Common pitfalls (we hit these)

- Mis-reading CSV_BI as simple `start,end,label` → produced all-zero masks
- Ignoring seizure codes beyond `seiz` (e.g., `cpsz`) → missing positives
- Not extracting duration from comments → mask length mismatch

Verification checklist

- Pick a known seizure CSV; ensure events are parsed with the correct counts and durations
- Spot-check: first seizure start/stop make sense; mask has non-zero fraction

Code anchors

- src/brain_brr/data/io.py: parse_tusz_csv, events_to_binary_mask
