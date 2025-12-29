"""SzCORE submission utilities (epilepsybenchmarks.com).

This package contains a self-contained inference entrypoint that:
- Reads SzCORE-format EDF (19 channels, `-Avg` suffix, fixed order)
- Remaps channels into `constants.CHANNEL_NAMES_10_20`
- Runs Brain-Go-Brr inference when a GPU is available
- Falls back to a lightweight CPU heuristic when GPU is unavailable (CI validation)
- Writes HED-SCORE compliant TSV output
"""

from __future__ import annotations
