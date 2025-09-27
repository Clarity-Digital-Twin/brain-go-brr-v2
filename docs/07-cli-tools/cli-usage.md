# CLI Usage

Entrypoint

- `python -m src ...`

Commands

- Validate config: `python -m src validate <config.yaml> [--phase data|model|training]`
- Train: `python -m src train <config.yaml> [--resume] [--device auto|cpu|cuda]`
- Build cache: `python -m src build-cache --data-dir <edf_dir> --cache-dir <cache_split_dir> [--split train|dev] [--limit-files N]`
- Scan cache to manifest: `python -m src scan-cache --cache-dir <cache_split_dir>`
- Evaluate: `python -m src evaluate <checkpoint.pt> <edf_dir> [--config <config.yaml>] [--device cuda] [--output-json out.json] [--output-csv-bi out.csv] [--dry-run]`

Validation and summary

- Add `--config <path>` ; the CLI summarizes key settings

Notes

- build-cache respects `--limit-files` first, then falls back to `BGB_LIMIT_FILES` if set.
- scan-cache exits with code 2 when no seizure windows are found (guards bad label paths).
- evaluate config resolution: uses `--config` if provided; else uses checkpoint-embedded config if present and not `None`; otherwise exits with an error.
- evaluate EDF discovery: exits with an error if no `*.edf` files are found under the provided `data_path`.
- CSV_BI export currently writes a single CSV for the evaluation run. Event times are stride-aware (60s windows, 10s stride), but not grouped per-recording.

Makefile commands are listed in `docs/07-cli-tools/makefile-commands.md`.
