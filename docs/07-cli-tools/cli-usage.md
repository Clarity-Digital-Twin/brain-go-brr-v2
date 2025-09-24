# CLI Usage

Entrypoint

- `python -m src ...`

Commands

- Validate config: `python -m src validate <config.yaml> [--phase data|model|training]`
- Train: `python -m src train <config.yaml> [--resume] [--device auto|cpu|cuda]`
- Build cache: `python -m src build-cache --data-dir <edf_dir> --cache-dir <cache_split_dir> [--split train|val]`
- Scan cache to manifest: `python -m src scan-cache --cache-dir <cache_split_dir>`
- Evaluate: `python -m src evaluate <checkpoint.pt> <edf_dir> [--config <config.yaml>] [--device cuda] [--output-json out.json] [--output-csv-bi out.csv]`

Validation and summary

- Add `--config <path>` ; the CLI summarizes key settings

Makefile commands are listed in `docs/07-cli-tools/makefile-commands.md`.
