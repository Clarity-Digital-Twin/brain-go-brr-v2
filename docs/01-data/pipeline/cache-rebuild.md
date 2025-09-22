Cache Rebuild Playbook

When to rebuild
- Parser/label set changes (e.g., added mysz)
- Windowing or preprocessing changes (size, stride, filters)
- Channel mapping changes
- Corrupted or stale manifest detected

Quick verification (must pass before training)
- `python -m src scan-cache --cache-dir <cache_dir>` → expect partial>0 or full>0
- Optionally force a rebuild on startup: `BGB_FORCE_MANIFEST_REBUILD=1`

Local commands
```bash
# Rebuild train/val caches and manifest
python -m src build-cache --data-dir data_ext4/tusz/edf/train --cache-dir cache/tusz/train
python -m src build-cache --data-dir data_ext4/tusz/edf/train --cache-dir cache/tusz/val
python -m src scan-cache --cache-dir cache/tusz/train
```

Modal commands
```bash
# Rebuild caches on persistent volume
python -m src build-cache --data-dir /data/edf/train --cache-dir /results/cache/tusz/train
python -m src build-cache --data-dir /data/edf/train --cache-dir /results/cache/tusz/val
python -m src scan-cache --cache-dir /results/cache/tusz/train
```

Notes
- Training auto-validates manifest and rebuilds if empty/stale.
- Use the env var for a one-off forced rebuild without manual deletion.
- For destructive resets (rare), delete the cache directory before rebuild.
