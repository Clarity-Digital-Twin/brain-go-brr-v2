# MODAL PIPELINE SETUP - CLOUD GPU TRAINING

## CRITICAL: Modal Pipeline IS READY with all fixes!

### Modal Setup (`deploy/modal/app.py`)

1. **Image includes all dependencies** (line numbers may vary)
   - PyTorch with CUDA 12.1
   - Mamba-SSM with CUDA kernels
   - All project dependencies
   - Copies `/src` and `/configs` (CSV_BI parser + balancing fixes)

2. **Data mounted from S3**
```python
# Example
data_mount = modal.CloudBucketMount(
    "brain-go-brr-eeg-data-20250919",  # Your S3 bucket
    key_prefix="tusz/",
    read_only=True,
)
```

3. **Config uses persistent storage** (`configs/tusz_train_a100.yaml`)
```yaml
data:
  data_dir: /data/edf/train         # S3 mount
  cache_dir: /results/cache/tusz    # Persistent volume
```

4. **Training function**
```python
def train(config_path="configs/tusz_train_a100.yaml", resume=False):
    ...
```

## How it works
1. First run builds cache from S3 data using fixed parser
2. Cache persists in `/results/cache/tusz`
3. Next runs reuse cache, build manifest, use BalancedSeizureDataset

## Running on Modal
```bash
# Deploy and run
modal run deploy/modal/app.py::train

# Different config
modal run deploy/modal/app.py::train --config-path configs/smoke_test.yaml

# Resume training
modal run deploy/modal/app.py::train --resume
```

## Notes
- S3 data must mirror local structure (edf/train with EDFs and CSVs)
- Persistent results volume stores checkpoints, logs, cache
- Balanced dataset is applied automatically via manifest

