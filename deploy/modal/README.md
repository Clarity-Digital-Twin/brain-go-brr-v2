# Modal Cloud Deployment

Deploy Brain-Go-Brr v2 training to Modal's GPU infrastructure.

> **Important**: This deployment uses NVIDIA CUDA development images for mamba-ssm compilation.
> First build takes ~10-15 minutes due to CUDA kernel compilation. Subsequent runs use cached images.

## Prerequisites

- Modal CLI installed and authenticated
- ~10GB free space for Docker image caching
- Patience for first-time mamba-ssm compilation (one-time cost)

## Setup

### 1. Install Modal CLI
```bash
pip install --upgrade modal
```

### 2. Authenticate
```bash
modal setup
# Opens browser for authentication
# Select or create a workspace (e.g., clarity-digital-twin)
# Close browser window after "API token created!" message
```

**Token Storage:**
- Token saved to `~/.modal.toml` (outside repo, gitignored)
- Format: `token_id` and `token_secret` for your workspace
- Never commit this file - it's your authentication credential
- Token persists across sessions (no need to re-authenticate)

### 3. (Optional) Add W&B Secret
```bash
# Only needed if using Weights & Biases tracking
modal secret create wandb-secret WANDB_API_KEY=<your-key>
# Then uncomment the secret in app.py
```

**Note:** Training works without any secrets. Only required for external API integrations.

## Usage

### Training

**Smoke Test** (quick validation):
```bash
modal run deploy/modal/app.py --action train --config configs/smoke_test.yaml
```

**Full Training** (production run):
```bash
modal run deploy/modal/app.py --action train --config configs/production.yaml --detach
```

### Evaluation

```bash
modal run deploy/modal/app.py --action evaluate --config /results/checkpoints/best.ckpt
```

## Data Management

### For Massive EEG Datasets (TUH/CHB-MIT)

**Option 1: S3/Cloud Storage (RECOMMENDED for >100GB)**

1. **Upload to S3 (one-time):**
```bash
# TUH is ~1.5TB, CHB-MIT is ~40GB
aws s3 sync ./tuh_eeg_seizure_v2.0.0 s3://your-eeg-bucket/tuh/
aws s3 sync ./chb-mit s3://your-eeg-bucket/chb-mit/
```

2. **Create Modal secret:**
```bash
modal secret create aws-s3-secret \
  AWS_ACCESS_KEY_ID=your_key \
  AWS_SECRET_ACCESS_KEY=your_secret \
  AWS_REGION=us-east-1
```

3. **Update app.py:** Uncomment CloudBucketMount section

**Option 2: Modal Volumes (for smaller/preprocessed data)**
```bash
# Only for smaller datasets or preprocessed windows
modal volume put brain-go-brr-data ./data/preprocessed /preprocessed
```

**Option 3: Stream from public URLs**
```python
# In your config, use URLs directly
data_url: "https://physionet.org/files/chb-mit/1.0.0/"
```

### Download Results
```bash
# Get checkpoints
modal volume get brain-go-brr-results /checkpoints ./results/checkpoints

# Get metrics
modal volume get brain-go-brr-results /evaluations ./results/evaluations
```

## GPU Options & Costs

| GPU | VRAM | $/hour | Use Case | vs RTX 4090 |
|-----|------|--------|----------|-------------|
| T4 | 16GB | $0.59 | Testing/debugging | ~0.3x speed |
| L40S | 48GB | $3.99 | Large models | ~1.5x speed |
| A100-40GB | 40GB | $3.99 | Fast training | ~2x speed |
| A100-80GB | 80GB | $5.59 | **Recommended** | ~3x speed |
| H100 | 80GB | $8.99 | Fastest (overkill) | ~5x speed |

**GPU Selection Guide:**
- If you have RTX 4090 (24GB): Use A100-80GB for meaningful speedup
- Default config uses A100-80GB for best price/performance
- Smoke tests complete in ~2-3 min on A100-80GB

## Tips

- Use `--detach` for long runs to avoid terminal disconnects
- Add `spot=True` to function decorators for 70% cost savings (may preempt)
- Monitor runs at https://modal.com/apps
- First run takes longer (~5 min) due to image building
- Subsequent runs reuse cached image (~2-3 min for smoke test)

## Troubleshooting

### "Token missing" error
```bash
# Run authentication
modal setup
```

### Mamba-SSM build failures

**Problem**: `nvcc was not found` or `bare_metal_version is not defined`

**Solution**: We use `nvidia/cuda:12.1.0-devel-ubuntu22.04` base image which includes:
- nvcc (CUDA compiler)
- CUDA development headers
- All tools for compiling CUDA kernels

**Why it happens**:
- `debian_slim` images lack CUDA development tools
- PyTorch runtime images don't include nvcc
- Mamba-SSM requires compiling custom CUDA kernels

### Long first-time build

**Expected**: First deployment takes 10-15 minutes
- Downloading CUDA dev image (~7GB)
- Installing PyTorch 2.2.2 + CUDA libs
- Compiling mamba-ssm CUDA kernels
- Building all dependencies

**After first build**: Image is cached, deployments take ~30 seconds

### Monitoring training progress
```bash
# After launching, you'll see:
# "Monitor at: https://modal.com/apps/<run-id>"
# Click the link to view logs in real-time
```

### Volume not found
```bash
# Volumes are created automatically on first use
# Or manually create:
modal volume create brain-go-brr-data
modal volume create brain-go-brr-results
```

## Technical Architecture

### Image Build Strategy
```python
# Uses NVIDIA CUDA devel image (not debian_slim!)
modal.Image.from_registry(
    "nvidia/cuda:12.1.0-devel-ubuntu22.04",
    add_python="3.11"
)
```

### Why This Architecture?
1. **CUDA Development Image**: Required for nvcc compiler
2. **PyTorch 2.2.2**: Specific version for mamba-ssm compatibility
3. **Build Order**: PyTorch → numpy → mamba-ssm (order matters!)
4. **A100-80GB**: 3x faster than RTX 4090, worth the cost

## File Structure

```
deploy/modal/
├── app.py      # Modal deployment script
└── README.md   # This file
```