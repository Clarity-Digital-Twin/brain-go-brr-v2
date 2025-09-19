# Deployment & Training Guide

## Current Status
- **Local Training**: Running 100-epoch TUSZ training
- **Modal Deployment**: Configured with S3 CloudBucketMount
- **Dataset**: 79GB TUSZ EEG data

## Local Training

### Quick Test
```bash
# Smoke test (few files, 1 epoch)
make train-local
```

### Full Training
```bash
# Full TUSZ training in tmux
tmux new -s tusz-full
source .venv/bin/activate
unset BGB_LIMIT_FILES
.venv/bin/python -m src train configs/tusz_train.yaml 2>&1 | tee tusz_full_training.log
# Ctrl+B, D to detach

# Monitor
tmux attach -t tusz-full
```

**Note**: Single epoch takes ~44 hours on local hardware. Full training (50-100 epochs) would take weeks.

## Modal Cloud Deployment

### Prerequisites
1. **Install Modal CLI**
```bash
pip install --upgrade modal
modal setup  # Authenticate via browser
```

2. **AWS S3 Setup (for massive datasets)**
```bash
# Install AWS CLI (already done)
pip install awscli

# Configure credentials (pending)
aws configure

# Create secure bucket
aws s3 mb s3://your-eeg-bucket --region us-east-1
aws s3api put-bucket-encryption --bucket your-eeg-bucket \
  --server-side-encryption-configuration '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'

# Upload TUSZ (79GB)
aws s3 sync /path/to/local/tusz/ \
  s3://your-eeg-bucket/tusz/ \
  --storage-class INTELLIGENT_TIERING \
  --sse AES256
```

3. **Configure Modal Secret**
```bash
modal secret create aws-s3-secret \
  AWS_ACCESS_KEY_ID=your_key \
  AWS_SECRET_ACCESS_KEY=your_secret \
  AWS_REGION=us-east-1
```

### Deploy & Run

```bash
# Deploy to Modal
cd deploy/modal
modal deploy app.py

# Run training (A100-80GB, ~3x faster than RTX 4090)
modal run app.py --action train --config configs/production.yaml --detach

# Monitor at https://modal.com/apps
```

### Performance Comparison
| Hardware | Single Epoch | 50 Epochs | Cost |
|----------|--------------|-----------|------|
| Local (CPU/GPU) | ~44 hours | ~90 days | $0 (electricity) |
| RTX 4090 | ~15 hours | ~30 days | $0 (owned) |
| Modal A100-80GB | ~5 hours | ~10 days | ~$1,340 |

## Data Management

### Option 1: S3 CloudBucketMount (Recommended)
- Best for massive datasets (TUH is 79GB)
- No upload to Modal needed
- Streams directly from S3
- See `deploy/modal/app.py` for CloudBucketMount config

### Option 2: Modal Volumes
- Only for preprocessed/smaller data
- `modal volume put brain-go-brr-data ./data /data`

### Option 3: Public URLs
- Stream from PhysioNet directly
- No storage needed

## Architecture
- **Model**: Bi-Mamba-2 + U-Net + ResCNN (48M params)
- **Input**: 19-channel EEG @ 256Hz
- **Processing**: 60s windows, 10s stride
- **Post**: Hysteresis (τ_on=0.86, τ_off=0.78) → morphology → duration filter

## Status
✅ Local training functional (100 epochs running)
✅ Modal deployment configured with S3
✅ AWS CLI installed and configured
✅ S3 bucket created: `brain-go-brr-eeg-data-20250919`
✅ Modal secret created: `aws-s3-secret`
⏳ TUH data uploading to S3 (79GB in progress)

## S3 Cost Breakdown
- **Storage**: $1.82/month for 79GB (Intelligent Tiering)
- **Transfer to Modal**: FREE (same region)
- **Auto-optimization**: Moves to cheaper tiers if not accessed

## Next Steps
1. ✅ ~~Get AWS credentials~~ DONE
2. ⏳ Wait for S3 upload to complete (~30 min)
3. Deploy to Modal: `modal deploy deploy/modal/app.py`
4. Run on A100s: `modal run deploy/modal/app.py --detach`