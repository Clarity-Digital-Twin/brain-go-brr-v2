# Deployment & Training Guide

## Current Status
- **Local Training**: Running 1-epoch test on TUSZ dataset (~44 hours ETA)
- **Modal Deployment**: Fixed and working, awaiting AWS S3 setup for data
- **Dataset**: 79GB TUSZ at `/home/jj/proj/brain-go-brr-v2/data_ext4/tusz/`

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
aws s3 mb s3://brain-go-brr-eeg-data --region us-east-1
aws s3api put-bucket-encryption --bucket brain-go-brr-eeg-data \
  --server-side-encryption-configuration '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'

# Upload TUSZ (79GB)
aws s3 sync /home/jj/proj/brain-go-brr-v2/data_ext4/tusz/ \
  s3://brain-go-brr-eeg-data/tusz/ \
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
✅ Local training functional
✅ Modal deployment fixed
✅ AWS CLI installed
⏸️ AWS credentials needed
⏸️ S3 bucket creation pending
⏸️ Data upload pending

## Next Steps
1. Get AWS credentials
2. Upload TUSZ to S3
3. Configure Modal secrets
4. Run full training on A100s