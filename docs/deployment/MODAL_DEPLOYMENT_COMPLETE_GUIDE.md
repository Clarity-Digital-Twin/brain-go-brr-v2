# 🚀 Modal Deployment Complete Guide for Brain-Go-Brr v2

Last updated: 2025-09-19

## ⚡ Quick Start (If Everything is Set Up)

> ⚠️ **CRITICAL**: Always use `--detach` flag to prevent disconnection from killing your training!

```bash
# ALWAYS use --detach to keep training running even if you disconnect
modal run deploy/modal/app.py --action train --config configs/smoke_test.yaml --detach
```

## 🎯 First-Time Setup (MANDATORY - DO IN ORDER!)

### Step 1: Configure Modal Workspace (CRITICAL!)

1. **Install Modal CLI**
   ```bash
   pip install --upgrade modal
   modal setup  # Login with your account
   ```

2. **Set Image Builder Version** ⚠️ **MOST IMPORTANT STEP** ⚠️
   - Go to: https://modal.com/settings/image-config
   - Select **"2025.06"** (NOT legacy!)
   - Click **Save**
   - **Why**: Legacy version causes Mamba-SSM compilation failures and dependency conflicts

### Step 2: Configure AWS S3 Access

1. **Create S3 Secret in Modal**
   ```bash
   modal secret create aws-s3-secret \
     --env AWS_ACCESS_KEY_ID=your_access_key \
     --env AWS_SECRET_ACCESS_KEY=your_secret_key \
     --env AWS_DEFAULT_REGION=us-east-1
   ```

2. **Verify Secret**
   ```bash
   modal secret list | grep aws-s3-secret
   ```

### Step 3: Verify Prerequisites

```bash
# Check Modal credits (need ~$140 for full training)
modal status

# Check local code quality
make q  # Must pass before deployment

# Verify S3 bucket name in app.py matches your bucket
grep "brain-go-brr-eeg-data" deploy/modal/app.py
```

## 📋 Deployment Steps

### 1. Smoke Test (ALWAYS DO FIRST!)

```bash
# Run 1-epoch test (~15 minutes, ~$1.40)
modal run deploy/modal/app.py --action train --config configs/smoke_test.yaml

# Monitor at the URL provided, e.g.:
# https://modal.com/apps/clarity-digital-twin/main/ap-xxxxx
```

**Expected Output:**
```
✓ Initialized. View run at https://modal.com/apps/...
✓ Created objects.
✓ Mamba-SSM imported successfully: 2.2.5
Running: python -m src train /tmp/tmp_xxx.yaml
```

### 2. Full Training (After Successful Smoke Test)

```bash
# Full 100-epoch training (detached mode)
modal run deploy/modal/app.py --action train --config configs/tusz_train_a100.yaml --detach

# Get the app ID from output, monitor at:
# https://modal.com/apps/clarity-digital-twin/main/<app-id>
```

### 3. Resume Training (If Interrupted)

```bash
# Resumes from last.pt checkpoint
modal run deploy/modal/app.py --action train --config configs/tusz_train_a100.yaml --resume true
```

## 🛠️ Management Commands

```bash
# List all apps
modal app list

# Stop a running app
modal app stop <app-id>

# View app logs
modal app logs <app-id>

# Check remaining credits
modal status
```

## 💰 Cost Breakdown

| Configuration | Time | Cost | Purpose |
|--------------|------|------|---------|
| smoke_test | ~15 min | ~$1.40 | Verify setup |
| tusz_train_a100 | ~25 hours | ~$140 | Full training |
| With early stopping | ~15-20 hours | ~$85-110 | Typical case |

**A100-80GB**: $5.59/hour

## ⚠️ Common Issues & Solutions

### Issue 1: "Using legacy Image Builder version"
**Solution**: Go to https://modal.com/settings/image-config → Select 2025.06 → Save

### Issue 2: Mamba-SSM Import Fails
**Causes & Fixes:**
- Missing CC/CXX env vars → Fixed in our app.py
- Wrong PyTorch version → We pin torch==2.2.2
- No build tools → We install build-essential + ninja-build

### Issue 3: Timeout Error
**Solution**: We set timeout=86400 (24 hours max for Modal)

### Issue 4: S3 Access Denied
**Solution**: Check aws-s3-secret is configured correctly

### Issue 5: OOM on Local Machine
**Solution**: Use configs/tusz_train_wsl2.yaml locally, tusz_train_a100.yaml on Modal

## 📂 File Structure

```
deploy/modal/
├── app.py                    # Main Modal deployment script
├── configs/
│   ├── smoke_test.yaml      # 1-epoch test config
│   ├── tusz_train_a100.yaml # Full A100 training
│   └── tusz_train_wsl2.yaml # Local WSL2-safe config
└── README.md                 # Basic instructions
```

## 🔍 Monitoring Training

1. **Modal Dashboard**: https://modal.com/apps/clarity-digital-twin
2. **Logs**: Click on app → View logs
3. **Costs**: Check "Usage" tab in Modal dashboard
4. **GPU Utilization**: Shown in app metrics

## 🚨 Pre-Flight Checklist

```markdown
- [ ] Modal CLI installed (`pip install modal`)
- [ ] Modal authenticated (`modal setup`)
- [ ] Image Builder set to 2025.06 (https://modal.com/settings/image-config)
- [ ] AWS S3 secret created (`modal secret create aws-s3-secret ...`)
- [ ] S3 bucket accessible (brain-go-brr-eeg-data-20250919)
- [ ] Modal credits sufficient ($140+)
- [ ] Local code passes quality checks (`make q`)
- [ ] Configs reviewed (batch_size, epochs, etc.)
```

## 📈 Performance Expectations

- **Data Loading**: First epoch slower (~5 min for S3 mount initialization)
- **Training Speed**: ~15 min/epoch on A100-80GB
- **Memory Usage**: ~40GB GPU VRAM with batch_size=64
- **Model Size**: ~25M parameters
- **Early Stopping**: Typically triggers around epoch 50-70

## 🔧 Advanced Configuration

### Custom S3 Bucket
Edit `deploy/modal/app.py`:
```python
data_mount = modal.CloudBucketMount(
    "your-bucket-name",  # Change this
    secret=s3_secret,
    key_prefix="tusz/",  # Adjust path
    read_only=True,
)
```

### Adjust GPU Type
```python
@app.function(
    gpu="A100-40GB",  # Or "A10G", "T4" for cheaper options
    # ...
)
```

### Change Timeout
```python
@app.function(
    timeout=43200,  # 12 hours instead of 24
    # ...
)
```

## 📊 Expected Training Output

```
[INFO] 🧠 Brain-Go-Brr v2 Training Pipeline
[INFO] Loading configuration from configs/tusz_train_a100.yaml
[INFO] Initializing TUSZ dataset from /data/edf/train
[INFO] Found 22,093 EDF files (limited to 50 for smoke test)
[INFO] Model: SeizureDetector (25.3M parameters)
[INFO] ✓ Using Mamba-SSM CUDA kernels (not fallback)

Epoch 1/100 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:15:23
├─ Loss: 0.312
├─ Sensitivity: 0.743
├─ Specificity: 0.892
└─ AUROC: 0.867
```

## 🆘 Emergency Procedures

### Stop All Training
```bash
modal app list | grep brain-go-brr | awk '{print $1}' | xargs -I {} modal app stop {}
```

### Check Remaining Credits
```bash
modal status
# Add more credits at: https://modal.com/settings/billing
```

### Debug Failed Deployment
```bash
# Check recent app logs
modal app list | head -5
modal app logs <failed-app-id> | tail -100
```

## 📝 Notes

- **Data**: TUH EEG Seizure Corpus v2.0.0 (84.6GB, 22,093 files)
- **Architecture**: Bi-Mamba-2 + U-Net + ResCNN
- **Metrics**: TAES score, sensitivity at various FA rates
- **Checkpoints**: Saved to Modal volume at /results/

---

**Last Updated**: 2025-01-19 (Evening)
**Validated**: Successfully deployed smoke test with Image Builder 2025.06
**Contact**: For issues, check MODAL_MAMBA_DEPLOYMENT_ISSUES.md
