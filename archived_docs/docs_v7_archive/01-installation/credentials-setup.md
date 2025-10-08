# Credentials Setup Guide

## Overview

Brain-Go-Brr uses standard CLI tools for cloud services. Credentials are stored in your home directory (NOT in the repo).

---

## AWS S3 (Optional - for cloud storage)

### Setup
```bash
aws configure
```

### What it creates
- `~/.aws/credentials` - Access keys
- `~/.aws/config` - Region settings

### Where it's used
- S3 data download scripts
- Cache backup/restore

---

## Modal (Optional - for cloud training)

### Setup
```bash
modal setup
```

### What it creates
- `~/.modal.toml` - API tokens

### Where it's used
- `deploy/modal/app.py` - Cloud training
- Modal volume operations

---

## Docker (Local training only)

### Setup
```bash
cp .env.example .env
# Edit .env and set CACHE_DIR to your repo path
```

### What it's for
- docker-compose.yml volume mounts
- NOT used by AWS/Modal

---

## For OSS Contributors

### Quick Start (No Cloud)
```bash
# No credentials needed for local training!
make setup
make setup-gpu
make train-local
```

### With Cloud (Optional)
```bash
# 1. AWS (for S3 data)
aws configure
# Enter your IAM keys

# 2. Modal (for A100 training)
modal setup
# Enter your Modal token

# 3. Docker (for containerized local training)
cp .env.example .env
# Update CACHE_DIR path
```

---

## Security Notes

- ✅ `.env` is gitignored (never committed)
- ✅ `~/.aws/` and `~/.modal.toml` are outside repo
- ❌ **NEVER** commit credentials to git
- ❌ **NEVER** share `.env` or `~/.aws/credentials`

---

## Troubleshooting

### "AWS credentials not found"
```bash
aws configure list  # Check if configured
cat ~/.aws/credentials  # Verify file exists
```

### "Modal authentication failed"
```bash
modal config show  # Check current config
modal setup  # Reconfigure
```

### Docker "CACHE_DIR not set"
```bash
cat .env  # Should have CACHE_DIR=/path/to/cache
cp .env.example .env  # Recreate if missing
```
