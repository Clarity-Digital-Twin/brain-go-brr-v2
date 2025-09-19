# Modal Cloud Deployment

Deploy Brain-Go-Brr v2 training to Modal's GPU infrastructure.

## Setup

### 1. Install Modal CLI
```bash
pip install --upgrade modal
```

### 2. Authenticate
```bash
modal setup
# Follow the browser flow to authenticate
```

### 3. (Optional) Add W&B Secret
```bash
modal secret create wandb-secret WANDB_API_KEY=<your-key>
# Then uncomment the secret in app.py
```

## Usage

### Training

**Smoke Test** (T4 GPU, cheaper):
```bash
modal run deploy/modal/app.py --action train --config configs/smoke_test.yaml
```

**Full Training** (L40S GPU, 48GB VRAM):
```bash
modal run deploy/modal/app.py --action train --config configs/production.yaml --detach
```

### Evaluation

```bash
modal run deploy/modal/app.py --action evaluate --config /results/checkpoints/best.ckpt
```

## Data Management

### Upload Training Data
```bash
# Create and populate data volume
modal volume put brain-go-brr-data ./data/tuh_eeg_seizure /tuh_eeg_seizure
modal volume put brain-go-brr-data ./data/chb-mit /chb-mit
```

### Download Results
```bash
# Get checkpoints
modal volume get brain-go-brr-results /checkpoints ./results/checkpoints

# Get metrics
modal volume get brain-go-brr-results /evaluations ./results/evaluations
```

## GPU Options & Costs

| GPU | VRAM | $/hour | Use Case |
|-----|------|--------|----------|
| T4 | 16GB | $0.59 | Testing/debugging |
| L40S | 48GB | $3.99 | **Recommended** for Mamba-2 |
| A100-40GB | 40GB | $3.99 | Fast training |
| A100-80GB | 80GB | $5.59 | Large batches |
| H100 | 80GB | $8.99 | Fastest (overkill) |

## Tips

- Use `--detach` for long runs to avoid terminal disconnects
- Add `spot=True` to function decorators for 70% cost savings (may preempt)
- Monitor runs at https://modal.com/apps

## File Structure

```
deploy/modal/
├── app.py      # Modal deployment script
└── README.md   # This file
```