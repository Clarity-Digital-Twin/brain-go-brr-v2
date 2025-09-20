# Modal Training Visibility Improvements

## Problem
Modal web UI and CLI don't show real-time training progress. Currently we get:
- No batch-level progress
- Hours of silence during epochs
- Buffered output that dumps all at once
- tqdm progress bars don't render properly

## Quick Fixes (Do These First)

### 1. Add Periodic Progress Prints
```python
# In src/brain_brr/train/loop.py, add every N batches:
if batch_idx % 100 == 0:  # Every 100 batches
    print(f"[PROGRESS] Epoch {epoch}/{total_epochs} | "
          f"Batch {batch_idx}/{total_batches} | "
          f"Loss: {loss.item():.4f} | "
          f"LR: {optimizer.param_groups[0]['lr']:.2e}",
          flush=True)  # CRITICAL: flush=True forces output
```

### 2. Force Flush All Prints
```python
# Add flush=True to EVERY print statement
print(f"Starting epoch {epoch}", flush=True)
print(f"Validation loss: {val_loss:.4f}", flush=True)
```

### 3. Add Heartbeat Logger
```python
# In training loop, add a simple heartbeat every 5 minutes
import time
last_heartbeat = time.time()

# In batch loop:
if time.time() - last_heartbeat > 300:  # 5 minutes
    print(f"[HEARTBEAT] Still training... Batch {batch_idx}/{total_batches}", flush=True)
    last_heartbeat = time.time()
```

## Proper Solutions

### 1. TensorBoard Integration (RECOMMENDED)
```python
# In src/brain_brr/train/loop.py
from torch.utils.tensorboard import SummaryWriter

# In train_one_epoch():
writer = SummaryWriter(log_dir=f"{output_dir}/runs")

# Log every batch:
global_step = epoch * len(dataloader) + batch_idx
writer.add_scalar('Loss/train', loss.item(), global_step)
writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], global_step)

# Log validation:
writer.add_scalar('Loss/val', val_loss, epoch)
writer.add_scalar('Metrics/TAES', taes, epoch)
writer.add_scalar('Metrics/AUROC', auroc, epoch)

# View locally:
# tensorboard --logdir results/runs --bind_all
# On Modal: serve via Modal's web endpoint or download logs
```

### 2. Weights & Biases Integration (BEST FOR MODAL)
```python
# In src/brain_brr/train/loop.py
import wandb

# Initialize (in main training function):
wandb.init(
    project="brain-go-brr-v2",
    config={
        "learning_rate": config.training.learning_rate,
        "epochs": config.training.epochs,
        "batch_size": config.training.batch_size,
        "model": "Bi-Mamba-2 + U-Net + ResCNN",
    }
)

# Log every batch:
wandb.log({
    "train/loss": loss.item(),
    "train/lr": optimizer.param_groups[0]['lr'],
    "train/epoch": epoch,
    "train/batch": batch_idx,
})

# Log validation:
wandb.log({
    "val/loss": val_loss,
    "val/taes": taes,
    "val/auroc": auroc,
    "val/sensitivity_10fa": sens_10fa,
})

# Log model checkpoints:
wandb.save(f"{checkpoint_path}")
```

### 3. Modal-Specific Logging
```python
# Use Modal's volume to write detailed logs
import json
from pathlib import Path

class ModalLogger:
    def __init__(self, log_dir="/results/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"training_{datetime.now().isoformat()}.jsonl"

    def log(self, metrics: dict):
        with open(self.log_file, 'a') as f:
            json.dump({**metrics, "timestamp": time.time()}, f)
            f.write('\n')
        # Also print for immediate visibility
        print(f"[LOG] {metrics}", flush=True)

# Usage:
logger = ModalLogger()
logger.log({"epoch": epoch, "batch": batch_idx, "loss": loss.item()})
```

### 4. Rich Progress Bars (Better than tqdm for Modal)
```python
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.console import Console

console = Console(force_terminal=True)  # Forces color output

with Progress(
    SpinnerColumn(),
    *Progress.get_default_columns(),
    TimeElapsedColumn(),
    console=console,
    refresh_per_second=1
) as progress:
    task = progress.add_task("[cyan]Training...", total=len(dataloader))
    for batch in dataloader:
        # ... training code ...
        progress.update(task, advance=1, description=f"[cyan]Loss: {loss.item():.4f}")
```

## Environment Variables for W&B on Modal

```python
# In deploy/modal/app.py, add to secrets:
app = modal.App(
    "brain-go-brr-v2",
    image=image,
    secrets=[
        modal.Secret.from_name("wandb-secret"),  # Create this in Modal dashboard
    ],
)

# Or in train function:
env["WANDB_API_KEY"] = "your-key-here"  # Better to use Modal secrets
env["WANDB_PROJECT"] = "brain-go-brr-v2"
```

## Quick Implementation Priority

1. **IMMEDIATE**: Add flush=True to all prints (5 min fix)
2. **TODAY**: Add batch progress prints every 100 batches
3. **NEXT RUN**: Setup W&B - it's perfect for Modal cloud training
4. **FUTURE**: TensorBoard for local, W&B for Modal

## Testing Locally First
```bash
# Test with minimal data
export BGB_LIMIT_FILES=2
python -m src train configs/smoke_test.yaml

# You should see:
# [PROGRESS] Epoch 1/10 | Batch 100/500 | Loss: 0.4532 | LR: 3.00e-04
# [HEARTBEAT] Still training... Batch 250/500
```

## For Modal deploy/modal/app.py
Add this to the train function:
```python
# Force unbuffered output
env["PYTHONUNBUFFERED"] = "1"
```

---

**CRITICAL**: Without these changes, you're flying blind on Modal for hours. Implement AT LEAST the flush=True fixes before next training run!