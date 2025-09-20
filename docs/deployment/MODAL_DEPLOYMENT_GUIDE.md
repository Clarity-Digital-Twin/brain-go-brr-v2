# Modal.com Deployment Guide for Brain-Go-Brr v2

Last updated: 2025-09-19

> IMPORTANT: Archived — contains legacy examples (modal_train.py, L40S/H100 multi-GPU flows).
> For the current, supported deployment, use `deploy/modal/app.py` and follow
> `docs/deployment/MODAL_DEPLOYMENT_COMPLETE_GUIDE.md`.

Canonical commands:
```
# Smoke
modal run deploy/modal/app.py --action train --config configs/smoke_test.yaml

# Full A100 training
modal run deploy/modal/app.py --action train --config configs/tusz_train_a100.yaml --detach

# Evaluate (checkpoint on Modal volume)
modal run deploy/modal/app.py --action evaluate --config /results/checkpoints/best.ckpt
```

## 🚀 Quick Start

This guide covers deploying Brain-Go-Brr v2's EEG seizure detection training to Modal.com's GPU cloud infrastructure.

### Prerequisites

1. **Install Modal CLI** (v1.1.4+):
```bash
pip install --upgrade modal
modal --version  # Should show 1.1.4+
```

2. **Authenticate with Modal**:
```bash
modal setup
```

3. **Create Modal Token** (for CI/CD):
```bash
modal token new
```

## 📦 Project Structure for Modal

Create `modal_train.py` in project root:

```python
"""Modal deployment for Brain-Go-Brr v2 training."""

import modal
from pathlib import Path
from typing import Optional

# Define image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    # Core ML dependencies
    .uv_pip_install(
        "torch==2.6.0",
        "torchvision",
        "torchaudio",
        find_links="https://download.pytorch.org/whl/cu124"  # CUDA 12.4
    )
    # Project dependencies
    .uv_pip_install(
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "h5py",
        "pydantic>=2.5",
        "pydantic-settings>=2.1",
        "wandb",
        "lightning>=2.0",
        "hydra-core>=1.3",
        "omegaconf>=2.3",
        "rich>=13.0",
        "click>=8.1",
        "mne>=1.7",
    )
    # Install Mamba-SSM for CUDA (requires GPU at build time)
    .pip_install(
        "mamba-ssm @ git+https://github.com/state-spaces/mamba.git@v2.2.0",
        gpu="A100"  # Build with A100 for compatibility
    )
    # Copy project code
    .copy_local_dir("src", "/app/src")
    .copy_local_dir("configs", "/app/configs")
    .copy_local_file("pyproject.toml", "/app/pyproject.toml")
    .workdir("/app")
    # Install project in editable mode
    .run_commands("pip install -e .")
)

# Create Modal app
app = modal.App(
    "brain-go-brr-v2",
    image=image,
    secrets=[
        modal.Secret.from_name("wandb-secret"),  # Add W&B API key
    ],
)

# Define volumes for data persistence
data_volume = modal.Volume.from_name("brain-go-brr-data", create_if_missing=True)
results_volume = modal.Volume.from_name("brain-go-brr-results", create_if_missing=True)

@app.function(
    gpu="L40S",  # 48GB VRAM, good for Mamba-2 + U-Net
    timeout=7200,  # 2 hours timeout
    volumes={
        "/data": data_volume,
        "/results": results_volume,
    },
    memory=32768,  # 32GB RAM
    cpu=8,  # 8 CPU cores
)
def train(
    config_path: str = "configs/smoke_test.yaml",
    wandb_project: Optional[str] = None,
    num_epochs: Optional[int] = None,
):
    """Run training on Modal GPU.

    Args:
        config_path: Path to Hydra config file
        wandb_project: Override W&B project name
        num_epochs: Override number of epochs
    """
    import subprocess
    import os

    # Set environment variables
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONPATH"] = "/app"

    # Build command
    cmd = ["python", "-m", "src", "train", config_path]

    # Add overrides
    overrides = []
    if wandb_project:
        overrides.append(f"experiment.wandb.project={wandb_project}")
    if num_epochs:
        overrides.append(f"train.num_epochs={num_epochs}")

    if overrides:
        cmd.extend(overrides)

    print(f"Running command: {' '.join(cmd)}")

    # Run training
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Training failed with error:\n{result.stderr}")
        raise RuntimeError(f"Training failed: {result.stderr}")

    print(f"Training completed successfully:\n{result.stdout}")

    # Return checkpoint path
    return "/results/checkpoints/best.ckpt"


@app.function(
    gpu="H100:8",  # 8x H100 for large-scale training
    timeout=28800,  # 8 hours
    volumes={
        "/data": data_volume,
        "/results": results_volume,
    },
    memory=131072,  # 128GB RAM
    cpu=32,
)
def train_large_scale(
    config_path: str = "configs/full_training.yaml",
    wandb_project: str = "brain-go-brr-v2-production",
):
    """Large-scale multi-GPU training on H100s.

    This uses 8x H100 GPUs for distributed training with PyTorch DDP.
    """
    import subprocess
    import os

    env = os.environ.copy()
    env["PYTHONPATH"] = "/app"
    env["MASTER_ADDR"] = "localhost"
    env["MASTER_PORT"] = "12355"
    env["WORLD_SIZE"] = "8"

    # Use torchrun for distributed training
    cmd = [
        "torchrun",
        "--nproc_per_node=8",
        "--master_port=12355",
        "-m", "src", "train",
        config_path,
        f"experiment.wandb.project={wandb_project}",
        "train.strategy=ddp",
        "train.devices=8",
    ]

    print(f"Running distributed training: {' '.join(cmd)}")

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Distributed training failed: {result.stderr}")

    return "/results/checkpoints/best.ckpt"


@app.function(
    gpu="A100",  # Single A100 for evaluation
    timeout=3600,
    volumes={
        "/data": data_volume,
        "/results": results_volume,
    },
)
def evaluate(
    checkpoint_path: str,
    dataset: str = "chb-mit",
):
    """Evaluate trained model on test dataset.

    Args:
        checkpoint_path: Path to model checkpoint
        dataset: Dataset to evaluate on
    """
    import subprocess

    cmd = [
        "python", "-m", "src", "evaluate",
        f"--checkpoint={checkpoint_path}",
        f"--dataset={dataset}",
        "--output-dir=/results/evaluations",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Evaluation failed: {result.stderr}")

    print(f"Evaluation results:\n{result.stdout}")
    return result.stdout


@app.function(schedule=modal.Period(days=1))
def scheduled_training():
    """Daily training run for continuous improvement."""
    train.remote(
        config_path="configs/daily_training.yaml",
        wandb_project="brain-go-brr-v2-scheduled",
    )


@app.local_entrypoint()
def main(
    action: str = "train",
    config: str = "configs/smoke_test.yaml",
    epochs: Optional[int] = None,
):
    """Local entrypoint for Modal deployment.

    Usage:
        modal run modal_train.py --action train --config configs/smoke_test.yaml
        modal run modal_train.py --action train-large --config configs/full_training.yaml
        modal run modal_train.py --action evaluate --checkpoint /results/checkpoints/best.ckpt
    """
    if action == "train":
        result = train.remote(
            config_path=config,
            num_epochs=epochs,
        )
        print(f"Training completed. Checkpoint: {result}")

    elif action == "train-large":
        result = train_large_scale.remote(config_path=config)
        print(f"Large-scale training completed. Checkpoint: {result}")

    elif action == "evaluate":
        # Expects checkpoint path in config arg for simplicity
        result = evaluate.remote(checkpoint_path=config)
        print(f"Evaluation completed:\n{result}")

    else:
        print(f"Unknown action: {action}")
        print("Available actions: train, train-large, evaluate")
```

## 🎯 Running Training on Modal

### 1. Quick Smoke Test (L40S GPU)
```bash
# Run 1 epoch smoke test
modal run modal_train.py --action train --config configs/smoke_test.yaml --epochs 1
```

### 2. Full Training (L40S GPU)
```bash
# Run full training with default config
modal run --detach modal_train.py --action train --config configs/full_training.yaml
```

### 3. Large-Scale Training (8x H100)
```bash
# Run distributed training on 8 H100s
modal run --detach modal_train.py --action train-large --config configs/full_training.yaml
```

### 4. Evaluation
```bash
# Evaluate trained model
modal run modal_train.py --action evaluate --checkpoint /results/checkpoints/best.ckpt
```

## 💰 Cost Optimization

### GPU Selection Guide

| Use Case | Recommended GPU | Cost/Hour | Memory | Notes |
|----------|----------------|-----------|---------|-------|
| Development/Testing | T4 | $0.59 | 16GB | Cheapest, good for debugging |
| Smoke Tests | L4 | $0.81 | 24GB | Better performance than T4 |
| Standard Training | L40S | $3.99 | 48GB | **Best value for Mamba-2** |
| Fast Training | A100-40GB | $3.99 | 40GB | Tensor cores, fast |
| Large Models | A100-80GB | $5.59 | 80GB | Maximum memory |
| Production | H100 | $8.99 | 80GB | Fastest single GPU |
| Scale Training | H100:8 | $71.92 | 640GB | Multi-GPU training |

### Cost Optimization Tips

1. **Use Spot Instances** (when available):
```python
@app.function(gpu="L40S", spot=True)  # Save ~70% on cost
```

2. **Optimize Batch Size**:
- L40S (48GB): batch_size=32-64
- A100-40GB: batch_size=24-48
- A100-80GB: batch_size=64-128

3. **Use Checkpointing**:
```python
# Save checkpoints to persistent volume
volumes={"/results": results_volume}
```

4. **Profile Before Scaling**:
- Start with L40S for initial experiments
- Only scale to H100 if compute-bound (not memory-bound)

## 📊 Data Management

### Upload Training Data
```bash
# Upload TUSZ dataset to Modal volume
modal volume put brain-go-brr-data ./data/tuh_eeg_seizure_v2.0.0 /tuh_eeg_seizure_v2.0.0

# Upload CHB-MIT for validation
modal volume put brain-go-brr-data ./data/chb-mit /chb-mit
```

### Download Results
```bash
# Download trained checkpoints
modal volume get brain-go-brr-results /checkpoints ./results/checkpoints

# Download evaluation metrics
modal volume get brain-go-brr-results /evaluations ./results/evaluations
```

## 🔧 Advanced Configuration

### Environment Variables
```python
@app.function(
    env={
        "SEIZURE_MAMBA_FORCE_FALLBACK": "0",  # Use CUDA kernels
        "BGB_LIMIT_FILES": "100",  # Process more files
        "WANDB_MODE": "online",  # Enable W&B logging
    }
)
```

### Multi-Node Training (Beta)
```python
# Contact Modal support to enable multi-node training
@app.function(
    gpu="H100:8",
    n_nodes=4,  # 32 H100s total
)
```

### Custom CUDA Kernels
```python
# Build custom CUDA extensions
image = image.run_commands(
    "cd /app && python setup.py build_ext --inplace",
    gpu="A100"  # Build on GPU
)
```

## 🚦 Monitoring & Debugging

### View Logs
```bash
# Stream logs in real-time
modal logs -f brain-go-brr-v2

# View specific run logs
modal logs <run-id>
```

### Monitor GPU Usage
```python
@app.function()
def monitor_gpu():
    import subprocess
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    print(result.stdout)
```

### Debug Interactively
```bash
# Open interactive shell in container
modal shell brain-go-brr-v2 --gpu L40S
```

## 📈 Performance Benchmarks

Expected training times on Modal GPUs:

| Dataset | Config | L40S | A100-40GB | H100 | H100:8 |
|---------|--------|------|-----------|------|--------|
| Smoke Test (1 epoch) | 1% data | 5 min | 3 min | 2 min | 30 sec |
| Small (10 epochs) | 10% data | 50 min | 30 min | 20 min | 5 min |
| Full TUSZ | 100% data | 8 hrs | 5 hrs | 3 hrs | 45 min |

## 🔐 Security & Secrets

### Add W&B API Key
```bash
modal secret create wandb-secret WANDB_API_KEY=<your-key>
```

### Add HuggingFace Token (if needed)
```bash
modal secret create huggingface-secret HF_TOKEN=<your-token>
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size
   - Use gradient accumulation
   - Switch to larger GPU (A100-80GB or H100)

2. **Mamba CUDA Kernel Issues**:
   - Set `SEIZURE_MAMBA_FORCE_FALLBACK=1` to use Conv1d fallback
   - Ensure CUDA 12.4+ compatibility

3. **Data Loading Slow**:
   - Increase `num_workers` in config
   - Use Modal volumes for faster I/O
   - Pre-process data into HDF5 format

4. **Connection Timeouts**:
   - Use `--detach` for long-running jobs
   - Increase timeout in function decorator

## 📚 Resources

- [Modal Documentation](https://modal.com/docs)
- [Modal GPU Guide](https://modal.com/docs/guide/gpu)
- [Modal Examples](https://modal.com/docs/examples)
- [Modal Discord](https://discord.gg/modal)
- [Brain-Go-Brr v2 Repo](https://github.com/yourusername/brain-go-brr-v2)

## 🎯 Next Steps

1. **Set up Modal account**: `modal setup`
2. **Create secrets**: Add W&B API key
3. **Upload data**: Transfer datasets to Modal volumes
4. **Run smoke test**: Verify setup works
5. **Scale training**: Move to larger GPUs for production

---

**Mission**: Leverage Modal's elastic GPU infrastructure to achieve O(N) clinical seizure detection at scale! 🚀
