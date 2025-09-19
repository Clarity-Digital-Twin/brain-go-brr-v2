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
        # modal.Secret.from_name("wandb-secret"),  # Uncomment when W&B secret is created
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

    # Mamba-specific environment
    env["SEIZURE_MAMBA_FORCE_FALLBACK"] = "0"  # Use CUDA kernels

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
    env["SEIZURE_MAMBA_FORCE_FALLBACK"] = "0"

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
    import os

    env = os.environ.copy()
    env["PYTHONPATH"] = "/app"
    env["CUDA_VISIBLE_DEVICES"] = "0"

    cmd = [
        "python", "-m", "src", "evaluate",
        f"--checkpoint={checkpoint_path}",
        f"--dataset={dataset}",
        "--output-dir=/results/evaluations",
    ]

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Evaluation failed: {result.stderr}")

    print(f"Evaluation results:\n{result.stdout}")
    return result.stdout


@app.function(
    schedule=modal.Period(days=1),
    volumes={
        "/data": data_volume,
        "/results": results_volume,
    },
)
def scheduled_training():
    """Daily training run for continuous improvement."""
    train.remote(
        config_path="configs/daily_training.yaml",
        wandb_project="brain-go-brr-v2-scheduled",
    )


@app.function(
    gpu="L40S",
    timeout=1800,
    volumes={
        "/data": data_volume,
        "/results": results_volume,
    },
)
def hyperparameter_sweep(
    base_config: str = "configs/smoke_test.yaml",
    n_trials: int = 8,
):
    """Run hyperparameter sweep with multiple configurations."""
    import subprocess
    import os
    import json
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Define hyperparameter grid
    hyperparams = [
        {"lr": 1e-3, "batch_size": 32, "dropout": 0.1},
        {"lr": 5e-4, "batch_size": 32, "dropout": 0.2},
        {"lr": 1e-4, "batch_size": 64, "dropout": 0.1},
        {"lr": 5e-4, "batch_size": 64, "dropout": 0.3},
        {"lr": 1e-3, "batch_size": 16, "dropout": 0.2},
        {"lr": 5e-5, "batch_size": 32, "dropout": 0.1},
        {"lr": 1e-4, "batch_size": 16, "dropout": 0.3},
        {"lr": 5e-4, "batch_size": 128, "dropout": 0.2},
    ][:n_trials]

    results = []

    def run_trial(trial_id: int, params: dict):
        env = os.environ.copy()
        env["PYTHONPATH"] = "/app"
        env["CUDA_VISIBLE_DEVICES"] = "0"

        # Build command with hyperparameter overrides
        cmd = [
            "python", "-m", "src", "train",
            base_config,
            f"train.learning_rate={params['lr']}",
            f"train.batch_size={params['batch_size']}",
            f"model.dropout={params['dropout']}",
            f"experiment.wandb.name=trial_{trial_id}",
            "train.num_epochs=1",  # Quick sweep
        ]

        print(f"Trial {trial_id}: {params}")
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)

        if result.returncode == 0:
            # Parse validation loss from output (you'll need to implement this)
            val_loss = 0.5  # Placeholder
            return {"trial": trial_id, "params": params, "val_loss": val_loss, "status": "success"}
        else:
            return {"trial": trial_id, "params": params, "status": "failed", "error": result.stderr}

    # Run trials in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(run_trial, i, params): i
            for i, params in enumerate(hyperparams)
        }

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"Completed trial {result['trial']}: {result.get('val_loss', 'N/A')}")

    # Find best configuration
    successful_trials = [r for r in results if r["status"] == "success"]
    if successful_trials:
        best_trial = min(successful_trials, key=lambda x: x["val_loss"])
        print(f"\nBest trial: {best_trial['trial']} with params: {best_trial['params']}")
        print(f"Best validation loss: {best_trial['val_loss']}")

    # Save results
    with open("/results/hyperparameter_sweep.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


@app.local_entrypoint()
def main(
    action: str = "train",
    config: str = "configs/smoke_test.yaml",
    epochs: Optional[int] = None,
    detach: bool = False,
):
    """Local entrypoint for Modal deployment.

    Usage:
        # Quick smoke test
        modal run modal_train.py --action train --config configs/smoke_test.yaml --epochs 1

        # Full training (detached)
        modal run modal_train.py --action train --config configs/full_training.yaml --detach

        # Large-scale multi-GPU
        modal run modal_train.py --action train-large --config configs/full_training.yaml --detach

        # Hyperparameter sweep
        modal run modal_train.py --action sweep

        # Evaluate model
        modal run modal_train.py --action evaluate --config /results/checkpoints/best.ckpt
    """
    if action == "train":
        if detach:
            # Use .spawn() for detached execution
            handle = train.spawn(
                config_path=config,
                num_epochs=epochs,
            )
            print(f"Training started in detached mode. Run ID: {handle.object_id}")
        else:
            result = train.remote(
                config_path=config,
                num_epochs=epochs,
            )
            print(f"Training completed. Checkpoint: {result}")

    elif action == "train-large":
        if detach:
            handle = train_large_scale.spawn(config_path=config)
            print(f"Large-scale training started. Run ID: {handle.object_id}")
        else:
            result = train_large_scale.remote(config_path=config)
            print(f"Large-scale training completed. Checkpoint: {result}")

    elif action == "sweep":
        results = hyperparameter_sweep.remote(base_config=config)
        print(f"Hyperparameter sweep completed with {len(results)} trials")

    elif action == "evaluate":
        # Config arg contains checkpoint path for evaluate action
        result = evaluate.remote(checkpoint_path=config)
        print(f"Evaluation completed:\n{result}")

    else:
        print(f"Unknown action: {action}")
        print("Available actions: train, train-large, sweep, evaluate")


if __name__ == "__main__":
    # Allow direct Python execution for debugging
    import sys
    if len(sys.argv) > 1:
        main(action=sys.argv[1])
    else:
        main()