"""Modal deployment for Brain-Go-Brr v2 training (aligned with repo CLI)."""

from pathlib import Path
from typing import Optional

import modal
from modal import gpu

# Define image with dependencies matching pyproject pins
# - Torch pinned to 2.2.x for mamba-ssm compatibility
# - Preinstall CUDA wheel (cu121) before installing our package
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # Install CUDA-enabled Torch first (per pyproject: >=2.2.2,<2.3.0)
        "torch==2.2.2",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    # Mamba-SSM GPU extra (prebuilt wheels)
    .pip_install("mamba-ssm>=2.0.0")
    # Copy project code
    .copy_local_dir("src", "/app/src")
    .copy_local_dir("configs", "/app/configs")
    .copy_local_file("pyproject.toml", "/app/pyproject.toml")
    .workdir("/app")
    # Install project (will bring in remaining deps from pyproject)
    .run_commands("pip install -e .[eval]")
)

# Create Modal app
app = modal.App("brain-go-brr-v2", image=image, secrets=[])

# Define volumes for data persistence
data_volume = modal.Volume.from_name("brain-go-brr-data", create_if_missing=True)
results_volume = modal.Volume.from_name("brain-go-brr-results", create_if_missing=True)


@app.function(
    gpu=gpu.L40S(),
    timeout=7200,
    volumes={
        "/data": data_volume,
        "/results": results_volume,
    },
    memory=32768,
    cpu=8,
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

    # Build command (our CLI takes positional config path only)
    cmd = ["python", "-m", "src", "train", config_path]

    print(f"Running command: {' '.join(cmd)}")

    # Run training
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Training failed with error:\n{result.stderr}")
        raise RuntimeError(f"Training failed: {result.stderr}")

    print(f"Training completed successfully:\n{result.stdout}")

    # Return checkpoint path
    return "/results/checkpoints/best.ckpt"


# Note: Multi-GPU DDP training is not wired in this repo's CLI.
# Add a distributed entrypoint here only after the training loop supports DDP.


@app.function(
    gpu=gpu.A100(),
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
    import os
    import subprocess

    env = os.environ.copy()
    env["PYTHONPATH"] = "/app"
    env["CUDA_VISIBLE_DEVICES"] = "0"

    # Map dataset to a default path under the mounted volume
    dataset_paths = {
        "chb-mit": "/data/chb-mit",
        "tuh": "/data/tuh_eeg_seizure_v2.0.0",
    }
    data_path = dataset_paths.get(dataset, dataset)

    # Save metrics here
    output_json = "/results/evaluations/metrics.json"

    cmd = [
        "python",
        "-m",
        "src",
        "evaluate",
        checkpoint_path,
        data_path,
        "--output-json",
        output_json,
    ]

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Evaluation failed: {result.stderr}")

    print(f"Evaluation results saved to: {output_json}")
    return output_json


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


# Note: Hyperparameter sweeps via CLI overrides are not supported by this repo's CLI.
# Consider integrating a config mutator or W&B sweeps before enabling this.


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

    elif action == "evaluate":
        # Config arg contains checkpoint path for evaluate action
        result = evaluate.remote(checkpoint_path=config)
        print(f"Evaluation completed:\n{result}")

    else:
        print(f"Unknown action: {action}")
        print("Available actions: train, evaluate")


if __name__ == "__main__":
    # Allow direct Python execution for debugging
    import sys
    if len(sys.argv) > 1:
        main(action=sys.argv[1])
    else:
        main()
