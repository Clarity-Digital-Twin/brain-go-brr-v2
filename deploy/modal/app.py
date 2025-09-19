"""Modal cloud deployment for Brain-Go-Brr v2."""

import os
import subprocess
from pathlib import Path
from typing import Optional

import modal

# Get repo root for proper path resolution
REPO_ROOT = Path(__file__).resolve().parents[2]

# Build the Modal image with exact dependencies from pyproject.toml
image = (
    modal.Image.debian_slim(python_version="3.11")
    # Install PyTorch 2.2.2 with CUDA (required for mamba-ssm)
    .pip_install(
        "torch==2.2.2",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    # Install mamba-ssm (requires torch to be installed first)
    .pip_install("mamba-ssm>=2.0.0")
    # Core dependencies
    .pip_install(
        "numpy<2.0",  # mamba-ssm constraint
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "mne>=1.5.0",
        "pyedflib>=0.1.30",
        "einops>=0.7.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0.0",
        "click>=8.1.7",
        "rich>=13.0.0",
        "tqdm>=4.64.0",
        "pandas>=2.0.0",  # For eval extras
    )
    # Add project code
    .add_local_dir(str(REPO_ROOT / "src"), "/app/src")
    .add_local_dir(str(REPO_ROOT / "configs"), "/app/configs")
    .add_local_file(str(REPO_ROOT / "pyproject.toml"), "/app/pyproject.toml")
    .workdir("/app")
    # Install the project
    .run_commands("pip install -e .")
)

# Modal app configuration
app = modal.App(
    "brain-go-brr-v2",
    image=image,
    secrets=[
        # Uncomment if using W&B:
        # modal.Secret.from_name("wandb-secret"),
    ],
)

# Persistent storage volumes
data_volume = modal.Volume.from_name("brain-go-brr-data", create_if_missing=True)
results_volume = modal.Volume.from_name("brain-go-brr-results", create_if_missing=True)


@app.function(
    gpu="A100-80GB",  # 80GB VRAM, 3x faster than 4090
    timeout=7200,  # 2 hours
    volumes={
        "/data": data_volume,
        "/results": results_volume,
    },
    memory=32768,  # 32GB RAM
    cpu=8,
)
def train(
    config_path: str = "configs/smoke_test.yaml",
):
    """Run training on Modal GPU.

    Args:
        config_path: Path to config YAML file (relative to /app)

    Returns:
        Path to checkpoint file
    """
    import os
    import subprocess

    # Set environment
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONPATH"] = "/app"
    env["SEIZURE_MAMBA_FORCE_FALLBACK"] = "0"  # Use CUDA kernels
    # Speed up smoke tests by limiting files if not explicitly set
    env.setdefault("BGB_LIMIT_FILES", "50")

    # Prepare a temp config to ensure data/output point to persistent volumes
    import tempfile
    import yaml

    cfg_abs = config_path
    if not config_path.startswith("/"):
        cfg_abs = str(Path("/app") / config_path)

    with open(cfg_abs, "r") as f:
        data = yaml.safe_load(f)

    # Auto-select dataset under /data if present
    preferred_roots = [
        "/data/chb-mit",
        "/data/tuh_eeg_seizure_v2.0.0",
    ]
    for root in preferred_roots:
        if os.path.isdir(root):
            data.setdefault("data", {})["data_dir"] = root
            break

    # Force outputs and cache into /results volume
    exp = data.setdefault("experiment", {})
    out_name = Path(exp.get("output_dir", "results/run")).name
    exp["output_dir"] = f"/results/{out_name}"
    cache = exp.get("cache_dir", f"cache/{out_name}")
    exp["cache_dir"] = f"/results/{cache}" if not str(cache).startswith("/") else str(cache)

    Path(exp["output_dir"]).mkdir(parents=True, exist_ok=True)
    (Path(exp["output_dir"]) / "checkpoints").mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump(data, tmp)
        tmp_cfg = tmp.name

    # Build command - our CLI takes positional config only
    cmd = ["python", "-m", "src", "train", tmp_cfg]

    print(f"Running: {' '.join(cmd)}")
    print(f"Config: {config_path}")
    print("-" * 50)

    # Run training
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Training failed:\n{result.stderr}")
        raise RuntimeError(f"Training failed: {result.stderr[:500]}")

    print(f"Training completed:\n{result.stdout[-1000:]}")  # Last 1000 chars
    # Return best checkpoint path under /results
    checkpoint_dir = Path(data["experiment"]["output_dir"]) / "checkpoints"
    # Our training saves best.pt
    return str(checkpoint_dir / "best.pt")


@app.function(
    gpu="A100",  # A100 for evaluation
    timeout=3600,  # 1 hour
    volumes={
        "/data": data_volume,
        "/results": results_volume,
    },
)
def evaluate(
    checkpoint_path: str,
    dataset: str = "chb-mit",
):
    """Evaluate model on test dataset.

    Args:
        checkpoint_path: Path to model checkpoint
        dataset: Dataset name or path

    Returns:
        Path to metrics JSON file
    """
    import os
    import subprocess

    env = os.environ.copy()
    env["PYTHONPATH"] = "/app"
    env["CUDA_VISIBLE_DEVICES"] = "0"

    # Map dataset shortcuts to paths
    dataset_paths = {
        "chb-mit": "/data/chb-mit",
        "tuh": "/data/tuh_eeg_seizure_v2.0.0",
    }
    data_path = dataset_paths.get(dataset, dataset)

    # Output path
    output_json = "/results/evaluations/metrics.json"

    # Build command
    cmd = [
        "python", "-m", "src", "evaluate",
        checkpoint_path,
        data_path,
        "--output-json", output_json,
    ]

    print(f"Running: {' '.join(cmd)}")
    print("-" * 50)

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Evaluation failed: {result.stderr[:500]}")

    print(f"Evaluation complete. Metrics saved to: {output_json}")
    return output_json


@app.local_entrypoint()
def main(
    action: str = "train",
    config: str = "configs/smoke_test.yaml",
    detach: bool = False,
):
    """Modal deployment entrypoint.

    Examples:
        # Quick smoke test
        modal run deploy/modal/app.py --action train --config configs/smoke_test.yaml

        # Full training (detached)
        modal run deploy/modal/app.py --action train --config configs/production.yaml --detach

        # Evaluate checkpoint
        modal run deploy/modal/app.py --action evaluate --config /results/checkpoints/best.ckpt
    """
    print("🚀 Brain-Go-Brr v2 Modal Deployment")
    print("=" * 50)

    if action == "train":
        if detach:
            handle = train.spawn(config_path=config)
            print(f"Training started (detached). Run ID: {handle.object_id}")
            print(f"Monitor at: https://modal.com/apps/{handle.object_id}")
        else:
            result = train.remote(config_path=config)
            print(f"✓ Training complete. Checkpoint: {result}")

    elif action == "evaluate":
        # For evaluate, config arg is actually checkpoint path
        result = evaluate.remote(checkpoint_path=config)
        print(f"✓ Evaluation complete. Metrics: {result}")

    else:
        print(f"Unknown action: {action}")
        print("Available actions: train, evaluate")


if __name__ == "__main__":
    main()
