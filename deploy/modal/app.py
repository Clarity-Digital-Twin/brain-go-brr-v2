"""Modal cloud deployment for Brain-Go-Brr v2."""

import os
import subprocess
from pathlib import Path
from typing import Optional

import modal

# Build the Modal image with CUDA development tools for mamba-ssm compilation
# Use lazy evaluation - only build when Modal needs it
image = (
    # Use NVIDIA CUDA devel image for nvcc compiler (required by mamba-ssm)
    modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])  # Clear entrypoint from CUDA image
    # Install build tools required for compiling CUDA extensions
    .apt_install("build-essential", "ninja-build")
    # Install PyTorch 2.2.2 with CUDA 12.1
    .pip_install(
        "torch==2.2.2",
        "torchvision==0.17.2",
        "numpy<2.0",  # mamba-ssm constraint
        index_url="https://download.pytorch.org/whl/cu121",
    )
    # Install mamba-ssm with CUDA kernels (nvcc now available)
    # Install build dependencies first
    .pip_install("packaging", "wheel", "setuptools")
    # Critical: Use CC and CXX env vars to specify compiler, install with verbose output
    .run_commands(
        "export CC=gcc CXX=g++ && pip install -v --no-build-isolation 'mamba-ssm>=2.0.0'"
    )
    # Core dependencies
    .pip_install(
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
        "tensorboard>=2.10.0",  # For training metrics
    )
    # Set working directory before adding local files
    .workdir("/app")
    # Add project code - MUST be last for Modal image caching
    # Use Path to resolve relative to script location
    .add_local_dir(str(Path(__file__).parent.parent.parent / "src"), "/app/src")
    .add_local_dir(str(Path(__file__).parent.parent.parent / "configs"), "/app/configs")
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

# S3 bucket mount for massive EEG data
s3_secret = modal.Secret.from_name("aws-s3-secret")
data_mount = modal.CloudBucketMount(
    "brain-go-brr-eeg-data-20250919",  # Your actual bucket!
    secret=s3_secret,
    key_prefix="tusz/",  # Mount just the TUH data (matches actual upload path)
    read_only=True,  # EEG data is read-only
)

# Persistent volumes for results
data_volume = modal.Volume.from_name("brain-go-brr-data", create_if_missing=True)
results_volume = modal.Volume.from_name("brain-go-brr-results", create_if_missing=True)


@app.function(
    gpu="A100-80GB",  # 80GB VRAM, 3x faster than 4090
    timeout=86400,  # 24 hours max (Modal limit)
    volumes={
        "/data": data_mount,  # S3 bucket with TUH data!
        "/results": results_volume,
    },
    memory=32768,  # 32GB RAM
    cpu=8,
)
def train(
    config_path: str = "configs/tusz_train_a100.yaml",  # A100-optimized config
    resume: bool = False,  # Resume training from last.pt in output_dir
):
    """Run training on Modal GPU.

    Args:
        config_path: Path to config YAML file (relative to /app)

    Returns:
        Path to checkpoint file
    """
    import os
    import subprocess

    # Test mamba-ssm import
    try:
        import mamba_ssm
        print(f"✓ Mamba-SSM imported successfully: {mamba_ssm.__version__}")
    except ImportError as e:
        print(f"⚠️ Mamba-SSM import failed: {e}")

    # Set environment
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONPATH"] = "/app"
    env["PYTHONUNBUFFERED"] = "1"  # CRITICAL: Force unbuffered output for real-time logs
    env["SEIZURE_MAMBA_FORCE_FALLBACK"] = "0"  # Use CUDA kernels
    # Only limit files for smoke tests
    if "smoke" in config_path.lower():
        env.setdefault("BGB_LIMIT_FILES", "50")
    # For production, use full dataset (no limit)

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
        "/data/edf/train",  # S3 mounted path: /data/tusz/edf/train
        "/data",  # Fallback to root of mount
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

    # Use built-in resume mechanism (relies on last.pt in output_dir/checkpoints)
    if resume:
        cmd.append("--resume")
        print("Resuming training from last.pt if present in output_dir")

    print(f"Running: {' '.join(cmd)}")
    print(f"Config: {config_path}")
    print("-" * 50)

    # Run training
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Training failed:\nSTDOUT:\n{result.stdout[-2000:]}\n\nSTDERR:\n{result.stderr[-2000:]}")
        raise RuntimeError(f"Training failed with exit code {result.returncode}")

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
    config: str = "configs/smoke_test.yaml",  # Default to smoke test for safety
    resume: bool = False,  # Resume training from last.pt
):
    """Modal deployment entrypoint.

    ⚠️ CRITICAL: Modal's --detach flag MUST go BEFORE the script name!

    CORRECT:  modal run --detach deploy/modal/app.py -- --action train ...
    WRONG:    modal run deploy/modal/app.py --action train ... --detach

    Examples:
        # Quick smoke test (Modal's --detach prevents disconnection)
        modal run --detach deploy/modal/app.py -- --action train --config configs/smoke_test.yaml

        # Full A100 training (Modal's --detach prevents disconnection)
        modal run --detach deploy/modal/app.py -- --action train --config configs/tusz_train_a100.yaml

        # Resume training from last.pt in output_dir
        modal run --detach deploy/modal/app.py -- --action train --resume true

        # Evaluate checkpoint
        modal run deploy/modal/app.py -- --action evaluate --config /results/checkpoints/best.ckpt
    """
    print("🚀 Brain-Go-Brr v2 Modal Deployment")
    print("=" * 50)

    if action == "train":
        # Always use train.remote() - Modal's --detach flag controls app lifecycle
        result = train.remote(config_path=config, resume=resume)
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
