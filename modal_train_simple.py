"""Simplified Modal deployment for Brain-Go-Brr v2 training."""

import os
import subprocess
from pathlib import Path
from typing import Optional

import modal

# Create the image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    # Install PyTorch with CUDA support
    .pip_install(
        "torch==2.2.2",
        "torchvision",
        "torchaudio",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    # Core dependencies
    .pip_install(
        "numpy<2.0",  # Required for mamba-ssm
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
        "pandas>=2.0.0",
    )
    # Add local code to the image
    .add_local_dir("src", "/app/src")
    .add_local_dir("configs", "/app/configs")
    .add_local_file("pyproject.toml", "/app/pyproject.toml")
    .workdir("/app")
    # Install the project
    .run_commands("pip install -e .")
)

# Create Modal app
app = modal.App("brain-go-brr-v2-simple", image=image)

# Define volumes for data persistence
data_volume = modal.Volume.from_name("brain-go-brr-data", create_if_missing=True)
results_volume = modal.Volume.from_name("brain-go-brr-results", create_if_missing=True)


@app.function(
    gpu="T4",  # Start with T4 for testing (cheaper)
    timeout=3600,  # 1 hour timeout
    volumes={
        "/data": data_volume,
        "/results": results_volume,
    },
)
def train_smoke_test():
    """Run a quick smoke test training."""

    # Set environment variables
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONPATH"] = "/app"

    # For testing, let's first check if our code is there
    print("Checking directory structure:")
    result = subprocess.run(["ls", "-la", "/app/"], capture_output=True, text=True)
    print(result.stdout)

    print("\nChecking src directory:")
    result = subprocess.run(["ls", "-la", "/app/src/"], capture_output=True, text=True)
    print(result.stdout)

    print("\nChecking configs:")
    result = subprocess.run(["ls", "-la", "/app/configs/"], capture_output=True, text=True)
    print(result.stdout)

    # Check if we can import our module
    print("\nTrying to import brain_brr:")
    try:
        import sys
        sys.path.insert(0, "/app")
        import src.brain_brr
        print("✓ Successfully imported brain_brr")
        print(f"  Version: {src.brain_brr.__version__}")
    except Exception as e:
        print(f"✗ Failed to import: {e}")

    # Check CUDA availability
    print("\nChecking CUDA:")
    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
    except Exception as e:
        print(f"  Error: {e}")

    # Try to run the actual training (simplified)
    print("\n" + "="*50)
    print("Starting training smoke test...")
    print("="*50)

    cmd = [
        "python", "-m", "src", "train",
        "configs/smoke_test.yaml"
    ]

    print(f"Running command: {' '.join(cmd)}")

    # Run with limited output for now
    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=300  # 5 minute timeout for smoke test
    )

    if result.returncode != 0:
        print(f"\n✗ Training failed with error:")
        print(result.stderr[:2000])  # First 2000 chars of error
        return {"status": "failed", "error": result.stderr[:1000]}

    print("\n✓ Training smoke test completed successfully!")
    print(result.stdout[:2000])  # First 2000 chars of output

    return {"status": "success", "message": "Smoke test completed"}


@app.local_entrypoint()
def main():
    """Run the smoke test."""
    print("🚀 Starting Modal deployment for Brain-Go-Brr v2")
    print("=" * 50)

    result = train_smoke_test.remote()

    print("\n" + "=" * 50)
    print("Result:", result)
    print("=" * 50)


if __name__ == "__main__":
    main()