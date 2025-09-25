"""Modal cloud deployment for Brain-Go-Brr v2."""

import os
import subprocess
from pathlib import Path
from typing import Optional

import modal

# Build the Modal image with CUDA development tools for mamba-ssm compilation
# CRITICAL: Must match EXACT versions from local setup (docs/03-operations/setup-guide.md)
image = (
    # Use NVIDIA CUDA devel image for nvcc compiler (required by mamba-ssm)
    modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])  # Clear entrypoint from CUDA image
    # Install build tools required for compiling CUDA extensions
    .apt_install("build-essential", "ninja-build", "git")
    # Set CUDA environment variables BEFORE any pip installs
    .env({
        "CUDA_HOME": "/usr/local/cuda-12.1",
        "PATH": "/usr/local/cuda-12.1/bin:$PATH",
        "LD_LIBRARY_PATH": "/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH",
        "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9;9.0",  # A100 is 8.0
    })
    # CRITICAL: Install EXACT PyTorch version from specific index
    # Modal's mirror can have wrong versions, so we force PyTorch index
    .run_commands(
        "pip install torch==2.2.2 torchvision==0.17.2 'numpy<2.0' --index-url https://download.pytorch.org/whl/cu121"
    )
    # Verify PyTorch is correct version (CUDA check happens at runtime, not build time)
    .run_commands(
        "python -c 'import torch; assert torch.__version__.startswith(\"2.2.2\"), f\"Wrong torch: {torch.__version__}\"'"
    )
    # Install build dependencies
    .pip_install("packaging", "wheel", "setuptools")
    # CRITICAL: Install EXACT versions with forced compilation
    # These MUST match local setup exactly (see setup-guide.md)
    .run_commands(
        "pip install --no-build-isolation --no-cache-dir causal-conv1d==1.4.0"
    )
    .run_commands(
        "pip install --no-build-isolation --no-cache-dir mamba-ssm==2.2.2"
    )
    # Verify mamba-ssm imports correctly (CUDA test happens at runtime)
    .run_commands(
        "python -c 'from mamba_ssm import Mamba2; print(\"✅ Mamba2 imports successfully\")'"
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
        "wandb",  # Weights & Biases for cloud tracking
        "pytorch-tcn",  # TCN implementation for optimal performance
    )
    # CRITICAL: Install PyTorch Geometric with exact versions for PyTorch 2.2.2 + CUDA 12.1
    # These MUST match our local setup exactly!
    .run_commands(
        "pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html"
    )
    .run_commands(
        "pip install torch-geometric==2.6.1"
    )
    # Verify PyG imports correctly
    .run_commands(
        "python -c 'import torch_geometric; print(f\"✅ PyG {torch_geometric.__version__} installed\")'"
    )
    # Set working directory before adding local files
    .workdir("/app")
    # Add project code - MUST be last for Modal image caching
    # Use Path to resolve relative to script location
    .add_local_dir(str(Path(__file__).parent.parent.parent / "src"), "/app/src")
    .add_local_dir(str(Path(__file__).parent.parent.parent / "configs"), "/app/configs")
    .add_local_dir(str(Path(__file__).parent), "/app/deploy/modal")  # Add deploy scripts
)

# Modal app configuration
app = modal.App(
    "brain-go-brr-v2",
    image=image,
    secrets=[
        # W&B tracking (optional - create "wandb-secret" in Modal dashboard):
        modal.Secret.from_name("wandb-secret"),
    ],
)

# S3 bucket mounts for EEG data and cache
s3_secret = modal.Secret.from_name("aws-s3-secret")

# Raw EDF data mount
data_mount = modal.CloudBucketMount(
    "brain-go-brr-eeg-data-20250919",  # Your actual bucket!
    secret=s3_secret,
    key_prefix="tusz/",  # Raw EDF data: tusz/{train,dev,eval}/
    read_only=True,  # EEG data is read-only
)

# Pre-built cache mount (will be added after local cache upload)
# Uncomment after uploading cache to S3:
# cache_mount = modal.CloudBucketMount(
#     "brain-go-brr-eeg-data-20250919",
#     secret=s3_secret,
#     key_prefix="cache/tusz/",  # Preprocessed NPZ cache
#     read_only=True,
# )

# Persistent volume for results and cache (310GB currently)
results_volume = modal.Volume.from_name("brain-go-brr-results", create_if_missing=True)
# NOTE: brain-go-brr-data volume deleted - it was empty and unused


# CPU-only: cache cleanup should not consume a GPU
@app.function(
    timeout=600,  # 10 min to include cache clean
    cpu=4,
    memory=4096,
    volumes={"/results": results_volume},  # Need volume for cache operations
)
def clean_cache():
    """Clean contaminated cache from before patient-disjoint fix."""
    import shutil
    from pathlib import Path

    print("\n" + "=" * 60)
    print("[CACHE CLEAN] Starting cache cleanup...")
    print("=" * 60)

    cache_paths = [
        Path("/results/cache/tusz"),
        Path("/results/cache/smoke"),
    ]

    for cache_path in cache_paths:
        if cache_path.exists():
            print(f"[CLEAN] Removing {cache_path}...")
            shutil.rmtree(cache_path, ignore_errors=True)
            print(f"[CLEAN] ✅ Removed {cache_path}")
        else:
            print(f"[CLEAN] Path does not exist: {cache_path}")

    # Recreate clean directories
    for cache_path in cache_paths:
        cache_path.mkdir(parents=True, exist_ok=True)
        (cache_path / "train").mkdir(exist_ok=True)
        (cache_path / "dev").mkdir(exist_ok=True)
        print(f"[CLEAN] ✅ Created clean structure: {cache_path}/{{train,dev}}/")

    print("\n[CACHE CLEAN] ✅ Cache cleanup complete!")
    print("Next training run will rebuild cache with patient-disjoint splits.")
    print("=" * 60 + "\n")
    return True


@app.function(
    gpu="A100",
    timeout=300,  # 5 min test
    cpu=16,  # Safe: 16 cores for testing
    memory=32768,  # Safe: 32GB RAM for tests
)
def test_mamba_cuda():
    """Test that Mamba CUDA kernels work properly."""
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    print(f"CUDA device: {torch.cuda.get_device_name()}", flush=True)

    # Test mamba-ssm import
    try:
        import mamba_ssm
        print(f"✓ mamba-ssm version: {mamba_ssm.__version__}", flush=True)
    except ImportError as e:
        print(f"✗ mamba-ssm import failed: {e}", flush=True)
        return False

    # Test causal_conv1d import (the actual CUDA kernels)
    try:
        import causal_conv1d
        print(f"✓ causal-conv1d imported", flush=True)
    except ImportError as e:
        print(f"✗ causal-conv1d import failed: {e}", flush=True)
        return False

    # Test Mamba2 creation and forward pass
    try:
        from mamba_ssm import Mamba2

        # Create a simple Mamba2 layer
        model = Mamba2(d_model=512, d_state=16, d_conv=4, expand=2).cuda()
        print("✓ Mamba2 model created", flush=True)

        # Test forward pass (no grad for speed)
        x = torch.randn(2, 100, 512).cuda()  # (batch, seq_len, d_model)
        with torch.no_grad():
            out = model(x)
        print(f"✓ Forward pass successful! Output shape: {out.shape}", flush=True)

        # Test backward pass (needs grad enabled)
        x_grad = torch.randn(2, 100, 512, requires_grad=True).cuda()
        out_grad = model(x_grad)
        loss = out_grad.sum()
        loss.backward()
        print("✓ Backward pass successful!", flush=True)

        return True

    except Exception as e:
        print(f"✗ Mamba2 test failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False


@app.function(
    gpu="A100-80GB",  # 80GB VRAM, 3x faster than 4090
    timeout=86400,  # 24 hours max (Modal limit)
    volumes={
        "/data": data_mount,  # S3 bucket with TUH data!
        "/results": results_volume,
    },
    memory=98304,  # SAFE: 96GB RAM (was 32GB, now 3x for safety)
    cpu=24,  # SAFE: 24 CPU cores (3 cores per 8 DataLoader workers)
)
def train(
    config_path: str = "configs/modal/smoke.yaml",  # Default to smoke test for safety
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

    # CRITICAL: Verify patient disjointness in data
    print("\n" + "=" * 60, flush=True)
    print("[PATIENT DISJOINTNESS] Verifying TUSZ splits...", flush=True)
    print("=" * 60, flush=True)

    from pathlib import Path
    train_dir = Path("/data/edf/train")
    dev_dir = Path("/data/edf/dev")

    if train_dir.exists() and dev_dir.exists():
        train_patients = {p.name for p in train_dir.iterdir() if p.is_dir()}
        dev_patients = {p.name for p in dev_dir.iterdir() if p.is_dir()}
        overlap = train_patients & dev_patients

        if overlap:
            raise RuntimeError(
                f"CRITICAL: Patient leakage detected! {len(overlap)} patients in both splits:\n"
                f"  {sorted(overlap)[:10]}"
            )

        print(f"[SPLITS] ✅ VERIFIED: {len(train_patients)} train, {len(dev_patients)} dev patients")
        print("[SPLITS] ✅ NO PATIENT OVERLAP - Data is clean!")
    else:
        print("[SPLITS] WARNING: Could not verify splits (dirs not found)")

    # Check if cache exists on Modal persistent volume
    print("\n" + "=" * 60, flush=True)
    print("[CACHE] Verifying cache location on Modal...", flush=True)
    print("=" * 60, flush=True)

    try:
        from pathlib import Path
        import shutil
        import json

        # Load config to get the actual cache path
        cfg_abs = config_path
        if not config_path.startswith("/"):
            cfg_abs = str(Path("/app") / config_path)

        import yaml
        with open(cfg_abs, "r") as f:
            config_data = yaml.safe_load(f)

        # Get cache path from config, with fallback (root of cache, not a split subdir)
        cache_dir = (
            config_data.get("data", {}).get("cache_dir")
            or config_data.get("experiment", {}).get("cache_dir", "/results/cache/tusz")
        )

        # For smoke tests, ensure we use a separate cache directory
        if "smoke" in config_path.lower() and "smoke" not in cache_dir:
            cache_dir = cache_dir.replace("/tusz", "/smoke")

        # CRITICAL: Cache structure should be cache_dir/{train,dev}/ for patient disjointness
        cache_train = Path(cache_dir) / "train"
        cache_dev = Path(cache_dir) / "dev"
        cache_path = cache_train  # Primary cache for reporting

        # CACHE VALIDATION: Check if cache was built with patient-disjoint splits
        cache_metadata_file = Path(cache_dir) / ".cache_metadata.json"
        cache_valid = False

        if cache_path.exists():
            npz_files = list(cache_path.glob("*.npz"))

            # Check if metadata exists and validates
            if cache_metadata_file.exists():
                try:
                    with open(cache_metadata_file) as f:
                        metadata = json.load(f)

                    # Check if built with official_tusz policy
                    if metadata.get("split_policy") == "official_tusz":
                        print(f"[CACHE] ✅ Cache built with official_tusz policy", flush=True)
                        cache_valid = True
                    else:
                        print(f"[CACHE] ⚠️ Cache built with old policy: {metadata.get('split_policy', 'unknown')}", flush=True)
                        cache_valid = False
                except Exception as e:
                    print(f"[CACHE] ⚠️ Could not read cache metadata: {e}", flush=True)
                    cache_valid = False
            else:
                # No metadata = old cache from before fix
                if len(npz_files) > 0:
                    print(f"[CACHE] ⚠️ No metadata found - cache built before patient fix!", flush=True)
                    print(f"[CACHE] ❌ MUST INVALIDATE {len(npz_files)} contaminated files", flush=True)
                else:
                    print("[CACHE] No metadata found - cache is empty (will build fresh)", flush=True)
                cache_valid = False

            if not cache_valid and len(npz_files) > 0:
                print("[CACHE] 🧹 Auto-cleaning contaminated cache...", flush=True)
                shutil.rmtree(cache_dir, ignore_errors=True)
                print("[CACHE] ✅ Old cache deleted", flush=True)

                # Recreate clean structure
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
                cache_train.mkdir(exist_ok=True)
                cache_dev.mkdir(exist_ok=True)

                # Write new metadata
                metadata = {
                    "split_policy": "official_tusz",
                    "created": str(Path("/app") / "configs" / "modal" / "smoke.yaml" if "smoke" in config_path else "train.yaml"),
                    "timestamp": str(Path(__file__).stat().st_mtime)
                }
                with open(cache_metadata_file, "w") as f:
                    json.dump(metadata, f, indent=2)
                print("[CACHE] ✅ Created clean cache structure with metadata", flush=True)

                npz_files = []  # Reset file count
            elif cache_valid:
                manifest = cache_path / "manifest.json"
                print(f"[CACHE] ✅ Using valid Modal SSD cache: {len(npz_files)} NPZ files", flush=True)
                if manifest.exists():
                    print(f"[CACHE] ✅ Manifest found at {manifest}", flush=True)
                print(f"[CACHE] Cache location: {cache_path}", flush=True)
                print(f"[CACHE] This is optimal - using fast local SSD storage", flush=True)
        else:
            print(f"[CACHE] Cache will be built at: {cache_path}", flush=True)
            print(f"[CACHE] First epoch will be slower while building cache", flush=True)

            # Create metadata for new cache
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            cache_train.mkdir(exist_ok=True)
            cache_dev.mkdir(exist_ok=True)

            metadata = {
                "split_policy": "official_tusz",
                "created": str(Path("/app") / "configs" / "modal" / "smoke.yaml" if "smoke" in config_path else "train.yaml"),
                "timestamp": str(Path(__file__).stat().st_mtime)
            }
            with open(cache_metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            print("[CACHE] ✅ Created cache metadata for validation", flush=True)

    except Exception as e:
        print(f"[WARNING] Could not verify cache: {e}", flush=True)

    print("=" * 60 + "\n", flush=True)

    # Set environment
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONPATH"] = "/app"
    env["PYTHONUNBUFFERED"] = "1"  # CRITICAL: Force unbuffered output for real-time logs
    env["PYTHONFAULTHANDLER"] = "1"  # Enable Python fault handler for better error traces
    # env["SEIZURE_MAMBA_FORCE_FALLBACK"] = "1"  # REMOVED - Mamba-SSM should work now!
    env["PYTHONTRACEMALLOC"] = "1"  # Track memory allocations for debugging
    # Only limit files for smoke tests
    if "smoke" in config_path.lower():
        env["BGB_LIMIT_FILES"] = "50"
        env["BGB_SMOKE_TEST"] = "1"
    else:
        # EXPLICITLY UNSET for full training to avoid inheritance
        env.pop("BGB_LIMIT_FILES", None)

    # Disable tqdm for Modal subprocess environments (causes issues with manifest generation)
    env["BGB_DISABLE_TQDM"] = "1"
    print(f"[ENV] BGB_DISABLE_TQDM={env.get('BGB_DISABLE_TQDM')}", flush=True)
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
        "/data/edf",  # Parent containing train/dev/eval (mounted from S3)
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

    # CRITICAL: Use cache with patient-disjoint structure
    # Cache location: /results/cache/{tusz,smoke}/{train,dev}/
    if "smoke" in config_path.lower():
        cache_dir = "/results/cache/smoke"
    else:
        # Use the persistent cache for full training
        cache_dir = "/results/cache/tusz"  # This MUST have train/ and dev/ subdirs

    # Ensure cache directories exist with correct structure
    from pathlib import Path
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    (Path(cache_dir) / "train").mkdir(exist_ok=True)
    (Path(cache_dir) / "dev").mkdir(exist_ok=True)

    # Set cache_dir in both data and experiment sections
    exp["cache_dir"] = cache_dir
    data.setdefault("data", {})["cache_dir"] = cache_dir

    print(f"[CONFIG] Using cache directory: {cache_dir}", flush=True)
    print(f"[CONFIG] Output directory: {exp['output_dir']}", flush=True)
    if "smoke" in config_path.lower():
        print(f"[CONFIG] BGB_LIMIT_FILES={env.get('BGB_LIMIT_FILES', 'not set')}", flush=True)

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

    # Run training with REAL-TIME output streaming
    print("Starting training process with real-time logging...")
    print(f"Data loading from S3 may take 10-20 minutes for large datasets", flush=True)

    # Use Popen for real-time output with proper buffering
    # bufsize=1 enables line buffering which is better for tqdm
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # Line buffering for better tqdm compatibility
    )

    # Stream output in real-time with error handling
    try:
        for line in process.stdout:
            print(line, end='', flush=True)
    except Exception as e:
        print(f"[ERROR] Output streaming failed: {e}", flush=True)

    # Wait for process to complete
    returncode = process.wait()

    if returncode != 0:
        raise RuntimeError(f"Training failed with exit code {returncode}")

    print(f"Training completed successfully!", flush=True)
    # Return best checkpoint path under /results
    checkpoint_dir = Path(data["experiment"]["output_dir"]) / "checkpoints"
    # Our training saves best.pt
    return str(checkpoint_dir / "best.pt")


@app.function(
    gpu="A100",  # A100 for evaluation
    timeout=3600,  # 1 hour
    volumes={
        "/data": data_mount,   # Use S3 mount for eval datasets
        "/results": results_volume,
    },
    memory=65536,  # SAFE: 64GB RAM for evaluation
    cpu=16,  # SAFE: 16 CPU cores for eval
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
    config: str = "configs/modal/smoke.yaml",  # Default to smoke test for safety
    resume: bool = False,  # Resume training from last.pt
):
    """Modal deployment entrypoint.

    ⚠️ CRITICAL: Modal's --detach flag MUST go BEFORE the script name!
    ⚠️ NO DOUBLE DASH (--) separator needed anymore in Modal CLI!

    Examples:
        # IMPORTANT: Clean old contaminated cache first!
        modal run deploy/modal/app.py --action clean-cache

        # Test Mamba CUDA kernels
        modal run deploy/modal/app.py --action test-mamba

        # Quick smoke test (Modal's --detach prevents disconnection)
        modal run --detach deploy/modal/app.py --action train --config configs/modal/smoke.yaml

        # Full A100 training (Modal's --detach prevents disconnection)
        modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml

        # Resume training from last.pt in output_dir
        modal run --detach deploy/modal/app.py --action train --config configs/modal/train.yaml --resume true

        # Evaluate checkpoint
        modal run deploy/modal/app.py --action evaluate --config /results/checkpoints/best.pt
    """
    print("🚀 Brain-Go-Brr v2 Modal Deployment")
    print("=" * 50)

    if action == "clean-cache":
        # Clean contaminated cache from before patient-disjoint fix
        print("🧹 Cleaning contaminated cache...")
        success = clean_cache.remote()
        if success:
            print("✅ Cache cleaned! Next training will rebuild with patient-disjoint splits.")
        else:
            print("❌ Cache cleaning failed!")
            raise RuntimeError("Failed to clean cache")

    elif action == "test-mamba":
        # Test Mamba CUDA kernels
        print("Testing Mamba CUDA kernels...")
        success = test_mamba_cuda.remote()
        if success:
            print("✅ Mamba CUDA test PASSED! Ready for training.")
        else:
            print("❌ Mamba CUDA test FAILED! Fix required before training.")
            raise RuntimeError("Mamba CUDA kernels not working")

    elif action == "train":
        # Always use train.remote() - Modal's --detach flag controls app lifecycle
        result = train.remote(config_path=config, resume=resume)
        print(f"✓ Training complete. Checkpoint: {result}")

    elif action == "evaluate":
        # For evaluate, config arg is actually checkpoint path
        result = evaluate.remote(checkpoint_path=config)
        print(f"✓ Evaluation complete. Metrics: {result}")

    else:
        print(f"Unknown action: {action}")
        print("Available actions: clean-cache, test-mamba, train, evaluate")


if __name__ == "__main__":
    main()
