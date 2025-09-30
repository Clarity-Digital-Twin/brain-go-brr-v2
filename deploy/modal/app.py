"""Modal cloud deployment for Brain-Go-Brr V3."""

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

import modal

# Module logger
logger = logging.getLogger(__name__)

# Build the Modal image with CUDA development tools for mamba-ssm compilation
# CRITICAL: Must match EXACT versions from local setup (docs/03-operations/setup-guide.md)
image = (
    # Use NVIDIA CUDA devel image for nvcc compiler (required by mamba-ssm)
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])  # Clear entrypoint from CUDA image
    # Install build tools required for compiling CUDA extensions
    .apt_install("build-essential", "ninja-build", "git")
    # Set CUDA environment variables BEFORE any pip installs
    .env({
        "CUDA_HOME": "/usr/local/cuda-12.4",
        "PATH": "/usr/local/cuda-12.4/bin:$PATH",
        "LD_LIBRARY_PATH": "/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH",
        "TORCH_CUDA_ARCH_LIST": "8.0;8.6;8.9;9.0",  # A100 is 8.0
        "FORCE_REBUILD": "2025-09-30-pr708-src",  # Bump to defeat Modal layer cache
        "TRITON_CACHE_DIR": "/tmp/triton_cache",
        "TORCHINDUCTOR_CACHE_DIR": "/tmp/torchinductor_cache",
    })
    # Upgrade pip to latest to stop annoying warnings
    .run_commands("pip install --upgrade pip")
    # CRITICAL: Install EXACT PyTorch version from specific index
    # Modal's mirror can have wrong versions, so we force PyTorch index
    .run_commands(
        "pip install torch==2.5.0 torchvision==0.20.0 'numpy<2.0' --index-url https://download.pytorch.org/whl/cu124"
    )
    # Verify PyTorch is correct version (CUDA check happens at runtime, not build time)
    .run_commands(
        "python -c 'import torch; assert torch.__version__.startswith(\"2.5.0\"), f\"Wrong torch: {torch.__version__}\"'"
    )
    # Install build dependencies
    .pip_install("packaging", "wheel", "setuptools")
    # CRITICAL: Install causal-conv1d first
    .run_commands(
        "pip install --no-build-isolation --no-cache-dir causal-conv1d==1.5.2"
    )
    # 🔧 CRITICAL: Patch mamba-ssm SOURCE before building
    # This fixes XID 31 MMU Fault on A100 with large batches (batch=64, d_model=512)
    # PR #708: https://github.com/state-spaces/mamba/pull/708
    # Strategy: Download sdist → Patch source → Build from patched source
    # This ensures Triton kernels compile from patched code, not cached wheels
    .run_commands(
        "python -c \""
        "import urllib.request, tarfile; "
        "from pathlib import Path; "
        "url = 'https://files.pythonhosted.org/packages/ba/2d/fbd909f6e6d48c491a9ed7ae68e8a890d8409aba4a6356741e2a9c6adad5/mamba_ssm-2.2.5.tar.gz'; "
        "dest = Path('/tmp/mamba_src'); "
        "dest.mkdir(parents=True, exist_ok=True); "
        "tgz = dest / 'mamba_ssm-2.2.5.tar.gz'; "
        "print(f'Downloading {url}...'); "
        "urllib.request.urlretrieve(url, tgz); "
        "print(f'Extracting {tgz}...'); "
        "tarfile.open(tgz, 'r:gz').extractall(dest); "
        "print(f'✅ Extracted to {dest}/mamba_ssm-2.2.5')"
        "\""
    )
    .add_local_file(
        str(Path(__file__).parent / "patch_mamba_pr708.py"),
        "/tmp/patch_source_pr708.py",
        copy=True
    )
    .run_commands(
        "python /tmp/patch_source_pr708.py --source /tmp/mamba_src/mamba_ssm-2.2.5"
    )
    # Build and install from patched source (no wheel caching, no binaries)
    .run_commands(
        "pip install --no-build-isolation --no-cache-dir "
        "--force-reinstall --no-deps --no-binary :all: "
        "/tmp/mamba_src/mamba_ssm-2.2.5"
    )
    # Verify patch landed in ALL installed Triton kernel files
    .run_commands(
        "python -c \""
        "from pathlib import Path; "
        "import mamba_ssm; "
        "files = ['ssd_chunk_scan.py', 'ssd_chunk_state.py', 'ssd_state_passing.py', 'ssd_combined.py']; "
        "for f in files: "
        "  p = Path(mamba_ssm.__file__).parent / 'ops' / 'triton' / f; "
        "  s = p.read_text(); "
        "  assert '.to(tl.int64)' in s, f'Missing int64 cast in {f}'; "
        "  print(f'✅ {f}'); "
        "\""
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
    # CRITICAL: Install PyTorch Geometric with exact versions for PyTorch 2.5.0 + CUDA 12.4
    # These MUST match our local setup exactly!
    .run_commands(
        "pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu124.html"
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

# ========================================================================
# STORAGE ARCHITECTURE (CRITICAL TO UNDERSTAND)
# ========================================================================
# 1. S3 BUCKET: brain-go-brr-eeg-data-20250919
#    - tusz/edf/: Raw EDF files (266GB)
#    - cache/tusz/: Preprocessed NPZ files (449GB)
#
# 2. MODAL MOUNTS (read-only from S3):
#    - /data/: Raw EDF files (from S3: tusz/)
#    - /cache/: Preprocessed NPZ cache (from S3: cache/tusz/)
#
# 3. PERSISTENCE VOLUME: brain-go-brr-results
#    - /results/: Training outputs ONLY (checkpoints, logs)
#    - NO CACHES HERE! Caches come from S3 mount
# ========================================================================

s3_secret = modal.Secret.from_name("aws-s3-secret")

# Raw EDF data mount (not used in training, but available if needed)
data_mount = modal.CloudBucketMount(
    "brain-go-brr-eeg-data-20250919",
    secret=s3_secret,
    key_prefix="tusz/",  # → Mounted at /data/{train,dev,eval}/
    read_only=True,
)

# REMOVED: We don't mount cache from S3 anymore!
# Cache lives on Modal SSD volume at /results/cache/tusz for performance.
# See populate_cache() function below for one-time S3→SSD copy.

# Persistent volume for TRAINING OUTPUTS ONLY (not caches!)
# Structure: /results/{smoke,train}/{{checkpoints,tensorboard,wandb}/
results_volume = modal.Volume.from_name("brain-go-brr-results", create_if_missing=True)
# NOTE: Old cache directories in volume have been cleaned up (Sep 25, 2025)


# One-time cache population from S3 to Modal SSD
@app.function(
    timeout=21600,  # 6 hours for 450GB copy (was timing out at 2 hours)
    cpu=24,  # More CPU for faster copying
    memory=65536,  # More memory for file operations
    volumes={
        "/results": results_volume,  # Destination: SSD volume
        "/s3_cache": modal.CloudBucketMount(
            "brain-go-brr-eeg-data-20250919",
            secret=s3_secret,
            key_prefix="cache/tusz/",
            read_only=True,
        ),  # Source: S3 bucket
    },
)
def populate_cache():
    """One-time copy of cache from S3 to Modal SSD volume.

    This copies ~450GB of preprocessed NPZ files from S3 to the Modal
    persistent SSD volume for fast, reliable training access.
    Run this ONCE when setting up, then reuse the cache forever.
    """
    from src.brain_brr.utils.logging_config import setup_logging
    # Use simple format for Modal (no Rich in container logs)
    setup_logging(format_style="simple", force=True)

    import shutil
    from pathlib import Path
    import time

    src = Path("/s3_cache")  # S3 mount
    dst = Path("/results/cache/tusz")  # SSD volume

    logger.info("\n" + "=" * 60)
    logger.info("[CACHE POPULATION] Starting S3 → SSD cache copy...")
    logger.info(f"Source: {src} (S3 mount)")
    logger.info(f"Destination: {dst} (Modal SSD)")
    logger.info("=" * 60 + "\n")

    start = time.time()

    # Create destination if needed
    dst.mkdir(parents=True, exist_ok=True)

    # Copy train split
    train_src = src / "train"
    train_dst = dst / "train"
    if train_src.exists():
        train_files = list(train_src.glob("*.npz"))
        logger.info(f"[COPY] Found {len(train_files)} train files to copy...")
        if train_dst.exists():
            logger.info(f"[COPY] Removing existing {train_dst}...")
            shutil.rmtree(train_dst)
        logger.info(f"[COPY] Copying {train_src} → {train_dst}...")
        shutil.copytree(train_src, train_dst)
        logger.info(f"[COPY] ✅ Copied {len(list(train_dst.glob('*.npz')))} train files")
    else:
        logger.info(f"[WARNING] No train split found at {train_src}")

    # Copy dev split (TUSZ 'dev' → cache 'dev')
    # CRITICAL: We use 'dev' naming to match TUSZ's official split naming!
    # TUSZ provides train/dev/eval - we use dev for validation/tuning during training.
    # DO NOT rename to 'val' - this causes confusion with TUSZ documentation.
    dev_src = src / "dev"
    dev_dst = dst / "dev"
    if dev_src.exists():
        dev_files = list(dev_src.glob("*.npz"))
        logger.info(f"[COPY] Found {len(dev_files)} dev files to copy...")
        if dev_dst.exists():
            logger.info(f"[COPY] Removing existing {dev_dst}...")
            shutil.rmtree(dev_dst)
        logger.info(f"[COPY] Copying {dev_src} → {dev_dst}...")
        shutil.copytree(dev_src, dev_dst)
        logger.info(f"[COPY] ✅ Copied {len(list(dev_dst.glob('*.npz')))} dev files")
    else:
        logger.info(f"[WARNING] No dev split found at {dev_src}")

    # Copy metadata file - CRITICAL for cache validation!
    metadata_src = src / ".cache_metadata.json"
    metadata_dst = dst / ".cache_metadata.json"
    if metadata_src.exists():
        logger.info(f"[COPY] Copying cache metadata file...")
        shutil.copy2(metadata_src, metadata_dst)
        logger.info(f"[COPY] ✅ Copied metadata file")
    else:
        logger.info(f"[WARNING] No metadata file found at {metadata_src}")
        logger.info(f"[WARNING] Creating metadata file to prevent cache deletion...")
        # Create metadata file to prevent auto-deletion
        import json
        metadata = {
            "split_policy": "official_tusz",
            "created": "2025-09-26T22:11:00",
            "timestamp": "1758939060",
            "note": "Cache built with patient-disjoint TUSZ official splits",
            "train_patients": 579,
            "dev_patients": 53,
            "train_files": 4667,
            "dev_files": 1832,
            "version": "v3.2.0"
        }
        with open(metadata_dst, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"[COPY] ✅ Created metadata file at {metadata_dst}")

    # Verify final state
    train_count = len(list((dst / "train").glob("*.npz")))
    dev_count = len(list((dst / "dev").glob("*.npz")))
    elapsed = time.time() - start

    logger.info("\n" + "=" * 60)
    logger.info("[CACHE POPULATION] ✅ COMPLETE!")
    logger.info(f"Train files: {train_count} (expected: 4600-4700)")
    logger.info(f"Dev files: {dev_count} (expected: 1800-1900)")
    logger.info(f"Time taken: {elapsed/60:.1f} minutes")
    logger.info(f"Cache location: {dst}")
    logger.info("Cache is now on fast Modal SSD - ready for training!")
    logger.info("=" * 60 + "\n")

    return train_count, dev_count


@app.function(
    timeout=300,
    cpu=2,
    memory=2048,
    volumes={"/results": results_volume},
)
def check_cache():
    """Verify Modal SSD cache completeness."""
    from pathlib import Path
    import json

    cache = Path("/results/cache/tusz")

    print("\n" + "=" * 70)
    print(" " * 20 + "MODAL SSD CACHE VERIFICATION")
    print("=" * 70 + "\n")

    # Check root metadata
    metadata_file = cache / ".cache_metadata.json"
    print(f"[ROOT] .cache_metadata.json: ", end="")
    if metadata_file.exists():
        print(f"✅ EXISTS ({metadata_file.stat().st_size} bytes)")
        with open(metadata_file) as f:
            meta = json.load(f)
        print(f"       Policy: {meta.get('split_policy')}")
        print(f"       Created: {meta.get('created')}")
        print(f"       Expected: {meta.get('train_files')} train + {meta.get('dev_files')} dev files")
    else:
        print("❌ MISSING")

    # Check train split
    print(f"\n[TRAIN SPLIT]")
    train_manifest = cache / "train" / "manifest.json"
    train_index = cache / "train" / "_dataset_index.json"
    train_dir = cache / "train"

    if train_dir.exists():
        train_npz = list(train_dir.glob("*.npz"))
        print(f"  manifest.json:         {'✅' if train_manifest.exists() else '❌'} ({train_manifest.stat().st_size if train_manifest.exists() else 0:,} bytes)")
        print(f"  _dataset_index.json:   {'✅' if train_index.exists() else '❌'} ({train_index.stat().st_size if train_index.exists() else 0:,} bytes)")
        print(f"  *.npz files:           {'✅' if len(train_npz) == 4667 else '⚠️ '} {len(train_npz):,} (expected 4667)")
    else:
        print(f"  ❌ train/ directory missing!")
        train_npz = []

    # Check dev split
    print(f"\n[DEV SPLIT]")
    dev_manifest = cache / "dev" / "manifest.json"
    dev_index = cache / "dev" / "_dataset_index.json"
    dev_dir = cache / "dev"

    if dev_dir.exists():
        dev_npz = list(dev_dir.glob("*.npz"))
        print(f"  manifest.json:         {'⚠️ ' if dev_manifest.exists() else '  '} ({dev_manifest.stat().st_size if dev_manifest.exists() else 0:,} bytes) [OPTIONAL - not used]")
        print(f"  _dataset_index.json:   {'✅' if dev_index.exists() else '❌'} ({dev_index.stat().st_size if dev_index.exists() else 0:,} bytes)")
        print(f"  *.npz files:           {'✅' if len(dev_npz) == 1832 else '⚠️ '} {len(dev_npz):,} (expected 1832)")
    else:
        print(f"  ❌ dev/ directory missing!")
        dev_npz = []

    # Summary
    print("\n" + "=" * 70)
    print(" " * 30 + "SUMMARY")
    print("=" * 70)

    missing = []
    if not metadata_file.exists(): missing.append(".cache_metadata.json")
    if not train_manifest.exists(): missing.append("train/manifest.json [CRITICAL]")
    if not train_index.exists(): missing.append("train/_dataset_index.json [OPTIONAL]")
    if not dev_index.exists(): missing.append("dev/_dataset_index.json [CRITICAL]")
    if len(train_npz) != 4667: missing.append(f"train/*.npz ({len(train_npz)}/4667)")
    if len(dev_npz) != 1832: missing.append(f"dev/*.npz ({len(dev_npz)}/1832)")

    if missing:
        print("\n❌ MISSING FILES:")
        for m in missing:
            print(f"   - {m}")
        print("\n💡 FIX:")
        print("   1. Update populate_cache() to copy manifests/indexes")
        print("   2. Run: modal run deploy/modal/app.py::populate_cache")
        print("   OR")
        print("   3. First training run will regenerate (~45-50 min delay)\n")
    else:
        print("\n✅ ALL REQUIRED FILES PRESENT - Cache is complete!\n")

    print("=" * 70 + "\n")


# CPU-only: cache cleanup should not consume a GPU
@app.function(
    timeout=600,  # 10 min to include cache clean
    cpu=4,
    memory=4096,
    volumes={"/results": results_volume},  # Need volume for cache operations
)
def clean_cache():
    """Clean contaminated cache from before patient-disjoint fix."""
    from src.brain_brr.utils.logging_config import setup_logging
    # Use simple format for Modal (no Rich in container logs)
    setup_logging(format_style="simple", force=True)

    import shutil
    from pathlib import Path

    logger.info("\n" + "=" * 60)
    logger.info("[CACHE CLEAN] Starting cache cleanup...")
    logger.info("=" * 60)

    cache_paths = [
        Path("/results/cache/tusz"),
        Path("/results/cache/smoke"),
    ]

    for cache_path in cache_paths:
        if cache_path.exists():
            logger.info(f"[CLEAN] Removing {cache_path}...")
            shutil.rmtree(cache_path, ignore_errors=True)
            logger.info(f"[CLEAN] ✅ Removed {cache_path}")
        else:
            logger.info(f"[CLEAN] Path does not exist: {cache_path}")

    # Recreate clean directories
    for cache_path in cache_paths:
        cache_path.mkdir(parents=True, exist_ok=True)
        (cache_path / "train").mkdir(exist_ok=True)
        (cache_path / "dev").mkdir(exist_ok=True)  # Use 'dev' to match TUSZ naming!
        logger.info(f"[CLEAN] ✅ Created clean structure: {cache_path}/{{train,dev}}/")

    logger.info("\n[CACHE CLEAN] ✅ Cache cleanup complete!")
    logger.info("Next training run will rebuild cache with patient-disjoint splits.")
    logger.info("=" * 60 + "\n")
    return True


@app.function(
    gpu="A100-80GB",
    timeout=300,  # 5 min test
    cpu=16,  # Safe: 16 cores for testing
    memory=32768,  # Safe: 32GB RAM for tests
)
def test_mamba_cuda():
    """Test that Mamba CUDA kernels work properly."""
    from src.brain_brr.utils.logging_config import setup_logging
    # Use simple format for Modal (no Rich in container logs)
    setup_logging(format_style="simple", force=True)

    import torch
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"CUDA device: {torch.cuda.get_device_name()}")

    # Test mamba-ssm import
    try:
        import mamba_ssm
        logger.info(f"✓ mamba-ssm version: {mamba_ssm.__version__}")
    except ImportError as e:
        logger.info(f"✗ mamba-ssm import failed: {e}")
        return False

    # Test causal_conv1d import (the actual CUDA kernels)
    try:
        import causal_conv1d
        logger.info(f"✓ causal-conv1d imported")
    except ImportError as e:
        logger.info(f"✗ causal-conv1d import failed: {e}")
        return False

    # Test Mamba2 creation and forward pass
    try:
        from mamba_ssm import Mamba2

        # Create a simple Mamba2 layer
        model = Mamba2(d_model=512, d_state=16, d_conv=4, expand=2).cuda()
        logger.info("✓ Mamba2 model created")

        # Test forward pass (no grad for speed)
        x = torch.randn(2, 100, 512).cuda()  # (batch, seq_len, d_model)
        with torch.no_grad():
            out = model(x)
        logger.info(f"✓ Forward pass successful! Output shape: {out.shape}")

        # Test backward pass (needs grad enabled)
        x_grad = torch.randn(2, 100, 512, requires_grad=True).cuda()
        out_grad = model(x_grad)
        loss = out_grad.sum()
        loss.backward()
        logger.info("✓ Backward pass successful!")

        return True

    except Exception as e:
        logger.info(f"✗ Mamba2 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


@app.function(
    gpu="A100-80GB",  # 80GB VRAM, 3x faster than 4090
    timeout=86400,  # 24 hours max (Modal limit)
    volumes={
        "/data": data_mount,  # S3 bucket with raw EDF data (optional)
        "/results": results_volume,  # SSD volume with cache AND outputs!
        # NO /cache mount! Cache is on SSD at /results/cache/tusz
    },
    memory=98304,  # SAFE: 96GB RAM (was 32GB, now 3x for safety)
    cpu=24,  # SAFE: 24 CPU cores (3 cores per 8 DataLoader workers)
)
def train(
    config_path: str = "configs/modal/smoke.yaml",  # Default to smoke test for safety
    resume: bool = False,  # Resume training from last.pt in output_dir
    cuda_launch_blocking: bool = False,  # Enable CUDA_LAUNCH_BLOCKING for diagnostics
    cuda_dsa: bool = False,  # Enable TORCH_USE_CUDA_DSA for diagnostics
    force_fallback: bool = False,  # Force Mamba Conv1d fallback
):
    """Run training on Modal GPU.

    Args:
        config_path: Path to config YAML file (relative to /app)

    Returns:
        Path to checkpoint file
    """
    from src.brain_brr.utils.logging_config import setup_logging
    # Use simple format for Modal (no Rich in container logs)
    setup_logging(format_style="simple", force=True)

    import os
    import subprocess

    # Set diagnostic environment variables if requested
    if cuda_launch_blocking:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        logger.info("[DIAGNOSTIC] CUDA_LAUNCH_BLOCKING=1 (synchronous kernel execution)")

    if cuda_dsa:
        os.environ["TORCH_USE_CUDA_DSA"] = "1"
        logger.info("[DIAGNOSTIC] TORCH_USE_CUDA_DSA=1 (device-side assertions)")

    if force_fallback:
        os.environ["SEIZURE_MAMBA_FORCE_FALLBACK"] = "1"
        logger.info("[DIAGNOSTIC] SEIZURE_MAMBA_FORCE_FALLBACK=1 (using Conv1d instead of Mamba CUDA)")

    # CRITICAL: Clear Triton/Inductor caches for first run after patch
    # This ensures we don't use stale cached kernels
    os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache_run"
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/tii_cache_run"

    # Test mamba-ssm import
    try:
        import mamba_ssm
        logger.info(f"✓ Mamba-SSM imported successfully: {mamba_ssm.__version__}")
    except ImportError as e:
        logger.info(f"⚠️ Mamba-SSM import failed: {e}")

    # CRITICAL: Verify patient disjointness in data
    logger.info("\n" + "=" * 60)
    logger.info("[PATIENT DISJOINTNESS] Verifying TUSZ splits...")
    logger.info("=" * 60)

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

        logger.info(f"[SPLITS] ✅ VERIFIED: {len(train_patients)} train, {len(dev_patients)} dev patients")
        logger.info("[SPLITS] ✅ NO PATIENT OVERLAP - Data is clean!")
    else:
        logger.info("[SPLITS] WARNING: Could not verify splits (dirs not found)")

    # Check if cache exists on Modal persistent volume
    logger.info("\n" + "=" * 60)
    logger.info("[CACHE] Verifying cache location on Modal...")
    logger.info("=" * 60)

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

        # ALWAYS use SSD cache on Modal, NOT S3!
        cache_dir = "/results/cache/tusz"  # Fixed path on SSD volume

        # CRITICAL: Cache structure should be cache_dir/{train,dev}/ for patient disjointness
        # We use 'dev' NOT 'val' to match TUSZ's official split naming convention!
        cache_train = Path(cache_dir) / "train"
        cache_dev = Path(cache_dir) / "dev"  # TUSZ calls it 'dev', so we keep it 'dev'!
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
                        logger.info(f"[CACHE] ✅ Cache built with official_tusz policy")
                        cache_valid = True
                    else:
                        logger.info(f"[CACHE] ⚠️ Cache built with old policy: {metadata.get('split_policy', 'unknown')}")
                        cache_valid = False
                except Exception as e:
                    logger.info(f"[CACHE] ⚠️ Could not read cache metadata: {e}")
                    cache_valid = False
            else:
                # No metadata = old cache from before fix
                if len(npz_files) > 0:
                    logger.info(f"[CACHE] ⚠️ No metadata found - cache built before patient fix!")
                    logger.info(f"[CACHE] ❌ MUST INVALIDATE {len(npz_files)} contaminated files")
                else:
                    logger.info("[CACHE] No metadata found - cache is empty (will build fresh)")
                cache_valid = False

            if not cache_valid and len(npz_files) > 0:
                logger.info("[CACHE] 🧹 Auto-cleaning contaminated cache...")
                shutil.rmtree(cache_dir, ignore_errors=True)
                logger.info("[CACHE] ✅ Old cache deleted")

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
                logger.info("[CACHE] ✅ Created clean cache structure with metadata")

                npz_files = []  # Reset file count
            elif cache_valid:
                manifest = cache_path / "manifest.json"
                logger.info(f"[CACHE] ✅ Using valid Modal SSD cache: {len(npz_files)} NPZ files")
                if manifest.exists():
                    logger.info(f"[CACHE] ✅ Manifest found at {manifest}")
                logger.info(f"[CACHE] Cache location: {cache_path}")
                logger.info(f"[CACHE] This is optimal - using fast local SSD storage")
        else:
            logger.info(f"[CACHE] Cache will be built at: {cache_path}")
            logger.info(f"[CACHE] First epoch will be slower while building cache")

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
            logger.info("[CACHE] ✅ Created cache metadata for validation")

    except Exception as e:
        logger.info(f"[WARNING] Could not verify cache: {e}")

    logger.info("=" * 60 + "\n")

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

    # 🚨 CRITICAL: NaN protection (REQUIRED for PyTorch 2.5.0+)
    env["BGB_SANITIZE_GRADS"] = "1"  # Prevent gradient explosion
    env["BGB_NAN_DEBUG"] = "1"        # Show NaN warnings
    logger.info("[ENV] BGB_SANITIZE_GRADS=1 BGB_NAN_DEBUG=1 (NaN protection enabled)")

    # Disable tqdm for Modal subprocess environments (causes issues with manifest generation)
    env["BGB_DISABLE_TQDM"] = "1"
    logger.info(f"[ENV] BGB_DISABLE_TQDM={env.get('BGB_DISABLE_TQDM')}")
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

    # CRITICAL: Cache architecture
    # - Cache is on Modal SSD volume at /results/cache/tusz
    # - Smoke tests use SAME cache with BGB_LIMIT_FILES=50
    # - NO SEPARATE SMOKE CACHE EXISTS OR IS NEEDED
    cache_dir = "/results/cache/tusz"  # SSD volume, NOT S3!

    # Ensure cache directories exist with correct structure
    from pathlib import Path
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    (Path(cache_dir) / "train").mkdir(exist_ok=True)
    (Path(cache_dir) / "dev").mkdir(exist_ok=True)

    # Set cache_dir in both data and experiment sections
    exp["cache_dir"] = cache_dir
    data.setdefault("data", {})["cache_dir"] = cache_dir

    logger.info(f"[CONFIG] Using cache directory: {cache_dir}")
    logger.info(f"[CONFIG] Output directory: {exp['output_dir']}")
    if "smoke" in config_path.lower():
        logger.info(f"[CONFIG] BGB_LIMIT_FILES={env.get('BGB_LIMIT_FILES', 'not set')}")

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
        logger.info("Resuming training from last.pt if present in output_dir")

    logger.info(f"Running: {' '.join(cmd)}")
    logger.info(f"Config: {config_path}")
    logger.info("-" * 50)

    # Run training with REAL-TIME output streaming
    logger.info("Starting training process with real-time logging...")
    logger.info(f"Loading from Modal SSD cache - dataset indices building...")

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
            # Stream output directly to stdout for real-time logging
            import sys
            sys.stdout.write(line)
            sys.stdout.flush()
    except Exception as e:
        logger.info(f"[ERROR] Output streaming failed: {e}")

    # Wait for process to complete
    returncode = process.wait()

    if returncode != 0:
        raise RuntimeError(f"Training failed with exit code {returncode}")

    logger.info(f"Training completed successfully!")
    # Return best checkpoint path under /results
    checkpoint_dir = Path(data["experiment"]["output_dir"]) / "checkpoints"
    # Our training saves best.pt
    return str(checkpoint_dir / "best.pt")


@app.function(
    gpu="A100-80GB",  # A100 for evaluation
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
    from src.brain_brr.utils.logging_config import setup_logging
    # Use simple format for Modal (no Rich in container logs)
    setup_logging(format_style="simple", force=True)

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

    logger.info(f"Running: {' '.join(cmd)}")
    logger.info("-" * 50)

    result = subprocess.run(cmd, env=env, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Evaluation failed: {result.stderr[:500]}")

    logger.info(f"Evaluation complete. Metrics saved to: {output_json}")
    return output_json


@app.local_entrypoint()
def main(
    action: str = "train",
    config: str = "configs/modal/smoke.yaml",  # Default to smoke test for safety
    resume: bool = False,  # Resume training from last.pt
    cuda_launch_blocking: bool = False,  # Enable CUDA_LAUNCH_BLOCKING for diagnostics
    cuda_dsa: bool = False,  # Enable TORCH_USE_CUDA_DSA for diagnostics
    force_fallback: bool = False,  # Force Mamba Conv1d fallback
):
    """Modal deployment entrypoint.

    ⚠️ CRITICAL: Modal's --detach flag MUST go BEFORE the script name!
    ⚠️ NO DOUBLE DASH (--) separator needed anymore in Modal CLI!

    Examples:
        # STEP 1: Populate cache from S3 to Modal SSD (ONE TIME ONLY - use --detach!)
        modal run --detach deploy/modal/app.py --action populate-cache

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
    logger.info("🚀 Brain-Go-Brr V3 Modal Deployment")
    logger.info("=" * 50)

    if action == "populate-cache":
        # ONE-TIME: Copy cache from S3 to Modal SSD
        logger.info("📦 Populating Modal SSD cache from S3...")
        logger.info("This will copy ~450GB and may take 1-2 hours...")
        train_count, dev_count = populate_cache.remote()
        if 4600 <= train_count <= 4700 and 1800 <= dev_count <= 1900:
            logger.info("✅ Cache populated successfully! Ready for training.")
        else:
            logger.info(f"⚠️ Cache populated with {train_count} train, {dev_count} dev files")
            logger.info("Expected: 4600-4700 train, 1800-1900 dev")
            if train_count > 0 and dev_count > 0:
                logger.info("Cache is usable but may be incomplete.")

    elif action == "clean-cache":
        # Clean contaminated cache from before patient-disjoint fix
        logger.info("🧹 Cleaning contaminated cache...")
        success = clean_cache.remote()
        if success:
            logger.info("✅ Cache cleaned! Next training will rebuild with patient-disjoint splits.")
        else:
            logger.info("❌ Cache cleaning failed!")
            raise RuntimeError("Failed to clean cache")

    elif action == "test-mamba":
        # Test Mamba CUDA kernels
        logger.info("Testing Mamba CUDA kernels...")
        success = test_mamba_cuda.remote()
        if success:
            logger.info("✅ Mamba CUDA test PASSED! Ready for training.")
        else:
            logger.info("❌ Mamba CUDA test FAILED! Fix required before training.")
            raise RuntimeError("Mamba CUDA kernels not working")

    elif action == "train":
        # Always use train.remote() - Modal's --detach flag controls app lifecycle
        result = train.remote(
            config_path=config,
            resume=resume,
            cuda_launch_blocking=cuda_launch_blocking,
            cuda_dsa=cuda_dsa,
            force_fallback=force_fallback,
        )
        logger.info(f"✓ Training complete. Checkpoint: {result}")

    elif action == "evaluate":
        # For evaluate, config arg is actually checkpoint path
        result = evaluate.remote(checkpoint_path=config)
        logger.info(f"✓ Evaluation complete. Metrics: {result}")

    else:
        logger.info(f"Unknown action: {action}")
        logger.info("Available actions: populate-cache, clean-cache, test-mamba, train, evaluate")
        logger.info("\n📌 IMPORTANT: Run 'populate-cache' ONCE before first training!")


if __name__ == "__main__":
    main()
