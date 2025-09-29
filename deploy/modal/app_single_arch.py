"""Modal cloud deployment for Brain-Go-Brr V3 - SINGLE ARCH A100 ONLY."""

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

import modal

logger = logging.getLogger(__name__)

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .apt_install("build-essential", "ninja-build", "git")
    .env({
        "CUDA_HOME": "/usr/local/cuda-12.1",
        "PATH": "/usr/local/cuda-12.1/bin:$PATH",
        "LD_LIBRARY_PATH": "/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH",
        "TORCH_CUDA_ARCH_LIST": "8.0",  # A100 ONLY (single-arch for Test 2B)
        "FORCE_REBUILD": "2025-09-29-test2b-single-arch",  # Force cache invalidation
    })
    .run_commands(
        "pip install torch==2.2.2 torchvision==0.17.2 'numpy<2.0' --index-url https://download.pytorch.org/whl/cu121"
    )
    .run_commands(
        "python -c 'import torch; assert torch.__version__.startswith(\"2.2.2\"), f\"Wrong torch: {torch.__version__}\"'"
    )
    .pip_install("packaging", "wheel", "setuptools")
    .run_commands("pip cache purge")
    .run_commands(
        "pip uninstall -y causal-conv1d || true"
    )
    .run_commands(
        "pip install --no-build-isolation --no-cache-dir --force-reinstall causal-conv1d==1.4.0"
    )
    .run_commands(
        "pip uninstall -y mamba-ssm || true"
    )
    .run_commands(
        "pip install --no-build-isolation --no-cache-dir --force-reinstall mamba-ssm==2.2.2"
    )
    .run_commands(
        "python -c 'from mamba_ssm import Mamba2; print(\"✅ Mamba2 imports successfully\")'"
    )
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
        "pandas>=2.0.0",
        "tensorboard>=2.10.0",
        "wandb",
        "pytorch-tcn",
    )
    .run_commands(
        "pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html"
    )
    .run_commands(
        "pip install torch-geometric==2.6.1"
    )
    .run_commands(
        "python -c 'import torch_geometric; print(f\"✅ PyG {torch_geometric.__version__} installed\")'"
    )
    .workdir("/app")
    .add_local_dir(str(Path(__file__).parent.parent.parent / "src"), "/app/src")
    .add_local_dir(str(Path(__file__).parent.parent.parent / "configs"), "/app/configs")
    .add_local_dir(str(Path(__file__).parent), "/app/deploy/modal")
)

app = modal.App(
    "brain-go-brr-v2-single-arch",
    image=image,
    secrets=[
        modal.Secret.from_name("wandb-secret"),
    ],
)

s3_secret = modal.Secret.from_name("aws-s3-secret")

data_mount = modal.CloudBucketMount(
    "brain-go-brr-eeg-data-20250919",
    secret=s3_secret,
    key_prefix="tusz/",
    read_only=True,
)

results_volume = modal.Volume.from_name("brain-go-brr-results", create_if_missing=True)


@app.function(
    gpu="A100-80GB",
    timeout=86400,
    volumes={
        "/data": data_mount,
        "/results": results_volume,
    },
    memory=98304,
    cpu=24,
)
def train(
    config_path: str = "configs/modal/smoke.yaml",
    resume: bool = False,
):
    """Run training on Modal GPU with single-arch A100 build."""
    from src.brain_brr.utils.logging_config import setup_logging
    setup_logging(format_style="simple", force=True)

    import os
    import subprocess

    logger.info("[TEST 2B] Single-arch A100 build (TORCH_CUDA_ARCH_LIST=8.0)")
    logger.info("[TEST 2B] All extensions rebuilt with --force-reinstall")
    logger.info("[TEST 2B] Cache purged before rebuild")

    try:
        import mamba_ssm
        logger.info(f"✓ Mamba-SSM imported successfully: {mamba_ssm.__version__}")
    except ImportError as e:
        logger.info(f"⚠️ Mamba-SSM import failed: {e}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONPATH"] = "/app"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONFAULTHANDLER"] = "1"
    env["PYTHONTRACEMALLOC"] = "1"

    if "smoke" in config_path.lower():
        env["BGB_LIMIT_FILES"] = "50"
        env["BGB_SMOKE_TEST"] = "1"
    else:
        env.pop("BGB_LIMIT_FILES", None)

    env["BGB_DISABLE_TQDM"] = "1"

    import tempfile
    import yaml

    cfg_abs = config_path
    if not config_path.startswith("/"):
        cfg_abs = str(Path("/app") / config_path)

    with open(cfg_abs, "r") as f:
        data = yaml.safe_load(f)

    preferred_roots = [
        "/data/edf",
        "/data",
    ]
    for root in preferred_roots:
        if os.path.isdir(root):
            data.setdefault("data", {})["data_dir"] = root
            break

    exp = data.setdefault("experiment", {})
    out_name = Path(exp.get("output_dir", "results/run")).name
    exp["output_dir"] = f"/results/{out_name}"

    data_cfg = data.setdefault("data", {})
    data_cfg["cache_dir"] = "/results/cache/tusz"

    fd, tmp_cfg = tempfile.mkstemp(suffix=".yaml", prefix="modal_cfg_", text=True)
    try:
        with os.fdopen(fd, "w") as tf:
            yaml.dump(data, tf)

        cmd = ["python", "-m", "src", "train", tmp_cfg]
        if resume:
            cmd.append("--resume")

        logger.info(f"Running: {' '.join(cmd)}")
        logger.info("-" * 50)

        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in proc.stdout:
            print(line, end="", flush=True)

        proc.wait()

        if proc.returncode != 0:
            raise RuntimeError(f"Training failed with exit code {proc.returncode}")

        results_volume.commit()

        best_ckpt = Path(exp["output_dir"]) / "checkpoints" / "best.pt"
        return str(best_ckpt) if best_ckpt.exists() else f"Complete (no checkpoint at {best_ckpt})"

    finally:
        try:
            os.unlink(tmp_cfg)
        except Exception:
            pass


@app.local_entrypoint()
def main(
    action: str = "train",
    config: str = "configs/modal/train.yaml",
    resume: bool = False,
):
    """Modal single-arch deployment entrypoint for Test 2B."""
    logger.info("🚀 Brain-Go-Brr V3 Modal Deployment - SINGLE ARCH TEST 2B")
    logger.info("=" * 50)

    if action == "train":
        result = train.remote(config_path=config, resume=resume)
        logger.info(f"✓ Training complete. Checkpoint: {result}")
    else:
        logger.info(f"Unknown action: {action}")
        logger.info("Only 'train' action supported in single-arch test")


if __name__ == "__main__":
    main()