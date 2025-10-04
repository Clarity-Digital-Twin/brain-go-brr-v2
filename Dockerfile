# Dockerfile for Brain-Go-Brr v3.6.1
# Multi-stage build: Compile GPU libraries → Lightweight runtime
# ============================================================================
# Stage 1: Builder - Compile GPU libraries
# ============================================================================
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda-12.4 \
    PATH=/usr/local/cuda-12.4/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9" \
    FORCE_CUDA=1

# Install build dependencies (add python3.11-venv for ensurepip)
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && apt-get update && \
    apt-get install -y \
      build-essential \
      ninja-build \
      git \
      python3.11 \
      python3.11-dev \
      python3.11-venv \
      python3-pip \
      wget \
    && rm -rf /var/lib/apt/lists/*

# Ensure Python 3.11 has pip (base image only ships pip for system Python)
RUN python3.11 -m ensurepip --upgrade

# Upgrade pip
RUN python3.11 -m pip install --upgrade pip setuptools wheel packaging

# CRITICAL: Install numpy FIRST to prevent PyTorch from pulling 2.x (breaks mamba-ssm)
RUN pip install numpy==1.26.4

# Install PyTorch (EXACT version from Modal)
RUN pip install \
    torch==2.5.0 \
    torchvision==0.20.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Verify CUDA
RUN python3.11 -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'"

# Compile causal-conv1d (5-10 min)
RUN pip install --no-build-isolation --no-cache-dir causal-conv1d==1.5.2

# Download mamba-ssm source
RUN mkdir -p /tmp/mamba_src && \
    cd /tmp/mamba_src && \
    wget https://files.pythonhosted.org/packages/ba/2d/fbd909f6e6d48c491a9ed7ae68e8a890d8409aba4a6356741e2a9c6adad5/mamba_ssm-2.2.5.tar.gz && \
    tar -xzf mamba_ssm-2.2.5.tar.gz

# Compile mamba-ssm (10-15 min)
RUN pip install --no-build-isolation --no-cache-dir /tmp/mamba_src/mamba_ssm-2.2.5

# Copy patch script
COPY deploy/modal/patch_mamba_pr708.py /tmp/

# Apply PR #708 patch
RUN python3.11 /tmp/patch_mamba_pr708.py

# Verify patch
RUN python3.11 -c "from pathlib import Path; \
    tri_dir = Path('/usr/local/lib/python3.11/dist-packages/mamba_ssm/ops/triton'); \
    files = ['ssd_chunk_scan.py', 'ssd_chunk_state.py', 'ssd_state_passing.py', 'ssd_combined.py']; \
    assert all('.to(tl.int64)' in (tri_dir / f).read_text() for f in files), 'Patch failed'; \
    print('✅ PR #708 patch verified')"

# Install PyG with pre-built wheels (EXACT URLs from Modal)
RUN pip install \
    torch_scatter \
    torch_sparse \
    torch_cluster \
    torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.5.0+cu124.html && \
    pip install torch-geometric==2.6.1

# Verify PyG
RUN python3.11 -c "import torch_geometric; print(f'✅ PyG {torch_geometric.__version__}')"

# Install remaining dependencies (EXACT versions from pyproject.toml)
RUN pip install \
    pytorch-tcn==1.2.3 \
    scipy==1.11.4 \
    scikit-learn==1.3.2 \
    matplotlib>=3.5.0 \
    seaborn>=0.11.0 \
    pandas>=2.0.0 \
    wandb>=0.16.0 \
    einops>=0.7.0 \
    mne>=1.5.0 \
    pydantic>=2.0.0 \
    pyedflib>=0.1.30 \
    pyyaml>=6.0.0 \
    click>=8.1.7 \
    rich>=13.0.0 \
    tqdm>=4.64.0

# ============================================================================
# Stage 2: Runtime - Lightweight production
# ============================================================================
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda-12.4 \
    PATH=/usr/local/cuda-12.4/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Install runtime Python (use deadsnakes for GA Python 3.11)
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && apt-get update && \
    apt-get install -y \
      python3.11 \
      python3.11-venv \
      python3-pip \
      git \
    && rm -rf /var/lib/apt/lists/*

# Ensure pip is available for Python 3.11 runtime layer
RUN python3.11 -m ensurepip --upgrade

# Copy compiled packages from builder
COPY --from=builder /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

# Copy source
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY pyproject.toml /app/
COPY README.md /app/

# Install editable
RUN pip install -e .

CMD ["python3.11", "-m", "src", "train", "/app/configs/local/train.yaml"]
