# syntax=docker/dockerfile:1

# SzCORE submission image for Brain-Go-Brr.
#
# Requirements:
# - Must run offline at inference time (no network).
# - Must read `/data/$INPUT` and write `/output/$OUTPUT`.
# - Must pass SzCORE PR CI which executes the container in CPU-only GitHub Actions.

ARG PYTORCH_IMAGE=pytorch/pytorch:2.5.0-cuda12.4-cudnn9
ARG UID=10001

# -----------------------------------------------------------------------------
# Builder: build CUDA wheels (mamba-ssm, causal-conv1d)
# -----------------------------------------------------------------------------
FROM ${PYTORCH_IMAGE}-devel AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /build

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
      build-essential ca-certificates git ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# IMPORTANT: compile for evaluation cluster GPUs (V100 sm70, A100 sm80) and local dev (RTX 4090 sm89).
ENV TORCH_CUDA_ARCH_LIST="70;80;89"
RUN python -m pip wheel --no-cache-dir --no-deps --no-build-isolation -w /build/wheels \
      mamba-ssm==2.2.5 \
      causal-conv1d==1.5.2

# -----------------------------------------------------------------------------
# Runtime: PyTorch CUDA runtime + app deps
# -----------------------------------------------------------------------------
FROM ${PYTORCH_IMAGE}-runtime AS runtime

ARG UID=10001

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Create a non-privileged user that the app will run under.
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/home/appuser" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    appuser

# System deps:
# - libgomp1: required by some scientific wheels (e.g., sklearn/numba deps via mne)
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY deploy/szcore/requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

# `flash-linear-attention` ships `fla.layers` used by our model code, but pulls
# heavyweight deps by default. Install it without deps to keep the image lean.
RUN python -m pip install --no-cache-dir --no-deps flash-linear-attention==0.3.2

COPY --from=builder /build/wheels /tmp/wheels
RUN python -m pip install --no-cache-dir --no-deps /tmp/wheels/*.whl && rm -rf /tmp/wheels

# Copy code + config + checkpoint.
COPY src/ /app/src/
COPY configs/szcore_inference.yaml /app/configs/szcore_inference.yaml
COPY results/local_fla_exp4_cyclic/checkpoints/best.pt /app/checkpoints/best.pt

USER appuser

VOLUME ["/data"]
VOLUME ["/output"]

ENV INPUT=""
ENV OUTPUT=""

# IMPORTANT: shell-form CMD so $INPUT/$OUTPUT expand.
CMD python -m src.brain_brr.szcore "/data/${INPUT}" "/output/${OUTPUT}"
