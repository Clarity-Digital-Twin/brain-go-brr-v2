# syntax=docker/dockerfile:1

# SzCORE submission image for Brain-Go-Brr.
#
# Requirements:
# - Must run offline at inference time (no network).
# - Must read `/data/$INPUT` and write `/output/$OUTPUT`.
# - Must pass SzCORE PR CI which executes the container in CPU-only GitHub Actions.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Create a non-privileged user that the app will run under.
ARG UID=10001
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
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (GPU deps are installed but only used when a GPU is available).
COPY deploy/szcore/requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

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
CMD python3 -m src.brain_brr.szcore "/data/${INPUT}" "/output/${OUTPUT}"

