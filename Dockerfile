# syntax=docker/dockerfile:1

# Stable RunPod PyTorch image:
# - PyTorch 2.8.0
# - CUDA 12.8.1
# - Ubuntu 24.04
# - RunPod /start.sh with Jupyter and SSH support
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ARG DEBIAN_FRONTEND=noninteractive
ARG ULTRALYTICS_VERSION=8.4.102
ARG SOCCERNET_VERSION=0.1.62

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MKL_THREADING_LAYER=GNU \
    OMP_NUM_THREADS=1 \
    WORKSPACE=/workspace \
    SOCCERNET_ROOT=/workspace/data/SoccerNet \
    SYNLOC_ROOT=/workspace/data/SoccerNet/SpiideoSynLoc \
    YOLO_ROOT=/workspace/data/SoccerNet/SpiideoSynLoc_yolo \
    DATASET_YAML=/workspace/data/SoccerNet/SpiideoSynLoc_yolo/synloc.yaml \
    PROJECT_DIRECTORY=/workspace/runs/synloc \
    MODEL=yolo26n.pt \
    SYNLOC_VERSION=fullhd \
    SYNLOC_SPLITS=train,valid,test,challenge \
    EPOCHS=100 \
    IMGSZ=1280 \
    BATCH=2 \
    WORKERS=8 \
    RUN_NAME=yolo26n_fullhd

# Only packages needed for:
# - cloning your repository
# - OpenCV/Ultralytics runtime support
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch and CUDA are already provided by the RunPod base image.
RUN python -m pip install --no-cache-dir \
        "ultralytics==${ULTRALYTICS_VERSION}" \
        "SoccerNet==${SOCCERNET_VERSION}" \
        "Pillow>=10,<12" \
        "PyYAML>=6,<7" \
    && python -m pip check

WORKDIR /workspace

# Preserve standard RunPod startup behavior:
# JupyterLab, SSH, and other RunPod services.
CMD ["/start.sh"]