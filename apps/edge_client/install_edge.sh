#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
VENV_PATH="${SCRIPT_DIR}/.venv"
PYTORCH_INDEX="https://pypi.jetson-ai-lab.io/jp6/cu126"

die() { echo "[ERROR] $*" >&2; exit 1; }
log() { echo "[INFO] $*"; }

log "Installing system dependencies"
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake git python3-dev python3-pip python3-venv \
    libavcodec-dev libavformat-dev libglib2.0-0 libgl1 \
    libjpeg-dev libopenblas-dev libswscale-dev zlib1g-dev

if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
    log "Creating ${VENV_PATH}"
    python3 -m venv --system-site-packages "$VENV_PATH"
fi

source "${VENV_PATH}/bin/activate"
python -m pip install --upgrade pip "setuptools<82" wheel packaging

FASTREID_PATH="${SCRIPT_DIR}/src/reid/fast-reid"
if [[ ! -f "${FASTREID_PATH}/fastreid/__init__.py" ]]; then
    if [[ ! -f "${REPOSITORY_ROOT}/.gitmodules" ]]; then
        die "fast-reid is missing and this checkout has no submodule metadata"
    fi
    log "Initializing fast-reid submodule"
    git -C "${REPOSITORY_ROOT}" submodule update --init --depth 1 -- \
        apps/edge_client/src/reid/fast-reid
fi

VGGT_SOURCE="${REPOSITORY_ROOT}/apps/calibration_worker/vggt-omega"
if [[ ! -f "${VGGT_SOURCE}/pyproject.toml" ]]; then
    log "Initializing vggt-omega submodule"
    git -C "${REPOSITORY_ROOT}" submodule update --init --depth 1 -- \
        apps/calibration_worker/vggt-omega
fi

shopt -s nullglob
COMMON_WHEELS=("${SCRIPT_DIR}"/wheels/dt_common-*.whl)
if (( ${#COMMON_WHEELS[@]} > 1 )); then
    die "multiple dt-common wheels found under ${SCRIPT_DIR}/wheels"
elif (( ${#COMMON_WHEELS[@]} == 1 )); then
    log "Installing bundled dt-common wheel"
    python -m pip install --no-deps "${COMMON_WHEELS[0]}"
elif [[ -f "${REPOSITORY_ROOT}/packages/dt_common/pyproject.toml" ]]; then
    log "Installing dt-common from the source checkout"
    python -m pip install --no-deps "${REPOSITORY_ROOT}/packages/dt_common"
else
    die "dt-common is missing; bundle edge dependencies with tools/vendor_dependencies.sh"
fi

VGGT_WHEELS=("${SCRIPT_DIR}"/wheels/vggt_omega-*.whl)
if (( ${#VGGT_WHEELS[@]} > 1 )); then
    die "multiple vggt-omega wheels found under ${SCRIPT_DIR}/wheels"
elif (( ${#VGGT_WHEELS[@]} == 1 )); then
    log "Installing bundled vggt-omega wheel"
    python -m pip install --no-deps "${VGGT_WHEELS[0]}"
elif [[ -f "${REPOSITORY_ROOT}/apps/calibration_worker/vggt-omega/pyproject.toml" ]]; then
    log "Installing vggt-omega from the source checkout"
    python -m pip install --no-deps --no-build-isolation \
        "${REPOSITORY_ROOT}/apps/calibration_worker/vggt-omega"
else
    die "vggt-omega is missing; bundle edge dependencies with tools/vendor_dependencies.sh"
fi

log "Installing PyTorch from Jetson AI Lab"
python -m pip install --no-cache-dir --extra-index-url "$PYTORCH_INDEX" \
    torch==2.11.0 torchvision==0.26.0

log "Installing edge client dependencies"
python -m pip install --no-cache-dir -r "${SCRIPT_DIR}/requirements.txt"

log "Installing torch2trt"
python -m pip install --no-deps --no-build-isolation \
    git+https://github.com/NVIDIA-AI-IOT/torch2trt.git

log "Downloading sample data and model files"
python "${SCRIPT_DIR}/src/utils/download_from_drive.py"

log "Verifying installation"
PYTHONPATH="${FASTREID_PATH}${PYTHONPATH:+:${PYTHONPATH}}" python - <<'PY'
import cv2
import onnx
import onnxoptimizer
import tensorrt
import torch
import torchvision
import ultralytics
import vggt_omega
import zenoh
import zstandard
import fastreid
from torch2trt import TRTModule, torch2trt

if not torch.cuda.is_available():
    raise SystemExit("CUDA-enabled PyTorch is required")

x = torch.ones((128, 128), device="cuda")
torch.cuda.synchronize()
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("TensorRT:", tensorrt.__version__)
print("CUDA device:", torch.cuda.get_device_name(0))
print("Installation complete")
PY

echo
echo "Activate later with: source ${VENV_PATH}/bin/activate"
