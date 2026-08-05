#!/usr/bin/env bash
set -Eeuo pipefail

# Digital Twin edge client installer for JetPack 6 / L4T R36.
# JetPack must provide CUDA, cuDNN, and the TensorRT Python API.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.txt"
VENV_PATH="${SCRIPT_DIR}/.venv"
PYTHON_CMD="python3"
TORCH_WHEEL=""
TORCHVISION_WHEEL=""
USE_VENV=1
INSTALL_APT=1
INSTALL_TORCH2TRT=1
DOWNLOAD_ASSETS=1
FORCE=0

log() { printf '\033[1;34m[INFO]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[WARN]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31m[ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'EOF'
Usage: apps/edge_client/install_root.sh [options]

Options:
  --venv PATH                 Virtual environment path (default: apps/edge_client/.venv)
  --system                    Install into the selected system Python
  --python COMMAND            Python command to use (default: python3)
  --torch-wheel PATH_OR_URL   JetPack-compatible PyTorch wheel
  --torchvision-wheel PATH_OR_URL
                               PyTorch-compatible torchvision wheel
  --skip-apt                  Do not install Ubuntu build/runtime packages
  --skip-torch2trt            Do not install torch2trt from its official repository
  --skip-assets               Do not download sample data and the pose checkpoint
  --force                     Reinstall supplied wheels and torch2trt
  -h, --help                  Show this help

PyTorch and torchvision must be CUDA-enabled builds compatible with the installed
JetPack release. If they are already available, wheel options may be omitted.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv)
            [[ $# -ge 2 ]] || die "--venv requires a path"
            VENV_PATH="$2"
            shift 2
            ;;
        --system)
            USE_VENV=0
            shift
            ;;
        --python)
            [[ $# -ge 2 ]] || die "--python requires a command"
            PYTHON_CMD="$2"
            shift 2
            ;;
        --torch-wheel|--wheel-url)
            [[ $# -ge 2 ]] || die "$1 requires a path or URL"
            TORCH_WHEEL="$2"
            shift 2
            ;;
        --torchvision-wheel)
            [[ $# -ge 2 ]] || die "--torchvision-wheel requires a path or URL"
            TORCHVISION_WHEEL="$2"
            shift 2
            ;;
        --skip-apt)
            INSTALL_APT=0
            shift
            ;;
        --skip-torch2trt)
            INSTALL_TORCH2TRT=0
            shift
            ;;
        --skip-assets)
            DOWNLOAD_ASSETS=0
            shift
            ;;
        --force)
            FORCE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "Unknown option: $1"
            ;;
    esac
done

[[ -f "$REQUIREMENTS_FILE" ]] || die "Requirements file not found: $REQUIREMENTS_FILE"
command -v "$PYTHON_CMD" >/dev/null 2>&1 || die "Python not found: $PYTHON_CMD"
[[ "$(uname -m)" == "aarch64" ]] || die "This installer supports Jetson aarch64 only"
[[ -r /etc/nv_tegra_release ]] || die "JetPack/L4T was not detected"

L4T_RELEASE="$(sed -n 's/^# R\([0-9]\+\).*/\1/p' /etc/nv_tegra_release | head -n 1)"
[[ "$L4T_RELEASE" == "36" ]] || die "JetPack 6 / L4T R36 is required (detected R${L4T_RELEASE:-unknown})"

log "Device: $(tr -d '\0' </proc/device-tree/model 2>/dev/null || printf Jetson)"
log "L4T: $(head -n 1 /etc/nv_tegra_release)"
log "Python: $($PYTHON_CMD --version 2>&1)"

if [[ "$INSTALL_APT" -eq 1 ]]; then
    log "Installing Ubuntu dependencies"
    sudo apt-get update
    sudo apt-get install -y \
        build-essential cmake git python3-dev python3-pip python3-venv \
        libavcodec-dev libavformat-dev libglib2.0-0 libgl1 \
        libjpeg-dev libopenblas-dev libswscale-dev zlib1g-dev
fi

if [[ "$USE_VENV" -eq 1 ]]; then
    if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
        log "Creating virtual environment: $VENV_PATH"
        "$PYTHON_CMD" -m venv --system-site-packages "$VENV_PATH"
    else
        log "Using existing virtual environment: $VENV_PATH"
    fi
    PY="${VENV_PATH}/bin/python"
else
    warn "Installing into system Python: $PYTHON_CMD"
    PY="$PYTHON_CMD"
fi

PIP=("$PY" -m pip)
"${PIP[@]}" install --upgrade pip setuptools wheel packaging

PIP_REINSTALL=()
[[ "$FORCE" -eq 1 ]] && PIP_REINSTALL=(--force-reinstall)

if [[ -n "$TORCH_WHEEL" ]]; then
    log "Installing the supplied PyTorch wheel"
    "${PIP[@]}" install --no-cache-dir "${PIP_REINSTALL[@]}" "$TORCH_WHEEL"
fi
if [[ -n "$TORCHVISION_WHEEL" ]]; then
    log "Installing the supplied torchvision wheel"
    "${PIP[@]}" install --no-cache-dir "${PIP_REINSTALL[@]}" "$TORCHVISION_WHEEL"
fi

log "Checking the JetPack ML runtime"
"$PY" - <<'PY'
import tensorrt
import torch
import torchvision

if not torch.cuda.is_available():
    raise SystemExit("CUDA-enabled PyTorch is required")

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("TensorRT:", tensorrt.__version__)
print("CUDA device:", torch.cuda.get_device_name(0))
PY

log "Installing edge client Python dependencies"
"${PIP[@]}" install --no-cache-dir -r "$REQUIREMENTS_FILE"

if [[ "$INSTALL_TORCH2TRT" -eq 1 ]]; then
    if [[ "$FORCE" -eq 1 ]] || ! "$PY" -c 'import torch2trt' >/dev/null 2>&1; then
        log "Installing torch2trt"
        "${PIP[@]}" install --no-deps --no-build-isolation \
            "git+https://github.com/NVIDIA-AI-IOT/torch2trt.git"
    else
        log "torch2trt is already installed"
    fi
fi

if [[ "$DOWNLOAD_ASSETS" -eq 1 ]]; then
    log "Downloading sample data and model files"
    "$PY" "${SCRIPT_DIR}/src/utils/download_from_drive.py"
fi

log "Verifying edge client imports"
"$PY" - <<'PY'
import cv2
import onnx
import onnxoptimizer
import pycuda.driver
import torch
import ultralytics
import zenoh
import zstandard

x = torch.ones((128, 128), device="cuda")
result = x @ x
torch.cuda.synchronize()
print("CUDA tensor test:", float(result[0, 0]))
print("Edge client dependencies are ready")
PY

log "Installation complete"
if [[ "$USE_VENV" -eq 1 ]]; then
    printf '\nActivate the environment with:\n  source %q/bin/activate\n' "$VENV_PATH"
fi
