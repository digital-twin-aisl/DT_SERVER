#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EDGE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPOSITORY_ROOT="$(cd -- "${EDGE_DIR}/../.." && pwd)"
COMMON_PROJECT="${REPOSITORY_ROOT}/packages/dt_common"
VGGT_PROJECT="${REPOSITORY_ROOT}/apps/calibration_worker/vggt-omega"
WHEEL_DIR="${EDGE_DIR}/wheels"

for project in "${COMMON_PROJECT}" "${VGGT_PROJECT}"; do
    if [[ ! -f "${project}/pyproject.toml" ]]; then
        echo "[ERROR] wheel source not found: ${project}" >&2
        exit 1
    fi
done

mkdir -p "${WHEEL_DIR}"
python -m pip wheel \
    --no-build-isolation \
    --no-deps \
    --wheel-dir "${WHEEL_DIR}" \
    "${COMMON_PROJECT}" \
    "${VGGT_PROJECT}"
echo "[INFO] Edge bundle dependencies written to ${WHEEL_DIR}"
