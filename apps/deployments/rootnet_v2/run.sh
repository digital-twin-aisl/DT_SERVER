#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
# SPDX-License-Identifier: LGPL-2.1-or-later
# Direct diagnostic launch; does not start/stop the edge manager or agent.
set -euo pipefail

usage() {
    echo 'Usage: bash run.sh {server|edge_1|edge_2} tcp/ROUTER_IP:7447 [inference options]'
    echo 'Example: bash run.sh server tcp/127.0.0.1:7447 --validate-only'
    echo 'Activate the inference Python environment first, or set PYTHON_BIN.'
}

if [[ "${1:-}" == '-h' || "${1:-}" == '--help' ]]; then
    usage
    exit 0
fi
if (( $# < 2 )); then
    usage >&2
    exit 2
fi
role="$1"
endpoint="$2"
shift 2
case "$role" in
    server) application='server_worker' ;;
    edge_1|edge_2) application='edge_client' ;;
    *) echo "Unknown role: $role" >&2; usage >&2; exit 2 ;;
esac
if [[ -z "$endpoint" || "$endpoint" == -* ]]; then
    echo 'Provide the router endpoint before inference options.' >&2
    exit 2
fi

profile_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$profile_dir/../../.." && pwd)"
# Edge focus paths are repository-relative; server focus paths are app-relative.
cd -- "$project_dir"
export PYTHONPATH="$project_dir/packages/dt_common/src:$project_dir${PYTHONPATH:+:$PYTHONPATH}"
echo "[rootnet_v2] role=$role; defaults: recorded dataset; model=B; v2 calibration override; PyTorch"
exec "${PYTHON_BIN:-python3}" "apps/$application/inference.py" \
    --runtime-config "$profile_dir/$role.json" \
    --zenoh-endpoint "$endpoint" "$@"
