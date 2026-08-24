#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DT_SERVER_DIR="${DT_SERVER_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
ISAAC_SIM_DIR="${ISAAC_SIM_DIR:-}"
ISAAC_PUBLIC_IP="${ISAAC_PUBLIC_IP:-}"
# The shorter names are kept for compatibility. The WEBRTC_* names can be
# shared with docker compose/frontend_api from the same environment file.
ISAAC_SIGNAL_PORT="${ISAAC_SIGNAL_PORT:-${ISAAC_WEBRTC_SIGNAL_PORT:-49100}}"
# This deployment only permits inbound UDP 10000-11000. Zenoh already uses
# 10020, so 10021 is the repository default for Isaac Sim media.
ISAAC_STREAM_PORT="${ISAAC_STREAM_PORT:-${ISAAC_WEBRTC_STREAM_PORT:-10021}}"
ISAAC_WEB_PORT="${ISAAC_WEB_PORT:-${ISAAC_WEBRTC_WEB_PORT:-8211}}"

if [[ -z "${ISAAC_SIM_DIR}" ]]; then
    echo "ISAAC_SIM_DIR must point to the Isaac Sim 4.2.0 installation." >&2
    exit 2
fi
if [[ -z "${ISAAC_PUBLIC_IP}" ]]; then
    echo "ISAAC_PUBLIC_IP must be the public IPv4 advertised to the remote WebRTC client." >&2
    exit 2
fi

LAUNCHER="${ISAAC_SIM_DIR}/isaac-sim.headless.webrtc.sh"
CLIENT_SCRIPT="${DT_SERVER_DIR}/apps/isaac_sim_client/meta_sejong_script.py"

if [[ ! -x "${LAUNCHER}" ]]; then
    echo "Isaac Sim WebRTC launcher is not executable: ${LAUNCHER}" >&2
    exit 2
fi
if [[ ! -f "${CLIENT_SCRIPT}" ]]; then
    echo "Meta Sejong Isaac client was not found: ${CLIENT_SCRIPT}" >&2
    exit 2
fi

for port_name in ISAAC_SIGNAL_PORT ISAAC_STREAM_PORT ISAAC_WEB_PORT; do
    port="${!port_name}"
    if [[ ! "${port}" =~ ^[0-9]+$ ]] || ((port < 1 || port > 65535)); then
        echo "${port_name} must be an integer between 1 and 65535 (got: ${port})." >&2
        exit 2
    fi
done

args=(
    "--/app/livestream/port=${ISAAC_SIGNAL_PORT}"
    "--/app/livestream/fixedHostPort=${ISAAC_STREAM_PORT}"
    "--/exts/omni.services.transport.server.http/port=${ISAAC_WEB_PORT}"
)

# Isaac Sim 4.2 uses the legacy Kit livestream settings. Advertising the
# reachable address and media port is required when the client is outside the
# server's LAN/NAT. For a mesh VPN, set this to the server's VPN IPv4 address.
if [[ -n "${ISAAC_PUBLIC_IP}" ]]; then
    args+=(
        "--/app/livestream/publicEndpointAddress=${ISAAC_PUBLIC_IP}"
        "--/app/livestream/publicEndpointPort=${ISAAC_STREAM_PORT}"
    )
fi

echo "Starting Isaac Sim 4.2 WebRTC"
echo "  viewer:   http://${ISAAC_PUBLIC_IP:-<server-ip>}:${ISAAC_WEB_PORT}/streaming/webrtc-demo/"
echo "  signaling: TCP ${ISAAC_SIGNAL_PORT}"
echo "  media:     UDP ${ISAAC_STREAM_PORT}"

exec "${LAUNCHER}" "${args[@]}" --exec "${CLIENT_SCRIPT}" "$@"
