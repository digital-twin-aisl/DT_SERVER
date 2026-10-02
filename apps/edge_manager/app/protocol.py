# SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Manager message policies and compatibility exports for the shared contract."""

from copy import deepcopy
import json
import time
from typing import Any
from uuid import uuid4

from dt_common.contracts.edge import (  # noqa: F401 - compatibility exports
    DEFAULT_TOPIC_ROOT, EDGE_ID_PATTERN, SCHEMA_VERSION, EdgeTopics,
    decode_json, encode_json, validate_edge_id,
)
from dt_common.contracts.chunks import (  # noqa: F401
    ChunkCollector, MAX_BUNDLE_BYTES as MAX_CALIBRATION_BUNDLE_BYTES,
)
from dt_common.infrastructure.zenoh import make_zenoh_config  # noqa: F401


def heartbeat_command(edge_id: str) -> dict[str, Any]:
    return {'schema_version': SCHEMA_VERSION, 'kind': 'command', 'edge_id': edge_id,
            'command_id': uuid4().hex, 'command': 'ping', 'sent_at': time.time(),
            'parameters': {'reason': 'manager_heartbeat', 'request_cameras': True}}



def camera_records_from_message(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate and whitelist the only camera fields the server may persist."""
    if message.get("kind") != "camera_status":
        raise ValueError("unexpected message kind")
    data = message.get("data")
    if not isinstance(data, dict) or not isinstance(data.get("cameras"), list):
        raise ValueError("camera status data.cameras must be a list")

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(data["cameras"]):
        if not isinstance(raw, dict):
            raise ValueError(f"camera status item {index} must be an object")
        camera_id = str(raw.get("camera_id") or "")
        if not camera_id.isdigit() or int(camera_id) <= 0:
            raise ValueError(f"camera status item {index} has an invalid camera_id")
        camera_id = str(int(camera_id))
        if camera_id in seen:
            raise ValueError(f"duplicate camera_id: {camera_id}")
        seen.add(camera_id)
        if not isinstance(raw.get("exists"), bool) or not isinstance(
            raw.get("ping"), bool
        ):
            raise ValueError(f"camera/{camera_id} exists and ping must be booleans")

        calibration = raw.get("calibration")
        if calibration is None:
            calibration = {}
        if not isinstance(calibration, dict):
            raise ValueError(f"camera/{camera_id} calibration must be an object")
        sanitized_calibration = {
            key: deepcopy(calibration.get(key))
            for key in ("intrinsic", "extrinsic", "distortion_coefficients")
        }
        # Validate that nested calibration values can safely be represented as JSON.
        json.dumps(sanitized_calibration, allow_nan=False)
        records.append(
            {
                "camera_id": camera_id,
                "exists": raw["exists"],
                "ping": raw["ping"],
                "calibration": sanitized_calibration,
            }
        )
    return records
