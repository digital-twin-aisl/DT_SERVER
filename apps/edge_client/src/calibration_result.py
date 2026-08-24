"""Validate and persist server-produced camera calibration on an edge."""

from __future__ import annotations

import math
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import yaml


def _finite_matrix(value: Any, shape: tuple[int, int], name: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != shape[0]:
        raise ValueError(f"{name} must have shape {shape[0]}x{shape[1]}")
    result: list[list[float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != shape[1]:
            raise ValueError(f"{name} must have shape {shape[0]}x{shape[1]}")
        converted = [float(item) for item in row]
        if not all(math.isfinite(item) for item in converted):
            raise ValueError(f"{name} must contain finite values")
        result.append(converted)
    return result


def _finite_vector(value: Any, length: int, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must contain {length} values")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain finite values")
    return result


def _atomic_write_yaml(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            yaml.safe_dump(
                document,
                destination,
                allow_unicode=True,
                sort_keys=False,
            )
            destination.flush()
            os.fsync(destination.fileno())
        os.chmod(temporary_name, 0o600)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def apply_calibration_result(
    config_path: str | Path,
    edge_id: str,
    payload: dict[str, Any],
) -> list[str]:
    """Apply one complete calibration result and return updated camera IDs."""
    request_id = str(payload.get("request_id") or "")
    if not request_id:
        raise ValueError("calibration result request_id is required")
    completed_at = float(payload.get("completed_at"))
    if not math.isfinite(completed_at):
        raise ValueError("calibration result completed_at must be finite")
    json.dumps(payload, allow_nan=False)
    cameras_payload = payload.get("cameras")
    if not isinstance(cameras_payload, list) or not cameras_payload:
        raise ValueError("calibration result cameras must be a non-empty list")

    path = Path(config_path)
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cameras = document.get("CAMERAS")
    if not isinstance(cameras, list):
        raise ValueError("camera config CAMERAS must be a list")
    local_by_id = {
        str(camera.get("id")): camera
        for camera in cameras
        if isinstance(camera, dict)
    }

    updates: list[tuple[dict[str, Any], dict[str, Any], list[list[float]] | None]] = []
    seen: set[str] = set()
    for item in cameras_payload:
        if not isinstance(item, dict):
            raise ValueError("calibration camera result must be an object")
        camera_id = str(item.get("camera_id") or "")
        if not camera_id.isdigit() or int(camera_id) <= 0:
            raise ValueError(f"invalid calibration camera_id: {camera_id}")
        camera_id = str(int(camera_id))
        external_key = str(item.get("camera_key") or "")
        if external_key != f"{edge_id}/camera/{camera_id}":
            raise ValueError(f"camera key does not belong to this edge: {external_key}")
        if camera_id in seen:
            raise ValueError(f"duplicate calibration camera_id: {camera_id}")
        seen.add(camera_id)
        local = local_by_id.get(camera_id)
        if local is None:
            raise ValueError(f"camera/{camera_id} is not registered on this edge")

        extrinsic = {
            "schema_version": 1,
            "request_id": request_id,
            "calibrated_at": completed_at,
            "marker_tree": str(payload.get("marker_tree") or ""),
            "coordinate_convention": payload.get("coordinate_convention") or {},
            "world_to_camera": _finite_matrix(
                item.get("world_to_camera"), (4, 4), "world_to_camera"
            ),
            "camera_to_world": _finite_matrix(
                item.get("camera_to_world"), (4, 4), "camera_to_world"
            ),
            "position_m": _finite_vector(item.get("position_m"), 3, "position_m"),
            "alignment": payload.get("alignment") or {},
        }
        processed_matrix = item.get("undistorted_camera_matrix")
        validated_processed = (
            _finite_matrix(
                processed_matrix, (3, 3), "undistorted_camera_matrix"
            )
            if processed_matrix is not None
            else None
        )
        updates.append((local, extrinsic, validated_processed))

    if set(local_by_id) != seen:
        missing = sorted(set(local_by_id) - seen)
        raise ValueError(
            "calibration result does not cover every registered camera: "
            + ", ".join(f"camera/{camera_id}" for camera_id in missing)
        )

    for local, extrinsic, processed_matrix in updates:
        local["extrinsic"] = extrinsic
        if processed_matrix is not None:
            intrinsic = local.get("intrinsic")
            if isinstance(intrinsic, dict):
                intrinsic["undistorted_camera_matrix"] = processed_matrix
    _atomic_write_yaml(path, document)
    return sorted(seen, key=int)
