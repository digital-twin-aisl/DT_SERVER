"""Validate and persist server-produced camera calibration on an edge."""

from __future__ import annotations

import math
import json
import os
from pathlib import Path
import tempfile
import time
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


def import_external_calibration_result(
    config_path: str | Path,
    result_path: str | Path,
) -> list[str]:
    """Import matching cameras from an existing calibration result into YAML."""
    source_path = Path(result_path).expanduser().resolve()
    result = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(result, dict) or not isinstance(result.get("cameras"), list):
        raise ValueError("calibration result must contain a cameras list")

    result_by_id: dict[str, dict[str, Any]] = {}
    for item in result["cameras"]:
        if not isinstance(item, dict):
            raise ValueError("calibration result camera must be an object")
        raw_id = item.get("edge_camera_id") or item.get("camera_id")
        camera_id = str(raw_id or "").rsplit("/", 1)[-1]
        if not camera_id.isdigit() or int(camera_id) <= 0:
            raise ValueError(f"invalid calibration camera_id: {raw_id}")
        camera_id = str(int(camera_id))
        if camera_id in result_by_id:
            raise ValueError(f"duplicate calibration camera_id: {camera_id}")
        result_by_id[camera_id] = item

    path = Path(config_path).expanduser().resolve()
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cameras = document.get("CAMERAS")
    if not isinstance(cameras, list) or not cameras:
        raise ValueError("camera config CAMERAS must be a non-empty list")

    imported: list[str] = []
    calibrated_at = time.time()
    input_data = result.get("input")
    if not isinstance(input_data, dict):
        input_data = {}
    request_id = str(input_data.get("request_id") or f"import:{source_path.name}")
    for camera in cameras:
        if not isinstance(camera, dict):
            raise ValueError("camera config entry must be an object")
        camera_id = str(camera.get("id") or "")
        if not camera_id.isdigit() or int(camera_id) <= 0:
            raise ValueError(f"invalid local camera ID: {camera_id}")
        camera_id = str(int(camera_id))
        item = result_by_id.get(camera_id)
        if item is None:
            raise ValueError(f"calibration result is missing camera/{camera_id}")

        intrinsic = camera.get("intrinsic")
        if not isinstance(intrinsic, dict):
            intrinsic = {}
            camera["intrinsic"] = intrinsic
        intrinsic["camera_matrix"] = _finite_matrix(
            item.get("camera_matrix"), (3, 3), "camera_matrix"
        )
        distortion = item.get("distortion_coefficients")
        if not isinstance(distortion, list) or len(distortion) not in {
            4,
            5,
            8,
            12,
            14,
        }:
            raise ValueError("distortion_coefficients has an invalid length")
        converted_distortion = [float(value) for value in distortion]
        if not all(math.isfinite(value) for value in converted_distortion):
            raise ValueError("distortion_coefficients must contain finite values")
        intrinsic["distortion_coefficients"] = converted_distortion
        if item.get("undistorted_camera_matrix") is not None:
            intrinsic["undistorted_camera_matrix"] = _finite_matrix(
                item["undistorted_camera_matrix"],
                (3, 3),
                "undistorted_camera_matrix",
            )

        camera["extrinsic"] = {
            "schema_version": 1,
            "request_id": request_id,
            "calibrated_at": calibrated_at,
            "marker_tree": str(result.get("marker_tree") or ""),
            "coordinate_convention": result.get("coordinate_convention") or {},
            "world_to_camera": _finite_matrix(
                item.get("world_to_camera"), (4, 4), "world_to_camera"
            ),
            "camera_to_world": _finite_matrix(
                item.get("camera_to_world"), (4, 4), "camera_to_world"
            ),
            "position_m": _finite_vector(item.get("position_m"), 3, "position_m"),
            "alignment": result.get("alignment") or {},
        }
        imported.append(camera_id)

    _atomic_write_yaml(path, document)
    return sorted(imported, key=int)
