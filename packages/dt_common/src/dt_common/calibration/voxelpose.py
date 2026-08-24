"""Convert USD-world calibration results to VoxelPose camera dictionaries."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np


MILLIMETRES_PER_METRE = 1000.0
CAMERA_ID_PATTERN = re.compile(r"^(?:camera/)?([1-9][0-9]*)$")


def camera_number(value: Any) -> int:
    """Return the physical camera number used by camera_N video names."""
    match = CAMERA_ID_PATTERN.fullmatch(str(value))
    if match is None:
        raise ValueError(f"invalid camera ID: {value!r}")
    return int(match.group(1))


def load_calibration_result(path: str | Path) -> dict[str, Any]:
    result_path = Path(path)
    with result_path.open(encoding="utf-8") as source:
        result = json.load(source)
    if not isinstance(result, dict) or not isinstance(result.get("cameras"), list):
        raise ValueError("calibration result must contain a cameras list")

    convention = result.get("coordinate_convention") or {}
    world_convention = str(convention.get("world", "")).lower()
    if re.search(r"\bmetres?\b", world_convention) is None:
        raise ValueError("calibration world coordinates must be expressed in metres")
    if "opencv" not in str(convention.get("camera", "")).lower():
        raise ValueError("calibration camera coordinates must use OpenCV convention")
    return result


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite array with shape {shape}")
    return array


def voxelpose_camera(
    record: dict[str, Any],
    *,
    world_origin_m: Iterable[float] = (0.0, 0.0, 0.0),
) -> dict[str, Any]:
    """Build the R/camera-centre/intrinsic representation used by VoxelPose.

    VoxelPose evaluates ``R @ (point - T)`` and therefore expects ``T`` to be
    the camera centre, not the translation column from a world-to-camera
    matrix. Model-space translations are millimetres.
    """
    camera_id = camera_number(record.get("camera_id"))
    world_to_camera = _finite_array(
        record.get("world_to_camera"), (4, 4), "world_to_camera"
    )
    camera_to_world = _finite_array(
        record.get("camera_to_world"), (4, 4), "camera_to_world"
    )
    if not np.allclose(world_to_camera @ camera_to_world, np.eye(4), atol=1e-5):
        raise ValueError(f"camera/{camera_id} extrinsic matrices are not inverses")

    rotation = world_to_camera[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5):
        raise ValueError(f"camera/{camera_id} rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-5):
        raise ValueError(f"camera/{camera_id} rotation must have determinant +1")

    camera_matrix = _finite_array(
        record.get("camera_matrix"), (3, 3), "camera_matrix"
    )
    distortion = np.asarray(
        record.get("distortion_coefficients"), dtype=np.float64
    ).reshape(-1)
    if distortion.size < 5 or not np.isfinite(distortion).all():
        raise ValueError("distortion_coefficients must contain five finite values")

    origin = _finite_array(list(world_origin_m), (3,), "world_origin_m")
    position_m = camera_to_world[:3, 3]
    declared_position = _finite_array(record.get("position_m"), (3,), "position_m")
    if not np.allclose(position_m, declared_position, atol=1e-5):
        raise ValueError(f"camera/{camera_id} position_m disagrees with its extrinsic")

    return {
        "id": camera_id,
        "R": rotation,
        "T": ((position_m - origin) * MILLIMETRES_PER_METRE).reshape(3, 1),
        "fx": float(camera_matrix[0, 0]),
        "fy": float(camera_matrix[1, 1]),
        "cx": float(camera_matrix[0, 2]),
        "cy": float(camera_matrix[1, 2]),
        "k": distortion[[0, 1, 4]].reshape(3, 1),
        "p": distortion[[2, 3]].reshape(2, 1),
    }


def select_voxelpose_cameras(
    calibration: dict[str, Any],
    camera_ids: Iterable[int | str],
    *,
    world_origin_m: Iterable[float] = (0.0, 0.0, 0.0),
) -> list[dict[str, Any]]:
    """Select cameras in the requested view order and reject missing IDs."""
    records: dict[int, dict[str, Any]] = {}
    for record in calibration["cameras"]:
        if not isinstance(record, dict):
            raise ValueError("each calibration camera must be an object")
        number = camera_number(record.get("camera_id"))
        if number in records:
            raise ValueError(f"duplicate calibration camera/{number}")
        records[number] = record

    requested = [camera_number(value) for value in camera_ids]
    if len(requested) != len(set(requested)):
        raise ValueError("requested camera IDs must be unique")
    missing = [number for number in requested if number not in records]
    if missing:
        raise ValueError(
            "calibration result is missing "
            + ", ".join(f"camera/{number}" for number in missing)
        )
    return [
        voxelpose_camera(records[number], world_origin_m=world_origin_m)
        for number in requested
    ]
