"""Estimate CCTV extrinsics in the Isaac Sim ArUco-marker coordinate system.

The calibration sequence is:

1. Load one still image per CCTV and sample frames from a reference video.
2. Undistort every image with the supplied camera calibration.
3. Run VGGT-Omega jointly over CCTV and reference images.
4. Detect known ArUco markers in the reference frames and unproject their
   corners with VGGT-Omega's depth/camera predictions.
5. Fit a metric similarity transform from VGGT coordinates to the marker-tree
   USD coordinates.
6. Apply that transform to each CCTV pose and write camera extrinsics as JSON.

VGGT-Omega returns camera-from-world extrinsics in OpenCV coordinates.  Output
``world_to_camera`` matrices use the same convention, with the world now being
the metric coordinate system authored in the marker-tree USD.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import itertools
import json
import logging
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import torch
import yaml

if str(REPOSITORY_ROOT := Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import apps  # noqa: E402,F401
from dt_common.calibration.protocol import (  # noqa: E402
    PREPROCESS_NAME,
    decode_feature_bundle,
    file_sha256,
)
from dt_common.calibration.preprocess import preprocess_frame  # noqa: E402


LOGGER = logging.getLogger("calibration_worker")
WORKER_DIR = Path(__file__).resolve().parent
VGGT_OMEGA_ROOT = WORKER_DIR / "vggt-omega"
DEFAULT_MARKER_TREE = (
    REPOSITORY_ROOT
    / "apps/isaac_sim_client/aruco_boards/aruco_marker_tree.usd"
)
DEFAULT_EDGE_REGISTRY = (
    REPOSITORY_ROOT / "apps/edge_manager/data/edges.json"
)
DEFAULT_RUNS_DIR = WORKER_DIR / "data" / "calibration_runs"
SUPPORTED_OFFLINE_IMAGE_SUFFIXES = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass(frozen=True)
class CameraInput:
    camera_id: str
    image_path: Path | None
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray


@dataclass(frozen=True)
class ReferenceVideoInput:
    video_path: Path
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    sample_count: int | None = None


@dataclass(frozen=True)
class CalibrationInput:
    cctv_cameras: tuple[CameraInput, ...]
    reference_video: ReferenceVideoInput


@dataclass(frozen=True)
class MarkerDefinition:
    dictionary_name: str
    marker_id: int
    marker_length_m: float
    world_corners: np.ndarray
    prim_path: str


@dataclass(frozen=True)
class SimilarityTransform:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    inlier_mask: np.ndarray
    residuals: np.ndarray

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        return self.scale * (points @ self.rotation.T) + self.translation

    def matrix4(self) -> np.ndarray:
        """Return the homogeneous VGGT-to-USD similarity matrix."""
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = self.scale * self.rotation
        matrix[:3, 3] = self.translation
        return matrix


@dataclass(frozen=True)
class PreparedInputs:
    image_paths: tuple[Path, ...]
    cctv_count: int
    cctv_new_camera_matrices: tuple[np.ndarray, ...]
    reference_frame_indices: tuple[int, ...]
    reference_new_camera_matrix: np.ndarray


@dataclass(frozen=True)
class RemotePreparedInputs:
    reference_images: np.ndarray
    cctv_count: int
    cctv_new_camera_matrices: tuple[np.ndarray, ...]
    reference_frame_indices: tuple[int, ...]
    reference_new_camera_matrix: np.ndarray


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object.")
    return value


def _matrix3(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 3x3 matrix.")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError(f"{name} focal lengths must be positive.")
    return matrix


def _distortion(value: Any, name: str) -> np.ndarray:
    coefficients = np.asarray(value, dtype=np.float64).reshape(-1)
    if coefficients.size not in {4, 5, 8, 12, 14}:
        raise ValueError(
            f"{name} must contain 4, 5, 8, 12, or 14 OpenCV coefficients."
        )
    if not np.isfinite(coefficients).all():
        raise ValueError(f"{name} must contain only finite values.")
    return coefficients


def _resolve_input_path(value: Any, base_dir: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


def load_calibration_input(config_path: str | Path) -> CalibrationInput:
    """Load and validate the calibration manifest.

    Paths in the manifest are resolved relative to the manifest itself.
    Intrinsics and distortion are mandatory: guessing CCTV lens distortion is
    deliberately outside this worker's responsibilities.
    """
    path = Path(config_path).expanduser().resolve()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Calibration config does not exist: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid calibration config JSON: {exc}") from exc

    root = _require_mapping(raw, "config")
    cameras_raw = root.get("cctv_cameras")
    if not isinstance(cameras_raw, list) or not cameras_raw:
        raise ValueError("config.cctv_cameras must be a non-empty array.")

    cameras: list[CameraInput] = []
    camera_ids: set[str] = set()
    for index, item in enumerate(cameras_raw):
        camera = _require_mapping(item, f"cctv_cameras[{index}]")
        camera_id = str(camera.get("camera_id", "")).strip()
        if not camera_id:
            raise ValueError(f"cctv_cameras[{index}].camera_id is required.")
        if camera_id in camera_ids:
            raise ValueError(f"Duplicate CCTV camera_id: {camera_id}")
        camera_ids.add(camera_id)
        cameras.append(
            CameraInput(
                camera_id=camera_id,
                image_path=_resolve_input_path(
                    camera.get("image_path"), path.parent, f"{camera_id}.image_path"
                ),
                camera_matrix=_matrix3(
                    camera.get("camera_matrix"), f"{camera_id}.camera_matrix"
                ),
                distortion_coefficients=_distortion(
                    camera.get("distortion_coefficients"),
                    f"{camera_id}.distortion_coefficients",
                ),
            )
        )

    reference_raw = _require_mapping(root.get("reference_video"), "reference_video")
    sample_count_raw = reference_raw.get("sample_count")
    sample_count = None if sample_count_raw is None else int(sample_count_raw)
    if sample_count is not None and sample_count <= 0:
        raise ValueError("reference_video.sample_count must be positive.")
    reference = ReferenceVideoInput(
        video_path=_resolve_input_path(
            reference_raw.get("video_path"), path.parent, "reference_video.video_path"
        ),
        camera_matrix=_matrix3(
            reference_raw.get("camera_matrix"), "reference_video.camera_matrix"
        ),
        distortion_coefficients=_distortion(
            reference_raw.get("distortion_coefficients"),
            "reference_video.distortion_coefficients",
        ),
        sample_count=sample_count,
    )
    return CalibrationInput(tuple(cameras), reference)


def load_remote_calibration_input(
    config_path: str | Path, metadata: dict[str, Any]
) -> CalibrationInput:
    """Load server reference input and edge-provided camera calibration metadata."""
    path = Path(config_path).expanduser().resolve()
    root = _require_mapping(json.loads(path.read_text(encoding="utf-8")), "config")
    cameras_raw = metadata.get("cameras")
    if not isinstance(cameras_raw, list) or not cameras_raw:
        raise ValueError("feature bundle has no camera metadata")
    cameras = tuple(
        CameraInput(
            camera_id=str(item["camera_key"]),
            image_path=None,
            camera_matrix=_matrix3(item.get("camera_matrix"), f"camera[{index}].camera_matrix"),
            distortion_coefficients=_distortion(
                item.get("distortion_coefficients"),
                f"camera[{index}].distortion_coefficients",
            ),
        )
        for index, item in enumerate(cameras_raw)
    )
    reference_raw = _require_mapping(root.get("reference_video"), "reference_video")
    sample_count_raw = reference_raw.get("sample_count")
    reference = ReferenceVideoInput(
        video_path=_resolve_input_path(
            reference_raw.get("video_path"), path.parent, "reference_video.video_path"
        ),
        camera_matrix=_matrix3(
            reference_raw.get("camera_matrix"), "reference_video.camera_matrix"
        ),
        distortion_coefficients=_distortion(
            reference_raw.get("distortion_coefficients"),
            "reference_video.distortion_coefficients",
        ),
        sample_count=None if sample_count_raw is None else int(sample_count_raw),
    )
    if reference.sample_count is not None and reference.sample_count <= 0:
        raise ValueError("reference_video.sample_count must be positive")
    return CalibrationInput(cameras, reference)


def estimate_max_images_from_gpu(
    device: str = "cuda", reserve_gib: float = 1.0
) -> int:
    """Conservatively estimate a VGGT-Omega frame limit from free GPU memory.

    The released 512px benchmark reports about 6.02 GiB for one frame and
    43.15 GiB for 500 frames.  Linear interpolation is only a planning estimate;
    ``--max-images`` remains the reliable override for a particular GPU.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("VGGT-Omega inference requires a CUDA GPU.")
    device_index = torch.device(device).index
    if device_index is None:
        device_index = torch.cuda.current_device()
    free_bytes, _ = torch.cuda.mem_get_info(device_index)
    free_gib = free_bytes / (1024**3)
    usable_gib = free_gib - float(reserve_gib)
    one_frame_gib = 6.02
    gib_per_extra_frame = (43.15 - one_frame_gib) / 499.0
    if usable_gib < one_frame_gib:
        raise RuntimeError(
            f"Only {free_gib:.2f} GiB is free on {device}; at least "
            f"{one_frame_gib + reserve_gib:.2f} GiB is recommended."
        )
    frame_count = 1 + math.floor((usable_gib - one_frame_gib) / gib_per_extra_frame)
    return max(1, min(500, frame_count))


def undistort_image(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
    *,
    alpha: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    if image is None or image.ndim not in {2, 3}:
        raise ValueError("A valid OpenCV image is required for undistortion.")
    height, width = image.shape[:2]
    new_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        distortion_coefficients,
        (width, height),
        alpha,
        (width, height),
    )
    map_x, map_y = cv2.initUndistortRectifyMap(
        camera_matrix,
        distortion_coefficients,
        None,
        new_matrix,
        (width, height),
        cv2.CV_32FC1,
    )
    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR), new_matrix


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write image: {path}")


def sample_video_frames(video_path: Path, sample_count: int) -> list[tuple[int, np.ndarray]]:
    """Read approximately uniform frames, including both ends of the video."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive.")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open reference video: {video_path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            raise RuntimeError(
                "Reference video does not expose a frame count; transcode it to "
                "a seekable MP4 before calibration."
            )
        target_count = min(sample_count, frame_count)
        indices = np.linspace(0, frame_count - 1, target_count, dtype=np.int64)
        results: list[tuple[int, np.ndarray]] = []
        for frame_index in np.unique(indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if not ok or frame is None:
                LOGGER.warning("Skipping unreadable video frame %d", frame_index)
                continue
            results.append((int(frame_index), frame))
        if not results:
            raise RuntimeError(f"No frames could be decoded from: {video_path}")
        return results
    finally:
        capture.release()


def prepare_inputs(
    calibration_input: CalibrationInput,
    output_dir: Path,
    max_images: int,
    *,
    undistort_alpha: float = 0.0,
) -> PreparedInputs:
    cctv_count = len(calibration_input.cctv_cameras)
    if max_images <= cctv_count:
        raise ValueError(
            f"max_images ({max_images}) must leave room for at least one reference "
            f"frame after {cctv_count} CCTV image(s)."
        )

    undistorted_dir = output_dir / "undistorted"
    paths: list[Path] = []
    cctv_new_matrices: list[np.ndarray] = []
    for camera in calibration_input.cctv_cameras:
        image = cv2.imread(str(camera.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read CCTV image: {camera.image_path}")
        corrected, new_matrix = undistort_image(
            image,
            camera.camera_matrix,
            camera.distortion_coefficients,
            alpha=undistort_alpha,
        )
        destination = undistorted_dir / "cctv" / f"{_safe_name(camera.camera_id)}.png"
        _write_image(destination, corrected)
        paths.append(destination)
        cctv_new_matrices.append(new_matrix)

    reference_slots = max_images - cctv_count
    requested = calibration_input.reference_video.sample_count
    if requested is not None:
        reference_slots = min(reference_slots, requested)
    sampled_frames = sample_video_frames(
        calibration_input.reference_video.video_path, reference_slots
    )

    reference_indices: list[int] = []
    reference_new_matrix: np.ndarray | None = None
    for sequence_index, (video_frame_index, frame) in enumerate(sampled_frames):
        corrected, new_matrix = undistort_image(
            frame,
            calibration_input.reference_video.camera_matrix,
            calibration_input.reference_video.distortion_coefficients,
            alpha=undistort_alpha,
        )
        if reference_new_matrix is None:
            reference_new_matrix = new_matrix
        destination = (
            undistorted_dir
            / "reference"
            / f"frame_{sequence_index:04d}_source_{video_frame_index:08d}.png"
        )
        _write_image(destination, corrected)
        paths.append(destination)
        reference_indices.append(video_frame_index)

    assert reference_new_matrix is not None
    return PreparedInputs(
        image_paths=tuple(paths),
        cctv_count=cctv_count,
        cctv_new_camera_matrices=tuple(cctv_new_matrices),
        reference_frame_indices=tuple(reference_indices),
        reference_new_camera_matrix=reference_new_matrix,
    )


def prepare_remote_inputs(
    calibration_input: CalibrationInput,
    metadata: dict[str, Any],
    output_dir: Path,
    max_images: int,
) -> RemotePreparedInputs:
    """Preprocess server-owned reference frames with the edge token contract."""
    cctv_count = len(calibration_input.cctv_cameras)
    if max_images <= cctv_count:
        raise ValueError("max_images must leave room for a reference frame")
    image_size_value = metadata.get("image_size")
    if (
        not isinstance(image_size_value, list)
        or len(image_size_value) != 2
        or image_size_value[0] != image_size_value[1]
    ):
        raise ValueError("only square distributed preprocessing is supported")
    image_size = int(image_size_value[0])
    if image_size <= 0 or image_size % 16:
        raise ValueError("distributed image_size must be a positive multiple of 16")
    if metadata.get("patch_size") != 16:
        raise ValueError("unsupported distributed patch size")
    if metadata.get("encoder_dtype") != "float16":
        raise ValueError("unsupported distributed encoder dtype")
    reference_slots = max_images - cctv_count
    if calibration_input.reference_video.sample_count is not None:
        reference_slots = min(reference_slots, calibration_input.reference_video.sample_count)
    sampled = sample_video_frames(calibration_input.reference_video.video_path, reference_slots)
    images: list[np.ndarray] = []
    indices: list[int] = []
    reference_matrix: np.ndarray | None = None
    artifact_dir = output_dir / "processed" / "reference"
    for sequence_index, (frame_index, frame) in enumerate(sampled):
        image, adjusted = preprocess_frame(
            frame,
            calibration_input.reference_video.camera_matrix,
            calibration_input.reference_video.distortion_coefficients,
            image_size,
        )
        images.append(image)
        indices.append(frame_index)
        reference_matrix = adjusted
        _write_image(
            artifact_dir / f"frame_{sequence_index:04d}_source_{frame_index:08d}.png",
            cv2.cvtColor((image.transpose(1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
        )
    assert reference_matrix is not None
    edge_cameras = metadata["cameras"]
    matrices = tuple(
        _matrix3(camera["processed_camera_matrix"], "processed_camera_matrix")
        for camera in edge_cameras
    )
    return RemotePreparedInputs(
        reference_images=np.stack(images),
        cctv_count=cctv_count,
        cctv_new_camera_matrices=matrices,
        reference_frame_indices=tuple(indices),
        reference_new_camera_matrix=reference_matrix,
    )


def load_marker_tree(tree_path: str | Path) -> dict[tuple[str, int], MarkerDefinition]:
    """Read marker identities, physical sizes, and metric world corners from USD."""
    try:
        from pxr import Gf, Usd, UsdGeom
    except ImportError as exc:
        raise RuntimeError(
            "Pixar USD Python modules are required to read the marker tree."
        ) from exc

    path = Path(tree_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Marker tree does not exist: {path}")
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"Could not open marker tree: {path}")
    markers_prim = stage.GetPrimAtPath("/ArUcoMarkerTree/Markers")
    if not markers_prim.IsValid():
        raise ValueError(f"USD is not an ArUco marker tree: {path}")

    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage) or 1.0)
    definitions: dict[tuple[str, int], MarkerDefinition] = {}
    for marker in markers_prim.GetChildren():
        dictionary_name = marker.GetCustomDataByKey("aruco:dictionary")
        marker_id = marker.GetCustomDataByKey("aruco:markerId")
        marker_length_mm = marker.GetCustomDataByKey("aruco:markerLengthMm")
        if dictionary_name is None or marker_id is None or marker_length_mm is None:
            raise ValueError(f"Marker metadata is incomplete: {marker.GetPath()}")
        identity = (str(dictionary_name), int(marker_id))
        if identity in definitions:
            raise ValueError(
                f"Marker tree has duplicate identity {identity[0]} ID {identity[1]}."
            )

        marker_length_m = float(marker_length_mm) / 1000.0
        half_in_stage_units = marker_length_m / meters_per_unit / 2.0
        transform = UsdGeom.Xformable(marker).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        local_corners = (
            (-half_in_stage_units, -half_in_stage_units, 0.0),
            (half_in_stage_units, -half_in_stage_units, 0.0),
            (half_in_stage_units, half_in_stage_units, 0.0),
            (-half_in_stage_units, half_in_stage_units, 0.0),
        )
        world_corners = np.asarray(
            [
                tuple(transform.Transform(Gf.Vec3d(*corner)))
                for corner in local_corners
            ],
            dtype=np.float64,
        ) * meters_per_unit
        definitions[identity] = MarkerDefinition(
            dictionary_name=identity[0],
            marker_id=identity[1],
            marker_length_m=marker_length_m,
            world_corners=world_corners,
            prim_path=marker.GetPath().pathString,
        )

    if not definitions:
        raise ValueError(f"Marker tree contains no markers: {path}")
    return definitions


def _aruco_dictionary(dictionary_name: str) -> Any:
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("Install opencv-contrib-python; cv2.aruco is unavailable.")
    dictionary_id = getattr(cv2.aruco, dictionary_name, None)
    if dictionary_id is None:
        raise ValueError(f"OpenCV does not support ArUco dictionary {dictionary_name}.")
    return cv2.aruco.getPredefinedDictionary(dictionary_id)


def _rgb_to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0) * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def detect_known_markers(
    rgb_image: np.ndarray,
    marker_definitions: dict[tuple[str, int], MarkerDefinition],
) -> list[tuple[MarkerDefinition, np.ndarray]]:
    """Return known markers and their OpenCV-ordered image corners."""
    gray = cv2.cvtColor(_rgb_to_uint8(rgb_image), cv2.COLOR_RGB2GRAY)
    detections: list[tuple[MarkerDefinition, np.ndarray]] = []
    dictionary_names = sorted({key[0] for key in marker_definitions})
    for dictionary_name in dictionary_names:
        dictionary = _aruco_dictionary(dictionary_name)
        if hasattr(cv2.aruco, "ArucoDetector"):
            detector = cv2.aruco.ArucoDetector(dictionary)
            corners, ids, _ = detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
        if ids is None:
            continue
        for marker_corners, marker_id_array in zip(corners, ids.reshape(-1), strict=True):
            identity = (dictionary_name, int(marker_id_array))
            definition = marker_definitions.get(identity)
            if definition is None:
                continue
            detections.append(
                (definition, np.asarray(marker_corners, dtype=np.float64).reshape(4, 2))
            )
    return detections


def _depth_at_corner(
    depth: np.ndarray,
    confidence: np.ndarray,
    corner_xy: np.ndarray,
    min_confidence: float,
    radius: int = 2,
) -> float | None:
    height, width = depth.shape
    x = int(round(float(corner_xy[0])))
    y = int(round(float(corner_xy[1])))
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    patch_depth = depth[y0:y1, x0:x1]
    patch_confidence = confidence[y0:y1, x0:x1]
    valid = (
        np.isfinite(patch_depth)
        & (patch_depth > 0)
        & np.isfinite(patch_confidence)
        & (patch_confidence >= min_confidence)
    )
    if not np.any(valid):
        return None
    return float(np.median(patch_depth[valid]))


def unproject_pixel(
    pixel_xy: np.ndarray,
    depth: float,
    world_to_camera: np.ndarray,
    camera_matrix: np.ndarray,
) -> np.ndarray:
    """Unproject one pixel into VGGT's reconstruction coordinate system."""
    u, v = (float(value) for value in pixel_xy)
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    camera_point = np.asarray(
        [(u - cx) / fx * depth, (v - cy) / fy * depth, depth],
        dtype=np.float64,
    )
    rotation = world_to_camera[:3, :3]
    translation = world_to_camera[:3, 3]
    return rotation.T @ (camera_point - translation)


def collect_marker_correspondences(
    processed_rgb: np.ndarray,
    depths: np.ndarray,
    depth_confidences: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    reference_start: int,
    marker_definitions: dict[tuple[str, int], MarkerDefinition],
    *,
    min_depth_confidence: float,
    artifact_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    reconstructed_points: list[np.ndarray] = []
    world_points: list[np.ndarray] = []
    observations: list[dict[str, Any]] = []

    for image_index in range(reference_start, len(processed_rgb)):
        detections = detect_known_markers(processed_rgb[image_index], marker_definitions)
        annotated = cv2.cvtColor(
            _rgb_to_uint8(processed_rgb[image_index]), cv2.COLOR_RGB2BGR
        )
        if detections:
            cv2.aruco.drawDetectedMarkers(
                annotated,
                [corners.astype(np.float32).reshape(1, 4, 2) for _, corners in detections],
                np.asarray([[definition.marker_id] for definition, _ in detections]),
            )
        accepted_corners = 0
        for definition, image_corners in detections:
            marker_accepted = 0
            for corner_index, image_corner in enumerate(image_corners):
                sampled_depth = _depth_at_corner(
                    depths[image_index],
                    depth_confidences[image_index],
                    image_corner,
                    min_depth_confidence,
                )
                if sampled_depth is None:
                    continue
                reconstructed_points.append(
                    unproject_pixel(
                        image_corner,
                        sampled_depth,
                        extrinsics[image_index],
                        intrinsics[image_index],
                    )
                )
                world_points.append(definition.world_corners[corner_index])
                accepted_corners += 1
                marker_accepted += 1
            observations.append(
                {
                    "image_index": image_index,
                    "dictionary": definition.dictionary_name,
                    "marker_id": definition.marker_id,
                    "accepted_corners": marker_accepted,
                }
            )
        LOGGER.info(
            "Reference image %d: %d known marker(s), %d usable corner(s)",
            image_index - reference_start,
            len(detections),
            accepted_corners,
        )
        if artifact_dir is not None:
            _write_image(
                artifact_dir / f"reference_{image_index - reference_start:04d}.png",
                annotated,
            )

    if len(reconstructed_points) < 3:
        raise RuntimeError(
            "Fewer than three usable ArUco corner correspondences were found. "
            "Ensure the reference video clearly sees markers registered in the USD."
        )
    return (
        np.asarray(reconstructed_points, dtype=np.float64),
        np.asarray(world_points, dtype=np.float64),
        observations,
    )


def fit_similarity_transform(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit ``target = scale * rotation @ source + translation`` (Umeyama)."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must have matching Nx3 shapes.")
    if len(source) < 3:
        raise ValueError("At least three point correspondences are required.")

    source_centered = source - source.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    source_variance = float(np.sum(source_centered**2) / len(source))
    if source_variance <= 1.0e-12 or np.linalg.matrix_rank(source_centered) < 2:
        raise ValueError("Source correspondences are degenerate.")
    if np.linalg.matrix_rank(target_centered) < 2:
        raise ValueError("Target correspondences are degenerate.")

    covariance = target_centered.T @ source_centered / len(source)
    left, singular_values, right_transpose = np.linalg.svd(covariance)
    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(left @ right_transpose) < 0:
        sign[-1] = -1.0
    rotation = left @ np.diag(sign) @ right_transpose
    scale = float(np.sum(singular_values * sign) / source_variance)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Estimated similarity scale is not positive.")
    translation = target.mean(axis=0) - scale * (rotation @ source.mean(axis=0))
    return scale, rotation, translation


def robust_similarity_transform(
    source: np.ndarray,
    target: np.ndarray,
    *,
    inlier_threshold_m: float = 0.10,
    max_hypotheses: int = 1000,
    random_seed: int = 0,
) -> SimilarityTransform:
    """RANSAC wrapper around the metric similarity fit."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must have matching Nx3 shapes.")
    if len(source) < 3:
        raise ValueError("At least three point correspondences are required.")
    if inlier_threshold_m <= 0:
        raise ValueError("inlier_threshold_m must be positive.")

    all_combinations: Iterable[tuple[int, int, int]] = itertools.combinations(
        range(len(source)), 3
    )
    combination_count = math.comb(len(source), 3)
    if combination_count <= max_hypotheses:
        samples = list(all_combinations)
    else:
        rng = np.random.default_rng(random_seed)
        sample_set: set[tuple[int, int, int]] = set()
        while len(sample_set) < max_hypotheses:
            sample_set.add(tuple(sorted(rng.choice(len(source), 3, replace=False).tolist())))
        samples = list(sample_set)

    best_mask: np.ndarray | None = None
    best_score = (-1, -math.inf)
    for indices in samples:
        try:
            scale, rotation, translation = fit_similarity_transform(
                source[list(indices)], target[list(indices)]
            )
        except ValueError:
            continue
        predicted = scale * (source @ rotation.T) + translation
        residuals = np.linalg.norm(predicted - target, axis=1)
        mask = residuals <= inlier_threshold_m
        inlier_count = int(mask.sum())
        if inlier_count < 3:
            continue
        median_error = float(np.median(residuals[mask]))
        score = (inlier_count, -median_error)
        if score > best_score:
            best_score = score
            best_mask = mask

    if best_mask is None:
        raise RuntimeError(
            "Could not find a non-degenerate ArUco alignment. Increase marker "
            "visibility or adjust --alignment-threshold-m."
        )
    scale, rotation, translation = fit_similarity_transform(
        source[best_mask], target[best_mask]
    )
    residuals = np.linalg.norm(
        scale * (source @ rotation.T) + translation - target,
        axis=1,
    )
    final_mask = residuals <= inlier_threshold_m
    if int(final_mask.sum()) >= 3 and not np.array_equal(final_mask, best_mask):
        scale, rotation, translation = fit_similarity_transform(
            source[final_mask], target[final_mask]
        )
        residuals = np.linalg.norm(
            scale * (source @ rotation.T) + translation - target,
            axis=1,
        )
    else:
        final_mask = best_mask
    return SimilarityTransform(scale, rotation, translation, final_mask, residuals)


def transform_camera_extrinsic(
    vggt_world_to_camera: np.ndarray,
    alignment: SimilarityTransform,
) -> tuple[np.ndarray, np.ndarray]:
    """Move a VGGT camera pose into the metric marker-tree world."""
    source_extrinsic = np.asarray(vggt_world_to_camera, dtype=np.float64)
    if source_extrinsic.shape != (3, 4):
        raise ValueError("VGGT extrinsic must have shape 3x4.")
    source_rotation = source_extrinsic[:, :3]
    source_translation = source_extrinsic[:, 3]
    source_camera_center = -source_rotation.T @ source_translation

    world_camera_center = alignment.transform_points(source_camera_center)
    world_to_camera_rotation = source_rotation @ alignment.rotation.T
    world_to_camera_translation = -world_to_camera_rotation @ world_camera_center

    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3, :3] = world_to_camera_rotation
    world_to_camera[:3, 3] = world_to_camera_translation
    camera_to_world = np.linalg.inv(world_to_camera)
    return world_to_camera, camera_to_world


def _load_vggt_api() -> tuple[Any, Any, Any]:
    if not VGGT_OMEGA_ROOT.is_dir():
        raise FileNotFoundError(
            f"VGGT-Omega submodule is missing: {VGGT_OMEGA_ROOT}. Run "
            "'git submodule update --init --recursive'."
        )
    module_root = str(VGGT_OMEGA_ROOT)
    if module_root not in sys.path:
        sys.path.insert(0, module_root)
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    return VGGTOmega, load_and_preprocess_images, encoding_to_camera


def _depth_outputs_to_numpy(
    depth: torch.Tensor,
    confidence: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize VGGT dense-head outputs to matching [N, H, W] arrays."""
    if depth.ndim != 5 or depth.shape[0] != 1 or depth.shape[-1] != 1:
        raise ValueError(f"unexpected VGGT depth shape: {tuple(depth.shape)}")
    depth_array = depth[0, ..., 0].detach().float().cpu().numpy()

    if confidence.ndim == 4 and confidence.shape[0] == 1:
        confidence_array = confidence[0].detach().float().cpu().numpy()
    elif (
        confidence.ndim == 5
        and confidence.shape[0] == 1
        and confidence.shape[-1] == 1
    ):
        confidence_array = confidence[0, ..., 0].detach().float().cpu().numpy()
    else:
        raise ValueError(
            f"unexpected VGGT depth confidence shape: {tuple(confidence.shape)}"
        )
    if depth_array.shape != confidence_array.shape:
        raise ValueError(
            "VGGT depth/confidence shapes do not match after conversion: "
            f"{depth_array.shape} vs {confidence_array.shape}"
        )
    return depth_array, confidence_array


def run_vggt_inference(
    image_paths: Sequence[Path],
    checkpoint_path: Path,
    *,
    device: str,
    image_resolution: int,
    resize_mode: str,
) -> dict[str, np.ndarray]:
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise RuntimeError("VGGT-Omega requires a CUDA device.")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"VGGT-Omega checkpoint not found: {checkpoint_path}")
    if image_resolution <= 0 or image_resolution % 16:
        raise ValueError("image_resolution must be a positive multiple of 16.")

    VGGTOmega, load_and_preprocess_images, encoding_to_camera = _load_vggt_api()
    LOGGER.info("Loading VGGT-Omega checkpoint: %s", checkpoint_path)
    model = VGGTOmega().eval()
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)
    model = model.to(device)

    images = load_and_preprocess_images(
        [str(path) for path in image_paths],
        mode=resize_mode,
        image_resolution=image_resolution,
    ).to(device)
    LOGGER.info("Running VGGT-Omega on tensor %s", tuple(images.shape))
    try:
        with torch.inference_mode():
            predictions = model(images)
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            "VGGT-Omega ran out of GPU memory; retry with a smaller --max-images."
        ) from exc

    extrinsics, intrinsics = encoding_to_camera(
        predictions["pose_enc"], predictions["images"].shape[-2:]
    )
    depth = predictions["depth"]
    depth_confidence = predictions["depth_conf"]
    LOGGER.info(
        "VGGT output shapes: images=%s depth=%s confidence=%s extrinsics=%s intrinsics=%s",
        tuple(predictions["images"].shape),
        tuple(depth.shape),
        tuple(depth_confidence.shape),
        tuple(extrinsics.shape),
        tuple(intrinsics.shape),
    )
    depth_array, depth_confidence_array = _depth_outputs_to_numpy(
        depth, depth_confidence
    )
    result = {
        "images": predictions["images"][0]
        .detach()
        .float()
        .cpu()
        .permute(0, 2, 3, 1)
        .numpy(),
        "depth": depth_array,
        "depth_conf": depth_confidence_array,
        "extrinsics": extrinsics[0].detach().float().cpu().numpy(),
        "intrinsics": intrinsics[0].detach().float().cpu().numpy(),
    }
    del predictions, images, model
    torch.cuda.empty_cache()
    return result


def run_vggt_feature_inference(
    edge_tokens: np.ndarray,
    reference_images: np.ndarray,
    checkpoint_path: Path,
    *,
    device: str,
) -> dict[str, np.ndarray]:
    """Encode server reference images and jointly infer from all patch tokens."""
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        raise RuntimeError("VGGT-Omega requires a CUDA device")
    VGGTOmega, _, encoding_to_camera = _load_vggt_api()
    model = VGGTOmega().eval()
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)
    model = model.to(device)
    reference_tensor = torch.from_numpy(reference_images).to(device)
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            reference_tokens = model.aggregator.encode_patch_tokens(reference_tensor.unsqueeze(0))[0]
        remote_tokens = torch.from_numpy(edge_tokens).to(device=device, dtype=reference_tokens.dtype)
        if remote_tokens.shape[1:] != reference_tokens.shape[1:]:
            raise ValueError(
                f"edge token shape {tuple(remote_tokens.shape[1:])} does not match "
                f"server token shape {tuple(reference_tokens.shape[1:])}"
            )
        tokens = torch.cat([remote_tokens, reference_tokens], dim=0).unsqueeze(0)
        placeholder = torch.zeros(
            (remote_tokens.shape[0], *reference_tensor.shape[1:]),
            device=device,
            dtype=reference_tensor.dtype,
        )
        images = torch.cat([placeholder, reference_tensor], dim=0).unsqueeze(0)
        predictions = model.forward_patch_tokens(tokens, images)
    extrinsics, intrinsics = encoding_to_camera(
        predictions["pose_enc"], predictions["images"].shape[-2:]
    )
    depth_array, depth_confidence_array = _depth_outputs_to_numpy(
        predictions["depth"], predictions["depth_conf"]
    )
    result = {
        "images": predictions["images"][0].detach().float().cpu().permute(0, 2, 3, 1).numpy(),
        "depth": depth_array,
        "depth_conf": depth_confidence_array,
        "extrinsics": extrinsics[0].detach().float().cpu().numpy(),
        "intrinsics": intrinsics[0].detach().float().cpu().numpy(),
    }
    del predictions, images, tokens, model
    torch.cuda.empty_cache()
    return result


def _json_matrix(matrix: np.ndarray) -> list[list[float]]:
    return np.asarray(matrix, dtype=np.float64).tolist()


def run_calibration(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    max_images = args.max_images
    if max_images is None:
        max_images = estimate_max_images_from_gpu(args.device)
        LOGGER.info("GPU memory estimate allows up to %d total images", max_images)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    feature_metadata: dict[str, Any] | None = None
    if args.feature_bundle:
        feature_payload = Path(args.feature_bundle).expanduser().resolve().read_bytes()
        feature_metadata, edge_tokens = decode_feature_bundle(feature_payload)
        if feature_metadata.get("preprocess") != PREPROCESS_NAME:
            raise ValueError("feature bundle preprocessing contract is unsupported")
        if feature_metadata.get("checkpoint_sha256") != file_sha256(checkpoint):
            raise ValueError("edge and server VGGT checkpoints do not match")
        calibration_input = load_remote_calibration_input(args.config, feature_metadata)
        prepared = prepare_remote_inputs(
            calibration_input, feature_metadata, output_dir, max_images
        )
    else:
        calibration_input = load_calibration_input(args.config)
        prepared = prepare_inputs(
            calibration_input,
            output_dir,
            max_images,
            undistort_alpha=args.undistort_alpha,
        )
    LOGGER.info(
        "Prepared %d CCTV image(s) and %d reference frame(s)",
        prepared.cctv_count,
        len(prepared.reference_frame_indices),
    )

    if feature_metadata is not None:
        predictions = run_vggt_feature_inference(
            edge_tokens,
            prepared.reference_images,
            checkpoint,
            device=args.device,
        )
    else:
        predictions = run_vggt_inference(
            prepared.image_paths,
            checkpoint,
            device=args.device,
            image_resolution=args.image_resolution,
            resize_mode=args.resize_mode,
        )
    marker_definitions = load_marker_tree(args.marker_tree)
    source_points, target_points, observations = collect_marker_correspondences(
        predictions["images"],
        predictions["depth"],
        predictions["depth_conf"],
        predictions["extrinsics"],
        predictions["intrinsics"],
        prepared.cctv_count,
        marker_definitions,
        min_depth_confidence=args.min_depth_confidence,
        artifact_dir=output_dir / "aruco_detections",
    )
    alignment = robust_similarity_transform(
        source_points,
        target_points,
        inlier_threshold_m=args.alignment_threshold_m,
    )
    inlier_residuals = alignment.residuals[alignment.inlier_mask]
    LOGGER.info(
        "Alignment: %d/%d inliers, scale %.6f, RMSE %.4f m",
        int(alignment.inlier_mask.sum()),
        len(alignment.inlier_mask),
        alignment.scale,
        float(np.sqrt(np.mean(inlier_residuals**2))),
    )

    cameras_output = []
    for index, camera in enumerate(calibration_input.cctv_cameras):
        world_to_camera, camera_to_world = transform_camera_extrinsic(
            predictions["extrinsics"][index], alignment
        )
        camera_output = {
                "camera_id": camera.camera_id,
                "source_image": str(camera.image_path) if camera.image_path else None,
                "camera_matrix": _json_matrix(camera.camera_matrix),
                "distortion_coefficients": camera.distortion_coefficients.tolist(),
                "undistorted_camera_matrix": _json_matrix(
                    prepared.cctv_new_camera_matrices[index]
                ),
                "world_to_camera": _json_matrix(world_to_camera),
                "camera_to_world": _json_matrix(camera_to_world),
                "position_m": camera_to_world[:3, 3].tolist(),
            }
        if feature_metadata is not None:
            edge_camera = feature_metadata["cameras"][index]
            camera_output.update(
                {
                    "edge_id": feature_metadata["edge_id"],
                    "edge_camera_id": edge_camera.get("camera_id"),
                    "name": edge_camera.get("name"),
                    "location": edge_camera.get("location"),
                    "twin_id": edge_camera.get("twin_id"),
                    "source_image_size": edge_camera.get("source_image_size"),
                }
            )
        cameras_output.append(camera_output)

    result = {
        "schema_version": 1,
        "coordinate_convention": {
            "world": "marker-tree USD coordinates in metres",
            "camera": "OpenCV: +X right, +Y down, +Z forward",
            "extrinsic": "world_to_camera maps homogeneous world points to camera points",
        },
        "marker_tree": str(Path(args.marker_tree).expanduser().resolve()),
        "input": {
            "mode": "edge_patch_tokens" if feature_metadata else "local_images",
            "edge_id": feature_metadata.get("edge_id") if feature_metadata else None,
            "request_id": feature_metadata.get("request_id") if feature_metadata else None,
            "checkpoint_sha256": file_sha256(checkpoint),
            "preprocess": feature_metadata.get("preprocess") if feature_metadata else None,
            "total_images": prepared.cctv_count + len(prepared.reference_frame_indices),
            "cctv_images": prepared.cctv_count,
            "reference_frames": len(prepared.reference_frame_indices),
            "reference_video_frame_indices": list(prepared.reference_frame_indices),
            "image_resolution": args.image_resolution,
            "resize_mode": args.resize_mode,
        },
        "alignment": {
            "scale": alignment.scale,
            "rotation": _json_matrix(alignment.rotation),
            "translation_m": alignment.translation.tolist(),
            "vggt_to_usd": _json_matrix(alignment.matrix4()),
            "correspondence_count": len(source_points),
            "inlier_count": int(alignment.inlier_mask.sum()),
            "rmse_m": float(np.sqrt(np.mean(inlier_residuals**2))),
            "max_error_m": float(np.max(inlier_residuals)),
            "inlier_threshold_m": args.alignment_threshold_m,
        },
        "marker_observations": observations,
        "cameras": cameras_output,
    }
    result_path = output_dir / "calibration_result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info("Saved calibration result: %s", result_path)
    return result_path


def _registered_camera_count(record: dict[str, Any]) -> int:
    cameras = record.get("cameras") or {}
    if isinstance(cameras, dict):
        count = sum(
            1
            for camera in cameras.values()
            if isinstance(camera, dict) and camera.get("exists", True)
        )
        if count:
            return count
    return int((record.get("last_status") or {}).get("camera_count") or 0)


def select_edge_interactively(
    edges: dict[str, dict[str, Any]], requested_edge_id: str | None = None
) -> tuple[str, dict[str, Any]]:
    if not edges:
        raise ValueError(
            "등록된 edge가 없습니다. 먼저 edge manager와 agent 등록을 완료하세요."
        )
    records = sorted(edges.values(), key=lambda item: str(item.get("edge_id")))
    print("\n등록된 edge")
    print("  번호  edge_id                  카메라  승인  상태")
    for index, record in enumerate(records, 1):
        print(
            f"  {index:<4}  {record['edge_id']:<23} "
            f"{_registered_camera_count(record):>4}대  "
            f"{'Y' if record.get('approved') else 'N':>3}  "
            f"{'ONLINE' if record.get('online') else 'OFFLINE'}"
        )

    answer = requested_edge_id or input(
        "\n캘리브레이션 대상 edge id 또는 번호: "
    ).strip()
    if answer.isdigit() and 1 <= int(answer) <= len(records):
        selected = records[int(answer) - 1]
    else:
        selected = next(
            (record for record in records if record.get("edge_id") == answer), None
        )
    if selected is None:
        raise ValueError("목록에 있는 edge id 또는 번호를 선택해주세요.")
    if not selected.get("approved"):
        raise ValueError(f"승인되지 않은 edge입니다: {selected['edge_id']}")
    if not selected.get("online"):
        raise ValueError(f"현재 offline 상태인 edge입니다: {selected['edge_id']}")
    if _registered_camera_count(selected) <= 0:
        raise ValueError(f"등록된 카메라가 없는 edge입니다: {selected['edge_id']}")
    return str(selected["edge_id"]), selected


def _read_intrinsic_sidecar(path: Path, label: str) -> dict[str, Any] | None:
    candidates = (
        Path(f"{path}.json"),
        path.with_suffix(".camera.json"),
    )
    for candidate in candidates:
        if not candidate.is_file():
            continue
        raw = json.loads(candidate.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"reference sidecar must be a JSON object: {candidate}")
        source = raw.get("intrinsic") if isinstance(raw.get("intrinsic"), dict) else raw
        return {
            "camera_matrix": _matrix3(
                source.get("camera_matrix"), f"{label} sidecar camera_matrix"
            ).tolist(),
            "distortion_coefficients": _distortion(
                source.get("distortion_coefficients"),
                f"{label} sidecar distortion_coefficients",
            ).tolist(),
            "camera_id": raw.get("camera_id"),
            "image_size": source.get("image_size"),
            "sidecar": str(candidate.resolve()),
        }
    return None


def _pinhole_intrinsic(width: int, height: int) -> tuple[list[list[float]], list[float]]:
    focal = float(max(width, height))
    return (
        [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    )


def _scale_sidecar_matrix(
    matrix: list[list[float]],
    calibration_size: Any,
    width: int,
    height: int,
    label: str,
) -> list[list[float]]:
    if calibration_size is None:
        return matrix
    if (
        not isinstance(calibration_size, list)
        or len(calibration_size) != 2
        or min(int(value) for value in calibration_size) <= 0
    ):
        raise ValueError(f"{label} sidecar image_size must be [width, height]")
    calibration_width, calibration_height = (
        int(value) for value in calibration_size
    )
    scaled = np.asarray(matrix, dtype=np.float64).copy()
    scaled[0, :] *= width / calibration_width
    scaled[1, :] *= height / calibration_height
    return scaled.tolist()


def _offline_image_paths(value: str | Path) -> list[Path]:
    source = Path(value).expanduser().resolve()
    if source.is_file():
        if source.suffix.lower() not in SUPPORTED_OFFLINE_IMAGE_SUFFIXES:
            raise ValueError(f"지원하지 않는 이미지 형식입니다: {source.suffix}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"사진 또는 사진 폴더가 없습니다: {source}")
    paths = sorted(
        path.resolve()
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_OFFLINE_IMAGE_SUFFIXES
    )
    if not paths:
        raise ValueError(f"폴더에 지원되는 사진이 없습니다: {source}")
    return paths


def _load_edge_camera_intrinsics(
    config_path: str | Path | None,
) -> dict[str, dict[str, Any] | None]:
    """Read only calibration metadata from an edge-local camera YAML file."""
    if config_path is None:
        return {}
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"카메라 설정 파일이 없습니다: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict) or not isinstance(raw.get("CAMERAS"), list):
        raise ValueError(f"카메라 설정의 CAMERAS 목록이 올바르지 않습니다: {path}")

    calibrations: dict[str, dict[str, Any] | None] = {}
    for camera in raw["CAMERAS"]:
        if not isinstance(camera, dict):
            continue
        number = str(camera.get("id", "")).strip()
        if not number.isdigit() or int(number) <= 0:
            continue
        intrinsic = camera.get("intrinsic")
        if not isinstance(intrinsic, dict):
            calibrations[number] = None
            continue
        calibrations[number] = {
            "camera_id": f"camera/{number}",
            "camera_matrix": _matrix3(
                intrinsic.get("camera_matrix"),
                f"camera/{number} camera_matrix",
            ).tolist(),
            "distortion_coefficients": _distortion(
                intrinsic.get("distortion_coefficients"),
                f"camera/{number} distortion_coefficients",
            ).tolist(),
            "image_size": intrinsic.get("image_size"),
            "source": str(path),
        }
    return calibrations


def _captured_camera_number(image_path: Path) -> str | None:
    match = re.match(r"^camera_(\d+)(?:_|$)", image_path.stem, re.IGNORECASE)
    return match.group(1) if match else None


def build_offline_manifest(
    image_source: str | Path,
    reference_video: str | Path,
    destination: str | Path,
    *,
    sample_count: int = 40,
    camera_config: str | Path | None = None,
) -> tuple[Path, list[str]]:
    """Build a local-image calibration manifest from a photo or directory."""
    image_paths = _offline_image_paths(image_source)
    configured_intrinsics = _load_edge_camera_intrinsics(camera_config)
    cameras: list[dict[str, Any]] = []
    descriptions: list[str] = []
    used_ids: set[str] = set()
    for index, image_path in enumerate(image_paths, 1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"사진을 읽을 수 없습니다: {image_path}")
        height, width = image.shape[:2]
        sidecar = _read_intrinsic_sidecar(image_path, f"photo {image_path.name}")
        calibration_source: str | None = None
        if sidecar is None and configured_intrinsics:
            camera_number_value = _captured_camera_number(image_path)
            if camera_number_value in configured_intrinsics:
                sidecar = configured_intrinsics[camera_number_value]
                if sidecar is None:
                    raise ValueError(
                        f"camera/{camera_number_value} intrinsic이 등록되지 않았습니다: "
                        f"{Path(camera_config).expanduser().resolve()}"
                    )
                calibration_source = (
                    f"edge camera config {sidecar['source']}의 "
                    f"camera/{camera_number_value} intrinsic"
                )
        requested_id = str(sidecar.get("camera_id") or "").strip() if sidecar else ""
        camera_id = requested_id or image_path.stem
        if requested_id and requested_id in used_ids:
            raise ValueError(f"중복된 offline camera_id입니다: {requested_id}")
        if not camera_id or camera_id in used_ids:
            camera_id = f"camera_{index}"
        if camera_id in used_ids:
            raise ValueError(f"중복된 offline camera_id입니다: {camera_id}")
        used_ids.add(camera_id)
        if sidecar is None:
            camera_matrix, distortion = _pinhole_intrinsic(width, height)
            descriptions.append(
                f"{camera_id}: sidecar 없음, {width}x{height} pinhole/zero distortion 가정"
            )
        else:
            camera_matrix = _scale_sidecar_matrix(
                sidecar["camera_matrix"],
                sidecar["image_size"],
                width,
                height,
                f"photo {image_path.name}",
            )
            distortion = sidecar["distortion_coefficients"]
            source = calibration_source or f"sidecar {sidecar['sidecar']}"
            descriptions.append(f"{camera_id}: {source} 사용")
        cameras.append(
            {
                "camera_id": camera_id,
                "image_path": str(image_path),
                "camera_matrix": camera_matrix,
                "distortion_coefficients": distortion,
            }
        )

    reference_manifest = Path(destination).with_name("reference_input.json")
    reference_path, reference_description = build_reference_manifest(
        reference_video,
        reference_manifest,
        sample_count=sample_count,
    )
    reference = json.loads(reference_path.read_text(encoding="utf-8"))[
        "reference_video"
    ]
    manifest = {"cctv_cameras": cameras, "reference_video": reference}
    output = Path(destination).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    descriptions.append(f"reference: {reference_description}")
    return output, descriptions


def build_reference_manifest(
    video_path: str | Path,
    destination: str | Path,
    *,
    sample_count: int = 40,
) -> tuple[Path, str]:
    """Create the server manifest from a video and an optional intrinsic sidecar."""
    video = Path(video_path).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"reference 영상이 없습니다: {video}")
    if sample_count <= 0:
        raise ValueError("reference sample_count must be positive")
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"reference 영상을 열 수 없습니다: {video}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"reference 영상 크기를 확인할 수 없습니다: {video}")
            height, width = frame.shape[:2]
    finally:
        capture.release()

    sidecar = _read_intrinsic_sidecar(video, "reference")
    if sidecar is None:
        camera_matrix, distortion = _pinhole_intrinsic(width, height)
        source_description = (
            f"sidecar 없음: {width}x{height} pinhole, zero distortion 가정"
        )
    else:
        camera_matrix = _scale_sidecar_matrix(
            sidecar["camera_matrix"],
            sidecar["image_size"],
            width,
            height,
            "reference",
        )
        distortion = sidecar["distortion_coefficients"]
        source_description = f"intrinsic sidecar 사용: {sidecar['sidecar']}"

    manifest = {
        "reference_video": {
            "video_path": str(video),
            "camera_matrix": camera_matrix,
            "distortion_coefficients": distortion,
            "sample_count": sample_count,
        }
    }
    output = Path(destination).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output, source_description


def _resolve_checkpoint(value: str | None) -> Path:
    candidates = []
    if value:
        candidates.append(Path(value).expanduser())
    environment_value = os.getenv("VGGT_OMEGA_CHECKPOINT")
    if environment_value:
        candidates.append(Path(environment_value).expanduser())
    candidates.extend(
        [
            WORKER_DIR / "checkpoints/VGGT-Omega-1B-512.pt",
            WORKER_DIR / "checkpoints/model.pt",
        ]
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(
        "VGGT-Omega checkpoint가 없습니다. --checkpoint 또는 "
        "VGGT_OMEGA_CHECKPOINT를 지정하세요."
    )


def select_cli_mode(requested_mode: str | None = None) -> str:
    print("\n캘리브레이션 실행 모드")
    print("  1. 온라인  - 등록 edge에서 DINO token 수신")
    print("  2. 오프라인 - 로컬 사진/폴더에서 pose 계산, JSON만 출력")
    answer = (requested_mode or input("모드 선택 [1/2]: ").strip()).lower()
    aliases = {
        "1": "online",
        "online": "online",
        "온라인": "online",
        "2": "offline",
        "offline": "offline",
        "오프라인": "offline",
    }
    try:
        return aliases[answer]
    except KeyError:
        raise ValueError("온라인(1) 또는 오프라인(2)을 선택해주세요.") from None


def _interactive_marker_tree(value: str | None) -> Path:
    answer = value
    if answer is None:
        answer = input(
            f"매핑용 Isaac Sim USD 위치 [{DEFAULT_MARKER_TREE}]: "
        ).strip() or str(DEFAULT_MARKER_TREE)
    path = Path(answer).expanduser().resolve()
    marker_definitions = load_marker_tree(path)
    print(f"USD 마커 트리 확인: {len(marker_definitions)}개 마커")
    return path


def _offline_json_destination(args: argparse.Namespace) -> Path:
    if args.json_output:
        destination = Path(args.json_output).expanduser().resolve()
    elif args.output_dir:
        destination = (
            Path(args.output_dir).expanduser().resolve() / "calibration_result.json"
        )
    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        destination = (Path.cwd() / f"calibration_result_{timestamp}.json").resolve()
    if destination.suffix.lower() != ".json":
        raise ValueError("오프라인 output 경로는 .json 파일이어야 합니다.")
    return destination


def run_offline_calibration(args: argparse.Namespace) -> Path:
    """Calibrate local photos and persist only the final JSON result."""
    image_source = args.images or input(
        "포즈를 구할 사진 또는 사진 폴더 경로: "
    ).strip()
    reference_video = args.reference_video or input(
        "reference 용 영상 경로를 입력해주세요: "
    ).strip()
    marker_tree = _interactive_marker_tree(args.marker_tree)
    checkpoint = _resolve_checkpoint(args.checkpoint)
    destination = _offline_json_destination(args)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="dt-offline-calibration-") as temporary:
        working_dir = Path(temporary)
        manifest_path, descriptions = build_offline_manifest(
            image_source,
            reference_video,
            working_dir / "offline_input.json",
            sample_count=args.reference_sample_count,
            camera_config=getattr(args, "camera_config", None),
        )
        print("\nOffline 입력 intrinsic")
        for description in descriptions:
            print(f"  - {description}")
        print("로컬 사진과 reference 영상을 결합해 VGGT-Omega 추론을 시작합니다.")

        args.config = str(manifest_path)
        args.feature_bundle = None
        args.checkpoint = str(checkpoint)
        args.output_dir = str(working_dir)
        args.marker_tree = str(marker_tree)
        temporary_result = run_calibration(args)
        result = json.loads(temporary_result.read_text(encoding="utf-8"))
        result["input"]["mode"] = "offline_images"
        result["input"]["offline_image_source"] = str(
            Path(image_source).expanduser().resolve()
        )
        destination.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print("오프라인 모드: registry와 edge 설정은 변경하지 않았습니다.")
    return destination


def run_interactive_calibration(args: argparse.Namespace) -> Path:
    """Run capture, distributed inference, registry update, and edge sync."""
    from apps.calibration_worker.distributed import (
        publish_calibration_result,
        request_calibration_features,
    )
    from apps.edge_manager.app.protocol import DEFAULT_TOPIC_ROOT
    from apps.edge_manager.app.registry import EdgeRegistry

    registry = EdgeRegistry(args.registry)
    edge_id, edge_record = select_edge_interactively(
        registry.snapshot()["edges"], args.edge_id
    )

    reference_answer = args.reference_video or input(
        "reference 용 영상 경로를 입력해주세요: "
    ).strip()
    reference_video = Path(reference_answer).expanduser().resolve()
    marker_tree = _interactive_marker_tree(args.marker_tree)

    checkpoint = _resolve_checkpoint(args.checkpoint)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (DEFAULT_RUNS_DIR / f"{timestamp}_{edge_id}").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path, reference_description = build_reference_manifest(
        reference_video,
        output_dir / "reference_input.json",
        sample_count=args.reference_sample_count,
    )
    print(f"Reference 전처리: {reference_description}")

    status = edge_record.get("last_status") or {}
    endpoint = args.endpoint or status.get("zenoh_endpoint")
    topic_root = args.topic_root or status.get("topic_root") or DEFAULT_TOPIC_ROOT
    print(f"\n[{edge_id}] 전체 카메라 capture 및 DINO token 생성을 요청합니다.")
    transfer = request_calibration_features(
        edge_id,
        endpoint=endpoint,
        zenoh_config=args.zenoh_config,
        topic_root=topic_root,
        timeout=args.transfer_timeout,
    )
    bundle_path = output_dir / f"{edge_id}_{transfer.request_id}_features.npz"
    bundle_path.write_bytes(transfer.payload)
    print(f"Feature bundle 수신 완료: {bundle_path}")

    args.config = str(manifest_path)
    args.feature_bundle = str(bundle_path)
    args.checkpoint = str(checkpoint)
    args.output_dir = str(output_dir)
    args.marker_tree = str(marker_tree)
    print("참조 영상 token과 edge token을 결합해 VGGT-Omega 추론을 시작합니다.")
    result_path = run_calibration(args)
    result = json.loads(result_path.read_text(encoding="utf-8"))

    completed_at = time.time()
    registry.record_calibration_result(
        edge_id,
        result,
        result_path=str(result_path),
        completed_at=completed_at,
        edge_sync_status="pending",
    )
    try:
        publish_calibration_result(
            edge_id,
            result,
            endpoint=endpoint,
            zenoh_config=args.zenoh_config,
            topic_root=topic_root,
            timeout=args.result_timeout,
            completed_at=completed_at,
        )
    except Exception as exc:
        registry.record_calibration_sync(
            edge_id,
            transfer.request_id,
            success=False,
            error=str(exc),
        )
        raise RuntimeError(
            f"서버 결과는 저장했지만 edge 로컬 반영에 실패했습니다: {exc}"
        ) from exc
    registry.record_calibration_sync(
        edge_id,
        transfer.request_id,
        success=True,
    )
    print(f"서버 registry 및 {edge_id} 로컬 카메라 설정 저장 완료")
    return result_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate CCTV extrinsics using VGGT-Omega and an ArUco marker tree."
    )
    parser.add_argument(
        "--config",
        help="Calibration input JSON manifest; omit to use the interactive flow",
    )
    parser.add_argument(
        "--feature-bundle",
        help="Edge-produced DINO patch token .npz; config then only needs reference_video",
    )
    parser.add_argument("--checkpoint", help="VGGT-Omega .pt checkpoint")
    parser.add_argument("--output-dir", help="Directory for artifacts/results")
    parser.add_argument(
        "--marker-tree",
        help="ArUco marker-tree USD path",
    )
    parser.add_argument(
        "--registry",
        default=str(DEFAULT_EDGE_REGISTRY),
        help="Edge manager edges.json path for interactive mode",
    )
    parser.add_argument(
        "--mode",
        choices=("online", "offline"),
        help="Interactive flow mode; omit to select at startup",
    )
    parser.add_argument("--edge-id", help="Skip interactive edge selection")
    parser.add_argument(
        "--images",
        help="Offline CCTV photo or directory; skips the offline image prompt",
    )
    parser.add_argument(
        "--camera-config",
        help=(
            "Offline edge cameras.local.yaml; camera_<number>_* images use its "
            "intrinsic/distortion when no image sidecar exists"
        ),
    )
    parser.add_argument("--reference-video", help="Skip reference-video prompt")
    parser.add_argument(
        "--json-output",
        help="Offline final JSON path; defaults to ./calibration_result_<time>.json",
    )
    parser.add_argument("--endpoint", default=os.getenv("ZENOH_ENDPOINT"))
    parser.add_argument("--zenoh-config")
    parser.add_argument("--topic-root", default=os.getenv("EDGE_TOPIC_ROOT"))
    parser.add_argument("--transfer-timeout", type=float, default=300.0)
    parser.add_argument("--result-timeout", type=float, default=30.0)
    parser.add_argument("--reference-sample-count", type=int, default=40)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Total CCTV + reference images; default estimates from free GPU memory",
    )
    parser.add_argument("--device", default="cuda", help="CUDA device, e.g. cuda:0")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument(
        "--resize-mode", choices=("balanced", "max_size"), default="balanced"
    )
    parser.add_argument(
        "--undistort-alpha",
        type=float,
        default=0.0,
        help="OpenCV undistortion alpha: 0 crops invalid borders, 1 keeps all pixels",
    )
    parser.add_argument("--min-depth-confidence", type=float, default=1.0)
    parser.add_argument("--alignment-threshold-m", type=float, default=0.10)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        if args.config is None:
            if args.reference_sample_count <= 0:
                raise ValueError("reference sample count must be greater than zero")
            mode = select_cli_mode(args.mode)
            if mode == "online":
                if args.transfer_timeout <= 0 or args.result_timeout <= 0:
                    raise ValueError("Zenoh timeout must be greater than zero")
                result_path = run_interactive_calibration(args)
            else:
                result_path = run_offline_calibration(args)
        else:
            if not args.checkpoint:
                raise ValueError("--checkpoint is required in manifest mode")
            if not args.output_dir:
                raise ValueError("--output-dir is required in manifest mode")
            if args.marker_tree is None:
                args.marker_tree = str(DEFAULT_MARKER_TREE)
            result_path = run_calibration(args)
    except Exception as exc:
        LOGGER.error("Calibration failed: %s", exc)
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.exception("Calibration traceback")
        return 1
    print(result_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
