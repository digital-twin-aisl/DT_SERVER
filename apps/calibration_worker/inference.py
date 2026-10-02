# SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
# SPDX-License-Identifier: LGPL-2.1-or-later
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
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
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
from apps.calibration_worker.domain.alignment import (  # noqa: E402,F401 - legacy exports
    SimilarityTransform, fit_similarity_transform,
    robust_similarity_transform, transform_camera_extrinsic,
)
from apps.calibration_worker.domain.point_cloud import export_depth_cloud, preserve_offline_cloud


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
BA_MAX_FEATURES_PER_IMAGE = 2_048
BA_MAX_MATCHES_PER_PAIR = 256
BA_DEFAULT_MAX_TRACKS = 300
BA_DEFAULT_MAX_IMAGES = 32
DEFAULT_REFERENCE_SAMPLE_COUNT = 130
BA_MAX_INITIAL_REPROJECTION_ERROR_PX = 32.0
BA_MAX_ITERATIONS = 300
BA_MIN_CCTV_OBSERVATIONS = 20
BA_MIN_REFERENCE_OBSERVATIONS = 8


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


@dataclass(frozen=True)
class _BundleTrack:
    """A matched image track and its VGGT-depth-derived initial 3D point."""

    observations: tuple[tuple[int, np.ndarray], ...]
    initial_xyz: np.ndarray


@dataclass(frozen=True)
class BundleAdjustmentResult:
    extrinsics: np.ndarray
    camera_count: int
    selected_image_count: int
    track_count: int
    observation_count: int
    mean_track_length: float
    multi_view_track_count: int
    cctv_observation_counts: tuple[int, ...]
    fixed_cctv_indices: tuple[int, ...]
    initial_rmse_px: float
    final_rmse_px: float
    solver_nfev: int
    solver_optimality: float
    solver_message: str
    solver_converged: bool
    variable_image_indices: tuple[int, ...]
    fixed_image_indices: tuple[int, ...]
    unobserved_image_indices: tuple[int, ...]
    image_budget: int
    track_budget: int
    iteration_budget: int


@dataclass(frozen=True)
class ColmapReconstructionResult:
    """Sparse COLMAP poses keyed by the prepared input-image index."""

    extrinsics: dict[int, np.ndarray]
    registered_image_count: int
    point_count: int
    match_pair_count: int
    sparse_model_dir: Path


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
    43.15 GiB for 500 frames. Linear interpolation is only a planning estimate.
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
    reference_sample_count: int,
    *,
    undistort_alpha: float = 0.0,
) -> PreparedInputs:
    cctv_count = len(calibration_input.cctv_cameras)
    if reference_sample_count <= 0:
        raise ValueError("reference sample count must be greater than zero")

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

    sampled_frames = sample_video_frames(
        calibration_input.reference_video.video_path, reference_sample_count
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


def build_vggt_preprocessed_intrinsics(
    image_paths: Sequence[Path],
    camera_matrices: Sequence[np.ndarray],
    *,
    image_resolution: int,
    resize_mode: str,
    patch_size: int = 16,
    return_valid_regions: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Map calibrated intrinsics into VGGT's crop/resize/padding coordinates."""
    if len(image_paths) != len(camera_matrices):
        raise ValueError("image paths and camera matrices must have the same length")
    if resize_mode not in {"balanced", "max_size"}:
        raise ValueError(f"Unsupported VGGT resize mode: {resize_mode}")

    records: list[tuple[np.ndarray, int, int]] = []
    target_shapes: list[tuple[int, int]] = []
    for image_path, camera_matrix in zip(image_paths, camera_matrices, strict=True):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read prepared image: {image_path}")
        height, width = image.shape[:2]
        crop_left = 0
        crop_top = 0
        cropped_width = width
        cropped_height = height
        aspect_ratio = height / max(width, 1)
        if aspect_ratio < 0.5:
            cropped_width = min(width, max(1, int(round(height / 0.5))))
            crop_left = max((width - cropped_width) // 2, 0)
        elif aspect_ratio > 2.0:
            cropped_height = min(height, max(1, int(round(width * 2.0))))
            crop_top = max((height - cropped_height) // 2, 0)

        aspect_ratio = cropped_height / max(cropped_width, 1)
        if resize_mode == "balanced":
            token_count = (image_resolution // patch_size) ** 2
            unrounded_width_patches = np.sqrt(token_count / aspect_ratio)
            width_patches = max(
                1, int(np.round(unrounded_width_patches))
            )
            height_patches = max(
                1, int(np.round(token_count / unrounded_width_patches))
            )
            target_width = width_patches * patch_size
            target_height = height_patches * patch_size
        else:
            if aspect_ratio >= 1.0:
                target_height = image_resolution
                target_width = max(
                    patch_size,
                    int(np.round((image_resolution / aspect_ratio) / patch_size))
                    * patch_size,
                )
            else:
                target_width = image_resolution
                target_height = max(
                    patch_size,
                    int(np.round((image_resolution * aspect_ratio) / patch_size))
                    * patch_size,
                )

        matrix = _matrix3(camera_matrix, f"prepared intrinsic {image_path}").copy()
        matrix[0, 2] -= crop_left
        matrix[1, 2] -= crop_top
        matrix[0, :] *= target_width / cropped_width
        matrix[1, :] *= target_height / cropped_height
        matrix[2, :] = (0.0, 0.0, 1.0)
        records.append((matrix, target_height, target_width))
        target_shapes.append((target_height, target_width))

    common_height = max(height for height, _ in target_shapes)
    common_width = max(width for _, width in target_shapes)
    adjusted: list[np.ndarray] = []
    valid_regions = []
    for matrix, target_height, target_width in records:
        matrix = matrix.copy()
        matrix[0, 2] += (common_width - target_width) // 2
        matrix[1, 2] += (common_height - target_height) // 2
        adjusted.append(matrix)
        left, top = (common_width-target_width)//2, (common_height-target_height)//2
        valid_regions.append([left, top, left+target_width, top+target_height])
    matrices = np.asarray(adjusted, dtype=np.float64)
    return (matrices, np.asarray(valid_regions, dtype=np.int32)) if return_valid_regions else matrices


def prepare_remote_inputs(
    calibration_input: CalibrationInput,
    metadata: dict[str, Any],
    output_dir: Path,
    reference_sample_count: int,
) -> RemotePreparedInputs:
    """Preprocess server-owned reference frames with the edge token contract."""
    cctv_count = len(calibration_input.cctv_cameras)
    if reference_sample_count <= 0:
        raise ValueError("reference sample count must be greater than zero")
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
    sampled = sample_video_frames(
        calibration_input.reference_video.video_path, reference_sample_count
    )
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
            "Pixar USD Python modules are required to read the marker tree "
            f"(pxr import failed: {exc})."
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


def _project_pixel(
    point_xyz: np.ndarray,
    world_to_camera: np.ndarray,
    camera_matrix: np.ndarray,
) -> np.ndarray | None:
    """Project a VGGT-world point, returning ``None`` when it is behind a camera."""
    camera_point = world_to_camera[:, :3] @ point_xyz + world_to_camera[:, 3]
    if not np.isfinite(camera_point).all() or camera_point[2] <= 1.0e-6:
        return None
    projected = camera_matrix @ camera_point
    return projected[:2] / projected[2]


class _DisjointSet:
    """Small union-find helper for turning pairwise matches into feature tracks."""

    def __init__(self) -> None:
        self._parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(self, item: tuple[int, int]) -> tuple[int, int]:
        parent = self._parent.setdefault(item, item)
        if parent != item:
            parent = self.find(parent)
            self._parent[item] = parent
        return parent

    def union(self, left: tuple[int, int], right: tuple[int, int]) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self._parent[right_root] = left_root

    def items(self) -> Iterable[tuple[int, int]]:
        return self._parent


def _ba_candidate_pairs(image_count: int, cctv_count: int) -> list[tuple[int, int]]:
    """Select useful, bounded matching pairs for local BA.

    Every CCTV still is matched against every reference frame, while reference
    frames are matched to their two temporal neighbours. This preserves links
    from each fixed camera to the moving reference camera without the quadratic
    all-pairs cost for a long reference video.
    """
    pairs: set[tuple[int, int]] = set()
    reference_start = cctv_count
    for cctv_index in range(cctv_count):
        for reference_index in range(reference_start, image_count):
            pairs.add((cctv_index, reference_index))
    for image_index in range(reference_start, image_count):
        for offset in (1, 2):
            other_index = image_index + offset
            if other_index < image_count:
                pairs.add((image_index, other_index))
    for left_index in range(cctv_count):
        for right_index in range(left_index + 1, cctv_count):
            pairs.add((left_index, right_index))
    return sorted(pairs)


def _mutual_sift_matches(
    left_descriptors: np.ndarray,
    right_descriptors: np.ndarray,
) -> list[cv2.DMatch]:
    """Return ratio-tested mutual SIFT matches, capped for BA tractability."""
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    left_knn = matcher.knnMatch(left_descriptors, right_descriptors, k=2)
    right_knn = matcher.knnMatch(right_descriptors, left_descriptors, k=2)

    def accepted(matches: list[list[cv2.DMatch]]) -> dict[int, cv2.DMatch]:
        return {
            pair[0].queryIdx: pair[0]
            for pair in matches
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
        }

    forward = accepted(left_knn)
    reverse = accepted(right_knn)
    matches = [
        match
        for query_index, match in forward.items()
        if (backward := reverse.get(match.trainIdx)) is not None
        and backward.trainIdx == query_index
    ]
    return sorted(matches, key=lambda match: match.distance)[:BA_MAX_MATCHES_PER_PAIR]


def _build_bundle_tracks(
    images: np.ndarray,
    depths: np.ndarray,
    depth_confidences: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    *,
    cctv_count: int,
    min_depth_confidence: float,
    max_tracks: int,
) -> list[_BundleTrack]:
    """Build repeatable SIFT tracks and initialize their points from VGGT depth."""
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("--use_ba requires OpenCV built with SIFT support.")
    sift = cv2.SIFT_create(nfeatures=BA_MAX_FEATURES_PER_IMAGE)
    keypoints_by_image: list[list[cv2.KeyPoint]] = []
    descriptors_by_image: list[np.ndarray | None] = []
    for image in images:
        rgb = _rgb_to_uint8(image)
        grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = sift.detectAndCompute(grayscale, None)
        keypoints_by_image.append(keypoints)
        descriptors_by_image.append(descriptors)

    matches = _DisjointSet()
    pair_count = 0
    for left_index, right_index in _ba_candidate_pairs(len(images), cctv_count):
        left_descriptors = descriptors_by_image[left_index]
        right_descriptors = descriptors_by_image[right_index]
        if left_descriptors is None or right_descriptors is None:
            continue
        pair_matches = _mutual_sift_matches(left_descriptors, right_descriptors)
        if len(pair_matches) >= 8:
            left_points = np.asarray(
                [keypoints_by_image[left_index][match.queryIdx].pt for match in pair_matches],
                dtype=np.float32,
            )
            right_points = np.asarray(
                [keypoints_by_image[right_index][match.trainIdx].pt for match in pair_matches],
                dtype=np.float32,
            )
            _, inlier_mask = cv2.findFundamentalMat(
                left_points,
                right_points,
                cv2.FM_RANSAC,
                2.0,
                0.999,
            )
            if inlier_mask is not None and int(inlier_mask.sum()) >= 8:
                pair_matches = [
                    match
                    for match, is_inlier in zip(
                        pair_matches, inlier_mask.reshape(-1), strict=True
                    )
                    if is_inlier
                ]
        if pair_matches:
            pair_count += 1
        for match in pair_matches:
            matches.union(
                (left_index, match.queryIdx), (right_index, match.trainIdx)
            )
    if not pair_count:
        raise RuntimeError(
            "--use_ba found no repeatable image features. Use images with "
            "overlapping scene content or run without --use_ba."
        )

    grouped: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for observation in matches.items():
        grouped.setdefault(matches.find(observation), []).append(observation)

    tracks: list[tuple[tuple[int, float], _BundleTrack]] = []
    for observations in grouped.values():
        # A valid feature track has exactly one keypoint per image and at least
        # two usable depth observations.
        image_indices = [image_index for image_index, _ in observations]
        if len(observations) < 2 or len(set(image_indices)) != len(observations):
            continue
        usable: list[tuple[int, np.ndarray, np.ndarray]] = []
        for image_index, keypoint_index in observations:
            xy = np.asarray(keypoints_by_image[image_index][keypoint_index].pt)
            sampled_depth = _depth_at_corner(
                depths[image_index],
                depth_confidences[image_index],
                xy,
                min_depth_confidence,
            )
            if sampled_depth is None:
                continue
            usable.append(
                (
                    image_index,
                    xy,
                    unproject_pixel(
                        xy,
                        sampled_depth,
                        extrinsics[image_index],
                        intrinsics[image_index],
                    ),
                )
            )
        if len(usable) < 2:
            continue
        initial_xyz = np.median(
            np.asarray([point for _, _, point in usable]), axis=0
        )
        errors: list[float] = []
        for image_index, xy, _ in usable:
            projected = _project_pixel(
                initial_xyz, extrinsics[image_index], intrinsics[image_index]
            )
            if projected is None:
                errors.append(float("inf"))
            else:
                errors.append(float(np.linalg.norm(projected - xy)))
        # A track with one observation behind its initial camera can still have
        # a small median error. Reject it entirely: retaining that observation
        # injects a 1,000 px penalty into BA and lets a false match dominate.
        max_error = float(np.max(errors))
        if not math.isfinite(max_error) or max_error > BA_MAX_INITIAL_REPROJECTION_ERROR_PX:
            continue
        tracks.append(
            (
                (-len(usable), float(np.median(errors))),
                _BundleTrack(
                    observations=tuple((index, xy) for index, xy, _ in usable),
                    initial_xyz=initial_xyz,
                ),
            )
        )
    tracks.sort(key=lambda item: item[0])
    ranked_tracks = [track for _, track in tracks]
    selected_indices: set[int] = set()
    # Reserve half of the track budget for balanced CCTV connectivity. Without
    # this quota, long reference-only tracks can occupy the whole budget while
    # one or more fixed CCTV images remain weakly constrained.
    per_cctv_quota = max(8, max_tracks // max(2 * cctv_count, 1))
    for cctv_index in range(cctv_count):
        candidates = [
            index
            for index, track in enumerate(ranked_tracks)
            if any(image_index == cctv_index for image_index, _ in track.observations)
        ]
        for index in candidates[:per_cctv_quota]:
            if len(selected_indices) >= max_tracks:
                break
            selected_indices.add(index)

    cctv_track_target = min(max_tracks, int(math.ceil(0.75 * max_tracks)))
    for index, track in enumerate(ranked_tracks):
        if len(selected_indices) >= cctv_track_target:
            break
        if any(image_index < cctv_count for image_index, _ in track.observations):
            selected_indices.add(index)
    for index in range(len(ranked_tracks)):
        if len(selected_indices) >= max_tracks:
            break
        selected_indices.add(index)
    return [ranked_tracks[index] for index in sorted(selected_indices)]


def _ba_selected_image_indices(
    image_count: int,
    cctv_count: int,
    max_images: int,
) -> np.ndarray:
    """Keep every CCTV image and evenly sample reference keyframes."""
    if not cctv_count or cctv_count >= image_count:
        raise ValueError("BA requires at least one CCTV and one reference image")
    if max_images <= cctv_count:
        raise ValueError("Internal BA image budget must leave room for reference keyframes")
    if image_count <= max_images:
        return np.arange(image_count, dtype=np.int64)
    reference_count = min(image_count - cctv_count, max_images - cctv_count)
    selected: set[int] = set()
    evenly_spaced = np.linspace(
        cctv_count, image_count - 1, reference_count, dtype=np.int64
    )
    for index in evenly_spaced:
        if len(selected) >= reference_count:
            break
        selected.add(int(index))
    if len(selected) < reference_count:
        for index in range(cctv_count, image_count):
            if len(selected) >= reference_count:
                break
            selected.add(index)
    reference_indices = np.asarray(sorted(selected), dtype=np.int64)
    return np.concatenate((np.arange(cctv_count, dtype=np.int64), reference_indices))


def bundle_adjustment_limits(image_count, cctv_count, *, max_images=None,
                             max_tracks=None, max_iterations=BA_MAX_ITERATIONS):
    """Default to all input views; scale the feature budget with selected views."""
    image_budget = image_count if max_images is None else min(max_images, image_count)
    if image_budget <= cctv_count or cctv_count < 1:
        raise ValueError('BA image budget must include every CCTV and at least one reference.')
    track_budget = (max(BA_DEFAULT_MAX_TRACKS, math.ceil(BA_DEFAULT_MAX_TRACKS * image_budget / BA_DEFAULT_MAX_IMAGES))
                    if max_tracks is None else max_tracks)
    if track_budget < 8 or max_iterations < 1:
        raise ValueError('BA requires at least 8 tracks and a positive iteration budget.')
    return dict(max_images=image_budget, max_tracks=track_budget, max_iterations=max_iterations)


def run_bundle_adjustment(
    predictions: dict[str, np.ndarray],
    *,
    cctv_count: int,
    min_depth_confidence: float,
    max_images: int | None = None,
    max_tracks: int | None = None,
    max_iterations: int = BA_MAX_ITERATIONS,
) -> BundleAdjustmentResult:
    """Refine VGGT poses and depth-initialized scene points from image tracks.

    Camera intrinsics stay fixed while matched 2D features jointly refine
    world-to-camera extrinsics and 3D track points. ArUco/USD geometry is
    deliberately excluded and is used only for post-BA coordinate alignment.
    """
    try:
        from scipy.optimize import least_squares
        from scipy.sparse import lil_matrix
    except ImportError as exc:
        raise RuntimeError("--use_ba requires scipy. Install requirements.txt.") from exc

    source_extrinsics = np.asarray(predictions["extrinsics"], dtype=np.float64).copy()
    source_intrinsics = np.asarray(predictions["intrinsics"], dtype=np.float64)
    intrinsics = np.asarray(
        predictions.get("ba_intrinsics", predictions["intrinsics"]),
        dtype=np.float64,
    )
    limits = bundle_adjustment_limits(len(source_extrinsics), cctv_count,
        max_images=max_images, max_tracks=max_tracks, max_iterations=max_iterations)
    max_images, max_tracks = limits['max_images'], limits['max_tracks']
    selected_indices = _ba_selected_image_indices(
        len(source_extrinsics),
        cctv_count,
        max_images,
    )
    LOGGER.info(
        "BA uses %d/%d images (%d CCTV + %d reference keyframes), up to %d tracks",
        len(selected_indices),
        len(source_extrinsics),
        cctv_count,
        len(selected_indices) - cctv_count,
        max_tracks,
    )
    extrinsics = source_extrinsics.copy()
    tracks = _build_bundle_tracks(
        predictions["images"][selected_indices],
        predictions["depth"][selected_indices],
        predictions["depth_conf"][selected_indices],
        source_extrinsics[selected_indices],
        source_intrinsics[selected_indices],
        cctv_count=cctv_count,
        min_depth_confidence=min_depth_confidence,
        max_tracks=max_tracks,
    )
    tracks = [
        _BundleTrack(
            observations=tuple(
                (int(selected_indices[image_index]), xy)
                for image_index, xy in track.observations
            ),
            initial_xyz=track.initial_xyz,
        )
        for track in tracks
    ]
    active_images = sorted(
        {
            image_index
            for track in tracks
            for image_index, _ in track.observations
        }
    )
    observation_count = sum(len(track.observations) for track in tracks)
    if len(active_images) < 3 or len(tracks) < 8:
        raise RuntimeError(
            "--use_ba needs at least three overlapping images and eight reliable "
            "depth-initialized feature tracks."
        )

    observation_counts = {
        image_index: sum(
            image_index in {index for index, _ in track.observations}
            for track in tracks
        )
        for image_index in active_images
    }
    # Feature-only BA has an arbitrary global frame, so retain the two strongest
    # poses as gauge anchors. Weakly connected poses also remain at their VGGT
    # initialization instead of being moved by insufficient track evidence.
    anchor_images = sorted(
        active_images,
        key=lambda image_index: observation_counts[image_index],
        reverse=True,
    )[:2]
    weak_images = [
        image_index
        for image_index in selected_indices
        if (
            image_index < cctv_count
            and observation_counts.get(image_index, 0) < BA_MIN_CCTV_OBSERVATIONS
        )
        or (
            image_index >= cctv_count
            and observation_counts.get(image_index, 0) < BA_MIN_REFERENCE_OBSERVATIONS
        )
    ]
    fixed_images = sorted(set(anchor_images) | set(weak_images))
    variable_images = [
        image_index for image_index in active_images if image_index not in fixed_images
    ]
    variable_index = {
        image_index: offset for offset, image_index in enumerate(variable_images)
    }
    point_offset = 6 * len(variable_images)
    point_end = point_offset + 3 * len(tracks)

    initial_parameters: list[float] = []
    for image_index in variable_images:
        rotation_vector, _ = cv2.Rodrigues(extrinsics[image_index, :, :3])
        initial_parameters.extend(rotation_vector.reshape(-1).tolist())
        initial_parameters.extend(extrinsics[image_index, :, 3].tolist())
    initial_parameters.extend(
        coordinate for track in tracks for coordinate in track.initial_xyz.tolist()
    )
    initial_parameters_array = np.asarray(initial_parameters, dtype=np.float64)

    def unpack(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        current_extrinsics = extrinsics.copy()
        for image_index, offset in variable_index.items():
            start = 6 * offset
            rotation, _ = cv2.Rodrigues(parameters[start : start + 3])
            current_extrinsics[image_index, :, :3] = rotation
            current_extrinsics[image_index, :, 3] = parameters[start + 3 : start + 6]
        points = parameters[point_offset:point_end].reshape(len(tracks), 3)
        return current_extrinsics, points

    def residual(parameters: np.ndarray) -> np.ndarray:
        current_extrinsics, points = unpack(parameters)
        values: list[float] = []
        for point, track in zip(points, tracks, strict=True):
            for image_index, xy in track.observations:
                projected = _project_pixel(
                    point, current_extrinsics[image_index], intrinsics[image_index]
                )
                if projected is None:
                    values.extend((1_000.0, 1_000.0))
                else:
                    values.extend((projected - xy).tolist())
        return np.asarray(values, dtype=np.float64)

    initial_residuals = residual(initial_parameters_array)
    parameter_count = len(initial_parameters_array)
    if len(initial_residuals) <= parameter_count:
        raise RuntimeError(
            "--use_ba has too few feature observations for its camera and point variables."
        )
    jacobian_sparsity = lil_matrix(
        (len(initial_residuals), parameter_count), dtype=np.int8
    )
    row = 0
    for track_index, track in enumerate(tracks):
        point_columns = slice(point_offset + 3 * track_index, point_offset + 3 * track_index + 3)
        for image_index, _ in track.observations:
            jacobian_sparsity[row : row + 2, point_columns] = 1
            if image_index in variable_index:
                camera_offset = 6 * variable_index[image_index]
                jacobian_sparsity[row : row + 2, camera_offset : camera_offset + 6] = 1
            row += 2
    solution = least_squares(
        residual,
        initial_parameters_array,
        loss="huber",
        f_scale=4.0,
        max_nfev=max_iterations,
        method="trf",
        jac_sparsity=jacobian_sparsity.tocsr(),
        tr_solver="lsmr",
        x_scale="jac",
    )
    refined_extrinsics, _ = unpack(solution.x)
    final_residuals = residual(solution.x)
    if not solution.success:
        initial_squared_error = float(np.dot(initial_residuals, initial_residuals))
        final_squared_error = float(np.dot(final_residuals, final_residuals))
        usable_limited_solution = (
            solution.status == 0
            and np.isfinite(solution.x).all()
            and np.isfinite(final_squared_error)
            and final_squared_error < 0.995 * initial_squared_error
        )
        if not usable_limited_solution:
            raise RuntimeError(f"--use_ba did not converge: {solution.message}")
        LOGGER.warning(
            "BA reached its iteration limit but retained a finite improving solution "
            "(squared error %.3g -> %.3g)",
            initial_squared_error,
            final_squared_error,
        )
    return BundleAdjustmentResult(
        extrinsics=refined_extrinsics,
        camera_count=len(active_images),
        selected_image_count=len(selected_indices),
        track_count=len(tracks),
        observation_count=observation_count,
        mean_track_length=float(observation_count / len(tracks)),
        multi_view_track_count=sum(
            len(track.observations) >= 3 for track in tracks
        ),
        cctv_observation_counts=tuple(
            sum(
                image_index == cctv_index
                for track in tracks
                for image_index, _ in track.observations
            )
            for cctv_index in range(cctv_count)
        ),
        fixed_cctv_indices=tuple(
            image_index
            for image_index in weak_images
            if image_index < cctv_count
        ),
        initial_rmse_px=float(np.sqrt(np.mean(initial_residuals**2))),
        final_rmse_px=float(np.sqrt(np.mean(final_residuals**2))),
        solver_nfev=int(solution.nfev),
        solver_optimality=float(solution.optimality),
        solver_message=str(solution.message),
        solver_converged=bool(solution.success),
        variable_image_indices=tuple(int(i) for i in variable_images),
        fixed_image_indices=tuple(int(i) for i in fixed_images),
        unobserved_image_indices=tuple(int(i) for i in selected_indices if i not in active_images),
        image_budget=max_images, track_budget=max_tracks, iteration_budget=max_iterations,
    )


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


def _camera_center(world_to_camera: np.ndarray) -> np.ndarray:
    rotation = np.asarray(world_to_camera, dtype=np.float64)[:, :3]
    translation = np.asarray(world_to_camera, dtype=np.float64)[:, 3]
    return -rotation.T @ translation


def _mean_rotation(rotations: Sequence[np.ndarray]) -> np.ndarray:
    if not rotations:
        raise ValueError("At least one rotation is required")
    left, _, right_transpose = np.linalg.svd(np.sum(rotations, axis=0))
    rotation = left @ right_transpose
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right_transpose
    return rotation


def fit_similarity_from_camera_poses(
    colmap_extrinsics: Sequence[np.ndarray],
    usd_extrinsics: Sequence[np.ndarray],
    *,
    inlier_threshold_m: float,
) -> SimilarityTransform:
    """Fit COLMAP-world to USD-world similarity from matched camera poses.

    Camera rotations determine the global orientation even for a reference video
    moving mostly along one corridor axis; camera centres then determine scale
    and translation. This avoids the VGGT-depth dependency in the COLMAP path.
    """
    if len(colmap_extrinsics) != len(usd_extrinsics) or len(colmap_extrinsics) < 2:
        raise ValueError("At least two matched COLMAP and USD camera poses are required")
    if inlier_threshold_m <= 0:
        raise ValueError("inlier_threshold_m must be positive")
    colmap = np.asarray(colmap_extrinsics, dtype=np.float64)
    usd = np.asarray(usd_extrinsics, dtype=np.float64)
    if colmap.shape[1:] != (3, 4) or usd.shape[1:] != (3, 4):
        raise ValueError("Camera extrinsics must have shape Nx3x4")
    rotation_candidates = [
        usd_pose[:, :3].T @ colmap_pose[:, :3]
        for colmap_pose, usd_pose in zip(colmap, usd, strict=True)
    ]
    source_centers = np.asarray([_camera_center(pose) for pose in colmap])
    target_centers = np.asarray([_camera_center(pose) for pose in usd])
    inlier_mask = np.ones(len(colmap), dtype=bool)
    for _ in range(3):
        rotation = _mean_rotation(
            [candidate for candidate, keep in zip(rotation_candidates, inlier_mask, strict=True) if keep]
        )
        rotated_source = source_centers @ rotation.T
        source_mean = rotated_source[inlier_mask].mean(axis=0)
        target_mean = target_centers[inlier_mask].mean(axis=0)
        source_centered = rotated_source[inlier_mask] - source_mean
        target_centered = target_centers[inlier_mask] - target_mean
        denominator = float(np.sum(source_centered**2))
        if denominator <= 1.0e-12:
            raise ValueError("COLMAP reference camera centres are degenerate")
        scale = float(np.sum(source_centered * target_centered) / denominator)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("COLMAP-to-USD pose alignment has a non-positive scale")
        translation = target_mean - scale * source_mean
        residuals = np.linalg.norm(
            scale * rotated_source + translation - target_centers, axis=1
        )
        new_mask = residuals <= inlier_threshold_m
        if new_mask.sum() < 2:
            break
        if np.array_equal(new_mask, inlier_mask):
            inlier_mask = new_mask
            break
        inlier_mask = new_mask
    if inlier_mask.sum() < 2:
        raise RuntimeError(
            "COLMAP-to-USD alignment found fewer than two consistent ArUco PnP poses"
        )
    rotation = _mean_rotation(
        [candidate for candidate, keep in zip(rotation_candidates, inlier_mask, strict=True) if keep]
    )
    rotated_source = source_centers @ rotation.T
    source_mean = rotated_source[inlier_mask].mean(axis=0)
    target_mean = target_centers[inlier_mask].mean(axis=0)
    scale = float(
        np.sum(
            (rotated_source[inlier_mask] - source_mean)
            * (target_centers[inlier_mask] - target_mean)
        )
        / np.sum((rotated_source[inlier_mask] - source_mean) ** 2)
    )
    translation = target_mean - scale * source_mean
    residuals = np.linalg.norm(scale * rotated_source + translation - target_centers, axis=1)
    return SimilarityTransform(scale, rotation, translation, inlier_mask, residuals)


def collect_colmap_pnp_alignment(
    prepared: PreparedInputs,
    colmap_extrinsics: dict[int, np.ndarray],
    marker_definitions: dict[tuple[str, int], MarkerDefinition],
    *,
    inlier_threshold_m: float,
) -> tuple[SimilarityTransform, list[dict[str, Any]], int]:
    """Anchor COLMAP's arbitrary world to the USD marker-tree via ArUco PnP."""
    colmap_poses: list[np.ndarray] = []
    usd_poses: list[np.ndarray] = []
    observations: list[dict[str, Any]] = []
    for image_index in range(prepared.cctv_count, len(prepared.image_paths)):
        if image_index not in colmap_extrinsics:
            continue
        image = cv2.imread(str(prepared.image_paths[image_index]), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read reference image for ArUco PnP: {prepared.image_paths[image_index]}")
        detections = detect_known_markers(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB), marker_definitions
        )
        object_points: list[np.ndarray] = []
        image_points: list[np.ndarray] = []
        marker_ids: list[int] = []
        for definition, corners in detections:
            object_points.extend(definition.world_corners)
            image_points.extend(corners)
            marker_ids.append(definition.marker_id)
        if len(object_points) < 4:
            continue
        success, rvec, translation, inliers = cv2.solvePnPRansac(
            np.asarray(object_points, dtype=np.float64),
            np.asarray(image_points, dtype=np.float64),
            prepared.reference_new_camera_matrix,
            None,
            flags=cv2.SOLVEPNP_EPNP,
            reprojectionError=3.0,
            confidence=0.999,
            iterationsCount=1_000,
        )
        if not success or inliers is None or len(inliers) < 4:
            continue
        inlier_indices = inliers.reshape(-1)
        rvec, translation = cv2.solvePnPRefineLM(
            np.asarray(object_points, dtype=np.float64)[inlier_indices],
            np.asarray(image_points, dtype=np.float64)[inlier_indices],
            prepared.reference_new_camera_matrix,
            None,
            rvec,
            translation,
        )
        rotation, _ = cv2.Rodrigues(rvec)
        usd_pose = np.column_stack((rotation, translation.reshape(3)))
        colmap_poses.append(colmap_extrinsics[image_index])
        usd_poses.append(usd_pose)
        observations.append(
            {
                "image_index": image_index,
                "reference_video_frame_index": prepared.reference_frame_indices[
                    image_index - prepared.cctv_count
                ],
                "marker_ids": sorted(set(marker_ids)),
                "corner_count": len(object_points),
                "pnp_inlier_count": len(inlier_indices),
            }
        )
    alignment = fit_similarity_from_camera_poses(
        colmap_poses,
        usd_poses,
        inlier_threshold_m=inlier_threshold_m,
    )
    return alignment, observations, len(colmap_poses)








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
            "VGGT-Omega ran out of GPU memory; retry with a smaller "
            "--reference-sample-count."
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


def _run_colmap(command: Sequence[str], *, cwd: Path) -> None:
    """Run one COLMAP CLI stage and surface its useful error output."""
    LOGGER.info("Running COLMAP: %s", " ".join(command))
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode:
        output = completed.stdout.strip()
        tail = output[-4_000:] if output else "no COLMAP output"
        raise RuntimeError(f"COLMAP command failed ({command[1]}):\n{tail}")
    if completed.stdout:
        LOGGER.debug("COLMAP %s output:\n%s", command[1], completed.stdout)


def _colmap_rotation_matrix(qvec: Sequence[float]) -> np.ndarray:
    """Convert COLMAP's [qw, qx, qy, qz] world-to-camera quaternion to R."""
    quaternion = np.asarray(qvec, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("COLMAP quaternion must contain four finite values")
    quaternion /= np.linalg.norm(quaternion)
    qw, qx, qy, qz = quaternion
    return np.asarray(
        (
            (1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)),
            (2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)),
            (2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)),
        ),
        dtype=np.float64,
    )


def _colmap_image_extrinsics(images_txt: Path) -> dict[str, np.ndarray]:
    """Read registered camera poses from COLMAP's human-readable images.txt."""
    poses: dict[str, np.ndarray] = {}
    for line in images_txt.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#") or len(fields) < 10:
            continue
        try:
            int(fields[0])
            qvec = [float(value) for value in fields[1:5]]
            translation = [float(value) for value in fields[5:8]]
            int(fields[8])
        except ValueError:
            continue
        extrinsic = np.empty((3, 4), dtype=np.float64)
        extrinsic[:, :3] = _colmap_rotation_matrix(qvec)
        extrinsic[:, 3] = translation
        poses[fields[9]] = extrinsic
    return poses


def _prepare_colmap_database(
    image_paths: Sequence[Path],
    camera_matrices: Sequence[np.ndarray],
    workspace: Path,
) -> dict[str, int]:
    """Pre-register one fixed PINHOLE camera for every undistorted image."""
    if len(image_paths) != len(camera_matrices):
        raise ValueError("COLMAP image paths and camera matrices must have equal length")
    image_dir = workspace / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    database_path = workspace / "database.db"
    _run_colmap(("colmap", "database_creator", "--database_path", str(database_path)), cwd=workspace)

    staged_names: dict[str, int] = {}
    connection = sqlite3.connect(database_path)
    try:
        with connection:
            for index, (source, intrinsic) in enumerate(
                zip(image_paths, camera_matrices, strict=True)
            ):
                image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise RuntimeError(f"Could not read COLMAP input image: {source}")
                height, width = image.shape[:2]
                matrix = _matrix3(intrinsic, f"COLMAP camera matrix for {source.name}")
                if matrix[0, 2] < 0 or matrix[1, 2] < 0:
                    raise ValueError(f"COLMAP camera principal point is invalid: {source}")
                staged_name = f"{index:04d}_{_safe_name(source.name)}"
                staged_path = image_dir / staged_name
                try:
                    staged_path.symlink_to(source.resolve())
                except OSError:
                    shutil.copy2(source, staged_path)
                cursor = connection.execute(
                    "INSERT INTO cameras(model, width, height, params, prior_focal_length) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        1,  # COLMAP CameraModelId.PINHOLE
                        int(width),
                        int(height),
                        np.asarray(
                            (matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]),
                            dtype=np.float64,
                        ).tobytes(),
                        1,
                    ),
                )
                connection.execute(
                    "INSERT INTO images(name, camera_id) VALUES (?, ?)",
                    (staged_name, int(cursor.lastrowid)),
                )
                staged_names[staged_name] = index
    finally:
        connection.close()
    return staged_names


def _colmap_candidate_pairs(
    staged_names: dict[str, int],
    cctv_count: int,
    *,
    reference_window: int = 5,
) -> list[tuple[str, str]]:
    """Build match pairs suited to fixed CCTV images plus a reference video.

    Every CCTV is compared with every reference frame, while reference frames
    are compared only with nearby frames in time. This keeps the bridge from
    each fixed camera into the video reconstruction without spending most of
    the matching time on distant video frames that are unlikely to overlap.
    """
    if cctv_count < 0 or cctv_count > len(staged_names):
        raise ValueError("COLMAP CCTV count is outside the staged image range")
    if reference_window <= 0:
        raise ValueError("COLMAP reference match window must be positive")

    names_by_index = {index: name for name, index in staged_names.items()}
    if set(names_by_index) != set(range(len(staged_names))):
        raise ValueError("COLMAP staged image indices must be contiguous")

    pairs: set[tuple[int, int]] = set()
    image_count = len(staged_names)
    # Fixed CCTV views may overlap one another.
    for left in range(cctv_count):
        for right in range(left + 1, cctv_count):
            pairs.add((left, right))
    # A fixed CCTV image may match any point along the reference walk-through.
    for cctv_index in range(cctv_count):
        for reference_index in range(cctv_count, image_count):
            pairs.add((cctv_index, reference_index))
    # Nearby video frames supply the continuous SfM backbone.
    for left in range(cctv_count, image_count):
        stop = min(image_count, left + reference_window + 1)
        for right in range(left + 1, stop):
            pairs.add((left, right))

    return [
        (names_by_index[left], names_by_index[right])
        for left, right in sorted(pairs)
    ]


def run_colmap_reconstruction(
    prepared: PreparedInputs,
    output_dir: Path,
    *,
    max_image_size: int = 1600,
) -> ColmapReconstructionResult:
    """Run fixed-intrinsic sparse COLMAP SfM on prepared local images."""
    if shutil.which("colmap") is None:
        raise RuntimeError("COLMAP CLI is not installed or is not on PATH")
    if max_image_size <= 0:
        raise ValueError("COLMAP max image size must be positive")
    camera_matrices = [*prepared.cctv_new_camera_matrices]
    camera_matrices.extend(
        prepared.reference_new_camera_matrix
        for _ in prepared.reference_frame_indices
    )
    if len(camera_matrices) != len(prepared.image_paths):
        raise RuntimeError("Prepared input count does not match COLMAP camera matrices")

    workspace = output_dir / "colmap"
    workspace.mkdir(parents=True, exist_ok=True)
    staged_names = _prepare_colmap_database(
        prepared.image_paths, camera_matrices, workspace
    )
    database_path = workspace / "database.db"
    image_dir = workspace / "images"
    _run_colmap(
        (
            "colmap", "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(image_dir),
            "--SiftExtraction.use_gpu", "0",
            "--SiftExtraction.max_image_size", str(max_image_size),
        ),
        cwd=workspace,
    )
    match_pairs = _colmap_candidate_pairs(staged_names, prepared.cctv_count)
    match_list_path = workspace / "match_pairs.txt"
    match_list_path.write_text(
        "".join(f"{left} {right}\n" for left, right in match_pairs),
        encoding="utf-8",
    )
    LOGGER.info(
        "COLMAP matching %d selected image pair(s) instead of %d exhaustive pair(s)",
        len(match_pairs),
        len(staged_names) * (len(staged_names) - 1) // 2,
    )
    _run_colmap(
        (
            "colmap", "matches_importer",
            "--database_path", str(database_path),
            "--match_list_path", str(match_list_path),
            "--match_type", "pairs",
            "--SiftMatching.use_gpu", "0",
            "--TwoViewGeometry.min_num_inliers", "10",
            "--TwoViewGeometry.min_inlier_ratio", "0.10",
        ),
        cwd=workspace,
    )
    sparse_dir = workspace / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    _run_colmap(
        (
            "colmap", "mapper",
            "--database_path", str(database_path),
            "--image_path", str(image_dir),
            "--output_path", str(sparse_dir),
            "--Mapper.multiple_models", "0",
            "--Mapper.max_num_models", "1",
            "--Mapper.min_model_size", "3",
            "--Mapper.min_num_matches", "10",
            "--Mapper.abs_pose_min_num_inliers", "15",
            "--Mapper.abs_pose_min_inlier_ratio", "0.10",
            "--Mapper.max_reg_trials", "5",
            "--Mapper.ba_refine_focal_length", "0",
            "--Mapper.ba_refine_principal_point", "0",
            "--Mapper.ba_refine_extra_params", "0",
        ),
        cwd=workspace,
    )
    models = sorted(
        path for path in sparse_dir.iterdir()
        if path.is_dir() and (path / "cameras.bin").is_file()
    ) if sparse_dir.is_dir() else []
    if not models:
        raise RuntimeError(
            "COLMAP produced no sparse reconstruction. Ensure CCTV and reference "
            "images share enough static, textured scene content."
    )
    model_dir = models[0]
    text_dir = workspace / "sparse_text"
    text_dir.mkdir(parents=True, exist_ok=True)
    _run_colmap(
        (
            "colmap", "model_converter",
            "--input_path", str(model_dir),
            "--output_path", str(text_dir),
            "--output_type", "TXT",
        ),
        cwd=workspace,
    )
    poses_by_name = _colmap_image_extrinsics(text_dir / "images.txt")
    poses = {
        staged_names[name]: extrinsic
        for name, extrinsic in poses_by_name.items()
        if name in staged_names
    }
    if len(poses) < 3:
        raise RuntimeError("COLMAP registered fewer than three input images")
    point_count = sum(
        1
        for line in (text_dir / "points3D.txt").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    )
    return ColmapReconstructionResult(
        extrinsics=poses,
        registered_image_count=len(poses),
        point_count=point_count,
        match_pair_count=len(match_pairs),
        sparse_model_dir=model_dir,
    )


def _json_matrix(matrix: np.ndarray) -> list[list[float]]:
    return np.asarray(matrix, dtype=np.float64).tolist()


def run_calibration(args: argparse.Namespace) -> Path:
    if not 1 <= getattr(args, 'point_cloud_max_points', 1_000_000) <= 2_000_000:
        raise ValueError('--point-cloud-max-points must be between 1 and 2000000.')
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    backend = getattr(args, "backend", "vggt")
    if backend not in {"vggt", "colmap"}:
        raise ValueError(f"Unsupported reconstruction backend: {backend}")
    if backend == "colmap" and getattr(args, "use_ba", False):
        raise ValueError("--use_ba is only available with --backend vggt")
    checkpoint = (
        Path(args.checkpoint).expanduser().resolve()
        if backend == "vggt" and args.checkpoint
        else None
    )
    feature_metadata: dict[str, Any] | None = None
    if args.feature_bundle:
        if backend == "colmap":
            raise ValueError(
                "--backend colmap is unavailable with --feature-bundle because "
                "COLMAP needs source CCTV images, not patch tokens."
            )
        assert checkpoint is not None
        feature_payload = Path(args.feature_bundle).expanduser().resolve().read_bytes()
        feature_metadata, edge_tokens = decode_feature_bundle(feature_payload)
        if feature_metadata.get("preprocess") != PREPROCESS_NAME:
            raise ValueError("feature bundle preprocessing contract is unsupported")
        if feature_metadata.get("checkpoint_sha256") != file_sha256(checkpoint):
            raise ValueError("edge and server VGGT checkpoints do not match")
        calibration_input = load_remote_calibration_input(args.config, feature_metadata)
    else:
        calibration_input = load_calibration_input(args.config)

    requested_reference_count = getattr(args, "reference_sample_count", None)
    if requested_reference_count is None:
        requested_reference_count = (
            calibration_input.reference_video.sample_count
            or DEFAULT_REFERENCE_SAMPLE_COUNT
        )
    requested_reference_count = int(requested_reference_count)
    if requested_reference_count <= 0:
        raise ValueError("reference sample count must be greater than zero")
    requested_total_count = (
        len(calibration_input.cctv_cameras) + requested_reference_count
    )
    if getattr(args, 'use_ba', False):
        bundle_adjustment_limits(requested_total_count, len(calibration_input.cctv_cameras),
            max_images=getattr(args, 'ba_max_images', None), max_tracks=getattr(args, 'ba_max_tracks', None),
            max_iterations=getattr(args, 'ba_max_iterations', BA_MAX_ITERATIONS))
    maximum_images = getattr(args, 'max_images', None)
    if maximum_images is not None and (maximum_images <= 0 or requested_total_count > maximum_images):
        raise ValueError(f'Requested {requested_total_count} images exceed --max-images {maximum_images}; reduce --reference-sample-count.')
    if backend == "vggt":
        gpu_image_limit = estimate_max_images_from_gpu(args.device)
        LOGGER.info(
            "Requested %d total image(s); GPU estimate allows up to %d",
            requested_total_count,
            gpu_image_limit,
        )
        if requested_total_count > gpu_image_limit:
            maximum_reference_count = max(
                1, gpu_image_limit - len(calibration_input.cctv_cameras)
            )
            raise RuntimeError(
                f"{requested_total_count} total images exceed the estimated GPU "
                f"limit ({gpu_image_limit}). Reduce --reference-sample-count to "
                f"{maximum_reference_count} or less."
            )

    if feature_metadata is not None:
        prepared = prepare_remote_inputs(
            calibration_input,
            feature_metadata,
            output_dir,
            requested_reference_count,
        )
    else:
        prepared = prepare_inputs(
            calibration_input,
            output_dir,
            requested_reference_count,
            undistort_alpha=args.undistort_alpha,
        )
    LOGGER.info(
        "Prepared %d CCTV image(s) and %d reference frame(s)",
        prepared.cctv_count,
        len(prepared.reference_frame_indices),
    )

    marker_definitions = load_marker_tree(args.marker_tree)
    ba_result: BundleAdjustmentResult | None = None
    colmap_result: ColmapReconstructionResult | None = None
    missing_cctv: list[str] = []
    intrinsics_diagnostics: dict[str, Any] | None = None
    if backend == "colmap":
        if not isinstance(prepared, PreparedInputs):
            raise RuntimeError("COLMAP requires local prepared images")
        colmap_result = run_colmap_reconstruction(
            prepared,
            output_dir,
            max_image_size=getattr(args, "colmap_max_image_size", 1600),
        )
        missing_cctv = [
            camera.camera_id
            for index, camera in enumerate(calibration_input.cctv_cameras)
            if index not in colmap_result.extrinsics
        ]
        if missing_cctv:
            LOGGER.warning(
                "COLMAP did not register CCTV image(s); the result will contain "
                "only geometrically verified cameras: %s",
                ", ".join(missing_cctv),
            )
        alignment, observations, alignment_pose_count = collect_colmap_pnp_alignment(
            prepared,
            colmap_result.extrinsics,
            marker_definitions,
            inlier_threshold_m=args.alignment_threshold_m,
        )
        camera_extrinsics = colmap_result.extrinsics
    else:
        assert checkpoint is not None
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
            calibrated_matrices = (
                list(prepared.cctv_new_camera_matrices)
                + [prepared.reference_new_camera_matrix]
                * len(prepared.reference_frame_indices)
            )
            calibrated_intrinsics, valid_regions = build_vggt_preprocessed_intrinsics(
                prepared.image_paths,
                calibrated_matrices,
                image_resolution=args.image_resolution,
                resize_mode=args.resize_mode,
                return_valid_regions=True,
            )
            predictions['valid_regions'] = valid_regions
            model_intrinsics = np.asarray(predictions["intrinsics"], dtype=np.float64)
            focal_relative_error = np.abs(
                np.stack(
                    (
                        model_intrinsics[:, 0, 0] - calibrated_intrinsics[:, 0, 0],
                        model_intrinsics[:, 1, 1] - calibrated_intrinsics[:, 1, 1],
                    ),
                    axis=1,
                )
                / np.stack(
                    (
                        calibrated_intrinsics[:, 0, 0],
                        calibrated_intrinsics[:, 1, 1],
                    ),
                    axis=1,
                )
            )
            principal_point_error = np.linalg.norm(
                model_intrinsics[:, :2, 2] - calibrated_intrinsics[:, :2, 2],
                axis=1,
            )
            intrinsics_diagnostics = {
                "source": "calibrated intrinsics transformed through VGGT crop/resize/padding",
                "model_focal_relative_error_mean": float(np.mean(focal_relative_error)),
                "model_focal_relative_error_max": float(np.max(focal_relative_error)),
                "model_principal_point_error_mean_px": float(np.mean(principal_point_error)),
                "model_principal_point_error_max_px": float(np.max(principal_point_error)),
            }
            # VGGT depth/extrinsics are geometrically coupled to the model's
            # predicted K, so retain that K for depth unprojection and final
            # ArUco alignment. Calibrated K is used by feature-track BA.
            predictions["ba_intrinsics"] = calibrated_intrinsics
            LOGGER.info(
                "Using calibrated preprocessed intrinsics for BA/alignment "
                "(VGGT focal difference mean %.1f%%, principal-point difference mean %.1f px)",
                100.0 * intrinsics_diagnostics["model_focal_relative_error_mean"],
                intrinsics_diagnostics["model_principal_point_error_mean_px"],
            )
        if getattr(args, "use_ba", False):
            if feature_metadata is not None:
                raise ValueError(
                    "--use_ba is unavailable with --feature-bundle because the server "
                    "receives CCTV patch tokens, not the source images needed to match tracks."
                )
            LOGGER.info(
                "Running feature-track BA without ArUco/USD constraints"
            )
            ba_result = run_bundle_adjustment(
                predictions,
                cctv_count=prepared.cctv_count,
                min_depth_confidence=args.min_depth_confidence,
                max_images=getattr(args, 'ba_max_images', None),
                max_tracks=getattr(args, 'ba_max_tracks', None),
                max_iterations=getattr(args, 'ba_max_iterations', BA_MAX_ITERATIONS),
            )
            predictions["extrinsics"] = ba_result.extrinsics
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
        # ArUco geometry is intentionally applied only after feature BA. It
        # estimates one similarity transform from the refined VGGT frame into
        # the USD map frame and never changes relative camera poses inside BA.
        alignment = robust_similarity_transform(
            source_points,
            target_points,
            inlier_threshold_m=args.alignment_threshold_m,
        )
        camera_extrinsics = {
            index: extrinsic for index, extrinsic in enumerate(predictions["extrinsics"])
        }
        alignment_pose_count = None

    inlier_residuals = alignment.residuals[alignment.inlier_mask]
    LOGGER.info(
        "Alignment: %d/%d inliers, scale %.6f, RMSE %.4f m",
        int(alignment.inlier_mask.sum()),
        len(alignment.inlier_mask),
        alignment.scale,
        float(np.sqrt(np.mean(inlier_residuals**2))),
    )
    if ba_result is not None:
        LOGGER.info(
            "BA has observations in %d camera(s) from %d selected image(s), using %d track(s) / %d observation(s) "
            "(mean %.2f, %d multi-view): RMSE %.2f px -> %.2f px",
            ba_result.camera_count,
            ba_result.selected_image_count,
            ba_result.track_count,
            ba_result.observation_count,
            ba_result.mean_track_length,
            ba_result.multi_view_track_count,
            ba_result.initial_rmse_px,
            ba_result.final_rmse_px,
        )
        LOGGER.info('BA optimized %d poses; fixed %d poses (%d without observations); converged=%s',
                    len(ba_result.variable_image_indices), len(ba_result.fixed_image_indices),
                    len(ba_result.unobserved_image_indices), ba_result.solver_converged)
        LOGGER.info(
            "BA CCTV feature observations: %s",
            ", ".join(
                f"{camera.camera_id}={ba_result.cctv_observation_counts[index]}"
                for index, camera in enumerate(calibration_input.cctv_cameras)
            ),
        )
        if ba_result.fixed_cctv_indices:
            LOGGER.warning(
                "BA kept weakly connected CCTV pose(s) fixed (<%d observations): %s",
                BA_MIN_CCTV_OBSERVATIONS,
                ", ".join(
                    calibration_input.cctv_cameras[index].camera_id
                    for index in ba_result.fixed_cctv_indices
                ),
            )

    cameras_output = []
    for index, camera in enumerate(calibration_input.cctv_cameras):
        if index not in camera_extrinsics:
            continue
        world_to_camera, camera_to_world = transform_camera_extrinsic(
            camera_extrinsics[index], alignment
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
        else:
            source_image = cv2.imread(str(camera.image_path))
            if source_image is None:
                raise ValueError(f'Cannot read camera source image size: {camera.camera_id}')
            camera_output['source_image_size'] = [int(source_image.shape[1]), int(source_image.shape[0])]
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
            "reconstruction_backend": backend,
            "edge_id": feature_metadata.get("edge_id") if feature_metadata else None,
            "request_id": feature_metadata.get("request_id") if feature_metadata else None,
            "checkpoint_sha256": file_sha256(checkpoint) if checkpoint else None,
            "preprocess": feature_metadata.get("preprocess") if feature_metadata else None,
            "total_images": prepared.cctv_count + len(prepared.reference_frame_indices),
            "cctv_images": prepared.cctv_count,
            "reference_frames": len(prepared.reference_frame_indices),
            "reference_video_frame_indices": list(prepared.reference_frame_indices),
            "image_resolution": args.image_resolution,
            "resize_mode": args.resize_mode,
            "intrinsics": intrinsics_diagnostics,
            "bundle_adjustment": (
                {
                    "enabled": True,
                    "camera_count": ba_result.camera_count,
                    "selected_image_count": ba_result.selected_image_count,
                    "track_count": ba_result.track_count,
                    "observation_count": ba_result.observation_count,
                    "mean_track_length": ba_result.mean_track_length,
                    "multi_view_track_count": ba_result.multi_view_track_count,
                    "cctv_observation_counts": {
                        camera.camera_id: ba_result.cctv_observation_counts[index]
                        for index, camera in enumerate(calibration_input.cctv_cameras)
                    },
                    "fixed_weak_cctv": [
                        calibration_input.cctv_cameras[index].camera_id
                        for index in ba_result.fixed_cctv_indices
                    ],
                    "initial_rmse_px": ba_result.initial_rmse_px,
                    "final_rmse_px": ba_result.final_rmse_px,
                    "solver_nfev": ba_result.solver_nfev,
                    "solver_optimality": ba_result.solver_optimality,
                    "solver_message": ba_result.solver_message,
                    "solver_converged": ba_result.solver_converged,
                    "image_budget": ba_result.image_budget,
                    "track_budget": ba_result.track_budget,
                    "iteration_budget": ba_result.iteration_budget,
                    "variable_image_indices": list(ba_result.variable_image_indices),
                    "fixed_image_indices": list(ba_result.fixed_image_indices),
                    "unobserved_image_indices": list(ba_result.unobserved_image_indices),
                    "coordinate_constraint": "none; ArUco is applied only after BA",
                }
                if ba_result is not None
                else {"enabled": False}
            ),
            "colmap": (
                {
                    "registered_image_count": colmap_result.registered_image_count,
                    "registered_cctv_count": sum(
                        index in colmap_result.extrinsics
                        for index in range(prepared.cctv_count)
                    ),
                    "unregistered_cctv": missing_cctv,
                    "complete": not missing_cctv,
                    "sparse_point_count": colmap_result.point_count,
                    "matched_pair_count": colmap_result.match_pair_count,
                    "aruco_pnp_pose_count": alignment_pose_count,
                    "intrinsics": "fixed per undistorted image (PINHOLE)",
                }
                if colmap_result is not None
                else None
            ),
        },
        "alignment": {
            "method": (
                "aruco_pnp_camera_poses"
                if backend == "colmap"
                else (
                    "vggt_ba_then_aruco_depth_corners"
                    if ba_result is not None
                    else "vggt_depth_corners"
                )
            ),
            "scale": alignment.scale,
            "rotation": _json_matrix(alignment.rotation),
            "translation_m": alignment.translation.tolist(),
            "reconstruction_to_usd": _json_matrix(alignment.matrix4()),
            # Kept for existing VGGT-result consumers; COLMAP uses the neutral
            # reconstruction_to_usd field above.
            "vggt_to_usd": (
                _json_matrix(alignment.matrix4()) if backend == "vggt" else None
            ),
            "correspondence_count": (
                alignment_pose_count if backend == "colmap" else len(source_points)
            ),
            "inlier_count": int(alignment.inlier_mask.sum()),
            "rmse_m": float(np.sqrt(np.mean(inlier_residuals**2))),
            "max_error_m": float(np.max(inlier_residuals)),
            "inlier_threshold_m": args.alignment_threshold_m,
        },
        "marker_observations": observations,
        "cameras": cameras_output,
    }
    if backend == 'vggt' and feature_metadata is None:
        sources = [dict(kind='cctv', camera_id=c.camera_id) for c in calibration_input.cctv_cameras]
        sources += [dict(kind='reference', frame_index=int(frame)) for frame in prepared.reference_frame_indices]
        result['point_cloud'] = export_depth_cloud(
            output_dir / 'point_cloud.npz', predictions, alignment, sources,
            min_confidence=args.min_depth_confidence,
            max_points=getattr(args, 'point_cloud_max_points', 1_000_000),
            ba_applied=ba_result is not None)
        LOGGER.info('Saved %d aligned point-cloud points', result['point_cloud']['point_count'])
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
    print("  2. 오프라인 - 로컬 사진/폴더에서 pose 계산, JSON 및 VGGT point cloud 출력")
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
    """Calibrate local photos and preserve the final JSON and paired point cloud."""
    image_source = args.images or input(
        "포즈를 구할 사진 또는 사진 폴더 경로: "
    ).strip()
    reference_video = args.reference_video or input(
        "reference 용 영상 경로를 입력해주세요: "
    ).strip()
    marker_tree = _interactive_marker_tree(args.marker_tree)
    checkpoint = (
        _resolve_checkpoint(args.checkpoint) if args.backend == "vggt" else None
    )
    destination = _offline_json_destination(args)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="dt-offline-calibration-") as temporary:
        working_dir = Path(temporary)
        manifest_path, descriptions = build_offline_manifest(
            image_source,
            reference_video,
            working_dir / "offline_input.json",
            sample_count=(
                args.reference_sample_count or DEFAULT_REFERENCE_SAMPLE_COUNT
            ),
            camera_config=getattr(args, "camera_config", None),
        )
        print("\nOffline 입력 intrinsic")
        for description in descriptions:
            print(f"  - {description}")
        print(
            "로컬 사진과 reference 영상을 결합해 "
            f"{args.backend.upper()} 추론을 시작합니다."
        )

        args.config = str(manifest_path)
        args.feature_bundle = None
        args.checkpoint = str(checkpoint) if checkpoint is not None else None
        args.output_dir = str(working_dir)
        args.marker_tree = str(marker_tree)
        temporary_result = run_calibration(args)
        result = json.loads(temporary_result.read_text(encoding="utf-8"))
        result["input"]["mode"] = "offline_images"
        result["input"]["offline_image_source"] = str(
            Path(image_source).expanduser().resolve()
        )
        preserve_offline_cloud(result, working_dir, destination)
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
    from dt_common.contracts.edge import DEFAULT_TOPIC_ROOT
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
        sample_count=(args.reference_sample_count or DEFAULT_REFERENCE_SAMPLE_COUNT),
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
    parser.add_argument("--checkpoint", help="VGGT-Omega .pt checkpoint (VGGT backend only)")
    parser.add_argument(
        "--backend",
        choices=("vggt", "colmap"),
        default="vggt",
        help="Pose reconstruction backend; COLMAP accepts only local images.",
    )
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
    parser.add_argument(
        "--reference-sample-count",
        type=int,
        default=None,
        help=f"Reference video frames to sample (default: {DEFAULT_REFERENCE_SAMPLE_COUNT}).",
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
    parser.add_argument('--max-images', type=int, help='Maximum total CCTV + reference images; does not bypass GPU memory checks.')
    parser.add_argument('--ba-max-images', type=int, help='Optional BA subsampling cap; default uses ALL input images.')
    parser.add_argument('--ba-max-tracks', type=int, help='BA track budget; default scales with the number of input images.')
    parser.add_argument('--ba-max-iterations', type=int, default=BA_MAX_ITERATIONS,
                        help='Maximum BA solver evaluations (default: 300); convergence is reported in JSON.')
    parser.add_argument('--point-cloud-max-points', type=int, default=1_000_000,
                        help='Saved local VGGT cloud point budget (1 to 2000000).')
    parser.add_argument(
        "--colmap-max-image-size",
        type=int,
        default=1600,
        help="Maximum image side used by COLMAP SIFT extraction.",
    )
    parser.add_argument(
        "--use_ba",
        action="store_true",
        help=(
            "Refine local-image VGGT poses with SIFT feature-track bundle "
            "adjustment; unavailable with --feature-bundle."
        ),
    )
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
        if (
            args.reference_sample_count is not None
            and args.reference_sample_count <= 0
        ):
            raise ValueError("reference sample count must be greater than zero")
        if args.config is None:
            mode = select_cli_mode(args.mode)
            if mode == "online":
                if args.backend == "colmap":
                    raise ValueError(
                        "--backend colmap is only supported in offline/local manifest mode"
                    )
                if args.transfer_timeout <= 0 or args.result_timeout <= 0:
                    raise ValueError("Zenoh timeout must be greater than zero")
                result_path = run_interactive_calibration(args)
            else:
                result_path = run_offline_calibration(args)
        else:
            if args.backend == "vggt" and not args.checkpoint:
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
