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
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import torch


LOGGER = logging.getLogger("calibration_worker")
WORKER_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = WORKER_DIR.parents[1]
VGGT_OMEGA_ROOT = WORKER_DIR / "vggt-omega"
DEFAULT_MARKER_TREE = (
    REPOSITORY_ROOT
    / "apps/isaac_sim_client/aruco_boards/aruco_marker_tree.usd"
)


@dataclass(frozen=True)
class CameraInput:
    camera_id: str
    image_path: Path
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


@dataclass(frozen=True)
class PreparedInputs:
    image_paths: tuple[Path, ...]
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
    result = {
        "images": predictions["images"][0]
        .detach()
        .float()
        .cpu()
        .permute(0, 2, 3, 1)
        .numpy(),
        "depth": predictions["depth"][0, ..., 0].detach().float().cpu().numpy(),
        "depth_conf": predictions["depth_conf"][0, ..., 0]
        .detach()
        .float()
        .cpu()
        .numpy(),
        "extrinsics": extrinsics[0].detach().float().cpu().numpy(),
        "intrinsics": intrinsics[0].detach().float().cpu().numpy(),
    }
    del predictions, images, model
    torch.cuda.empty_cache()
    return result


def _json_matrix(matrix: np.ndarray) -> list[list[float]]:
    return np.asarray(matrix, dtype=np.float64).tolist()


def run_calibration(args: argparse.Namespace) -> Path:
    calibration_input = load_calibration_input(args.config)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    max_images = args.max_images
    if max_images is None:
        max_images = estimate_max_images_from_gpu(args.device)
        LOGGER.info("GPU memory estimate allows up to %d total images", max_images)
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

    predictions = run_vggt_inference(
        prepared.image_paths,
        Path(args.checkpoint).expanduser().resolve(),
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
        cameras_output.append(
            {
                "camera_id": camera.camera_id,
                "source_image": str(camera.image_path),
                "camera_matrix": _json_matrix(camera.camera_matrix),
                "distortion_coefficients": camera.distortion_coefficients.tolist(),
                "undistorted_camera_matrix": _json_matrix(
                    prepared.cctv_new_camera_matrices[index]
                ),
                "world_to_camera": _json_matrix(world_to_camera),
                "camera_to_world": _json_matrix(camera_to_world),
                "position_m": camera_to_world[:3, 3].tolist(),
            }
        )

    result = {
        "schema_version": 1,
        "coordinate_convention": {
            "world": "marker-tree USD coordinates in metres",
            "camera": "OpenCV: +X right, +Y down, +Z forward",
            "extrinsic": "world_to_camera maps homogeneous world points to camera points",
        },
        "marker_tree": str(Path(args.marker_tree).expanduser().resolve()),
        "input": {
            "total_images": len(prepared.image_paths),
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


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Estimate CCTV extrinsics using VGGT-Omega and an ArUco marker tree."
    )
    parser.add_argument("--config", required=True, help="Calibration input JSON manifest")
    parser.add_argument("--checkpoint", required=True, help="VGGT-Omega .pt checkpoint")
    parser.add_argument("--output-dir", required=True, help="Directory for artifacts/results")
    parser.add_argument(
        "--marker-tree",
        default=str(DEFAULT_MARKER_TREE),
        help="ArUco marker-tree USD path",
    )
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
