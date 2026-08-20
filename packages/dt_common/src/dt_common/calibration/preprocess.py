"""Deterministic image preprocessing shared by edge and calibration server."""

from __future__ import annotations

import cv2
import numpy as np

from .protocol import DEFAULT_IMAGE_SIZE


def preprocess_frame(
    frame: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    image_size: int = DEFAULT_IMAGE_SIZE,
    calibration_image_size: tuple[int, int] | list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Undistort then RGB-letterbox to a deterministic square canvas."""
    height, width = frame.shape[:2]
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64).copy()
    if calibration_image_size is not None:
        calibration_width, calibration_height = (
            int(value) for value in calibration_image_size
        )
        if calibration_width <= 0 or calibration_height <= 0:
            raise ValueError("calibration image dimensions must be positive")
        camera_matrix[0, :] *= width / calibration_width
        camera_matrix[1, :] *= height / calibration_height
    new_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix, distortion, (width, height), 0.0, (width, height)
    )
    corrected = cv2.undistort(frame, camera_matrix, distortion, None, new_matrix)
    scale = min(image_size / width, image_size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = cv2.resize(
        corrected, (resized_width, resized_height), interpolation=cv2.INTER_AREA
    )
    left = (image_size - resized_width) // 2
    top = (image_size - resized_height) // 2
    canvas = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    canvas[top : top + resized_height, left : left + resized_width] = resized
    adjusted = new_matrix.copy()
    adjusted[0, :] *= scale
    adjusted[1, :] *= scale
    adjusted[0, 2] += left
    adjusted[1, 2] += top
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return (
        np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0,
        adjusted,
    )
