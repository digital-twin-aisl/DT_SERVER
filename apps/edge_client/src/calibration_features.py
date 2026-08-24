"""Capture local CCTV frames and turn them into transportable DINO tokens."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml


from dt_common.calibration.protocol import (
    DEFAULT_IMAGE_SIZE,
    PREPROCESS_NAME,
    encode_feature_bundle,
    file_sha256,
)
from dt_common.calibration.preprocess import preprocess_frame


def _matrix3(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"{name}: camera_matrix must be a finite 3x3 matrix")
    return matrix


def capture_and_encode(
    camera_config: str | Path,
    checkpoint: str | Path,
    edge_id: str,
    request_id: str,
    *,
    device: str = "cuda",
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> bytes:
    """Capture one frame per camera. Raw pixels never leave this function."""
    if image_size <= 0 or image_size % 16:
        raise ValueError("image_size must be a positive multiple of 16")
    config = yaml.safe_load(Path(camera_config).read_text(encoding="utf-8")) or {}
    cameras = config.get("CAMERAS") or []
    if not cameras:
        raise ValueError("no cameras are registered")

    images: list[np.ndarray] = []
    camera_metadata: list[dict[str, Any]] = []
    for camera in cameras:
        intrinsic = camera.get("intrinsic")
        if not isinstance(intrinsic, dict):
            raise ValueError(f"camera/{camera.get('id')} has no intrinsic calibration")
        matrix = _matrix3(intrinsic.get("camera_matrix"), f"camera/{camera.get('id')}")
        distortion = np.asarray(intrinsic.get("distortion_coefficients"), dtype=np.float64).reshape(-1)
        if distortion.size not in {4, 5, 8, 12, 14} or not np.isfinite(distortion).all():
            raise ValueError(f"camera/{camera.get('id')}: invalid distortion coefficients")
        calibration_size = intrinsic.get("image_size")
        if (
            not isinstance(calibration_size, (list, tuple))
            or len(calibration_size) != 2
            or min(int(value) for value in calibration_size) <= 0
        ):
            raise ValueError(f"camera/{camera.get('id')}: invalid intrinsic image_size")
        capture = cv2.VideoCapture(camera["url"])
        try:
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok or frame is None:
            raise RuntimeError(f"camera/{camera.get('id')}: failed to capture a frame")
        image, adjusted_matrix = preprocess_frame(
            frame,
            matrix,
            distortion,
            image_size,
            calibration_image_size=calibration_size,
        )
        images.append(image)
        camera_metadata.append(
            {
                "camera_id": str(camera["id"]),
                "camera_key": f"{edge_id}/camera/{camera['id']}",
                "name": str(camera.get("name") or camera["id"]),
                "location": camera.get("location"),
                "twin_id": camera.get("twin_id"),
                "camera_matrix": matrix.tolist(),
                "distortion_coefficients": distortion.tolist(),
                "source_image_size": [int(frame.shape[1]), int(frame.shape[0])],
                "processed_camera_matrix": adjusted_matrix.tolist(),
            }
        )

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"VGGT-Omega checkpoint not found: {checkpoint_path}")
    from vggt_omega.models.aggregator import _build_patch_embed

    encoder = _build_patch_embed(patch_size=16, embed_dim=1024).eval()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    prefix = "aggregator.patch_embed."
    patch_state = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
    if not patch_state:
        raise ValueError("checkpoint does not contain aggregator.patch_embed weights")
    encoder.load_state_dict(patch_state, strict=True)
    encoder = encoder.to(device)
    tensor = torch.from_numpy(np.stack(images)).to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    with torch.inference_mode():
        with torch.autocast(
            device_type=torch.device(device).type, dtype=torch.float16
        ):
            output = encoder((tensor - mean) / std)
        if isinstance(output, dict):
            output = output["x_norm_patchtokens"]
    tokens = output.detach().float().cpu().numpy()
    return encode_feature_bundle(
        {
            "request_id": request_id,
            "edge_id": edge_id,
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "preprocess": PREPROCESS_NAME,
            "image_size": [image_size, image_size],
            "patch_size": 16,
            "encoder_dtype": "float16",
            "cameras": camera_metadata,
        },
        tokens,
    )
