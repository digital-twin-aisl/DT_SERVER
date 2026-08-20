"""Build and publish privacy-limited camera status snapshots over Zenoh."""

from concurrent.futures import ThreadPoolExecutor
import socket
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

import yaml
import zenoh

from apps.edge_client.src.protocol.edge import SCHEMA_VERSION, EdgeTopics, encode_json
from apps.edge_client.src.protocol.zenoh import make_zenoh_config


DEFAULT_CAMERA_PING_TIMEOUT = 1.0


def _numbered_cameras(cameras: list[Any]) -> list[tuple[str, dict[str, Any]]]:
    """Return the same stable numeric identifiers used by camera_setup.py."""
    mappings = [camera for camera in cameras if isinstance(camera, dict)]
    reserved = {
        str(camera.get("id"))
        for camera in mappings
        if str(camera.get("id", "")).isdigit() and int(camera["id"]) > 0
    }
    used: set[str] = set()
    fallback = 1
    numbered: list[tuple[str, dict[str, Any]]] = []
    for camera in mappings:
        number = str(camera.get("id", ""))
        if not number.isdigit() or int(number) <= 0 or number in used:
            while str(fallback) in used or str(fallback) in reserved:
                fallback += 1
            number = str(fallback)
        used.add(number)
        numbered.append((number, camera))
    return numbered


def load_registered_cameras(config_path: str | Path) -> list[dict[str, Any]]:
    path = Path(config_path)
    if not path.exists():
        return []
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError("camera config must be a YAML object")
    cameras = config.get("CAMERAS") or []
    if not isinstance(cameras, list):
        raise ValueError("camera config CAMERAS must be a list")
    return [camera for camera in cameras if isinstance(camera, dict)]


def _ping_camera(camera: dict[str, Any], timeout: float) -> bool:
    """Check whether the camera's RTSP socket is reachable from this edge."""
    try:
        parsed = urlsplit(str(camera.get("url") or ""))
        if parsed.scheme not in {"rtsp", "rtsps"} or not parsed.hostname:
            return False
        port = parsed.port or (322 if parsed.scheme == "rtsps" else 554)
        with socket.create_connection((parsed.hostname, port), timeout=timeout):
            return True
    except (OSError, ValueError):
        return False


def _calibration(camera: dict[str, Any]) -> dict[str, Any]:
    intrinsic_source = camera.get("intrinsic")
    if not isinstance(intrinsic_source, dict):
        intrinsic_source = {}

    distortion = intrinsic_source.get("distortion_coefficients")
    if distortion is None:
        distortion = camera.get("distortion_coefficients")

    intrinsic = {
        key: intrinsic_source[key]
        for key in (
            "camera_matrix",
            "image_size",
            "method",
            "rms_error",
            "schema_version",
            "calibrated_at",
            "checkerboard",
            "undistorted_camera_matrix",
        )
        if key in intrinsic_source
    }
    if not intrinsic and camera.get("camera_matrix") is not None:
        intrinsic = {"camera_matrix": camera["camera_matrix"]}
        if camera.get("image_size") is not None:
            intrinsic["image_size"] = camera["image_size"]

    extrinsic = camera.get("extrinsic")
    if extrinsic is None:
        top_level_extrinsic = {
            key: camera[key]
            for key in (
                "world_to_camera",
                "camera_to_world",
                "position_m",
                "coordinate_convention",
            )
            if key in camera
        }
        extrinsic = top_level_extrinsic or None

    return {
        "intrinsic": intrinsic or None,
        "extrinsic": extrinsic,
        "distortion_coefficients": distortion,
    }


def build_camera_status_message(
    edge_id: str,
    config_path: str | Path,
    *,
    ping_timeout: float = DEFAULT_CAMERA_PING_TIMEOUT,
    sent_at: float | None = None,
    pinger: Callable[[dict[str, Any], float], bool] = _ping_camera,
) -> dict[str, Any]:
    if ping_timeout <= 0:
        raise ValueError("camera ping timeout must be greater than zero")
    numbered = _numbered_cameras(load_registered_cameras(config_path))
    workers = min(16, max(1, len(numbered)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        reachable = list(
            executor.map(lambda item: pinger(item[1], ping_timeout), numbered)
        )

    cameras = [
        {
            "camera_id": number,
            "exists": True,
            "ping": bool(ping),
            "calibration": _calibration(camera),
        }
        for (number, camera), ping in zip(numbered, reachable)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "camera_status",
        "edge_id": edge_id,
        "sent_at": time.time() if sent_at is None else sent_at,
        "data": {"cameras": cameras},
    }


def publish_camera_status_once(
    edge_id: str,
    config_path: str | Path,
    *,
    endpoint: str | None,
    zenoh_config: str | None,
    topic_root: str,
    ping_timeout: float = DEFAULT_CAMERA_PING_TIMEOUT,
) -> dict[str, Any]:
    message = build_camera_status_message(
        edge_id,
        config_path,
        ping_timeout=ping_timeout,
    )
    config = make_zenoh_config(endpoint, zenoh_config)
    with zenoh.open(config) as session:
        session.put(EdgeTopics(edge_id, topic_root).cameras, encode_json(message))
        # Give an asynchronous transport a short opportunity to flush before close.
        time.sleep(0.05)
    return message
