"""Zenoh orchestration for privacy-preserving distributed calibration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import threading
import time
from typing import Any
from uuid import uuid4

import zenoh

if str(REPOSITORY_ROOT := Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import apps  # noqa: E402,F401
from dt_common.calibration.protocol import ChunkCollector  # noqa: E402
from apps.edge_manager.app.protocol import (  # noqa: E402
    DEFAULT_TOPIC_ROOT,
    SCHEMA_VERSION,
    EdgeTopics,
    decode_json,
    encode_json,
    make_zenoh_config,
    validate_edge_id,
)


@dataclass(frozen=True)
class FeatureTransfer:
    edge_id: str
    request_id: str
    command_id: str
    payload: bytes
    ack: dict[str, Any]


def request_calibration_features(
    edge_id: str,
    *,
    endpoint: str | None = None,
    zenoh_config: str | None = None,
    topic_root: str = DEFAULT_TOPIC_ROOT,
    timeout: float = 300.0,
) -> FeatureTransfer:
    """Ask one edge to capture all cameras and receive its DINO token bundle."""
    edge_id = validate_edge_id(edge_id)
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")
    request_id = uuid4().hex
    command_id = uuid4().hex
    topics = EdgeTopics(edge_id, topic_root)
    transfer_done = threading.Event()
    ack_done = threading.Event()
    state: dict[str, Any] = {}
    collector: ChunkCollector | None = None
    pending_chunks: dict[int, bytes] = {}

    def fail(exc: Exception) -> None:
        state.setdefault("error", str(exc))
        transfer_done.set()

    def on_manifest(sample: Any) -> None:
        nonlocal collector
        try:
            message = decode_json(sample.payload)
            if message.get("kind") != "calibration_features":
                return
            if message.get("request_id") != request_id:
                return
            if message.get("edge_id") != edge_id:
                raise ValueError("calibration manifest edge_id does not match")
            if str(sample.key_expr) != topics.calibration:
                raise ValueError("calibration manifest topic does not match")
            collector = ChunkCollector(
                int(message["chunk_count"]),
                str(message["sha256"]),
                int(message["byte_count"]),
            )
            for index, payload in pending_chunks.items():
                collector.add(index, payload)
            pending_chunks.clear()
            if collector.complete:
                transfer_done.set()
        except Exception as exc:
            fail(exc)

    def on_chunk(sample: Any) -> None:
        try:
            topic = str(sample.key_expr)
            prefix = topics.calibration_chunk(request_id, "")
            if not topic.startswith(prefix):
                raise ValueError("calibration chunk topic does not match")
            index = int(topic.rsplit("/", 1)[1])
            payload = (
                sample.payload.to_bytes()
                if hasattr(sample.payload, "to_bytes")
                else bytes(sample.payload)
            )
            if collector is None:
                pending_chunks.setdefault(index, payload)
            else:
                collector.add(index, payload)
                if collector.complete:
                    transfer_done.set()
        except Exception as exc:
            fail(exc)

    def on_ack(sample: Any) -> None:
        try:
            candidate = decode_json(sample.payload)
            if candidate.get("command_id") != command_id:
                return
            if candidate.get("edge_id") != edge_id:
                raise ValueError("calibration ACK edge_id does not match")
            state["ack"] = candidate
            ack_done.set()
            if not candidate.get("success"):
                transfer_done.set()
        except Exception as exc:
            fail(exc)

    command = {
        "schema_version": SCHEMA_VERSION,
        "kind": "command",
        "edge_id": edge_id,
        "command_id": command_id,
        "command": "capture_calibration_features",
        "sent_at": time.time(),
        "parameters": {"request_id": request_id},
    }
    config = make_zenoh_config(endpoint, zenoh_config)
    with zenoh.open(config) as session:
        manifest_subscriber = session.declare_subscriber(
            topics.calibration, on_manifest
        )
        chunk_subscriber = session.declare_subscriber(
            topics.calibration_chunk(request_id), on_chunk
        )
        ack_subscriber = session.declare_subscriber(topics.ack, on_ack)
        session.put(topics.command, encode_json(command))
        if not transfer_done.wait(timeout):
            raise TimeoutError(
                f"feature transfer timed out after {timeout:.1f}s"
            )
        if state.get("error"):
            raise ValueError(f"invalid feature transfer: {state['error']}")
        if not ack_done.wait(5.0):
            raise TimeoutError("feature transfer completed but ACK timed out")
        _ = manifest_subscriber, chunk_subscriber, ack_subscriber

    ack = state["ack"]
    if not ack.get("success"):
        raise RuntimeError(f"edge capture failed: {ack.get('error')}")
    if collector is None:
        raise RuntimeError("edge acknowledged capture without a feature manifest")
    return FeatureTransfer(
        edge_id=edge_id,
        request_id=request_id,
        command_id=command_id,
        payload=collector.assemble(),
        ack=ack,
    )


def calibration_result_payload(
    result: dict[str, Any], *, completed_at: float | None = None
) -> dict[str, Any]:
    """Reduce a worker result to the fields the edge is allowed to persist."""
    input_data = result.get("input") or {}
    cameras = []
    for camera in result.get("cameras") or []:
        camera_id = str(camera.get("edge_camera_id") or "")
        cameras.append(
            {
                "camera_id": camera_id,
                "camera_key": camera.get("camera_id"),
                "world_to_camera": camera.get("world_to_camera"),
                "camera_to_world": camera.get("camera_to_world"),
                "position_m": camera.get("position_m"),
                "undistorted_camera_matrix": camera.get(
                    "undistorted_camera_matrix"
                ),
            }
        )
    return {
        "request_id": input_data.get("request_id"),
        "completed_at": time.time() if completed_at is None else completed_at,
        "marker_tree": result.get("marker_tree"),
        "coordinate_convention": result.get("coordinate_convention"),
        "alignment": result.get("alignment"),
        "cameras": cameras,
    }


def publish_calibration_result(
    edge_id: str,
    result: dict[str, Any],
    *,
    endpoint: str | None = None,
    zenoh_config: str | None = None,
    topic_root: str = DEFAULT_TOPIC_ROOT,
    timeout: float = 30.0,
    completed_at: float | None = None,
) -> dict[str, Any]:
    """Send calibrated poses to an edge and wait until its YAML is saved."""
    edge_id = validate_edge_id(edge_id)
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")
    command_id = uuid4().hex
    topics = EdgeTopics(edge_id, topic_root)
    response: dict[str, Any] = {}
    received = threading.Event()

    def on_ack(sample: Any) -> None:
        try:
            candidate = decode_json(sample.payload)
            if candidate.get("command_id") != command_id:
                return
            if candidate.get("edge_id") != edge_id:
                return
            response.update(candidate)
            received.set()
        except Exception:
            return

    message = {
        "schema_version": SCHEMA_VERSION,
        "kind": "command",
        "edge_id": edge_id,
        "command_id": command_id,
        "command": "apply_calibration_result",
        "sent_at": time.time(),
        "parameters": calibration_result_payload(result, completed_at=completed_at),
    }
    config = make_zenoh_config(endpoint, zenoh_config)
    with zenoh.open(config) as session:
        subscriber = session.declare_subscriber(topics.ack, on_ack)
        session.put(topics.command, encode_json(message))
        if not received.wait(timeout):
            raise TimeoutError(
                f"calibration result ACK timed out after {timeout:.1f}s"
            )
        _ = subscriber
    if not response.get("success"):
        raise RuntimeError(f"edge rejected calibration result: {response.get('error')}")
    return response
