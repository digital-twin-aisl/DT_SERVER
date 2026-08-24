import json

from apps.edge_client.src.camera_status import build_camera_status_message
from apps.edge_client.src.protocol.edge import EdgeTopics as ClientEdgeTopics
from apps.edge_manager.app.cli import _heartbeat_command
from apps.edge_manager.app.protocol import (
    EdgeTopics as ManagerEdgeTopics,
    camera_records_from_message,
)
from apps.edge_manager.app.registry import EdgeRegistry


def test_camera_snapshot_contains_only_allowed_camera_information(tmp_path):
    config_path = tmp_path / "cameras.yaml"
    config_path.write_text(
        """
CAMERAS:
  - id: '1'
    name: loading-dock
    url: rtsp://admin:secret-password@10.0.0.8:554/live
    location: private-building
    twin_id: private-twin
    intrinsic:
      camera_matrix: [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]]
      distortion_coefficients: [-0.1, 0.02, 0, 0, 0]
      image_size: [1920, 1080]
      method: opencv_checkerboard
    extrinsic:
      world_to_camera: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]
""",
        encoding="utf-8",
    )

    message = build_camera_status_message(
        "edge-1",
        config_path,
        ping_timeout=0.1,
        sent_at=123.0,
        pinger=lambda camera, timeout: True,
    )

    camera = message["data"]["cameras"][0]
    assert camera == {
        "camera_id": "1",
        "exists": True,
        "ping": True,
        "calibration": {
            "intrinsic": {
                "camera_matrix": [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]],
                "image_size": [1920, 1080],
                "method": "opencv_checkerboard",
            },
            "extrinsic": {
                "world_to_camera": [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 1],
                    [0, 0, 0, 1],
                ]
            },
            "distortion_coefficients": [-0.1, 0.02, 0, 0, 0],
        },
    }
    serialized = json.dumps(message)
    assert "secret-password" not in serialized
    assert "loading-dock" not in serialized
    assert "private-building" not in serialized
    assert "private-twin" not in serialized


def test_manager_whitelists_camera_fields_and_topics_match():
    message = {
        "kind": "camera_status",
        "data": {
            "cameras": [
                {
                    "camera_id": "01",
                    "exists": True,
                    "ping": False,
                    "url": "rtsp://secret/stream",
                    "name": "private-name",
                    "calibration": {
                        "intrinsic": {"camera_matrix": [[1, 0, 0]]},
                        "extrinsic": None,
                        "distortion_coefficients": [0, 0, 0, 0],
                        "private": "discard-me",
                    },
                }
            ]
        },
    }
    records = camera_records_from_message(message)
    assert records == [
        {
            "camera_id": "1",
            "exists": True,
            "ping": False,
            "calibration": {
                "intrinsic": {"camera_matrix": [[1, 0, 0]]},
                "extrinsic": None,
                "distortion_coefficients": [0, 0, 0, 0],
            },
        }
    ]
    assert ClientEdgeTopics("edge-1").cameras == ManagerEdgeTopics("edge-1").cameras


def test_registry_marks_missing_cameras_and_offline_pings(tmp_path):
    registry = EdgeRegistry(tmp_path / "edges.json")
    calibration = {
        "intrinsic": {"camera_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
        "extrinsic": None,
        "distortion_coefficients": [0, 0, 0, 0],
    }
    registry.observe_cameras(
        "edge-1",
        [
            {
                "camera_id": "1",
                "exists": True,
                "ping": True,
                "calibration": calibration,
            },
            {
                "camera_id": "2",
                "exists": True,
                "ping": True,
                "calibration": calibration,
            },
        ],
        edge_sent_at=10.0,
        received_at=20.0,
    )
    registry.observe_cameras(
        "edge-1",
        [{"camera_id": "2", "exists": True, "ping": True, "calibration": calibration}],
        edge_sent_at=30.0,
        received_at=40.0,
    )

    cameras = registry.snapshot()["edges"]["edge-1"]["cameras"]
    assert cameras["1"]["exists"] is False
    assert cameras["1"]["ping"] is False
    assert cameras["1"]["calibration"] == calibration
    assert cameras["2"]["exists"] is True
    assert cameras["2"]["ping"] is True

    assert registry.mark_offline(5.0, now=50.0) == ["edge-1"]
    assert registry.snapshot()["edges"]["edge-1"]["cameras"]["2"]["ping"] is False


def test_manager_heartbeat_requests_a_camera_refresh():
    message = _heartbeat_command("edge-1")
    assert message["kind"] == "command"
    assert message["command"] == "ping"
    assert message["parameters"]["request_cameras"] is True
