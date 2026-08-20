import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from apps.edge_client.src.root.utils.cameras import project_pose  # noqa: E402
from apps.edge_client.src.utils.calibration import CalibrationData  # noqa: E402
from apps.edge_client.src.utils.input import discover_dataset_videos  # noqa: E402
from dt_common.calibration.voxelpose import (  # noqa: E402
    load_calibration_result,
    select_voxelpose_cameras,
)


def _record(camera_id=2):
    return {
        "camera_id": f"camera/{camera_id}",
        "camera_matrix": [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]],
        "distortion_coefficients": [0, 0, 0, 0, 0],
        "world_to_camera": [
            [1, 0, 0, -10],
            [0, 1, 0, -20],
            [0, 0, 1, -30],
            [0, 0, 0, 1],
        ],
        "camera_to_world": [
            [1, 0, 0, 10],
            [0, 1, 0, 20],
            [0, 0, 1, 30],
            [0, 0, 0, 1],
        ],
        "position_m": [10, 20, 30],
    }


def _result(*records):
    return {
        "coordinate_convention": {
            "world": "marker-tree USD coordinates in metres",
            "camera": "OpenCV: +X right, +Y down, +Z forward",
        },
        "cameras": list(records),
    }


def test_modern_dataset_videos_are_sorted_by_physical_camera_id(tmp_path):
    for name in ("camera_8.mkv", "camera_2.mkv", "camera_6.mkv", "camera_4.mkv"):
        (tmp_path / name).touch()

    sources = discover_dataset_videos(tmp_path)

    assert [camera_id for camera_id, _ in sources] == [2, 4, 6, 8]


def test_usd_camera_uses_camera_centre_in_millimetres():
    cameras = select_voxelpose_cameras(_result(_record()), [2])
    camera = cameras[0]
    optical_axis_point = torch.tensor([[10000.0, 20000.0, 40000.0]])

    projected = project_pose(optical_axis_point, camera)

    assert camera["T"].shape == (3, 1)
    assert camera["T"].reshape(-1).tolist() == [10000, 20000, 30000]
    assert torch.allclose(projected, torch.tensor([[960.0, 540.0]]), atol=1e-3)


def test_dataset_calibration_selects_video_camera_order(tmp_path):
    for camera_id in (4, 2):
        (tmp_path / f"camera_{camera_id}.mkv").touch()
    result_path = tmp_path / "calibration_result_test.json"
    result_path.write_text(
        json.dumps(_result(_record(2), _record(4))),
        encoding="utf-8",
    )
    cfg = SimpleNamespace(
        NETWORK=SimpleNamespace(
            IMAGE_SIZE_ORIG=np.array([1920, 1080]),
            IMAGE_SIZE=np.array([960, 512]),
        )
    )

    CalibrationData(cfg, example_path=tmp_path)

    assert [camera["id"] for camera in cfg.CAMS] == [2, 4]


def test_calibration_rejects_non_metre_world(tmp_path):
    result = _result(_record())
    result["coordinate_convention"]["world"] = "USD coordinates in centimetres"
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(ValueError, match="metres"):
        load_calibration_result(path)
