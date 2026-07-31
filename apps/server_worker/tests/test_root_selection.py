from pathlib import Path
import sys
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from apps.server_worker import inference  # noqa: E402


class FakeCalibration:
    def __init__(self, camera_ids_by_edge):
        self._metadata = {
            edge_id: SimpleNamespace(
                cams=[{"id": camera_id} for camera_id in camera_ids]
            )
            for edge_id, camera_ids in camera_ids_by_edge.items()
        }

    def __getitem__(self, edge_id):
        return self._metadata[edge_id]


def observation(edge_id, camera_id, global_id, center, frame=1):
    x, y = center
    return {
        "edge_id": edge_id,
        "cam": camera_id,
        "global_id": global_id,
        "frame": frame,
        "bbox": [x - 1, y - 1, x + 1, y + 1],
    }


def project_xy(roots, _camera):
    return roots[:, :2]


def test_duplicate_global_id_uses_highest_root_confidence(monkeypatch):
    monkeypatch.setattr(inference.cameras, "project_pose", project_xy)
    edge_ids = ["0_edge", "1_edge"]
    calibration = FakeCalibration({"0_edge": [1, 2], "1_edge": [1]})
    roots = torch.tensor(
        [
            [[0, 0, 0, 0, 0.40], [50, 0, 0, 0, 0.30]],
            [[100, 0, 0, 0, 0.90], [150, 0, 0, 0, 0.20]],
        ],
        dtype=torch.float32,
    )
    observations = [
        observation("0_edge", 1, 7, (0, 0)),
        observation("0_edge", 2, 7, (0, 0)),
        observation("1_edge", 1, 7, (100, 0)),
    ]

    ids = inference.assign_global_ids(
        observations,
        roots,
        edge_ids,
        calibration,
    )

    assert ids == [[None, None], [7, None]]


def test_edge_association_is_one_to_one(monkeypatch):
    monkeypatch.setattr(inference.cameras, "project_pose", project_xy)
    edge_ids = ["0_edge"]
    calibration = FakeCalibration({"0_edge": [1]})
    roots = torch.tensor(
        [[[0, 0, 0, 0, 0.80], [100, 0, 0, 0, 0.70]]],
        dtype=torch.float32,
    )
    observations = [
        observation("0_edge", 1, 10, (5, 0)),
        observation("0_edge", 1, 11, (95, 0)),
    ]

    candidates = inference.associate_global_ids_to_roots(
        observations,
        roots,
        edge_ids,
        calibration,
    )

    assert {
        (candidate.global_id, candidate.root_index) for candidate in candidates
    } == {(10, 0), (11, 1)}


def test_confidence_tie_uses_lower_reprojection_error(monkeypatch):
    monkeypatch.setattr(inference.cameras, "project_pose", project_xy)
    edge_ids = ["0_edge", "1_edge"]
    calibration = FakeCalibration({"0_edge": [1], "1_edge": [1]})
    roots = torch.tensor(
        [
            [[0, 0, 0, 0, 0.80]],
            [[100, 0, 0, 0, 0.80]],
        ],
        dtype=torch.float32,
    )
    observations = [
        observation("0_edge", 1, 3, (10, 0)),
        observation("1_edge", 1, 3, (101, 0)),
    ]

    ids = inference.assign_global_ids(
        observations,
        roots,
        edge_ids,
        calibration,
    )

    assert ids == [[None], [3]]


def test_root_beyond_reprojection_gate_is_not_assigned(monkeypatch):
    monkeypatch.setattr(inference.cameras, "project_pose", project_xy)
    edge_ids = ["0_edge"]
    calibration = FakeCalibration({"0_edge": [1]})
    roots = torch.tensor(
        [[[0, 0, 0, 0, 0.90]]],
        dtype=torch.float32,
    )
    observations = [
        observation("0_edge", 1, 7, (500, 0)),
    ]

    ids = inference.assign_global_ids(
        observations,
        roots,
        edge_ids,
        calibration,
    )

    assert ids == [[None]]
