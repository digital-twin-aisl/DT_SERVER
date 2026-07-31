from pathlib import Path
import json
import sys

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from apps.server_worker import inference  # noqa: E402
from apps.server_worker.src.protocol.scene_zenoh import encode_scene  # noqa: E402
from apps.server_worker.src.protocol.zmq import Protocol  # noqa: E402


def make_pose_result():
    predictions = torch.zeros((2, 2, 15, 5), dtype=torch.float32)
    predictions[1, 0, :, :3] = torch.arange(
        45,
        dtype=torch.float32,
    ).reshape(15, 3)
    result_roots = torch.tensor(
        [
            [[0, 0, 0, -1, 0.4], [0, 0, 0, -1, 0.3]],
            [[0, 0, 0, 0, 0.9], [0, 0, 0, -1, 0.2]],
        ],
        dtype=torch.float32,
    )
    return {
        "pred": predictions,
        "grid_centers": result_roots,
    }


def test_scene_output_contains_root_for_low_lod_and_pose_for_lod2():
    roots = torch.tensor(
        [
            [[10, 20, 30, 0, 0.4], [0, 0, 0, -1, 0.3]],
            [[100, 200, 300, 0, 0.9], [0, 0, 0, -1, 0.2]],
        ],
        dtype=torch.float32,
    )
    scene = inference.build_scene_output(
        make_pose_result(),
        roots,
        ids=[[5, None], [7, None]],
        lod_by_id={5: 1, 7: 2},
        timestamps=[100.0, 100.02],
        edge_ids=["0_edge", "1_edge"],
        sync_spread=0.02,
    ).to_dict()

    assert scene["schema_version"] == 1
    assert scene["timestamp"] == pytest.approx(100.01)
    assert [person["global_id"] for person in scene["people"]] == [7, 5]

    lod2_person, lod1_person = scene["people"]
    assert lod2_person["lod"] == 2
    assert lod2_person["root"] == {
        "edge_id": "1_edge",
        "candidate_index": 0,
        "position": [100.0, 200.0, 300.0],
        "confidence": 0.8999999761581421,
        "timestamp": 100.02,
    }
    assert lod2_person["pose"]["joint_format"] == "voxelpose_15j_xyz"
    assert len(lod2_person["pose"]["joints"]) == 15

    assert lod1_person["lod"] == 1
    assert lod1_person["root"]["position"] == [10.0, 20.0, 30.0]
    assert lod1_person["pose"] is None


def test_scene_output_respects_max_people_limit():
    roots = torch.tensor(
        [
            [
                [0, 0, 0, 0, 0.3],
                [1, 0, 0, 0, 0.9],
            ]
        ],
        dtype=torch.float32,
    )
    pose_result = {
        "pred": torch.zeros((1, 2, 15, 5), dtype=torch.float32),
        "grid_centers": roots.clone(),
    }
    scene = inference.build_scene_output(
        pose_result,
        roots,
        ids=[[1, 2]],
        lod_by_id={1: 1, 2: 1},
        timestamps=[10.0],
        edge_ids=["0_edge"],
        sync_spread=0.0,
        max_people=1,
    )

    assert [person.global_id for person in scene.people] == [2]


def test_save_scene_for_viser_appends_scenes_as_jsonl(tmp_path):
    output_path = tmp_path / "viser_scenes.jsonl"
    first_scene = inference.SceneOutput(
        timestamp=10.0,
        sync_spread_seconds=0.01,
        people=[],
    )
    second_scene = inference.SceneOutput(
        timestamp=11.0,
        sync_spread_seconds=0.02,
        people=[],
    )

    inference.save_scene_for_viser(first_scene, output_path)
    inference.save_scene_for_viser(second_scene, output_path)

    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert records == [first_scene.to_dict(), second_scene.to_dict()]


class FakeSocket:
    def __init__(self):
        self.sent = None

    def send_json(self, payload):
        self.sent = payload


def test_protocol_sends_scene_as_json():
    protocol = Protocol.__new__(Protocol)
    protocol.socket = FakeSocket()
    payload = {
        "schema_version": 1,
        "timestamp": 10.0,
        "sync_spread_seconds": 0.0,
        "people": [],
    }

    assert protocol.send_scene(payload)
    assert protocol.socket.sent == payload


def test_encode_scene_produces_compact_utf8_json():
    payload = {
        "schema_version": 1,
        "timestamp": 10.0,
        "sync_spread_seconds": 0.0,
        "people": [{"global_id": 7, "label": "사람"}],
    }

    encoded = encode_scene(payload)

    assert b" " not in encoded
    assert json.loads(encoded.decode("utf-8")) == payload
