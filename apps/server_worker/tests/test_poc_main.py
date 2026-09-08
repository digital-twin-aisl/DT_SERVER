import json
import sys
from types import SimpleNamespace

import numpy as np
import torch
import pytest

from apps.server_worker import inference
from apps.server_worker.tests.test_realtime_input import frame
from dt_common.inference_codec import encode_frame


@pytest.mark.parametrize("policy", ["rank", "zone"])
def test_replay_runs_server_loop_with_held_decisions_and_completed_output(
    tmp_path, monkeypatch, policy
):
    recordings = tmp_path / "recordings"
    for edge_index, edge in enumerate(("a", "b")):
        (recordings / edge).mkdir(parents=True)
        for index in range(7):
            values = frame(
                edge, index / 10, index, clock="dataset", calibration_digest="same"
            )
            roots = np.zeros((1, 10, 5), dtype=np.float32)
            roots[:, :, 3] = -1
            roots[0, 0] = [edge_index * 2000, 0, 0, 0, 0.9]
            values["roots"] = roots
            (recordings / edge / f"{index:09d}.dtframe").write_bytes(
                encode_frame(values)
            )
    manifest = tmp_path / "deployment.json"
    manifest.write_text(json.dumps({"poc": {"hazards_metres": [[0, 0]]}}))

    class Metadata:
        edge_ids = ["a", "b"]
        topics = {"a": "a", "b": "b"}

        def __init__(self, *args):
            pass

        def __getitem__(self, edge):
            return SimpleNamespace(
                cams=[{"id": i} for i in range(1, 5)], transform=None
            )

    monkeypatch.setattr(inference, "EdgeMetadataLoader", Metadata)
    monkeypatch.setattr(inference, "setup_cuda", lambda: torch.device("cpu"))
    monkeypatch.setattr(inference, "load_pose_model", lambda *args: None)
    monkeypatch.setattr(inference, "calibration_digest", lambda cameras: "same")

    def pose_models(models, heatmaps, roots, edge_ids, ids, lods):
        assert roots.shape == (2, 10, 5)
        return {
            "grid_centers": inference.select_lod2_roots(roots, ids, lods),
            "pred": torch.zeros((2, 10, 15, 5)),
        }

    monkeypatch.setattr(inference, "run_pose_models", pose_models)
    output = tmp_path / "decisions.jsonl"
    scenes = tmp_path / "scenes.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inference.py",
            "--deployment",
            str(manifest),
            "--input-clock",
            "dataset",
            "--input-mode",
            "strict",
            "--replay-inputs",
            str(recordings),
            "--no-scene-zenoh",
            "--no-zmq",
            "--no-metrics",
            "--decision-output",
            str(output),
            "--viser-debug-output",
            str(scenes),
            "--lod-policy",
            policy,
            "--lod-edge-zones",
            '{"a": 2, "b": 1}',
        ],
    )
    inference.main()
    decisions = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(decisions) == 7
    assert [row["timestamp"] for row in decisions if row["decision_updated"]] == [
        0,
        0.5,
    ]
    assert all(len(row["selected_ids"]) == 1 for row in decisions)
    rendered = [json.loads(line) for line in scenes.read_text().splitlines()]
    assert len(rendered) == 7
    assert all(len(row["people"]) == 2 for row in rendered)
    assert rendered[-1]["runtime"]["priority"]["interval_s"] == 0.5
