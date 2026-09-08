import json
from pathlib import Path

from apps.edge_client.agent import build_parser, configure_edge_identity
from apps.edge_client.inference import parse_args, resolve_edge_path
from apps.server_worker.inference import get_parser
from dt_common.runtime_config import parse_runtime_args


def test_runtime_profile_paths_are_profile_relative_and_cli_wins(tmp_path, monkeypatch):
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "arguments": {
                    "priority_interval": 0.5,
                    "deployment": "scene.json",
                    "no_metrics": True,
                },
            }
        )
    )
    monkeypatch.chdir("/tmp")
    args = parse_runtime_args(
        get_parser(), ["--runtime-config", str(profile), "--priority-interval", "1"]
    )
    assert args.priority_interval == 1
    assert args.deployment == str(tmp_path / "scene.json")
    assert args.no_metrics is True


def test_documented_cli_deployment_is_relative_to_working_directory(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    args = parse_args(["--deployment", "../deployments/scene_0812_poc.json"])
    assert (
        resolve_edge_path(args.deployment)
        == root.parent / "deployments" / "scene_0812_poc.json"
    )


def test_zone_profile_accepts_json_objects():
    root = Path(__file__).resolve().parents[3]
    profile = root / "apps/deployments/poc/server.offline.zone.json"
    args = parse_runtime_args(get_parser(), ["--runtime-config", str(profile)])
    assert args.lod_edge_zones == {"edge_1": 2, "edge_2": 1}


def test_online_profile_requires_initialized_identity(tmp_path, monkeypatch):
    import pytest
    from apps.edge_client import inference

    monkeypatch.setattr(
        inference,
        "parse_args",
        lambda: parse_args(
            ["--edge-id-file", str(tmp_path / "missing.json"), "--require-identity"]
        ),
    )
    with pytest.raises(ValueError, match="agent.py init"):
        inference.main()
    assert not (tmp_path / "missing.json").exists()


def test_init_selects_four_deployment_cameras_from_eight_camera_registry(tmp_path):
    cameras = tmp_path / "cameras.yaml"
    cameras.write_text(
        "CAMERAS:\n" + "".join(f"  - id: '{index}'\n" for index in range(1, 9))
    )
    manifest = tmp_path / "scene.json"
    manifest.write_text(
        json.dumps({"edges": [{"id": "edge_1", "camera_ids": [2, 4, 6, 8]}]})
    )
    identity = tmp_path / "edge.json"
    args = build_parser().parse_args(
        [
            "init",
            "--edge-id",
            "edge_1",
            "--identity-file",
            str(identity),
            "--camera-config",
            str(cameras),
            "--deployment",
            str(manifest),
            "--endpoint",
            "tcp/server:7447",
        ]
    )
    result = configure_edge_identity(args)
    assert result["camera_ids"] == [2, 4, 6, 8]
    assert result["topics"]["inference"] == "dt/edges/edge_1/inference"
    assert result["zenoh_endpoint"] == "tcp/server:7447"
