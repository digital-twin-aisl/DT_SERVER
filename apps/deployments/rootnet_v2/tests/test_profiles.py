# SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Offline checks: no camera, router, CUDA, or model loading."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml


PROFILE_DIR = Path(__file__).resolve().parents[1]
ROOT = PROFILE_DIR.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packages/dt_common/src"))


def read_json(name):
    return json.loads((PROFILE_DIR / name).read_text(encoding="utf-8"))


class VisualizationProfilesTest(unittest.TestCase):
    def test_shared_geometry_and_camera_order(self):
        actual = read_json("deployment.json")
        original = json.loads(
            (ROOT / "apps/deployments/scene_0812_poc.json").read_text()
        )
        for key in ("workspace", "edges", "resolution", "world_origin_m"):
            self.assertEqual(actual[key], original[key])
        for edge in actual["edges"]:
            role = edge["id"]
            args = read_json(f"{role}.json")["arguments"]
            identity = read_json(args["edge_id_file"])
            self.assertEqual(identity["edge_id"], role)
            self.assertEqual(identity["camera_ids"], edge["camera_ids"])
            self.assertEqual(args["topic_root"], actual["topic_root"])
            self.assertTrue(args["no_reid"])

    def test_focus_paths_and_checkpoint_are_consistent(self):
        checkpoints = []
        model_configs = []
        for role in ("edge_1", "edge_2", "server"):
            args = read_json(f"{role}.json")["arguments"]
            focus = yaml.safe_load(
                (PROFILE_DIR / args["cfg_focus"]).read_text()
            )["POSENET"]
            # These are the applications' distinct focus-path conventions.
            base = ROOT / "apps/server_worker" if role == "server" else ROOT
            checkpoints.append((base / focus["CKPT"]).resolve())
            model_path = (base / focus["CONFIG"]).resolve()
            self.assertTrue(model_path.is_file(), model_path)
            model_configs.append(yaml.safe_load(model_path.read_text()))
            self.assertFalse(focus["TENSORRT"])
            self.assertFalse(args["tensorrt"])
            self.assertEqual(args["deployment"], "deployment.json")
        self.assertEqual(len(set(checkpoints)), 1)
        self.assertEqual(
            checkpoints[0], ROOT / "apps/edge_client/models/POC_posenet.pth.tar"
        )
        for config in model_configs:
            self.assertEqual(config["NUM_VIEWS"], 4)
            self.assertEqual(config["NETWORK"]["IMAGE_SIZE_ORIG"], [1920, 1080])
            self.assertEqual(config["MULTI_PERSON"]["THRESHOLD"], 0.2)

    def test_server_has_all_pose_output_and_separate_topics(self):
        args = read_json("server.json")["arguments"]
        self.assertEqual(args["lod_policy"], "all")
        self.assertTrue(args["root_fallback"])
        self.assertTrue(args["no_zmq"])
        self.assertEqual(args["input_mode"], "independent")
        self.assertEqual(args["input_clock"], "dataset")
        self.assertTrue(args["priority_hazard"])
        self.assertEqual(args["scene_zenoh_topic"], "meta-sejong/rootnet-v2/scene/v1")
        self.assertNotEqual(read_json("deployment.json")["topic_root"], "dt/edges")

    def test_recorded_video_profiles_explicitly_opt_into_v2_calibration(self):
        from apps.edge_client.inference import parse_args

        for edge in ("edge_1", "edge_2"):
            args = parse_args(["--runtime-config", str(PROFILE_DIR / f"{edge}.json")])
            self.assertTrue(args.dataset)
            self.assertTrue(args.allow_dataset_calibration_override)
            self.assertIsNone(args.camera_config)
            self.assertEqual(
                Path(args.example_folder),
                ROOT / f"apps/edge_client/data/data_0812_1_{edge}",
            )

    def test_dataset_calibration_override_is_opt_in_and_keeps_workspace_guard(self):
        from contextlib import redirect_stdout
        import hashlib
        import io
        from types import SimpleNamespace
        from apps.edge_client.inference import validate_dataset_spatial_context

        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "calibration_result_test.json"
            original = '{"test": "recorded calibration"}'
            source.write_text(original)
            context = SimpleNamespace(calibration_path=Path(folder) / "applied.json",
                                      calibration_sha256="different-calibration")
            with self.assertRaisesRegex(ValueError, "does not match"):
                validate_dataset_spatial_context(folder, context)
            with self.assertRaisesRegex(ValueError, "requires --deployment"):
                validate_dataset_spatial_context(folder, None, allow_calibration_override=True)
            output = io.StringIO()
            with redirect_stdout(output):
                validate_dataset_spatial_context(folder, context, allow_calibration_override=True)
            self.assertIn("WARNING: explicit dataset calibration override", output.getvalue())
            self.assertIn("applied_sha256=different-calibration", output.getvalue())
            self.assertEqual(source.read_text(), original)
            context.calibration_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
            output = io.StringIO()
            with redirect_stdout(output):
                validate_dataset_spatial_context(folder, context)
            self.assertEqual(output.getvalue(), "")

    def test_dataset_override_cannot_be_enabled_for_live_input(self):
        from contextlib import redirect_stderr
        import io
        from apps.edge_client.inference import parse_args

        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--allow-dataset-calibration-override"])
        args = parse_args(["--runtime-config", str(PROFILE_DIR / "edge_1.json"),
                           "--no-allow-dataset-calibration-override"])
        self.assertFalse(args.allow_dataset_calibration_override)

    def test_calibration_snapshot_contains_only_spatial_camera_fields(self):
        snapshot = read_json(read_json("deployment.json")["calibration_result"])
        keys = {
            "camera_id", "camera_matrix", "distortion_coefficients",
            "undistorted_camera_matrix", "world_to_camera", "camera_to_world",
            "position_m",
        }
        self.assertEqual(
            [c["camera_id"] for c in snapshot["cameras"]],
            [f"camera/{i}" for i in range(1, 9)],
        )
        for camera in snapshot["cameras"]:
            # Calibration results may keep the capture file name, never a path.
            self.assertEqual(set(camera) - {"source_image"}, keys)
            self.assertNotIn("/", camera.get("source_image", ""))

    def test_snapshot_matches_local_v2_when_available(self):
        path = ROOT / "apps/edge_client/config/cameras.local_v2.yaml"
        if not path.is_file():
            self.skipTest("Private local camera config is not installed")
        local = {
            int(c["id"]): c for c in yaml.safe_load(path.read_text())["CAMERAS"]
        }
        reference = read_json("deployment.json")["calibration_result"]
        if Path(reference).name != "calibration.from_cameras_v2.json":
            self.skipTest("Profile does not use the cameras_v2 calibration snapshot")
        snapshot = read_json(reference)
        for camera in snapshot["cameras"]:
            number = int(camera["camera_id"].split("/")[-1])
            if number not in local:
                continue  # An edge may install only its assigned four sources.
            for key in ("camera_matrix", "distortion_coefficients", "undistorted_camera_matrix"):
                self.assertEqual(camera[key], local[number]["intrinsic"][key])
            for key in ("world_to_camera", "camera_to_world", "position_m"):
                self.assertEqual(camera[key], local[number]["extrinsic"][key])

    def test_launcher_outside_repository_and_cli_forwarding(self):
        with tempfile.TemporaryDirectory() as folder:
            for role, application in (("server", "server_worker"), ("edge_1", "edge_client")):
                result = subprocess.run(
                    ["bash", str(PROFILE_DIR / "run.sh"), role,
                     "tcp/127.0.0.1:7447", "--validate-only"],
                    cwd=folder, env={**os.environ, "PYTHON_BIN": "/bin/echo"},
                    capture_output=True, text=True, check=True,
                )
                self.assertIn(f"apps/{application}/inference.py", result.stdout)
                self.assertIn(str(PROFILE_DIR / f"{role}.json"), result.stdout)
                self.assertIn("--zenoh-endpoint tcp/127.0.0.1:7447 --validate-only", result.stdout)

    def test_actual_edge_server_camera_and_workspace_contracts(self):
        from copy import deepcopy
        import numpy as np
        from dt_common.calibration.identity import calibration_digest
        from dt_common.spatial.workspace import build_edge_workspace, load_spatial_context
        from apps.edge_client.src.root.core.config import config as edge_config
        from apps.edge_client.src.root.core.config import update_config as update_edge
        from apps.edge_client.src.utils.calibration import CalibrationData
        from apps.server_worker.src.pose.core.config import config as server_config
        from apps.server_worker.src.pose.core.config import update_config as update_server
        from apps.server_worker.src.utils.edgemetadata import EdgeMetadataLoader

        manifest_path = PROFILE_DIR / "deployment.json"
        scene = read_json("deployment.json")["scene"]
        for key in ("ground_usd", "ground_cache"):
            if not (PROFILE_DIR / scene[key]).is_file():
                self.skipTest(f"Local spatial asset not installed: {key}")
        update_edge(str(ROOT / "apps/edge_client/config/cam4_posenet.yaml"))
        update_server(str(ROOT / "apps/server_worker/config/cam4_posenet.yaml"))
        context = load_spatial_context(manifest_path)
        server = EdgeMetadataLoader(server_config, path=manifest_path)
        for edge_id, ids in context.edge_camera_ids.items():
            cfg = deepcopy(edge_config)
            CalibrationData(cfg, calibration_path=context.calibration_path,
                            camera_ids=ids, world_origin_m=context.world_origin_m)
            edge_ws = build_edge_workspace(
                context, edge_id, cfg.CAMS, cfg.NETWORK.IMAGE_SIZE_ORIG,
                cfg.MULTI_PERSON.SPACE_SIZE, cfg.MULTI_PERSON.INITIAL_CUBE_SIZE,
            )
            server_ws = build_edge_workspace(
                context, edge_id, server[edge_id].cams,
                server_config.NETWORK.IMAGE_SIZE_ORIG,
                server_config.MULTI_PERSON.SPACE_SIZE,
                server_config.MULTI_PERSON.INITIAL_CUBE_SIZE,
            )
            self.assertEqual(edge_ws.workspace_id, server_ws.workspace_id)
            self.assertTrue(np.array_equal(edge_ws.valid_mask, server_ws.valid_mask))
            self.assertEqual(calibration_digest(cfg.CAMS), calibration_digest(server[edge_id].cams))

    def test_launcher_rejects_missing_endpoint_and_invalid_role(self):
        for args in ([], ["edge_1"], ["edge_1", "--validate-only"], ["edge_3", "tcp/localhost:7447"]):
            result = subprocess.run(
                ["bash", str(PROFILE_DIR / "run.sh"), *args],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
