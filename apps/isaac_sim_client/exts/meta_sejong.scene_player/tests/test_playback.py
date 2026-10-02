# SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
# SPDX-License-Identifier: LGPL-2.1-or-later
import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

EXTENSION = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXTENSION))
sys.path.insert(0, str(ROOT / "packages/dt_common/src"))

from meta_sejong.scene_player.playback import Playback, SceneRecording, validate_scene
from apps.server_worker.src.utils.scene_recording import SceneOutputRecorder, SceneRecordingError


def scene(timestamp=0.0, people=None):
    return {"schema_version": 1, "timestamp": timestamp, "sync_spread_seconds": 0.0,
            "coordinate_system": {"frame": "USD world", "up_axis": "Z", "unit": "millimetre"},
            "people": [] if people is None else people}


def person(identity=1, pose=True):
    return {"global_id": identity, "lod": 2 if pose else 1,
            "root": {"position": [80000.0, 6000.0, 900.0], "confidence": 0.9,
                     "edge_id": "edge_1", "timestamp": 0.0, "candidate_index": 0},
            "pose": {"joint_format": "voxelpose_15j_xyz",
                     "joints": [[80000.0 + i * 10, 6000.0, 900.0 + i * 30] for i in range(15)]}
                    if pose else None}


class InterfaceTextTests(unittest.TestCase):
    def test_extension_text_literals_use_ascii_for_kit_font_compatibility(self):
        package = EXTENSION / "meta_sejong" / "scene_player"
        for path in package.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    with self.subTest(file=path.name, line=node.lineno):
                        self.assertTrue(node.value.isascii(), repr(node.value))


class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "session.jsonl"

    def record(self, scenes):
        with SceneOutputRecorder(self.path) as writer:
            for value in scenes:
                writer.write_payload(json.dumps(value, allow_nan=False).encode())
        return SceneRecording.load(self.path)

    def test_server_writer_roundtrip_including_empty_scene_and_runtime(self):
        values = [scene(12, [person()]), scene(12.5, [person(2, False)]), scene(13)]
        values[0]["runtime"] = {"timestamp_kind": "dataset_relative", "input_status": {}}
        data = self.record(values)
        self.assertEqual(data.duration, 1.0)
        self.assertEqual([data.frame(i) for i in range(3)], values)

    def test_existing_recording_is_never_overwritten_or_appended(self):
        self.record([scene()])
        before = self.path.read_bytes()
        with self.assertRaises(FileExistsError):
            SceneOutputRecorder(self.path)
        self.assertEqual(self.path.read_bytes(), before)

    def test_writer_failure_is_explicit(self):
        writer = SceneOutputRecorder(self.path)
        writer.close()
        with self.assertRaises(SceneRecordingError):
            writer.write_payload(b"{}")
        writer._stream = Mock()
        writer._stream.write.side_effect = OSError("disk full")
        with self.assertRaisesRegex(SceneRecordingError, "disk full"):
            writer.write_payload(b"{}")

    def test_partial_tail_recovery_but_invalid_middle_line_rejected(self):
        self.path.write_bytes(json.dumps(scene()).encode() + b'\n{"schema_version":')
        data = SceneRecording.load(self.path)
        self.assertEqual(len(data.offsets), 1)
        self.assertIn("incomplete final", data.warning)
        self.path.write_text(json.dumps(scene()) + '\ninvalid\n' + json.dumps(scene(1)))
        with self.assertRaisesRegex(ValueError, "Line 2"):
            SceneRecording.load(self.path)

    def test_empty_and_backwards_timestamps_rejected(self):
        self.path.write_text("")
        with self.assertRaisesRegex(ValueError, "no complete"):
            SceneRecording.load(self.path)
        self.path.write_text(json.dumps(scene(3)) + '\n' + json.dumps(scene(2)) + '\n')
        with self.assertRaisesRegex(ValueError, "backwards"):
            SceneRecording.load(self.path)

    def test_corrupt_geometry_and_wrong_units_rejected(self):
        values = []
        value = scene(); value["timestamp"] = float("nan"); values.append(value)
        value = scene(); value["coordinate_system"]["unit"] = "metre"; values.append(value)
        value = scene(0, [person()]); value["people"][0]["pose"]["joints"].pop(); values.append(value)
        value = scene(0, [person()]); value["people"][0]["root"]["position"][0] = float("inf"); values.append(value)
        values.append(scene(0, [person(), person()]))
        for value in values:
            with self.assertRaises(ValueError):
                validate_scene(value)

    def test_play_pause_speed_seek_loop_and_end(self):
        player = Playback(self.record([scene(100), scene(101), scene(103)]))
        player.play(); player.advance(0.5)
        self.assertEqual(player.index, 0)
        player.playing = False; player.advance(10)
        self.assertEqual(player.elapsed, 0.5)
        player.speed = 2; player.play(); player.advance(0.25)
        self.assertEqual(player.index, 1)
        player.advance(1)
        self.assertEqual(player.index, 2)
        self.assertFalse(player.playing)
        player.play(); self.assertEqual(player.index, 0)
        player.loop = True; player.advance(2)
        self.assertTrue(player.playing)
        self.assertAlmostEqual(player.elapsed, 1)
        player.seek_frame(-10); self.assertEqual(player.index, 0)
        player.seek_frame(500); self.assertEqual(player.index, 2)

    def test_equal_timestamps_and_single_scene(self):
        data = self.record([scene(0), scene(0, [person()]), scene(1)])
        player = Playback(data)
        player.play(); player.advance(0)
        self.assertEqual(player.index, 1)
        single = Playback(SceneRecording(self.path, [data.offsets[0]], [0]))
        single.loop = True; single.play(); single.advance(10)
        self.assertEqual(single.index, 0)
        self.assertFalse(single.playing)

    def test_speed_and_dt_validation(self):
        player = Playback(self.record([scene(), scene(1)]))
        player.play()
        for speed in (0, -1, 9, float("nan")):
            player.speed = speed
            with self.assertRaises(ValueError): player.advance(1)
        player.speed = 1
        with self.assertRaises(ValueError): player.advance(-1)

    def test_cancelled_indexing_exits_before_reading(self):
        import threading
        self.record([scene()])
        cancelled = threading.Event()
        cancelled.set()
        with self.assertRaisesRegex(ValueError, "cancelled"):
            SceneRecording.load(self.path, cancelled)

    def test_runtime_profile_resolves_recording_path_and_cli_overrides(self):
        from apps.server_worker.inference import get_parser
        from dt_common.runtime_config import parse_runtime_args
        profile = Path(self.temp.name) / "profile.json"
        profile.write_text(json.dumps({"schema_version": 1, "arguments": {"scene_recording": "out/run.jsonl"}}))
        args = parse_runtime_args(get_parser(), ["--runtime-config", str(profile)])
        self.assertEqual(args.scene_recording, str(Path(self.temp.name) / "out/run.jsonl"))
        args = parse_runtime_args(get_parser(), ["--runtime-config", str(profile), "--scene-recording", "other.jsonl"])
        self.assertEqual(args.scene_recording, "other.jsonl")


class RendererTests(unittest.TestCase):
    def setUp(self):
        try:
            from pxr import Usd, UsdGeom
            from meta_sejong.scene_player.renderer import SceneRenderer
        except ImportError:
            self.skipTest("USD Python bindings are not installed")
        self.stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.stage, 0.01)
        # A transformed /World must not shift playback world coordinates.
        from pxr import Gf
        UsdGeom.Xform.Define(self.stage, "/World").AddTranslateOp().Set(Gf.Vec3d(100, 0, 0))
        self.before = self.stage.GetRootLayer().ExportToString()
        self.session_before = self.stage.GetSessionLayer().ExportToString()
        self.renderer = SceneRenderer(self.stage)
        self.addCleanup(self.renderer.close)

    def test_millimetres_to_stage_units_pose_and_root_fallback(self):
        from pxr import UsdGeom
        self.renderer.apply(scene(0, [person()]))
        path = self.renderer.root_path + "/Person_1"
        prim = self.stage.GetPrimAtPath(path)
        world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0).ExtractTranslation()
        self.assertEqual(list(world), [8000, 600, 90])
        skeleton = UsdGeom.Imageable(self.stage.GetPrimAtPath(path + "/Skeleton"))
        capsule = UsdGeom.Imageable(self.stage.GetPrimAtPath(path + "/Root"))
        self.assertNotEqual(skeleton.ComputeVisibility(), UsdGeom.Tokens.invisible)
        self.assertEqual(capsule.ComputeVisibility(), UsdGeom.Tokens.invisible)
        self.renderer.apply(scene(1, [person(pose=False)]))
        self.assertEqual(skeleton.ComputeVisibility(), UsdGeom.Tokens.invisible)
        self.assertNotEqual(capsule.ComputeVisibility(), UsdGeom.Tokens.invisible)

    def test_empty_scene_hides_people_and_backward_seek_restores_them(self):
        from pxr import UsdGeom
        data = scene(0, [person()])
        self.renderer.apply(data)
        prim = self.stage.GetPrimAtPath(self.renderer.root_path + "/Person_1")
        self.renderer.apply(scene(1))
        self.assertEqual(UsdGeom.Imageable(prim).ComputeVisibility(), UsdGeom.Tokens.invisible)
        self.renderer.apply(data)
        self.assertNotEqual(UsdGeom.Imageable(prim).ComputeVisibility(), UsdGeom.Tokens.invisible)

    def test_cleanup_preserves_original_root_and_session_layers(self):
        self.renderer.apply(scene(0, [person()]))
        root = self.renderer.root_path
        self.assertEqual(self.stage.GetRootLayer().ExportToString(), self.before)
        self.renderer.close()
        self.assertFalse(self.stage.GetPrimAtPath(root).IsValid())
        self.assertEqual(self.stage.GetRootLayer().ExportToString(), self.before)
        self.assertEqual(self.stage.GetSessionLayer().ExportToString(), self.session_before)


if __name__ == "__main__":
    unittest.main()
