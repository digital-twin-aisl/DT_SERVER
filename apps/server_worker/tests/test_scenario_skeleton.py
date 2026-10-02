# SPDX-FileCopyrightText: 2025-2026 DT_SERVER contributors
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Physical constraints, CSV semantics, and safe offline recording generation."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import scenario_skeleton as demo

PLAYER = demo.PROJECT / "apps/isaac_sim_client/exts/meta_sejong.scene_player/meta_sejong/scene_player/playback.py"
spec = importlib.util.spec_from_file_location("scenario_test_player", PLAYER)
player = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = player
spec.loader.exec_module(player)


class FlatGround:
    def height_mm(self, x, y):
        return 14000.0


class ScenarioTests(unittest.TestCase):
    def test_selection_modes_hold_previous_row_and_switch_at_boundary(self):
        track = demo.Track(0, np.array([0., 1., 2.]),
            np.array([[0., 0.], [1000., 0.], [2000., 0.]]),
            [{"zone_selected": 0, "rank_selected": 1},
             {"zone_selected": 1, "rank_selected": 0},
             {"zone_selected": 0, "rank_selected": 1}])
        for t, zone in ((0., 0), (.999999, 0), (1., 1), (1.5, 1), (2., 0)):
            baseline = demo.synthetic_person(track, t, t, FlatGround())
            for column, selected in (("zone_selected", zone), ("rank_selected", 1-zone)):
                person = demo.synthetic_person(track, t, t, FlatGround(), column)
                self.assertEqual(person["global_id"], 0)
                self.assertEqual(person["root"], baseline["root"])
                self.assertEqual(person["lod"], 2 if selected else 1)
                self.assertEqual(person["pose"], baseline["pose"] if selected else None)
                self.assertEqual(person["scenario"]["selected"], selected)
                player.validate_scene({"schema_version": 1,
                    "coordinate_system": {"frame": "USD world", "up_axis": "Z", "unit": "millimetre"},
                    "timestamp": t, "people": [person]})

    def test_selection_rejects_missing_and_nonbinary_flags_before_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            path = folder / "tracks.csv"
            for value in ("", "2", "-1"):
                path.write_text(f"time_stamp,global_id,x_mm,y_mm,zone_selected\n0,0,0,0,{value}\n")
                args = argparse.Namespace(csv=path, fps=30., max_gap=1.01, time_offset=0.,
                    selection_column="zone_selected", output=folder / "out.jsonl")
                with self.assertRaisesRegex(ValueError, "zone_selected must be 0 or 1"):
                    demo.generate(args)
                self.assertFalse(args.output.exists())

    def test_rigid_limbs_and_ground_contact_over_walk_cycle(self):
        track = demo.Track(0, np.array([0., 4.]), np.array([[0., 0.], [4000., 0.]]), [{}, {}])
        last_stance = None
        for t in np.linspace(0., 4., 161):
            person = demo.synthetic_person(track, t, t, FlatGround())
            joints = np.array(person["pose"]["joints"])
            np.testing.assert_allclose(joints[2, :2], [t * 1000., 0.], atol=1e-8)
            self.assertGreaterEqual(min(joints[[8, 14], 2]), 14060. - 1e-8)
            for a, b, length in ((6, 7, 440), (7, 8, 440), (12, 13, 440),
                                 (13, 14, 440), (3, 4, 285), (4, 5, 260), (9, 10, 285), (10, 11, 260)):
                self.assertAlmostEqual(np.linalg.norm(joints[a] - joints[b]), length, places=6)
            if 1.05 < t < 1.45:
                if last_stance is not None:
                    np.testing.assert_allclose(joints[8], last_stance)
                last_stance = joints[8]

    def test_stationary_person_has_still_grounded_feet(self):
        track = demo.Track(4, np.array([0., 3.]), np.zeros((2, 2)), [{}, {}])
        poses = [demo.synthetic_person(track, t, t, FlatGround())["pose"]["joints"] for t in (0., 1., 3.)]
        np.testing.assert_allclose(poses[0], poses[1])
        np.testing.assert_allclose(poses[1], poses[2])
        self.assertEqual(poses[0][8][2], 14060.)

    def test_sloped_terrain_does_not_stretch_legs(self):
        class Slope:
            def height_mm(self, x, y):
                return 14000. + .25 * x
        track = demo.Track(0, np.array([0., 4.]), np.array([[0., 0.], [4000., 0.]]), [{}, {}])
        for t in np.linspace(0., 4., 101):
            joints = np.array(demo.synthetic_person(track, t, t, Slope())["pose"]["joints"])
            for hip, knee, ankle in ((6, 7, 8), (12, 13, 14)):
                self.assertAlmostEqual(np.linalg.norm(joints[hip] - joints[knee]), 440.)
                self.assertAlmostEqual(np.linalg.norm(joints[knee] - joints[ankle]), 440.)
                self.assertGreaterEqual(joints[ankle, 2] + 1e-8, Slope().height_mm(*joints[ankle, :2]) + 60.)

    def test_csv_bom_id_zero_gaps_and_duplicate_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tracks.csv"
            path.write_text("\ufefftime_stamp,global_id,x_mm,y_mm\n0,0,0,0\n1,0,1000,0\n4,0,4000,0\n")
            tracks = demo.read_tracks(path)
            self.assertEqual(len(tracks), 2)
            self.assertEqual(tracks[0].identity, 0)
            np.testing.assert_allclose(tracks[0].at(.5), [500, 0])
            path.write_text("time_stamp,global_id,x_mm,y_mm\n0,0,0,0\n0,0,1,1\n")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                demo.read_tracks(path)
            path.write_text("time_stamp,global_id,x_mm,y_mm\n0,0,nan,0\n")
            with self.assertRaisesRegex(ValueError, "non-finite"):
                demo.read_tracks(path)

    def test_recording_time_offset_gaps_and_existing_file_protection(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            csv = folder / "tracks.csv"
            csv.write_text("time_stamp,global_id,x_mm,y_mm\n2,0,0,0\n3,0,1000,0\n6,0,2000,0\n7,0,3000,0\n")
            cache = folder / "ground.npz"
            np.savez(cache, format_version=1, source_sha256="test", triangles_mm=np.array([
                [[-10000, -10000, 14000], [20000, -10000, 14000], [-10000, 20000, 14000]]]))
            reference = folder / "reference.jsonl"
            reference.write_text('{"timestamp":0}\n{"timestamp":10}\n')
            args = argparse.Namespace(csv=csv, ground_cache=cache, reference_recording=reference,
                output=folder / "out.jsonl", fps=2., max_gap=1.01, time_offset=1.)
            result = demo.generate(args)
            self.assertEqual(result["frames"], 21)
            recording = player.SceneRecording.load(args.output)
            self.assertEqual(len(recording.timestamps), 21)
            scenes = [json.loads(line) for line in args.output.read_text().splitlines()]
            active = [scene["timestamp"] for scene in scenes if scene["people"]]
            self.assertEqual(active, [3., 3.5, 4., 7., 7.5, 8.])
            self.assertTrue(scenes[7]["runtime"]["scenario"]["synthetic_pose"])
            self.assertEqual(scenes[7]["people"][0]["root"]["position"][0], 500.)
            original = args.output.read_bytes()
            with self.assertRaises(FileExistsError):
                demo.generate(args)
            self.assertEqual(args.output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
