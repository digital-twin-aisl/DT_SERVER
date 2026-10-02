# SPDX-FileCopyrightText: 2025-2026 DT_SERVER contributors
# SPDX-License-Identifier: LGPL-2.1-or-later
"""The synthetic sample must stay a valid, clearly-marked SceneOutput stream."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import generate_scene  # noqa: E402


class SyntheticSceneTests(unittest.TestCase):
    def test_generated_scenes_follow_the_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "scene.jsonl"
            generate_scene.main(["--output", str(output), "--people", "3",
                                 "--seconds", "2", "--fps", "5"])
            scenes = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual(len(scenes), 10)
        timestamps = [scene["timestamp"] for scene in scenes]
        self.assertEqual(timestamps, sorted(timestamps))
        x0, y0, x1, y1 = generate_scene.DEFAULT_AREA_MM
        for scene in scenes:
            self.assertEqual(scene["schema_version"], 1)
            self.assertEqual(scene["coordinate_system"]["unit"], "millimetre")
            self.assertTrue(scene["runtime"]["playback"])
            self.assertEqual(sorted(p["global_id"] for p in scene["people"]), [0, 1, 2])
            for person in scene["people"]:
                x, y, _ = person["root"]["position"]
                self.assertTrue(x0 <= x <= x1 and y0 <= y <= y1)
                if person["pose"] is not None:
                    self.assertEqual(person["lod"], 2)
                    self.assertEqual(person["pose"]["joint_format"], "voxelpose_15j_xyz")
                    self.assertEqual(len(person["pose"]["joints"]), 15)
                    self.assertEqual(person["pose"]["joints"][2], person["root"]["position"])

    def test_bundled_sample_is_valid(self):
        lines = (HERE / "synthetic_scene.jsonl").read_text().splitlines()
        self.assertGreater(len(lines), 0)
        for line in lines:
            self.assertTrue(json.loads(line)["runtime"]["playback"])


if __name__ == "__main__":
    unittest.main()
