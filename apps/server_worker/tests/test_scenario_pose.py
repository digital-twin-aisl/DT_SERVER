# SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
# SPDX-License-Identifier: LGPL-2.1-or-later
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from apps.server_worker.tools.scenario_pose import sample_roots, choose_edge, anchor_pose, packet_heatmaps
from apps.server_worker.tools.scenario_skeleton import Track
from dt_common.calibration.identity import calibration_digest
import numpy as np
import torch


class ScenarioPoseTests(unittest.TestCase):
    def test_csv_identity_interpolation_height_and_no_extrapolation(self):
        track = Track(0, np.array([1.,2.]), np.array([[10.,20.],[30.,40.]]), [{'rank_selected':0},{'rank_selected':1}])
        ground = SimpleNamespace(height_mm=lambda x,y: 123.)
        self.assertEqual(sample_roots([track], .9, ground, 900), [])
        self.assertEqual(sample_roots([track], 2.1, ground, 900), [])
        person = sample_roots([track], 1.5, ground, 900)[0]
        self.assertEqual(person['id'], 0)
        np.testing.assert_array_equal(person['root'], [20.,30.,1023.])
        self.assertEqual(person['flags']['rank_selected'], 0)

    def test_anchor_only_translates_and_preserves_relative_pose(self):
        raw = np.arange(45).reshape(15,3).astype(float)
        root = np.array([83000.123,5400.456,1900.789])
        result, shift = anchor_pose(raw, root)
        np.testing.assert_array_equal(result[2], root)
        np.testing.assert_allclose(result-result[2], raw-raw[2], atol=1e-10)
        np.testing.assert_array_equal(shift, root-raw[2])
        np.testing.assert_array_equal(raw, np.arange(45).reshape(15,3))
        with self.assertRaises(ValueError): anchor_pose(np.zeros((17,3)), root)
        with self.assertRaises(ValueError): anchor_pose(np.full((15,3), np.nan), root)

    def test_edge_selection_is_single_stable_and_visibility_checked(self):
        def scene(views):
            return SimpleNamespace(cameras=[], visible=lambda p,c: np.array([views]),
                workspace=SimpleNamespace(footprint_xy_mm=np.array([[0,0],[4,0],[4,4],[0,4]])))
        scenes = {'edge_1':scene(4), 'edge_2':scene(4)}
        edge, info = choose_edge([2,2,900], scenes, 'edge_1')
        self.assertEqual(edge, 'edge_1'); self.assertTrue(info['inside_edge_aoi'])
        self.assertIsNone(choose_edge([2,2,900], {'edge_1':scene(1)})[0])

    def test_packet_heatmaps_ignore_rootnet_and_preserve_normalization(self):
        cameras = [dict(id=i,R=np.eye(3),T=np.zeros((3,1)),fx=1,fy=1,cx=0,cy=0,k=np.zeros(3),p=np.zeros(2)) for i in range(4)]
        scene = SimpleNamespace(edge_id='edge_1', cameras=cameras, heatmap_size=np.array([4,2]))
        frame = dict(edge_id='edge_1', sequence=3, timestamp_kind='dataset_relative', time=.1,
            camera_ids=list(range(4)), frame_ids=[3]*4, frame_timestamps=[.1]*4,
            calibration_digest=calibration_digest(cameras),
            allheatmaps=[np.full((1,15,2,4), 255, np.uint8) for _ in range(4)], roots=np.full((1,10,5), np.nan))
        first = packet_heatmaps(frame, scene, 3, .1)
        del frame['roots']
        second = packet_heatmaps(frame, scene, 3, .1)
        for a,b in zip(first,second):
            self.assertTrue(torch.equal(a,b)); self.assertTrue(torch.equal(a,torch.ones_like(a)))
        frame['calibration_digest'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'calibration differs'): packet_heatmaps(frame, scene, 3, .1)


if __name__ == '__main__': unittest.main()
