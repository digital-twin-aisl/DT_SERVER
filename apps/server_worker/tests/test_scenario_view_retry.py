# SPDX-FileCopyrightText: 2025-2026 DT_SERVER contributors
# SPDX-License-Identifier: LGPL-2.1-or-later
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from apps.server_worker.tools.scenario_pose import PoseOnly, anchor_pose
from apps.server_worker.tools.pose_view_retry import (
    RetryConfig, heatmap_peaks, score_views, select_drop, retry_once,
)
from apps.server_worker.src.pose.models.project_layer import ProjectLayer


class ViewRetryTests(unittest.TestCase):
    def setUp(self):
        self.config = RetryConfig()
        self.cameras = [dict(id=i * 2 + 2) for i in range(4)]
        self.scene = SimpleNamespace(cameras=self.cameras, heatmap_size=np.array([64, 64]),
            heatmap_points=lambda pose, camera: (pose[:, :2], np.ones(15, dtype=bool)),
            visible=lambda points, cameras: np.array([len(cameras)]))
        self.root = np.array([20., 20., 1000.])
        self.raw = np.tile(self.root, (15, 1))
        self.heatmaps = [torch.zeros(1, 15, 64, 64) for _ in range(4)]
        for i, hm in enumerate(self.heatmaps): hm[:, :, 20, 35 if i == 2 else 20] = .8
        self.evidence = heatmap_peaks(self.heatmaps, .1)

    def test_worst_view_by_heatmap_pixel_error(self):
        scores = score_views(self.scene, self.raw, self.evidence, self.config)
        self.assertEqual([s['median_error_px'] for s in scores], [0., 0., 15., 0.])
        self.assertEqual(select_drop(self.scene, self.root, scores, self.config), (2, 'retry'))

    def test_nearest_peak_not_other_person_global_argmax(self):
        self.heatmaps[0][:, :, 50, 50] = 1.
        scores = score_views(self.scene, self.raw, heatmap_peaks(self.heatmaps, .1), self.config)
        self.assertEqual(scores[0]['median_error_px'], 0.)

    def test_missing_evidence_and_offscreen_are_not_bad_views(self):
        self.heatmaps[2].zero_()
        self.heatmaps[2][:, :5, 20, 35] = .8  # Below the six-joint minimum.
        scores = score_views(self.scene, self.raw, heatmap_peaks(self.heatmaps, .1), self.config)
        self.assertIsNone(scores[2]['median_error_px'])
        self.assertEqual(select_drop(self.scene, self.root, scores, self.config)[1], 'below_threshold')
        outside = self.raw.copy(); outside[:, 0] = -1
        scores = score_views(self.scene, outside, self.evidence, self.config)
        self.assertTrue(all(s['valid_joints'] == 0 for s in scores))
        self.assertEqual(select_drop(self.scene, self.root, scores, self.config)[1], 'insufficient_heatmap_evidence')

    def test_leave_at_least_two_root_views(self):
        self.scene.visible = lambda points, cameras: np.array([1])
        scores = score_views(self.scene, self.raw, self.evidence, self.config)
        self.assertEqual(select_drop(self.scene, self.root, scores, self.config),
                         (None, 'fewer_than_two_remaining_visible_root_views'))

    def test_only_one_retry_even_when_result_still_bad(self):
        # Make the retry worse in all retained views, but keep the external root.
        retried = self.raw.copy(); retried[:, 0] += 10; retried[2] = self.root
        predict = Mock(return_value=(retried[None], np.array([.7])))
        raw, peak, report = retry_once(predict, anchor_pose, 'edge_1', self.heatmaps, self.scene,
                                      self.root, self.raw, .8, self.evidence, self.config)
        self.assertEqual(predict.call_count, 1)
        self.assertEqual(predict.call_args.kwargs, {'excluded_view': 2})
        self.assertEqual(report['retry_count'], 1)
        self.assertEqual(report['excluded_camera_id'], 6)
        self.assertTrue(report['retry_used'])
        self.assertGreater(report['retained_comparison']['retry_mean_px'],
                           report['retained_comparison']['initial_mean_px'])
        np.testing.assert_array_equal(raw, retried)
        np.testing.assert_array_equal(anchor_pose(raw, self.root)[0][2], self.root)

    def test_empty_retry_keeps_initial_and_does_not_retry_again(self):
        predict = Mock(return_value=(self.raw[None], np.array([0.])))
        raw, peak, report = retry_once(predict, anchor_pose, 'edge_1', self.heatmaps, self.scene,
                                      self.root, self.raw, .8, self.evidence, self.config)
        self.assertEqual(predict.call_count, 1)
        self.assertFalse(report['retry_used'])
        self.assertEqual(peak, .8)
        np.testing.assert_array_equal(raw, self.raw)

    def test_no_retry_below_threshold(self):
        predict = Mock()
        _, _, report = retry_once(predict, anchor_pose, 'edge_1', self.heatmaps, self.scene,
            self.root, self.raw, .8, self.evidence, RetryConfig(threshold_px=15.))
        predict.assert_not_called()
        self.assertEqual(report['retry_count'], 0)

    def test_matching_heatmap_camera_removal_and_restoration_on_error(self):
        runner = PoseOnly.__new__(PoseOnly)
        runner.device = 'cpu'; runner.peaks = {'edge_1': np.array([.8])}
        layer = SimpleNamespace(cams=self.cameras)
        calls = []
        class Model:
            project_layer = layer
            def __call__(model, maps, centers, indices):
                calls.append(([c['id'] for c in layer.cams], [float(h[0,0,0,0]) for h in maps]))
                if len(calls) == 3: raise RuntimeError('test failure')
                return torch.zeros(1, 15, 3)
        runner.models = {'edge_1': Model()}
        maps = [torch.full((1,15,2,2), float(i)) for i in range(4)]
        runner.predict('edge_1', maps, [self.root], excluded_view=1)
        self.assertEqual(calls[0], ([2,6,8], [0.,2.,3.]))
        self.assertIs(layer.cams, self.cameras)
        runner.predict('edge_1', maps, [self.root])
        self.assertEqual(calls[1], ([2,4,6,8], [0.,1.,2.,3.]))
        with self.assertRaises(RuntimeError): runner.predict('edge_1', maps, [self.root], excluded_view=2)
        self.assertIs(layer.cams, self.cameras)

    def test_actual_project_layer_normalizes_by_retained_views(self):
        cfg = SimpleNamespace(
            NETWORK=SimpleNamespace(IMAGE_SIZE=[16,16], IMAGE_SIZE_ORIG=[16,16], HEATMAP_SIZE=[16,16]),
            MULTI_PERSON=SimpleNamespace(SPACE_CENTER=[0,0,100], SPACE_SIZE=[2,2,2]),
            PICT_STRUCT=SimpleNamespace(CUBE_SIZE=[2,2,2], GRID_SIZE=[2,2,2]))
        camera = dict(R=np.eye(3), T=np.zeros((3,1)), fx=1., fy=1., cx=8., cy=8., k=np.zeros((3,1)), p=np.zeros((2,1)))
        layer = ProjectLayer(cfg, np.array([[1,0,0],[0,1,0]]), [camera]*4, 'posenet')
        maps = [torch.ones(1,15,16,16) for _ in range(3)] + [torch.zeros(1,15,16,16)]
        centers = torch.tensor([[0.,0.,100.]])
        indices = torch.zeros(1, dtype=torch.long)
        four, _ = layer.project_local_cubes(maps, centers, indices)
        layer.cams = [camera]*3
        three, _ = layer.project_local_cubes(maps[:3], centers, indices)
        torch.testing.assert_close(four, torch.full_like(four, .75), atol=1e-6, rtol=0)
        torch.testing.assert_close(three, torch.ones_like(three), atol=1e-6, rtol=0)

    def test_config_rejects_nonfinite_or_invalid_values(self):
        for kwargs in [dict(threshold_px=float('nan')), dict(threshold_px=0),
                       dict(peak_confidence=0), dict(peak_confidence=1.1), dict(min_joints=16)]:
            with self.assertRaises(ValueError): RetryConfig(**kwargs)


if __name__ == '__main__': unittest.main()
