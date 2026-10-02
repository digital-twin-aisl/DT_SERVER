# SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
# SPDX-License-Identifier: LGPL-2.1-or-later
import unittest
from unittest.mock import patch
import numpy as np
from apps.calibration_worker.inference import (
    bundle_adjustment_limits, _ba_selected_image_indices, _ba_candidate_pairs, build_argument_parser,
    run_bundle_adjustment, _BundleTrack,
)


class BALimitsTests(unittest.TestCase):
    def test_unobserved_views_are_reported_fixed_not_claimed_as_refined(self):
        points=[np.array([i*.1,(i%3)*.2,4.]) for i in range(8)]
        tracks=[_BundleTrack(tuple((j,p[:2]/p[2]) for j in (0,2,3)),p) for p in points]
        predictions=dict(images=np.zeros((5,2,2,3)),depth=np.ones((5,2,2)),
            depth_conf=np.ones((5,2,2)),extrinsics=np.tile(np.eye(4)[:3],(5,1,1)),
            intrinsics=np.tile(np.eye(3),(5,1,1)))
        with patch('apps.calibration_worker.inference._build_bundle_tracks',return_value=tracks) as build:
            result=run_bundle_adjustment(predictions,cctv_count=2,min_depth_confidence=1.)
        self.assertEqual(len(build.call_args.args[0]),5)
        self.assertEqual(result.selected_image_count,5)
        self.assertEqual(result.unobserved_image_indices,(1,4))
        self.assertIn(1,result.fixed_cctv_indices)
        self.assertNotIn(1,result.variable_image_indices)
        self.assertNotIn(4,result.variable_image_indices)
        self.assertTrue(result.solver_converged)
        np.testing.assert_array_equal(result.extrinsics,predictions['extrinsics'])

    def test_90_reference_frames_are_not_silently_subsampled(self):
        args=build_argument_parser().parse_args(['--reference-sample-count','90','--max-images','98','--use_ba'])
        limits=bundle_adjustment_limits(98,8,max_images=args.ba_max_images,max_tracks=args.ba_max_tracks,
            max_iterations=args.ba_max_iterations)
        self.assertEqual(limits,dict(max_images=98,max_tracks=919,max_iterations=300))
        np.testing.assert_array_equal(_ba_selected_image_indices(98,8,limits['max_images']),np.arange(98))
        pairs=set(_ba_candidate_pairs(98,8))
        self.assertTrue(all((c,r) in pairs for c in range(8) for r in range(8,98)))

    def test_explicit_caps_preserve_all_cctv_and_spread_reference_views(self):
        limits=bundle_adjustment_limits(98,8,max_images=32,max_tracks=300,max_iterations=100)
        selected=_ba_selected_image_indices(98,8,limits['max_images'])
        self.assertEqual(len(selected),32);np.testing.assert_array_equal(selected[:8],np.arange(8))
        self.assertEqual(selected[-1],97);self.assertEqual(limits['max_tracks'],300)

    def test_invalid_budgets_rejected(self):
        for options in [dict(max_images=8),dict(max_images=0),dict(max_tracks=7),dict(max_iterations=0)]:
            with self.assertRaises(ValueError):bundle_adjustment_limits(98,8,**options)


if __name__=='__main__':unittest.main()
