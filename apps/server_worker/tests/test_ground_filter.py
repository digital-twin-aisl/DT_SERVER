# SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
# SPDX-License-Identifier: LGPL-2.1-or-later
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO/'packages/dt_common/src')]
import numpy as np
from apps.server_worker.domain.ground_filter import (
    GroundFilterConfig, GroundHeightFilter, add_ground_filter_arguments, make_ground_filter,
)
from apps.server_worker.domain.scene_state import SceneState
from dt_common.contracts.scene import PersonEntity, RootOutput, PoseOutput, SceneOutput


def person(identity=1, root=900., pelvis=900., ankles=(80.,80.), pose=True):
    joints = np.tile([100.,200.,10900.], (15,1))
    joints[2,2] = 10000+pelvis; joints[8,2] = 10000+ankles[0]; joints[14,2] = 10000+ankles[1]
    return PersonEntity(identity, 2 if pose else 1,
        RootOutput('edge_1',identity,[100.,200.,10000+root],.9,1.),
        PoseOutput('voxelpose_15j_xyz',joints.tolist()) if pose else None)


class GroundTests(unittest.TestCase):
    def setUp(self):
        self.ground = SimpleNamespace(heights_mm=lambda xy: np.full(len(xy),10000.))
        self.filter = GroundHeightFilter(self.ground)

    def test_world_elevation_not_absolute_z_and_no_mutation(self):
        p = person(); before = json.dumps(p.to_dict())
        kept, report = self.filter.filter_people([p])
        self.assertIs(kept[0], p)
        self.assertEqual(json.dumps(p.to_dict()), before)
        self.assertEqual(report['rejected_people'],0)

    def test_excess_root_rejects_whole_person(self):
        self.assertEqual(self.filter.assess(person(root=1201))['reason'],'root_too_high')

    def test_pose_pelvis_also_checked(self):
        self.assertEqual(self.filter.assess(person(pelvis=1300))['reason'],'pelvis_too_high')

    def test_both_feet_high_rejected(self):
        result = self.filter.assess(person(ankles=(351.,450.)))
        self.assertFalse(result['keep']); self.assertEqual(result['reason'],'both_feet_too_high')

    def test_one_foot_swing_kept(self):
        self.assertTrue(self.filter.assess(person(ankles=(80.,800.)))['keep'])
        self.assertTrue(self.filter.assess(person(ankles=(800.,80.)))['keep'])

    def test_threshold_equality_kept(self):
        self.assertTrue(self.filter.assess(person(root=1200,pelvis=1200,ankles=(350,350)))['keep'])

    def test_each_ankle_queries_its_own_ground_on_steps(self):
        p = person(ankles=(800,80)); joints = np.array(p.pose.joints); joints[8,0] = 600
        p = replace(p,pose=PoseOutput('voxelpose_15j_xyz',joints.tolist()))
        stairs = SimpleNamespace(heights_mm=lambda xy: 10000+np.where(xy[:,0]>500,750,0))
        result = GroundHeightFilter(stairs).assess(p)
        self.assertEqual(result['ankle_clearance_mm'],[50.,80.]); self.assertTrue(result['keep'])

    def test_root_only_lod_is_filtered(self):
        self.assertFalse(self.filter.assess(person(root=1300,pose=False))['keep'])
        self.assertTrue(self.filter.assess(person(pose=False))['keep'])

    def test_missing_ground_is_not_zero_and_reported(self):
        unknown = SimpleNamespace(heights_mm=lambda xy: np.full(len(xy),np.nan))
        f = GroundHeightFilter(unknown)
        kept, report = f.filter_people([person()])
        self.assertEqual(len(kept),1); self.assertEqual(report['missing_ground_ids'],[1])
        f = GroundHeightFilter(unknown,GroundFilterConfig(missing_ground='drop'))
        kept, report = f.filter_people([person()])
        self.assertEqual(kept,[]); json.dumps(report,allow_nan=False)

    def test_one_unknown_ankle_does_not_prove_floating(self):
        p = person(ankles=(450,450)); joints = np.array(p.pose.joints); joints[8,0] = 999
        p = replace(p,pose=PoseOutput('voxelpose_15j_xyz',joints.tolist()))
        partial = SimpleNamespace(heights_mm=lambda xy: np.where(xy[:,0]==999,np.nan,10000))
        result = GroundHeightFilter(partial).assess(p)
        self.assertTrue(result['keep']); self.assertTrue(result['missing_ground'])

    def test_known_root_violation_wins_over_unknown_ankle(self):
        partial = SimpleNamespace(heights_mm=lambda xy: np.array([10000,np.nan,np.nan,np.nan]))
        self.assertFalse(GroundHeightFilter(partial).assess(person(root=1300))['keep'])

    def test_invalid_coordinates_rejected(self):
        self.assertEqual(self.filter.assess(person(root=np.nan))['reason'],'invalid_root')
        self.assertEqual(self.filter.assess(person(ankles=(np.inf,80)))['reason'],'invalid_pose')

    def test_unsupported_format_does_not_guess_joint_indices(self):
        p = replace(person(),pose=PoseOutput('unknown',[[0,0,0]]))
        kept, report = self.filter.filter_people([p])
        self.assertEqual(kept,[p]); self.assertEqual(report['unsupported_pose_ids'],[1])

    def test_disabling_allows_no_ground_and_preserves_output(self):
        f = GroundHeightFilter(None,GroundFilterConfig(enabled=False))
        p = person(root=4000)
        self.assertEqual(f.filter_people([p])[0],[p])
        with self.assertRaises(ValueError): GroundHeightFilter(None)

    def test_invalid_options_rejected(self):
        for options in [dict(max_root_mm=0),dict(max_root_mm=np.nan),dict(max_feet_mm=-1),
                        dict(max_feet_mm=np.inf),dict(missing_ground='zero'),dict(enabled='false')]:
            with self.subTest(options=options), self.assertRaises(ValueError): GroundFilterConfig(**options)

    def test_cli_overrides(self):
        parser = argparse.ArgumentParser(); add_ground_filter_arguments(parser)
        default = make_ground_filter(parser.parse_args([]),self.ground)
        self.assertEqual(default.config.max_root_mm,1200)
        args = parser.parse_args(['--ground-max-root-mm','1350','--ground-max-feet-mm','400','--ground-missing','drop'])
        self.assertEqual(make_ground_filter(args,self.ground).config,GroundFilterConfig(True,1350,400,'drop'))
        self.assertFalse(make_ground_filter(parser.parse_args(['--no-ground-filter']),None).config.enabled)

    def test_state_does_not_resurrect_rejected_current_edge(self):
        state = SceneState(['edge_1','edge_2'])
        state.update(SceneOutput(1.,0.,[person()]),['edge_1'],1.,{1:2},10)
        kept, report = self.filter.filter_people([person(ankles=(450,450))])
        output = state.update(SceneOutput(1.1,0.,kept,{'ground_filter':report}),['edge_1'],1.1,{1:2},10)
        self.assertEqual(output.people,[])
        self.assertEqual(state.update(SceneOutput(1.2,0.,[]),[],1.2,{1:2},10).people,[])

    def test_server_filters_before_output_cap(self):
        import torch
        from apps.server_worker.inference import build_scene_output
        p1, p2 = person(root=1300), person(identity=2)
        roots = torch.tensor([[[*p1.root.position,0.,.99],[*p2.root.position,0.,.8]]])
        predicted = torch.tensor([[p1.pose.joints,p2.pose.joints]])
        scene = build_scene_output({'pred':predicted,'grid_centers':roots},roots,
            [[1,2]],{1:2,2:2},[1.],['edge_1'],0.,max_people=1,ground_filter=self.filter)
        self.assertEqual([p.global_id for p in scene.people],[2])
        self.assertEqual(scene.runtime['ground_filter']['rejected_people'],1)


if __name__ == '__main__': unittest.main()
