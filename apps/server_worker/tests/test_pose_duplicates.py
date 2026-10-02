# SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
# SPDX-License-Identifier: LGPL-2.1-or-later
import json
from dataclasses import replace
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT),str(ROOT/'packages/dt_common/src')]
import numpy as np
import torch
from dt_common.contracts.scene import PersonEntity,RootOutput,PoseOutput
from apps.server_worker.domain.pose_duplicates import PoseDuplicateFilter,PoseDuplicateConfig
from apps.server_worker.domain.root_tracks import RootTracks


def person(index=0,x=0.,edge='edge_1',confidence=.7,root_x=None):
    joints = np.array([[0,0,1400],[0,0,1600],[0,0,900],[-200,0,1400],[-300,0,1150],[-350,0,900],
                       [-120,0,900],[-120,0,500],[-120,0,50],[200,0,1400],[300,0,1150],[350,0,900],
                       [120,0,900],[120,0,500],[120,0,50]],float)
    joints[:,0] += x
    return PersonEntity(index+1,2,RootOutput(edge,index,[x if root_x is None else root_x,0.,900.],confidence,0.),
                        PoseOutput('voxelpose_15j_xyz',joints.tolist()))


def projection(pose,cid):
    return np.asarray(pose)[:,[0,2]]*.1+400,np.ones(15)


def observations(*people):
    detections = [dict(keypoints=np.c_[projection(p.pose.joints,'1')[0],np.ones(15)].tolist()) for p in people]
    return {'1':detections,'2':detections}


class PoseDuplicateTests(unittest.TestCase):
    def filter(self,people,obs=None):
        return PoseDuplicateFilter(projection).filter_people(people,observations(people[0]) if obs is None else obs,0.)

    def test_same_edge_duplicate_uses_pose_not_raw_root(self):
        a,b = person(),person(1,50,root_x=650)
        kept,report = self.filter([a,b])
        self.assertEqual(kept,[a]);self.assertEqual(report['rejected_people'],1)
        self.assertEqual(report['rejected'][0]['same_views'],['1','2'])

    def test_cross_edge_also_suppressed(self):
        a,b = person(),person(0,50,edge='edge_2')
        self.assertEqual(self.filter([a,b])[0],[a])

    def test_world_coordinates_not_root_aligned_shape(self):
        a,b = person(),person(1,400)
        self.assertEqual(self.filter([a,b])[0],[a,b])

    def test_no_evidence_keeps_both(self):
        a,b = person(),person(1,50)
        kept,report = self.filter([a,b],{})
        self.assertEqual(kept,[a,b]);self.assertEqual(report['pairs'][0]['reason'],'insufficient_evidence')

    def test_one_shared_view_is_insufficient(self):
        a,b = person(),person(1,50);obs = observations(a);obs.pop('2')
        self.assertEqual(self.filter([a,b],obs)[0],[a,b])

    def test_close_real_people_with_distinct_2d_instances_survive(self):
        a,b = person(),person(1,150)
        kept,report = self.filter([a,b],observations(a,b))
        self.assertEqual(kept,[a,b]);self.assertEqual(report['pairs'][0]['reason'],'different_2d_people')

    def test_one_contrary_view_vetoes_other_shared_views(self):
        a,b = person(),person(1,150);obs = observations(a)
        obs['3'] = observations(a,b)['1']
        self.assertEqual(self.filter([a,b],obs)[0],[a,b])

    def test_ambiguous_duplicate_detector_boxes_do_not_prove_agreement(self):
        a,b = person(),person(1,50);obs = observations(a,a)
        self.assertEqual(self.filter([a,b],obs)[0],[a,b])

    def test_quality_can_override_higher_root_confidence(self):
        a,b = person(confidence=.4),person(1,50,confidence=.99)
        self.assertEqual(self.filter([a,b])[0],[a])

    def test_no_transitive_chain_suppression(self):
        a,b,c = person(),person(1,150),person(2,300)
        self.assertEqual(self.filter([a,b,c])[0],[a,c])

    def test_no_mutation_and_json_finite(self):
        a,b = person(),person(1,50);old = json.dumps([p.to_dict() for p in [a,b]])
        kept,report = self.filter([a,b])
        self.assertIs(kept[0],a);self.assertEqual(old,json.dumps([p.to_dict() for p in [a,b]]))
        json.dumps(report,allow_nan=False)

    def test_unsupported_or_missing_pose_kept(self):
        a,b = person(),replace(person(1),pose=None)
        self.assertEqual(self.filter([a,b])[0],[a,b])

    def test_missing_joints_are_not_negative_evidence(self):
        a,b = person(),person(1,50);obs = observations(a)
        for ds in obs.values():
            for det in ds:
                for k in det['keypoints'][4:]:k[2] = 0
        self.assertEqual(self.filter([a,b],obs)[0],[a,b])

    def test_history_only_breaks_small_quality_difference(self):
        a,b = person(root_x=0),person(1,10,root_x=650)
        f = PoseDuplicateFilter(projection)
        f.filter_people([b],observations(b),0.)
        self.assertEqual(f.filter_people([a,b],observations(a),.03)[0],[b])
        # Expired continuity is not a permanent lock on an inferior candidate.
        self.assertEqual(f.filter_people([a,b],observations(a),1.)[0],[a])

    def test_timestamp_reset_and_duplicate_keys_fail(self):
        f = PoseDuplicateFilter(projection);p = person()
        f.filter_people([p],observations(p),0.)
        with self.assertRaises(ValueError):f.filter_people([p],observations(p),0.)
        with self.assertRaises(ValueError):self.filter([p,p])

    def test_invalid_config(self):
        for kw in [dict(max_pelvis_mm=0),dict(max_joint_median_mm=np.nan),dict(min_joints=14),
                   dict(min_joint_confidence=2),dict(min_shared_views=1.5)]:
            with self.subTest(kw=kw),self.assertRaises(ValueError):PoseDuplicateConfig(**kw)

    def test_legacy_root_suppression_disable_protects_exactly_coincident_roots(self):
        roots = torch.tensor([[[0.,0.,900.,0.,.8]],[[0.,0.,900.,0.,.7]]])
        default = RootTracks().assign([[None],[None]],roots,['edge_1','edge_2'],[0.,0.])
        enabled = RootTracks(duplicate_distance_mm=0).assign([[None],[None]],roots,['edge_1','edge_2'],[0.,0.])
        self.assertIsNone(default[1][0]);self.assertIsNotNone(enabled[1][0])
        self.assertNotEqual(enabled[0][0],enabled[1][0])


if __name__ == '__main__':unittest.main()
