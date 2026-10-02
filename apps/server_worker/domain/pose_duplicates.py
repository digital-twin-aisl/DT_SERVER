# SPDX-FileCopyrightText: 2025-2026 DT_SERVER contributors
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Conservative, absolute-world pose NMS with synchronized 2D instance evidence.

Does not move joints, average skeletons, or infer absence from a missing detection.
Call after ground rejection and before assigning new track IDs. Evidence is an
inference input, not an independent quality measurement.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np

@dataclass(frozen=True)
class PoseDuplicateConfig:
    max_pelvis_mm: float = 300.
    max_joint_median_mm: float = 200.
    max_torso_median_mm: float = 250.
    max_reprojection_px: float = 30.
    min_joint_confidence: float = .5
    min_joints: int = 5
    match_margin_px: float = 3.
    min_shared_views: int = 2
    distinct_view_veto: int = 1
    history_seconds: float = .15
    history_root_mm: float = 250.
    history_bonus_px: float = 3.

    def __post_init__(self):
        for key,value in asdict(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if self.min_joint_confidence > 1 or not 1 <= self.min_joints <= 13:
            raise ValueError('Invalid 2D joint confidence/count')
        for key in ['min_joints','min_shared_views','distinct_view_veto']:
            if type(getattr(self,key)) is not int:raise ValueError(f'{key} must be an integer')


def candidate_key(person):
    return f'{person.root.edge_id}:{person.root.candidate_index}'


def valid_pose(person):
    if person.pose is None or person.pose.joint_format != 'voxelpose_15j_xyz':return None
    pose = np.asarray(person.pose.joints,dtype=float)
    return pose if pose.shape == (15,3) and np.isfinite(pose).all() else None


def pose_distances(a,b):
    distances = np.linalg.norm(a-b,axis=1)
    return dict(pelvis_mm=float(distances[2]),joint_median_mm=float(np.median(distances)),
                torso_median_mm=float(np.median(distances[[0,3,6,9,12]])))


class PoseDuplicateFilter:
    def __init__(self, project, config=None):
        """project(pose_mm, camera_id) -> (original pixel xy, positive depth)."""
        self.project = project
        self.config = config or PoseDuplicateConfig()
        self.previous = []
        self.previous_time = -math.inf

    def match_views(self,pose,observations):
        matches = {}
        for cid,detections in sorted(observations.items()):
            xy,depth = self.project(pose,cid)
            xy,depth = np.asarray(xy),np.asarray(depth)
            if xy.shape != (15,2) or depth.shape != (15,):raise ValueError('Invalid pose projection')
            choices = []
            for index,detection in enumerate(detections):
                points = np.asarray(detection['keypoints'],float)
                if points.shape != (15,3) or not np.isfinite(points).all():raise ValueError('Invalid 2D instance')
                valid = (points[:,2] >= self.config.min_joint_confidence) & (depth > 0) & np.isfinite(xy).all(1)
                valid[[0,2]] = False  # Do not count virtual neck/pelvis twice.
                if valid.sum() < self.config.min_joints:continue
                errors = np.linalg.norm(xy-points[:,:2],axis=1)[valid]
                choices.append((float(np.median(errors)),index,int(valid.sum())))
            choices.sort()
            if not choices or choices[0][0] > self.config.max_reprojection_px:continue
            if len(choices)>1 and choices[1][0]-choices[0][0] < self.config.match_margin_px:continue
            error,index,count = choices[0]
            matches[str(cid)] = dict(detection=index,error_px=error,joints=count)
        return matches

    def geometry_ok(self,distances):
        c = self.config
        return (distances['pelvis_mm'] < c.max_pelvis_mm
                and distances['joint_median_mm'] < c.max_joint_median_mm
                and distances['torso_median_mm'] < c.max_torso_median_mm)

    def filter_people(self,people,observations,timestamp):
        if not math.isfinite(timestamp) or timestamp <= self.previous_time:
            raise ValueError('Pose NMS timestamps must strictly increase; reset at clip boundaries')
        keys = [candidate_key(p) for p in people]
        if len(keys) != len(set(keys)):raise ValueError('Duplicate input candidate key')
        poses = [valid_pose(p) for p in people]
        geometries = {}
        for i,a in enumerate(poses):
            if a is None:continue
            for j in range(i):
                if poses[j] is None:continue
                distances = pose_distances(a,poses[j])
                if self.geometry_ok(distances):geometries[j,i] = distances
        involved = {i for pair in geometries for i in pair}
        evidence = {i:self.match_views(poses[i],observations) for i in involved}
        near_pairs = []; duplicates = {}
        for (i,j),distances in geometries.items():
            a,b = evidence[i],evidence[j]
            common = a.keys() & b.keys()
            same = sorted(c for c in common if a[c]['detection'] == b[c]['detection'])
            different = sorted(c for c in common if a[c]['detection'] != b[c]['detection'])
            # Any unambiguous contrary view protects both people by default.
            reason = ('different_2d_people' if len(different) >= self.config.distinct_view_veto else
                      'same_2d_person' if len(same) >= self.config.min_shared_views else 'insufficient_evidence')
            info = dict(candidates=[keys[i],keys[j]],same_views=same,distinct_views=different,
                        reason=reason,**distances)
            near_pairs.append(info)
            if reason == 'same_2d_person':duplicates[i,j] = info
        quality = {}
        for i in involved:
            views = evidence[i]
            error = float(np.median([v['error_px'] for v in views.values()])) if views else None
            continuity = timestamp-self.previous_time <= self.config.history_seconds and any(
                p.root.edge_id == people[i].root.edge_id
                and np.linalg.norm(np.asarray(p.root.position)-people[i].root.position) < self.config.history_root_mm
                and np.linalg.norm(valid_pose(p)[2]-poses[i][2]) < self.config.max_pelvis_mm
                for p in self.previous if valid_pose(p) is not None)
            quality[i] = dict(supported_views=len(views),median_reprojection_px=error,continuity=bool(continuity))
        def rank(i):
            q = quality.get(i,dict(supported_views=0,median_reprojection_px=None,continuity=False))
            err = q['median_reprojection_px'] if q['median_reprojection_px'] is not None else 1e6
            adjusted = err-self.config.history_bonus_px*q['continuity']
            return (-q['supported_views'],adjusted,-float(people[i].root.confidence),keys[i])
        kept = []; rejected = []
        for i in sorted(range(len(people)),key=rank):
            winner = next((j for j in kept if (min(i,j),max(i,j)) in duplicates),None)
            if winner is None:
                kept.append(i)
            else:
                info = duplicates[min(i,winner),max(i,winner)]
                rejected.append(dict(info,removed=keys[i],kept=keys[winner],
                                     removed_quality=quality[i],kept_quality=quality[winner]))
        # Preserve original ordering/candidate indices. No transitive union: each
        # suppressed candidate must be directly justified by a retained winner.
        selected = [p for i,p in enumerate(people) if i in kept]
        self.previous = selected;self.previous_time = timestamp
        return selected,dict(config=asdict(self.config),input_people=len(people),kept_people=len(selected),
            rejected_people=len(rejected),pairs=near_pairs,rejected=rejected,
            scope='Absolute world 15-joint pose NMS before tracking; synchronized 2D instance agreement required; no coordinate changes')
