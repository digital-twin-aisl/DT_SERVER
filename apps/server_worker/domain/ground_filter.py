# SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Output-only height rejection in USD world Z-up millimetres.

Never snap a skeleton onto the floor. The supplied surface must describe the
same world frame as the inference output; a missing height is not zero metres.
"""
import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class GroundFilterConfig:
    enabled: bool = True
    max_root_mm: float = 1200.0
    max_feet_mm: float = 350.0
    missing_ground: str = 'keep'

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError('ground_filter must be a boolean')
        if (not math.isfinite(self.max_root_mm) or self.max_root_mm <= 0
                or not math.isfinite(self.max_feet_mm) or self.max_feet_mm < 0):
            raise ValueError('Ground limits must be finite; root > 0 and feet >= 0')
        if self.missing_ground not in ('keep', 'drop'):
            raise ValueError('Missing-ground policy must be keep or drop')


def add_ground_filter_arguments(parser):
    parser.add_argument('--ground-filter', action=argparse.BooleanOptionalAction, default=True,
                        help='Reject excessively elevated people in final scene output')
    parser.add_argument('--ground-max-root-mm', type=float, default=1200.,
                        help='Maximum root AND predicted pelvis clearance above local ground (mm)')
    parser.add_argument('--ground-max-feet-mm', type=float, default=350.,
                        help='Reject only if BOTH ankle clearances exceed this value (mm)')
    parser.add_argument('--ground-missing', choices=['keep', 'drop'], default='keep',
                        help='When a queried XY has no ground, keep with diagnostic or drop')


def make_ground_filter(args, ground):
    return GroundHeightFilter(ground, GroundFilterConfig(args.ground_filter,
        args.ground_max_root_mm, args.ground_max_feet_mm, args.ground_missing))


class GroundHeightFilter:
    def __init__(self, ground, config=None):
        self.config = config or GroundFilterConfig()
        self.ground = ground
        if self.config.enabled and ground is None:
            raise ValueError('Ground filtering requires a spatial ground surface; configure deployment or --no-ground-filter')

    def assess(self, person):
        """One decision per entity; checks raw root plus pelvis and ankle joints."""
        if not self.config.enabled:
            return dict(keep=True, reason='disabled')
        detail = dict(global_id=person.global_id, edge_id=person.root.edge_id,
                      keep=True, reason='accepted', missing_ground=False,
                      pose_checked=False, unsupported_pose_format=False)
        root = np.asarray(person.root.position, dtype=np.float64)
        if root.shape != (3,) or not np.isfinite(root).all():
            return dict(detail, keep=False, reason='invalid_root')
        points = [root]
        if person.pose is not None:
            if person.pose.joint_format == 'voxelpose_15j_xyz':
                joints = np.asarray(person.pose.joints, dtype=np.float64)
                if joints.shape != (15, 3) or not np.isfinite(joints).all():
                    return dict(detail, keep=False, reason='invalid_pose')
                # Panoptic/SelfPose/FVP 15-joint ordering: pelvis=2, ankles=8/14.
                points.extend(joints[[2, 8, 14]])
                detail['pose_checked'] = True
            else:
                detail['unsupported_pose_format'] = True
        points = np.asarray(points)
        heights = np.asarray(self.ground.heights_mm(points[:, :2]), dtype=np.float64)
        if heights.shape != (len(points),):
            raise ValueError('Ground query returned an unexpected shape')
        clearance = points[:, 2] - heights
        known = np.isfinite(clearance)
        detail['missing_ground'] = not bool(known.all())
        detail['root_clearance_mm'] = float(clearance[0]) if known[0] else None
        if len(points) == 4:
            detail['pelvis_clearance_mm'] = float(clearance[1]) if known[1] else None
            detail['ankle_clearance_mm'] = [float(h) if k else None for h,k in zip(clearance[2:],known[2:])]
        if known[0] and clearance[0] > self.config.max_root_mm:
            return dict(detail, keep=False, reason='root_too_high')
        if len(points) == 4:
            if known[1] and clearance[1] > self.config.max_root_mm:
                return dict(detail, keep=False, reason='pelvis_too_high')
            if known[2:].all() and (clearance[2:] > self.config.max_feet_mm).all():
                return dict(detail, keep=False, reason='both_feet_too_high')
        if detail['missing_ground']:
            return dict(detail, keep=self.config.missing_ground == 'keep', reason='missing_ground')
        return detail

    def filter_people(self, people):
        kept = []; rejected = []; counts = Counter(); unknown = []; unsupported = []
        for person in people:
            decision = self.assess(person)
            if decision.get('missing_ground'): unknown.append(person.global_id)
            if decision.get('unsupported_pose_format'): unsupported.append(person.global_id)
            if decision['keep']:
                kept.append(person)
            else:
                rejected.append(decision); counts[decision['reason']] += 1
        return kept, dict(config=asdict(self.config), input_people=len(people),
            kept_people=len(kept), rejected_people=len(rejected), reasons=dict(counts),
            rejected=rejected, missing_ground_ids=unknown, unsupported_pose_ids=unsupported,
            scope='Incoming observations before output cap/state; no coordinate modification')
