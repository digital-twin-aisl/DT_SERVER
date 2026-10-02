# SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
# SPDX-License-Identifier: LGPL-2.1-or-later
"""One-shot camera rejection for the synchronous CSV-root replay runner.

All distances are in original heatmap pixels, not input-image pixels. A
multi-person heatmap has multiple peaks: use the nearest confident local maximum
for each projected joint, never the global argmax belonging to another person.
This is a consistency heuristic, not a camera calibration or pose accuracy test.
"""
from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RetryConfig:
    threshold_px: float = 6.0
    peak_confidence: float = 0.1
    min_joints: int = 6

    def __post_init__(self):
        if not np.isfinite(self.threshold_px) or self.threshold_px <= 0:
            raise ValueError('Retry threshold must be finite and positive')
        if not 0 < self.peak_confidence <= 1 or not 1 <= self.min_joints <= 15:
            raise ValueError('Invalid peak confidence or minimum joint count')

    def describe(self):
        return dict(asdict(self), max_retries=1, score_pose='CSV-pelvis-anchored output',
                    metric='median nearest confident 5x5 local-maximum distance in heatmap pixels',
                    minimum_remaining_visible_root_views=2)


def heatmap_peaks(heatmaps, confidence):
    """Compute once per edge/frame, then share between people and retry scores."""
    result = []
    for tensor in heatmaps:
        maps = tensor.detach().cpu().numpy()[0]
        joints = []
        for hm in maps:
            maxima = cv2.dilate(hm, np.ones((5, 5), np.uint8))
            y, x = np.nonzero((hm >= confidence) & (hm == maxima))
            joints.append(np.column_stack((x, y)).astype(np.float64))
        result.append(joints)
    return result


def score_views(scene, pose, evidence, config):
    scores = []
    for camera, peaks in zip(scene.cameras, evidence):
        xy, visible = scene.heatmap_points(pose, camera)
        xy = np.asarray(xy)
        visible = (np.asarray(visible, dtype=bool) & np.isfinite(xy).all(axis=1)
                   & (xy >= 0).all(axis=1) & (xy <= scene.heatmap_size - 1).all(axis=1))
        errors = [None] * len(pose)
        for joint, candidates in enumerate(peaks):
            if visible[joint] and len(candidates):
                errors[joint] = float(np.linalg.norm(candidates - xy[joint], axis=1).min())
        valid = [e for e in errors if e is not None]
        scores.append(dict(camera_id=int(camera['id']), valid_joints=len(valid),
                           median_error_px=float(np.median(valid)) if len(valid) >= config.min_joints else None,
                           joint_errors_px=errors))
    return scores


def select_drop(scene, root, scores, config):
    valid = [i for i, s in enumerate(scores) if s['median_error_px'] is not None]
    if not valid:
        return None, 'insufficient_heatmap_evidence'
    worst = max(valid, key=lambda i: scores[i]['median_error_px'])
    if scores[worst]['median_error_px'] <= config.threshold_px:
        return None, 'below_threshold'
    remaining = [c for i, c in enumerate(scene.cameras) if i != worst]
    if len(remaining) < 2 or int(scene.visible(np.asarray(root)[None], remaining)[0]) < 2:
        return None, 'fewer_than_two_remaining_visible_root_views'
    return worst, 'retry'


def retry_once(predict, anchor, edge, heatmaps, scene, root, raw, peak, evidence, config):
    """Call predict at most once; never recursively reject a second camera.

    predict(edge, heatmaps, roots, excluded_view=...) must remove the matching
    camera AND heatmap from projection/normalization, not zero one heatmap.
    """
    report = dict(retry_count=0, excluded_camera_id=None, retry_used=False)
    if peak <= 0:
        report['reason'] = 'empty_initial_cube'
        return raw, peak, report
    initial_pose, _ = anchor(raw, root)
    initial = score_views(scene, initial_pose, evidence, config)
    report['initial_views'] = initial
    excluded, report['reason'] = select_drop(scene, root, initial, config)
    if excluded is None:
        return raw, peak, report
    report.update(retry_count=1, excluded_camera_id=int(scene.cameras[excluded]['id']),
                  initial_raw_joints_mm=np.asarray(raw).tolist())
    predictions, peaks = predict(edge, heatmaps, [root], excluded_view=excluded)
    retry_raw, retry_peak = predictions[0], float(peaks[0])
    report.update(retry_raw_joints_mm=np.asarray(retry_raw).tolist(), retry_cube_peak=retry_peak)
    if retry_peak <= 0 or not np.isfinite(retry_raw).all():
        report['reason'] = 'invalid_retry_keep_initial'
        return raw, peak, report
    retry_pose, _ = anchor(retry_raw, root)
    final = score_views(scene, retry_pose, evidence, config)
    report.update(final_views=final, retry_used=True, reason='retried_once')
    # Compare identical visible/evidence-supported joints in retained cameras.
    # Simply dropping the worst camera would trivially lower an aggregate score.
    pairs = [(a, b) for i, (before, after) in enumerate(zip(initial, final)) if i != excluded
             for a, b in zip(before['joint_errors_px'], after['joint_errors_px'])
             if a is not None and b is not None]
    if pairs:
        report['retained_comparison'] = dict(joint_pairs=len(pairs),
            initial_mean_px=float(np.mean([p[0] for p in pairs])),
            retry_mean_px=float(np.mean([p[1] for p in pairs])))
    # User requested one retry, not best-of-N selection. Use the valid retry even
    # when this consistency metric worsens; retain diagnostics for comparison.
    return retry_raw, retry_peak, report


def compact_report(report):
    """Full per-joint/initial raw data live in diagnostics, not each scene entity."""
    return {k: ([{a: b for a, b in s.items() if a != 'joint_errors_px'} for s in v]
                if k in ('initial_views', 'final_views') else v)
            for k, v in report.items() if k not in ('initial_raw_joints_mm', 'retry_raw_joints_mm')}
