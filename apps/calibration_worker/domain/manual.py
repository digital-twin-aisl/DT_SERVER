# SPDX-FileCopyrightText: 2025-2026 DT_SERVER contributors
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Pure geometry for manual extrinsic editing (USD metres, OpenCV cameras)."""
from copy import deepcopy
from datetime import datetime, timezone

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def rigid(value):
    m = np.asarray(value, dtype=np.float64)
    if (m.shape != (4, 4) or not np.isfinite(m).all()
            or np.max(np.abs(m[:3, 3])) > 1e6
            or not np.allclose(m[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(m[:3, :3].T @ m[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(m[:3, :3]), 1, atol=1e-5)):
        raise ValueError('Expected a finite rigid camera_to_world matrix (metres).')
    return m


def move_camera(base, translation, rotation):
    """World XYZ translation, camera-local XYZ Euler increments relative to base."""
    base = rigid(base)
    t, r = np.asarray(translation, dtype=float), np.asarray(rotation, dtype=float)
    if t.shape != (3,) or r.shape != (3,) or not np.isfinite(np.r_[t, r]).all():
        raise ValueError('Offsets must be finite XYZ vectors.')
    result = base.copy()
    result[:3, 3] += t
    result[:3, :3] = base[:3, :3] @ Rotation.from_euler('xyz', r, degrees=True).as_matrix()
    return result


def display_intrinsic(camera, size, calibration_size, mode):
    k = np.asarray(camera['camera_matrix'], dtype=float).copy()
    d = np.asarray(camera['distortion_coefficients'], dtype=float).reshape(-1)
    if k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError('Invalid camera matrix.')
    if len(d) not in (4, 5, 8, 12, 14) or not np.isfinite(d).all():
        raise ValueError('Unsupported OpenCV distortion coefficients.')
    source = camera.get('source_image_size', calibration_size)
    if len(source) != 2 or min(source) <= 0: raise ValueError('Invalid calibration image size.')
    k[0] *= size[0] / source[0]; k[1] *= size[1] / source[1]
    if mode == 'rectified':
        new, _ = cv2.getOptimalNewCameraMatrix(k, d, tuple(size), 0, tuple(size))
        return new, np.zeros_like(d), k, d
    if mode == 'pinhole':
        # For already-undistorted/simulation inputs; never silently infer this.
        new = np.asarray(camera.get('undistorted_camera_matrix', camera['camera_matrix']), dtype=float).copy()
        new[0] *= size[0] / source[0]; new[1] *= size[1] / source[1]
        if new.shape != (3, 3) or not np.isfinite(new).all() or min(new[0, 0], new[1, 1]) <= 0:
            raise ValueError('Invalid undistorted camera matrix.')
        return new, np.zeros_like(d), k, d
    if mode != 'raw': raise ValueError('Unknown image mode.')
    return k, d, k, d


def project(pose, points, k, distortion):
    pose = rigid(pose)
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if not np.isfinite(points).all() or np.any(np.abs(points) > 1e6): raise ValueError('Invalid world point (metres).')
    if not len(points): return np.empty((0, 2)), np.empty(0, dtype=bool)
    world_to_camera = np.linalg.inv(pose)
    rotation, _ = cv2.Rodrigues(world_to_camera[:3, :3])
    pixels, _ = cv2.projectPoints(points, rotation, world_to_camera[:3, 3], k, distortion)
    pixels = pixels.reshape(-1, 2)
    depth = (points - pose[:3, 3]) @ pose[:3, :3][:, 2]
    valid = (depth > .01) & np.isfinite(pixels).all(axis=1) & (np.abs(pixels) < 1e7).all(axis=1)
    return pixels, valid


def evaluate(pose, pairs, k, distortion):
    pixels, visible = project(pose, [p['world'] for p in pairs], k, distortion)
    rows = []
    for p, xy, valid in zip(pairs, pixels, visible):
        measured = p.get('image')
        error = float(np.linalg.norm(xy - measured)) if valid and measured is not None else None
        rows.append(dict(projected=xy.tolist() if valid else None, error_px=error,
                         holdout=p.get('holdout', False)))
    metrics = {}
    for name, subset in [('all', rows), ('fit', [r for r in rows if not r['holdout']]),
                         ('holdout', [r for r in rows if r['holdout']])]:
        errors = [r['error_px'] for r in subset if r['error_px'] is not None]
        metrics[name] = dict(count=len(errors), rmse_px=float(np.sqrt(np.mean(np.square(errors)))) if errors else None)
    return dict(points=rows, metrics=metrics)


def refine(pose, pairs, k, distortion):
    training = [p for p in pairs if p.get('image') is not None and not p.get('holdout', False)]
    if len(training) < 6: raise ValueError('Mark at least 6 fit correspondences; holdout points are not used.')
    world = np.asarray([p['world'] for p in training], dtype=float)
    image = np.asarray([p['image'] for p in training], dtype=float)
    if np.linalg.matrix_rank(world - world.mean(0), tol=1e-5) < 2:
        raise ValueError('Points are collinear. Use spatially spread structural landmarks.')
    if np.linalg.matrix_rank(image - image.mean(0), tol=1e-5) < 2:
        raise ValueError('Image points are collinear.')
    if not project(pose, world, k, distortion)[1].all():
        raise ValueError('Fit points must start in front of this camera.')
    def residual(x):
        candidate = move_camera(pose, x[:3], x[3:])
        pixels, valid = project(candidate, world, k, distortion)
        error = np.nan_to_num(pixels - image, nan=1e6, posinf=1e6, neginf=-1e6)
        error[~valid] = 1e6
        return error.ravel()
    limits = np.array([2, 2, 2, 15, 15, 15])
    result = least_squares(residual, np.zeros(6), bounds=(-limits, limits),
                           loss='soft_l1', f_scale=3, x_scale=[.1, .1, .1, 1, 1, 1],
                           max_nfev=200)
    candidate = move_camera(pose, result.x[:3], result.x[3:])
    if not result.success or not project(candidate, world, k, distortion)[1].all():
        raise ValueError('Refinement did not converge to a valid forward-facing pose.')
    return dict(camera_to_world=candidate.tolist(), offsets=result.x.tolist(),
                at_bound=bool(np.any(np.abs(result.x) > limits * .98)),
                coplanar=bool(np.linalg.matrix_rank(world - world.mean(0), tol=1e-5) == 2),
                before=evaluate(pose, pairs, k, distortion), after=evaluate(candidate, pairs, k, distortion))


def export_calibration(document, edits, source_sha256):
    output = deepcopy(document)
    known = {c['camera_id'] for c in document['cameras']}
    if not set(edits).issubset(known): raise ValueError('Unknown camera ID in edits.')
    for c in output['cameras']:
        if c['camera_id'] not in edits: continue
        pose = rigid(edits[c['camera_id']])
        c.update(camera_to_world=pose.tolist(), world_to_camera=np.linalg.inv(pose).tolist(),
                 position_m=pose[:3, 3].tolist())
    output['manual_correction'] = dict(source_sha256=source_sha256,
        created_utc=datetime.now(timezone.utc).isoformat(), edited_camera_ids=sorted(edits),
        intrinsics_changed=False, automatically_deployed=False, source_metrics_not_recomputed=True,
        note='User-edited extrinsics; not certified calibration. Keep the correspondence project for validation.')
    return output
