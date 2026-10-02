# SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Inverse OpenCV lens map for image-matched rendering and landmark picking.

The output raster has the video's exact W/H. Each texel stores the normalized
camera ray for its pixel centre. Invalid/noninvertible lens regions are masked,
not silently replaced with a pinhole model. No camera pose is baked into this map.
"""
import cv2
import numpy as np


def inverse_lens_map(k, distortion, width, height):
    k = np.asarray(k, dtype=np.float64)
    d = np.asarray(distortion, dtype=np.float64)
    if min(width, height) < 1 or max(width, height) > 8192:
        raise ValueError('Unsupported render resolution (maximum dimension: 8192).')
    yy, xx = np.mgrid[:height, :width]
    pixels = np.column_stack((xx.ravel() + .5, yy.ravel() + .5))
    if np.count_nonzero(d):
        rays = cv2.undistortPointsIter(pixels[:, None], k, d, None, None,
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-8)).reshape(-1, 2)
    else:
        rays = (pixels - k[:2, 2]) / np.array([k[0, 0], k[1, 1]])
    # OpenCV's polynomial distortion can fold outside the calibrated region.
    # Require an accurate forward round-trip; a failed inverse is not usable.
    finite = np.isfinite(rays).all(axis=1) & (np.abs(rays) < 20).all(axis=1)
    safe = np.where(finite[:, None], rays, 0)
    restored, _ = cv2.projectPoints(np.column_stack((safe, np.ones(len(safe)))),
                                   np.zeros(3), np.zeros(3), k, d)
    errors = np.linalg.norm(restored.reshape(-1, 2) - pixels, axis=1)
    valid = finite & (errors <= .05)
    if not valid.any(): raise ValueError('No invertible pixels for this lens model.')
    # Overscan covers rays beyond the undistorted camera's original sensor crop.
    # Otherwise applying barrel distortion would produce artificial cut-off edges.
    low = rays[valid].min(axis=0) - 2 / np.array([k[0, 0], k[1, 1]])
    high = rays[valid].max(axis=0) + 2 / np.array([k[0, 0], k[1, 1]])
    span = high - low
    source_size = np.minimum(4096, np.maximum(2, np.ceil(span * [k[0, 0], k[1, 1]]))).astype(int)
    source_k = np.array([[source_size[0]/span[0], 0, -low[0]*source_size[0]/span[0]],
                         [0, source_size[1]/span[1], -low[1]*source_size[1]/span[1]], [0, 0, 1]])
    data = rays.astype('<f4')
    data[~valid] = 1e9
    meta = dict(width=width, height=height, camera_matrix=k.tolist(), distortion_coefficients=d.tolist(),
                ray_bounds=[float(low[0]), float(low[1]), float(high[0]), float(high[1])],
                source_size=source_size.tolist(), source_camera_matrix=source_k.tolist(),
                invalid_fraction=float(1 - valid.mean()), max_roundtrip_error_px=float(errors[valid].max()),
                encoding='little-endian-float32-RG', row_order='top-to-bottom', pixel_centres='x+0.5,y+0.5')
    return meta, data.reshape(height, width, 2)
