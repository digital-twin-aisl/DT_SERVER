# SPDX-FileCopyrightText: 2025-2026 DT_SERVER contributors
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Metric alignment of VGGT coordinates to the marker-defined world."""

from dataclasses import dataclass
import itertools
import math
from typing import Iterable
import numpy as np


@dataclass(frozen=True)
class SimilarityTransform:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    inlier_mask: np.ndarray
    residuals: np.ndarray

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        return self.scale * (points @ self.rotation.T) + self.translation

    def matrix4(self) -> np.ndarray:
        """Return the homogeneous VGGT-to-USD similarity matrix."""
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = self.scale * self.rotation
        matrix[:3, 3] = self.translation
        return matrix



def fit_similarity_transform(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit ``target = scale * rotation @ source + translation`` (Umeyama)."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must have matching Nx3 shapes.")
    if len(source) < 3:
        raise ValueError("At least three point correspondences are required.")

    source_centered = source - source.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    source_variance = float(np.sum(source_centered**2) / len(source))
    if source_variance <= 1.0e-12 or np.linalg.matrix_rank(source_centered) < 2:
        raise ValueError("Source correspondences are degenerate.")
    if np.linalg.matrix_rank(target_centered) < 2:
        raise ValueError("Target correspondences are degenerate.")

    covariance = target_centered.T @ source_centered / len(source)
    left, singular_values, right_transpose = np.linalg.svd(covariance)
    sign = np.ones(3, dtype=np.float64)
    if np.linalg.det(left @ right_transpose) < 0:
        sign[-1] = -1.0
    rotation = left @ np.diag(sign) @ right_transpose
    scale = float(np.sum(singular_values * sign) / source_variance)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Estimated similarity scale is not positive.")
    translation = target.mean(axis=0) - scale * (rotation @ source.mean(axis=0))
    return scale, rotation, translation



def robust_similarity_transform(
    source: np.ndarray,
    target: np.ndarray,
    *,
    inlier_threshold_m: float = 0.10,
    max_hypotheses: int = 1000,
    random_seed: int = 0,
) -> SimilarityTransform:
    """RANSAC wrapper around the metric similarity fit."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must have matching Nx3 shapes.")
    if len(source) < 3:
        raise ValueError("At least three point correspondences are required.")
    if inlier_threshold_m <= 0:
        raise ValueError("inlier_threshold_m must be positive.")

    all_combinations: Iterable[tuple[int, int, int]] = itertools.combinations(
        range(len(source)), 3
    )
    combination_count = math.comb(len(source), 3)
    if combination_count <= max_hypotheses:
        samples = list(all_combinations)
    else:
        rng = np.random.default_rng(random_seed)
        sample_set: set[tuple[int, int, int]] = set()
        while len(sample_set) < max_hypotheses:
            sample_set.add(tuple(sorted(rng.choice(len(source), 3, replace=False).tolist())))
        samples = list(sample_set)

    best_mask: np.ndarray | None = None
    best_score = (-1, -math.inf)
    for indices in samples:
        try:
            scale, rotation, translation = fit_similarity_transform(
                source[list(indices)], target[list(indices)]
            )
        except ValueError:
            continue
        predicted = scale * (source @ rotation.T) + translation
        residuals = np.linalg.norm(predicted - target, axis=1)
        mask = residuals <= inlier_threshold_m
        inlier_count = int(mask.sum())
        if inlier_count < 3:
            continue
        median_error = float(np.median(residuals[mask]))
        score = (inlier_count, -median_error)
        if score > best_score:
            best_score = score
            best_mask = mask

    if best_mask is None:
        raise RuntimeError(
            "Could not find a non-degenerate ArUco alignment. Increase marker "
            "visibility or adjust --alignment-threshold-m."
        )
    scale, rotation, translation = fit_similarity_transform(
        source[best_mask], target[best_mask]
    )
    residuals = np.linalg.norm(
        scale * (source @ rotation.T) + translation - target,
        axis=1,
    )
    final_mask = residuals <= inlier_threshold_m
    if int(final_mask.sum()) >= 3 and not np.array_equal(final_mask, best_mask):
        scale, rotation, translation = fit_similarity_transform(
            source[final_mask], target[final_mask]
        )
        residuals = np.linalg.norm(
            scale * (source @ rotation.T) + translation - target,
            axis=1,
        )
    else:
        final_mask = best_mask
    return SimilarityTransform(scale, rotation, translation, final_mask, residuals)



def transform_camera_extrinsic(
    vggt_world_to_camera: np.ndarray,
    alignment: SimilarityTransform,
) -> tuple[np.ndarray, np.ndarray]:
    """Move a VGGT camera pose into the metric marker-tree world."""
    source_extrinsic = np.asarray(vggt_world_to_camera, dtype=np.float64)
    if source_extrinsic.shape != (3, 4):
        raise ValueError("VGGT extrinsic must have shape 3x4.")
    source_rotation = source_extrinsic[:, :3]
    source_translation = source_extrinsic[:, 3]
    source_camera_center = -source_rotation.T @ source_translation

    world_camera_center = alignment.transform_points(source_camera_center)
    world_to_camera_rotation = source_rotation @ alignment.rotation.T
    world_to_camera_translation = -world_to_camera_rotation @ world_camera_center

    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3, :3] = world_to_camera_rotation
    world_to_camera[:3, 3] = world_to_camera_translation
    camera_to_world = np.linalg.inv(world_to_camera)
    return world_to_camera, camera_to_world

