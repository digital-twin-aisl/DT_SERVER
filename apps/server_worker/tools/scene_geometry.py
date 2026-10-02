# SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Per-edge voxel scenes for offline pose tools; model-space positions are mm.

Builds each enabled edge's workspace grid from a deployment manifest and a
calibration result without changing either file on disk.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from dt_common.calibration.voxelpose import load_calibration_result, select_voxelpose_cameras
from dt_common.spatial.workspace import (
    _points_in_polygon, build_edge_workspace, load_spatial_context,
)
from apps.server_worker.src.pose.utils.cameras import project_pose
from apps.server_worker.src.utils.transforms import get_affine_transform, get_scale


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_value(value):
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    """Write a new JSON file; never overwrite an existing result."""
    with Path(path).open("x") as stream:
        json.dump(value, stream, default=json_value, indent=2, allow_nan=False)
        stream.write("\n")


@dataclass
class Scene:
    edge_id: str
    cameras: list
    grid: torch.Tensor
    terrain_mask: torch.Tensor
    shape: tuple
    spacing: np.ndarray
    workspace: object
    ground: object
    image_size: np.ndarray
    heatmap_size: np.ndarray
    transform: np.ndarray
    min_views: int

    def pixels(self, points, camera):
        points = torch.as_tensor(points, dtype=torch.float32).reshape(-1, 3)
        with torch.no_grad():
            xy = project_pose(points, camera).numpy()
        r = np.asarray(camera["R"])
        t = np.asarray(camera["T"]).reshape(3)
        depth = ((points.numpy() - t) @ r.T)[:, 2]
        visible = (depth > 0) & np.isfinite(xy).all(axis=1)
        visible &= (xy >= 0).all(axis=1) & (xy < self.image_size).all(axis=1)
        return xy, visible

    def heatmap_points(self, points, camera):
        xy, visible = self.pixels(points, camera)
        xy = xy @ self.transform[:, :2].T + self.transform[:, 2]
        return xy, visible

    def visible(self, points, cameras):
        return np.stack([self.pixels(points, camera)[1] for camera in cameras]).sum(axis=0)


def load_scenes(config):
    """config: {"deployment", "calibration", "runtime_config"} file paths."""
    context = load_spatial_context(config["deployment"])
    if context is None or any(context.world_origin_m):
        raise ValueError("this tool requires an explicit scene with zero world_origin_m")
    # Override only in memory; the deployed manifest is not changed.
    context = replace(context, calibration_path=Path(config["calibration"]),
                      calibration_sha256=sha256(config["calibration"]))
    calibration = load_calibration_result(config["calibration"])
    runtime = yaml.safe_load(Path(config["runtime_config"]).read_text())
    manifest = json.loads(Path(config["deployment"]).read_text())
    enabled = {e["id"] for e in manifest["edges"] if e.get("enabled", True)}
    network = runtime["NETWORK"]
    image_size = np.asarray(network["IMAGE_SIZE_ORIG"])
    model_size = np.asarray(network["IMAGE_SIZE"])
    heatmap_size = np.asarray(network["HEATMAP_SIZE"])
    transform = get_affine_transform(image_size / 2, get_scale(image_size, model_size), 0, model_size)
    transform = transform * (heatmap_size / model_size)[:, None]
    scenes = {}
    for edge, ids in context.edge_camera_ids.items():
        if edge not in enabled:
            continue
        if len(ids) != 4:
            raise ValueError("the tool requires four cameras per edge")
        cameras = select_voxelpose_cameras(calibration, ids)
        w = build_edge_workspace(context, edge, cameras, image_size,
                                 runtime["MULTI_PERSON"]["SPACE_SIZE"],
                                 runtime["MULTI_PERSON"]["INITIAL_CUBE_SIZE"])
        axes = [np.linspace(-s / 2, s / 2, int(n)) for s, n in zip(w.size_mm, w.cube_size)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        grid[..., :2] = grid[..., :2] @ w.xy_axes + w.center_mm[:2]
        grid[..., 2] += w.center_mm[2]
        inside = _points_in_polygon(grid[:, :, 0, :2].reshape(-1, 2), w.footprint_xy_mm).reshape(grid.shape[:2])
        clearance = grid[..., 2] - w.ground_height_mm[:, :, None]
        terrain = inside[:, :, None] & np.isfinite(clearance)
        terrain &= (clearance >= w.root_clearance_mm[0]) & (clearance <= w.root_clearance_mm[1])
        scenes[edge] = Scene(edge, cameras, torch.from_numpy(grid.reshape(-1, 3).astype(np.float32)),
                             torch.from_numpy(terrain.reshape(-1)), tuple(int(n) for n in w.cube_size),
                             w.voxel_spacing_mm, w, context.ground_surface,
                             image_size, heatmap_size, transform, context.policy.min_views)
    if not scenes:
        raise ValueError("no enabled edges")
    return scenes, context
