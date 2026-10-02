# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Derived from microsoft/voxelpose-pytorch; modified for intelligent-synchronization.
# SPDX-License-Identifier: MIT
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import cameras
from ..utils.transforms import affine_transform_pts_cuda as do_transform
from dt_common.spatial.workspace import query_ground_heights_mm


class ProjectLayer(nn.Module):
    def __init__(self, cfg, cube_size=None, workspace=None):
        super(ProjectLayer, self).__init__()
        self.workspace = workspace
        self.trans = getattr(cfg, "TRANSFORM", None)
        self.cams = getattr(cfg, "CAMS", None)
        if self.trans is None:
            raise ValueError(
                "cfg.TRANSFORM must be initialized before ProjectLayer construction"
            )
        if not self.cams:
            raise ValueError(
                "cfg.CAMS must be initialized before ProjectLayer construction"
            )

        self.img_size = cfg.NETWORK.IMAGE_SIZE
        self.img_size_orig = cfg.NETWORK.IMAGE_SIZE_ORIG
        self.heatmap_size = cfg.NETWORK.HEATMAP_SIZE
        spatial_context = getattr(cfg, "SPATIAL_CONTEXT", None)
        self.ground_surface = (
            None if spatial_context is None else spatial_context.ground_surface
        )
        root_clearance = (
            (400.0, 1400.0)
            if workspace is None
            else workspace.root_clearance_mm
        )
        self.root_clearance_min_mm = float(root_clearance[0])
        self.root_clearance_max_mm = float(root_clearance[1])

        self.cube_size = (
            cfg.MULTI_PERSON.INITIAL_CUBE_SIZE if cube_size is None else cube_size
        )
        self.nbins = self.cube_size[0] * self.cube_size[1] * self.cube_size[2]

        self._voxel_cache = {}

    def compute_grid(
        self,
        boxSize,
        boxCenter,
        nBins,
        device=None,
        xy_axes=None,
    ):
        if isinstance(boxSize, int) or isinstance(boxSize, float):
            boxSize = [boxSize, boxSize, boxSize]
        if isinstance(nBins, int):
            nBins = [nBins, nBins, nBins]

        size = torch.as_tensor(boxSize, dtype=torch.float32, device=device)
        center = torch.as_tensor(boxCenter, dtype=torch.float32, device=device)
        axes = torch.as_tensor(
            [[1.0, 0.0], [0.0, 1.0]] if xy_axes is None else xy_axes,
            dtype=torch.float32,
            device=device,
        )
        grid1Dx = torch.linspace(-size[0] / 2, size[0] / 2, nBins[0], device=device)
        grid1Dy = torch.linspace(-size[1] / 2, size[1] / 2, nBins[1], device=device)
        grid1Dz = torch.linspace(-size[2] / 2, size[2] / 2, nBins[2], device=device)
        gridx, gridy, gridz = torch.meshgrid(
            grid1Dx,
            grid1Dy,
            grid1Dz,
            indexing="ij",
        )
        local_xy = torch.stack([gridx, gridy], dim=-1).reshape(-1, 2)
        world_xy = local_xy @ axes + center[:2]
        world_z = (gridz + center[2]).reshape(-1, 1)
        grid = torch.cat([world_xy, world_z], dim=1)
        return grid

    def get_voxel(
        self,
        grid_size,
        grid_center,
        cube_size,
        device=None,
        xy_axes=None,
        workspace_valid_mask=None,
        min_views=1,
    ):
        sample_grids = []
        w, h = self.heatmap_size
        width, height = self.img_size_orig
        n = len(self.cams)
        bounding = torch.zeros(1, 1, self.nbins, n, device=device)
        grid = self.compute_grid(
            grid_size,
            grid_center,
            cube_size,
            device=device,
            xy_axes=xy_axes,
        )
        if workspace_valid_mask is not None:
            voxel_valid = torch.as_tensor(
                workspace_valid_mask,
                dtype=torch.bool,
                device=device,
            ).reshape(-1)
            if voxel_valid.numel() != self.nbins:
                raise ValueError("workspace valid mask does not match cube_size")
        else:
            voxel_valid = torch.ones(self.nbins, dtype=torch.bool, device=device)
        if (
            workspace_valid_mask is None
            and self.ground_surface is not None
        ):
            grid_volume = grid.view(
                self.cube_size[0],
                self.cube_size[1],
                self.cube_size[2],
                3,
            )
            xy_grid = grid_volume[:, :, 0, :2].detach().cpu().numpy()
            ground_heights = torch.as_tensor(
                query_ground_heights_mm(self.ground_surface, xy_grid),
                dtype=grid.dtype,
                device=device,
            )
            clearance = grid_volume[:, :, :, 2] - ground_heights[:, :, None]
            voxel_valid = (
                torch.isfinite(clearance)
                & (clearance >= self.root_clearance_min_mm)
                & (clearance <= self.root_clearance_max_mm)
            ).reshape(-1)
        for c in range(n):
            trans = torch.as_tensor(
                self.trans,
                dtype=torch.float,
                device=device,
            )
            cam = self.cams[c].copy()

            xy = cameras.project_pose(grid, cam)
            depth = cameras.world_to_camera_frame(
                grid,
                cam["R"],
                cam["T"],
            )[:, 2]
            bounding[0, 0, :, c] = (
                (depth > 0)
                & (xy[:, 0] >= 0)
                & (xy[:, 1] >= 0)
                & (xy[:, 0] < width)
                & (xy[:, 1] < height)
                & voxel_valid
            )
            xy = torch.clamp(xy, -1.0, max(width, height))
            xy = do_transform(xy, trans)
            xy = (
                xy
                * torch.tensor([w, h], dtype=torch.float, device=device)
                / torch.tensor(self.img_size, dtype=torch.float, device=device)
            )
            sample_grid = (
                xy
                / torch.tensor([w - 1, h - 1], dtype=torch.float, device=device)
                * 2.0
                - 1.0
            )
            sample_grid = torch.clamp(sample_grid.view(1, 1, self.nbins, 2), -1.1, 1.1)
            sample_grids.append(sample_grid)

        voxel_valid = voxel_valid & (
            bounding[0, 0].sum(dim=-1) >= int(min_views)
        )
        return sample_grids, bounding, grid, voxel_valid.view(*self.cube_size)

    def project_workspace(self, heatmaps, flip_xcoords=None):
        """Project heatmaps into the pre-resolved edge workspace."""

        del flip_xcoords  # Projection coordinates already follow calibrated views.
        if self.workspace is None:
            raise RuntimeError("project_workspace requires an EdgeWorkspace")
        device = heatmaps[0].device
        key = (
            device.type,
            device.index,
            self.workspace.workspace_id,
        )
        cached = self._voxel_cache.get(key)
        if cached is None:
            cached = self.get_voxel(
                self.workspace.size_mm,
                self.workspace.center_mm,
                self.workspace.cube_size,
                device=device,
                xy_axes=self.workspace.xy_axes,
                workspace_valid_mask=self.workspace.valid_mask,
                min_views=self.workspace.min_views,
            )
            self._voxel_cache[key] = cached
        sample_grids, bounding, _, voxel_valid = cached
        return self.project(heatmaps, sample_grids, bounding), voxel_valid

    def project(self, heatmaps, sample_grids, bounding):
        n = len(heatmaps)
        batch_size = heatmaps[0].shape[0]
        num_joints = heatmaps[0].shape[1]
        if any(heatmap.shape[0] != batch_size for heatmap in heatmaps):
            raise ValueError("all camera heatmaps must use the same batch size")

        # Treat camera as another batch dimension so CUDA launches one
        # grid_sample kernel instead of one kernel per camera and sample.
        heatmap_batch = torch.stack(heatmaps, dim=1).reshape(
            batch_size * n,
            num_joints,
            heatmaps[0].shape[-2],
            heatmaps[0].shape[-1],
        )
        grid_batch = torch.cat(sample_grids, dim=0)
        grid_batch = (
            grid_batch.unsqueeze(0)
            .expand(batch_size, -1, -1, -1, -1)
            .reshape(batch_size * n, 1, self.nbins, 2)
        )
        cubes = F.grid_sample(
            heatmap_batch,
            grid_batch,
            align_corners=True,
        )
        cubes = cubes.reshape(
            batch_size,
            n,
            num_joints,
            1,
            self.nbins,
        ).permute(0, 2, 3, 4, 1)
        cubes = torch.sum(torch.mul(cubes, bounding), dim=-1) / (
            torch.sum(bounding, dim=-1) + 1e-6
        )
        cubes = cubes.clone()
        cubes[cubes != cubes] = 0.0
        cubes = cubes.clamp(0.0, 1.0)

        cubes = cubes.view(
            batch_size,
            num_joints,
            self.cube_size[0],
            self.cube_size[1],
            self.cube_size[2],
        )
        return cubes

    def forward(
        self,
        heatmaps,
        grid_size,
        grid_centers,
        cube_size,
        flip_xcoords=None,
        xy_axes=None,
        return_valid_mask=False,
    ):
        device = heatmaps[0].device
        center = torch.as_tensor(
            grid_centers[0],
            dtype=torch.float32,
            device=device,
        )
        key = (
            device.type,
            device.index,
            tuple(float(value) for value in center.detach().cpu().tolist()),
            tuple(float(value) for value in grid_size),
            tuple(
                float(value)
                for value in torch.as_tensor(
                    [[1.0, 0.0], [0.0, 1.0]] if xy_axes is None else xy_axes
                ).reshape(-1)
            ),
        )
        cached = self._voxel_cache.get(key)
        if cached is None:
            cached = self.get_voxel(
                grid_size,
                center,
                cube_size,
                device=device,
                xy_axes=xy_axes,
            )
            self._voxel_cache[key] = cached
        sample_grids, bounding, _, voxel_valid = cached
        cubes = self.project(heatmaps, sample_grids, bounding)
        if return_valid_mask:
            return cubes, voxel_valid
        return cubes
