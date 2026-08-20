# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import cameras
from ..utils.transforms import affine_transform_pts_cuda as do_transform
from dt_common.spatial.workspace import query_ground_heights_mm


class ProjectLayer(nn.Module):
    def __init__(self, cfg, mode="", cube_size=None, workspace=None):
        super(ProjectLayer, self).__init__()
        self.mode = mode
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
        self.grid_center = torch.tensor(
            cfg.MULTI_PERSON.SPACE_CENTER, dtype=torch.float32
        )
        self.grid_size = torch.tensor(cfg.MULTI_PERSON.SPACE_SIZE, dtype=torch.float32)
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

        if self.mode == "rootnet":
            self.cube_size = (
                cfg.MULTI_PERSON.INITIAL_CUBE_SIZE if cube_size is None else cube_size
            )
        elif self.mode == "posenet":
            self.clamp_cube_size = cfg.PICT_STRUCT.CUBE_SIZE
            self.clamp_grid_size = cfg.PICT_STRUCT.GRID_SIZE
            self.cube_size = [
                int(
                    round(
                        self.clamp_cube_size[0]
                        * self.grid_size[0].item()
                        / self.clamp_grid_size[0]
                    )
                ),
                int(
                    round(
                        self.clamp_cube_size[1]
                        * self.grid_size[1].item()
                        / self.clamp_grid_size[1]
                    )
                ),
                int(
                    round(
                        self.clamp_cube_size[2]
                        * self.grid_size[2].item()
                        / self.clamp_grid_size[2]
                    )
                ),
            ]
            self.clamp_nbins = (
                self.clamp_cube_size[0]
                * self.clamp_cube_size[1]
                * self.clamp_cube_size[2]
            )
        self.nbins = self.cube_size[0] * self.cube_size[1] * self.cube_size[2]

        self.first_inference = True
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
            and self.mode == "rootnet"
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
        if self.mode != "rootnet" or self.workspace is None:
            raise RuntimeError("project_workspace requires a rootnet EdgeWorkspace")
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
        device = heatmaps[0].device

        # 결과 텐서를 미리 할당
        # final_cubes = torch.zeros(batch_size, num_joints, self.nbins, device=device)
        # weight_sum = torch.zeros(batch_size, num_joints, self.nbins, device=device)
        cubes = torch.zeros(batch_size, num_joints, 1, self.nbins, n, device=device)
        for i in range(batch_size):
            for c in range(n):
                cubes[i : i + 1, :, :, :, c] += F.grid_sample(
                    heatmaps[c][i : i + 1, :, :, :],
                    sample_grids[c],
                    align_corners=True,
                )
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
        # for i in range(n):
        #     heatmap = heatmaps[i]
        #     sample_grid = sample_grids[i].expand(batch_size, -1, -1, -1)

        #     # grid_sample 수행
        #     sampled_cube = F.grid_sample(heatmap, sample_grid, align_corners=True)
        #     sampled_cube = sampled_cube.squeeze(2)  # (batch_size, num_joints, nbins)

        #     # 바운딩 마스크 적용
        #     mask = bounding[0, 0, :, i].unsqueeze(0).unsqueeze(0)  # (1, 1, nbins)
        #     mask = mask.expand(batch_size, num_joints, -1)

        #     # 가중 평균을 위한 누적
        #     final_cubes += sampled_cube * mask
        #     weight_sum += mask

        #     # 즉시 메모리 해제
        #     # del sampled_cube, sample_grid, mask
        #     # torch.cuda.empty_cache()

        # # 최종 평균 계산
        # final_cubes = final_cubes / (weight_sum + 1e-6)
        # final_cubes = torch.nan_to_num(final_cubes, nan=0.0)
        # final_cubes = torch.clamp(final_cubes, 0.0, 1.0)

        # # 큐브 형태로 재구성
        # final_cubes = final_cubes.view(batch_size, num_joints, self.cube_size[0], self.cube_size[1], self.cube_size[2])

        # return final_cubes

    def clamp_cubes(self, cubes, clamp_grid_centers, batch_indices):
        device = cubes.device
        num_candidates = len(batch_indices)
        num_joints = cubes.shape[1]

        clamp_cube_size_tensor = torch.tensor(self.clamp_cube_size, device=device)
        cube_size_tensor = torch.tensor(self.cube_size, device=device)
        clamp_cubes = torch.zeros(
            num_candidates,
            num_joints,
            clamp_cube_size_tensor[0],
            clamp_cube_size_tensor[1],
            clamp_cube_size_tensor[2],
            device=device,
        )
        grids = torch.zeros(num_candidates, self.clamp_nbins, 3, device=device)

        grid_center = self.grid_center.to(device)
        grid_size = self.grid_size.to(device)

        centers_3d = clamp_grid_centers[:, :3]
        cube_grid_scale = grid_size / cube_size_tensor
        center_indices = torch.round(
            (centers_3d - grid_center + grid_size / 2) / cube_grid_scale
        ).long()
        half_clamp_size = clamp_cube_size_tensor // 2

        # Calculate source cube slicing indices
        zeros_tensor = torch.zeros(3, device=device, dtype=torch.long)
        src_start = torch.max(zeros_tensor, center_indices - half_clamp_size)
        src_end = torch.min(cube_size_tensor, center_indices + half_clamp_size)

        # Calculate destination clamp_cube slicing indices
        dst_start = half_clamp_size - (center_indices - src_start)
        dst_end = half_clamp_size + (src_end - center_indices)

        # Iterate over candidates to perform the slicing and assignment
        for j in range(num_candidates):
            s_x, s_y, s_z = src_start[j]
            e_x, e_y, e_z = src_end[j]

            d_s_x, d_s_y, d_s_z = dst_start[j]
            d_e_x, d_e_y, d_e_z = dst_end[j]

            # Check if there is a valid volume to copy
            if (e_x > s_x) and (e_y > s_y) and (e_z > s_z):
                clamp_cubes[j, :, d_s_x:d_e_x, d_s_y:d_e_y, d_s_z:d_e_z] = cubes[
                    batch_indices[j], :, s_x:e_x, s_y:e_y, s_z:e_z
                ]

            grids[j] = self.compute_grid(
                self.clamp_grid_size, centers_3d[j], self.clamp_cube_size, device=device
            )
        return clamp_cubes, grids

    def forward(
        self,
        heatmaps,
        grid_size,
        grid_centers,
        cube_size,
        batch_indices=None,
        flip_xcoords=None,
        xy_axes=None,
        return_valid_mask=False,
    ):
        device = heatmaps[0].device
        if self.mode == "rootnet":
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
        if self.mode == "posenet":
            if self.first_inference:
                (
                    self.sample_grids,
                    self.bounding,
                    self.grid,
                    _,
                ) = self.get_voxel(
                    self.grid_size,
                    self.grid_center,
                    self.cube_size,
                    device=device,
                )
                self.first_inference = False
            cubes = self.project(heatmaps, self.sample_grids, self.bounding)
            cubes, grids = self.clamp_cubes(cubes, grid_centers, batch_indices)

            return cubes, grids
