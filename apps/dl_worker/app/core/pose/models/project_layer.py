# This file is part of DT_SERVER.
# 
# DT_SERVER is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or
# (at your option) any later version.
# 
# DT_SERVER is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with DT_SERVER; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import cameras
from ..utils.transforms import get_affine_transform as get_transform
from ..utils.transforms import affine_transform_pts_cuda as do_transform


class ProjectLayer(nn.Module):
    def __init__(self, cfg, mode=""):
        super(ProjectLayer, self).__init__()
        self.mode = mode
        self.trans = getattr(cfg, "TRANSFORM", None)
        self.cams = getattr(cfg, "CAMS", None)

        self.img_size = cfg.NETWORK.IMAGE_SIZE
        self.img_size_orig = cfg.NETWORK.IMAGE_SIZE_ORIG
        self.heatmap_size = cfg.NETWORK.HEATMAP_SIZE
        self.grid_center = torch.tensor(cfg.MULTI_PERSON.SPACE_CENTER, dtype=torch.float32)
        self.grid_size = torch.tensor(cfg.MULTI_PERSON.SPACE_SIZE, dtype=torch.float32)

        if self.mode == "rootnet":
            self.cube_size = cfg.MULTI_PERSON.INITIAL_CUBE_SIZE
        elif self.mode == "posenet":
            self.clamp_cube_size = cfg.PICT_STRUCT.CUBE_SIZE
            self.clamp_grid_size = cfg.PICT_STRUCT.GRID_SIZE
            self.cube_size = [
                int(round(self.clamp_cube_size[0] * self.grid_size[0].item() / self.clamp_grid_size[0])),
                int(round(self.clamp_cube_size[1] * self.grid_size[1].item() / self.clamp_grid_size[1])),
                int(round(self.clamp_cube_size[2] * self.grid_size[2].item() / self.clamp_grid_size[2])),
            ]
            self.clamp_nbins = self.clamp_cube_size[0] * self.clamp_cube_size[1] * self.clamp_cube_size[2]
        self.nbins = self.cube_size[0] * self.cube_size[1] * self.cube_size[2]

        self.first_inference = True

    def compute_grid(self, boxSize, boxCenter, nBins, device=None):
        if isinstance(boxSize, int) or isinstance(boxSize, float):
            boxSize = [boxSize, boxSize, boxSize]
        if isinstance(nBins, int):
            nBins = [nBins, nBins, nBins]

        grid1Dx = torch.linspace(-boxSize[0] / 2, boxSize[0] / 2, nBins[0], device=device)
        grid1Dy = torch.linspace(-boxSize[1] / 2, boxSize[1] / 2, nBins[1], device=device)
        grid1Dz = torch.linspace(-boxSize[2] / 2, boxSize[2] / 2, nBins[2], device=device)
        gridx, gridy, gridz = torch.meshgrid(
            grid1Dx + boxCenter[0],
            grid1Dy + boxCenter[1],
            grid1Dz + boxCenter[2],
        )
        gridx = gridx.contiguous().view(-1, 1)
        gridy = gridy.contiguous().view(-1, 1)
        gridz = gridz.contiguous().view(-1, 1)
        grid = torch.cat([gridx, gridy, gridz], dim=1)
        return grid
    
    def get_voxel(self, grid_size, grid_center, cube_size, device=None):
        sample_grids = []
        w, h = self.heatmap_size
        width, height = self.img_size_orig
        n = len(self.cams)
        bounding = torch.zeros(1, 1, self.nbins, n, device=device)
        grid = self.compute_grid(grid_size, grid_center, cube_size, device=device)
        for c in range(n):
            trans = torch.as_tensor(
                self.trans,
                dtype=torch.float,
                device=device,
            )
            cam = self.cams[c].copy()

            xy = cameras.project_pose(grid, cam)
            bounding[0, 0, :, c] = (xy[:, 0] >= 0) & (xy[:, 1] >= 0) & (xy[:, 0] < width) & (
                                xy[:, 1] < height)
            xy = torch.clamp(xy, -1.0, max(width, height))
            xy = do_transform(xy, trans)
            xy = xy * torch.tensor(
                [w, h], dtype=torch.float, device=device) / torch.tensor(
                self.img_size, dtype=torch.float, device=device)
            sample_grid = xy / torch.tensor(
                [w - 1, h - 1], dtype=torch.float, device=device
            ) * 2.0 - 1.0
            sample_grid = torch.clamp(sample_grid.view(1, 1, self.nbins, 2), -1.1, 1.1)
            sample_grids.append(sample_grid)

        return sample_grids, bounding, grid
    
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
                cubes[i:i + 1, :, :, :, c] += F.grid_sample(heatmaps[c][i:i + 1, :, :, :], sample_grids[c], align_corners=True)
        cubes = torch.sum(torch.mul(cubes, bounding), dim=-1) / (torch.sum(bounding, dim=-1) + 1e-6)
        cubes = cubes.clone()
        cubes[cubes != cubes] = 0.0
        cubes = cubes.clamp(0.0, 1.0)

        cubes = cubes.view(batch_size, num_joints, self.cube_size[0], self.cube_size[1], self.cube_size[2]) 
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
            num_candidates, num_joints,
            clamp_cube_size_tensor[0], clamp_cube_size_tensor[1], clamp_cube_size_tensor[2],
            device=device
        )
        grids = torch.zeros(num_candidates, self.clamp_nbins, 3, device=device)

        grid_center = self.grid_center.to(device)
        grid_size = self.grid_size.to(device)

        centers_3d = clamp_grid_centers[:, :3]
        cube_grid_scale = grid_size / cube_size_tensor
        center_indices = torch.round((centers_3d - grid_center + grid_size / 2) / cube_grid_scale).long()
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
                clamp_cubes[j, :, d_s_x:d_e_x, d_s_y:d_e_y, d_s_z:d_e_z] = \
                    cubes[batch_indices[j], :, s_x:e_x, s_y:e_y, s_z:e_z]
                
            grids[j] = self.compute_grid(self.clamp_grid_size, centers_3d[j], self.clamp_cube_size, device=device)
        return clamp_cubes, grids

    def forward(self, heatmaps, grid_size, grid_centers, cube_size, batch_indices=None, flip_xcoords=None, meta=None):
        '''
        meta: List[Dict[str, Any]], len(meta) == number of cameras
        '''
        device = heatmaps[0].device
        if self.first_inference or (isinstance(meta, list) and len(meta) > 0 and 'cameras' in meta[0]):
            # self.cams = [meta[i]['cameras'] for i in range(len(meta))]
            self.sample_grids, self.bounding, self.grid = self.get_voxel(self.grid_size, self.grid_center, self.cube_size, device=device)
            self.first_inference = False
        cubes = self.project(heatmaps, self.sample_grids, self.bounding)
        if self.mode == "rootnet":
            return cubes
        if self.mode == "posenet":
            cubes, grids = self.clamp_cubes(cubes, grid_centers, batch_indices)

            return cubes, grids