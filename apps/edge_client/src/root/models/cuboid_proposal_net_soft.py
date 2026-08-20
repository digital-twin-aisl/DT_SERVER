# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from .v2v_net import V2VNet
from .project_layer import ProjectLayer
from ..core.proposal import nms


class ProposalLayerSoft(nn.Module):
    def __init__(self, cfg, cube_size=None, workspace=None):
        super(ProposalLayerSoft, self).__init__()
        self.workspace = workspace
        self.grid_size = torch.as_tensor(
            cfg.MULTI_PERSON.SPACE_SIZE
            if workspace is None
            else workspace.size_mm
        )
        self.cube_size = torch.as_tensor(
            (
                cfg.MULTI_PERSON.INITIAL_CUBE_SIZE
                if cube_size is None
                else cube_size
            )
            if workspace is None
            else workspace.cube_size
        )
        self.grid_center = torch.as_tensor(
            cfg.MULTI_PERSON.SPACE_CENTER
            if workspace is None
            else workspace.center_mm
        )
        self.xy_axes = (
            [[1.0, 0.0], [0.0, 1.0]]
            if workspace is None
            else workspace.xy_axes
        )
        self.num_cand = cfg.MULTI_PERSON.MAX_PEOPLE_NUM
        self.root_id = cfg.DATASET.ROOTIDX
        self.num_joints = cfg.NETWORK.NUM_JOINTS
        self.threshold = cfg.MULTI_PERSON.THRESHOLD

        self._cached_constants = {}

    def filter_proposal(self, topk_index, gt_3d, num_person):
        batch_size = topk_index.shape[0]
        cand_num = topk_index.shape[1]
        cand2gt = torch.zeros(batch_size, cand_num)

        for i in range(batch_size):
            cand = topk_index[i].reshape(cand_num, 1, -1)
            gt = gt_3d[i, : num_person[i]].reshape(1, num_person[i], -1)

            dist = torch.sqrt(torch.sum((cand - gt) ** 2, dim=-1))
            min_dist, min_gt = torch.min(dist, dim=-1)

            cand2gt[i] = min_gt
            cand2gt[i][min_dist > 500.0] = -1.0

        return cand2gt

    def get_real_loc(
        self,
        index,
        grid_center=None,
        grid_size=None,
        xy_axes=None,
    ):
        device = index.device
        center = torch.as_tensor(
            self.grid_center if grid_center is None else grid_center,
            dtype=torch.float,
            device=device,
        )
        size = torch.as_tensor(
            self.grid_size if grid_size is None else grid_size,
            dtype=torch.float,
            device=device,
        )
        axes = torch.as_tensor(
            self.xy_axes if xy_axes is None else xy_axes,
            dtype=torch.float,
            device=device,
        )
        cube_size = self.cube_size.to(device=device, dtype=torch.float)
        local = index.float() * (size / (cube_size - 1.0)) - size / 2.0
        world = torch.empty_like(local)
        world[:, :, :2] = local[:, :, :2] @ axes + center[:2]
        world[:, :, 2] = local[:, :, 2] + center[2]
        return world

    def forward(
        self,
        root_cubes,
        grid_center=None,
        grid_size=None,
        xy_axes=None,
    ):
        batch_size = root_cubes.shape[0]

        topk_values, topk_unravel_index = nms(root_cubes.detach(), self.num_cand)
        topk_unravel_index = self.get_real_loc(
            topk_unravel_index,
            grid_center,
            grid_size,
            xy_axes,
        )

        grid_centers = torch.zeros(
            batch_size, self.num_cand, 5, device=root_cubes.device
        )
        grid_centers[:, :, 0:3] = topk_unravel_index
        grid_centers[:, :, 4] = topk_values

        grid_centers[:, :, 3] = (
            topk_values > self.threshold
        ).float() - 1.0  # if ground-truths are not available.

        return grid_centers


class CuboidProposalNetSoft(nn.Module):
    def __init__(self, cfg):
        super(CuboidProposalNetSoft, self).__init__()
        self.workspace = getattr(cfg, "EDGE_WORKSPACE", None)
        self.cube_size = (
            cfg.MULTI_PERSON.INITIAL_CUBE_SIZE
            if self.workspace is None
            else self.workspace.cube_size
        )
        self.root_id = cfg.DATASET.ROOTIDX
        self.rootnet_roothm = cfg.NETWORK.ROOTNET_ROOTHM
        self.rootnet_train_synth = cfg.NETWORK.ROOTNET_TRAIN_SYNTH
        self.max_num_people = cfg.MULTI_PERSON.MAX_PEOPLE_NUM
        self.root_nms_distance = float(cfg.MULTI_PERSON.ROOT_NMS_DISTANCE)
        spatial_context = getattr(cfg, "SPATIAL_CONTEXT", None)
        self.ground_surface = (
            None if spatial_context is None else spatial_context.ground_surface
        )
        root_clearance = (
            (400.0, 1400.0)
            if self.workspace is None
            else self.workspace.root_clearance_mm
        )
        self.root_clearance_min_mm = float(root_clearance[0])
        self.root_clearance_max_mm = float(root_clearance[1])
        self.grid_size = (
            cfg.MULTI_PERSON.SPACE_SIZE
            if self.workspace is None
            else self.workspace.size_mm
        )
        self.grid_center = (
            cfg.MULTI_PERSON.SPACE_CENTER
            if self.workspace is None
            else self.workspace.center_mm
        )
        self.xy_axes = (
            [[1.0, 0.0], [0.0, 1.0]]
            if self.workspace is None
            else self.workspace.xy_axes
        )

        self.project_layer = ProjectLayer(
            cfg,
            mode="rootnet",
            cube_size=self.cube_size,
            workspace=self.workspace,
        )
        if self.rootnet_roothm:
            self.v2v_net = V2VNet(1, 1)
        else:
            self.v2v_net = V2VNet(cfg.NETWORK.NUM_JOINTS, 1)
        self.proposal_layer = ProposalLayerSoft(
            cfg,
            cube_size=self.cube_size,
            workspace=self.workspace,
        )

    def forward(self, all_heatmaps, flip_xcoords=None):
        # all_heatmaps_copy = [
        #     a[:, self.root_id, :, :][:, None] for a in all_heatmaps
        # ]
        all_heatmaps_copy = [
            a[:, self.root_id, :, :][:, None].clone() for a in all_heatmaps
        ]

        if self.workspace is None:
            initial_cubes, voxel_valid = self.project_layer(
                all_heatmaps_copy,
                self.grid_size,
                [self.grid_center],
                self.cube_size,
                flip_xcoords=flip_xcoords,
                xy_axes=self.xy_axes,
                return_valid_mask=True,
            )
        else:
            initial_cubes, voxel_valid = self.project_layer.project_workspace(
                all_heatmaps_copy,
                flip_xcoords=flip_xcoords,
            )
        root_cubes = self.v2v_net(initial_cubes).squeeze(1)
        root_cubes = root_cubes.masked_fill(
            ~voxel_valid.unsqueeze(0),
            torch.finfo(root_cubes.dtype).min,
        )
        grid_centers = self.proposal_layer(root_cubes)
        grid_centers = self.filter_proposals(grid_centers)

        return root_cubes, None, None, grid_centers

    def filter_proposals(self, candidates):
        batch_size = candidates.shape[0]
        terrain_valid = torch.ones(
            candidates.shape[:2],
            dtype=torch.bool,
            device=candidates.device,
        )
        ground_surface = getattr(self, "ground_surface", None)
        if ground_surface is not None:
            positions = candidates[:, :, :3]
            ground_heights = ground_surface.heights_mm_torch(
                positions[:, :, :2].reshape(-1, 2)
            ).view(candidates.shape[:2])
            clearance = positions[:, :, 2] - ground_heights
            terrain_valid = (
                torch.isfinite(clearance)
                & (clearance >= self.root_clearance_min_mm)
                & (clearance <= self.root_clearance_max_mm)
            )
        merged = torch.zeros(
            batch_size,
            self.max_num_people,
            5,
            dtype=candidates.dtype,
            device=candidates.device,
        )
        merged[:, :, 3] = -1.0

        for batch_index in range(batch_size):
            ordered = torch.argsort(
                candidates[batch_index, :, 4],
                descending=True,
            )
            selected = []
            for candidate_index in ordered.tolist():
                candidate = candidates[batch_index, candidate_index]
                if candidate[3] < 0 or not terrain_valid[batch_index, candidate_index]:
                    continue
                if selected:
                    positions = torch.stack([item[:3] for item in selected])
                    if torch.any(
                        torch.linalg.vector_norm(
                            positions - candidate[:3],
                            dim=1,
                        )
                        < self.root_nms_distance
                    ):
                        continue
                selected.append(candidate)
                if len(selected) == self.max_num_people:
                    break
            for output_index, candidate in enumerate(selected):
                merged[batch_index, output_index] = candidate
        return merged
