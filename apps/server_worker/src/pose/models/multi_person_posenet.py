# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

from . import pose_resnet
from .cuboid_proposal_net_soft import CuboidProposalNetSoft
from .pose_regression_net import PoseRegressionNet

from torch.utils.flop_counter import FlopCounterMode

def get_flops(model, inp):
    flop_counter = FlopCounterMode(mods=model, display=False, depth=None)
    with flop_counter:
        model(inp)
    return flop_counter.get_total_flops()

class MultiPersonPoseNetSSV(nn.Module):
    def __init__(self, backbone, cfg, transform=None, cams=None, inference_mode="posenet"):
        super(MultiPersonPoseNetSSV, self).__init__()
        if cams is not None and len(cams) != cfg.NUM_VIEWS:
            raise ValueError(
                f"Expected {cfg.NUM_VIEWS} calibrated cameras, got {len(cams)}"
            )

        self.backbone = backbone
        self.root_net = CuboidProposalNetSoft(cfg, transform, cams)
        self.pose_net = PoseRegressionNet(cfg, transform, cams)
        self.num_joints = cfg.NETWORK.NUM_JOINTS
        self.num_cand = cfg.MULTI_PERSON.MAX_PEOPLE_NUM
        self.num_views = cfg.NUM_VIEWS
        self.heatmap_size = tuple(int(v) for v in cfg.NETWORK.HEATMAP_SIZE)

        self.inference_mode = inference_mode

    def _cal_root_distance(self, root, distance):
        if distance is None or distance == 0:
            return True
        return torch.norm(root).item() < distance

    def _validate_inputs(self, all_heatmaps, grid_centers):
        if not isinstance(all_heatmaps, (list, tuple)) or not all_heatmaps:
            raise ValueError("input_heatmaps must be a non-empty view list")
        if len(all_heatmaps) != self.num_views:
            raise ValueError(
                f"Expected {self.num_views} heatmap views, "
                f"got {len(all_heatmaps)}"
            )

        batch_size = all_heatmaps[0].shape[0]
        heatmap_width, heatmap_height = self.heatmap_size
        expected_shape = (
            batch_size,
            self.num_joints,
            heatmap_height,
            heatmap_width,
        )
        for view_index, heatmap in enumerate(all_heatmaps):
            if tuple(heatmap.shape) != expected_shape:
                raise ValueError(
                    f"Heatmap view {view_index} must have shape "
                    f"{expected_shape}, got {tuple(heatmap.shape)}"
                )
            if heatmap.dtype != torch.float32:
                raise ValueError(
                    f"Heatmap view {view_index} must use float32"
                )
            if heatmap.device != all_heatmaps[0].device:
                raise ValueError("Every heatmap view must use one device")

        if grid_centers is None:
            return
        expected_root_shape = (batch_size, self.num_cand, 5)
        if tuple(grid_centers.shape) != expected_root_shape:
            raise ValueError(
                f"grid_centers must have shape {expected_root_shape}, "
                f"got {tuple(grid_centers.shape)}"
            )
        if grid_centers.dtype != torch.float32:
            raise ValueError("grid_centers must use float32")
        if grid_centers.device != all_heatmaps[0].device:
            raise ValueError("grid_centers and heatmaps must use one device")

    def forward(
        self,
        views=None,
        input_heatmaps=None,
        grid_centers=None,
        meta=None,
    ):
        if views is not None:
            all_heatmaps = []
            for view in views:
                heatmaps = self.backbone(view)
                all_heatmaps.append(heatmaps)
        else:
            all_heatmaps = input_heatmaps

        self._validate_inputs(all_heatmaps, grid_centers)
        batch_size = all_heatmaps[0].shape[0]
        device = all_heatmaps[0].device
        
        if grid_centers is None:
            _, grid_centers = self.root_net(all_heatmaps, meta)
        
        # result = {
        #     'grid_centers': grid_centers,
        #     'heatmaps': all_heatmaps,
        # }
        result = {
            'grid_centers': grid_centers,

        }
        if self.inference_mode == 'rootnet':
            return result

        pred = torch.zeros(batch_size, self.num_cand, self.num_joints, 5, device=device)
        pred[:, :, :, 3:] = grid_centers[:, :, 3:].reshape(batch_size, -1, 1, 2)
    
        valid_mask = grid_centers[:, :, 3] >= 0  # [batch_size, num_cand]
        if torch.any(valid_mask):
        # 유효한 후보자들의 인덱스 추출
            batch_indices, cand_indices = torch.where(valid_mask)
            if len(batch_indices) > 0:
                # 배치 처리를 위한 데이터 준비
                batch_grid_centers = grid_centers[batch_indices, cand_indices]  # [N, 5]
                batch_poses = self.pose_net(all_heatmaps, batch_grid_centers, batch_indices, meta)  # [N, num_joints, 3]

                pred[batch_indices, cand_indices, :, 0:3] = batch_poses

        result['pred'] = pred
        return result

def  get_multi_person_pose_net(cfg, transform=None, cams=None, inference_mode="posenet"):
    backbone = pose_resnet.get_pose_net(cfg)
    model = MultiPersonPoseNetSSV(backbone, cfg, transform, cams, inference_mode)
    return model
