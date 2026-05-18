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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import torch
import torch.nn as nn
import time

from . import pose_resnet
from .cuboid_proposal_net_soft import CuboidProposalNetSoft
from .pose_regression_net import PoseRegressionNet

from torch.utils.flop_counter import FlopCounterMode

stream = torch.cuda.Stream()

def get_flops(model, inp):
    flop_counter = FlopCounterMode(mods=model, display=False, depth=None)
    with flop_counter:
        model(inp)
    return flop_counter.get_total_flops()

class MultiPersonPoseNetSSV(nn.Module):
    def __init__(self, backbone, cfg, inference_mode="posenet"):
        super(MultiPersonPoseNetSSV, self).__init__()
        self.backbone = backbone
        self.root_net = CuboidProposalNetSoft(cfg)
        self.pose_net = PoseRegressionNet(cfg)
        self.num_joints = cfg.NETWORK.NUM_JOINTS
        self.num_cand = cfg.MULTI_PERSON.MAX_PEOPLE_NUM

        self.inference_mode = inference_mode

    def _cal_root_distance(self, root, distance):
        if distance is None or distance == 0:
            return True
        return torch.norm(root).item() < distance

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

        # 이상 포즈 제거 (0929)
        z_vals = pred[:, :, [8, 14], 2]
        min_z = torch.min(z_vals, dim=2)[0]
        invalid_mask = (min_z < -50) | (min_z > 250)
        grid_centers[:, :, 3] = torch.where(invalid_mask, -1, grid_centers[:, :, 3])
        pred[:, :, :, 3] = torch.where(invalid_mask.unsqueeze(-1), -1, pred[:, :, :, 3])
        
        result['pred'] = pred
        return result

def  get_multi_person_pose_net(cfg, inference_mode="posenet"):
    backbone = pose_resnet.get_pose_net(cfg)
    model = MultiPersonPoseNetSSV(backbone, cfg, inference_mode)
    return model