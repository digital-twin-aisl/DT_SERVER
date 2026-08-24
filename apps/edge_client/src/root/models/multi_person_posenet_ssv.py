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

class MultiPersonPoseNetSSV(nn.Module):
    def __init__(self, backbone, cfg, inference_mode="rootnet"):
        super(MultiPersonPoseNetSSV, self).__init__()
        self.inference_mode = inference_mode
        self.backbone = backbone
        self.root_net = CuboidProposalNetSoft(cfg)
        # Edge inference stops at root proposals.  Keep the pose regression
        # network entirely off the edge GPU; the server owns that stage.
        self.pose_net = (
            None if inference_mode == "rootnet" else PoseRegressionNet(cfg)
        )
        self.num_joints = cfg.NETWORK.NUM_JOINTS
        self.num_cand = cfg.MULTI_PERSON.MAX_PEOPLE_NUM
        self.num_views = cfg.NUM_VIEWS

    def _cal_root_distance(self, root, distance):
        if distance is None or distance == 0:
            return True
        return torch.norm(root).item() < distance

    def forward(
        self,
        views=None,
        input_heatmaps=None,
        grid_centers=None,
    ):
        if views is not None:
            # Every edge frame contains one tensor per camera.  Running the
            # backbone once per view leaves the GPU idle between four small
            # TensorRT enqueues.  Concatenate the camera dimension, execute a
            # single larger batch, then restore the model's view-list API.
            view_batch_sizes = [view.shape[0] for view in views]
            batched_heatmaps = self.backbone(torch.cat(views, dim=0))
            all_heatmaps = list(
                batched_heatmaps.split(view_batch_sizes, dim=0)
            )
        else:
            all_heatmaps = input_heatmaps

        batch_size = all_heatmaps[0].shape[0]
        device = all_heatmaps[0].device
        
        if grid_centers is None:
            _, _, _, grid_centers = self.root_net(all_heatmaps)

        if self.inference_mode == 'rootnet':
            return None, all_heatmaps, grid_centers

        pred = torch.zeros(batch_size, self.num_cand, self.num_joints, 5, device=device)
        pred[:, :, :, 3:] = grid_centers[:, :, 3:].reshape(batch_size, -1, 1, 2)
    
        valid_mask = grid_centers[:, :, 3] >= 0  # [batch_size, num_cand]
        if torch.any(valid_mask):
        # 유효한 후보자들의 인덱스 추출
            batch_indices, cand_indices = torch.where(valid_mask)
            if len(batch_indices) > 0:
                # 배치 처리를 위한 데이터 준비
                batch_grid_centers = grid_centers[batch_indices, cand_indices]  # [N, 5]
                batch_poses = self.pose_net(all_heatmaps, batch_grid_centers, batch_indices)  # [N, num_joints, 3]

                pred[batch_indices, cand_indices, :, 0:3] = batch_poses
        return pred, all_heatmaps, grid_centers

def  get_multi_person_pose_net(cfg, inference_mode="posenet"):
    backbone = pose_resnet.get_pose_net(cfg)
    model = MultiPersonPoseNetSSV(backbone, cfg, inference_mode)
    return model
