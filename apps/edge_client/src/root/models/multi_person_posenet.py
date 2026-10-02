# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Derived from microsoft/voxelpose-pytorch; modified for DT_SERVER.
# SPDX-License-Identifier: MIT
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

from . import pose_resnet
from .cuboid_proposal_net_soft import CuboidProposalNetSoft

class MultiPersonPoseNet(nn.Module):
    """Edge-only pose model that stops after root proposal generation."""

    def __init__(self, backbone, cfg):
        super(MultiPersonPoseNet, self).__init__()
        self.backbone = backbone
        self.root_net = CuboidProposalNetSoft(cfg)

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

        if grid_centers is None:
            _, _, _, grid_centers = self.root_net(all_heatmaps)

        return None, all_heatmaps, grid_centers

def get_multi_person_pose_net(cfg):
    backbone = pose_resnet.get_pose_net(cfg)
    model = MultiPersonPoseNet(backbone, cfg)
    return model
