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

from .v2v_net import V2VNet
from .project_layer import ProjectLayer


class SoftArgmaxLayer(nn.Module):
    def __init__(self, cfg):
        super(SoftArgmaxLayer, self).__init__()
        self.beta = cfg.NETWORK.BETA

    def forward(self, x, grids):
        batch_size = x.size(0)
        channel = x.size(1)
        x = x.reshape(batch_size, channel, -1, 1)
        # x = F.softmax(x, dim=2)
        x = F.softmax(self.beta * x, dim=2)
        grids = grids.unsqueeze(1)
        x = torch.mul(x, grids)
        x = torch.sum(x, dim=2)
        return x


class PoseRegressionNet(nn.Module):
    def __init__(self, cfg):
        super(PoseRegressionNet, self).__init__()
        self.grid_size = cfg.PICT_STRUCT.GRID_SIZE
        self.cube_size = cfg.PICT_STRUCT.CUBE_SIZE

        self.project_layer = ProjectLayer(cfg, "posenet")
        self.v2v_net = V2VNet(cfg.NETWORK.NUM_JOINTS, cfg.NETWORK.NUM_JOINTS)
        self.soft_argmax_layer = SoftArgmaxLayer(cfg)

    def forward(self, all_heatmaps, batch_grid_centers, batch_indices, meta=None):
        batch_size = all_heatmaps[0].shape[0]
        num_joints = all_heatmaps[0].shape[1]
        num_grid_centers = batch_grid_centers.shape[0]
        device = all_heatmaps[0].device
        pred = torch.zeros(batch_size, num_joints, 3, device=device)
        cubes, grids = self.project_layer(all_heatmaps,
                                          self.grid_size, batch_grid_centers, self.cube_size, batch_indices= batch_indices, meta=meta)

        # index = grid_centers[:, 3] >= 0
        valid_cubes = torch.zeros_like(cubes, device=device)
        for i in range(num_grid_centers):
            valid_cubes[i:i+1] = self.v2v_net(cubes[i:i+1])
        # valid_cubes = self.v2v_net(cubes)
        pred = self.soft_argmax_layer(valid_cubes, grids)
        return pred