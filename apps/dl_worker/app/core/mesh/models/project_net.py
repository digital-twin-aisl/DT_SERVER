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

import torch
import torch.nn as nn

from ..core.config import cfg


class OptimzeCamLayer(nn.Module):
    def __init__(self, crop_size):
        super(OptimzeCamLayer, self).__init__()

        self.img_res = crop_size / 2
        self.cam_param = nn.Parameter(torch.rand((1,3)))

    def forward(self, pose3d):
        output = pose3d[:, :, :2] + self.cam_param[None, :, 1:]
        output = output * self.cam_param[None, :, :1] * self.img_res + self.img_res
        return output


def get_model(crop_size):
    model = OptimzeCamLayer(crop_size)

    return model


