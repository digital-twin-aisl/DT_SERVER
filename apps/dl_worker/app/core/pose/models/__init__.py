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


import sys
# print(sys.path)  # test

from .......dl_worker.app.core.pose.models import pose_resnet
# import models.pose_resnet_dpi
from .......dl_worker.app.core.pose.models import v2v_net
from .......dl_worker.app.core.pose.models import project_layer
from .......dl_worker.app.core.pose.models import cuboid_proposal_net_soft
from .......dl_worker.app.core.pose.models import pose_regression_net
from .......dl_worker.app.core.pose.models import multi_person_posenet_ssv
from .model_loader import load_model