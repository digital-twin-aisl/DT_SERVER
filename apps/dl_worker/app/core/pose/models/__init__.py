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