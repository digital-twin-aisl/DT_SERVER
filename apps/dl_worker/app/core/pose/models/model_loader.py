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
from . import multi_person_posenet_ssv
from ..core.config import config, update_config

def load_model(cfg_file, model_checkpoint, gpus) -> torch.nn.Module:
    update_config(cfg_file)
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED    # set to False if you want to disable cuDNN
    torch.backends.cudnn.benchmark = config.CUDNN.BENCHMARK    # set to True if input sizes are the same
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC    # set to True if you want reproducible results
    try:
        pose_model = multi_person_posenet_ssv.get_multi_person_pose_net(
                config, is_train=True)
        with torch.no_grad():
            pose_model = torch.nn.DataParallel(pose_model, device_ids=gpus).cuda()
        pose_model.module.load_state_dict(torch.load(model_checkpoint), strict=False)
        print("=> loaded model from {}".format(model_checkpoint))
        return pose_model
    except:
        raise ValueError("Fail to load model from {}".format(model_checkpoint))