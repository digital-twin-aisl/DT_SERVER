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

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from easydict import EasyDict as edict
import numpy as np
import yaml

# from SelfPose3d.lib.core.config import config as cfg

config = edict()
config.NUM_VIEWS = 4
config.BATCHSIZE = 1

# POSENET
config.POSENET = edict()
config.POSENET.CKPT = "models/POC_posenet.pth.tar"
config.POSENET.CONFIG = "config/cam4_posenet.yaml"
config.POSENET.TENSORRT = False
config.POSENET.INPUT_SHAPE = None

# Example
config.EXAMPLE = edict()
config.EXAMPLE.SOURCES = None
config.EXAMPLE.START_FRAME = 0
config.EXAMPLE.END_FRAME = None
config.CALIBRATION_PATH = '/data_disk/home/minseok/Server_DT/data'
# lod
config.DISTANCE = 1500
config.LOD=[2,1,0]

config.ZONE=[1,2,3]
config.SERVER='localhost'

def _update_dict(k, v):
    if k == 'DATASET':
        if 'MEAN' in v and v['MEAN']:
            v['MEAN'] = np.array(
                [eval(x) if isinstance(x, str) else x for x in v['MEAN']])
        if 'STD' in v and v['STD']:
            v['STD'] = np.array(
                [eval(x) if isinstance(x, str) else x for x in v['STD']])
    if k == 'NETWORK':
        if 'HEATMAP_SIZE' in v:
            if isinstance(v['HEATMAP_SIZE'], int):
                v['HEATMAP_SIZE'] = np.array(
                    [v['HEATMAP_SIZE'], v['HEATMAP_SIZE']])
            else:
                v['HEATMAP_SIZE'] = np.array(v['HEATMAP_SIZE'])
        if 'IMAGE_SIZE' in v:
            if isinstance(v['IMAGE_SIZE'], int):
                v['IMAGE_SIZE'] = np.array([v['IMAGE_SIZE'], v['IMAGE_SIZE']])
            else:
                v['IMAGE_SIZE'] = np.array(v['IMAGE_SIZE'])
    # add new keys
    if k not in config or not isinstance(config[k], dict):
        config[k] = edict()
    for vk, vv in v.items():
        config[k][vk] = vv
        # if vk in config[k]:
        #     config[k][vk] = vv
        # else:
        #     raise ValueError("{}.{} not exist in config.py".format(k, vk))

def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(k, v)
                else:
                    if k == 'SCALES':
                        config[k][0] = (tuple(v))
                    else:
                        config[k] = v
            else:
                # raise ValueError("{} not exist in config.py".format(k))
                if isinstance(v, dict):
                    config[k] = edict(v)
                else:
                    config[k] = v
