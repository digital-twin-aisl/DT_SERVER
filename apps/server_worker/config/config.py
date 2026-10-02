# SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
# SPDX-License-Identifier: LGPL-2.1-or-later
from easydict import EasyDict as edict
import numpy as np
import yaml


config = edict()
config.NUM_VIEWS = 4

# POSENET
config.POSENET = edict()
config.POSENET.CKPT = "models/POC_posenet.pth.tar"
config.POSENET.CONFIG = "config/cam4_posenet.yaml"
config.POSENET.TENSORRT = False
config.POSENET.INPUT_SHAPE = None
config.ZMQ_SERVER = 'localhost'
config.ZMQ_PORT = 5555
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
