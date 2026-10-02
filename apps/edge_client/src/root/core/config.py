# SPDX-FileCopyrightText: 2025-2026 DT_SERVER contributors
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Runtime configuration for the edge heatmap and root-proposal model."""

import numpy as np
import yaml
from easydict import EasyDict as edict


config = edict()
config.NUM_VIEWS = 4
config.BATCH_SIZE = 1

config.CUDNN = edict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True

config.NETWORK = edict()
config.NETWORK.NUM_JOINTS = 15
config.NETWORK.HEATMAP_SIZE = np.array([80, 80])
config.NETWORK.IMAGE_SIZE = np.array([320, 320])
config.NETWORK.IMAGE_SIZE_ORIG = np.array([1920, 1080])
config.NETWORK.ROOTNET_ROOTHM = True

config.POSE_RESNET = edict()
config.POSE_RESNET.NUM_LAYERS = 50
config.POSE_RESNET.DECONV_WITH_BIAS = False
config.POSE_RESNET.NUM_DECONV_LAYERS = 3
config.POSE_RESNET.NUM_DECONV_FILTERS = [256, 256, 256]
config.POSE_RESNET.NUM_DECONV_KERNELS = [4, 4, 4]
config.POSE_RESNET.FINAL_CONV_KERNEL = 1

config.DATASET = edict()
config.DATASET.ROOTIDX = 2

config.MULTI_PERSON = edict()
config.MULTI_PERSON.SPACE_SIZE = np.array([4000.0, 5200.0, 2400.0])
config.MULTI_PERSON.SPACE_CENTER = np.array([300.0, 300.0, 300.0])
config.MULTI_PERSON.INITIAL_CUBE_SIZE = np.array([24, 32, 16])
config.MULTI_PERSON.MAX_PEOPLE_NUM = 10
config.MULTI_PERSON.THRESHOLD = 0.1
config.MULTI_PERSON.ROOT_NMS_DISTANCE = 500.0


def _update_dict(section, values):
    if section == "NETWORK":
        for key in ("HEATMAP_SIZE", "IMAGE_SIZE", "IMAGE_SIZE_ORIG"):
            if key in values:
                value = values[key]
                values[key] = np.array(
                    [value, value] if isinstance(value, int) else value
                )
    for key, value in values.items():
        if key not in config[section]:
            raise ValueError(f"{section}.{key} not exist in config.py")
        config[section][key] = value


def update_config(config_file):
    with open(config_file, encoding="utf-8") as stream:
        experiment = edict(yaml.load(stream, Loader=yaml.FullLoader))
    for section, values in experiment.items():
        if section not in config:
            raise ValueError(f"{section} not exist in config.py")
        if isinstance(values, dict):
            _update_dict(section, values)
        else:
            config[section] = values
