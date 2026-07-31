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
config.CALIBRATION_PATH = '/home/min4090/Server_DT/data'
# lod
config.DISTANCE = 1500
config.LOD=[2,1,0]

config.ZONE=[0,1]
config.SERVER='192.168.0.73:20522'
config.ZMQ_SERVER = 'localhost'
config.ZMQ_PORT = 5555
config.ReidDataPATH = "/home/min4090/Server_DT/"
config.time_diff = 0.15
config.MESH=edict()
config.MESH.CKPT = "/home/min4090/Server_DT/models/mesh.pth.tar"
config.GRAPH_JOINT_NUM = 17
config.GRAPH_SKELETON = (
        (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2),
        (2, 3), (0, 4), (4, 5), (5, 6))
config.GRAPH_FLIP_PAIRS = ((1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13))
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