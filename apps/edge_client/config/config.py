from ast import literal_eval
from pathlib import Path

from easydict import EasyDict as edict
import numpy as np
import yaml

config = edict()
config.NUM_VIEWS = 4
config.BATCHSIZE = 1

# POSENET
config.POSENET = edict()
config.POSENET.CKPT = "models/POC_posenet.pth.tar"
config.POSENET.CONFIG = "config/cam4_posenet.yaml"
config.POSENET.TENSORRT = False
config.POSENET.INPUT_SHAPE = [960, 512]

# Example
config.EXAMPLE = edict()
config.EXAMPLE.SOURCES = None
config.EXAMPLE.START_FRAME = 0
config.EXAMPLE.END_FRAME = None

# lod
config.DISTANCE = 1500
config.CAMERAS = []
local_camera_config = Path(__file__).with_name("cameras.local.yaml")
if local_camera_config.exists():
    local_config = yaml.safe_load(local_camera_config.read_text()) or {}
    config.CAMERAS = local_config.get("CAMERAS", [])
config.SERVER = "localhost:7447"
config.OUTPUT_TOPIC = "0_edge"
config.YOLO = edict()
config.YOLO.MODEL = "models/yolo11n-pose.pt"
config.YOLO.IMAGE_SIZE = 640


def _update_dict(k, v):
    if k == "DATASET":
        for field in ("MEAN", "STD"):
            if v.get(field):
                v[field] = np.array(
                    [literal_eval(x) if isinstance(x, str) else x for x in v[field]]
                )
    elif k == "NETWORK":
        for field in ("HEATMAP_SIZE", "IMAGE_SIZE"):
            if field in v:
                value = v[field]
                v[field] = np.array([value, value] if isinstance(value, int) else value)

    if k not in config or not isinstance(config[k], dict):
        config[k] = edict()
    config[k].update(v)


def update_config(config_file):
    with open(config_file) as f:
        exp_config = edict(yaml.safe_load(f))
        for k, v in exp_config.items():
            if isinstance(v, dict):
                _update_dict(k, v)
            elif k == "SCALES" and k in config:
                config[k][0] = tuple(v)
            else:
                config[k] = v
