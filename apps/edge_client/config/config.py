from ast import literal_eval

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
config.CAMERAS = [
    {
        "id": "0",
        "url": "rtsp://admin:byda13245@10.1.1.3:554",
    },
    {
        "id": "1",
        "url": "rtsp://admin:byda13245@10.1.1.2:554",
    },
    {
        "id": "2",
        "url": "rtsp://admin:byda13245@10.1.1.4:554",
    },
    {
        "id": "3",
        "url": "rtsp://admin:byda13245@10.1.1.5:554",
    }
]
config.SERVER = "localhost:7447"
config.OUTPUT_TOPIC = "0_edge"
config.YOLO = edict()
config.YOLO.MODEL = "models/yolo11n-pose.pt"


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
