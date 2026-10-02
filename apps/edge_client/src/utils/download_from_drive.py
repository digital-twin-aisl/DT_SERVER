# SPDX-FileCopyrightText: 2025-2026 DT_SERVER contributors
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Download the model files the edge needs into their default locations.

Trained PoseNet weights are not bundled with the source. Set DT_POSENET_URL to
a URL (e.g. https://drive.google.com/uc?id=<file id>) for a checkpoint you are licensed to
use, or place the file at apps/edge_client/models/POC_posenet.pth.tar yourself.
"""
import os
import os.path as osp

import gdown

PWD = osp.dirname(osp.abspath(__file__))
MODELS_DIR = osp.join(PWD, '..', '..', 'models')
FASTREID_WEIGHTS_DIR = osp.join(PWD, '..', 'reid', 'fast-reid', 'weights')

# FastReID's published Market-1501 baseline (Apache-2.0).
FASTREID_URL = (
    'https://github.com/JDAI-CV/fast-reid/releases/download/'
    'v0.1.1/market_bot_R50-ibn.pth'
)


def download(url, output_path):
    if osp.exists(output_path):
        print(f"{osp.basename(output_path)} already exists")
        return
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    gdown.download(url, output_path, quiet=False)
    print(f"Downloaded {osp.basename(output_path)}")


def run():
    posenet = osp.join(MODELS_DIR, 'POC_posenet.pth.tar')
    posenet_url = os.environ.get('DT_POSENET_URL', '')
    if posenet_url:
        download(posenet_url, posenet)
    elif not osp.exists(posenet):
        print("POC_posenet.pth.tar is missing; set DT_POSENET_URL or copy it "
              f"to {osp.normpath(posenet)}")
    download(FASTREID_URL, osp.join(FASTREID_WEIGHTS_DIR, 'market_bot_R50-ibn.pth'))
    print("Download completed")


if __name__ == '__main__':
    run()
