# SPDX-FileCopyrightText: 2025-2026 DT_SERVER contributors
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Download the server PoseNet checkpoint into its default location.

Trained weights are not bundled with the source. Set DT_POSENET_URL to a URL
(e.g. https://drive.google.com/uc?id=<file id>) for a checkpoint you are licensed to use, or
place the file at apps/server_worker/models/POC_posenet.pth.tar yourself.
"""
import os
import os.path as osp
import sys

import gdown

PWD = osp.dirname(osp.abspath(__file__))
POSENET_PATH = osp.join(PWD, '..', '..', 'models', 'POC_posenet.pth.tar')


def run():
    if osp.exists(POSENET_PATH):
        print("POC_posenet.pth.tar already exists")
        return
    url = os.environ.get('DT_POSENET_URL', '')
    if not url:
        sys.exit("Set DT_POSENET_URL or copy the checkpoint to "
                 f"{osp.normpath(POSENET_PATH)}")
    os.makedirs(osp.dirname(POSENET_PATH), exist_ok=True)
    gdown.download(url, POSENET_PATH, quiet=False)
    print("Download completed")


if __name__ == '__main__':
    run()
