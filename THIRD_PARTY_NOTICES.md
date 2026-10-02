# Third-Party Notices

DT_SERVER is licensed under LGPL-2.1-or-later (see `LICENSE`). This file lists
third-party material that is included in, derived into, or used by DT_SERVER.

## 1. Code derived into this repository

### VoxelPose (Microsoft) — MIT License

The following files are derived from
[microsoft/voxelpose-pytorch](https://github.com/microsoft/voxelpose-pytorch)
and keep their original MIT notice (SPDX identifier `MIT`):

- `apps/server_worker/src/pose/models/*.py` (except `__init__.py`)
- `apps/server_worker/src/pose/core/proposal.py`, `apps/server_worker/src/pose/utils/cameras.py`
- `apps/edge_client/src/root/models/*.py` (except `__init__.py`)
- `apps/edge_client/src/root/core/proposal.py`, `apps/edge_client/src/root/utils/cameras.py`

```
MIT License

Copyright (c) Microsoft Corporation.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE
```

The multi-view root/pose network design follows VoxelPose (Tu et al., ECCV 2020)
and the root-heatmap formulation studied in SelfPose3d (Srivastav et al., CVPR 2024).
No SelfPose3d source code is included.

## 2. Git submodules (not distributed; fetched from upstream on demand)

| Path | Upstream | License |
| --- | --- | --- |
| `apps/edge_client/src/reid/fast-reid` | https://github.com/JDAI-CV/fast-reid | Apache-2.0 |
| `apps/calibration_worker/vggt-omega` | https://github.com/facebookresearch/vggt-omega | FAIR Noncommercial Research License (non-commercial use only) |

`apps/calibration_worker/patches/vggt-omega-patch-tokens.patch` modifies the
VGGT-Omega submodule and is therefore distributed under the FAIR Noncommercial
Research License (`LICENSES/LicenseRef-FAIR-Noncommercial-Research.txt`), not
under the LGPL. It is needed only for distributed calibration.

## 3. Optional components installed by the user

| Component | License | Notes |
| --- | --- | --- |
| Ultralytics YOLO (`ultralytics`, YOLO11 pose weights) | AGPL-3.0 or Ultralytics Enterprise | Edge ReID only; `apps/edge_client/requirements-reid-ultralytics.txt` |
| VGGT-Omega checkpoints | FAIR Noncommercial Research License | Hugging Face `facebook/VGGT-Omega` |
| FastReID Market1501 weights | Apache-2.0 | Downloaded from the FastReID release page |
| NVIDIA TensorRT, torch2trt, JetPack | NVIDIA SLA / MIT (torch2trt) | Edge acceleration |
| NVIDIA Isaac Sim | NVIDIA Omniverse License | `apps/isaac_sim_client` runs inside Isaac Sim |

## 4. Runtime dependencies (installed from PyPI / npm, not vendored)

| Package | License |
| --- | --- |
| eclipse-zenoh | EPL-2.0 OR Apache-2.0 |
| torch, torchvision | BSD-3-Clause |
| numpy, scipy, scikit-learn, pyzmq, zstandard, httpx | BSD-3-Clause |
| opencv-contrib-python-headless, onnx, safetensors, yacs, viser | Apache-2.0 |
| easydict | LGPL-3.0 |
| usd-core (OpenUSD) | Tomorrow Open Source Technology License 1.0 |
| fastapi, PyYAML, einops, gdown, tabulate, termcolor | MIT |
| uvicorn, websockets | BSD-3-Clause |
| Pillow | MIT-CMU (HPND) |
| matplotlib | Matplotlib License (PSF-based) |
| tqdm | MPL-2.0 AND MIT |
| three (npm) | MIT |
| esbuild, @gltf-transform/cli (npm, build only) | MIT |

The built viewer bundle `apps/frontend_api/app/static/dist/viewer.js` (generated
by `npm run build`, not committed) contains three.js under the MIT License.

Versions are pinned in `requirements*.txt`, `apps/*/requirements*.txt` and
`apps/frontend_api/web/package-lock.json`. Check each package's metadata for the
authoritative license of the version you install.
