# 서버측 Python 환경 설정

이 문서는 서버에서 실행하는 다음 구성요소의 환경을 다룹니다.

- `apps/server_worker`: CUDA 기반 3D pose 추론 및 Zenoh scene 발행
- `apps/edge_manager`: Zenoh edge 등록/승인 CLI
- `apps/calibration_worker`: VGGT-Omega와 ArUco marker tree를 이용한 카메라 정합
- `apps/isaac_sim_client`: Isaac Sim 전용 scene 구독 및 USD 반영

`server_worker`, `edge_manager`, `calibration_worker`는 루트의 동일한
`.venv-server`를 사용합니다. `isaac_sim_client`만은 `omni`, `pxr` ABI 때문에 이
가상환경에서 실행하지 않고 **Isaac Sim에 포함된 `python.sh`**를 사용합니다.

## 1. 요구 환경

- Ubuntu 22.04 권장
- Python 3.10 또는 3.11 (`server_worker`는 Python 3.8에서 실행되지 않음)
- NVIDIA GPU, 호환 드라이버와 CUDA 지원 PyTorch
- Git과 Python venv
- Isaac client 실행 시 Isaac Sim 4.2.0 및 별도 GPU/그래픽 환경

먼저 GPU와 Python을 확인합니다.

```bash
nvidia-smi
python3.10 --version
```

Ubuntu에서 기본 도구가 없다면 다음 패키지가 필요합니다.

```bash
sudo apt update
sudo apt install -y git python3.10 python3.10-venv python3.10-dev build-essential
```

> `calibration_worker`는 GPU 메모리를 많이 사용하므로 보통 상시 추론 중인
> `server_worker`와 동시에 실행하지 않습니다.

## 2. 공통 서버 가상환경 생성

아래 명령은 모두 저장소 루트에서 실행합니다.

```bash
cd /path/to/DT_SERVER
python3.10 -m venv .venv-server
source .venv-server/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### CUDA PyTorch를 먼저 설치

`torch`/`torchvision`은 서버 드라이버와 CUDA에 맞는 wheel을 먼저 설치해야
합니다. [PyTorch 설치 선택기](https://pytorch.org/get-started/locally/)에서 Linux,
Pip, Python과 서버에 맞는 CUDA 버전을 선택해 표시되는 명령을 실행합니다.
이 프로젝트는 VGGT-Omega 요구사항에 맞춰 `torch>=2.3`,
`torchvision>=0.18`을 요구합니다.

설치 직후 CUDA가 실제로 잡히는지 확인합니다.

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

마지막 값이 `False`이면 나머지 패키지를 설치하기 전에 드라이버와 PyTorch wheel
조합을 수정해야 합니다.

### 나머지 공통 패키지 설치

```bash
python -m pip install -r requirements.txt
```

`requirements.txt`는 ArUco가 포함된 headless OpenCV 하나만 설치합니다.
`opencv-python`, `opencv-python-headless`, `opencv-contrib-python`을 추가로 함께
설치하면 같은 `cv2` 경로를 덮어쓸 수 있으므로 혼용하지 않습니다.

## 3. Calibration Worker 추가 준비

VGGT-Omega는 Git 서브모듈입니다. 저장소에 남아 있는 다른 오래된 gitlink의
영향을 피하도록 필요한 서브모듈만 명시해서 받습니다.

```bash
git submodule update --init apps/calibration_worker/vggt-omega
python -m pip install --no-deps -e apps/calibration_worker/vggt-omega
```

`--no-deps`는 이미 루트 requirements로 설치한 `numpy<2`와
`opencv-contrib-python-headless`를 VGGT-Omega의 일반 OpenCV 의존성이 다시
덮어쓰지 않게 합니다. 현재 고정된 VGGT-Omega 커밋은 Python 3.10 이상,
PyTorch 2.3 이상을 요구합니다.

추가로 준비할 자산은 다음과 같습니다.

1. Hugging Face의 `facebook/VGGT-Omega`에서 사용 승인을 받은 뒤
   `VGGT-Omega-1B-512` 체크포인트를 서버 로컬 경로에 저장합니다.
2. Isaac Sim의 ArUco extension으로
   `apps/isaac_sim_client/aruco_boards/aruco_marker_tree.usd`를 생성합니다.
   현재 저장소에는 생성된 marker tree가 포함되어 있지 않습니다.
3. CCTV 이미지, 기준 영상, 각 카메라의 camera matrix와 distortion 값을 담은
   JSON manifest를 준비합니다. 형식은
   `apps/calibration_worker/README.md`를 참고합니다.

## 4. 설치 검증

공통 환경을 활성화한 상태에서 아래 smoke test를 실행합니다.

```bash
source .venv-server/bin/activate

python - <<'PY'
import cv2
import numpy
import scipy
import sklearn
import torch
import torchvision
import zenoh
import zmq
import zstandard
from pxr import Usd

assert hasattr(cv2, "aruco"), "cv2.aruco가 없습니다"
assert torch.cuda.is_available(), "CUDA PyTorch가 아닙니다"
print("server Python environment: OK")
print("torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("opencv:", cv2.__version__, "numpy:", numpy.__version__)
PY

python -m apps.edge_manager --help
python apps/server_worker/inference.py --help
python apps/calibration_worker/inference.py --help
```

VGGT 서브모듈까지 설치했다면 다음도 확인합니다.

```bash
python -c "from vggt_omega.models import VGGTOmega; print('VGGT-Omega: OK')"
```

## 5. 서비스별 실행

### Edge Manager

Zenoh router가 `127.0.0.1:7447`에서 동작한다고 가정한 예입니다.

```bash
source .venv-server/bin/activate
python -m apps.edge_manager --endpoint 127.0.0.1:7447 serve
```

Registry 기본 파일은 `apps/edge_manager/data/edges.json`입니다. 다른 터미널에서
`list`, `approve`, `command`, `revoke`, `remove` 명령을 사용할 수 있습니다.

### Server Worker

저장소에 기본 pose 설정과 `models/POC_posenet.pth.tar`가 포함되어 있습니다.
현재 `inference.py`의 edge 목록은 `0_edge`로 지정되어 있으므로
`apps/server_worker/data/edge.json`의 해당 metadata/topic과 실제 edge 발행 토픽이
일치해야 합니다.

```bash
source .venv-server/bin/activate
python apps/server_worker/inference.py \
  --zenoh-endpoint 127.0.0.1:7447 \
  --scene-zenoh-topic meta-sejong/scene/v1
```

CPU fallback은 없으며 CUDA가 없으면 즉시 종료됩니다. 기본 설정은 TensorRT를
사용하지 않습니다.

TensorRT를 켜려면 서버의 CUDA/TensorRT/PyTorch 버전에 맞는 TensorRT Python
binding, `pycuda`, `torch2trt`를 별도로 설치한 뒤 `--tensorrt`를 사용합니다.
이 세 패키지는 CUDA ABI에 종속되므로 공통 requirements에는 넣지 않았습니다.
NVIDIA 환경에 맞지 않는 일반 PyPI wheel을 임의로 설치하지 마세요.

scene 발행만 먼저 확인하려면 다음 구독 도구를 별도 터미널에서 실행합니다.

```bash
source .venv-server/bin/activate
python apps/server_worker/tools/scene_zenoh_subscriber.py \
  --endpoint 127.0.0.1:7447 \
  --topic meta-sejong/scene/v1
```

### Calibration Worker

```bash
source .venv-server/bin/activate
python apps/calibration_worker/inference.py \
  --config /path/to/calibration_input.json \
  --checkpoint /path/to/vggt_omega_1b_512.pt \
  --marker-tree apps/isaac_sim_client/aruco_boards/aruco_marker_tree.usd \
  --output-dir results/calibration_001 \
  --max-images 48
```

GPU OOM이 발생하면 `--max-images`를 낮춥니다. 결과의
`alignment.rmse_m`, `alignment.max_error_m`를 확인한 뒤 생성된 extrinsics를
server worker metadata에 반영합니다.

## 6. Isaac Sim Client 전용 환경

일반 `.venv-server`가 아니라 Isaac Sim 설치 디렉터리에서 실행합니다.
`omni`와 `pxr`은 Isaac Sim이 제공하므로 별도로 pip 설치하지 않습니다.

```bash
export DT_SERVER_DIR=/path/to/DT_SERVER
export ISAAC_SIM_DIR=/path/to/isaac-sim-standalone-4.2.0

cd "$ISAAC_SIM_DIR"
./python.sh -m pip install \
  -r "$DT_SERVER_DIR/requirements-isaac-sim.txt"
./python.sh -c "import omni.usd, zenoh; from pxr import Usd; print('Isaac Python: OK')"
```

Isaac Sim에 이미 `cv2.aruco`가 있다면 OpenCV를 다시 설치할 필요가 없습니다.
먼저 확인하고, 필요한 경우에만 전용 requirements의 OpenCV 항목을 설치하는 것이
가장 안전합니다.

```bash
./python.sh -c "import cv2; print(cv2.__version__, hasattr(cv2, 'aruco'))"

# 위 명령의 마지막 값이 False일 때만 실행
./python.sh -m pip install "numpy<2" \
  "opencv-contrib-python-headless>=4.8,<4.12"
```

실행에 필요한 환경변수를 설정합니다.

```bash
export ISAAC_USD_PATH=/absolute/path/to/2025_SejongUniv_All.usd
export ISAAC_SCENE_TRANSPORT=zenoh
export ISAAC_ZENOH_ENDPOINT=127.0.0.1:7447
export ISAAC_ZENOH_TOPIC=meta-sejong/scene/v1
```

Headless WebRTC 실행:

```bash
cd "$ISAAC_SIM_DIR"
./isaac-sim.headless.webrtc.sh \
  --exec "$DT_SERVER_DIR/apps/isaac_sim_client/meta_sejong_script.py"
```

WebSocket transport를 사용할 때만 `ISAAC_SCENE_TRANSPORT=websocket` 또는
`both`로 바꾸고 `SIM_BACKEND_WS_URL`을 지정합니다. Zenoh만 사용할 때는
`sim_backend` WebSocket 연결이 필요하지 않습니다.

## 7. 운영 시 확인 사항

- Zenoh router의 TCP 7447(또는 설정한 포트)을 서버/edge/Isaac Sim 사이에서
  허용합니다.
- 실행 전 `nvidia-smi`로 드라이버 상태와 남은 GPU 메모리를 확인합니다.
- `.venv-server`, VGGT 체크포인트, 입력 영상과 생성 calibration 결과는 Git에
  커밋하지 않습니다.
- `edge_manager` registry와 `server_worker`의 edge metadata는 현재 자동 동기화되지
  않으므로 승인된 edge ID/topic을 수동으로 일치시킵니다.
- Isaac USD의 상대 참조 texture/하위 USD도 동일한 디렉터리 구조로 서버에
  복사합니다.
