# 2. 서버 설치

GPU 서버 한 대에 다음을 준비합니다. 서버는 구역 실행의 중심이며, 엣지와 브라우저는
모두 이 서버의 Zenoh router와 관리 서비스에 접속합니다.

| 구성 | 실행 형태 | 기본 주소 |
| --- | --- | --- |
| Zenoh router (`zenohd` 1.9.0) | 상시 | `tcp/0.0.0.0:7447` |
| `edge_manager serve` (관리 API·ControlPlane) | 상시 | `127.0.0.1:8001` |
| `frontend_api` (GUI·뷰어) | 상시 | `127.0.0.1:8005` |
| `server_worker` (추론) | manager가 구역 시작 시 실행 | - |
| `calibration_worker` (보정) | manager가 보정 작업 시 실행 | - |

## 1) 요구 환경

- Ubuntu 22.04 권장, Python 3.10 또는 3.11
- NVIDIA GPU, 드라이버, CUDA 지원 PyTorch (`server_worker`에는 CPU 대체 경로가 없음)
- Node.js 22 이상 (GUI 빌드), Git

```bash
nvidia-smi
python3 --version
sudo apt install -y git python3-venv python3-dev build-essential
```

## 2) 저장소와 Python 환경

```bash
git clone --recurse-submodules <이 저장소 URL> DT_SERVER
cd DT_SERVER
python3 -m venv .venv-server
. .venv-server/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### CUDA PyTorch를 먼저 설치

[PyTorch 설치 선택기](https://pytorch.org/get-started/locally/)에서 서버 CUDA에 맞는
명령으로 `torch>=2.3`, `torchvision>=0.18`을 설치한 뒤 확인합니다.

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

마지막 값이 `False`이면 다음 단계로 가기 전에 드라이버와 wheel 조합을 고칩니다.

### 공통 패키지, 관리 서비스, GUI

```bash
python -m pip install -r requirements.txt          # dt-common(editable) 포함
python -m pip install -r apps/edge_manager/requirements.txt -r apps/frontend_api/requirements.txt
npm ci --prefix apps/frontend_api/web
npm run build --prefix apps/frontend_api/web
```

`requirements.txt`는 ArUco가 포함된 `opencv-contrib-python-headless` 하나만 설치합니다.
다른 OpenCV 패키지(`opencv-python` 등)를 함께 설치하면 `cv2`가 덮어써지므로 섞지 않습니다.

## 3) 모델 가중치

학습된 가중치는 소스 저장소에 포함되지 않습니다. 사용 권한이 있는 PoseNet
체크포인트를 `apps/server_worker/models/POC_posenet.pth.tar`에 두거나, URL을 지정해
내려받습니다.

```bash
DT_POSENET_URL='https://drive.google.com/uc?id=<file id>' \
  python apps/server_worker/src/utils/download_from_drive.py
```

경로는 `apps/server_worker/config/config.py`의 `POSENET.CKPT`, 모델 구조는
`apps/server_worker/config/cam4_posenet.yaml`이 정합니다. 가중치마다 라이선스가
다를 수 있으므로 [라이선스 준수](09-license-compliance.md)를 확인합니다.

## 4) (선택) 자동 보정: VGGT-Omega

자동 카메라 보정은 VGGT-Omega(**FAIR Noncommercial Research License**)를 사용합니다.
비상업 연구 목적이 아니면 설치하지 말고 [수동 보정 편집기](05-new-region.md#3-카메라-보정)를
사용합니다.

```bash
git submodule update --init apps/calibration_worker/vggt-omega
git -C apps/calibration_worker/vggt-omega apply ../patches/vggt-omega-patch-tokens.patch
python -m pip install --no-deps -e apps/calibration_worker/vggt-omega
```

패치는 엣지가 보낸 patch token으로 보정을 이어 가는 분산 보정에 필요하며, 패치 자체도 FAIR 비상업
라이선스를 따릅니다([patches/README.md](../../apps/calibration_worker/patches/README.md)).

`--no-deps`는 루트 requirements의 `numpy<2`와 OpenCV를 덮어쓰지 않게 합니다.
Hugging Face `facebook/VGGT-Omega`에서 사용 승인을 받은 뒤 체크포인트를 서버 로컬에 둡니다.

## 5) 설치 확인

```bash
python - <<'PY'
import cv2, numpy, torch, zenoh, zstandard
from pxr import Usd
assert hasattr(cv2, "aruco"), "cv2.aruco가 없습니다"
assert torch.cuda.is_available(), "CUDA PyTorch가 아닙니다"
print("OK torch", torch.__version__, "opencv", cv2.__version__, "numpy", numpy.__version__)
PY
python -m apps.edge_manager --help
python apps/server_worker/inference.py --help
```

## 6) 상시 서비스 실행

세 프로세스를 실행합니다. 개발 중에는 터미널 세 개로, 운영에서는 systemd 등으로
상시 실행합니다(아래 예시).

```bash
zenohd -c apps/edge_manager/config/zenoh-router-poc.json5
python -m apps.edge_manager serve
python -m uvicorn apps.frontend_api.app.main:app --host 127.0.0.1 --port 8005
```

| 환경변수 | 기본값 | 설명 |
| --- | --- | --- |
| `ZENOH_ENDPOINT` | `tcp/127.0.0.1:7447` | manager가 접속할 router |
| `SCENE_ZENOH_ENDPOINT` | `tcp/127.0.0.1:7447` | GUI 뷰어가 장면을 구독할 router |
| `DT_REGION_CATALOG` | `apps/deployments/regions.json` | 구역 catalog 파일 |
| `DT_DATA_DIR` | `data/manager` | registry·run 기록 저장 위치 |
| `DT_SERVER_PYTHON` | manager와 같은 Python | `server_worker`를 실행할 Python |
| `DT_CALIBRATION_PYTHON` | manager와 같은 Python | `calibration_worker`를 실행할 Python |
| `DT_MANAGER_URL` | `http://127.0.0.1:8001` | CLI·GUI가 호출할 관리 API |
| `DT_MANAGER_TOKEN` | 없음 | 관리 API를 localhost 밖에 열 때 **필수** Bearer 토큰 |
| `DT_RECORDING_MAX_BYTES` | 1 GiB | 장면 기록 파일 상한 |

systemd 사용자 서비스 예시 (`~/.config/systemd/user/dt-manager.service`):

```ini
[Unit]
Description=DT_SERVER edge manager
After=network-online.target

[Service]
WorkingDirectory=/opt/DT_SERVER
ExecStart=/opt/DT_SERVER/.venv-server/bin/python -m apps.edge_manager serve
Restart=on-failure

[Install]
WantedBy=default.target
```

router와 frontend도 같은 방식으로 만들고 `systemctl --user enable --now dt-manager`로
켭니다. 로그인 전에 시작하려면 `loginctl enable-linger "$USER"`를 한 번 실행합니다.
manager가 재시작되면 끝나지 않은 run은 `interrupted`로 표시되고 자동으로 재개되지 않습니다.

Docker로 router와 GUI만 띄우는 선택지도 있습니다(`docker-compose.yml`). GPU worker와
manager는 호스트 환경이 필요합니다.

## 7) 네트워크와 보안

- 엣지·Isaac 호스트가 router의 TCP 7447에 접속할 수 있어야 합니다.
- Zenoh 전송은 **암호화·인증이 없는 설정**입니다. 신뢰된 내부망이나 VPN 안에서만
  사용하고, 인터넷에 직접 노출해야 하면 Zenoh TLS/QUIC 인증서 구성을 별도로 적용합니다.
- 관리 API(8001)와 GUI(8005)는 기본적으로 loopback에만 열립니다. 원격 접속은 SSH/VPN
  터널을 권장합니다. 외부 바인딩 시 `DT_MANAGER_TOKEN`이 없으면 manager가 시작을 거부합니다.
- RTSP 주소와 카메라 계정은 서버에 두지 않습니다(엣지 로컬 파일에만 저장).

## 8) 개발·진단용 단독 실행

`apps/server_worker/inference.py`를 직접 실행하는 방법은
[poc-runtime.md](../../apps/server_worker/docs/poc-runtime.md)에 있습니다. 관리 중인
구역과 **동시에 실행하지 마세요**. 발행되는 장면은 다음 도구로 확인합니다.

```bash
python apps/server_worker/tools/scene_zenoh_subscriber.py \
  --endpoint 127.0.0.1:7447 --topic meta-sejong/scene/v1
```

TensorRT 가속은 서버 CUDA/TensorRT에 맞는 TensorRT Python binding, `pycuda`,
`torch2trt`를 직접 설치한 뒤 `--tensorrt`로 켭니다. ABI 의존성 때문에 공통
requirements에는 넣지 않았습니다.
