# DT_SERVER

Jetson 엣지에서 멀티 카메라 영상을 처리하고, ReID 및 3D pose 중간 결과를 Zenoh로 서버에 전달하는 디지털 트윈 백엔드입니다. 전체 서비스 구성은 [ARCHITECTURE.md](ARCHITECTURE.md)를 참고하세요.

## 구성

- `apps/edge_client`: Jetson 카메라 입력, YOLO pose/ReID, 3D pose 전처리 및 Zenoh 송신
- `apps/edge_manager`: 엣지 연결과 데이터 수집
- `apps/dl_worker`: 서버 측 딥러닝 처리
- `apps/camera_manager`: 카메라 및 캘리브레이션 관리
- `apps/sim_backend`: Isaac Sim/프론트엔드용 WebSocket 송신
- `apps/frontend_api`: 대시보드 API

## 서버 실행

`server_worker`, `edge_manager`, `calibration_worker`, `isaac_sim_client`의 Python
가상환경과 GPU/Isaac Sim 준비 절차는 [SERVER_SETUP.md](SERVER_SETUP.md)를 먼저
확인하세요.

Docker와 Docker Compose를 설치한 서버에서 다음을 실행합니다.

```bash
docker compose up --build -d
docker compose ps
```

Zenoh router를 별도로 사용하는 경우 서버에서 외부 연결을 받을 수 있도록 실행합니다.

```bash
zenohd --listen tcp/0.0.0.0:7447
```

방화벽이나 보안 그룹에서도 TCP 7447 포트를 허용해야 합니다.

## Jetson edge client 설치

### 요구 환경

- Jetson Orin 계열
- JetPack 6 / L4T R36
- Python 3.10
- JetPack에 포함된 CUDA, cuDNN, TensorRT
- JetPack 버전과 호환되는 CUDA PyTorch 및 torchvision

PyTorch wheel 선택은 [NVIDIA PyTorch for Jetson 설치 문서](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)를 기준으로 합니다. `torch2trt`는 [NVIDIA 공식 저장소](https://github.com/NVIDIA-AI-IOT/torch2trt)에서 설치됩니다.

### 기본 설치

CUDA PyTorch와 torchvision이 이미 설치된 Jetson에서는 저장소 루트에서 다음을 실행합니다.

```bash
chmod +x apps/edge_client/install_root.sh
apps/edge_client/install_root.sh
source apps/edge_client/.venv/bin/activate
```

스크립트는 다음 작업을 수행합니다.

1. JetPack 6/L4T R36 및 CUDA PyTorch 확인
2. `apps/edge_client/.venv` 생성
3. `apps/edge_client/requirements.txt` 설치
4. `torch2trt` 설치
5. 샘플 영상, 캘리브레이션 데이터, pose checkpoint 다운로드
6. CUDA와 주요 Python import 검증

PyTorch 또는 torchvision을 별도 wheel로 설치해야 한다면 다음처럼 지정합니다.

```bash
apps/edge_client/install_root.sh \
  --torch-wheel /path/to/torch-wheel.whl \
  --torchvision-wheel /path/to/torchvision-wheel.whl
```

주요 옵션:

| 옵션 | 설명 |
| --- | --- |
| `--venv PATH` | 가상환경 위치 변경 |
| `--system` | 가상환경 대신 시스템 Python 사용 |
| `--python COMMAND` | 사용할 Python 명령 지정 |
| `--skip-apt` | Ubuntu 패키지 설치 생략 |
| `--skip-torch2trt` | torch2trt 설치 생략 |
| `--skip-assets` | 샘플 데이터 및 모델 다운로드 생략 |
| `--force` | 지정 wheel과 torch2trt 재설치 |

전체 옵션은 `apps/edge_client/install_root.sh --help`로 확인할 수 있습니다.

## Edge client 설정 및 실행

카메라는 `python apps/edge_client/camera_setup.py`로 엣지 로컬 설정에 등록합니다. Zenoh endpoint, topic과 모델 경로는 `apps/edge_client/config/config.py` 또는 `--cfg_focus`로 전달하는 YAML 파일에서 설정합니다.

샘플 데이터로 실행:

```bash
cd apps/edge_client
source .venv/bin/activate
python inference.py \
  --dataset \
  --example_folder data/data_0705 \
  --tensorrt \
  --zenoh-endpoint SERVER_IP:7447
```

RTSP 카메라로 실행:

```bash
cd apps/edge_client
source .venv/bin/activate
python inference.py --tensorrt --zenoh-endpoint SERVER_IP:7447
```

Zenoh 송신 없이 로컬 추론만 확인하려면 `--no-zenoh`를 추가합니다.

## 종료 및 상태 확인

```bash
docker compose logs -f
docker compose down
```

운영 환경에서는 설정 파일에 RTSP 비밀번호를 직접 커밋하지 말고 별도 비밀 관리 수단을 사용하세요.
