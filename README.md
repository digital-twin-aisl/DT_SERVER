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

Jetson에서는 저장소 루트에서 다음을 실행합니다.

```bash
chmod +x apps/edge_client/install_edge.sh
apps/edge_client/install_edge.sh
source apps/edge_client/.venv/bin/activate
```

스크립트는 다음 작업을 수행합니다.

1. 시스템 패키지와 `apps/edge_client/.venv` 준비
2. Jetson용 CUDA PyTorch와 edge client 의존성 설치
3. 누락된 FastReID/VGGT-Omega 서브모듈 초기화
4. `torch2trt` 설치
5. 샘플 데이터, pose checkpoint, ReID checkpoint 다운로드
6. CUDA와 주요 Python import 검증

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

`data/data_0812_1`처럼 두 edge의 영상 8개와 `edge_info.txt`가 함께 있는
데이터셋은 실행할 edge identity를 지정하면 해당 4개 카메라를 자동 선택합니다.
저장소 루트에서도 동일하게 실행할 수 있습니다.

```bash
apps/edge_client/.venv/bin/python apps/edge_client/inference.py \
  --dataset \
  --example-folder apps/edge_client/data/data_0812_1 \
  --edge-id edge_1 \
  --edge-id-file apps/edge_client/config/edge_1.dataset.json \
  --deployment apps/deployments/scene_0812_2.json \
  --tensorrt \
  --no-zenoh
```

첫 프레임만 빠르게 검증하려면 `--max-frames 1`을, ReID 엔진/모델을 제외하고
3D pose 경로만 검증하려면 `--no-reid`를 추가합니다.

USD calibration 데이터셋은 `--deployment`가 필수입니다. 이 manifest의 Ground와
edge AOI를 `workspace.py` 정책으로 계산한 결과 현재 `edge_1`은 `328x108x20`,
`edge_2`는 `300x108x16` cube를 사용합니다. deployment 없이 학습 기준의 고정
`80x80x20` cube로 실행하는 오류는 시작 단계에서 차단됩니다.

Edge 모델은 backbone heatmap과 root candidate까지만 계산합니다. 관절별 3D pose
regression은 server worker가 담당하며, edge rootnet 모드에서는 `PoseRegressionNet`
가 생성되거나 checkpoint에서 GPU로 적재되지 않습니다.

`--tensorrt`를 처음 사용하면 pose와 FastReID checkpoint의 SHA-256, 입력/cube
크기, precision과 Jetson/TensorRT 버전을 조합한 전용 FP16 엔진을 각각 생성하고
PyTorch 출력과 비교 검증합니다. 같은 설정의 다음 실행부터는 생성된 엔진을 자동
재사용합니다. 체크포인트나 설정이 바뀌면 별도 엔진을 자동 생성하며, 두 엔진을
강제로 다시 만들려면 `--rebuild-tensorrt`를 사용합니다.
`config.POSENET.TENSORRT: true`로 설정해도 `--tensorrt`와 동일하게 동작합니다.

## 종료 및 상태 확인

```bash
docker compose logs -f
docker compose down
```

운영 환경에서는 설정 파일에 RTSP 비밀번호를 직접 커밋하지 말고 별도 비밀 관리 수단을 사용하세요.
