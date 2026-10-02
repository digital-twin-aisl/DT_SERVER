# 3. 엣지 설치 (Jetson)

엣지는 담당 카메라(보통 4대)의 RTSP 영상을 받아 2D heatmap·root 후보·(선택) ReID 특징을
계산하고, ZNH2 패킷으로 서버에 보냅니다. 엣지에는 **상시 agent** 하나만 띄워 두고,
추론 프로세스는 서버 manager의 명령으로 agent가 시작·중지합니다.

상세 옵션은 [apps/edge_client/README.md](../../apps/edge_client/README.md)와
[edge agent 문서](../../apps/edge_client/docs/edge-agent.md)에 있습니다. 이 문서는 처음
설치하는 순서만 정리합니다.

## 요구 환경

- Jetson Orin 계열, JetPack 6 (L4T R36), Python 3.10, CUDA·cuDNN·TensorRT(JetPack 포함)
- 서버 router(TCP 7447)에 도달 가능한 네트워크
- 담당 카메라의 RTSP 주소·계정, 체커보드로 측정한 intrinsic (없으면 `camera_setup.py`로 측정)

## 1) 설치

```bash
git clone --recurse-submodules <이 저장소 URL> DT_SERVER && cd DT_SERVER
# 선택 구성요소: 라이선스 확인 후 1로 켭니다 (09-license-compliance.md)
export DT_WITH_ULTRALYTICS=0      # ReID 사람 검출 (AGPL-3.0)
export DT_WITH_VGGT=1             # 관리형 자동 보정 캡처 (FAIR 비상업)
export DT_POSENET_URL='https://drive.google.com/uc?id=<file id>'   # 사용 권한이 있는 가중치
apps/edge_client/install_edge.sh
```

설치 스크립트는 `apps/edge_client/.venv`를 만들고 Jetson용 PyTorch, `torch2trt`,
dt-common, FastReID 서브모듈을 준비한 뒤 CUDA·TensorRT import를 확인합니다.

## 2) 카메라 등록

```bash
cp apps/edge_client/config/cameras.example.yaml apps/edge_client/config/cameras.local.yaml
apps/edge_client/.venv/bin/python apps/edge_client/camera_setup.py        # 대화형 등록·스트림 확인
apps/edge_client/.venv/bin/python apps/edge_client/camera_setup.py list
```

- 카메라 `id`는 배포 manifest가 이 엣지에 배정한 번호와 같아야 합니다(예: edge_1 → 2, 4, 6, 8).
- `cameras.local.yaml`과 `edge.local.json`은 git에서 제외되어 있습니다. **커밋하지 마세요.**
- intrinsic 측정, 스트림 녹화 방법은 [camera-setup.md](../../apps/edge_client/docs/camera-setup.md).

## 3) agent 최초 설정과 실행

```bash
apps/edge_client/.venv/bin/python apps/edge_client/agent.py init \
  --edge-id edge_1 --display-name site-a-edge-1 \
  --endpoint tcp/SERVER_IP:7447 \
  --camera-config apps/edge_client/config/cameras.local.yaml \
  --deployment apps/deployments/scene_0812_poc.json \
  --calibration-result apps/deployments/calibration_result_20260814-123941.json \
  --tensorrt

apps/edge_client/.venv/bin/python apps/edge_client/agent.py run
```

`edge_id`는 배포 manifest의 엣지 ID입니다(구역 소속을 ID에 넣지 않습니다). `run`은 계속
실행해 둡니다. systemd 사용자 서비스로 등록하면 재부팅 후에도 자동으로 올라옵니다.

온라인 보정을 사용할 엣지는 `agent.py run`에 `--calibration-checkpoint /path/to/VGGT-Omega.pt`를
추가합니다(`DT_WITH_VGGT=1` 필요).

## 4) 서버에서 승인

```bash
python -m apps.edge_manager list            # 새 엣지가 discovered로 보입니다
python -m apps.edge_manager approve edge_1
```

GUI(`http://SERVER:8005/`)의 장치 화면에서도 승인할 수 있습니다. 승인 후 구역을 시작하면
manager가 이 엣지의 추론을 시작합니다([운영](06-operations.md)).

## ReID 없이 운영하기

Ultralytics를 설치하지 않은 엣지는 ReID를 끈 상태로 실행해야 합니다. 이때 전역 ID는
서버의 위치 기반 추적으로 부여되며, 사람이 교차하거나 가려지면 ID가 바뀔 수 있습니다.
ReID 없이 ReID를 요청하면 엣지 추론은 `--no-reid`를 안내하는 오류로 즉시 종료합니다.
구역 단위 설정은 서버 프로파일과 엣지 런타임 설정의 `no_reid` 항목으로 지정합니다
(예: `apps/deployments/rootnet_v2/edge_1.json`).

## 직접 실행 (개발·진단용)

manager가 소유한 추론을 먼저 중지한 뒤에만 사용합니다.

```bash
apps/edge_client/.venv/bin/python apps/edge_client/inference.py --no-zenoh   # 로컬 확인
```
