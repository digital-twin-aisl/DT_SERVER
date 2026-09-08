# Edge Client

Jetson에서 RTSP 카메라 영상을 추론하고 결과를 Zenoh router를 통해 서버로
전송합니다. 장비별 ID, 서버 주소, 카메라 설정과 추론 기본값은 로컬 파일에 한 번
저장되며, 이후에는 같은 인자를 반복해서 입력하지 않습니다.

두 엣지 + 한 서버의 회의 기반 실행 설정은 [PoC 실행 문서](../server_worker/docs/poc-runtime.md)를
사용합니다. 0.5초 Rank/LOD, 공통 calibration, TCP router 및 오프라인 Rank/Zone 비교 명령을 포함합니다.

## 1. 설치

지원 환경은 Jetson Orin 계열, JetPack 6/L4T R36, Python 3.10입니다. 저장소
루트에서 실행합니다.

```bash
chmod +x apps/edge_client/install_edge.sh
apps/edge_client/install_edge.sh
```

설치 스크립트는 다음 작업을 수행합니다.

- `apps/edge_client/.venv` 생성
- Jetson용 CUDA PyTorch와 Python 의존성 설치
- FastReID 및 VGGT-Omega 서브모듈 준비
- `torch2trt`, 모델 및 샘플 데이터 설치
- CUDA, TensorRT와 주요 Python 패키지 검증

스크립트 실행에는 인터넷 연결과 `sudo` 권한이 필요합니다. JetPack, CUDA,
cuDNN과 TensorRT 드라이버는 Jetson에 미리 설치되어 있어야 합니다.

설치 스크립트만으로 장비 설정과 서버 등록까지 완료되지는 않습니다. 다음 항목은
설치 후 별도로 준비합니다.

- 서버의 Zenoh router 및 edge manager 실행
- 엣지 ID, 표시 이름과 서버 주소 최초 설정
- `cameras.local.yaml`의 RTSP 카메라 및 캘리브레이션 정보
- 서버에서 새 엣지 승인

## 2. 서버 준비

가장 간단한 구성은 Zenoh 1.9의 인증서 없는 reliable UDP, 즉 unsecure QUIC을
사용하는 것입니다. 암호화나 서버 인증이 없으므로 신뢰할 수 있는 내부망 또는
VPN에서만 사용해야 합니다.

서버의 저장소 루트에서 router와 manager를 각각 실행합니다.

```bash
zenohd -c apps/edge_manager/config/zenoh-router-quic.json5
```

```bash
python -m apps.edge_manager \
  --endpoint 'udp/127.0.0.1:10020?rel=1' \
  serve
```

서버 방화벽에서도 UDP `10020` 포트를 허용해야 합니다. 인터넷에 직접 노출되는
환경에서는 인증서가 필요한 secure `quic/` 구성을 사용해야 합니다.

서버의 전체 설치와 실행 방법은 [SERVER_SETUP.md](../../SERVER_SETUP.md), 엣지
등록 관리는 [edge manager README](../edge_manager/README.md)를 참고하세요.

## 3. 최초 설정: `agent.py init`

`init`은 장비마다 최초 한 번 실행합니다. `edge-id`는 deployment에 정의된 논리
ID이고, `display-name`은 사람이 장비를 구분하기 위한 이름입니다.

```bash
apps/edge_client/.venv/bin/python apps/edge_client/agent.py init \
  --edge-id edge_1 \
  --display-name dt_jetson_1 \
  --endpoint 'udp/SERVER_IP:10020?rel=1' \
  --camera-config apps/edge_client/config/cameras.local.yaml \
  --deployment apps/deployments/scene_0812_2.json \
  --tensorrt
```

`SERVER_IP`를 실제 서버 주소로 바꿉니다. 이 명령은 다음 정보를 Git에서 제외된
`apps/edge_client/config/edge.local.json`에 저장합니다.

- `edge_id`와 `display_name`
- Zenoh router 주소
- 카메라 설정 파일과 카메라 ID 목록
- deployment와 TensorRT 사용 여부
- 엣지별 status, command, calibration 및 inference 토픽

이미 다른 ID가 저장된 장비의 ID를 바꿀 때만 `--replace-edge-id`를 추가합니다.
ID를 변경하면 서버에서 새 ID로 다시 승인해야 합니다.

기존 캘리브레이션 결과 JSON을 `cameras.local.yaml`로 가져와야 한다면 최초 설정에
다음을 추가할 수 있습니다.

```bash
--calibration-result apps/deployments/calibration_result_20260814-123941.json
```

저장된 ID는 다음 명령으로 확인합니다.

```bash
apps/edge_client/.venv/bin/python apps/edge_client/agent.py show-id
```

## 4. 카메라 설정

실시간 추론 전에 `apps/edge_client/config/cameras.local.yaml`에 각 카메라의 RTSP
주소, intrinsic, distortion과 extrinsic이 있어야 합니다. 카메라를 대화형으로
등록하거나 확인하려면 다음 명령을 사용합니다.

```bash
apps/edge_client/.venv/bin/python apps/edge_client/camera_setup.py
apps/edge_client/.venv/bin/python apps/edge_client/camera_setup.py list
```

자세한 등록, 스트리밍 확인, 녹화 및 intrinsic 측정 방법은
[카메라 설정 문서](docs/camera-setup.md)를 참고하세요. RTSP URL과 인증정보가 들어간
`cameras.local.yaml`은 Git에 커밋하지 않습니다.

## 5. 엣지 등록과 승인

엣지에서 agent를 실행합니다. `run`은 `init`에서 저장된 서버 주소와 ID를 자동으로
읽으므로 인자를 다시 입력할 필요가 없습니다.

```bash
apps/edge_client/.venv/bin/python apps/edge_client/agent.py run
```

agent는 계속 실행해 둡니다. 서버의 다른 터미널에서 새 엣지를 확인하고 승인합니다.

```bash
python -m apps.edge_manager list
python -m apps.edge_manager \
  --endpoint 'udp/127.0.0.1:10020?rel=1' \
  approve edge_1 --name dt_jetson_1
```

승인이 완료되면 엣지의 `edge.local.json`에서 `approved`가 `true`로 갱신됩니다.

## 6. 추론 실행

최초 설정, 카메라 캘리브레이션과 서버 승인이 끝났다면 별도 인자 없이 실행합니다.

```bash
apps/edge_client/.venv/bin/python apps/edge_client/inference.py
```

추론기는 `edge.local.json`에서 ID, router 주소, deployment와 TensorRT 설정을 읽고,
`cameras.local.yaml`에서 RTSP 주소와 모든 캘리브레이션 정보를 읽습니다. 명령행
옵션을 전달하면 저장된 기본값보다 우선합니다.

Zenoh 송신 없이 로컬에서만 확인하려면 다음과 같이 실행합니다.

```bash
apps/edge_client/.venv/bin/python apps/edge_client/inference.py --no-zenoh
```

데이터셋 추론은 실시간 기본 실행과 달리 데이터 위치를 지정해야 합니다. 예시는
[edge agent 문서](docs/edge-agent.md)와 저장소 루트 [README](../../README.md)를
참고하세요.

## 실행 순서 요약

```text
Edge:   install_edge.sh
Server: zenohd + edge manager serve
Edge:   agent.py init                 # 장비당 최초 한 번
Edge:   agent.py run                  # 계속 실행
Server: edge manager approve edge_1   # 최초 등록/ID 변경 후
Edge:   inference.py                  # 이후 무인자 실행
```

설정 파일을 직접 수정하기보다 `agent.py init`을 다시 실행하는 편이 안전합니다.
상세 agent 동작과 저장 항목은 [edge agent 문서](docs/edge-agent.md)를 참고하세요.
