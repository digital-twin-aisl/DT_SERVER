# Edge Agent CLI

Edge agent는 로컬에 안정적인 `edge_id`를 생성하고 Zenoh로 heartbeat를 보내
서버가 엣지를 자동 발견할 수 있게 합니다.

## 실행

처음 한 번만 논리 ID, 표시 이름, Zenoh 연결, 카메라 설정과 추론 기본값을
저장합니다. 논리 ID는 server deployment의 edge ID와 같아야 하고 표시 이름은
사람이 장비를 구분하는 이름입니다.

```bash
python apps/edge_client/agent.py init \
  --edge-id edge_1 \
  --display-name dt_jetson_1 \
  --replace-edge-id \
  --endpoint 'udp/SERVER_IP:10020?rel=1' \
  --camera-config apps/edge_client/config/cameras.local.yaml \
  --deployment apps/deployments/scene_0812_2.json \
  --calibration-result apps/deployments/calibration_result_20260814-123941.json \
  --tensorrt

python -m apps.edge_client.agent show-id
```

이미 다른 ID가 저장된 장비에서만 `--replace-edge-id`가 필요합니다. ID를 바꾸면
서버에서는 새 edge로 다시 승인해야 합니다. 위 UDP endpoint는 Zenoh 1.9의
인증서 없는 reliable QUIC이며 신뢰 네트워크 또는 VPN 안에서만 사용합니다.

Agent를 실행합니다.

```bash
python apps/edge_client/agent.py run
```

온라인 calibration capture까지 사용할 때만 checkpoint를 추가합니다.

```bash
python apps/edge_client/agent.py run \
  --calibration-checkpoint /models/VGGT-Omega-1B-512/model.pt
```

Zenoh 세션이 연결되면 등록 카메라 상태를 `dt/edges/{edge_id}/cameras`로 한 번
발행합니다. 이후 manager의 `ping` heartbeat를 받을 때마다 카메라 RTSP 포트
연결을 다시 확인하고 최신 상태를 발행합니다. 카메라별 연결 확인 제한시간은
`--camera-ping-timeout`(기본 1초)으로 조정할 수 있습니다.

`capture_calibration_features` 명령을 받으면 등록된 전체 카메라에서 한 프레임을
캡처하고 로컬 intrinsic/distortion으로 왜곡을 제거한 뒤 DINO patch token만
전송합니다. `apply_calibration_result` 명령은 서버가 구한 USD 좌표계 pose를
`cameras.local.yaml`에 원자적으로 저장하고 ACK를 보냅니다. 두 명령 모두 승인된
edge에서만 실행됩니다.

ID는 기본적으로 `apps/edge_client/config/edge.local.json`에 저장되고 Git에서
제외됩니다. `run` 전에 `init`하지 않아도 최초 실행 시 자동 생성됩니다.

서버에서 승인하면 같은 파일에 서버 메타데이터가 추가됩니다.

```json
{
  "edge_id": "edge_1",
  "approved": true,
  "display_name": "dt_jetson_1",
  "camera_config": "cameras.local.yaml",
  "camera_ids": [2, 4, 6, 8],
  "zenoh_endpoint": "udp/SERVER_IP:10020?rel=1",
  "topic_root": "dt/edges",
  "topics": {
    "status": "dt/edges/edge_1/status",
    "inference": "dt/edges/edge_1/inference",
    "command": "dt/edges/edge_1/command",
    "config": "dt/edges/edge_1/config",
    "ack": "dt/edges/edge_1/ack",
    "cameras": "dt/edges/edge_1/cameras"
  }
}
```

저장된 `zenoh_endpoint`, `topic_root`, `camera_config`, `camera_ids`와 `inference`
기본값은 다음 agent 및 inference 실행부터 자동 사용됩니다. CLI 인자를 전달하면
저장값보다 CLI가 우선합니다.

추론기도 같은 identity 파일을 읽으므로 토픽을 따로 전달하지 않습니다.

```bash
apps/edge_client/.venv/bin/python apps/edge_client/inference.py
```

Identity에 `camera_config`와 `camera_ids`가 있으면 실시간 추론은 해당 로컬
RTSP 설정을 읽고 deployment의 카메라 순서와 대조합니다. `camera_config`의 상대
경로는 identity 파일이 있는 디렉터리를 기준으로 합니다.

실시간 모델의 intrinsic, distortion, extrinsic은 모두 같은
`cameras.local.yaml`에서 읽습니다. 각 카메라에 `intrinsic`과 `extrinsic`이 모두
있어야 하며, 누락되면 inference는 어떤 레거시 파일도 대신 사용하지 않고 즉시
오류를 냅니다. 기존 calibration 결과가 있으면 위 `init --calibration-result`가
해당 edge 카메라의 값을 YAML로 가져옵니다. 온라인 calibration을 새로 수행한
경우에도 agent가 `extrinsic`을 같은 YAML에 저장하므로 그다음부터 위 무인자
명령으로 실행할 수 있습니다.

기본 추론 토픽은 `dt/edges/{edge_id}/inference`입니다. 이전
`--zenoh-topic` 옵션은 기존 실행 명령과의 호환만을 위해 숨김 상태로 남아
있습니다.
