# Edge Agent CLI

Edge agent는 로컬에 안정적인 `edge_id`를 생성하고 Zenoh로 heartbeat를 보내
서버가 엣지를 자동 발견할 수 있게 합니다.

## 실행

저장할 ID를 직접 지정하거나 자동 생성할 수 있습니다.

```bash
python -m apps.edge_client.agent init --edge-id edge-001
python -m apps.edge_client.agent show-id
```

Agent를 실행합니다.

```bash
python -m apps.edge_client.agent run \
  --endpoint SERVER_IP:7447 \
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
  "edge_id": "edge-001",
  "approved": true,
  "display_name": "jetson-01",
  "camera_config": "cameras.local.yaml",
  "camera_ids": [2, 4, 6, 8],
  "zenoh_endpoint": "SERVER_IP:7447",
  "topic_root": "dt/edges",
  "topics": {
    "status": "dt/edges/edge-001/status",
    "inference": "dt/edges/edge-001/inference",
    "command": "dt/edges/edge-001/command",
    "config": "dt/edges/edge-001/config",
    "ack": "dt/edges/edge-001/ack",
    "cameras": "dt/edges/edge-001/cameras"
  }
}
```

저장된 `zenoh_endpoint`와 `topic_root`는 다음 agent 및 inference 실행부터
기본값으로 사용됩니다. CLI 인자를 전달하면 저장값보다 CLI가 우선합니다.

추론기도 같은 identity 파일을 읽으므로 토픽을 따로 전달하지 않습니다.

```bash
python apps/edge_client/inference.py \
  --edge-id-file apps/edge_client/config/edge_1.dataset.json \
  --deployment apps/deployments/scene_0812_2.json \
  --tensorrt \
  --zenoh-endpoint SERVER_IP:7447
```

Identity에 `camera_config`와 `camera_ids`가 있으면 실시간 추론은 해당 로컬
RTSP 설정을 읽고 deployment의 카메라 순서와 대조합니다. `camera_config`의 상대
경로는 identity 파일이 있는 디렉터리를 기준으로 합니다.

기본 추론 토픽은 `dt/edges/{edge_id}/inference`입니다. 이전
`--zenoh-topic` 옵션은 기존 실행 명령과의 호환만을 위해 숨김 상태로 남아
있습니다.
