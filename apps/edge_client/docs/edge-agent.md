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
  --camera-count 4
```

ID는 기본적으로 `apps/edge_client/config/edge.local.json`에 저장되고 Git에서
제외됩니다. `run` 전에 `init`하지 않아도 최초 실행 시 자동 생성됩니다.

추론기도 같은 identity 파일을 읽으므로 토픽을 따로 전달하지 않습니다.

```bash
python apps/edge_client/inference.py \
  --tensorrt \
  --zenoh-endpoint SERVER_IP:7447
```

기본 추론 토픽은 `dt/edges/{edge_id}/inference`입니다. 이전
`--zenoh-topic` 옵션은 기존 실행 명령과의 호환만을 위해 숨김 상태로 남아
있습니다.
