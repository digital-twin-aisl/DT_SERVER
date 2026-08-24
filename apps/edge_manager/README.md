# Zenoh Edge Manager CLI

엣지의 첫 `status` 메시지를 등록 요청으로 취급하는 최소 CLI 관리자입니다.
별도 HTTP, gRPC, Redis 서비스는 사용하지 않습니다.

## 실행

먼저 Zenoh router에 연결해 엣지를 감시합니다.

```bash
python -m apps.edge_manager --endpoint localhost:7447 serve
```

Manager는 기본 5초마다 online edge에 Zenoh `ping` heartbeat를 보내고, edge는
응답과 함께 최신 카메라 상태를 다시 발행합니다. 주기는
`serve --heartbeat-interval 10`처럼 변경할 수 있습니다.

다른 터미널에서 발견된 엣지를 관리합니다.

```bash
python -m apps.edge_manager list
python -m apps.edge_manager --endpoint SERVER_IP:7447 \
  approve EDGE_ID --name jetson-01
python -m apps.edge_manager command EDGE_ID ping
python -m apps.edge_manager revoke EDGE_ID
python -m apps.edge_manager remove EDGE_ID
```

카메라 캘리브레이션의 표준 진입점은 calibration worker입니다.

```bash
python apps/calibration_worker/inference.py
```

worker는 이 registry에서 edge ID와 등록 카메라 수를 읽고, 캘리브레이션 완료 후
카메라별 intrinsic/distortion/extrinsic과 정합 품질을 다시 저장합니다. 엣지 agent도
같은 checkpoint로 `--calibration-checkpoint` 옵션을 주어 실행해야 합니다. 카메라
intrinsic은 먼저 `python apps/edge_client/camera_setup.py intrinsic`으로 등록합니다.

Docker Compose로 manager를 실행 중이라면 같은 컨테이너의 registry를 사용하는
CLI를 다음처럼 호출합니다.

```bash
docker compose exec edge_manager python -m app list
```

`--endpoint`, `--topic-root`, `--registry`는 subcommand 앞에 둡니다. 환경 변수
`ZENOH_ENDPOINT`, `EDGE_TOPIC_ROOT`, `EDGE_REGISTRY_PATH`로도 지정할 수 있습니다.

Registry 기본 위치는 `apps/edge_manager/data/edges.json`입니다. 엣지는 처음
발견되면 `approved=false`로 저장되며 heartbeat가 15초 동안
없으면 offline 처리됩니다. 각 edge의 `cameras`에는 카메라 ID별 `exists`,
`ping`, `calibration`을 저장합니다. 일반 camera status는 intrinsic, extrinsic,
distortion coefficients로 제한됩니다. 캘리브레이션 작업이 명시적으로 실행될 때만
서비스 매핑에 필요한 이름, 위치, Twin ID가 결과와 함께 저장됩니다. RTSP URL,
인증정보, 원본 프레임은 어떤 경우에도 서버로 전송하지 않습니다.

## 토픽

```text
dt/edges/{edge_id}/status
dt/edges/{edge_id}/inference
dt/edges/{edge_id}/command
dt/edges/{edge_id}/ack
dt/edges/{edge_id}/cameras
dt/edges/{edge_id}/calibration
dt/edges/{edge_id}/calibration/{request_id}/chunks/{index}
```

현재 agent 명령은 `ping`, `info`, `shutdown`, `capture_calibration_features`,
`apply_calibration_result`를 지원합니다. `shutdown`은 Jetson을
종료하는 명령이 아니라 edge agent 프로세스만 정상 종료합니다.

`approve`는 `dt/edges/{edge_id}/config`로 설정을 발행하고, 엣지가
`edge.local.json` 저장을 완료했다는 ACK를 보낸 뒤에만 registry를 승인 상태로
변경합니다. 엣지가 보고한 endpoint 대신 다른 외부 주소를 저장해야 한다면
`approve EDGE_ID --edge-endpoint PUBLIC_IP:7447`을 사용합니다.

현재 최소 구현에서는 registry와 `server_worker`의 승인 엣지 목록을 자동으로
동기화하지 않습니다. 서버 추론을 붙일 때는 승인된 엣지의 ID와
`dt/edges/{edge_id}/inference` 토픽을 server worker 설정에 사용해야 합니다.
