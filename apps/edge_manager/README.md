# Zenoh Edge Manager CLI

엣지의 첫 `status` 메시지를 등록 요청으로 취급하는 최소 CLI 관리자입니다.
별도 HTTP, gRPC, Redis 서비스는 사용하지 않습니다.

## 실행

먼저 Zenoh router에 연결해 엣지를 감시합니다.

```bash
python -m apps.edge_manager --endpoint localhost:7447 serve
```

다른 터미널에서 발견된 엣지를 관리합니다.

```bash
python -m apps.edge_manager list
python -m apps.edge_manager approve EDGE_ID --name jetson-01
python -m apps.edge_manager command EDGE_ID ping
python -m apps.edge_manager revoke EDGE_ID
python -m apps.edge_manager remove EDGE_ID
```

Docker Compose로 manager를 실행 중이라면 같은 컨테이너의 registry를 사용하는
CLI를 다음처럼 호출합니다.

```bash
docker compose exec edge_manager python -m app list
```

`--endpoint`, `--topic-root`, `--registry`는 subcommand 앞에 둡니다. 환경 변수
`ZENOH_ENDPOINT`, `EDGE_TOPIC_ROOT`, `EDGE_REGISTRY_PATH`로도 지정할 수 있습니다.

Registry 기본 위치는 `apps/edge_manager/data/edges.json`입니다. 엣지는 처음
발견되면 `approved=false`로 저장되며 heartbeat가 15초 동안
없으면 offline 처리됩니다.

## 토픽

```text
dt/edges/{edge_id}/status
dt/edges/{edge_id}/inference
dt/edges/{edge_id}/command
dt/edges/{edge_id}/ack
```

현재 명령은 `ping`, `info`, `shutdown`을 지원합니다. `shutdown`은 Jetson을
종료하는 명령이 아니라 edge agent 프로세스만 정상 종료합니다.

현재 최소 구현에서는 registry와 `server_worker`의 승인 엣지 목록을 자동으로
동기화하지 않습니다. 서버 추론을 붙일 때는 승인된 엣지의 ID와
`dt/edges/{edge_id}/inference` 토픽을 server worker 설정에 사용해야 합니다.
