# 구역·장치 관리자

`python -m apps.edge_manager`는 GUI와 동일한 관리 API를 호출하는 CLI입니다. GPU 라이브러리는 관리 프로세스에 import하지 않으며, 준비된 Python 환경의 worker를 자식 프로세스로 감독합니다.

최초 설치와 구역 시작은 [루트 README](../../README.md)를 참고하세요.

```bash
python -m apps.edge_manager serve
python -m apps.edge_manager list
python -m apps.edge_manager approve edge_1
python -m apps.edge_manager region list
python -m apps.edge_manager region start center-b1-corridor
python -m apps.edge_manager region status center-b1-corridor
python -m apps.edge_manager region logs center-b1-corridor --edge-id edge_1
python -m apps.edge_manager region stop center-b1-corridor
```

서버가 상시 실행된 이후 CLI에 Zenoh 주소와 registry 경로를 매번 전달하지 않습니다. `DT_MANAGER_URL`은 기본 `http://127.0.0.1:8001`이며 토큰이 있으면 `DT_MANAGER_TOKEN`을 사용합니다. `--endpoint`, `--registry`, `--catalog`는 **serve의 설정**입니다. 기존 `list/approve/revoke/remove/command`의 이름은 유지하지만 이제 같은 관리 API를 거칩니다.

## 관리 API

| 요청 | 의미 |
| --- | --- |
| GET /regions | 구역 → 엣지 → 카메라와 실제 실행 상태 |
| GET /regions/{id} | 특정 구역 상태 |
| PUT /regions/{id} | 중지된 구역 등록/구성. name, deployment, server_profile, scene_topic |
| DELETE /regions/{id} | 중지된 구역 등록 해제 |
| POST /regions/{id}/start | 구역 실시간 실행. 선택적 `{"record": true}` |
| POST /regions/{id}/stop | 구역 작업 종료 |
| GET /regions/{id}/logs?edge_id=... | 서버 또는 엣지 로그 끝부분 |
| GET /regions/{id}/recordings | 작업·기록 목록 |
| GET /regions/{id}/recordings/{run_id}/download | SceneOutput JSONL 다운로드 |
| POST /regions/{id}/replay | `{"recording_id": "..."}` 출력 장면 재생 |
| POST /regions/{id}/calibrate | edge_id, reference_video, marker_tree, checkpoint |
| GET /edges | 발견된 장치·승인·연결 상태 |
| POST /edges/{id}/approve | 장치 승인. 선택적 name, endpoint |
| POST /edges/{id}/revoke | 승인 해제와 lease 갱신 중단 |
| POST /edges/{id}/ping, info, shutdown | 기존 agent 명령 |

GET `/docs`는 OpenAPI 문서입니다. frontend는 `/api/v1/control/...`로 그대로 중계합니다. 외부 바인딩은 관리 토큰이 있어야 하며 cross-origin 요청은 허용하지 않습니다.

## 모듈 경계

- `topology.py`: 구역 소속 검증과 배포 참조
- `registry.py`: 기존 장치 발견·승인·카메라 상태 영속화
- `control.py`: CLI/GUI에 공통인 작업 수명주기
- `transport.py`: edge 명령과 ACK 상관관계, 타임아웃
- `service.py`: Zenoh 구독, HTTP 서비스, 상태 갱신
- `recording.py`, `scene_replay.py`: 추론과 독립된 출력 기록·재생
- `dt_common.process`: 서버·엣지 공통 프로세스 감독

Registry의 기본 위치는 `apps/edge_manager/data/edges.json`입니다. 기존 v1 파일을 그대로 읽습니다. 구역 정보는 배포 manifest를 참조하므로 기존 카메라/승인 레코드를 삭제하거나 재번호 매기지 않습니다. 모니터링은 5초 heartbeat, 15초 offline 판정을 기본으로 사용합니다.
