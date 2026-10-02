# 6. 운영

설치가 끝나면 서버·엣지의 추론 스크립트를 직접 실행하지 않습니다. 모든 작업은 manager의
CLI 또는 GUI(`http://SERVER:8005/`)로 하며, 둘은 같은 관리 API를 호출합니다.

## 일상 작업

```bash
python -m apps.edge_manager region list
python -m apps.edge_manager region start center-b1-corridor
python -m apps.edge_manager region status center-b1-corridor          # --json 으로 기계 판독
python -m apps.edge_manager region logs center-b1-corridor --edge-id edge_1
python -m apps.edge_manager region stop center-b1-corridor
```

원격 서버의 manager를 쓰려면 `DT_MANAGER_URL`(과 필요 시 `DT_MANAGER_TOKEN`)을 지정합니다.

## 장치 관리

```bash
python -m apps.edge_manager list               # 발견·승인 상태, 마지막 연결, 카메라 상태
python -m apps.edge_manager approve edge_1
python -m apps.edge_manager revoke edge_1      # 승인 취소
python -m apps.edge_manager remove edge_1      # registry에서 삭제
```

## 상태의 의미

프로세스 생성, agent heartbeat, 실제 관측 도착은 서로 다른 증거입니다. 실행 요청이 성공했다고
`running`이 되지 않습니다.

| 상태 | 의미 |
| --- | --- |
| `starting` | 설정 검증, 프로세스 준비, 첫 관측 대기 |
| `running` | 서버 동작 + 모든 참여 엣지 승인·연결 + 새로운 유효 관측(`runtime.input_status`) 도착 |
| `degraded` | 일부 엣지 실패/관측 누락. 정상 엣지 처리는 유지 |
| `failed` | 설정 검증 실패 또는 서버 재시도 한도(3회) 초과. 로그에 원인 |
| `stopping` / `stopped` | 정지 처리 중 / 로컬 정지 완료. ACK 없는 엣지는 30초 lease 만료로 정지 |
| `interrupted` | manager 재시작 전에 끝나지 않은 run. 자동 재개하지 않음 |

- 추론 프로세스는 지수 대기로 최대 3회 재시도합니다.
- 엣지는 30초 동안 lease 갱신이 없으면 관리 중인 추론을 스스로 중지합니다.
- 재생된 장면(`runtime.playback=true`)은 실시간 관측으로 집계하지 않습니다.
- 같은 구역에서 실시간·재생·보정 작업은 동시에 실행되지 않습니다. 옵션을 바꾸려면 중지 후 다시 시작합니다.

## 기록과 재생

```bash
python -m apps.edge_manager region start center-b1-corridor --record
python -m apps.edge_manager region stop center-b1-corridor
python -m apps.edge_manager region recordings center-b1-corridor
python -m apps.edge_manager region replay center-b1-corridor RECORDING_RUN_ID
```

- 기록은 원본 영상이 아니라 `SceneOutput` JSONL(`data/manager/runs/<run_id>/scenes.jsonl`)입니다.
  사람 ID·위치·자세가 포함되므로 **익명성이나 개인정보 적합성을 보증하지 않습니다.**
- 기록은 명시적으로 켤 때만 시작하며, 파일 상한은 `DT_RECORDING_MAX_BYTES`(기본 1 GiB),
  남은 디스크가 256 MiB 미만이면 중단합니다. 저장 실패는 실시간 처리를 멈추지 않고 누락 수로 표시됩니다.
- 보관 기간과 삭제는 자동으로 결정하지 않습니다. 운영 정책에 따라 관리하세요.
- GUI의 기록 목록에서 내려받거나, 뷰어의 **JSONL 파일 열기**로 재생합니다.

## 보정

```bash
python -m apps.edge_manager region calibrate center-b1-corridor \
  --edge-id edge_1 --reference-video /data/reference.mp4 \
  --marker-tree /data/marker-tree.usd --checkpoint /models/VGGT-Omega.pt
```

구역을 중지한 상태에서 실행합니다. 엣지 적용 ACK와 좌표계가 확인된 결과만 다음 실행부터
공통 보정에 병합됩니다. 실제 정합 품질은 현장에서 눈으로 확인해야 합니다.

## run 기록 위치

`data/manager/runs/<run_id>/`에 manifest, 생성된 서버 설정, 서버·엣지 로그, 선택적
`scenes.jsonl`이 남습니다. 문제를 보고할 때 이 폴더의 manifest와 로그 끝부분을 첨부하면
원인 파악이 빠릅니다(RTSP 주소는 포함되지 않지만, 올리기 전에 한 번 확인하세요).

## 업그레이드

1. 구역을 모두 중지합니다.
2. [CHANGELOG.md](../../CHANGELOG.md)에서 호환성 변경(특히 ZNH2 코덱, `SceneOutput`)을 확인합니다.
3. 서버와 **모든 엣지**를 같은 버전으로 올립니다(`git pull` 후 각 환경에서 의존성 재설치).
   `dt-common` 버전이 다르면 관리 경로가 동작하지 않습니다.
4. GUI를 다시 빌드합니다: `npm ci --prefix apps/frontend_api/web && npm run build --prefix apps/frontend_api/web`
5. manager와 agent를 재시작하고 구역을 시작합니다.
