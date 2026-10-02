# 7. 데이터 계약

다른 시스템이 장면을 구독하거나, 자체 엣지를 만들어 서버에 입력을 보낼 때 지켜야 하는 형식입니다.
코드 기준 정의는 `packages/dt_common/src/dt_common/contracts/`와 `inference_codec.py`이며,
이 문서와 코드가 다르면 **코드가 기준**입니다.

## 전송: Zenoh 1.9.0

| 키 | 방향 | 내용 |
| --- | --- | --- |
| `dt/edges/{edge_id}/status` | 엣지 → manager | heartbeat, 승인·run 상태 (JSON) |
| `dt/edges/{edge_id}/cameras` | 엣지 → manager | 카메라별 연결 상태 (JSON) |
| `dt/edges/{edge_id}/command` | manager → 엣지 | 시작·중지·lease 갱신 등 명령 (JSON) |
| `dt/edges/{edge_id}/ack` | 엣지 → manager | 명령 결과 (JSON) |
| `dt/edges/{edge_id}/config` | manager → 엣지 | 승인 후 설정 (JSON) |
| `dt/edges/{edge_id}/inference` | 엣지 → 서버 | 2D 추론 결과 (ZNH2 바이너리) |
| `{scene_topic}` 예: `meta-sejong/scene/v1` | 서버 → 구독자 | `SceneOutput` (JSON) |

- `dt/edges`는 manifest의 `topic_root`로 바꿀 수 있습니다. 엣지 ID에 구역을 넣지 않습니다.
- 카메라 식별자는 `{edge_id}/camera/{번호}`입니다.
- router 기본 주소는 `tcp/127.0.0.1:7447`(서버 로컬)입니다.

## SceneOutput v1

서버가 발행하고, 기록 파일(JSONL)의 한 줄이 되는 장면입니다.

```json
{
  "schema_version": 1,
  "coordinate_system": {"frame": "USD world", "up_axis": "Z", "unit": "millimetre"},
  "timestamp": 1786349132.91,
  "sync_spread_seconds": 0.012,
  "people": [
    {
      "global_id": 3,
      "lod": 2,
      "root": {"edge_id": "edge_1", "candidate_index": 0,
               "position": [80123.4, 512.0, 14800.0], "confidence": 0.82, "timestamp": 1786349132.90},
      "pose": {"joint_format": "voxelpose_15j_xyz", "joints": [[80110.0, 500.0, 15300.0], "... 15개"]}
    }
  ],
  "runtime": {"input_status": {"edge_1": "..."}, "playback": false}
}
```

| 필드 | 의미 |
| --- | --- |
| `timestamp` | 장면 시각(초). 실시간은 Unix 시각, 데이터셋 재생은 `frame/fps` |
| `sync_spread_seconds` | 장면에 합쳐진 엣지 입력 간 최대 시각 차 |
| `global_id` | 구역 안에서 유일한 사람 ID (0 이상 정수) |
| `lod` | 0=위치, 1=위치+ID, 2=3D 자세 포함 |
| `root.position` | 골반 중심(mid-hip), USD world mm |
| `pose.joints` | `voxelpose_15j_xyz` 순서의 15개 관절, USD world mm, 자세가 없으면 `pose: null` |
| `runtime` | 선택. `input_status`(엣지별 입력 상태), `playback`(재생 여부) 등 |

`voxelpose_15j_xyz` 관절 순서: 0 목, 1 코, 2 골반 중심(root), 3 왼어깨, 4 왼팔꿈치, 5 왼손목,
6 왼엉덩이, 7 왼무릎, 8 왼발목, 9 오른어깨, 10 오른팔꿈치, 11 오른손목, 12 오른엉덩이,
13 오른무릎, 14 오른발목.

Python에서는 `dt_common.contracts.scene`의 `SceneOutput`/`PersonEntity`를 사용해 만들고
`to_dict()`로 직렬화합니다([합성 장면 생성기](../../examples/synthetic_region/generate_scene.py) 참고).
구독 예시는 `apps/server_worker/tools/scene_zenoh_subscriber.py`입니다.

## 단위와 좌표계

| 대상 | 좌표계 | 단위 |
| --- | --- | --- |
| `SceneOutput`, 모델 내부 root/관절 | USD world, Z-up | mm |
| 배포 manifest AOI, 보정 결과 위치 | USD world, Z-up | m |
| 우선순위·위험 지점 입력 | USD world, Z-up | m |
| Isaac 스테이지 (개발 현장) | USD world, Z-up | `metersPerUnit=0.01` |

## ZNH2 추론 패킷 (엣지 → 서버)

- 헤더 `!4sdI`: magic `ZNH2`, timestamp(float64), 압축 전 크기(uint32)
- 본문: zstd 압축된 JSON 메타데이터 + 연속된 ndarray 바이트
- 허용 dtype: `|u1 <f4 <f8 <i4 <i8`, 차원 ≤ 5, 메시지 ≤ 32 MiB. **pickle은 사용하지 않습니다.**
- 서버는 카메라 순서, 유효 보정 digest, scene/workspace ID, 텐서 크기, clock 종류,
  session/sequence를 검사해 맞지 않는 패킷을 거부합니다.
- 구버전 ZNH1은 거부합니다. 코덱을 바꾸면 엣지와 서버를 함께 업그레이드해야 합니다.

인코딩·디코딩은 `dt_common.inference_codec.encode_frame` / `decode_frame`을 사용하세요.

## 관리 REST API (manager, 기본 127.0.0.1:8001)

| 요청 | 의미 |
| --- | --- |
| `GET /healthz` | 상태 확인 |
| `GET /regions`, `GET /regions/{id}` | 구역·엣지·카메라와 실행 상태 |
| `PUT /regions/{id}` / `DELETE /regions/{id}` | 중지된 구역 등록·수정 / 해제 |
| `POST /regions/{id}/start` | 실시간 실행, 본문 `{"record": true}` 선택 |
| `POST /regions/{id}/stop` | 실행 종료 |
| `GET /regions/{id}/logs?edge_id=` | 서버 또는 엣지 로그 끝부분 |
| `GET /regions/{id}/recordings`, `.../recordings/{run_id}/download` | 기록 목록·다운로드 |
| `POST /regions/{id}/replay` | 기록 재생 실행 |
| `POST /regions/{id}/calibrate` | 보정 작업 (`edge_id`, `reference_video`, `marker_tree`, `checkpoint`) |
| `GET /edges`, `POST /edges/{edge_id}/{approve|revoke|remove}` | 장치 목록·관리 |

`DT_MANAGER_TOKEN`이 설정되어 있으면 `Authorization: Bearer <token>` 헤더가 필요합니다.
GUI(8005)는 같은 API를 `/api/v1/control/...` 경로로 중계합니다.
