# 5. 새 구역 구성 (우리 현장에 적용하기)

저장소의 예시 구역 `center-b1-corridor`(세종대 센터 지하 1층 복도, 엣지 2대 × 카메라 4대)를
참고해 자기 현장의 구역을 만드는 절차입니다. 원본 배포·보정 파일은 실행 중에 수정되지 않으므로,
예시 파일을 복사해 새 이름으로 작성합니다.

```text
apps/deployments/
├── regions.json                     # 구역 catalog (manager가 읽음)
├── scene_0812_poc.json              # 배포 manifest 예시
├── calibration_result_*.json        # 카메라 보정 결과 (USD world, metre)
├── cache/scene_0812_2_ground.npz    # 지면 삼각형 캐시
└── poc/server.online.json           # 서버 프로파일 (추론 인자)
```

## 1) 좌표계 정하기

- 모든 공간 데이터는 **USD world, Z-up**을 사용합니다. 보정 결과와 manifest의 AOI는 미터,
  모델 내부와 `SceneOutput`은 밀리미터입니다.
- 현장의 3D 모델(USD)이 있으면 그 world 좌표계를 기준으로 삼습니다. 없으면 바닥면만 담은
  `Ground.usd`를 만들어 기준으로 사용할 수 있습니다.

## 2) 지면(Ground)과 캐시

서버·엣지는 USD를 직접 읽지 않고 지면 삼각형 캐시(`.npz`)를 사용합니다. OpenUSD(`pxr`)가 있는
환경에서 한 번 생성합니다.

```bash
python -m dt_common.spatial.ground export Ground.usd apps/deployments/cache/mysite_ground.npz
```

캐시에는 원본 USD의 SHA-256이 기록되며, USD가 바뀌면 다시 생성해야 합니다.

## 3) 카메라 보정

각 카메라의 intrinsic(체커보드)과 world 기준 extrinsic이 필요합니다. 결과는
`calibration_result_*.json` 하나로 모읍니다(`camera_id: "camera/<번호>"`, `camera_matrix`,
`distortion_coefficients`, `world_to_camera`, `camera_to_world`, `position_m`).

| 방법 | 필요한 것 | 라이선스 |
| --- | --- | --- |
| **수동 보정 편집기** ([manual_editor.md](../../apps/calibration_worker/manual_editor.md)) | 각 카메라 영상 한 장면, 3D 지도 | LGPL (추가 의존성 없음) |
| 자동 보정 ([calibration_worker](../../apps/calibration_worker/README.md)) | ArUco 마커 트리, 기준 영상, VGGT-Omega 체크포인트 | VGGT-Omega: FAIR 비상업 |

1. intrinsic: 엣지에서 `camera_setup.py`의 체커보드 측정을 사용합니다.
2. extrinsic: 수동 편집기에서 지도와 영상의 대응점을 맞추거나, 자동 보정을 실행합니다.
3. 운영 중 보정은 구역을 중지한 뒤 manager로 실행합니다(`region calibrate`). 엣지 ACK와 좌표계가
   확인된 결과만 `calibration-overrides.json`에 채택되며, 원본 파일은 바뀌지 않습니다.

## 4) 배포 manifest 작성

`scene_0812_poc.json`을 복사해 수정합니다. 핵심 필드는 다음과 같습니다.

```json
{
  "version": 3,
  "coordinate_system": "USD world, Z-up",
  "world_unit": "metre",
  "model_unit": "millimetre",
  "world_origin_m": [0.0, 0.0, 0.0],
  "scene": {"id": "mysite_v1", "ground_usd": "../../Ground.usd", "ground_cache": "cache/mysite_ground.npz"},
  "calibration_result": "calibration_result_mysite.json",
  "resolution": {"width": 1920, "height": 1080},
  "fps": 30,
  "topic_root": "dt/edges",
  "workspace": {"strategy": "explicit_aoi", "min_views": 2, "root_height_mm": 900.0},
  "edges": [
    {"id": "edge_1", "enabled": true, "camera_ids": [2, 4, 6, 8],
     "workspace": {"polygon_xy_m": [[76.7, -7.4], [71.9, -2.9], [83.2, 9.1], [88.0, 4.6]]}}
  ]
}
```

- 경로는 manifest 파일 기준 상대 경로입니다.
- `camera_ids`는 엣지별로 중복 없는 양의 정수이며, 이 순서가 모델 입력 순서가 됩니다.
- `workspace.polygon_xy_m`(AOI)은 사람을 찾을 바닥 영역입니다. 최소 `min_views`대 카메라에
  보이는 지면 안쪽으로 잡습니다. AOI를 생략하면 여러 카메라에 보이는 가장 큰 지면 영역을 씁니다.
- 한 엣지는 한 구역에만 속할 수 있습니다.

## 5) 서버 프로파일

`poc/server.online.json`을 복사해 `"deployment"`가 새 manifest를 가리키게 합니다. 주요 인자:

| 인자 | 의미 |
| --- | --- |
| `priority_interval` | 전역 Rank/LOD 재계산 주기(초), 기본 0.5 |
| `lod2_count` | 동시에 3D 자세를 계산할 최대 인원 |
| `max_input_age` | 이보다 오래된 엣지 입력은 버림(초) |
| `input_mode` / `input_clock` | `independent`/`live` = 실시간, `dataset` = 녹화 재생 시각 |
| `lod_policy` | `rank`(우선순위) 또는 `all`(모두 자세 계산) |

## 6) 구역 등록

`apps/deployments/regions.json`에 추가하거나 manager API로 등록합니다.

```json
{
  "version": 1,
  "regions": {
    "mysite-lobby": {
      "name": "본관 로비",
      "deployment": "mysite.json",
      "server_profile": "poc/server.mysite.json",
      "scene_topic": "mysite/scene/v1"
    }
  }
}
```

```bash
python -m apps.edge_manager region register mysite-lobby --name "본관 로비" \
  --deployment mysite.json --server-profile poc/server.mysite.json --scene-topic mysite/scene/v1
# 또는 REST API
curl -X PUT http://127.0.0.1:8001/regions/mysite-lobby \
  -H 'Content-Type: application/json' \
  -d '{"name":"본관 로비","deployment":"mysite.json","server_profile":"poc/server.mysite.json","scene_topic":"mysite/scene/v1"}'
```

검증 규칙: 서버 프로파일과 구역은 같은 manifest를 사용해야 하고, 구역마다 장면 토픽이 달라야
하며, manager의 `--topic-root`와 manifest의 `topic_root`가 같아야 합니다.

## 7) 모든 호스트에 같은 파일 배포

엣지와 서버는 같은 manifest, 보정 결과, 지면 캐시를 가져야 합니다. 엣지는 시작 시 배포 파일
일치를 검증하고, 서버는 패킷마다 카메라 순서·보정 digest·scene/workspace ID를 비교해 맞지 않는
입력을 거부합니다. 저장소를 같은 커밋으로 맞추는 것이 가장 간단합니다.

## 8) 브라우저 지도 (선택)

현장 USD를 브라우저용 `map.glb`/`map.json`으로 변환하는 방법은
[frontend_api README](../../apps/frontend_api/README.md#맵-준비)에 있습니다. 지도가 없어도
구역 제어와 사람 표시는 동작합니다.
