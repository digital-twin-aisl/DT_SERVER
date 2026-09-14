# 브라우저 디지털 트윈

`server_worker → Zenoh meta-sejong/scene/v1 → frontend_api /ws/scene → Three.js`

기존 `SceneOutput` schema v1을 그대로 중계합니다. 렌더링은 접속한 브라우저의 GPU에서 수행하며 서버에는 Isaac Sim, CUDA, RTX 또는 Redis가 필요하지 않습니다. 추론 worker와 edge는 기존대로 실행합니다.

- 대시보드: `http://SERVER:8005/`
- 독립 뷰어: `http://SERVER:8005/viewer`
- 상태: `/api/v1/viewer/status` (라우터 구독 준비와 실제 장면 수신 상태를 구분)
- HTTP(S)와 WS(S)가 같은 origin을 사용합니다. VS Code 터널에는 **8005 하나만** 전달합니다. 기존 Isaac WebRTC용 8211/49100/UDP 미디어 전달은 필요하지 않습니다.

## 맵 준비

원본 USD와 참조 파일은 저장소에서 추적하지 않는 로컬 자산입니다. `All/2025_SejongUniv_All.usd`, `All/Props`, `All/Materials`를 기존 서버와 같은 구조로 준비합니다. 원본 파일은 변경하지 않습니다.

Python 3.11+와 Node.js 22 환경에서:

```bash
python3.11 -m venv .venv-three
.venv-three/bin/pip install -r apps/frontend_api/tools/requirements.txt
npm ci --prefix apps/frontend_api/web
MAP_PYTHON=.venv-three/bin/python apps/frontend_api/tools/prepare_map.sh
```

`VIEWER_USD_PATH`와 `VIEWER_DEPLOYMENT`로 다른 USD/배포 manifest를 지정할 수 있습니다. 기본 manifest는 `apps/deployments/scene_0812_poc.json`입니다. manifest가 참조하는 calibration JSON과 ground cache도 필요합니다.

이 작업은 배포 전 한 번 실행하며 `apps/frontend_api/assets/map.glb`, `map.json`을 만듭니다. OpenUSD는 변환 시에만 필요합니다. 재배포 서버에는 이 두 파일만 복사해도 됩니다. 맵 수정 시 다시 변환하고 브라우저를 새로고침합니다.

변환은 USD 가시성, 인스턴스, 메시 변환, normals/UV, material subsets를 처리합니다. glTF로 옮길 때 Z-up→Y-up 회전과 stage unit→metre 변환을 적용합니다. `map.json`에는 원본 `/World/MetaSejong_People`의 누적 변환을 metres로 저장하므로, 이 그룹의 회전·이동·스케일도 사람 렌더링에 유지됩니다. 해당 그룹이 없으면 `/World` 변환 또는 항등 변환을 사용합니다.

MDL은 base-color texture/상수 색상·roughness·metallic을 PBR로 근사합니다. 절차적 MDL, RTX 효과, 광원·애니메이션은 복제하지 않습니다. 변환 경고는 `map.json`에 기록됩니다. Meshopt와 WebP, GPU instancing을 사용하고 형상 단순화는 끕니다. 현재 로컬 캠퍼스 자산은 약 175.5MB에서 19.8MB로 줄었습니다. 원본 맵 자체의 삼각형 수가 많으므로 전체 캠퍼스 화면의 FPS는 브라우저 GPU에 따라 달라집니다.

## 배포

기존 PoC router가 같은 호스트의 `tcp/10020`에 실행 중이면:

```bash
docker compose -f docker-compose.viewer.yml up -d --build browser_viewer
```

**라우터가 없는 서버에서만** 기존 PoC 설정의 라우터를 함께 시작합니다:

```bash
docker compose -f docker-compose.viewer.yml --profile router up -d --build
```

기존 라우터를 사용하는 경우 `router` profile을 켜지 마세요. 다른 라우터/토픽에 연결하려면:

```bash
SCENE_ZENOH_ENDPOINT=tcp/192.168.0.73:20522 \
SCENE_ZENOH_TOPIC=meta-sejong/scene/v1 \
docker compose -f docker-compose.viewer.yml up -d browser_viewer
```

`VIEWER_PORT` 기본값은 8005, `VIEWER_BIND_HOST` 기본값은 0.0.0.0입니다. 기존 VPN/인증 터널로 접속하는 경우 그 주소를 사용하세요. 인터넷 공개용 별도 인증은 이 서비스에 포함하지 않습니다. HTTPS reverse proxy에는 WebSocket Upgrade 전달을 설정합니다.

기존 `docker-compose.yml`의 `frontend_api`도 동일 렌더러를 사용합니다. 독립 compose와 동시에 8005에 띄우지 않습니다.

## SceneOutput 동작

- 서버의 root/joints는 mm이며 기존 Isaac 코드처럼 `MetaSejong_People` 좌표계로 해석합니다.
- `voxelpose_15j_xyz`의 같은 14개 limb와 ID별 색상을 사용합니다.
- pose가 없거나 유효하지 않으면 root 위치에 캡슐을 표시합니다.
- 완전한 snapshot에서 빠진 사람은 제거하고 GPU material도 해제합니다. 빈 snapshot은 모든 사람을 제거합니다.
- 네트워크/브라우저 모두 최신 snapshot 하나만 유지합니다. 중계는 최대 30Hz, 느린 클라이언트는 별도 길이 1 queue를 사용합니다.
- 연결 끊김은 자동 재시도합니다. 5초간 수신이 없으면 남은 사람을 제거하고 수신 지연을 표시합니다. 첫 데이터가 없을 때 가짜 사람을 표시하지 않습니다.
- 정지한 화면은 다시 그리지 않아 대기 중 GPU 사용을 줄입니다.
- `가려진 사람 표시`는 차양·건물 뒤의 사람도 보여 줍니다. 해제하면 실제 깊이에 따라 가립니다. 위치 좌표는 바뀌지 않습니다.
- 관측 영역과 카메라 위치·방향은 deployment/calibration에서 읽습니다. `관측 구역`, `캠퍼스 전체`, `사람 따라가기`로 탐색합니다.

## 검증 및 운영

```bash
.venv-three/bin/pip install -r apps/frontend_api/requirements.txt pytest
.venv-three/bin/python -m pytest apps/frontend_api/tests -q
npm test --prefix apps/frontend_api/web
curl --fail http://127.0.0.1:8005/healthz
curl --fail http://127.0.0.1:8005/api/v1/viewer/status
docker compose -f docker-compose.viewer.yml logs --tail=50 browser_viewer
```

브라우저 smoke test:

```bash
npx --prefix apps/frontend_api/web playwright install chromium
node apps/frontend_api/web/smoke.mjs
```

`smoke.mjs`는 실제 GLB가 로딩되고 브라우저 예외가 없는지 검사합니다. 단위 테스트는 단위/좌표 변환, pose fallback, 삭제, bounded fan-out, 늦은 구독자와 WebSocket 수명주기를 검증합니다.

기존 inference 실행에서 `--no-scene-zenoh`를 사용하지 않아야 합니다. 출력 router를 입력 router와 다르게 쓰는 경우 worker의 `--scene-zenoh-endpoint`를 뷰어와 같은 router로 설정합니다. 현재 worker가 꺼져 있으면 맵은 표시되고 상태는 장면 대기로 유지됩니다.

중지:

```bash
docker compose -f docker-compose.viewer.yml stop browser_viewer
```

Isaac 기록·카메라 생성 도구는 `apps/isaac_sim_client`에 그대로 남아 있습니다. 실시간 시각화에는 실행할 필요가 없습니다.

구현 참고: [Three.js GLTFLoader](https://threejs.org/docs/pages/GLTFLoader.html), [OpenUSD XformCache](https://openusd.org/dev/api/class_usd_geom_xform_cache.html).
