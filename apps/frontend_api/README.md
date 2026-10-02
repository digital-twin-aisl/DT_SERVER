> 운영 대시보드는 구역 관리와 3D 뷰어를 제공합니다. [루트 README](../../README.md)의 상시 서비스 설치 후 사용하세요. 아래는 지도 자산 준비와 선택적 Docker 배포 설명입니다.

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

기존 PoC router가 같은 호스트의 `tcp/7447`에 실행 중이면:

```bash
docker compose -f docker-compose.yml up -d --build frontend_api
```

**라우터가 없는 서버에서만** 기존 PoC 설정의 라우터를 함께 시작합니다:

```bash
docker compose -f docker-compose.yml --profile router up -d --build
```

기존 라우터를 사용하는 경우 `router` profile을 켜지 마세요. 다른 라우터/토픽에 연결하려면:

```bash
ZENOH_ENDPOINT=tcp/192.0.2.73:20522 \
docker compose -f docker-compose.yml up -d frontend_api
```

기본 접속 주소는 `127.0.0.1:8005`입니다. 원격 접속은 해당 포트의 SSH/VPN 터널을 사용하세요. HTTPS reverse proxy에는 WebSocket Upgrade 전달을 설정합니다. 구역 화면에서 여는 뷰어는 관리 API에서 해당 구역의 장면 토픽을 조회합니다.

systemd frontend와 Docker frontend를 동시에 같은 8005 포트에 띄우지 않습니다. Docker frontend를 사용해도 관리 서비스와 GPU worker 환경은 호스트에 필요합니다.

## 서버 출력 기록과 파일 재생

기존 실시간 실행은 그대로 유지됩니다. `/viewer`는 기본적으로 실시간 모드이며,
**JSONL 파일 열기**로 전환하면 그 브라우저의 WebSocket만 닫습니다. 서버/엣지 실행,
다른 브라우저, Isaac Sim 구독에는 영향을 주지 않고 Zenoh로 재발행하지 않습니다.

### 1. 출력 저장

관리 대시보드에서는 **장면 기록**을 체크한 뒤 **구역 시작**을 누릅니다. 기존
구역 실행/중지/기록 기능을 그대로 사용합니다. 수동 RootNet v2 실행에서는:

```bash
bash apps/deployments/rootnet_v2/run.sh server tcp/127.0.0.1:7447 \
  --scene-recording apps/server_worker/data/recordings/rootnet_B_v2_001.jsonl
```

엣지는 기존 명령으로 실행합니다. 서버 기록은 기존 파일을 덮어쓰지 않으므로 매번
새 파일명을 사용하세요. `--no-scene-zenoh`를 추가하지 않으면 저장하면서 실시간
뷰어도 계속 볼 수 있습니다. 영상/MKV나 엣지 입력 패킷이 아니라 **최종 SceneOutput
JSONL**을 저장하며, Isaac의 [Scene Player](../isaac_sim_client/exts/meta_sejong.scene_player/docs/README.md)와 같은 파일을 사용합니다.

### 2. 브라우저에서 재생

- 수동 저장 파일: `http://SERVER:8005/viewer` → **JSONL 파일 열기** → 파일 선택 → **재생**.
  파일 선택 창은 **브라우저를 실행한 PC** 기준입니다. 원격 서버의 파일은 먼저
  내려받으세요. 선택한 파일을 서버로 업로드하지 않습니다.
- 관리 서비스의 기록: 대시보드 **작업과 기록 → 뷰어 재생**. 기존 다운로드 API로
  파일을 읽어 새 탭에서만 재생합니다. **구역 재생**은 기존처럼 관리 서비스의 공유
  토픽 재생이므로 서로 다른 기능입니다. 기존 작업이 있는 구역의 실행 규칙도 유지됩니다.
- 파일은 첫 장면에서 일시정지한 상태로 열립니다. 재생/일시정지, 처음/이전/다음,
  장면 슬라이더, 0.25–8배속, 반복 재생을 지원합니다. 끝에서는 마지막 장면을 유지합니다.
- **실시간**을 누르면 파일 재생을 해제하고 원래 구역의 WebSocket에 다시 연결합니다.
  파일 불러오기/다운로드 도중에도 취소할 수 있습니다. 별도 탭을 열면 실시간과
  기록을 동시에 비교할 수 있습니다.

재생 간격은 JSONL의 `timestamp` 차이입니다. dataset 모드에서는 영상 시간 기준이며,
추론 당시의 처리 지연을 다시 재현하지는 않습니다. 보간 없이 기록된 snapshot을
표시하고 빠른 배속에서는 중간 장면을 건너뛸 수 있습니다. 프레임 이동은 모든 저장
장면에 접근합니다. 빈 snapshot은 사람을 제거하며, **일시정지한 파일 장면에는
실시간 5초 만료 규칙을 적용하지 않습니다**. 백그라운드 탭에서는 재생 시계도 멈춥니다.

파일 모드는 Isaac Scene Player와 같이 선언된 `USD world / Z-up / millimetre` 좌표를
그대로 사용합니다. 기존 실시간 모드의 `MetaSejong_People` 부모 변환은 호환성을 위해
유지합니다. 따라서 원본 USD의 해당 부모 변환이 항등이 아니면 두 모드의 표시 위치가
다를 수 있습니다. 기록 파일에 캠퍼스 맵/캘리브레이션은 포함되지 않으므로, 배경 지도와
관측 영역은 현재 서비스의 `map.glb`, `map.json`을 사용합니다.

로컬 파일은 바이트 위치와 시간만 인덱싱하고 필요한 장면을 읽습니다. 관리 서비스
기록은 먼저 다운로드하므로 브라우저 메모리/다운로드 시간이 필요합니다. 파일당
1 GiB, 장면당 16 MiB, 최대 200만 장면/장면당 1,000명으로 제한합니다. 잘못된 좌표,
역순 시간, 중간의 손상된 JSON은 거부하고 중단된 기록의 불완전한 마지막 행만 경고 후
제외합니다. 파일 재생에는 추론 서버/엣지/Zenoh/관리 서비스 연결이 필요하지 않습니다
(관리 기록 다운로드 시 관리 서비스 필요). frontend와 지도 자산은 필요합니다.

### 프런트엔드 반영 및 테스트

소스 실행 환경은 `npm run build --prefix apps/frontend_api/web` 후 브라우저를 새로고침합니다.
Docker 배포라면 `docker compose up -d --build frontend_api`로 frontend만 다시 빌드합니다.
기존 구역/엣지/worker를 재시작할 필요는 없습니다.

```bash
npm test --prefix apps/frontend_api/web
python3 -m unittest discover -s apps/frontend_api/tests -v
# Playwright + Chromium이 설치된 테스트 환경에서 (가짜 장면/맵, 실제 브라우저 UI 검사)
npm run test:browser --prefix apps/frontend_api/web
```

브라우저 검사에는 별도 설치된 `playwright` 또는 `playwright-core`가 필요합니다.
`PLAYWRIGHT_MODULE=/절대경로/playwright-core/index.mjs`, `CHROME_BIN=/절대경로/chrome`으로
기존 설치를 지정할 수 있습니다. 실제 운영 서비스에 접속하거나 구역 실행을 변경하지 않습니다.

## SceneOutput 실시간 동작

- 서버의 root/joints는 mm이며 기존 Isaac 코드처럼 `MetaSejong_People` 좌표계로 해석합니다.
- `voxelpose_15j_xyz`의 같은 14개 limb와 ID별 색상을 사용합니다.
- pose가 없거나 유효하지 않으면 root 위치에 캡슐을 표시합니다.
- 완전한 snapshot에서 빠진 사람은 제거하고 GPU material도 해제합니다. 빈 snapshot은 모든 사람을 제거합니다.
- 네트워크/브라우저 모두 최신 snapshot 하나만 유지합니다. 중계는 최대 30Hz, 느린 클라이언트는 별도 길이 1 queue를 사용합니다.
- 연결 끊김은 자동 재시도합니다. 5초간 수신이 없으면 남은 사람을 제거하고 수신 지연을 표시합니다. 첫 데이터가 없을 때 가짜 사람을 표시하지 않습니다.
- 정지한 화면은 다시 그리지 않아 대기 중 GPU 사용을 줄입니다.
- `가려진 사람 표시`는 차양·건물 뒤의 사람도 보여 줍니다. 해제하면 실제 깊이에 따라 가립니다. 위치 좌표는 바뀌지 않습니다.
- 관측 영역과 카메라 위치·방향은 deployment/calibration에서 읽습니다. `관측 구역`, `캠퍼스 전체`, `사람 따라가기`로 탐색합니다.

## 운영 확인

```bash
curl --fail http://127.0.0.1:8005/healthz
curl --fail http://127.0.0.1:8005/api/v1/viewer/status
docker compose -f docker-compose.yml logs --tail=50 frontend_api
```

기존 inference 실행에서 `--no-scene-zenoh`를 사용하지 않아야 합니다. 출력 router를 입력 router와 다르게 쓰는 경우 worker의 `--scene-zenoh-endpoint`를 뷰어와 같은 router로 설정합니다. 현재 worker가 꺼져 있으면 맵은 표시되고 상태는 장면 대기로 유지됩니다.

중지:

```bash
docker compose -f docker-compose.yml stop frontend_api
```

Isaac 기록·카메라 생성 도구는 `apps/isaac_sim_client`에 그대로 남아 있습니다. 실시간 시각화에는 실행할 필요가 없습니다.

구현 참고: [Three.js GLTFLoader](https://threejs.org/docs/pages/GLTFLoader.html), [OpenUSD XformCache](https://openusd.org/dev/api/class_usd_geom_xform_cache.html).
