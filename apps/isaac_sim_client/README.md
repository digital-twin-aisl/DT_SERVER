# Meta Sejong Isaac Sim 연동 클라이언트

`meta_sejong_script.py`는 기본적으로 `server_worker/inference.py`가 발행하는
`SceneOutput` JSON을 Zenoh로 구독하여 Isaac Sim의 USD Stage에 반영합니다.
기존 `sim_backend` WebSocket 경로도 선택적으로 사용할 수 있습니다.

## 작동 원리

* 기본 Zenoh 토픽은 `meta-sejong/scene/v1`입니다.
* Zenoh callback은 최신 Scene 하나만 queue에 저장합니다.
* Isaac Sim update loop가 queue를 소비하므로 네트워크 스레드에서 USD를 직접
* Isaac Sim update loop가 queue를 소비하므로 네트워크 스레드에서 USD를 직접
  수정하지 않습니다.
* 수신된 `global_id`별로 `/World/MetaSejong_People/Person_<id>` 캡슐을
  생성하거나 이동합니다.
* 현재 단계에서는 `root.position`만 표시합니다. skeleton은 통신과 좌표 정렬을
  확인한 다음 추가합니다.

## 시작하기 전에

작업 컴퓨터와 GPU 서버의 설치 경로가 다를 수 있으므로 다음 값을 각 환경에
맞게 설정합니다. `DT_SERVER_DIR`와 `ISAAC_SIM_DIR`는 아래 명령에서 사용하는
셸 변수이고, `ISAAC_USD_PATH`는 `meta_sejong_script.py`가 직접 읽는 환경변수입니다.

```bash
export DT_SERVER_DIR=/path/to/DT_SERVER
export ISAAC_SIM_DIR=/path/to/isaac-sim-standalone-4.2.0
export ISAAC_USD_PATH=/path/to/2025_SejongUniv_All.usd
export ISAAC_SCENE_TRANSPORT=zenoh
export ISAAC_ZENOH_ENDPOINT=192.168.0.73:20522
export ISAAC_ZENOH_TOPIC=meta-sejong/scene/v1
```

`ISAAC_USD_PATH`에는 GPU 서버에 복사한 USD 파일의 절대경로를 지정합니다.
USD에서 상대경로로 참조하는 텍스처와 하위 USD 파일도 동일한 디렉터리 구조로
복사해야 합니다. 환경변수를 생략하면 개발 환경의 기존 경로
`/home/dojan/All/2025_SejongUniv_All.usd`를 사용합니다.

`ISAAC_ZENOH_ENDPOINT`는 `inference.py`와 Isaac Sim 양쪽에서 접근 가능한
Zenoh router 주소여야 합니다. 같은 머신에 기본 포트로 zenohd를 실행한다면
`127.0.0.1:7447`을 사용할 수 있습니다.

Isaac Sim 전용 Python 환경에 저장소와 같은 Zenoh 버전을 설치합니다.

```bash
cd "$ISAAC_SIM_DIR"
./python.sh -m pip install eclipse-zenoh==1.9.0
./python.sh -c "import zenoh; print('zenoh import OK')"
```

WebSocket transport도 사용할 경우에만 다음 패키지가 추가로 필요합니다.

```bash
./python.sh -m pip install websockets
```

## inference.py에서 SceneOutput 발행

기존 `inference.py` 실행 명령에 아래 옵션을 추가합니다. Scene publisher endpoint를
생략하면 입력에 사용하는 `--zenoh-endpoint`를 그대로 사용합니다.

```bash
cd "$DT_SERVER_DIR"
python apps/server_worker/inference.py \
    <기존 inference 옵션> \
    --zenoh-endpoint 192.168.0.73:20522 \
    --scene-zenoh-topic meta-sejong/scene/v1
```

입력과 출력에 서로 다른 Zenoh router를 사용하려면 다음 옵션을 추가합니다.

```bash
--scene-zenoh-endpoint <Isaac Sim이 접속할 router IP:PORT>
```

정상적으로 초기화되면 server worker 로그에 다음 메시지가 표시됩니다.

```text
Zenoh scene publisher is ready: topic=meta-sejong/scene/v1 ...
[6/8] Zenoh scene output is ready: topic=meta-sejong/scene/v1
```

Isaac Sim을 실행하기 전에 일반 Python으로 토픽을 확인할 수 있습니다.

```bash
cd "$DT_SERVER_DIR"
python apps/server_worker/tools/scene_zenoh_subscriber.py \
    --endpoint 192.168.0.73:20522 \
    --topic meta-sejong/scene/v1
```

추론 결과가 발행되면 다음과 같이 출력됩니다.

```text
timestamp=... people=1 global_ids=[0]
```

## WebRTC Headless 실행 (Isaac Sim 4.2.0)

웹 브라우저(Frontend API 대시보드)에서 Isaac Sim의 극실사 화면을 직접 보고(`Pixel Streaming`) 마우스/키보드로 제어하려면, Isaac Sim을 **Headless WebRTC 모드**로 구동해야 합니다.

대시보드도 함께 사용할 경우 DT_SERVER 루트에서 백엔드와 대시보드를
실행합니다. Zenoh 통신 자체에는 `sim_backend` WebSocket이 필요하지 않습니다.

```bash
cd "$DT_SERVER_DIR"
docker compose up -d --build
curl --fail http://localhost:8005/api/v1/system/status
```

그 다음 터미널에서 Isaac Sim 설치 경로로 이동한 뒤, 4.2.0에 포함된 WebRTC
전용 런처를 실행합니다. 아래 명령은 **씬 로드, WebRTC 활성화, Python
스크립트 실행을 한 번에 처리**합니다.

```bash
cd "$ISAAC_SIM_DIR"
./isaac-sim.headless.webrtc.sh \
    --exec "$DT_SERVER_DIR/apps/isaac_sim_client/meta_sejong_script.py"
```

로그에 아래 문구가 모두 표시되는지 확인합니다.

```text
The frontend interface is available at http://localhost:8211/streaming/webrtc-demo
[Meta Sejong] Loading stage from <ISAAC_USD_PATH에 지정한 경로> ...
[Meta Sejong] Zenoh subscriber is ready: endpoint=... topic=meta-sejong/scene/v1
[Meta Sejong] First Zenoh scene received: timestamp=... people=...
```

1. 최초 씬 로딩에는 이 환경에서 약 50~60초가 걸립니다. `Isaac Sim Headless WebRTC App is loaded` 로그가 나온 뒤 진행합니다.
2. `ISAAC_SCENE_TRANSPORT=zenoh`에서는 WebSocket 연결을 만들지 않으므로
   `/api/v1/sim/status`의 `connections` 값은 Zenoh 연결 상태를 나타내지 않습니다.
3. `curl --fail http://localhost:8211/streaming/webrtc-demo/`로 WebRTC 페이지가 응답하는지 확인합니다.
4. 웹 브라우저에서 서버의 프론트엔드(`http://<서버IP>:8005/`)로 접속합니다.
5. 자동 재생이 제한된 브라우저에서는 iframe 중앙의 빨간 재생 버튼을 한 번 누릅니다.
6. "3D 트윈 뷰어" 탭 내부에 **Isaac Sim의 화면이 iframe(포트 8211)으로 스트리밍**되어 나타납니다.
7. 웹 상에서 클릭 및 마우스 드래그를 하면 그 제어값이 백그라운드의 Isaac Sim 뷰포트에 전달됩니다.

## Transport 선택

```bash
# inference.py의 SceneOutput을 직접 수신하는 기본 모드
export ISAAC_SCENE_TRANSPORT=zenoh

# 기존 sim_backend WebSocket만 사용
export ISAAC_SCENE_TRANSPORT=websocket

# 두 수신 경로를 모두 실행
export ISAAC_SCENE_TRANSPORT=both
```

`both`는 두 소스가 동일한 사람을 갱신하면 위치가 서로 덮어써질 수 있으므로
마이그레이션 확인 용도로만 사용하는 것이 좋습니다.
