# Meta Sejong Isaac Sim 연동 클라이언트

## 9-view 영상 녹화 Extension

현재 Viewport와 보정 카메라 8대를 동시에 MP4로 저장하는
`Meta Sejong Multi-Camera Recorder` Extension이
`exts/meta_sejong.multi_camera_recorder`에 있습니다. 설치, 출력 구조와 사용법은
[Extension 문서](exts/meta_sejong.multi_camera_recorder/docs/README.md)를
참고하십시오.

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

## 외부 인터넷에서 보기

Isaac Sim 4.2.0 WebRTC는 웹 페이지만 열어 주는 구조가 아닙니다. 브라우저가
아래 세 경로 모두에 도달해야 실제 영상이 나옵니다.

| 포트 | 프로토콜 | 용도 |
| --- | --- | --- |
| `8211` | TCP | 내장 WebRTC 웹 클라이언트 |
| `49100` | TCP | WebRTC 시그널링 |
| `47998` | UDP | 영상/입력 미디어 |

위 표는 기본값입니다. 포트 정책에 맞춰 UDP 미디어 포트는 바꿀 수 있습니다.
현재처럼 `10000~11000/UDP`만 허용되고 `10020/UDP`를 Zenoh가 사용한다면,
WebRTC에는 충돌하지 않는 `10021/UDP`를 사용하면 됩니다.

### 현재 서버 권장 구성: GitHub/VS Code Remote Tunnel + 10021/UDP

이 방식은 Isaac Sim의 인증 없는 HTTP/시그널링 포트를 인터넷에 공개하지 않습니다.
서버에서 이미 동작 중인 Remote Tunnel의 outbound `443/TCP` 연결을 그대로 쓰므로,
Compose에서 `edge_manager`가 사용하는 host `443/TCP`와도 충돌하지 않습니다.

```text
외부 Chrome ── VS Code Remote Tunnel(outbound 443) ──> 8005/TCP (Dashboard)
                                                    ├─> 8211/TCP (Web client)
                                                    └─> 49100/TCP (Signaling)
외부 Chrome ─────────────── direct Internet ──────────> 10021/UDP (Media)
Edge Zenoh  ─────────────── direct Internet ──────────> 10020/UDP
```

3090 서버에서 외부 클라이언트가 접속할 서버의 **공인 IPv4**를 지정하고 Isaac
Sim을 실행합니다.

```bash
# 아래 TEST-NET 주소를 실제 3090 서버 공인 IPv4로 바꾸십시오.
export ISAAC_PUBLIC_IP=203.0.113.10
export ISAAC_WEBRTC_STREAM_PORT=10021
export ISAAC_WEBRTC_SIGNAL_PORT=49100
export ISAAC_WEBRTC_WEB_PORT=8211

cd "$DT_SERVER_DIR"
apps/isaac_sim_client/run_remote_webrtc.sh
```

같은 3090 서버에서 `frontend_api`도 실행합니다. VS Code Desktop이 원격 포트를
로컬의 같은 번호로 전달할 것이므로 브라우저 기준 주소는 `127.0.0.1`로 설정합니다.

```bash
export ISAAC_WEBRTC_PUBLIC_URL=http://127.0.0.1:8211
export ISAAC_WEBRTC_SERVER=127.0.0.1
export ISAAC_WEBRTC_INTERNAL_URL=http://host.docker.internal:8211
export ISAAC_WEBRTC_STREAM_PORT=10021

docker compose up -d --build frontend_api
```

서버 방화벽에서는 `10021/UDP`만 외부 클라이언트 공인 IP로 제한하는 것이 가장
안전합니다. UFW를 쓴다면 아래 TEST-NET 주소를 실제 접속 PC의 공인 IP로
바꿉니다.

```bash
export ISAAC_VIEWER_PUBLIC_IP=198.51.100.25
sudo ufw allow from "$ISAAC_VIEWER_PUBLIC_IP" to any port 10021 proto udp \
  comment 'temporary Isaac WebRTC media'
```

외부 PC에서는 **VS Code Desktop**으로 현재 GitHub 인증 Remote Tunnel에 접속한
후 `PORTS` 패널에서 다음 세 원격 포트를 추가합니다.

| 원격 포트 | 로컬 주소 | 표시 이름 |
| --- | --- | --- |
| `8005` | `127.0.0.1:8005` | DT Dashboard |
| `8211` | `127.0.0.1:8211` | Isaac Web Client |
| `49100` | `127.0.0.1:49100` | Isaac Signaling |

자동 배정된 로컬 포트가 다르면 포트를 우클릭하고 **Change Local Address Port**로
원격 포트와 같은 번호로 맞춥니다. 특히 `49100`은 반드시 로컬에서도
`49100`이어야 합니다. 브라우저용 `app.github.dev` HTTPS 주소가 아니라 VS Code
Desktop이 표시하는 `127.0.0.1` 주소를 사용해야 mixed-content/TLS 문제를
피할 수 있습니다.

그 후 외부 PC의 Chrome/Chromium에서 다음 주소를 엽니다.

```text
대시보드: http://127.0.0.1:8005/
Isaac 직접: http://127.0.0.1:8211/streaming/webrtc-demo/?server=127.0.0.1
```

두 주소 중 하나를 사용하면 됩니다.

시청이 끝나면 Isaac Sim을 종료하고 VS Code의 전달 포트를 닫습니다. 임시 UFW
규칙을 추가했다면 같이 제거합니다.

```bash
export ISAAC_VIEWER_PUBLIC_IP=198.51.100.25
sudo ufw delete allow from "$ISAAC_VIEWER_PUBLIC_IP" to any port 10021 proto udp
```

`10000~11000/UDP` 범위가 이미 전체 인터넷에 항상 열려 있다면 규칙을 추가할
필요는 없지만, Isaac Sim이 실행 중인 동안 `10021/UDP`에는 실제 미디어 서비스가
열립니다. 가능하면 기존 범위 규칙도 소스 IP 제한 방식으로 좁히십시오.

`frontend_api` 대시보드는 별도로 `8005/TCP`를 사용합니다. Isaac Sim 4.2의
내장 스트리밍 엔드포인트에는 인증과 TLS가 없습니다. 일반적으로는 mesh VPN도
가능하지만, 현재 서버에서는 위의 **Remote Tunnel + 제한된 `10021/UDP` 방식**을
사용합니다.

> Jetson/Orin 같은 `aarch64` 장비에서는 공식 Isaac Sim 4.2.0 데스크톱 앱을
> 실행하는 구성을 전제로 하면 안 됩니다. 이 경우 `frontend_api`와 나머지
> DT_SERVER 서비스는 현재 장비에 두고, Isaac Sim과 이 디렉터리의 실행 스크립트는
> 별도 `x86_64 + RTX/NVENC` 호스트에서 실행합니다. 브라우저는 두 호스트 모두에
> 도달해야 합니다.

### 대안: mesh VPN 주소로 실행

Isaac Sim 호스트의 VPN IPv4가 `100.80.10.20`, DT_SERVER 호스트의 VPN IPv4가
`100.80.10.10`이라고 가정합니다. 두 역할이 같은 RTX 서버라면 같은 IP를
사용하면 됩니다.

Isaac Sim 호스트에서 저장소와 Isaac Sim 4.2.0을 준비한 후 실행합니다.

```bash
export ISAAC_SIM_DIR=/path/to/isaac-sim-standalone-4.2.0
export ISAAC_PUBLIC_IP=100.80.10.20

cd "$DT_SERVER_DIR"
apps/isaac_sim_client/run_remote_webrtc.sh
```

DT_SERVER/Frontend API 호스트에서는 브라우저가 접속할 Isaac 주소와 상태 확인용
내부 주소를 설정합니다.

```bash
export ISAAC_WEBRTC_PUBLIC_URL=http://100.80.10.20:8211
export ISAAC_WEBRTC_SERVER=100.80.10.20
export ISAAC_WEBRTC_INTERNAL_URL=http://100.80.10.20:8211

cd "$DT_SERVER_DIR"
docker compose up -d --build frontend_api
```

같은 VPN에 로그인한 외부 PC의 Chrome/Chromium에서 다음 주소를 엽니다.

```text
http://100.80.10.10:8005/
```

대시보드가 아니라 Isaac Sim 화면만 바로 보려면 다음 주소를 엽니다.

```text
http://100.80.10.20:8211/streaming/webrtc-demo/?server=100.80.10.20
```

### 공인 IP/포트포워딩으로 직접 연결

VPN을 사용할 수 없을 때만 공유기 또는 클라우드 보안 그룹에서 `8211/TCP`,
`49100/TCP`, `10021/UDP`를 서버로 전달합니다. 대시보드까지 열려면
`8005/TCP`도 추가합니다. 가능한 경우 소스 CIDR을 접속할 외부 PC의 공인
IP(`/32`)로 제한하십시오.

```bash
# 예시는 문서용 TEST-NET 주소입니다. 실제 서버 공인 IPv4로 바꾸십시오.
export ISAAC_PUBLIC_IP=203.0.113.10
export ISAAC_WEBRTC_PUBLIC_URL=http://203.0.113.10:8211
export ISAAC_WEBRTC_SERVER=203.0.113.10

docker compose up -d --build frontend_api
apps/isaac_sim_client/run_remote_webrtc.sh
```

`run_remote_webrtc.sh`는 4.2.0의 legacy Kit 설정에 공인/VPN 주소, 고정 시그널링
포트, 고정 UDP 미디어 포트를 넘긴 뒤 기존 `meta_sejong_script.py`를 실행합니다.
추가 Isaac Sim 인자는 명령 마지막에 그대로 붙일 수 있습니다.

```bash
apps/isaac_sim_client/run_remote_webrtc.sh --/app/window/width=1920 --/app/window/height=1080
```

### HTTPS와 터널 사용 시 주의

Cloudflare Tunnel 같은 HTTP 터널 하나로 `frontend_api:8005`만 공개하면
대시보드 자체는 보이지만 Isaac Sim 영상은 나오지 않습니다. 4.2.0 WebRTC는
별도의 TCP 시그널링과 UDP 미디어 연결이 필요하기 때문입니다. 또한 HTTPS
대시보드 안의 HTTP iframe은 브라우저의 mixed-content 정책으로 차단됩니다.
이 경우 대시보드의 **새 창에서 열기**를 사용하더라도 위 WebRTC 포트의 직접
도달성은 필요합니다. 한 개의 HTTPS/TCP 터널만 허용되는 환경에서는 WebRTC
대신 noVNC 또는 HLS 같은 별도 TCP 기반 화면 중계 계층을 구성해야 합니다.

### 확인 순서

서버에서:

```bash
ss -lnt | grep -E ':(8005|8211|49100) '
ss -lnu | grep ':10021 '
curl --fail http://127.0.0.1:8211/streaming/webrtc-demo/
curl --fail http://127.0.0.1:8005/api/v1/viewer/status
```

외부 PC에서 먼저 `http://<접속주소>:8211/...`를 직접 열어 영상이 나오는지
확인한 뒤 대시보드 iframe을 확인합니다. 웹 페이지는 열리지만 영상이 검다면
대부분 `49100/TCP`, `10021/UDP`, 또는 `ISAAC_PUBLIC_IP` 광고 주소 문제입니다.
Firefox보다 Chrome/Chromium을 사용하십시오.

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
