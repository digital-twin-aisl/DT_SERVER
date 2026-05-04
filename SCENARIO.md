# PoC(Proof of Concept) 환경: 엣지-서버 원격 분산 네트워크 연결 및 예상 시나리오

이 문서는 개발이 완료된 백엔드 뼈대를 바탕으로, 서로 다른 물리적 로컬 네트워크(LAN) 상에 존재하는 **Jetson 엣지 디바이스(현장 야외 복도)** 와 **중앙 집중식 AI 서버(교내 전산실 등)** 가 어떻게 상호작용하는지 설명합니다.

---

## 1. 네트워크 연결 개요 (Network Connectivity)

엣지 노드(예: 192.168.0.x / 4G LTE LTE 망)와 클라우드/학내 서버(예: 공인 IP 210.xx.xx.xx)가 통신할 때 겪는 `NAT(Network Address Translation) 트래버셜` 또는 방화벽 인바운드 차단을 우회하기 위한 구조입니다.

### 🛡️ 연결 시나리오 : 하이브리드 프로토콜 (WebRTC DataChannel + gRPC Keep-Alive)

실시간 분석 시스템의 지연 시간(Low Latency) 보장을 위해, 엣지 노드와 중앙 서버는 데이터 유형을 나누어 통신합니다.

*   **고대역폭 실시간 데이터 (WebRTC UDP):** 10 FPS 수준의 무거운 영상 텐서는 Nginx를 통해 443 포트로 시그널링(`POST https://<서버IP>/api/edge/webrtc/offer`)을 맺고 P2P-UDP 채널을 수립합니다. 약간의 패킷 손실이 있더라도 최신 프레임을 100ms 내로 서버로 직통 전송합니다. 
*   **신뢰성 및 제어 명령 (gRPC TCP):** Nginx가 단일로 열어둔 `443` 파이프로 터널을 뚫어두며, HTTP/2를 이용해 서버 $\rightarrow$ 엣지 역방향으로 카메라 조향 제어나 설정값을 안전하게 하달합니다.

#### 동작 플로우:
*   **[1단계]** `Edge` 전원 인가 시, Nginx의 `443` 포트를 통해 `/api/edge` 로 WebRTC SDP Offer를 던지고 텐서 전송용 UDP 세션을 수립.
*   **[2단계]** 동시에, 수립된 `server_ip:443` (경로 `/edge_communication.EdgeManager`) 으로 gRPC TCP 커넥션을 영구적으로 열어 (`Keep-Alive`) 하베스팅 데이터 및 원격 제어 대기.
*   **[3단계]** `Server` 내부의 `camera_manager`에서 엣지를 제어할 땐, 이 gRPC 터널을 통해 명령을 하달.
*   **[결과]** 실시간 10 FPS는 Head-of-Line Blocking 없이 빠르고, 제어 명령은 방화벽 문제 없이 100% 신뢰성 있게 도달합니다.

---

## 2. 예상 시운전 튜토리얼 (Tutorial Scenario)

사용자가 이 디지털 트윈 시스템을 켜서 초기화하고, 교내 야외 복도를 렌더링하는 시나리오입니다.

### 🎥 Step 1. 전원 인가 및 카메라 설정 (Initialization)
1. **[현장]** 엣지 디바이스(Jetson Orin 01)의 전원이 켜지고 카메라 3대에 대한 RTSP 스트림 파싱을 시작합니다.
2. **[서버]** `edge_manager`에 디바이스가 붙었음을 감지하고, `camera_manager` API를 사용하여 ONVIF 제어 명령(PTZ 기본 화각 정렬 등)을 하달하여 현장 카메라들을 일제히 정렬시킵니다.
    *   *내부 통신:* `Frontend -> camera_manager -> Redis -> edge_manager -> gRPC -> Edge`

### 📐 Step 2. 캘리브레이션 트리거 (Calibration Trigger)
1. 사용자는 Frontend GUI에서 "캘리브레이션 시작" 버튼을 누릅니다. (`/api/calibration/request`)
2. 각 엣지 디바이스는 카메라 3대에서 체커보드나 특징점을 딴 스냅샷 배열(Tensor)을 `gRPC`를 통해 `edge_manager`로 쏩니다.
3. `camera_manager`는 엣지가 연산해 온 기초 데이터를 바탕으로 고도화된 `Intrinsic / Extrinsic Matrix`를 연산하여 PostgreSQL에 저장하고 배포합니다.

### 🏃 Step 3. 실시간 동적 추론 (Dynamic Inference Pipeline)
1. 캘리브레이션이 완료되면, 엣지 디바이스는 가벼운 특징값(Feature Vector) 패킷을 WebRTC DataChannel을 통해 100ms 파이프라인(10FPS)으로 무한 송신합니다.
2. 서버는 `edge_manager` 내부에서 `Redis Stream`으로 텐서를 직접 던지고, 연결된 `dl_worker`가 이를 즉각 Pull해 GPU 추론(3D Pose, BBox 산출)을 거칩니다.
3. 추론된 경량화 JSON 형태 `{"persons": [...], "timestamp": ...}` 데이터는 다시 Redis Pub/Sub에 발행되어 디지털 트윈 환경으로 흘러갑니다.

### 📺 Step 4. 디지털 트윈 동기화 구현 (Digital Twin Rendezvous)
1. 교내 통합 관제실(또는 프론트엔드 URL `https://<서버IP>:443/`)에 접속하면, SPA 방식의 **웹 대시보드**가 구동됩니다.
2. 사용자는 "3D 트윈 뷰어" 탭을 통해 백그라운드의 Isaac Sim 렌더링 화면을 WebRTC 픽셀 스트리밍으로 직접 관찰합니다.
3. `sim_backend`의 WebSocket 채널을 통해 뿌려지는 메타데이터를 Isaac Sim 내부의 `meta_sejong_script.py` Extension이 실시간 구독하여 3D 아바타 모션을 업데이트합니다.

---

## 3. 검증 및 테스트 명령어 가이드

현 아키텍처의 WebRTC UDP(데이터) / gRPC TCP(제어) 하이브리드 병렬 동작을 테스트하려면 아래 명령어를 사용합니다.

```bash
# [서버] 1. 인프라 전체 기동 (Nginx 리버스 프록시 포함)
docker compose up -d --build

# [서버] 2. 컨테이너 상태 및 Redis/dl_worker 정상 대기 확인
docker ps
docker logs -f dt_server-dl_worker-1

# [로컬/클라이언트] 3. 더미 엣지 디바이스 구동 (WebRTC 10FPS + gRPC KeepAlive)
# (단일 443 포트 사용. 테스트 환경이므로 호스트 네트워크를 공유하여 접속 테스트)
docker run --rm --network host -v $(pwd)/apps:/apps \
  python:3.10-slim sh -c \
  "pip install grpcio aiortc aiohttp requests protobuf dotenv urllib3 > /dev/null 2>&1 && \
   HTTP_SERVER_URL=https://localhost:443/api/edge \
   gRPC_SERVER_IP_PORT=localhost:443 \
   PYTHONPATH=/ \
   python /apps/edge_client/test_dummy_edge.py"

# [서버] 4. 데이터 플로우 확인 (WebRTC 수신 -> Redis -> dl_worker 추론)
docker logs dt_server-edge_manager-1 | grep WebRTC
docker logs dt_server-dl_worker-1 | grep "추론 및 좌표 변환 완료"
```
