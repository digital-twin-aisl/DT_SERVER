# PoC(Proof of Concept) 환경: 엣지-서버 원격 분산 네트워크 연결 및 예상 시나리오

이 문서는 개발이 완료된 백엔드 뼈대를 바탕으로, 서로 다른 물리적 로컬 네트워크(LAN) 상에 존재하는 **Jetson 엣지 디바이스(현장 야외 복도)** 와 **중앙 집중식 AI 서버(교내 전산실 등)** 가 어떻게 상호작용하는지 설명합니다.

---

## 1. 네트워크 연결 개요 (Network Connectivity)

엣지 노드(예: 192.168.0.x / 4G LTE LTE 망)와 클라우드/학내 서버(예: 공인 IP 210.xx.xx.xx)가 통신할 때 겪는 `NAT(Network Address Translation) 트래버셜` 또는 방화벽 인바운드 차단을 우회하기 위한 구조입니다.

### 🛡️ 연결 시나리오 : Edge Initiated gRPC Streaming (Client $\rightarrow$ Server 단방향 수립)

서버(방화벽 내부)가 직접 현장의 엣지 디바이스로 연결(TCP PUSH)을 시도하면 연결이 실패할 확률이 높습니다(포트 포워딩 필요).
따라서 **엣지 디바이스가 서버의 공인(또는 VPN) IP의 `50051 (gRPC)` 포트로 먼저 연결을 맺고(Handshake), 이 거대한 튜브를 계속 열어두는(Keep-Alive) 방식**을 채택했습니다. (현재 작성된 `edge_manager`의 방식입니다.)

*   **[1단계]** `Edge` 전원 인가 시, `test_dummy_edge.py` 스크립트가 실행되어 `server_ip:50051` 로 gRPC `StreamDataAndControl` 호출.
*   **[2단계]** `Server`는 해당 커넥션을 절대 끊지 않음. (HTTP/2의 멀티플렉싱).
*   **[3단계]** `Server` 내부의 `camera_manager`에서 엣지의 카메라를 제어하고 싶을 때, 별도의 포트로 쏘지 않고 **열려있는 기존 gRPC 터널을 통해 응답(Yield)으로명령을 하달**.
*   **[결과]** 엣지는 방화벽에 포트를 뚫어둘 필요 없이 아웃바운드 인터넷 연결만 가능하면 양방향 제어가 가능합니다.

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
1. 캘리브레이션이 완료되면, 엣지 디바이스는 가벼운 특징값(Feature Vector) 패킷을 초당 N프레임 단위로 무한 송신(Yield)합니다.
2. 서버의 `dl_worker`는 GPU를 사용하여 수신된 패킷에서 3D 뼈대(Pose), 동적 모션, Bounding Box 등을 다듬어 추출해냅니다.
3. 무거운 연산이 끝나는 즉시, 경량화된 JSON 형태 `{"persons": [...], "timestamp": ...}` 로 환산되어 Redis Pub/Sub에 올라갑니다.

### 📺 Step 4. 디지털 트윈 동기화 구현 (Digital Twin Rendezvous)
1. 교내 통합 관제실(또는 프론트엔드 URL `http://<서버IP>:8005/`)에 접속하면, SPA 방식의 **웹 대시보드**가 구동됩니다. 사용자는 "3D 트윈 뷰어" 탭을 켜고, 내장된 iframe을 통해 서버 백그라운드에 구동 중인 Isaac Sim의 화면(Pixel Steaming, WebRTC 포트 8211)을 직접 눈으로 확인합니다.
2. `sim_backend`의 WebSocket 포트(`8004`)가 뿌려주는 3D 동적 메타데이터(`UpdateTransforms`)를 Isaac Sim의 내부 Extension `meta_sejong_script.py` 가 실시간으로 수신받아 3D 인간 아바타를 생성(Spawn)시킵니다.
3. 웹페이지에 접속한 사용자가 마우스 클릭이나 래그를 하면(WebRTC Client JavaScript 작동), 이 제어 인풋값이 실시간으로 백그라운드의 Isaac Sim 뷰포트에 전송되어 서버 엔진의 실제 관찰 시점이 회전하고 이동합니다.
