# ROS 2 기반 차세대 디지털 트윈 아키텍처 기획안 (V2)

본 문서는 기존의 커스텀 하이브리드 통신망(WebRTC+gRPC)과 커스텀 웹소켓 백엔드를 걷어내고, **로보틱스 생태계의 글로벌 표준인 ROS 2 기반으로 전체 시스템을 전면 재구성(Refactoring)** 하는 아키텍처 V2 기획서입니다.

---

## 1. 패러다임 전환 (Paradigm Shift)

기존 시스템의 방대한 커스텀 통신/렌더링 코드를 오픈소스 생태계의 검증된 모듈로 대체하여 유지보수성을 극대화합니다.

| 기존 아키텍처 (V1) | 🚀 ROS 2 기반 아키텍처 (V2) | 도입 효과 및 기대 이점 |
| :--- | :--- | :--- |
| 커스텀 gRPC + WebRTC + Redis | **ROS 2 + Zenoh (`rmw_zenoh`)** | 원격지 통신망(WAN) 라우팅을 Zenoh가 전담. ROS 2의 QoS(Reliable/Best Effort)를 그대로 사용하며, Nginx 포트 포워딩이나 TURN 서버 구축 불필요. |
| 커스텀 딥러닝 큐 (`dl_worker`) | **ROS 2 Inference Node** | Redis 인프라 걷어냄. `/camera/tensor` 토픽을 구독하여 추론 후 `/tf` (Transform) 토픽으로 발행. |
| 커스텀 `sim_backend` (WebSocket) | **Isaac Sim ROS 2 Bridge** | Isaac Sim에 내장된 OmniGraph ROS 2 익스텐션 활성화. 커스텀 통신 코드 없이 `/tf` 토픽을 받으면 아바타가 즉시 연동. |
| 커스텀 Web 3D 대시보드 (`frontend`) | **Foxglove Studio + Foxglove Bridge** | Three.js 커스텀 코딩 불필요. 현존 최고의 로봇 관제 UI로 카메라, 3D Pose, 텔레메트리 즉시 시각화. |

---

## 2. V2 아키텍처 노드 및 네트워크 토폴로지

전체 시스템은 중앙 메세지 브로커 없이 각 마이크로서비스가 독립적인 **ROS 2 노드(Node)** 로 동작합니다.

### 🌐 인프라: 통신 레이어
*   **Zenoh Router:** 엣지(Jetson)와 서버(Cloud) 간의 WAN 통신을 책임집니다. 엣지의 ROS 2 노드들은 로컬 네트워크처럼 통신하지만, 밑단에서는 `rmw_zenoh`가 이를 가로채 UDP/QUIC 프로토콜로 암호화하여 클라우드로 초저지연 라우팅합니다. 방화벽 및 NAT 트래버설을 자체 지원합니다.

### 💻 엣지 파트 (Jetson Orin)
1. **`camera_capture_node`**: RTSP 카메라 스트림을 읽어 ROS 2의 `sensor_msgs/Image` 로 퍼블리시.
2. **`edge_feature_node`**: 무거운 원본 영상 대신 특징점(Tensor)만 추출하여 커스텀 `TensorMsg`로 패키징 후 Best-Effort QoS로 퍼블리시.

### ☁️ 클라우드/서버 파트
1. **`inference_node`**: 서버 GPU를 독점. 엣지에서 넘어온 `TensorMsg`를 구독(Subscribe). 무거운 PyTorch 3D Pose 연산 후, 절대 좌표계 위치를 로보틱스 표준인 `tf2_msgs/TFMessage` 및 `geometry_msgs/PoseArray` 로 변환하여 퍼블리시.
2. **`foxglove_bridge_node`**: Nginx를 거쳐 들어오는 웹 클라이언트(관제실) 접속 요청을 처리. ROS 2 토픽들을 WebSocket(또는 WebRTC)으로 자동 변환하여 프론트엔드로 송출.

### 🎮 디지털 트윈 렌더링 파트
1. **Isaac Sim (ROS 2 Bridge Extension)**: 어떠한 커스텀 Python 네트워크 스크립트도 불필요합니다. OmniGraph 내에서 `ROS2 Subscriber` 노드를 드래그 앤 드롭으로 배치하고 `/tf` 토픽을 아바타 관절(Articulation)에 연결하면 자동으로 동기화됩니다.

---

## 3. 새로운 시운전 시나리오 (Tutorial Scenario V2)

커스텀 로직이 대폭 사라지며 시스템 기동이 극도로 단순해집니다.

### 🎥 Step 1. 전원 인가 및 디스커버리 (Zero-Config Network)
1. 서버 인프라(Zenoh Router, Foxglove, Inference Node)를 `docker compose up`으로 기동.
2. 현장의 Jetson 엣지에 전원이 들어오면, 내장된 `rmw_zenoh`가 클라우드 라우터 IP를 찾아 자동으로 연결 수립. 별도의 시그널링이나 커넥션 맺기 로직 불필요.

### 🏃 Step 2. 실시간 동적 추론 파이프라인
1. Jetson에서 `/edge/camera1/tensor` 토픽(Best Effort QoS)으로 10FPS 데이터가 쏟아짐.
2. 서버의 `inference_node` 콜백 함수가 이를 받아 즉각 추론 후, `/world_to_person_01` 이라는 표준 TF 토픽(Reliable QoS) 발행.

### 📺 Step 3. 관제 및 시각화 (No Code Rendering)
1. 관리자는 웹 브라우저로 `https://<서버IP>/foxglove` 접속. Foxglove Studio UI에서 클릭 몇 번으로 3D Scene에 `/tf` 좌표계를 올리면 현장 상황 모니터링 즉시 완료.
2. 렌더링 서버에서는 Isaac Sim이 켜져 있으며, ROS Bridge가 `/tf`를 읽어들여 언리얼/유니티 수준의 고품질 픽셀 렌더링 화면을 송출.

---

## 4. 검증 및 테스트 (향후 Action Plan)

기존 코드를 모두 버리는 것이 아니라 **통신과 데이터 규격만 ROS 2 생태계로 래핑(Wrapping)** 하는 작업입니다.
*   **우선 과제 1:** 기존 FastAPI 기반의 `dl_worker` 추론 코드를 `rclpy` (ROS 2 Python) 노드 콜백 함수로 마이그레이션.
*   **우선 과제 2:** Isaac Sim 내부의 커스텀 웹소켓 확장 프로그램을 비활성화하고, 기본 탑재된 `omni.isaac.ros2_bridge` 활성화 후 TF 트리 구성 연동 테스트.
*   **우선 과제 3:** LAN 환경에서 ROS 2 동작 확인 후, `rmw_fastrtps`를 `rmw_zenoh`로 교체하여 LTE 망 테스트 진행.
