# 실시간 동적 엣지-서버 분산 컴퓨팅 디지털트윈 백엔드 시스템

## 1. 프로젝트 개요 (Overview)

본 프로젝트는 원격지에 위치한 Jetson Orin 기반의 엣지 디바이스 군(Fleet)이 수집한 실시간 비디오(RTSP) 데이터 및 딥러닝 특징값(Feature)을 중앙 서버로 고속 전송하여, 무거운 딥러닝 추론을 거친 뒤 Isaac Sim 기반의 디지털 트윈(Digital Twin) 환경과 동기화하는 **실시간 3D 렌더링/분석 백엔드 인프라**입니다.

- **대상 공간**: 교내 야외 복도 환경 (실제 환경과 디지털 트윈 환경 간의 포즈, 동적 모션 동기화)
- **주요 워크플로우**: `Edge (Data Extraction)` $\rightarrow$ `Server (Deep Learning)` $\rightarrow$ `Isaac Sim (3D Rendering)` $\rightarrow$ `Frontend (GUI 제어)`

---

## 2. 아키텍처 및 모듈 구성 (Microservices Architecture)

빠른 장애 복구 및 역할 분담을 위해 5개의 독립된 마이크로서비스 컨테이너와 2개의 기반 인프라(DB, Message Queue)로 설계되었습니다.

1. **`edge_manager` (포트 : 8001 / gRPC 50051)**
   - **역할:** 다수의 엣지 디바이스와 양방향 `gRPC` 스트림 채널을 맺고 데이터를 수집/명령을 하달하는 게이트웨이.
   - **통신:** 딥러닝용 대용량 텐서(Tensor) 데이터의 직렬화(Protobuf) 고속 처리.
2. **`dl_worker` (포트 : 8002)**
   - **역할:** 엣지로부터 받은 중간 결과물을 Redis Queue를 통해 수신한 뒤, PyTorch/TensorRT 기반의 무거운 추론 모델을 돌려 객체의 3D 위치(Pose, Bounding Box)를 연산.
   - **특징:** 도커 구동 시 GPU 리소스를 전면적으로 할당(`capabilities: [gpu]`) 받음.
3. **`camera_manager` (포트 : 8003)**
   - **역할:** 엣지에 연결된 카메라들의 ONVIF 메타데이터 관리, 캘리브레이션 파라미터(Intrinsic/Extrinsic) 연산 및 PostgreSQL DB 저장.
4. **`sim_backend` (포트 : 8004)**
   - **역할:** 딥러닝 추론이 끝난 3D 동적 데이터를 WebSocket을 통해 Isaac Sim 환경과 Frontend 3D 뷰어에 실시간(Real-time) 브로드캐스팅.
5. **`frontend_api` (포트 : 8005)**
   - **역할:** 사용자가 접속하여 시스템 전체의 상태(엣지 연결 상태, 카메라 조작 등)를 모니터링하고 제어 명령을 내릴 수 있는 통합 API 엔드포인트(BFF - Backend For Frontend).

### 2.1. 인프라 요소 (Data Layer)
- **`Redis (alpine)`**: 마이크로서비스 간 초저지연 비동기 통신 버퍼.
  - 대용량 비디오 데이터: `Redis Stream` (최신 프레임 유지, 유실 허용)
  - 제어 명령/메타데이터: `Redis Pub/Sub` (즉각적인 명령 하달)
- **`PostgreSQL (15-alpine)`**: 엣지 디바이스 및 카메라의 물리적 설정, 캘리브레이션 파라미터 영구 저장(Persistence).

---

## 3. 현재까지 진행 상황 (Progress)

- [x] **Phase 1: 인프라 셋업** - FastAPI 기반 5개 모듈 스캐폴딩 및 `docker-compose` 멀티 컨테이너 통합 구현.
- [x] **Phase 2: 엣지 통신 (gRPC)** - Protobuf(`edge_communication.proto`) 작성 및 Bi-directional (양방향) 고속 스트리밍 구현 (`edge_manager`).
- [x] **Phase 3: 비동기 큐 (Message Broker)** - 레이턴시 최소화를 위한 Redis 인메모리 컨테이너 연동 완료.
- [x] **Phase 4: DB 및 제어 레이어** - `camera_manager` 내 PostgreSQL 연동 및 캘리브레이션 명령 로직 개발, Redis Pub/Sub을 타고 엣지로 내려가는 Downlink 파이프라인 검증 완료.
- [x] **Phase 5: 데이터 파이프라인 (E2E Test)** - 더미 엣지 클라이언트 송신 $\rightarrow$ gRPC 수신 $\rightarrow$ 비동기 딥러닝 추론 대기열(`dl_worker`) $\rightarrow$ WebSocket (`sim_backend`) 브로드캐스트까지 완벽한 동기화 테스트 완료.
- [x] **Phase 6: 핵심 알고리즘 통합** - PyTorch(pose, reid, mesh 파이프라인) 실시간 추론 로직 병합 및 의존성 이식, Camera Manager 캘리브레이션 연동을 통한 절대 좌표계(World Coordinate) 변환 로직 완전 적용.
- [x] **Phase 7: Frontend API 개편 및 대시보드 연동** - FastAPI `frontend_api` 에 HTML/Three.js 기반 3D SPA 대시보드를 서빙하여, 웹 브라우저 상에서 디지털 트윈 현황(카메라, 마우스 제어 등) 실시간 연동.
- [x] **Phase 8: Isaac Sim 클라이언트 렌더러 연동** - `isaac_sim_client` 디렉토리 내 Python Extension 구성(WebSocket 구독) 및 Headless WebRTC 스트리밍 파이프라인. Isaac Sim의 픽셀 프레임 자체를 WebRTC (port 8211)로 직접 릴레이하여 웹 통합 게이트웨이에 임베딩하는 방식 입증 완료.
- [ ] **Phase 9: Production 고도화** - 실제 엣지 디바이스 하드웨어 프로파일링, 로드밸런싱 및 K8s 기반 MSA 배포, SMPL 메쉬 파라미터 세밀 조정.
