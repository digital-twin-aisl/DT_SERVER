# intelligent-synchronization 활용 가이드

디지털 트윈 동기화 엔진을 처음 설치하는 사람부터 다른 시스템과 연동하는 개발자까지를
대상으로 합니다. 위에서부터 순서대로 읽으면 실제 현장 배포까지 진행할 수 있습니다.

| # | 문서 | 이런 분께 | 필요 장비 |
| --- | --- | --- | --- |
| 1 | [빠른 시작](01-quickstart.md) | 무엇을 하는 엔진인지 먼저 보고 싶다 | 노트북 (GPU 불필요) |
| 2 | [서버 설치](02-install-server.md) | GPU 서버에 관리 서비스와 추론 환경을 준비한다 | NVIDIA GPU 서버 |
| 3 | [엣지 설치](03-install-edge.md) | Jetson에 카메라를 연결하고 서버에 등록한다 | Jetson Orin + RTSP 카메라 |
| 4 | [Isaac Sim 연동](04-isaac-sim.md) | 결과를 Isaac Sim 스테이지에 반영한다 | RTX GPU 호스트 |
| 5 | [새 구역 구성](05-new-region.md) | 우리 현장(건물·카메라)에 맞게 구성한다 | 현장 카메라, 마커 |
| 6 | [운영](06-operations.md) | 매일 시작·중지하고 상태·기록을 관리한다 | - |
| 7 | [데이터 계약](07-contracts.md) | 다른 시스템이 장면을 구독하거나 입력을 보낸다 | - |
| 8 | [문제 해결](08-troubleshooting.md) | 동작하지 않을 때 | - |
| 9 | [라이선스 준수](09-license-compliance.md) | 재배포·상용 사용 전 의무를 확인한다 | - |

## 용어

| 용어 | 의미 |
| --- | --- |
| 구역 (region) | 운영 단위. 한 개 이상의 엣지와 서버 프로파일, 장면 토픽을 묶습니다. 예: `center-b1-corridor` |
| 엣지 (edge) | 카메라 4대 내외를 담당하는 Jetson 장비. 2D 추론 결과를 ZNH2 패킷으로 보냅니다. |
| 배포 manifest | 구역의 카메라 순서, 작업 영역(AOI), 보정·지면 파일 참조를 정의한 JSON |
| SceneOutput | 서버가 발행하는 장면 메시지 (schema v1, USD world, Z-up, mm) |
| LOD | 사람별 처리 수준. 0=위치, 1=위치+ID, 2=15관절 3D 자세 |
| run | 구역 한 번의 실행. `data/manager/runs/<run_id>/`에 manifest·로그·기록이 남습니다. |

## 지원 범위

- 검증된 조합: Ubuntu 22.04 + Python 3.10/3.11 + CUDA PyTorch(서버), Jetson Orin + JetPack 6(엣지),
  Zenoh 1.9.0, Isaac Sim 4.2.0
- 모든 호스트는 같은 `dt-common` 버전을 사용해야 합니다. 업그레이드 시 서버와 엣지를 함께 올립니다.
- 질문과 버그는 [SUPPORT.md](../../SUPPORT.md)의 절차를 따릅니다.
