# Changelog

이 프로젝트의 주요 변경을 기록합니다. 형식은 [Keep a Changelog](https://keepachangelog.com/ko/1.1.0/)를,
버전은 [Semantic Versioning](https://semver.org/lang/ko/)을 따릅니다. 엔진 버전은 `packages/dt_common`의
`dt-common` 버전과 같습니다.

**호환성 표기**: 서버·엣지·뷰어를 함께 업그레이드해야 하는 변경은 `[계약]`으로 표시합니다
(Zenoh 키, ZNH2 코덱, `SceneOutput`, 배포 manifest 스키마).

## [Unreleased]

## [0.3.0] - 2026-10-02

첫 오픈소스 공개 버전입니다.

### Added
- 구역(region) 단위 운영: `edge_manager`의 `ControlPlane`, REST API, CLI, 웹 GUI
- 엣지 상시 agent(명령/ACK/lease)와 관리형 엣지 추론
- 서버 파이프라인: 입력 검증 → 전역 ID/root 연결 → 0.5초 Rank/LOD → LOD-2 3D 자세 → TTL 장면 병합 → 지면 높이 필터
- `SceneOutput` v1 JSONL 기록·다운로드·재생, Three.js 브라우저 뷰어, Isaac Sim 구독·Scene Player
- 카메라 보정: ArUco 마커 트리 + VGGT-Omega(선택), 수동 보정 편집기
- 합성 장면 생성기 `examples/synthetic_region` (GPU·카메라 없이 체험)
- 활용 가이드 `docs/guide/`, LGPL-2.1-or-later 라이선스, REUSE 준수, CI

### Changed
- `[계약]` 엣지-서버 추론 패킷을 ZNH2로 고정 (ZNH1/pickle 거부)
- Ultralytics YOLO(AGPL-3.0)를 엣지 ReID 전용 선택 의존성으로 분리 (`--no-reid`로 미설치 운영 가능)
- 서버 3D 자세 설정을 런타임 키만 받는 최소 설정으로 정리
- 학습 가중치와 샘플 영상은 저장소에서 제외 (`DT_POSENET_URL`로 지정)

### Removed
- gRPC → Redis → dl_worker → sim_backend 구 구조
