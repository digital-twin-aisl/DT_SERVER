# 기여 가이드

intelligent-synchronization에 관심을 가져 주셔서 감사합니다. 버그 보고, 문서 개선, 코드 기여 모두 환영합니다.

## 시작하기 전에

- 사용 중 질문은 [SUPPORT.md](SUPPORT.md)의 절차를 따릅니다.
- 보안 문제나 자격증명 노출은 공개 이슈 대신 [SECURITY.md](SECURITY.md)로 알려 주세요.
- 큰 변경(새 기능, 메시지 계약 변경)은 먼저 이슈로 방향을 논의해 주세요.

## 개발 환경

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt          # CUDA torch는 먼저 설치
pip install -r apps/edge_manager/requirements.txt -r apps/frontend_api/requirements.txt
npm ci --prefix apps/frontend_api/web
```

## 테스트

```bash
python -m pytest apps/server_worker/tests apps/frontend_api/tests \
  apps/calibration_worker/tests apps/deployments/rootnet_v2/tests examples
npm test --prefix apps/frontend_api/web
ruff check apps packages examples tools
reuse lint
```

테스트는 `unittest` 형식이며 각 파일이 저장소 루트와 `packages/dt_common/src`를 `sys.path`에
추가하므로 설치 없이도 실행됩니다. GPU, 카메라, router가 필요한 검사는 해당 자원이 없으면 건너뛰도록
작성합니다.

## 코드 규칙

- 모든 호스트가 공유하는 계약(`packages/dt_common`)은 CPU 전용입니다. 모델 코드와 장치 정책을 넣지 않습니다.
- Zenoh 키, ZNH2 코덱, `SceneOutput`, 단위·좌표계를 바꾸는 변경은 서버·엣지·뷰어를 함께 수정하고
  [데이터 계약](docs/guide/07-contracts.md)과 CHANGELOG의 호환성 항목을 갱신합니다.
- 호환성 shim(`apps/server_worker/{scene_state,root_tracks,...}.py` 등)이 아니라 실제 모듈을 수정합니다.
- 주변 코드의 스타일과 주석 밀도를 따릅니다. 사용자 문서는 한국어로 작성합니다.

## 라이선스와 파일 헤더

기여한 코드는 프로젝트 라이선스(LGPL-2.1-or-later)로 배포되는 데 동의한 것으로 간주합니다.
새 소스 파일 맨 위에 다음 헤더를 둡니다.

```python
# SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
# SPDX-License-Identifier: LGPL-2.1-or-later
```

다른 프로젝트의 코드를 가져올 때는 원 저작권 고지와 라이선스를 유지하고, LGPL-2.1-or-later와 호환되는지
확인합니다([라이선스 준수](docs/guide/09-license-compliance.md#기여자에게)). 비상업·AGPL·GPL-only 코드는
받을 수 없습니다.

## 절대 커밋하지 말 것

- RTSP 주소, 카메라 계정, 토큰 (`cameras.local.yaml`, `edge.local.json`, `.env`)
- 사람이 찍힌 영상·이미지, 녹화된 `scenes.jsonl`
- 모델 가중치(`*.pt`, `*.pth*`, `*.engine`)

`.gitignore`가 대부분을 막지만, PR 전에 `git diff --cached --stat`으로 한 번 더 확인해 주세요.
CI는 민감정보 검사(`python tools/scan_sensitive.py .`)를 실행합니다.

## Pull Request

1. `main`에서 브랜치를 만듭니다.
2. 테스트와 `reuse lint`를 통과시킵니다.
3. PR 템플릿의 체크리스트를 채웁니다. 사용자에게 보이는 변경은 `CHANGELOG.md`의 `Unreleased`에 추가합니다.
