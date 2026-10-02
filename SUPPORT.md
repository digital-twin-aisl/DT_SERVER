# 지원

## 먼저 확인할 문서

1. [활용 가이드](docs/guide/README.md), 특히 [문제 해결](docs/guide/08-troubleshooting.md)
2. 이미 등록된 GitHub Issues (닫힌 이슈 포함)

## 질문·버그·제안

GitHub Issues에서 알맞은 양식을 골라 주세요.

| 양식 | 용도 |
| --- | --- |
| 버그 보고 | 문서대로 했는데 동작이 다를 때 |
| 사용 질문 | 설치·구성·운영 방법 문의 |
| 기능 제안 | 새 기능이나 개선 아이디어 |

버그 보고에는 다음을 포함하면 빠르게 도와드릴 수 있습니다.

- DT_SERVER 버전(태그 또는 커밋), `dt-common` 버전
- 호스트 종류(서버 GPU/드라이버/CUDA, Jetson JetPack 버전), Python 버전
- 실행한 명령과 `region status --json` 출력
- `data/manager/runs/<run_id>/`의 로그 끝부분

**올리기 전에 RTSP 주소, 계정, IP, 사람이 보이는 이미지를 지워 주세요.**

## 응답과 업데이트

- 이슈에는 보통 영업일 기준 5일 안에 라벨을 붙이고 1차 답변을 드립니다.
- 릴리스는 [CHANGELOG.md](CHANGELOG.md)와 GitHub Releases에 공지합니다. 저장소를 Watch → Releases로
  구독하면 업데이트 알림을 받을 수 있습니다.
- 보안 문제는 [SECURITY.md](SECURITY.md)의 비공개 채널을 사용해 주세요.
