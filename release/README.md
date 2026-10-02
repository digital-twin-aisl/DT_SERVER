# 오픈소스 배포 절차 (dev 전용)

이 디렉터리는 개발 브랜치에만 있고 공개 배포본에는 포함되지 않습니다.

## 원칙

- **dev 브랜치가 원본**입니다. LICENSE, 가이드, CI, 예제를 포함한 모든 공개 자산은 dev에서 고칩니다.
- 공개 배포본은 `release/oss` 브랜치이며, dev의 커밋 하나를 `release/export.py`로 걸러 낸 결과입니다.
  dev와 **히스토리를 공유하지 않는 고아 브랜치**이므로, 과거 dev 커밋의 민감정보가 따라가지 않습니다.
- 배포본에서 직접 수정하지 않습니다. 외부 PR은 dev로 옮겨 반영한 뒤 다시 export합니다.

## 무엇이 빠지는가

1. `.gitignore` 대상 (로컬 데이터, 영상, 가중치, `*.local.*` 자격증명 파일): 애초에 커밋되지 않음
2. [oss-exclude.txt](oss-exclude.txt)의 경로 (dev 도구, 내부 실험 기록)
3. 커밋되지 않은 작업 트리 변경

export 전에 `tools/scan_sensitive.py`가 RTSP 자격증명, 사설 IP, 홈 경로, 이메일, Google Drive 링크,
영상·이미지·가중치, 5 MB 초과 파일을 검사하고, 하나라도 있으면 중단합니다. 과거에 유출된 비밀번호 같은
문자열은 **git에 넣지 말고** `data/local/release-denylist.txt`(git 제외)에 한 줄씩 적어 두면 함께 검사합니다.

## 업데이트 절차

```bash
# 0) dev에서 작업을 커밋하고 테스트·라이선스 검사를 통과시킨다
python -m pytest apps/server_worker/tests apps/frontend_api/tests apps/calibration_worker/tests \
  apps/deployments/rootnet_v2/tests examples
reuse lint && python tools/scan_sensitive.py .

# 1) 버전과 변경 기록
#    packages/dt_common/pyproject.toml 의 version 과 CHANGELOG.md 를 함께 올리고 커밋

# 2) export (기본 worktree: ../DT_SERVER-oss, 브랜치: release/oss)
python release/export.py               # 스테이징 후 변경 요약 출력
python release/export.py --commit --message "Release v0.3.1"

# 3) 태그와 공개
cd ../DT_SERVER-oss
git tag -a v0.3.1 -m "v0.3.1"
git push oss release/oss:main --follow-tags     # oss = 공개 저장소 remote
```

공개 저장소 remote는 한 번만 추가합니다: `git -C ../DT_SERVER-oss remote add oss <공개 저장소 URL>`.
GitHub Releases에 CHANGELOG의 해당 절을 붙여 공지합니다.

## 버전 정책

- `MAJOR.MINOR.PATCH`, `dt-common` 버전과 동일하게 유지합니다.
- `[계약]` 변경(Zenoh 키, ZNH2, `SceneOutput`, manifest 스키마)이 있으면 0.x 동안은 MINOR, 1.0 이후는 MAJOR를 올립니다.
- 버그 수정·문서만 바뀌면 PATCH를 올립니다.

## 공개 전 확인표

- [ ] `reuse lint` 통과, 새 파일에 SPDX 헤더
- [ ] `tools/scan_sensitive.py` 0건
- [ ] 선택 구성요소(Ultralytics, VGGT-Omega, 가중치)가 배포본에 들어가지 않았음
- [ ] CHANGELOG, 가이드의 버전·명령이 실제와 일치
- [ ] 배포 worktree에서 새 가상환경으로 빠른 시작(`docs/guide/01-quickstart.md`)이 동작
