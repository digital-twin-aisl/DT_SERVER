# 9. 라이선스 준수 가이드

> 이 문서는 법률 자문이 아닙니다. 상용 배포나 제품 탑재 전에는 소속 기관의 법무 검토를 받으세요.

## intelligent-synchronization의 라이선스

intelligent-synchronization는 **GNU Lesser General Public License v2.1 이상(LGPL-2.1-or-later)** 으로 배포됩니다.
저작권자는 한국전자통신연구원(ETRI)과 세종대학교이고, 전문은 [LICENSE](../../LICENSE)에 있으며, 각 소스 파일에는 SPDX 헤더가 있습니다. 저장소는
[REUSE](https://reuse.software) 규격을 따르므로 `reuse lint`로 모든 파일의 라이선스를 확인할 수 있습니다.

VoxelPose(Microsoft, MIT)에서 유래한 모델 코드(`apps/*/src/{pose,root}/{models,core/proposal.py,utils/cameras.py}`)는
원 저작권 고지와 함께 **MIT** 라이선스를 유지합니다. MIT는 LGPL과 함께 배포할 수 있습니다.

## 사용·수정·재배포할 때의 의무 (요약)

| 하려는 일 | 해야 할 일 |
| --- | --- |
| 내부에서 그대로 사용 | 별도 의무 없음 |
| 수정 없이 재배포 (소스 또는 바이너리) | LICENSE 사본과 저작권 고지 유지, 소스 제공(또는 제공 약속) |
| 수정 후 재배포 | 위 사항 + 수정 사실과 날짜 표시, **수정한 intelligent-synchronization 부분의 소스를 LGPL로 공개** |
| 자기 프로그램이 intelligent-synchronization를 라이브러리로 사용 (예: `dt-common` import, Zenoh로 장면 구독) | 자기 프로그램은 다른 라이선스 가능. 단, 사용자가 intelligent-synchronization 부분을 교체·수정할 수 있어야 함(동적 링크/별도 패키지 형태 권장) |
| 컨테이너·장비 이미지로 배포 | 이미지 안의 intelligent-synchronization 소스(또는 정확한 소스 위치)와 LICENSE를 함께 제공 |

네트워크로 장면을 구독하는 것만으로는 LGPL 의무가 생기지 않습니다. `SceneOutput`을 소비하는
별도 프로그램은 자유롭게 라이선스를 정할 수 있습니다.

## 포함되지 않는 선택 구성요소

다음 구성요소는 이 저장소에 **포함되어 있지 않으며**(서브모듈은 원 저장소의 위치만 가리킵니다),
설치 여부는 사용자가 결정합니다. 설치하면 해당 라이선스가 그 구성요소와 결합된 배포물에 적용됩니다.

| 구성요소 | 라이선스 | 상용 사용 | intelligent-synchronization에서의 역할 | 대안 |
| --- | --- | --- | --- | --- |
| [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) + `yolo11n-pose.pt` | AGPL-3.0 (또는 유료 Enterprise) | 결합 배포물 전체가 AGPL 의무를 질 수 있음 | 엣지 ReID의 사람 검출 | `--no-reid` |
| [VGGT-Omega](https://github.com/facebookresearch/vggt-omega) (서브모듈) + 체크포인트 | FAIR Noncommercial Research License | **불가** (비상업 연구만) | 자동 카메라 보정 | 수동 보정 편집기 |
| [FastReID](https://github.com/JDAI-CV/fast-reid) (서브모듈) + Market1501 가중치 | Apache-2.0 | 가능 (고지 유지) | ReID 특징 추출 | `--no-reid` |
| 개발팀의 PoseNet 가중치 (`POC_posenet.pth.tar`) | CC BY-NC-SA 4.0 (SelfPose3d 사전학습 모델을 미세조정한 파생물, CMU Panoptic 학습 데이터 조건 포함) | **불가** (비상업 연구만) | 2D heatmap·3D 자세 | 라이선스가 허용된 데이터·초기값으로 직접 학습한 가중치 |

- `apps/edge_client/install_edge.sh`는 `DT_WITH_ULTRALYTICS=0`(기본)·`DT_WITH_VGGT`로 설치 여부를 고릅니다.
- 분산 보정용 `apps/calibration_worker/patches/vggt-omega-patch-tokens.patch`는 VGGT-Omega 코드의
  수정본이므로 **FAIR 비상업 라이선스**를 따르는 별도 파일입니다(REUSE.toml에 명시).
- 서버 `requirements.txt`에는 Ultralytics가 없습니다. 엣지의 Ultralytics는
  `apps/edge_client/requirements-reid-ultralytics.txt`로 분리되어 있습니다.
- Apache-2.0은 LGPL-2.1"만"과는 호환되지 않지만, intelligent-synchronization는 "or later"이므로 LGPL-3.0 조건으로
  함께 배포할 수 있습니다.

## 데이터와 개인정보

- 저장소에는 사람이 촬영된 영상·이미지가 **포함되지 않습니다.** 예제는 계산으로 만든 합성 장면입니다.
- 배포 예시(`apps/deployments/`)의 지도·보정 값은 카메라 배치 정보입니다. 자기 현장의 값을 공개할지는
  보안 정책에 따라 판단하세요.
- RTSP 주소·계정은 엣지 로컬 파일(`cameras.local.yaml`, `edge.local.json`)에만 두고 커밋하지 않습니다.
- 운영 중 기록되는 `SceneOutput`에는 사람 ID·위치·자세가 들어갑니다. 고지·동의·보관 기간은 운영자가
  현지 법령(예: 개인정보 보호법)에 맞게 정해야 합니다.

## 기여자에게

기여한 코드는 LGPL-2.1-or-later로 배포됩니다([CONTRIBUTING.md](../../CONTRIBUTING.md)). 다른 프로젝트의
코드를 가져올 때는 원 라이선스가 LGPL-2.1-or-later와 호환되는지 확인하고(MIT, BSD, Apache-2.0 가능,
GPL-only·AGPL·비상업 라이선스 불가), 원 저작권 고지와 SPDX 헤더를 유지하세요.
