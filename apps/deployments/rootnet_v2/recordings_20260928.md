# 개선된 RootNet-B + PoseNet-B의 실제 영상 출력 녹화

## 완료 파일

2026-09-28 실제 영상 추론과 JSONL 검사 완료. 아래 파일을 Isaac Scene Player나
frontend_api의 JSONL 파일 재생에서 열면 된다. 기존 `rootnet_B_v2_001.jsonl` 등은 보존했다.

| 데이터셋 | JSONL 파일 | 전체 프레임 | 영상 분량 | pose 포함 프레임 |
|---|---|---:|---:|---:|
| 0812_1 | [rootnet_B_posenet_B_v2_0812_1.jsonl](../../server_worker/data/recordings/rootnet_B_posenet_B_v2_0812_1.jsonl) | 1,406 | 46.91초 | 1,323 |
| 0812_2 | [rootnet_B_posenet_B_v2_0812_2.jsonl](../../server_worker/data/recordings/rootnet_B_posenet_B_v2_0812_2.jsonl) | 843 | 28.13초 | 679 |
| 0812_3 | [rootnet_B_posenet_B_v2_0812_3.jsonl](../../server_worker/data/recordings/rootnet_B_posenet_B_v2_0812_3.jsonl) | 1,334 | 44.51초 | 1,334 |

모든 프레임에서 schema, timestamp 순서/간격, 두 엣지 timestamp 차이 0,
root/15관절 pose 유한값 검사를 통과했다. 실제 디코딩된 두 엣지 개수는 각 데이터셋에서
같았으므로 추가로 제외한 엣지 tail은 0이다. 사람을 검출하지 못한 프레임도 빈 장면으로
저장했다. 처리 성공은 포즈/검출의 실제 정확도 보장을 뜻하지 않는다.

person/pose 관측 수(프레임마다 합산)는 각각 4,754 / 2,116 / 7,087이다.
서로 다른 track ID 수는 183 / 70 / 122이며, 이는 고유 인원수가 아니다.
Re-ID가 꺼진 root tracking에서 ID 변경·중복·오검출이 생길 수 있다.

JSONL SHA-256:

```text
0812_1 708fff7ec167c4e15ddf9c52461f138943445756d8ab7692f91d258aacceadc1
0812_2 63703942e5c35a0d6345504c002cdbedfb05497053029846a4e8d51e3d8f2e69
0812_3 f484956c24ffa5d90c0dc9672ef4b3ed39bfc28f1b7dc6d0f2aa5e2891ecd3f4
```

결합 가중치 SHA-256:
`9d70a95c03288e798605260e477dfe87020d73dcbc78fdd2692688621ad9bfc0`.

## 실행 구성

- 입력: `apps/edge_client/data/data_0812_{1,2,3}_edge_{1,2}`.
- RootNet: `SelfPose3d/output_root_robustness/pilot_v2/B_export.pth.tar`.
- PoseNet: `SelfPose3d/output_pose_synthetic/full_20260928/B_export.pth.tar`
  (validation으로 선택한 9,000-step PoseNet).
- 두 export의 Backbone이 bit-identical인지 확인한 뒤, RootNet은 첫 모델에서,
  PoseNet은 두 번째 모델에서 가져온 전체 체크포인트를 생성한다.
- 카메라: 학습에 사용한 `calibration.from_cameras_v2.json` 숫자 스냅샷.
  현재 기존 deployment의 August calibration 및 POC focus 파일은 수정하지 않는다.
- 두 엣지 모두 기존 PyTorch inference를 실행한다. TensorRT/Re-ID/네트워크 송출은 끈다.
  서버는 root tracking fallback과 `lod_policy=all`로 PoseNet을 실행한다.
- 엣지 ZNH2 패킷을 먼저 보존하고, 서버의 `ReplayDataLoader`를 `strict/dataset` 모드로
  재생한다. 실제 서버의 `SceneOutputRecorder`가 최종 JSONL을 저장한다.
- 원본 영상, 기존 녹화 파일, 기존 실행 프로파일, 운영 프로세스는 변경하지 않는다.

## 영상 끝과 동기화

OpenCV가 MKV에서 읽는 `CAP_PROP_FRAME_COUNT`는 실제 디코딩 프레임 수와 1프레임
차이가 있었다. 최종 실행은 그 예상값으로 자르지 않고 실제 EOF까지 추론한다.
각 엣지 내부는 4개 카메라 중 가장 짧은 영상까지, 서버 출력은 두 엣지에서 모두
존재하는 공통 프레임까지 처리한다. 한 엣지에만 남는 tail이 있다면 원본 패킷은
그대로 보존하고 `alignment.json`에 제외 개수를 기록한다.

서버 replay용 패킷은 원본 패킷의 hard link여서 내용 변형/중복 디스크 복사가 없다.
메타데이터와 실제 개수가 크게 다르거나, 중간 프레임이 빠지거나, replay 동기화가
실패하면 완료로 기록하지 않는다. 빈 장면도 JSONL에서 생략하지 않는다.

동기화는 **프레임 인덱스와 영상 FPS 기준**이다. 카메라의 물리적 촬영 시차를
새로 추정/보정한 것이 아니다. 영상 동봉 캘리브레이션 대신 v2를 사용하는 시험이며,
실제 공간 정확도나 학습 전후 정확도 개선을 인증하지 않는다. ID는 Re-ID가 아니라
root 위치 추적 결과이므로 동일한 사람이 여러 ID를 가질 수 있다.

## 재실행

저장소 루트에서, 출력 이름과 run 디렉터리가 아직 없는 값으로 실행한다.

```bash
python apps/deployments/rootnet_v2/record_datasets.py \
  --run-dir apps/server_worker/data/recordings/runs/root_B_pose_B_v2_NEW \
  --output-prefix rootnet_B_posenet_B_v2_NEW
```

전체 3개 dataset 대신 하나만 실행하려면 `--datasets 0812_2`, 사전 검증은
`--max-frames 3`을 추가한다. 기존 출력/체크포인트/로그 파일은 덮어쓰지 않는다.
기존 router나 Isaac Sim을 실행할 필요가 없다. 생성한 SceneOutput JSONL은
Isaac Scene Player 또는 frontend_api의 JSONL 파일 재생에서 열 수 있다.

## 실행 증거

최종 run: `apps/server_worker/data/recordings/runs/root_B_pose_B_v2_20260928_run2/`.

- `plan.json`: 원본 가중치 및 결합 가중치 SHA-256, 영상 메타데이터, 캘리브레이션,
  실행 정책. 결합 가중치는 `root_B_pose_B.pth.tar`.
- `deployment.json`, `edge.focus.yaml`, `server.focus.yaml`: 격리된 실제 실행 설정.
- dataset별 `edge_1.log`, `edge_2.log`, `server.log`: 실제 추론 로그.
- dataset별 `packets/`: 실제 영상에서 계산한 Backbone 히트맵 및 RootNet 출력.
- dataset별 `alignment.json`: 실제 디코딩 개수와 공통 구간.
- dataset별 `metrics/`: 서버 추론 및 입력 버림/거부 통계.
- dataset별 `result.json`: JSONL 해시, 프레임 수, 사람/pose 관측 수, timestamp 검사.
- `completed.json`: 세 파일의 검사를 모두 마친 뒤 생성.

첫 `_20260928` run은 MKV 메타데이터 개수 검사에서 중단된 edge_1 패킷만 보존한
실행이다. `_smoke_20260928`은 첫 3프레임의 저장 기능 검사다. 최종 결과와 구분한다.

가중치 결합, Backbone 불일치 차단, 기존 파일 덮어쓰기 방지, 공통 구간/원본 tail
보존, 빈 장면 유지, timestamp 누락 검출, pose 좌표 검사의 테스트 5개가 통과했다.

```bash
python -m unittest discover \
  -s apps/deployments/rootnet_v2/tests -p test_record_datasets.py -v
```
