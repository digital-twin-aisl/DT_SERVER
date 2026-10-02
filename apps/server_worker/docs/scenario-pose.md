# CSV root 고정 + 실제 영상 PoseNet 추론

`tools/scenario_pose.py`는 `scenario (1)/scenario_N.csv`를 `0812_N` 실제 영상에
대응시킨다. RootNet 검출과 Re-ID/위치 기반 ID 재할당을 건너뛰고, CSV root 주변에서
**실제 서버 PoseRegressionNet**만 추론한다. 기존 `scenario_skeleton.py`의 합성
보행이나 이전 출력 포즈를 복사하는 방식이 아니다.

## 2026-09-28 완료 결과

| 입력 대응 | JSONL 파일 | 전체 프레임 | 인물/pose 포함 프레임 | pose 관측 수 |
|---|---|---:|---:|---:|
| scenario_1 → 0812_1 | [scenario_root_posenet_B_v2_0812_1.jsonl](../data/recordings/scenario_root_posenet_B_v2_0812_1.jsonl) | 1,406 | 1,056 | 4,496 |
| scenario_2 → 0812_2 | [scenario_root_posenet_B_v2_0812_2.jsonl](../data/recordings/scenario_root_posenet_B_v2_0812_2.jsonl) | 843 | 630 | 1,986 |
| scenario_3 → 0812_3 | [scenario_root_posenet_B_v2_0812_3.jsonl](../data/recordings/scenario_root_posenet_B_v2_0812_3.jsonl) | 1,334 | 1,274 | 5,943 |

실제 PoseNet 추론에 약 160초가 소요됐다(히트맵 생성은 이전 실행의 캐시 사용).
세 파일 모두 CSV에 활성 상태인 인물과 출력 ID 집합이 매 프레임 일치했고,
ID는 0~4만 존재했다. 총 12,425개 인물 관측 모두에서 pose를 계산했다.
출력 재조회로 타임스탬프 연속성, 15관절 유한값, CSV selection 열 보존, SHA-256을 확인했다.
보간된 CSV XY 및 Ground+900mm와 출력 root의 최대 차이, 출력 골반과 root의 최대 차이는
모두 0mm다. **이는 고정 제약 검증이며 3D 자세 정확도 지표가 아니다.**

원시 PoseNet 골반을 지정 root로 옮긴 거리:

| 시나리오 | 중앙값 | P95 | 최대 |
|---|---:|---:|---:|
| 1 | 209.7mm | 485.3mm | 673.2mm |
| 2 | 213.5mm | 482.5mm | 658.1mm |
| 3 | 205.1mm | 435.6mm | 660.1mm |

상당한 차이가 있으므로 root 고정으로 캘리브레이션/높이/시간 정합 문제가 해결됐다고
해석하지 않는다. 원시 관절과 보정량은 각 시나리오의 진단 파일에 보존되어 있다.
신규 4개 + 기존 시나리오 7개, 총 11개 테스트가 통과했다.

출력 SHA-256:

```text
0812_1 d066b4aade593e8aa48130a5e50d72bf0bd8fa55a7c04cf8c8057c26d3c4aec4
0812_2 2e85f502e19f11201b4c3b287890d89990c3977eec500c0c00180e5d429c6028
0812_3 a540265f4d9dbe2fc6781842cc692b3d027ba0d66ad6cd3585892309ea0509ba
```

## root와 시간의 의미

- CSV의 `global_id`와 X/Y(mm)는 고정한다. CSV 행 사이에서는 X/Y를 선형 보간한다.
- CSV에 Z가 없으므로 지면 캐시 높이 + 900mm를 골반 높이로 가정한다.
  `--root-height-mm`으로 바꿀 수 있다. 측정한 실제 사람 키/골반 높이가 아니다.
- 시간은 `영상 시각 = CSV 시각 + time-offset`이며 기본 오프셋은 0초다.
  각 인물의 관측 범위 밖으로 외삽하지 않고, 1.01초보다 큰 CSV 공백도 보간하지 않는다.
- 영상은 처음부터 끝까지 기록한다. CSV에 인물이 없는 시각에는 빈 장면을 출력한다.
  CSV 시작 시각은 scenario_1=8.258초, scenario_2=5.838667초, scenario_3=0.617033초다.
  사람이 이보다 일찍 영상에 보여도 새로 검출해서 CSV에 없는 인물을 추가하지 않는다.
- `zone_selected`/`rank_selected`는 메타데이터로 보존하되 이번 기본 실행에서는
  CSV에 있는 모든 인물의 포즈를 계산한다. 해당 열로 표시/추론 대상을 제한하지 않는다.

## PoseNet 입력과 출력

- 앞서 `record_datasets.py`가 실제 RGB 영상으로 생성한 uint8 히트맵 패킷을 재사용한다.
  기존 패킷의 **`roots` 필드는 사용하지 않는다.** Backbone도 다시 실행할 필요가 없다.
- PoseNet-B의 validation 선택 모델(9,000 step)을 사용한다. Backbone/RootNet
  모듈은 생성하지 않으며 `pose_net.*`만 실제 서버 PoseRegressionNet에 strict-load한다.
- 카메라는 동일한 v2 캘리브레이션이다. 패킷의 카메라 순서, calibration digest,
  프레임 인덱스/시각, 히트맵 shape/dtype을 검증하고 uint8/255로 복원한다.
- root가 2개 이상 카메라에서 관측 가능한 엣지 중 AOI 포함 여부, 관측 카메라 수,
  직전 엣지 소유권 순서로 하나를 선택한다. 두 엣지의 8개 카메라를 한 모델로 합치지 않는다.
- 관측 가능한 엣지가 없거나 투영된 로컬 히트맵 큐브가 완전히 비어 있으면
  CSV root와 ID는 유지하되 `pose=null`로 기록한다. 합성 포즈로 빈 값을 채우지 않는다.
- 추론 결과의 골반(관절 2)이 지정한 XYZ root와 정확히 일치하도록 모든 관절에
  같은 평행이동을 적용한다. 관절 상대 위치/골격 길이는 이 연산으로 바뀌지 않는다.
  이동 전 원시 관절과 이동량은 `raw_predictions.jsonl`에 전부 기록한다.
- root confidence=1은 **외부에서 지정한 제약**이라는 뜻이지 검출/포즈 신뢰도 100%가 아니다.
- 출력은 기존 SceneOutput JSONL이며, `runtime.scenario`와 `people[].scenario`에
  CSV/실영상/PoseNet/후처리 출처를 구분한다. `synthetic_pose=false`는 보행 합성을
  쓰지 않았다는 뜻으로, 모든 관절이 정확하거나 ground truth라는 의미가 아니다.

root를 강제로 맞추면 원래 PoseNet의 골반 오차가 뷰어에서 가려질 수 있다.
따라서 원시 골반과 지정 root의 차이를 진단 기록에 보존한다. 이 차이는 CSV의 시간·좌표
정합, Z 가정, 캘리브레이션, 모델 오류가 섞인 값이며 ground-truth MPJPE가 아니다.
평행이동이 영상 재투영 일치를 개선한다고 보장하지도 않는다.

## 실행

저장소 루트에서 실행한다. 모든 출력은 새 파일이어야 하며 기존 파일을 덮어쓰지 않는다.

```bash
# CPU로 CSV 범위와 카메라 관측 범위 먼저 확인 (출력 파일 생성 없음)
python apps/server_worker/tools/scenario_pose.py \
  --run-dir apps/server_worker/data/recordings/runs/scenario_root_pose_B_v2_NEW \
  --preflight-only

# 세 시나리오 전체, 실제 PoseNet 추론
python apps/server_worker/tools/scenario_pose.py \
  --run-dir apps/server_worker/data/recordings/runs/scenario_root_pose_B_v2_NEW \
  --output-prefix scenario_root_posenet_B_v2_NEW
```

다른 설정은 `--scenarios 1`, `--time-offset 0`, `--root-height-mm 900`,
`--source-run PATH`, `--checkpoint PATH`를 지정한다. 예를 들어 CSV 8.258초를
영상 0초에 맞추려면 `--time-offset -8.258`이다. 자동으로 이 정합을 추정하지 않는다.
짧은 실제 검사는 `--scenarios 1 --start-frame 300 --max-frames 4`를 사용한다.
FPS와 영상 프레임 수는 기존 영상 추론 run에서 읽어 실제 영상 시간축을 유지한다.

기본 히트맵 입력:
`apps/server_worker/data/recordings/runs/root_B_pose_B_v2_20260928_run2/`.

이번 전체 실행 기록:
`apps/server_worker/data/recordings/runs/scenario_root_pose_B_v2_20260928/`.

`plan.json`에 CSV/모델/캘리브레이션/코드 해시와 가정을 저장한다.
각 시나리오 폴더의 `raw_predictions.jsonl`, `result.json`에 원시 결과와 요약을 저장하고,
세 시나리오를 끝낸 뒤 `completed.json`을 생성한다. 운영 설정/서비스 및 기존 녹화는 변경하지 않는다.

## 테스트

```bash
python -m unittest discover \
  -s apps/server_worker/tests -p 'test_scenario*.py' -v
```

새 경로는 CSV ID 0/보간/관측 범위/높이 가정, root anchoring 및 상대 관절 불변,
엣지 선택, 저장된 RootNet 값의 독립성을 검사한다. 기존 합성 보행 경로의 테스트도 유지한다.

## 최악 뷰 제외 후 1회 재추론 (opt-in)

현재 CSV root 고정 오프라인 경로에 `--view-retry`를 추가했다. 기존 기본 실행과
라이브 서버/엣지 설정은 변경하지 않는다. 학습 가중치도 동일한 PoseNet-B다.

```bash
python apps/server_worker/tools/scenario_pose.py \
  --view-retry \
  --view-error-threshold 6 \
  --view-peak-confidence 0.1 \
  --view-min-joints 6 \
  --run-dir apps/server_worker/data/recordings/runs/scenario_root_pose_B_v2_viewdrop1_NEW \
  --output-prefix scenario_root_posenet_B_v2_viewdrop1_NEW
```

처리 단위는 **사람 × 프레임 × 선택된 엣지**다. 한 사람의 가림 때문에 같은 프레임의
다른 사람에게 필요한 카메라까지 제외하지 않는다.

1. 기존처럼 4개 히트맵으로 PoseNet을 실행하고 CSV root에 골반을 맞춘다.
2. 표시될 15관절을 각 카메라의 왜곡/영상 affine 변환을 거쳐 **원래 히트맵 좌표**로
   재투영한다. 원본 영상 픽셀과 히트맵 픽셀을 혼용하지 않는다.
3. 각 관절 채널에서 신뢰도 ≥ 0.1인 5×5 지역 최대점을 추출한다. 여러 사람이 있으므로
   전체 argmax 대신 재투영 지점에서 **가장 가까운 피크까지의 유클리드 거리**를 사용한다.
   카메라 뒤/화면 밖/히트맵 밖 관절과 피크 없는 관절은 제외한다.
4. 유효 관절이 6개 이상인 뷰만 거리 중앙값을 계산한다. 가장 큰 중앙값이
   **6 heatmap pixels 초과**일 때 해당 카메라 하나를 제외한다.
5. 제외 후 root가 보이는 카메라가 최소 2대 있어야 재추론한다. 그렇지 않으면 최초 결과를
   유지한다. 최악 뷰를 제외할 수 없다고 차순위 뷰를 대신 제외하지 않는다.
6. 해당 카메라의 히트맵과 카메라 파라미터를 **둘 다** 제거하여 남은 3개 뷰로
   로컬 3D 큐브를 다시 투영/정규화한 뒤 동일한 PoseNet을 한 번만 실행한다.
   히트맵을 0으로 바꾸고 4개 뷰로 계속 나누는 방식이 아니다.
7. 정상적인 재추론 결과를 출력한다. 여전히 오차가 크거나 오히려 늘어도 추가 재시도하지
   않는다. 재투영 큐브가 비면 최초 결과를 유지한다. 모든 경우 CSV ID/root는 유지한다.

이 기능은 동기식 오프라인 runner 안에서 카메라 목록을 일시 교체하고 `finally`로
원복한다. 병렬 요청을 받는 라이브 서버에 해당 모델 인스턴스를 그대로 공유하면 안 된다.
온라인 적용 시에는 요청별 projector 또는 동기화가 별도로 필요하다.

### 진단과 한계

- `people[].scenario.view_retry`: 재시도 0/1, 사용 여부, 제외 카메라 ID, 전후 뷰별 오차/
  유효 관절 수/보류 이유를 기록한다. 카메라 ID와 배열 인덱스는 구분한다.
- `raw_predictions.jsonl`: 최초/재추론 원시 관절과 관절별 오차까지 기록한다.
  `raw_joints_mm`은 최종 채택한 원시 관절이다.
- `result.json`: 재추론 횟수/카메라별 제외 횟수, **같은 남은 뷰의 같은 유효 관절**에 대한
  전후 평균 거리, 개선된 사람-프레임 수를 기록한다. 큰 오차 뷰를 빼기만 해도 평균이
  낮아지는 착시를 피하려고 비교 대상 관절/뷰를 동일하게 맞춘다.
- 평가 기준은 **CSV 골반 보정 후 pose**다. CSV XY/높이/시간 또는 캘리브레이션의
  오류도 재투영 오차에 섞인다. 고정한 root 자체가 잘못된 경우 뷰 제외로 해결되지 않는다.
- 가까운 피크도 다른 사람의 관절일 수 있다. 이 방식은 2D 인물 ID 매칭을 보장하지 않는다.
  피크가 없는 뷰는 높은 오차로 간주하지 않으며, 신뢰할 증거가 없으면 제외하지 않는다.
- 6픽셀/0.1/6관절은 초기 휴리스틱이며 실제 정답으로 최적화한 값이 아니다.
  히트맵 일치도 개선은 3D 정확도 개선을 뜻하지 않는다. 시각화 비교가 필요하다.
- 정상 재추론이 최초 결과보다 나빠도 사용한다. '개선된 경우에만 채택'하는 best-of-two
  정책은 이번 요청의 단일 재추론 실험에 추가하지 않았다.

신규 10개 + 기존 11개 테스트로 1회 제한, 다중 피크, 증거 부족/화면 밖,
최소 관측 뷰, 카메라-히트맵 동시 제거/원복, 실제 projector의 3뷰 정규화를 검증한다.

### 2026-09-28 전체 실행 결과

실행 폴더: `data/recordings/runs/scenario_root_pose_B_v2_viewdrop1_20260928/`.
세 영상 처리 시간은 약 363초다. 아래 오차는 재추론한 사람-프레임의 **동일한 남은
뷰·관절 쌍**에 대한 평균 히트맵 픽셀 거리이며, 사람-프레임마다 같은 가중치로 평균했다.
'개선 비율'의 분모도 전체 관측 수가 아니라 재추론 수다.

| 비교 JSONL | 프레임 | 전체 pose 관측 | 재추론 관측 | 평균 오차 전→후 | 개선 비율 |
|---|---:|---:|---:|---:|---:|
| [0812_1](../data/recordings/scenario_root_posenet_B_v2_viewdrop1_0812_1.jsonl) | 1,406 | 4,496 | 4,460 | 10.754→10.928 px | 49.35% |
| [0812_2](../data/recordings/scenario_root_posenet_B_v2_viewdrop1_0812_2.jsonl) | 843 | 1,986 | 1,836 | 10.408→10.664 px | 49.78% |
| [0812_3](../data/recordings/scenario_root_posenet_B_v2_viewdrop1_0812_3.jsonl) | 1,334 | 5,943 | 5,909 | 10.219→10.243 px | 75.04% |

전체 파일명은 `scenario_root_posenet_B_v2_viewdrop1_0812_N.jsonl`이다.
세 영상 모두 pose 누락 없이 완료됐다. 임계값 이하로 유지한 관측은 각각 27/135/19개,
제외 후 root 관측 카메라가 2개 미만이라 유지한 관측은 각각 9/15/15개다.

0812_3은 대부분의 재추론 관측에서 조금 개선됐지만, 일부 악화된 관측 때문에 평균은
소폭 증가했다(관측별 변화 중앙값 −0.100px, P95 +1.196px, 최대 +8.279px).
따라서 **무조건 재추론 결과를 채택하는 이 실험을 전반적인 품질 개선으로 판정하지 않는다.**
GT 3D 정확도와 시각적 자연스러움은 별도 평가가 필요하다.

전체 3,583프레임/12,425개 pose를 다시 읽어 다음을 확인했다.

- 기존 JSONL 세 파일 SHA-256 보존, 동일 히트맵 입력 패킷 해시.
- 모든 최초 원시 pose가 기존 실행의 원시 pose와 정확히 동일(최대 차이 0mm).
- CSV root/ID/시간/selection flags 유지, 유한한 15관절, 골반=root 정확히 일치.
- 사람-프레임별 재추론 횟수 0 또는 1, 임계값 초과한 최악 카메라를 제외.
- 재추론하지 않은 pose는 기존 pose와 정확히 동일.

검증 요약은 실행 폴더의 `audit.json`, 전후 관절별 수치는 각 시나리오의
`raw_predictions.jsonl`에 있다. 기존 baseline 및 운영 프로필/모델 가중치는 변경하지 않았다.
