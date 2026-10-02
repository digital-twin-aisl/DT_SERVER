# CSV 경로 기반 데모 스켈레톤

`tools/scenario_skeleton.py`는 CSV의 인물 ID와 XY를 고정하고 걷기/정지 동작을
합성하여 기존 브라우저 뷰어·Isaac Scene Player용 JSONL을 생성한다.
**영상의 실제 동작을 추정한 결과가 아니다.** 모델 추론/성능 평가와 분리된 데모다.
실행 중인 서버, 엣지 설정, 원본 녹화는 변경하지 않는다. Python과 NumPy만 필요하다.

저장소 루트에서:

```bash
python apps/server_worker/tools/scenario_skeleton.py \
  --csv 'scenario (1)/scenario_1.csv' \
  --reference-recording apps/server_worker/data/recordings/rootnet_B_v2_001.jsonl \
  --ground-cache apps/deployments/cache/scene_0812_2_ground.npz \
  --fps 29.97002997002997 \
  --output apps/server_worker/data/recordings/scenario_1_synthetic.jsonl
```

기존 `/viewer`의 **JSONL 파일 열기** 또는 Isaac의 **Meta Sejong Scene Player**로
출력 파일을 연다. 이미 존재하는 출력은 덮어쓰지 않으므로 다시 생성할 때 새 이름을 지정한다.

## Zone / Rank 표시 버전

위 명령에 `--selection-column zone_selected` 또는 `--selection-column rank_selected`를
추가하면 해당 CSV 열에 따라 표시를 전환한다. 출력 이름도 각각
`scenario_1_zone_synthetic.jsonl`, `scenario_1_rank_synthetic.jsonl`처럼 지정한다.
시나리오 2·3은 CSV와 출력 파일명을 바꾸고 `--reference-recording`을 생략하면
기존에 생성한 각 CSV의 시작/종료 시각을 유지한다.

- 선택값 **0**: `pose=null`, 내부 `lod=1` → 기존 뷰어의 위치 표시용 캡슐(실린더 형태).
- 선택값 **1**: 15관절 `pose`, 내부 `lod=2` → 스켈레톤.
- 값은 인물별 직전 CSV 행에서 가져와 다음 행의 시각까지 유지한다. 선택값은 보간하지 않는다.
- ID, XYZ 위치, 시간축, 스켈레톤 동작은 두 버전에서 동일하다. 선택에 따른 표시만 다르다.
- 선택한 열이 없거나 값이 0/1이 아니면 생성 전에 실패한다.
- 기본 `--selection-column all`은 기존처럼 모든 인물을 스켈레톤으로 표시한다.

- `global_id` 0~4를 그대로 사용하며 XY는 CSV 시각 사이에서 선형 보간한다.
  각 인물의 첫/마지막 관측 밖으로는 외삽하지 않는다. 기본 1.01초보다 큰 CSV 공백은
  별도 구간으로 나누며 공백 동안 인물을 표시하지 않는다 (`--max-gap`으로 조정).
- 기준 녹화에서는 시작/종료 시각만 읽는다. 기존 오검출, ID, 포즈, 처리 간격은 복사하지 않는다.
  일정 FPS로 생성하며 마지막 프레임은 원본 종료 시각을 정확히 포함한다.
- 기본 시간 변환은 `출력 시각 = CSV 시각 + 0초`다. 따라서 이 데이터는
  0초부터 재생하되 첫 인물은 약 8.258초에 등장한다. 마지막 인물은 43.508초까지다.
  영상과 CSV의 실제 동기화/좌표 정합은 검증되지 않았다. 필요 시 `--time-offset`을
  명시한다. 예: CSV 8.258초를 영상 0초에 대응시키려면 `--time-offset -8.258`.
  기준 녹화를 생략하면 CSV의 처음부터 마지막 시각까지만 생성한다.
- CSV 좌표를 USD world, Z-up, mm로 해석한다. 발 높이는 기존 Ground 캐시로 조회하고,
  골반 높이는 지면 및 두 발까지의 도달 거리에 맞춘다. 지면이 없으면 오류로 종료한다.
- 보행 위상은 이동 거리에 따라 진행한다. 지지 중인 발 위치를 고정하고, 스윙하는 발은
  지면 위로 들어 올린다. 다리는 두 구간 역기구학으로 각 440mm를 유지하며 팔은 반대로 흔든다.
  이것은 단순 보행 합성이므로 달리기, 급회전, 실제 제스처, 충돌 회피를 재현하지 않는다.
- 관절은 기존 `voxelpose_15j_xyz` 순서다. `zone_selected`와 `rank_selected`는
  `people[].scenario`에 보존하고, 선택한 열을 표시 전환에 사용한다. 온라인 추론 정책은 바꾸지 않는다.
- `runtime.scenario`와 `people[].scenario`에 합성 출처를 명시한다. 입력 SHA-256,
  시간 오프셋, FPS를 기록한다. root의 confidence=1은 CSV 지정값이지 모델 신뢰도가 아니다.
  기존 뷰어는 합성 여부를 별도 배지로 표시하지 않으므로 파일명도 `_synthetic`을 사용한다.

검증:

```bash
python -m unittest discover -s apps/server_worker/tests -p 'test_scenario_skeleton.py' -v
```
