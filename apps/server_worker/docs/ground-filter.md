# 최종 추론 출력의 지면 높이 필터

2026-09-29부터 다음 실행 경로에서 기본 활성화된다.

- `apps/server_worker/inference.py` (실시간/녹화 패킷 replay, `run.sh server` 포함)
- `Faster-VoxelPose/run/real_adapt.py record`
- `apps/server_worker/tools/scenario_pose.py` (CSV root + 실제 Pose)

학습과 raw-head 정량 audit, 합성 gait 생성기에는 적용하지 않는다. 이미 저장한
원본/학습본 JSONL이나 가중치, 카메라 값, ground mesh는 수정하지 않았다.
실행 중인 프로세스를 재시작하지 않았으며, 다음 실행부터 새 기본값을 사용한다.

## 기본 동작

`apps/server_worker/domain/ground_filter.py`의 공통 구현을 사용한다.

| 검사 | 제거 조건 |
|---|---|
| 검출 root | 해당 XY의 ground보다 1,200mm 초과 |
| 예측 pelvis | 해당 XY의 ground보다 1,200mm 초과 |
| 양발 | 양쪽 발목이 각각 자기 XY의 ground보다 모두 350mm 초과 |

높이는 절대 Z가 아니라 `Z - ground(X,Y)`이다. 15관절 포맷의 pelvis=2,
left ankle=8, right ankle=14를 검사한다. 발바닥이 아닌 발목 관절 기준이다.
한쪽 발만 들어 올린 보행은 이 발 조건으로 제거하지 않는다. 한계를 정확히
만나는 값도 유지한다. 계단·경사를 위해 각 관절 위치마다 지면을 조회한다.
관절이 없는 LOD0/1은 root만 검사한다. 다른 관절 포맷은 인덱스를 추측하지
않고 root만 검사하며 `unsupported_pose_ids`에 남긴다.

위반한 사람 entity 전체를 그 snapshot에서 제외한다. root를 지면으로 내리거나
자세를 이동/변형하지 않는다. 신규 관측의 필터는 인원 cap과 SceneState 업데이트
전에 적용하므로 같은 edge의 이전 관측을 현재 관측 대신 되살리지 않는다.
내부 ReID/공간 track 상태를 삭제하지는 않는다.

`runtime.ground_filter`에 기준값, 입력/유지/제거 건수, 제거 ID·edge·사유,
해당 root/pelvis/발목 높이를 저장한다. 이 통계는 **현재 들어온 관측 기준**이다.
부분 edge 업데이트의 최종 합쳐진 사람 수나 cap으로 제거한 수를 뜻하지 않는다.
아직 Pose가 없는 관측에서는 양발 상태를 판정할 수 없다.

지면 밖 위치는 높이 0으로 간주하지 않는다. 기본은 확인 불가 관측을 유지하고
`missing_ground_ids`에 기록한다. `--ground-missing drop`이면 그 관측을 제거한다.
지면 표면 자체가 전혀 설정되지 않았다면 활성화 상태로 시작할 수 없다.
정확한 deployment/ground를 주거나 명시적으로 `--no-ground-filter`를 사용한다.

## 설정

```bash
bash apps/deployments/rootnet_v2/run.sh server tcp/127.0.0.1:7447 \
  --ground-max-root-mm 1200 \
  --ground-max-feet-mm 350 \
  --ground-missing keep
```

동일 옵션을 FVP `record` 및 `scenario_pose.py`에도 사용할 수 있다.
이전 필터 없는 결과를 재현하려면 `--no-ground-filter`를 붙인다.

서버 runtime JSON profile의 `arguments` 키로도 설정할 수 있다.

```json
{
  "ground_filter": true,
  "ground_max_root_mm": 1200.0,
  "ground_max_feet_mm": 350.0,
  "ground_missing": "keep"
}
```

이 값은 사람 자세/신장/현장 지면 오차에 맞춰 조정할 초기 휴리스틱이지 학습된
확률이 아니다. 점프나 높은 골반의 정상 사람도 제거할 수 있다. 기존 RootNet
workspace의 400–1,400mm 검색 범위는 변경하지 않았다. 필터 상한을 늘려도
모델이 원래 검색하지 않은 위치의 사람을 새로 검출할 수는 없다.

## 확인한 결과

기존 FVP 학습본 B의 세 기록을 읽기만 하여 기본 필터를 적용할 경우를 집계했다.
이전 기록은 변경하지 않았다. 관측 건수는 고유한 사람 수가 아니다.

| 데이터 | 기존 관측 | 유지 | 높은 root | 높은 pelvis | 양발 높음 |
|---|---:|---:|---:|---:|---:|
| 0812_1 | 7,876 | 6,778 | 891 | 44 | 163 |
| 0812_2 | 3,054 | 2,487 | 426 | 82 | 59 |
| 0812_3 | 10,878 | 7,731 | 2,634 | 41 | 472 |

이 집계는 이미 저장된 최대 10명 출력에 적용한 진단이다. 새 추론에서는 cap
앞에서 검사하므로, 걸러진 후보 대신 다른 정상 후보가 들어와 집계가 달라질 수 있다.
**제거 수는 정확도가 좋아졌다는 증거가 아니며**, 독립 정답으로 검증하지 않았다.
현재 파일에서는 지면 확인 불가 관측은 없었다.

실제 FVP B 모델의 320프레임 추론/JSONL 기록을 별도 `/tmp` 폴더에서 실행했고,
모든 출력이 높이 기준을 통과하는지 및 Isaac 파일 로더 호환성을 검증했다.
서버 실제 프로필 `--validate-only`도 통과했다.

필터(18개), Pose/뷰 재시도, FVP 연결부, 녹화/Isaac 관련 테스트 56개를 통과했다.
별도 frontend 테스트 4개도 통과했다. 다만 60개를 한 번에 실행했을 때 기존
WebSocket 테스트의 구독 해제 즉시 검사 1건이 실패(59 통과)했고, 분리 재실행에서는
통과했다. 종료 처리의 타이밍 영향이 의심되며, 이번 작업에서 프런트엔드 코드는
변경하지 않았다. 그 테스트의 비결정성을 해결했다고 주장하지 않는다.

현재 ground는 여전히 deployment에 등록된 USD 표면/캐시다. VGGT point cloud의
지면으로 자동 전환한 것이 아니다. 지면 좌표·기울기·높이가 틀리면 정상 사람도
제거될 수 있으므로 이 필터를 카메라 캘리브레이션 보정으로 해석하면 안 된다.
