# 현재 캘리브레이션 진단과 다음 보정안

후속 실행: [부분 보정·Faster-VoxelPose 재학습 결과](../../../Faster-VoxelPose/docs/calibration_refined_20260929.md).
아래는 실행 전 진단/설계안이며, 후속 실험에서는 2·4·8번 회전만 채택했다.

## 확인한 사실

FVP/SelfPose 실제 영상 비교가 사용하는 파일:
`apps/calibration_worker/data/vggt_cloud_90.NWJ28r/calibration_result.json`.
SHA256 `b58fd180b2347a1a106ab206a32b003f48124297ced993b382a3b01ec96ede67`.
이 문서는 이 파일에 대한 진단이며 다른 수동 수정본까지 같은 상태라고 단정하지 않는다.

`input.bundle_adjustment`를 확인한 결과:

- 98장 중 CCTV 8장, reference 90장.
- camera/2, /4, /6, /8의 feature observation 수가 **11, 15, 11, 15**.
- `fixed_weak_cctv`에 위 네 카메라가 모두 포함된다. 모두 edge_1 카메라다.
- 현 코드의 `BA_MIN_CCTV_OBSERVATIONS=20`보다 적어, 카메라 포즈 최적화에서
  제외하고 VGGT 초기 포즈에 고정한다. 뒤의 전역 ArUco 정합은 적용되지만 각
  카메라의 상대적인 포즈 오차를 독립적으로 보정하는 것은 아니다.
- 전체 fitting RMSE는 8.786 → 2.444px로 줄었지만 `solver_converged=false`,
  `solver_nfev=300`이며 최대 함수 평가 횟수에 도달했다. 이 값은 BA 입력 영상
  좌표계의 fitting 오차이지 1920×1080 녹화의 holdout 오차가 아니다.
- 현재 구현은 유한하고 오차가 감소한 반복 제한 해를 경고 후 보존한다.
- K/왜곡은 고정한다. 원래 intrinsic이 잘못됐으면 extrinsic BA만으로 모두
  해결할 수 없다. VGGT 추정 K와 입력 K의 차이도 실제 K의 정답 오차는 아니다.

따라서 캘리브레이션이 주요 병목이라는 의심을 뒷받침하는 근거가 있다. 그러나
수렴 실패만으로 모든 포즈가 틀렸다고 단정하거나, 현재 pose 오차 전부를 카메라
오차로 환산할 수는 없다. 시간 동기화·2D 검출·ground·모델 검출 문제도 분리해야 한다.

## 권장 순서 — 아직 구현/실행하지 않은 다음 실험

1. **추론에 사용하는 바로 그 0812 녹화에서 대응점 확보.** 현재 0815 reference
   및 CCTV 사진과 0812 녹화 사이에 카메라가 움직였는지 정지 배경으로 확인한다.
   해상도/크롭/K 스케일/왜곡을 한 번만 적용했는지, camera ID/order도 확인한다.
   사람을 대응시킬 때는 같은 프레임 번호뿐 아니라 실제 영상의 시간 오프셋을 확인한다.
2. **우선 camera/2, /4, /6, /8에 다중뷰 정지 배경 대응점을 보강.** 타일 모서리,
   벽·문·기둥 모서리를 이용하되 바닥 한 평면에만 몰리지 않게 높이·깊이·화면 위치를
   분산한다. 자동 대응이 부족하면 같은 점을 서로 다른 영상에서 직접 클릭하는
   **2D↔2D 대응**을 보탠다. 이 단계는 부정확한 USD/VGGT 점의 XYZ를 정답으로
   고정하지 않고, 점의 3D 위치와 카메라를 함께 조정하는 BA를 목표로 한다.
   단순히 관측 수 제한을 낮춰 약한 카메라를 강제로 움직이는 것은 피한다.
3. **배경이 부족할 때 사람을 이동하는 보정 패턴으로 사용.** 기존 독립 2D 검출과
   cross-view ID 대응을 이용해 여러 프레임의 관절을 추가한다. 8개 고정 CCTV의
   extrinsic은 전체 시간에 하나씩 공유하고, 프레임별 3D 관절/위치는 별도 변수로 둔다.
   robust reprojection + 뼈 길이 일관성 + 시간 연속성 + 신뢰할 수 있는 지면의
   접지 프레임만 약한 제약으로 사용한다. 양발 항상 접지는 강제하지 않는다.
   추론은 계속 4뷰씩 해도 보정은 연결된 8뷰 전체를 함께 최적화할 수 있다.
4. **변수와 자유도 제한.** 먼저 2D/3D 네트워크 가중치, K, 왜곡을 동결하고 작은
   extrinsic 보정부터 시작한다. 3D network가 만든 의사 정답에 카메라를 다시
   맞추는 순환 최적화는 피한다. 공간 전체의 평행이동/회전/스케일 자유도를 고정할
   기준 pose와 스케일 제약이 필요하다. 신뢰할 실측 길이가 없다면 기존 metric
   scale을 유지하는 상대 보정까지만 주장한다. 지면만으로 절대 스케일은 정해지지 않는다.
   residual이 렌즈 문제를 가리킬 때만 충분한 분포의 정적 관측으로 일부 intrinsic을
   제한적으로 풀고, 결과를 별도로 검증한다.
5. **모델을 고정한 보정 전후 비교.** 보정에 안 쓴 배경 대응점 및 시간 구간에서
   카메라별 reprojection 중앙값/P95, epipolar 잔차, floor clearance 분포,
   검출 누락·중복을 함께 본다. 지면 필터로 지운 뒤 예뻐진 화면이나 BA fitting
   평균만으로 보정을 승인하지 않는다. 새 JSON으로 내보내고 원본은 유지한다.

VGGT cloud는 초기 구조/시각화에 유용하지만 동일 추정에서 나온 카메라와 cloud가
서로 맞는 것만으로 실제 정확도를 검증할 수 없다. 현재 수동 editor의 고정 cloud
2D↔3D 보정과, 위의 다중뷰 2D↔2D 공동 보정은 다른 기능이다. 신뢰할 정지점과
시야 중첩이 전혀 없고 독립 스케일 기준도 없으면 영상만으로 정확한 metric
world calibration을 유일하게 복구할 수 있다는 보장은 없다.

## 근거 자료

- [Wide-Baseline Multi-Camera Calibration Using Person Re-Identification, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Xu_Wide-Baseline_Multi-Camera_Calibration_Using_Person_Re-Identification_CVPR_2021_paper.html):
  사람의 cross-view 대응과 다중뷰 기하/BA를 사용하는 연구 사례. 현장 적용 성공을 보장하는 자료는 아니다.
- [Multi-View Person Matching and 3D Pose Estimation with Arbitrary Uncalibrated Camera Networks](https://arxiv.org/abs/2312.01561):
  캘리브레이션되지 않은 카메라 네트워크에서 사람 대응, triangulation 및 BA를 결합한다.
- [COLMAP FAQ: Fix intrinsics / Principal point refinement](https://colmap.github.io/faq.html#fix-intrinsics):
  내부 파라미터 자유도를 제한하고, 특히 principal point 최적화의 불량 조건을 주의한다.

이번 변경에서는 출력의 지면 필터만 구현했다. 카메라 값이나 모델 가중치를
자동으로 보정/배포하지 않았고, 위 내용은 다음 실험의 설계안이다.
