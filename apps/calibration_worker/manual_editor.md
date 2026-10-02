# Local Camera Pose Workbench

`manual_editor.py`는 0812 녹화와 3D 맵을 보면서 카메라 **외부 파라미터만** 수정하는
독립적인 로컬 웹 편집기다. VGGT, GPU 추론, Isaac Sim, Zenoh, edge manager가 필요 없다.
UI는 영어다. 영상/맵/편집값은 외부 서비스에 전송하지 않는다.

## 시작

저장소 루트에서:

```bash
bash apps/calibration_worker/build_manual_editor.sh
python apps/calibration_worker/manual_editor.py --port 8092
```

브라우저에서 <http://127.0.0.1:8092/>를 연다. 종료는 실행 터미널에서 Ctrl+C.
`127.0.0.1`에만 바인딩하므로 다른 컴퓨터에는 공개되지 않는다.
원격 개발 PC라면 SSH 로컬 포트 포워딩을 이용한다. 공개 웹 호스팅 도구가 아니다.

기존 `apps/frontend_api/web/node_modules`의 Three.js/esbuild를 재사용한다.
없으면 해당 디렉터리에서 `npm ci` 후 빌드한다. 기존 뷰어 코드나 lockfile은 변경하지 않는다.
Python에는 numpy, scipy, OpenCV, fastapi, uvicorn이 필요하다. 현재 Anaconda 환경에는
설치되어 있다. 작은 6변수 CPU 보정에서 Intel OpenMP 심볼 충돌을 피하기 위해 시작 시
`MKL_THREADING_LAYER`가 미지정이면 `SEQUENTIAL`을 설정한다. 이미 충돌하는 값을
지정한 셸에서는 `MKL_THREADING_LAYER=SEQUENTIAL`을 명시하여 실행한다.

### 기본 입력

- 캘리브레이션: `apps/deployments/rootnet_v2/calibration.from_cameras_v2.json`.
  이전 v2 camera YAML에서 추출한 숫자 전용 JSON이며, 이전 실험의 v2 초기값이다.
- 영상: `apps/edge_client/data/data_0812_{1,2,3}_edge_{1,2}/camera_N.mkv`.
  짝수 카메라는 edge_1, 홀수 카메라는 edge_2다.
- 맵: `apps/frontend_api/assets/map.glb`, USD 월드 미터 좌표를 보존한 기존 export.
- 카메라별 `source_image_size`가 없으면 intrinsic 기준 해상도를 1920×1080으로
  **명시적으로 가정**한다. 현재 녹화는 이 해상도다.

```bash
# 다른 초기 캘리브레이션으로 시작
python apps/calibration_worker/manual_editor.py \
  --calibration apps/calibration_worker/calibration_result_20260814-123941.json \
  --port 8092
```

추가 옵션: `--data-dir`, `--assets-dir`, `--calibration-width`, `--calibration-height`.
이 도구는 OpenCV 카메라 좌표계, USD Z-up 월드 미터 단위의 현재 프로젝트용이다.
다른 단위/맵 좌표계 파일을 넣어도 자동 정합하지 않는다.

## 권장 작업 순서

### VGGT-omega 포인트클라우드를 기준으로 사용

로컬 VGGT `inference.py` 실행은 이제 `calibration_result.json`과 짝지어진
포인트클라우드 NPZ를 함께 저장한다. 기존 명령에 출력 폴더를 추가하면 된다:

```bash
python apps/calibration_worker/inference.py \
  --mode offline \
  --images apps/calibration_worker/data/cctv_photos \
  --camera-config apps/edge_client/config/cameras.local.yaml \
  --reference-video apps/calibration_worker/data/260815_센B1복도.mp4 \
  --checkpoint apps/calibration_worker/vggt-omega/vggt_omega_1b_512.pt \
  --marker-tree apps/isaac_sim_client/aruco_boards/aruco_marker_tree.usd \
  --reference-sample-count 90 --max-images 98 --use_ba \
  --point-cloud-max-points 2000000 \
  --output-dir apps/calibration_worker/data/vggt_cloud_90

python apps/calibration_worker/manual_editor.py \
  --calibration apps/calibration_worker/data/vggt_cloud_90/calibration_result.json \
  --port 8092
```

`3D reference`는 짝지어진 cloud가 있으면 **VGGT point cloud**로 시작한다.
BA도 기본적으로 CCTV 8장과 reference 90장 전체를 사용한다. 관측 부족으로 고정된
포즈나 solver 반복 한도 도달 여부는 결과 JSON의 `input.bundle_adjustment`에서 확인한다.
**USD map**으로 다시 전환할 수 있다. JSON과 해당 NPZ는 같은 폴더에 두어야 하며,
다른 실행의 cloud를 섞거나 파일이 변조되면 로드를 거부한다. 수동 수정 JSON을
다운로드해 다시 열 때도 해당 NPZ를 옆에 복사해야 한다. cloud는 움직이지 않고
편집한 카메라만 이동한다. 기존 투영/왜곡/빨간 점 가이드/기준점 선택 기능은 유지된다.

- **Min confidence**: 저장된 점에서 신뢰도가 낮은 점을 더 숨긴다. 신뢰도는 확률이 아니다.
- **Point size (m)**: 점의 표시 크기와 점 선택 허용 반경을 조절한다.
- **Hide selected camera's points**: 현재 CCTV가 만든 점을 제외하고 다른 입력 뷰의 점으로 비교한다.
  이는 자기 영상 재투영에 의존하는 정도를 줄일 뿐, 독립적인 정확도 검증은 아니다.
- 저장은 입력 padding을 제외하고 유효 깊이/신뢰도를 필터링한 뒤 뷰별 균등 예산으로 다운샘플링한다.
  움직이는 사람/반사면을 자동 제거하거나 여러 뷰의 점을 표면으로 융합하지는 않는다.
- `--use_ba` 이후 최종 외부 파라미터로 dense depth를 역투영한다. 깊이와 결합된
  **VGGT 예측 K**를 사용하고, 카메라와 같은 `reconstruction_to_usd`를 적용한다.
  BA는 dense depth 자체를 보정하지 않으므로 중복 표면이나 어긋남이 남을 수 있다.
- 이 cloud는 실측 정답이 아니다. USD mesh는 사용하지 않지만 marker-tree의 기준
  좌표/스케일에 대한 의존성은 남는다. 0812 녹화와 0815 사진 사이 카메라 이동도 별도 확인한다.

### 편집 순서

1. **Recording / Camera**를 선택한다. 영상은 실시간 재생 대신 원하는 프레임으로
   이동하는 정지 프레임 편집 방식이다. Time/슬라이더/±1 프레임 버튼을 사용할 수 있다.
2. 아래 표에 따라 **Image model**을 선택한다. 원본 여부를 자동 추측하지 않는다.
3. 맵의 기본 **Matched camera** 뷰는 녹화 영상과 동일한 fx/fy/cx/cy, 왜곡 모델,
   W/H 및 현재 편집 중인 pose로 렌더링한다. 카메라를 수정하면 즉시 갱신된다.
   **Pick landmark** → 맵의 실제 구조물 모서리 클릭 → 영상의 동일 지점 클릭.
   다른 각도에서 기준점을 찾으려면 **Free orbit**을 선택한다. 이 자유 시점은 비교용이 아니다.
   맵 좌표를 이미 알면 World point XYZ에 직접 입력할 수 있다.
4. 선택한 점을 영상에서 다시 클릭하면 관측 위치를 수정한다. 표의 행으로 점을 선택하고
   Delete selected로 제거한다. 삭제/pose 변경/관측 변경은 Undo로 복구할 수 있다.
5. Position offset/Local rotation offset 숫자·슬라이더·± 버튼으로 조정한다.
   청록색은 수정 pose의 재투영, 황색은 원본, 흰 십자는 관측점, 붉은 선은 잔차다.
6. 숫자만 맞추지 말고 화면 전체에 분산된 점, 서로 다른 높이·깊이의 점으로 비교한다.
   일부 점은 **Holdout**으로 표시해 보정에는 쓰지 않고 검증에만 사용한다.
7. 선택적으로 **Preview refinement**를 누른다. fit 대응점 최소 6개가 필요하며
   고정 intrinsic/왜곡값 아래에서 현재 pose 주변 위치 ±2m/회전축별 ±15° 범위로
   soft-L1 재투영 잔차를 줄인다. 결과와 holdout 오차를 보고 **Apply proposed pose**를
   눌러야 적용된다. 한계에 도달하거나 점들이 같은 평면에 있으면 경고한다.
8. **Save project**로 대응점과 편집 상태를 저장하고, **Export calibration**으로
   수정 JSON을 다운로드한다. 다운로드 위치는 브라우저 설정을 따른다.

| Image model | 표시 영상 | 재투영 모델 / 맵 오버레이 |
|---|---|---|
| Raw + lens distortion | 녹화 프레임 그대로 | 원래 K와 왜곡계수, 왜곡 적용 맵 및 오버레이 가능 |
| Undistort raw video | OpenCV alpha=0으로 왜곡 보정 | 같은 new K와 zero distortion, 맵 오버레이 가능 |
| Already undistorted / simulation | 녹화 프레임 그대로 | 저장된 undistorted K(없으면 K)와 zero distortion, 오버레이 가능 |

이미 보정된 영상에 `Undistort raw video`를 적용하면 이중 보정되어 잘못된다.
시뮬레이션 영상은 생성 때 사용한 K와 `pinhole` 모드의 K가 일치하는지도 확인한다.
모드/녹화/카메라마다 관측점을 별도로 관리하므로, 서로 다른 영상 좌표를 섞지 않는다.
동일 카메라 pose는 녹화 선택을 바꾸어도 유지된다. 녹화 사이 실제 카메라가 이동했다면
별도 프로젝트/수정 JSON으로 관리해야 한다.

### 영상과 동일한 reference map 렌더링

Recorded view를 클릭하면 두 화면의 **동일한 영상 픽셀 좌표**에 작은 빨간 점이
누적 표시된다. 여러 개를 찍을 수 있으며, 패널 표시 크기가 달라도 원본 W/H를 기준으로
위치를 맞춘다. 왜곡을 추가 적용하거나 3D 표면으로 스냅하는 도구는 아니다.
**Guide only**를 켜면 선택된 기준점의 관측 좌표를 바꾸지 않고 가이드만 놓는다.
끄면 기존 관측점 지정 기능과 함께 동작한다(기준점 미선택 시 가이드만 표시).
어느 화면에서든 빨간 점을 다시 클릭하면 해당 가이드만 양쪽에서 삭제된다.
Tab으로 점에 포커스를 옮겨 Enter/Space로도 삭제할 수 있다. 기준점은 삭제하지 않는다.
**Clear all guides**로 모두 지울 수 있다. 프레임/카메라/녹화/영상 모델 변경 시 초기화되고,
pose 조정 중에는 같은 픽셀에 유지된다. Free orbit에서는 맵 가이드를 숨긴다.
가이드는 보조 표시이며 프로젝트/캘리브레이션에 저장하지 않는다.

`Matched camera`와 영상의 `Map overlay`는 같은 렌즈 렌더러를 사용한다.
맵 카메라의 임의 FOV/aspect를 사용하는 것이 아니다. 최종 렌더 버퍼는 영상의 실제
W×H(현재 1920×1080), 픽셀 비율은 1로 고정하며 화면에서는 종횡비를 유지해 축소 표시한다.
맵 아래에 적용 중인 fx/fy/cx/cy·왜곡 계수·해상도를 표시한다.

Raw 모드에서는 출력 픽셀 중심마다 `cv2.undistortPointsIter`로 역방향 정규화 광선을
계산한다. 넓힌 pinhole 렌더 타깃에서 이 광선을 샘플링하는 GPU 후처리로 **원래 렌즈
왜곡을 다시 적용**한다. 중간 렌더 영역을 넓히므로 배럴 왜곡의 가장자리가 기존 pinhole
화각에서 잘리는 것을 방지한다. 중간 타깃만 최대 4096픽셀로 제한하며 최종 W/H는 바꾸지 않는다.
맵에서 기준점을 고르는 raycast도 같은 역왜곡 광선을 사용한다.

기존 OpenCV Brown 계열 왜곡계수(4/5/8/12/14개)를 그대로 사용한다. 픽셀 → 광선 →
픽셀 왕복 오차가 0.05px를 넘거나 유효한 역변환이 없는 위치는 마스킹하고 선택하지 않는다.
이 구간은 화면에 비율을 표시하며, pinhole로 몰래 대체하지 않는다. 현재 v2의 일부 Raw
카메라는 가장자리에 이런 구간이 있다. 이는 카메라 파라미터가 그 영역에서 유효한지
검토할 신호이지, 해당 영역을 임의로 늘려 맞춰야 한다는 뜻은 아니다.
Rectified/Pinhole 모드는 녹화 표시와 동일한 K 및 zero distortion을 사용한다.

렌즈 lookup은 pose와 무관하므로 카메라/영상 모드를 바꿀 때만 생성한다.
서버는 최근 2개 모델만 캐시하고, 브라우저는 선택 모델만 유지한다.

## 좌표와 회전

- 월드 XYZ는 미터, Z-up. pose 추론의 mm 좌표를 그대로 입력하면 안 된다.
- 카메라는 OpenCV +X right, +Y down, +Z forward다.
- Position offset은 현재 기준 pose에 **월드 XYZ 이동량**을 더한다.
- Local rotation offset은 현재 기준 카메라 축에 대해
  `R_new = R_base @ Rz(z) @ Ry(y) @ Rx(x)`를 적용한다. yaw/pitch/roll을 월드 축으로
  돌리는 방식과 다르며 단위는 도다. 서버와 Three.js가 동일한 회전 순서를 사용한다.
- Zero controls here는 현재 pose를 조절 기준으로 삼고 입력값만 0으로 만든다.
  Reset camera pose는 해당 카메라의 원본 pose로 되돌린다. 대응점은 유지한다.
- Three.js 표시용 Y-up 변환은 편집 UI 안에서만 수행한다. 맵 클릭 좌표는 역변환해
  원래 USD 월드 좌표로 저장하며, export JSON에는 Three.js 좌표를 저장하지 않는다.

## 오차를 해석할 때

RMSE는 **표시 영상의 원래 해상도 픽셀 단위**다. CSS 화면 크기를 바꿔도 관측 좌표는
원영상 좌표로 저장된다. 카메라 뒤의 점은 투영되지 않으며 유효 점 수/관측점 수를
같이 표시한다. 점이 뒤로 사라져 평균 오차만 작아진 것을 개선으로 보지 않는다.

USD 맵 자체가 실제 환경과 다르거나, 영상 K/왜곡 모델이 틀리면 pose만으로 맞출 수 없다.
바닥의 한 구역만 맞추거나 움직이는 사람의 추론 pose에 맞춰 카메라를 수정하지 않는다.
가급적 건물/계단 등 정적인 구조물과 여러 높이의 기준점을 사용한다.
수동 fitting RMSE는 3D pose 정확도나 캘리브레이션 인증 결과가 아니다.

## 저장과 안전

- **Export calibration**: 원본 JSON을 복제하고 편집한 카메라의 `camera_to_world`,
  역행렬 `world_to_camera`, `position_m`을 함께 갱신한다. intrinsic/왜곡값은 보존한다.
  `manual_correction`에 원본 SHA-256/시각/변경 카메라 목록을 기록한다.
  원본의 alignment/품질 메타데이터는 재계산한 값이 아니며 이 사실도 함께 표시한다.
- **Save project / Open project**: 모든 카메라 pose, 모드별 대응점·holdout, 선택
  카메라/녹화/시간을 저장·복원한다. 원본 SHA-256이 다른 프로젝트는 불러오지 않는다.
- 원본 JSON, 카메라 YAML, registry, 배포 manifest에 **쓰는 API가 없다**.
  수정 JSON을 추론에 적용하는 것은 별도 명시적 작업이다.
- 편집 상태는 브라우저 메모리에만 있다. 자동 저장하지 않는다. 닫기 전에 프로젝트를
  저장한다. 새로고침도 저장하지 않은 상태를 잃는다.
- localhost host/origin 검사, 고정 영상·맵 경로만 허용하며 RTSP 비밀번호를 읽지 않는다.
- `sites-building`의 로컬 작업 원칙을 따라 외부 사이트 등록/호스팅을 생략했다.

## 검증

```bash
python -m unittest discover \
  -s apps/calibration_worker/tests -p 'test_manual_editor.py' -v
```

Python 편집기 19개 및 cloud 8개 테스트: rigid pose/회전/단위, OpenCV 투영·왜곡 보정, holdout 분리,
합성 대응점 보정 회복, export 원본 불변/역행렬 정합, query 인자·비정상 입력·접근 제어.
렌즈 lookup의 왜곡 왕복·오버스캔·비가역 영역 마스킹·영상 모드 일치도 포함한다.
브라우저용 `manual_web/geometry.test.js`의 15개 테스트는 기존 esbuild로 Node 번들하여
검사한다. 특히 Three.js 오버레이와 OpenCV pinhole 픽셀 좌표의 일치를 확인한다.
Matched view의 역왜곡 raycast, 좌표계 및 픽셀 중심/상하 방향도 검사한다.

WebMCP를 지원하는 브라우저에는 `read_camera_pose`, `stage_camera_offsets`가 등록된다.
이들은 UI와 같은 상태를 읽거나 조정하고 배포하지 않는다. 미지원 브라우저에서도 모든
UI 기능은 동일하다. 1cm 이동/원복과 잘못된 입력 거절을 실제 로컬 페이지에서 확인했다.
일반적인 브라우저 클릭·시각적 QA 전체를 자동 검증했다고 주장하지 않는다.

실제 로컬 서버에서도 0812_1/2/3 × 8개 카메라의 원본 프레임 24개와
추가 영상 모드 2개를 읽어 해상도/디코딩을 확인했다. 실제 캘리브레이션에서 생성한
**합성 대응점**으로 보정 API를 검사해 알려진 pose를 회복했고, export의 행렬 일관성 및
원본 SHA-256 불변을 확인했다. 이는 실제 환경 캘리브레이션 정확도 검증이 아니다.
