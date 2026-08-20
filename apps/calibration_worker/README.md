# Calibration Worker

VGGT-Omega가 복원한 임의 좌표계를 Isaac Sim ArUco marker tree의 미터 단위
좌표계에 정합하고 CCTV 카메라 extrinsics를 계산합니다.

## 준비 사항

- CUDA GPU와 CUDA 지원 PyTorch
- VGGT-Omega 체크포인트 (`VGGT-Omega-1B-512`)
- `opencv-contrib-python`의 `cv2.aruco`
- marker tree를 읽기 위한 Pixar USD Python 모듈 (`pxr`)
- CCTV 및 스마트폰 카메라의 OpenCV camera matrix와 distortion coefficients

서브모듈과 VGGT-Omega 의존성은 저장소 루트에서 다음과 같이 준비합니다.

```bash
git submodule update --init --recursive
pip install -r apps/calibration_worker/vggt-omega/requirements.txt
pip install -e apps/calibration_worker/vggt-omega
```

체크포인트는 VGGT-Omega Hugging Face 페이지에서 라이선스 승인을 받은 후 별도로
다운로드해야 합니다. 체크포인트 파일은 Git 또는 서브모듈에 커밋하지 않습니다.

엣지 agent와 calibration worker가 같은 파일을 사용하도록 환경 변수로 지정할 수
있습니다.

```bash
export VGGT_OMEGA_CHECKPOINT=/models/VGGT-Omega-1B-512/model.pt
```

## 대화형 분산 캘리브레이션

표준 CLI 진입점은 `inference.py`입니다. 인자 없이 실행하면 등록된 edge와 카메라
수를 먼저 표시하고 다음 순서로 edge, reference 영상, marker-tree USD를 선택합니다.

```bash
python apps/calibration_worker/inference.py
```

대화 없이 자동화하려면 다음처럼 동일한 값을 옵션으로 전달합니다.

```bash
python apps/calibration_worker/inference.py \
  --edge-id edge-001 \
  --reference-video /data/reference.mp4 \
  --marker-tree apps/isaac_sim_client/aruco_boards/aruco_marker_tree.usd \
  --checkpoint /models/VGGT-Omega-1B-512/model.pt
```

reference 영상과 같은 위치의 `<video>.json` 또는 `<stem>.camera.json`에
`camera_matrix`와 `distortion_coefficients`가 있으면 자동 사용합니다. sidecar가
없으면 영상 크기 기반 pinhole과 zero distortion을 가정하고 터미널에 경고합니다.
광각 reference 카메라는 반드시 sidecar를 제공하는 편이 안전합니다.

실행 결과는 `apps/calibration_worker/data/calibration_runs/<시각>_<edge_id>/`와
`apps/edge_manager/data/edges.json`에 저장됩니다. 이어서 Zenoh로 pose를 전송하고 edge의
`cameras.local.yaml` 저장 ACK를 받습니다. registry의 `last_calibration.edge_sync`는
`pending`, `applied`, `failed` 중 하나로 동기화 상태를 남깁니다.

### 오프라인 모드

CLI 첫 화면에서 `2. 오프라인`을 선택하면 edge registry와 Zenoh를 사용하지
않습니다. CCTV 사진 한 장 또는 사진 폴더, reference 영상, marker-tree USD를
입력받아 같은 VGGT-Omega·ArUco·USD 정합을 수행합니다.

```bash
python apps/calibration_worker/inference.py

# 자동화
python apps/calibration_worker/inference.py \
  --mode offline \
  --images /data/cctv_photos \
  --camera-config apps/edge_client/config/cameras.local.yaml \
  --reference-video /data/reference.mp4 \
  --marker-tree apps/isaac_sim_client/aruco_boards/aruco_marker_tree.usd \
  --checkpoint /models/VGGT-Omega-1B-512/model.pt \
  --json-output /data/calibration_result.json
```

폴더는 하위 폴더까지 검색하며 `jpg`, `jpeg`, `png`, `bmp`, `tif`, `tiff`,
`webp`를 읽습니다. 이미지별 intrinsic은 `<image>.json` 또는
`<stem>.camera.json` sidecar에서 읽습니다.

`camera_setup.py capture`로 새 사진을 만들면 RTSP 주소나 계정은 제외하고
`camera/<번호>`, camera matrix, distortion, 교정 해상도만 담은
`<stem>.camera.json`을 사진 옆에 자동 저장합니다. 예전에 캡처해서 sidecar가
없는 `camera_<번호>_*` 사진은 `--camera-config`로 edge-local
`cameras.local.yaml`을 명시하면 해당 번호의 intrinsic을 읽어 사용할 수 있습니다.
설정의 RTSP 필드는 manifest나 결과 JSON으로 복사하지 않습니다.

```json
{
  "camera_id": "north-gate",
  "camera_matrix": [[1200, 0, 960], [0, 1200, 540], [0, 0, 1]],
  "distortion_coefficients": [-0.1, 0.02, 0, 0, 0],
  "image_size": [1920, 1080]
}
```

`image_size`가 현재 사진 크기와 다르면 camera matrix를 현재 해상도에 맞춰
자동 스케일링합니다.

sidecar와 `--camera-config`가 모두 없으면 해당 사진 해상도 기반 pinhole과 zero
distortion을 가정하고 터미널에 표시합니다. 이 값은 실행 편의를 위한 근사치이며
렌즈 왜곡을 보정할 수 없으므로 정밀 pose 결과로 취급하면 안 됩니다. reference
영상도 실제 카메라 촬영물이라면 같은 sidecar 규칙으로 intrinsic/distortion을
제공해야 합니다. 중간 이미지와 검출 artifact는 임시 디렉터리에서 처리 후
삭제하며, 최종 JSON만 남깁니다. `--json-output`을 생략하면 현재 디렉터리에
`calibration_result_<시각>.json`으로 저장합니다. 오프라인 모드는
`edges.json`이나 edge의 `cameras.local.yaml`을 수정하지 않습니다.

### 결과 카메라 pose USD 시각화

결과 JSON의 `camera_to_world`와 intrinsic으로 OpenCV 카메라의 +Z 전방을 나타내는
wireframe 사각뿔을 생성할 수 있습니다. 출력 Stage는 Z-up·미터 단위이며 결과에
기록된 ArUco marker tree도 함께 reference합니다.

```bash
python apps/calibration_worker/export_camera_frustums.py \
  apps/calibration_worker/calibration_result_20260813-150010.json \
  --frustum-depth-m 2.0
```

기본 출력은 입력 JSON 옆의 `<result_stem>_cameras.usda`입니다. 카메라가 멀리
떨어져 있어 사각뿔이 작게 보이면 `--frustum-depth-m 5`처럼 시각화 길이만
키울 수 있으며 pose 값 자체는 바뀌지 않습니다. `--no-marker-tree`를 지정하면
카메라와 월드 축만 생성합니다.

## 입력 manifest

이미지와 영상 경로는 JSON 파일 위치를 기준으로 해석됩니다. 모든 카메라의 내부
파라미터와 렌즈 왜곡값은 필수입니다.

```json
{
  "cctv_cameras": [
    {
      "camera_id": "north_gate_01",
      "image_path": "inputs/north_gate_01.jpg",
      "camera_matrix": [
        [1820.1, 0.0, 960.0],
        [0.0, 1818.7, 540.0],
        [0.0, 0.0, 1.0]
      ],
      "distortion_coefficients": [-0.21, 0.08, 0.0, 0.0, -0.01]
    }
  ],
  "reference_video": {
    "video_path": "inputs/reference_phone.mp4",
    "camera_matrix": [
      [1450.0, 0.0, 960.0],
      [0.0, 1450.0, 540.0],
      [0.0, 0.0, 1.0]
    ],
    "distortion_coefficients": [-0.10, 0.03, 0.0, 0.0, 0.0],
    "sample_count": 40
  }
}
```

## 실행

```bash
python apps/calibration_worker/inference.py \
  --config apps/calibration_worker/calibration_input.json \
  --checkpoint /models/VGGT-Omega-1B-512/model.pt \
  --marker-tree apps/isaac_sim_client/aruco_boards/aruco_marker_tree.usd \
  --output-dir results/calibration_001 \
  --max-images 48
```

`--max-images`는 CCTV 이미지와 스마트폰 샘플 프레임을 합친 개수입니다. 생략하면
현재 GPU 여유 메모리와 공개된 512px 메모리 측정값으로 상한을 추정합니다. 실제
GPU에서 OOM이 발생하면 값을 낮춰 다시 실행합니다.

## 엣지 patch token 입력

분산 모드에서는 CCTV 원본 대신 엣지가 만든 DINO patch token bundle을 받습니다.
서버 manifest에는 CCTV 항목을 넣지 않고 서버가 보유한 참조 영상만 정의합니다.

```json
{
  "reference_video": {
    "video_path": "inputs/reference_phone.mp4",
    "camera_matrix": [[1450, 0, 960], [0, 1450, 540], [0, 0, 1]],
    "distortion_coefficients": [-0.1, 0.03, 0, 0, 0],
    "sample_count": 40
  }
}
```

```bash
python apps/calibration_worker/inference.py \
  --config reference_input.json \
  --feature-bundle results/edge-001_features.npz \
  --checkpoint /models/VGGT-Omega-1B-512/model.pt \
  --output-dir results/calibration_001
```

엣지와 서버는 동일한 체크포인트를 사용해야 하며 SHA-256이 다르면 추론을
거부합니다. 분산 전처리는 왜곡 제거 후 512×512 RGB letterbox로 고정됩니다.
bundle은 `allow_pickle=False`로 읽고 FP16 patch token과 JSON 메타데이터만
포함합니다. RTSP 주소와 계정, 원본 프레임은 포함하지 않습니다.

## 출력

- `undistorted/cctv`: 왜곡 제거된 CCTV 이미지
- `undistorted/reference`: 균등 샘플링 후 왜곡 제거된 스마트폰 프레임
- `aruco_detections`: VGGT 입력 크기에서 검출된 marker 확인 이미지
- `calibration_result.json`: 좌표 정합 품질과 CCTV별 `world_to_camera`,
  `camera_to_world`, 위치, intrinsic/distortion 및 엣지 카메라 메타데이터

`alignment.vggt_to_usd`는 VGGT 복원 좌표를 marker-tree USD의 미터 좌표로 옮기는
4×4 similarity matrix입니다.

출력 extrinsics는 OpenCV 카메라 좌표계(`+X` 오른쪽, `+Y` 아래, `+Z` 전방)를
사용하고, world 좌표는 marker tree USD에 저장된 미터 단위 좌표입니다. 기준
영상에서 USD에 등록된 marker가 선명하게 보여야 하며, 정합 오차는 결과 JSON의
`alignment.rmse_m`와 `alignment.max_error_m`로 확인합니다.
