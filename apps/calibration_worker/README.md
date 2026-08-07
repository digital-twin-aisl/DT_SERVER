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

## 출력

- `undistorted/cctv`: 왜곡 제거된 CCTV 이미지
- `undistorted/reference`: 균등 샘플링 후 왜곡 제거된 스마트폰 프레임
- `aruco_detections`: VGGT 입력 크기에서 검출된 marker 확인 이미지
- `calibration_result.json`: 좌표 정합 품질과 CCTV별 `world_to_camera`,
  `camera_to_world`, 위치 및 내부 파라미터

출력 extrinsics는 OpenCV 카메라 좌표계(`+X` 오른쪽, `+Y` 아래, `+Z` 전방)를
사용하고, world 좌표는 marker tree USD에 저장된 미터 단위 좌표입니다. 기준
영상에서 USD에 등록된 marker가 선명하게 보여야 하며, 정합 오차는 결과 JSON의
`alignment.rmse_m`와 `alignment.max_error_m`로 확인합니다.
