# Meta Sejong Multi-Camera Recorder

Isaac Sim에서 현재 활성 Viewport와 캠퍼스 보정 카메라 8대를 동시에 녹화하는
Extension입니다. `녹화 시작` 시 9개의 RenderProduct가 동기화되어 캡처되고,
`녹화 종료 및 저장`을 누르면 각 스트림의 MP4 컨테이너를 마무리합니다.

## 출력

한 번 녹화할 때 다음 디렉터리가 생성됩니다.

```text
apps/isaac_sim_client/recordings/recording_YYYYMMDD_HHMMSS/
├── viewport.mp4
├── camera_1.mp4
├── camera_2.mp4
├── ...
├── camera_8.mp4
└── recording.json
```

`recording.json`에는 시작/종료 시각, 설정 해상도와 FPS, 각 MP4의 프레임 수와
드롭 수, 녹화에 사용한 카메라 행렬/pose가 기록됩니다. 녹화 중에는
`*.part.mp4`에 쓰고 정상적으로 종료된 파일만 최종 `*.mp4` 이름으로 바뀝니다.

카메라 2/4/6/8 pose는
`apps/edge_client/config/cameras.local.yaml`에서 하드코딩했습니다. 현재 로컬
YAML은 edge 1의 짝수 카메라 4대만 포함하므로, 총 9개 화면 요구사항을 맞추기
위해 카메라 1/3/5/7은 같은 배포의
`apps/deployments/calibration_result_20260814-123941.json` 값을 함께
하드코딩했습니다.

OpenCV extrinsic의 카메라 축(`+X` 오른쪽, `+Y` 아래, `+Z` 전방)은 USD 카메라
축(`+X` 오른쪽, `+Y` 위, `-Z` 전방)으로 변환됩니다. 생성한 Camera Prim은 현재
Stage의 익명 session layer에만 존재하고 녹화 종료 시 제거되므로 원본 캠퍼스
USD는 변경되거나 저장되지 않습니다.

## 설치

Isaac Sim 4.2 Extension Manager의 Extension Search Path에 다음 경로를
추가합니다.

```text
<DT_SERVER>/apps/isaac_sim_client/exts
```

`Meta Sejong Multi-Camera Recorder`를 검색해 활성화합니다. 창을 닫은 뒤에는
상단 `Window > Meta Sejong Multi-Camera Recorder`에서 다시 열 수 있습니다.

이 Extension은 MP4 인코딩에 시스템 `ffmpeg`와 `libx264`를 사용합니다. Isaac Sim을
실행하는 호스트에서 다음 명령이 성공해야 합니다.

```bash
ffmpeg -hide_banner -encoders | grep libx264
```

Ubuntu에서 없다면 설치합니다.

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

## 사용

1. 녹화할 USD Stage를 열고 Viewport를 원하는 위치/방향으로 맞춥니다.
2. 저장 폴더, 해상도, FPS를 확인합니다.
3. `녹화 시작`을 누릅니다.
4. 녹화를 끝낼 때 `녹화 종료 및 저장`을 누릅니다.
5. 상태가 `9개 MP4 저장 완료`로 바뀐 뒤 출력 디렉터리를 확인합니다.

기본값은 9개 동시 렌더링 부하를 고려해 `1280x720 / 10 FPS`입니다. 해상도와
FPS를 올리면 GPU 렌더링 및 CPU H.264 인코딩 부하가 크게 증가합니다. 인코더가
처리 속도를 따라가지 못할 때 Isaac Sim UI를 멈추지 않도록 해당 스트림의 프레임을
드롭하며, 수치는 `recording.json`에 남습니다.

활성 Viewport는 녹화 시작 시 선택된 카메라 Prim을 사용합니다. 자유 시점 카메라를
녹화 중 움직이면 같은 Prim의 갱신된 화면이 기록되지만, 녹화 도중 Viewport 자체를
다른 Camera Prim으로 전환해도 녹화 대상은 시작 시 카메라로 유지됩니다.
