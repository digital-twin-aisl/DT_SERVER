# Edge camera setup

저장소 루트에서 대화형 등록 스크립트를 실행합니다.

```bash
python apps/edge_client/camera_setup.py
```

실행 시 기존 카메라 목록을 먼저 보여준 뒤 새 카메라 등록을 시작합니다. 목록만
확인하려면 다음과 같이 실행합니다.

```bash
python apps/edge_client/camera_setup.py list
```

스트리밍을 직접 보려면 목록 뒤에 번호를 입력합니다. `0`은 전체 카메라를
각각의 창으로 표시하며, `q` 또는 `Esc`로 종료합니다. 선택한 카메라별
`probe_rtsp` 결과(코덱·해상도·FPS)도 터미널에 출력됩니다.

```bash
python apps/edge_client/camera_setup.py show
```

한 장씩 JPEG로 캡처하려면 다음을 실행합니다. `0`을 입력하면 모든 카메라를
캡처하며 결과는 기본적으로 `apps/edge_client/data/captures`에 저장됩니다.
이 경우에도 카메라별 `probe_rtsp` 결과가 먼저 터미널에 출력됩니다.

```bash
python apps/edge_client/camera_setup.py capture
```

연결 가능한 카메라를 모두 녹화하려면 `record`를 사용합니다. 등록 목록을 출력한 뒤
각 카메라의 연결을 확인하여 연결 가능한 목록만 다시 보여주며, 사용자 확인 후 녹화를
시작합니다. 관제 창에는 카메라 화면이 행렬로 배치되고 카메라 번호와 녹화 시간이
표시됩니다. `q`, `Esc` 또는 `Ctrl+C`로 종료합니다.

```bash
python apps/edge_client/camera_setup.py record
```

원본 RTSP 비디오 패킷은 재인코딩하지 않고 카메라별 MKV 파일로 저장되며, 기본 저장
위치는 `apps/edge_client/data/recordings/<시작시각>/`입니다. 관제 화면만 기본 640x360,
5 FPS로 디코딩하므로 Jetson 부하를 낮춥니다. 필요하면 더 낮출 수 있습니다.

```bash
python apps/edge_client/camera_setup.py record \
  --preview-width 480 --preview-height 270 --preview-fps 3
```

`config/edge.local.json`에 유효한 `edge_id`가 있어야 합니다. 파일 또는 `edge_id`가
없다면 먼저 다음을 실행해야 합니다.

```bash
python apps/edge_client/agent.py init
```

카메라는 숫자 번호로 등록되며 내부 키는 `camera/<번호>` 형식입니다. 외부 서비스에
노출할 때만 `edge_id/camera/<번호>` 형식으로 edge ID를 앞에 붙입니다. 이후 카메라
이름, RTSP 주소와 인증정보, 위치 및 Twin ID를 순서대로 입력합니다.
저장 전 로컬에서 RTSP 첫 프레임을 읽고 코덱, 해상도 및 FPS를 확인합니다.
첫 프레임을 읽지 못하면 저장 여부를 묻고, 명시적으로 동의한 경우에만 저장합니다.
결과는 엣지 로컬의 다음 파일에 저장되고 `inference.py`가 자동으로 읽습니다.

```text
apps/edge_client/config/cameras.local.yaml
```

Intrinsic을 나중에 입력하거나 체커보드로 측정하려면 다음 명령을 사용합니다.

```bash
python apps/edge_client/camera_setup.py intrinsic
```

직접 입력 시 `fx fy cx cy`, OpenCV 순서의 distortion coefficients, 보정 영상 크기를
저장합니다. 체커보드 측정은 내부 코너 수, 한 칸 크기와 촬영 장수를 입력받습니다.
분산 캘리브레이션이 완료되면 같은 카메라 항목의 `extrinsic`에 USD 좌표계 pose가
저장됩니다.

카메라를 새로 등록하거나 intrinsic을 변경하면 현재 등록 목록을
`dt/edges/{edge_id}/cameras` 토픽으로 즉시 동기화합니다. 서버로 보내는 값은
카메라 번호, 등록 존재 여부, 엣지에서 카메라 RTSP 포트까지의 ping 결과와
intrinsic/extrinsic/distortion calibration뿐입니다. RTSP URL과 인증정보, 이름,
위치, Twin ID는 로컬 설정을 벗어나지 않습니다. 서버 주소는 edge identity에
저장된 값을 사용하며 필요하면 `--endpoint SERVER_IP:7447`로 덮어쓸 수 있습니다.

이 파일은 Git에서 제외되고 `0600` 권한으로 생성되지만 암호화되지는 않습니다.
