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

이 파일은 Git에서 제외되고 `0600` 권한으로 생성되지만 암호화되지는 않습니다.
