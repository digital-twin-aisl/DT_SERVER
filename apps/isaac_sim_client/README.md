# Meta Sejong Isaac Sim 연동 클라이언트 (Python Extension)

`meta_sejong_script.py` 스크립트는 DT_SERVER의 `sim_backend`에서 발송하는 WebSocket 이벤트(`{"action": "UpdateTransforms", ...}`)를 실시간으로 구독하여 Isaac Sim (NVIDIA Omniverse) 환경 내 `Meta Sejong` 디지털 트윈에 현실 사람을 반영(Spawn & Translate)하는 코드입니다.

## 작동 원리
* `websockets` 라이브러리를 통해 비동기적으로 실시간 포지션/포즈/메쉬 파라미터를 수신합니다.
* 수신된 `track_id`에 매치되는 USD Prim(임시 캡슐 형상)을 `UsdGeom` API를 사용해 생성(Spawn) 또는 이동(Transform) 시킵니다.

## 시작하기 전에
Isaac Sim에 `websockets` 라이브러리를 설치해야 합니다.
```bash
./python.sh -m pip install websockets
```

## [중요] WebRTC Headless 렌더링 및 스크립트 자동 실행 (Isaac Sim 4.2.0 기준)
웹 브라우저(Frontend API 대시보드)에서 Isaac Sim의 극실사 화면을 직접 보고(`Pixel Streaming`) 마우스/키보드로 제어하려면, Isaac Sim을 **Headless WebRTC 모드**로 구동해야 합니다.

터미널(Linux)을 열고 Isaac Sim 설치 경로로 이동한 뒤, 아래 명령어를 실행하면 **씬 로드, WebRTC 활성화, Python 스크립트 실행이 한 번에 자동화**됩니다.

```bash
./isaac-sim.sh --no-window \
    --enable omni.services.streamclient.webrtc \
    --exec /home/dojan/DT_SERVER/isaac_sim_client/meta_sejong_script.py \
    --/app/window/hideUi=true \
    --open ~/All/2025_SejongUniv_All.usd
```
3. 웹 브라우저에서 서버의 프론트엔드(`http://<서버IP>:8005/`)로 접속합니다.
4. "3D 트윈 뷰어" 탭 내부에 **Isaac Sim의 화면이 Iframe(포트 8211)으로 스트리밍**되어 나타납니다.
5. 웹 상에서 클릭 및 마우스 드래그를 하면 그 제어값이 백그라운드의 Isaac Sim 뷰포트에 전달됩니다.
