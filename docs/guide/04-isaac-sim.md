# 4. Isaac Sim 연동

서버가 발행하는 `SceneOutput`을 NVIDIA Isaac Sim의 USD 스테이지에 반영합니다. 브라우저
뷰어만 쓸 경우 이 단계는 필요 없습니다. 상세 내용은
[apps/isaac_sim_client/README.md](../../apps/isaac_sim_client/README.md)에 있습니다.

## 구성요소

| 구성 | 역할 |
| --- | --- |
| `meta_sejong_script.py` | Zenoh로 장면을 구독해 `/World/MetaSejong_People/Person_<id>`를 생성·이동 |
| `exts/meta_sejong.scene_player` | 기록된 JSONL을 재생 (Zenoh·모델 불필요) |
| `exts/meta_sejong.aruco_board` | 보정용 ArUco 마커 트리 USD 생성 |
| `exts/meta_sejong.multi_camera_recorder` | 보정 카메라 시점 MP4 동시 녹화 |

## 1) Isaac Sim Python에 의존성 설치

Isaac Sim은 자체 Python(`python.sh`)과 `omni`/`pxr` ABI를 사용합니다. 서버 venv가 아니라
Isaac Sim 설치 디렉터리에서 설치합니다.

```bash
export DT_SERVER_DIR=/path/to/DT_SERVER
export ISAAC_SIM_DIR=/path/to/isaac-sim-standalone-4.2.0
cd "$ISAAC_SIM_DIR"
./python.sh -m pip install -r "$DT_SERVER_DIR/requirements-isaac-sim.txt"
./python.sh -c "import omni.usd, zenoh; from pxr import Usd; print('Isaac Python: OK')"
```

## 2) 실시간 구독

```bash
export ISAAC_USD_PATH=/path/to/your_stage.usd   # 생략 시 저장소의 Ground.usd
export ISAAC_ZENOH_ENDPOINT=SERVER_IP:7447
export ISAAC_ZENOH_TOPIC=meta-sejong/scene/v1   # 구역의 scene_topic
cd "$ISAAC_SIM_DIR"
./isaac-sim.headless.webrtc.sh --exec "$DT_SERVER_DIR/apps/isaac_sim_client/meta_sejong_script.py"
```

- 스크립트는 스테이지의 `metersPerUnit`(개발 현장은 0.01=cm)을 읽어 `SceneOutput`의 mm 좌표를
  스테이지 단위로 변환합니다. 스테이지의 world 좌표계가 보정 결과와 같은 USD world(Z-up)여야 합니다.
- 캠퍼스 전체 USD는 이 저장소에 포함되지 않습니다. 자기 현장의 USD를 지정하고, 상대 경로로
  참조하는 텍스처·하위 USD도 같은 구조로 복사합니다.
- 원격 WebRTC 스트리밍과 VPN 구성은 Isaac 클라이언트 README의 원격 실행 절차를 따릅니다.

## 3) 기록 재생

Isaac Sim에서 **Meta Sejong Scene Player** 확장을 켜고 JSONL 파일
(예: `examples/synthetic_region/synthetic_scene.jsonl`)을 엽니다. 사용법은
[Scene Player 문서](../../apps/isaac_sim_client/exts/meta_sejong.scene_player/docs/README.md)에 있습니다.
