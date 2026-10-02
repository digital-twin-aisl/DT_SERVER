# 녹화 영상 + RootNet B + 새 v2 캘리브레이션 시각화

**개선된 RootNet과 PoseNet을 함께 적용해 0812_1/2/3을 JSONL로 저장하는 실행**은
[실제 영상 출력 녹화 안내](recordings_20260928.md)를 참고하세요. 아래 내용은 기존
수동 실행 경로이며, 새 녹화 스크립트는 현재 사용자가 수정한 프로파일을 덮어쓰지 않습니다.

기존 운영 설정을 바꾸지 않는 **수동 진단 실행**입니다. 합성 데이터로 학습한 B의
녹화 RGB 영상에서 확인하는 용도이며, 실환경 정확도가 검증된 배포는 아닙니다.
새로 학습한 부분은 RootNet뿐이고 backbone/PoseNet은 원본 SelfPose3D 가중치입니다.
**RTSP 실시간 입력이 아닙니다.** 영상을 순서대로 추론해서 Zenoh로 발행하며,
실제 재생 속도는 처리 성능에 따라 달라집니다(원본 29.97 FPS 실시간 재생 보장 없음).

## 고정 설정

- 가중치: `apps/edge_client/models/POC_posenet.pth.tar` (엣지/서버 동일, git에 포함되지 않음).
- 캘리브레이션: 현재 `deployment.json`은 `../calibration_result_20260814-123941.json`을
  참조합니다. 아래 v2 snapshot(`calibration.from_cameras_v2.json`)으로 되돌릴 수 있습니다.
- v2 snapshot 출처:
  원본 `calibration_result_simplified_cli_smoke.json`이 현재 작업 폴더에서 없어져,
  이전에 변환한 `cameras.local_v2.yaml`의 8대 K/distortion/extrinsic을 그대로
  추출한 실행용 snapshot입니다. 원본 파일 전체를 복구한 것은 아니며, 재추정하거나
  이전 캘리브레이션을 사용하지 않았습니다. RTSP 주소/자격 증명은 포함하지 않습니다.
- edge_1 영상: `apps/edge_client/data/data_0812_1_edge_1/camera_{2,4,6,8}.mkv`.
- edge_2 영상: `apps/edge_client/data/data_0812_1_edge_2/camera_{1,3,5,7}.mkv`.
- edge_1: 카메라 **2, 4, 6, 8** / edge_2: **1, 3, 5, 7**, 기존 AOI/Ground 유지.
- PyTorch 실행, TensorRT 끔, 엣지 Re-ID 끔, 서버 root tracking fallback 켬.
  모든 유효 root에 LOD 2를 지정해 PoseNet을 실행합니다. ID는 위치 기반 임시 ID라
  교차/가림 시 바뀔 수 있고, Re-ID 품질 평가용 설정이 아닙니다.
- 입력 토픽: `dt/rootnet-v2/edges/{edge_id}/inference`.
- Isaac 출력 토픽: `meta-sejong/rootnet-v2/scene/v1`.
- 서버 `input_clock=dataset`: Unix 시각 대신 `frame_index / fps`를 사용합니다.
  `input_mode=independent`이므로 두 엣지의 재생 시점을 자동으로 맞추지는 않습니다.

영상에 동봉된 `calibration_result_20260814-123941.json`과 새 v2는 서로 다릅니다.
카메라 위치 차이는 약 0.33~1.26m입니다. 사용자 선택에 따라 **새 v2를 적용하는
시험**으로 구성했고, 두 엣지 프로파일에 `allow_dataset_calibration_override=true`를
명시했습니다. 시작 로그에 동봉/적용 캘리브레이션 경로와 SHA-256 및 경고가 나옵니다.
기본 불일치 차단은 다른 실행에서 그대로 유지됩니다. 영상 폴더와 동봉 JSON은
수정하지 않습니다. 결과의 오차에는 모델뿐 아니라 캘리브레이션 불일치도 섞입니다.

## 1. 각 머신 준비

서버와 두 엣지에 같은 버전의 저장소를 두고 추론용 Python 환경을 활성화합니다.
`run.sh`는 현재 환경의 `python3`를 사용합니다. 다른 인터프리터는
`PYTHON_BIN=/절대경로/환경/bin/python bash .../run.sh ...`로 지정할 수 있습니다.
`dt_common` 소스 경로는 스크립트가 설정하지만 PyTorch/CUDA 등 의존성은 설치되어
있어야 합니다. 기존 설치 방법은 `docs/guide/02-install-server.md`, `apps/edge_client/README.md` 참고.

다음 파일은 Git에서 제외될 수 있으므로 **서버·두 엣지에 별도로 동일하게 배치**합니다.

```text
SelfPose3d/output_root_robustness/pilot_v2/B_export.pth.tar
apps/deployments/rootnet_v2/calibration.from_cameras_v2.json
Ground.usd
apps/deployments/cache/scene_0812_2_ground.npz
```

각 엣지에는 위에 지정한 자기 영상 폴더(동봉 JSON 포함)가 필요합니다. 이번 입력은
로컬 MKV이므로 RTSP 연결이나 `cameras.local_v2.yaml`은 실행에 필요하지 않습니다.
서버에는 영상이 필요 없습니다. 원본 가중치/전체 학습 데이터도 필요하지 않습니다.
머신 간 공통 파일 일치는 아래 해시를 비교할 수 있습니다.

```bash
sha256sum SelfPose3d/output_root_robustness/pilot_v2/B_export.pth.tar \
  apps/deployments/rootnet_v2/calibration.from_cameras_v2.json Ground.usd \
  apps/deployments/cache/scene_0812_2_ground.npz
```

기존 agent/manager가 동일 엣지 추론을 실행 중이라면 먼저 운영 절차로 해당 추론을
중지하십시오. 이 스크립트는 서비스를 자동 중지하거나 운영 identity를 덮어쓰지
않습니다. 네트워크 토픽은 분리되지만 GPU 자원은 공유합니다.

## 2. 설정 사전 검사

아래 예시는 각 머신의 저장소 루트에서 실행합니다. `SERVER_IP`는 **엣지와 Isaac
머신에서 접속 가능한 Zenoh router IP**로 바꾸십시오. 엣지에서 `127.0.0.1`을 쓰면
서버가 아니라 엣지 자신을 가리킵니다. 기본 TCP 포트는 7447이며 실제 router와
일치해야 합니다.

```bash
# 서버
bash apps/deployments/rootnet_v2/run.sh server tcp/127.0.0.1:7447 --validate-only

# 첫 번째 엣지
bash apps/deployments/rootnet_v2/run.sh edge_1 tcp/SERVER_IP:7447 --validate-only

# 두 번째 엣지
bash apps/deployments/rootnet_v2/run.sh edge_2 tcp/SERVER_IP:7447 --validate-only
```

`Configuration validation passed`를 확인합니다. 이 검사는 설정/파일/카메라 순서/
workspace 구성을 확인하고 첫 영상의 FPS를 읽습니다. 전체 영상 디코딩·Zenoh 접속이나
GPU 모델 실행을 시험하지 않습니다. 명시적으로 허용한 캘리브레이션 경고는 정상입니다.

프로파일 자체의 회귀 검사(카메라/네트워크 접속 없음)는 다음과 같습니다.

```bash
python3 -m unittest discover -s apps/deployments/rootnet_v2/tests -v
```

## 3. router → 서버 → 엣지 순서로 실행

각 명령은 별도 터미널에서 계속 실행합니다. 이미 같은 TCP endpoint의 router가
실행 중이면 재사용하고 중복 실행하지 마십시오. 신뢰하는 내부 네트워크에서만
router 포트를 허용합니다.

```bash
# 서버 터미널 1: router가 없을 때만
zenohd -c apps/edge_manager/config/zenoh-router-poc.json5

# 서버 터미널 2
bash apps/deployments/rootnet_v2/run.sh server tcp/127.0.0.1:7447

# 엣지 1 터미널
bash apps/deployments/rootnet_v2/run.sh edge_1 tcp/SERVER_IP:7447

# 엣지 2 터미널
bash apps/deployments/rootnet_v2/run.sh edge_2 tcp/SERVER_IP:7447
```

서버는 두 엣지를 독립적으로 처리하므로 먼저 엣지 하나만 켜도 확인할 수 있습니다.
각 엣지는 자기 네 영상 중 가장 짧은 영상이 끝나면 종료합니다(현재 약 47초 분량).
서버는 입력을 기다리며 남고 Isaac에는 마지막 결과가 남을 수 있으므로 Ctrl+C로
서버를 종료합니다. **처음부터 다시 재생할 때는 서버도 재시작**해야 이전 프레임의
timestamp/sequence 상태 때문에 새 재생 패킷이 거부되지 않습니다.
실행 종료는 해당 터미널의 Ctrl+C입니다. 기존 운영 프로파일의 실행 경로는 바꾸지
않았습니다. 실행 로그/통계는 기본 `apps/server_worker/data/metrics/`에 기록됩니다.

두 엣지의 시작 시각/추론 속도가 다르면 서로 다른 영상 시점이 표시되거나 느린
엣지의 사람이 TTL 때문에 사라질 수 있습니다. 첫 기하 확인은 한 엣지씩 수행해도
됩니다. 프레임 단위 동기화/A-B 정량 비교가 필요하면 `--record-output`으로 두 엣지의
패킷을 저장한 후 서버의 `--replay-inputs ... --input-mode strict` 경로를 사용합니다.
온라인 Zenoh에서 `strict`만 켜는 것으로 무손실 동기화가 보장되지는 않습니다.

## 4. Isaac Sim Script Editor

나중에 오프라인으로 보려면 서버 실행에 아래 옵션을 추가합니다.

```bash
--scene-recording apps/server_worker/data/recordings/rootnet_B_v2_001.jsonl
```

저장한 파일은 **Meta Sejong Scene Player** Extension으로 재생할 수 있습니다.
[오프라인 재생 안내](../../isaac_sim_client/exts/meta_sejong.scene_player/docs/README.md)를
참고하세요. 아래 Script Editor 방식은 기존 실시간 Zenoh 수신에 해당합니다.

같은 JSONL 파일은 Isaac 없이 `frontend_api`의 `/viewer` → **JSONL 파일 열기**로도
재생할 수 있습니다. [브라우저 기록 재생 안내](../../frontend_api/README.md#서버-출력-기록과-파일-재생)를
참고하세요. 파일 재생은 기존 실시간 추론/송출을 중지하지 않습니다.

Isaac Sim 전용 Python에 저장소에서 사용하는 `eclipse-zenoh==1.9.0`가 필요합니다.
이미 설치되어 있다면 다시 설치할 필요가 없습니다. **일반 Python으로
`meta_sejong_script.py`를 실행하지 않습니다.**

```bash
# Isaac Sim 설치 디렉터리에서, 설치가 필요한 경우만
./python.sh -m pip install eclipse-zenoh==1.9.0
```

Isaac Sim의 Script Editor에서 아래 코드를 실행합니다. `project`, router 주소,
USD 경로를 **Isaac Sim이 실행되는 머신 기준**으로 바꾸십시오. 서버와 같은
머신이면 router 주소는 `tcp/127.0.0.1:7447`로 둘 수 있습니다.

```python
import os
from pathlib import Path

project = Path("/path/to/DT_SERVER")
os.environ["ISAAC_USD_PATH"] = "/path/to/campus.usd"
os.environ["ISAAC_SCENE_TRANSPORT"] = "zenoh"
os.environ["ISAAC_ZENOH_ENDPOINT"] = "tcp/SERVER_IP:7447"
os.environ["ISAAC_ZENOH_TOPIC"] = "meta-sejong/rootnet-v2/scene/v1"

script = project / "apps/isaac_sim_client/meta_sejong_script.py"
exec(compile(script.read_text(encoding="utf-8"), str(script), "exec"), globals())
```

스크립트는 지정 USD Stage를 엽니다. 현재 Stage의 저장하지 않은 작업은 먼저
저장하십시오. 기존 수신 스크립트는 재실행 시 이전 listener/task를 정리하지 않으므로
**새 Isaac 세션에서 한 번만 실행**합니다. 주소/토픽 변경 후 재시험할 때도 Isaac을
재시작합니다. 전체 건물 USD와 그 참조 자산은 Isaac 머신에만 필요합니다.

정상 로그는 `Zenoh subscriber is ready`, 이어서 `First Zenoh scene received`입니다.
pose가 있으면 15관절 skeleton, root만 있으면 capsule이 표시됩니다.
`/World/MetaSejong_People` 아래에 생성되며, 카메라 시점을 작업 구역으로 이동해야
보일 수 있습니다. 들어오는 좌표는 USD world 기준으로 적용됩니다.

## 안 보일 때 확인 순서

1. **엣지:** 시작 설정의 `dataset: true`, `example_folder`, 카메라 ID와
   `Workspace ready`를 확인합니다. 영상은 1920×1080, 약 29.97 FPS입니다.
   새 CLI 옵션을 모른다는 오류가 나면 엣지의 `inference.py`도 업데이트해야 합니다.
2. **서버:** 입력 토픽이 `dt/rootnet-v2/...`인지, 두 머신의 calibration/workspace
   digest가 일치하는지 확인합니다. 구버전 deployment 패킷은 거부될 수 있습니다.
3. **출력:** Isaac과 별개로 아래 구독기로 `people`, `global_ids`를 확인합니다.

   ```bash
   python3 apps/server_worker/tools/scene_zenoh_subscriber.py \
     --endpoint tcp/127.0.0.1:7447 --topic meta-sejong/rootnet-v2/scene/v1
   ```

4. timestamp domain/freshness 오류라면 서버의 `input_clock=dataset`을 확인합니다.
   `live` 모드에서는 0부터 시작하는 영상 timestamp를 거부합니다. 재생을 처음부터
   다시 시작했다면 서버도 재시작합니다.
5. skeleton이 바닥/벽 밖에 나타나면 새 캘리브레이션의 world 정렬과 실제 기하 오차,
   촬영 동기화, root 오검출을 구분해야 합니다. 시각화 성공만으로 캘리브레이션이나
   모델 정확도가 검증되는 것은 아닙니다.

A/B 비교 시에는 동일 입력·캘리브레이션·threshold·LOD 설정을 유지해야 합니다.
이 프로파일은 B 전용입니다. 운영 `POC_posenet`은 backbone도 달라 엄밀한 A baseline이
아니고, 원본 `cam5_posenet.pth.tar`도 attention 키를 포함하므로 파일 경로만 바꿔
직접 strict-load할 수는 없습니다.
