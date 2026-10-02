# Meta Sejong Scene Player

서버의 최종 SceneOutput(root, 15관절 pose, ID, LOD, timestamp)을 JSONL로 저장하고
Isaac Sim에서 오프라인 재생합니다. **MP4 녹화가 아니라 3D 추론 결과 녹화**입니다.
재생 중 카메라·모델·Zenoh·server_worker는 필요하지 않습니다. 과거 metrics 로그만으로는
전체 장면을 복원할 수 없으므로 저장 옵션을 켠 새 추론 실행이 필요합니다.

## 1. 서버 출력 저장

저장소 루트에서 기존 서버 실행 명령에 `--scene-recording`을 추가합니다.

```bash
bash apps/deployments/rootnet_v2/run.sh server tcp/127.0.0.1:7447 \
  --scene-recording apps/server_worker/data/recordings/rootnet_B_v2_001.jsonl
```

이전과 같이 두 엣지를 실행하면 서버가 최종 결과를 저장하면서 Zenoh로도 발행합니다.
Isaac을 켜지 않고 저장만 할 때는 위 명령에 `--no-scene-zenoh`를 추가할 수 있습니다.
엣지 입력 수신에는 여전히 router가 필요합니다.

- 한 줄은 최종 SceneOutput JSON 한 개이며, root/pose는 USD world **mm 단위**입니다.
- 네트워크 송신 큐에 넣기 **전** 결과를 기록하므로 네트워크 드롭과 독립적입니다.
- 사람이 없는 장면도 기록합니다. live 입력에서 사람 TTL이 만료되는 빈 장면도 포함합니다.
- 폴더는 자동 생성하지만 **기존 파일은 덮어쓰거나 이어 쓰지 않습니다.** 재실행할 때
  `...002.jsonl`처럼 새 이름을 사용하십시오. 기본값은 저장 OFF입니다.
- 매 장면을 동기적으로 쓰고 flush합니다. 디스크 I/O 시간이 추가되며, 디스크가 가득
  차는 등 저장 실패 시 불완전한 기록을 조용히 남기지 않고 추론을 오류로 종료합니다.
  flush는 전원 차단에 대한 fsync 내구성 보장은 아닙니다.
- Ctrl+C로 정상 종료한 파일을 복사해 재생하십시오. 녹화 중인 파일을 따라가는 모드는
  아닙니다. 중단된 마지막 JSON 조각은 경고와 함께 제외하고 완성된 장면만 복구합니다.
- 원래 `--viser-debug-output`이나 manager SceneRecorder로 저장한 파일도 동일한
  schema_version=1 / USD world Z-up mm 형식이면 읽을 수 있습니다.

## 2. Extension 활성화

Isaac Sim 4.2의 Extension Manager에서 Extension Search Path에 다음을 추가합니다.

```text
<DT_SERVER>/apps/isaac_sim_client/exts
```

`Meta Sejong Scene Player` (`meta_sejong.scene_player`)를 활성화합니다.
창을 닫았다면 `Window > Meta Sejong Scene Player`로 다시 열 수 있습니다.
이 재생에는 `meta_sejong_script.py --exec`가 필요 없습니다. 혼동을 피하려면
실시간 수신 스크립트가 실행되지 않은 Isaac 세션을 사용하십시오.

## 3. 파일 재생

1. 녹화 좌표계와 일치하는 Z-up USD(예: `2025_SejongUniv_All.usd`)를 먼저 엽니다.
2. 서버에서 저장한 JSONL을 Isaac 머신으로 복사합니다.
3. Extension 상단에 파일 **절대경로**를 입력하고 `Open File`을 누릅니다.
4. 첫 장면이 정지 상태로 표시됩니다. `Play`를 누릅니다.
5. `Pause`, `First`, `Previous`/`Next`, 프레임 슬라이더/숫자 입력으로 탐색합니다.
6. `Speed`로 0.1–8 배속, `Loop`로 반복 재생을 선택합니다. `Unload`는 임시 장면을 제거합니다.

Isaac 기본 폰트의 글자 깨짐을 피하기 위해 UI 문구는 영문 ASCII로 표시합니다.
변경 전 UI가 남아 있으면 Extension을 비활성화한 뒤 다시 활성화하십시오.

배속은 **기록된 장면 timestamp 차이** 기준입니다. dataset timestamp이면 영상 시간,
live timestamp이면 Unix 시간의 차이를 사용합니다. GPU가 느리게 추론한 대기 시간은
별도로 재현하지 않습니다. 같은 timestamp의 여러 장면은 재생 중 마지막 것으로
갱신되지만 이전/다음 버튼으로 모두 확인할 수 있습니다. UI가 느리면 중간 표시 프레임을
건너뛸 수 있으며 보간은 하지 않습니다. 저장 시 이미 생긴 검출 오류/캘리브레이션 오차/
엣지 간 시간 차이를 보정하는 기능은 아닙니다.

pose가 있으면 skeleton, root만 있으면 capsule이 표시됩니다. 사람이 사라진 장면에서는
이전 사람을 숨깁니다. 상단 시간 옆의 인원수가 0이면 정상적인 빈 장면일 수 있습니다.
인원수가 있는데 화면 밖이라면 Stage 목록의 `/MetaSejong_Playback_...` 아래 사람을
선택하고 Viewport를 해당 위치로 이동하십시오.

## USD 안전성과 대용량 파일

- 현재 열린 Stage를 사용하며 다른 USD 파일을 자동으로 열거나 저장하지 않습니다.
- `/MetaSejong_Playback_<고유번호>` 아래에 재생 전용 prim을 만듭니다. mm를 Stage의
  `metersPerUnit`에 맞게 변환하며, 건물 `/World` 변환을 중복 적용하지 않습니다.
- 소유한 익명 session sublayer에만 씁니다. 닫기·Stage 변경·Extension 종료 시 이
  레이어만 제거하므로 기존 USD와 실시간 `/World/MetaSejong_People`은 건드리지 않습니다.
  재생 도중 Stage flatten/export를 하면 임시 geometry도 포함될 수 있습니다.
- 로딩은 백그라운드에서 파일을 검증하고 byte offset/timestamp 인덱스만 보관합니다.
  pose 전체를 메모리에 올리지 않습니다. 로딩 중 Stage가 바뀌면 다시 열어야 합니다.
- 잘못된 단위, 지원하지 않는 pose, timestamp 역행, 중간 JSON 손상은 오류로 표시합니다.
  한 줄 16 MiB, 최대 200만 장면까지 지원합니다. 로딩 이후 파일을 수정하지 마십시오.

## 검증

```bash
python3 -m unittest discover \
  -s apps/isaac_sim_client/exts/meta_sejong.scene_player/tests -v
```

일반 Python에서 저장/로딩/재생 시간/탐색을 테스트하고, `pxr`가 있으면 실제 USD
geometry와 session layer 정리도 검사합니다. UI는 Kit의
[IntSlider](https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/ui/widgets/intslider.html)와
[update event](https://docs.omniverse.nvidia.com/dev-guide/latest/python-snippets/events/subscribe-app-update-event.html)를 사용합니다.

Isaac Sim 설치 폴더에서 새 빈 headless Stage를 사용하는 UI smoke test도 가능합니다.

```bash
./python.sh <DT_SERVER>/apps/isaac_sim_client/exts/meta_sejong.scene_player/tests/run_isaac_smoke.py \
  --/app/settings/loadUserConfig=false --/app/settings/persistent=false
```
