# 1. 빠른 시작: GPU·카메라 없이 장면 재생

이 단계에서는 실제 사람이 찍힌 데이터 없이, 계산으로 만든 합성 장면을 브라우저
뷰어에서 재생합니다. 엔진이 최종적으로 무엇을 내보내는지(`SceneOutput`) 확인하는
것이 목적입니다.

## 준비물

- Linux 또는 macOS, Python 3.10 이상, Node.js 22 이상
- 인터넷 연결 (pip/npm 패키지 설치)

## 1) 설치

```bash
git clone --recurse-submodules <이 저장소 URL> DT_SERVER
cd DT_SERVER
python3 -m venv .venv && . .venv/bin/activate
python -m pip install -e packages/dt_common -r apps/frontend_api/requirements.txt
npm ci --prefix apps/frontend_api/web
npm run build --prefix apps/frontend_api/web
```

`npm run build`는 `apps/frontend_api/app/static/dist/viewer.js`를 생성합니다.

## 2) 합성 장면 만들기 (선택)

저장소에 이미 `examples/synthetic_region/synthetic_scene.jsonl`이 들어 있습니다.
사람 수나 길이를 바꾸려면 직접 생성합니다.

```bash
python examples/synthetic_region/generate_scene.py \
  --output /tmp/demo.jsonl --people 6 --seconds 30
```

모든 장면에는 `runtime.playback=true`가 붙습니다. 운영 시스템은 이 표시가 있는
장면을 실시간 관측으로 집계하지 않습니다.

## 3) 뷰어 실행

```bash
python -m uvicorn apps.frontend_api.app.main:app --host 127.0.0.1 --port 8005
```

Zenoh router가 없다는 `Scene subscriber reconnecting` 로그가 반복되어도 정상입니다.
이 단계에서는 실시간 구독을 사용하지 않습니다.

1. 브라우저에서 <http://127.0.0.1:8005/viewer>를 엽니다.
2. **JSONL 파일 열기**를 누르고 `synthetic_scene.jsonl`을 선택합니다.
   파일은 브라우저 안에서만 읽히며 서버로 업로드되지 않습니다.
3. 재생 막대로 시간을 이동합니다. 일부 사람은 15관절 자세(LOD 2), 나머지는 위치만(LOD 1)
   표시됩니다. 서버가 계산 자원을 나누는 방식을 흉내 낸 것입니다.

캠퍼스 3D 지도(`apps/frontend_api/assets/map.glb`)가 함께 표시됩니다. 지도가 없는
환경에서도 사람 표시와 재생은 동작합니다.

## 4) 장면 파일 구조 보기

```bash
head -n 1 examples/synthetic_region/synthetic_scene.jsonl | python -m json.tool | head -40
```

필드 의미는 [데이터 계약](07-contracts.md#sceneoutput-v1)에 있습니다. 같은 파일은
Isaac Sim의 Scene Player 확장에서도 열 수 있습니다([Isaac Sim 연동](04-isaac-sim.md)).

## 다음 단계

- 실제 카메라로 운영하려면 [서버 설치](02-install-server.md) → [엣지 설치](03-install-edge.md)
- 다른 프로그램에서 장면을 만들거나 구독하려면 `generate_scene.py`의 `dt_common.contracts.scene`
  사용 예를 참고합니다.
