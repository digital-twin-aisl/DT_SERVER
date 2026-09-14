import asyncio
import os
import httpx
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pathlib
from contextlib import asynccontextmanager, suppress
import json

from .viewer_config import assets_dir, viewer_config
from .scene_bridge import SceneBridge

@asynccontextmanager
async def lifespan(app):
    bridge = SceneBridge(
        os.getenv("SCENE_ZENOH_ENDPOINT", "tcp/127.0.0.1:10020"),
        os.getenv("SCENE_ZENOH_TOPIC", "meta-sejong/scene/v1"),
        viewer_config()["stale_seconds"],
    )
    app.state.scene_bridge = bridge
    await bridge.start()
    try:
        yield
    finally:
        await bridge.close()


app = FastAPI(title="Frontend API Gateway", lifespan=lifespan)

# 프론트엔드(React, Vue 등) 브라우저에서 직접 API를 찌를 수 있도록 CORS 허용 세팅
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 배포 환경에서는 실제 도메인으로 대체 요망
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 웹 브라우저 제공 설정
static_dir = pathlib.Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# 내부 Docker 컨테이너 주소 (docker-compose의 서비스명 기준, 기본 컨테이너 포트 80 사용)
CAMERA_MANAGER_URL = os.getenv("CAMERA_MANAGER_URL", "http://camera_manager:80")
EDGE_MANAGER_URL = os.getenv("EDGE_MANAGER_URL", "http://edge_manager:80")


@app.get("/")
def read_root():
    # 웰컴 페이지 대신 메인 대시보드(SPA) 제공
    index_file = pathlib.Path(__file__).parent / "static" / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "message": "Frontend API Gateway is up. Static dashboard not found.",
        "docs": "Access /docs for Swagger UI"
    }

@app.get("/api/v1/system/status")
async def get_system_status():
    """간단한 시스템 헬스체크 및 통합 상태 정보"""
    # 실제로는 Redis에 각 모듈의 상태를 핑 쳐서 취합하거나, 각 모듈의 루트("/")를 찔러서 확인합니다.
    return {
        "edge_manager": "online",
        "dl_worker": "online",
        "camera_manager": "online",
        "sim_backend": "online"
    }


@app.get("/api/v1/viewer/config")
async def get_viewer_config():
    """Same-origin browser renderer configuration."""
    return viewer_config()


@app.get("/viewer")
async def browser_viewer():
    return FileResponse(static_dir / "viewer.html")


@app.get("/healthz")
async def health():
    return {"status": "ok", "renderer": "threejs"}


@app.get("/api/v1/viewer/status")
async def get_viewer_status():
    return {**app.state.scene_bridge.status(),
            "renderer": "threejs", "map_ready": (assets_dir() / "map.glb").is_file()}


@app.get("/assets/{name}")
async def get_map_asset(name: str):
    if name not in {"map.glb", "map.json"}:
        raise HTTPException(404)
    path = assets_dir() / name
    if not path.is_file():
        raise HTTPException(404, "Map is not exported. Run tools/export_map.py first.")
    return FileResponse(path, media_type="model/gltf-binary" if name.endswith(".glb") else "application/json")


@app.websocket("/ws/scene")
async def scene_socket(websocket: WebSocket):
    # Viewer streams are read-only. Reject cross-origin browser subscriptions.
    origin = websocket.headers.get("origin")
    if origin:
        from urllib.parse import urlsplit
        if urlsplit(origin).netloc != websocket.headers.get("host"):
            await websocket.close(code=1008)
            return
    await websocket.accept()
    bridge = app.state.scene_bridge
    queue = bridge.subscribe()

    async def send():
        while True:
            try:
                payload = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                payload = json.dumps({"type": "status", **bridge.status()})
            await asyncio.wait_for(websocket.send_text(payload), timeout=5.0)

    async def receive():
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect": return

    tasks = [asyncio.create_task(send()), asyncio.create_task(receive())]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks: task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        bridge.clients.discard(queue)
        with suppress(RuntimeError, OSError):
            await websocket.close()

@app.get("/api/v1/edges")
async def get_edges_list():
    """Camera Manager로부터 엣지 및 카메라 메타데이터 데이터베이스 목록을 가져옵니다."""
    async with httpx.AsyncClient() as client:
        try:
            # camera_manager 컨테이너의 내부망 포트 80으로 요청
            response = await client.get(f"{CAMERA_MANAGER_URL}/api/edges", timeout=5.0)
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Camera Manager 연동 에러: {e}")

@app.post("/api/v1/control/calibration")
async def request_calibration_trigger(edge_id: str, camera_id: int):
    """
    Frontend 사용자가 '캘리브레이션 시작' 버튼을 눌렀을 때, 
    요청을 일관된 포멧으로 래핑하여 Camera Manager 쪽으로 푸시 (Proxy) 합니다.
    """
    async with httpx.AsyncClient() as client:
        try:
            # camera_manager 컨테이너로 파라미터 전달 및 Proxy 호출
            response = await client.post(
                f"{CAMERA_MANAGER_URL}/api/calibration/request",
                params={"edge_id": edge_id, "camera_id": camera_id},
                timeout=5.0
            )
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Camera Manager 제어 에러: {e}")
