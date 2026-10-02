# SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
# SPDX-License-Identifier: LGPL-2.1-or-later
import asyncio
import os
import httpx
from urllib.parse import quote
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pathlib
from contextlib import asynccontextmanager, suppress
import json

from .viewer_config import assets_dir, viewer_config
from .scene_bridge import SceneBridge
from .control_proxy import router as control_router, manager_url

@asynccontextmanager
async def lifespan(app):
    bridge = SceneBridge(
        os.getenv("SCENE_ZENOH_ENDPOINT", "tcp/127.0.0.1:7447"),
        os.getenv("SCENE_ZENOH_TOPIC", "meta-sejong/scene/v1"),
        viewer_config()["stale_seconds"],
    )
    app.state.scene_bridge = bridge
    app.state.region_bridges = {}
    app.state.bridge_lock = asyncio.Lock()
    await bridge.start()
    try:
        yield
    finally:
        await bridge.close()
        for regional in app.state.region_bridges.values():
            await regional.close()


async def region_bridge(region=None):
    if region is None:
        return app.state.scene_bridge
    headers = {}
    if os.getenv('DT_MANAGER_TOKEN'):
        headers['authorization'] = 'Bearer ' + os.environ['DT_MANAGER_TOKEN']
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f'{manager_url()}/regions/{quote(region, safe="")}', headers=headers)
            response.raise_for_status()
            topic = response.json()['scene_topic']
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        raise HTTPException(503, '구역의 장면 설정을 가져올 수 없습니다.') from exc
    async with app.state.bridge_lock:
        current = app.state.region_bridges.get(region)
        if current is not None and current.topic != topic:
            await current.close()
            current = None
        if current is None:
            current = SceneBridge(os.getenv('SCENE_ZENOH_ENDPOINT', 'tcp/127.0.0.1:7447'), topic,
                                  viewer_config()['stale_seconds'])
            await current.start()
            app.state.region_bridges[region] = current
        return current


app = FastAPI(title="Frontend API Gateway", lifespan=lifespan)
app.include_router(control_router)

# 관리 기능과 뷰어는 같은 origin에서 제공한다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 웹 브라우저 제공 설정
static_dir = pathlib.Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

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

@app.get("/api/v1/viewer/config")
async def get_viewer_config(region: str | None = None):
    """Same-origin browser renderer configuration."""
    config = viewer_config()
    if region:
        await region_bridge(region)
        config['websocket_path'] += '?region=' + quote(region, safe='')
    return config


@app.get("/viewer")
async def browser_viewer():
    return FileResponse(static_dir / "viewer.html")


@app.get("/healthz")
async def health():
    return {"status": "ok", "renderer": "threejs"}


@app.get("/api/v1/viewer/status")
async def get_viewer_status(region: str | None = None):
    return {**(await region_bridge(region)).status(),
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
async def scene_socket(websocket: WebSocket, region: str | None = None):
    # Viewer streams are read-only. Reject cross-origin browser subscriptions.
    origin = websocket.headers.get("origin")
    if origin:
        from urllib.parse import urlsplit
        if urlsplit(origin).netloc != websocket.headers.get("host"):
            await websocket.close(code=1008)
            return
    try:
        bridge = await region_bridge(region)
    except HTTPException:
        await websocket.close(code=1013)
        return
    await websocket.accept()
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
