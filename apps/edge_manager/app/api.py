# SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Management HTTP interface; the CLI and frontend use these same operations."""

import hmac
import os
from urllib.parse import urlsplit

from fastapi import Body, FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse


def create_app(control):
    app = FastAPI(title="Digital Twin Manager")

    @app.middleware("http")
    async def access(request: Request, call_next):
        token = os.getenv("DT_MANAGER_TOKEN")
        if token and not hmac.compare_digest(
            request.headers.get("authorization", ""), f"Bearer {token}"
        ):
            return JSONResponse(
                {"detail": "manager authentication required"}, status_code=401
            )
        origin = request.headers.get("origin")
        if origin and urlsplit(origin).netloc != request.headers.get("host"):
            return JSONResponse(
                {"detail": "cross-origin management requests are not allowed"},
                status_code=403,
            )
        return await call_next(request)

    @app.exception_handler(KeyError)
    async def missing(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(TimeoutError)
    async def timeout(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=504)

    @app.exception_handler(OSError)
    async def unavailable(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=503)

    @app.get("/healthz")
    def health():
        return {"status": "ready" if not control.closing else "stopping"}

    @app.get("/regions")
    def regions():
        return control.regions()

    @app.get("/regions/{region_id}")
    def region(region_id: str):
        return control.status(region_id)

    @app.put("/regions/{region_id}")
    def register(region_id: str, body: dict = Body(...)):
        return control.configure_region(region_id, body)

    @app.delete("/regions/{region_id}")
    def remove(region_id: str):
        return control.configure_region(region_id, remove=True)

    @app.post("/regions/{region_id}/start")
    def start(region_id: str, body: dict = Body(default={})):
        if set(body) - {"record"}:
            raise ValueError("start accepts only record")
        return control.start(region_id, record=body.get("record", False))

    @app.post("/regions/{region_id}/stop")
    def stop(region_id: str):
        return control.stop(region_id)

    @app.get("/regions/{region_id}/logs")
    def logs(region_id: str, edge_id: str | None = None):
        return control.logs(region_id, edge_id)

    @app.get("/regions/{region_id}/recordings")
    def recordings(region_id: str):
        return control.recordings(region_id)

    @app.get("/regions/{region_id}/recordings/{run_id}/download")
    def download(region_id: str, run_id: str):
        return FileResponse(
            control.recording_path(region_id, run_id),
            filename=f"{run_id}.jsonl",
            media_type="application/x-ndjson",
        )

    @app.post("/regions/{region_id}/replay")
    def replay(region_id: str, body: dict = Body(...)):
        if set(body) != {"recording_id"}:
            raise ValueError("replay requires recording_id")
        return control.replay(region_id, body["recording_id"])

    @app.post("/regions/{region_id}/calibrate")
    def calibrate(region_id: str, body: dict = Body(...)):
        if set(body) != {"edge_id", "reference_video", "marker_tree", "checkpoint"}:
            raise ValueError(
                "calibrate requires edge_id, reference_video, marker_tree and checkpoint"
            )
        return control.calibrate(region_id, **body)

    @app.get("/edges")
    def edges():
        return list(control.registry.snapshot()["edges"].values())

    @app.post("/edges/{edge_id}/{action}")
    def edge_action(edge_id: str, action: str, body: dict = Body(default={})):
        if set(body) - {"name", "endpoint"}:
            raise ValueError("edge action accepts only name and endpoint")
        return control.edge_action(edge_id, action, **body)

    return app
