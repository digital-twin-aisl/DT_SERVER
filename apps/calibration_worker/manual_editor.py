#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Local-only camera pose workbench. Never writes calibration or deployment files."""
import argparse
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
import sys
from typing import Literal
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
# This small CPU-only fit does not need threaded BLAS. Avoid an unresolved
# Intel OpenMP symbol in the supplied Anaconda SciPy/MKL environment.
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware
from apps.calibration_worker.domain.manual import (
    rigid, move_camera, display_intrinsic, evaluate, refine, export_calibration,
)
from apps.calibration_worker.domain.lens_render import inverse_lens_map
from apps.calibration_worker.domain.point_cloud import load_cloud_buffer

DEFAULT_CALIBRATION = REPO/'apps/deployments/rootnet_v2/calibration.from_cameras_v2.json'
WEB = Path(__file__).with_name('manual_web')


class Pair(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra='forbid')
    world: list[float] = Field(min_length=3, max_length=3)
    image: list[float] | None = Field(default=None, min_length=2, max_length=2)
    holdout: bool = False


class PoseRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra='forbid')
    camera_id: str
    dataset: Literal[1, 2, 3] = 1
    mode: Literal['raw', 'rectified', 'pinhole'] = 'raw'
    camera_to_world: list[list[float]]
    translation: list[float] = Field(default=[0., 0., 0.], min_length=3, max_length=3)
    rotation: list[float] = Field(default=[0., 0., 0.], min_length=3, max_length=3)
    pairs: list[Pair] = Field(default=[], max_length=200)


class ExportRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra='forbid')
    source_sha256: str
    edits: dict[str, list[list[float]]]


class Workbench:
    def __init__(self, calibration, data_dir, assets_dir, calibration_size):
        self.calibration = Path(calibration)
        content = self.calibration.read_bytes()
        self.sha = hashlib.sha256(content).hexdigest()
        self.document = json.loads(content)
        self.cameras = {c['camera_id']: c for c in self.document['cameras']}
        if len(self.cameras) != len(self.document['cameras']): raise ValueError('Duplicate camera IDs.')
        for c in self.cameras.values():
            rigid(c['camera_to_world'])
            if not np.allclose(np.asarray(c['world_to_camera']) @ rigid(c['camera_to_world']), np.eye(4), atol=1e-5):
                raise ValueError('Inconsistent camera transforms.')
        self.data_dir, self.assets_dir = Path(data_dir), Path(assets_dir)
        self.calibration_size = tuple(calibration_size)

    def camera(self, camera_id):
        if camera_id not in self.cameras: raise ValueError('Unknown camera ID.')
        return self.cameras[camera_id]

    @lru_cache(maxsize=1)
    def cloud_buffer(self):
        metadata = self.document.get('point_cloud')
        if not metadata: raise ValueError('No point cloud is paired with this calibration.')
        return load_cloud_buffer(self.calibration, metadata)

    def video(self, dataset, camera_id):
        self.camera(camera_id)
        # Resolve from a known ID, never accept a client-supplied filesystem path.
        n = int(camera_id.split('/')[-1])
        if dataset not in (1, 2, 3): raise ValueError('Unknown dataset.')
        path = self.data_dir/f'data_0812_{dataset}_edge_{1 if n % 2 == 0 else 2}'/f'camera_{n}.mkv'
        if not path.is_file(): raise ValueError(f'Missing video for {camera_id}, 0812_{dataset}.')
        return path

    @lru_cache(maxsize=24)
    def video_info(self, dataset, camera_id):
        capture = cv2.VideoCapture(str(self.video(dataset, camera_id)))
        try:
            if not capture.isOpened(): raise ValueError('Cannot open video.')
            width, height = [int(capture.get(p)) for p in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)]
            fps = float(capture.get(cv2.CAP_PROP_FPS)); count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if min(width, height, fps, count) <= 0: raise ValueError('Invalid video metadata.')
            # MKV metadata can over-count EOF by one; verify the last decodable frame.
            for _ in range(min(count, 5)):
                capture.set(cv2.CAP_PROP_POS_FRAMES, count - 1)
                if capture.read()[0]: break
                count -= 1
            else: raise ValueError('Cannot seek near the end of this video; check the recording index.')
            return dict(width=width, height=height, fps=fps, frames=count, duration=(count - 1)/fps)
        finally: capture.release()

    def geometry(self, camera_id, dataset, mode):
        c = self.camera(camera_id); info = self.video_info(dataset, camera_id)
        return display_intrinsic(c, (info['width'], info['height']), self.calibration_size, mode)

    @lru_cache(maxsize=2)
    def lens_model(self, camera_id, dataset, mode):
        info = self.video_info(dataset, camera_id)
        k, d, _, _ = self.geometry(camera_id, dataset, mode)
        meta, rays = inverse_lens_map(k, d, info['width'], info['height'])
        meta.update(source_sha256=self.sha, camera_id=camera_id, mode=mode)
        return meta, rays.tobytes()


def create_app(calibration=DEFAULT_CALIBRATION, data_dir=REPO/'apps/edge_client/data',
               assets_dir=REPO/'apps/frontend_api/assets', calibration_size=(1920, 1080)):
    work = Workbench(calibration, data_dir, assets_dir, calibration_size)
    app = FastAPI(title='Camera Pose Workbench', docs_url=None, redoc_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost', 'testserver'])

    @app.middleware('http')
    async def local_only(request: Request, call_next):
        origin = request.headers.get('origin')
        if origin and urlsplit(origin).netloc != request.headers.get('host'):
            return JSONResponse({'detail': 'Cross-origin requests are not allowed.'}, status_code=403)
        if request.headers.get('sec-fetch-site') == 'cross-site' and request.url.path != '/':
            return JSONResponse({'detail': 'Cross-site resource access is not allowed.'}, status_code=403)
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        # GLTF ImageBitmapLoader fetches embedded textures via local blob URLs.
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self'; img-src 'self' blob: data:; connect-src 'self' blob:; worker-src 'self' blob:; frame-ancestors 'none'"
        if request.url.path.startswith('/api/'): response.headers['Cache-Control'] = 'no-store'
        return response

    @app.exception_handler(ValueError)
    async def invalid(request, error): return JSONResponse({'detail': str(error)}, status_code=422)

    @app.get('/')
    def index(): return FileResponse(WEB/'index.html')

    @app.get('/api/config')
    def config(dataset: int = Query(1, ge=1, le=3)):
        cameras = []
        for c in work.cameras.values():
            info = work.video_info(dataset, c['camera_id'])
            k, d, _, _ = work.geometry(c['camera_id'], dataset, 'raw')
            cameras.append(dict(camera_id=c['camera_id'], camera_to_world=c['camera_to_world'],
                video=info, camera_matrix=k.tolist(), distortion_coefficients=d.tolist()))
        return dict(source_name=work.calibration.name, source_sha256=work.sha,
            point_cloud=work.document.get('point_cloud'),
            calibration_size_assumed=list(calibration_size), cameras=cameras, dataset=dataset,
            coordinate_system='USD world Z-up metres / OpenCV camera +X right, +Y down, +Z forward')

    @app.get('/api/frame')
    def frame(camera_id: str, dataset: int = Query(1, ge=1, le=3), index: int = 0,
              mode: Literal['raw', 'rectified', 'pinhole'] = 'raw'):
        info = work.video_info(dataset, camera_id)
        if not 0 <= index < info['frames']: raise ValueError('Frame index out of range.')
        capture = cv2.VideoCapture(str(work.video(dataset, camera_id)))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index); ok, image = capture.read()
            if not ok: raise ValueError('Frame decode failed.')
        finally: capture.release()
        if mode == 'rectified':
            new, _, k, d = work.geometry(camera_id, dataset, mode)
            image = cv2.undistort(image, k, d, None, new)
        ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 93])
        if not ok: raise ValueError('Frame encoding failed.')
        return Response(encoded.tobytes(), media_type='image/jpeg', headers={'X-Frame-Index': str(index)})

    @app.get('/api/lens-model')
    def lens_model(camera_id: str, dataset: int = Query(1, ge=1, le=3),
                   mode: Literal['raw', 'rectified', 'pinhole'] = 'raw'):
        meta, payload = work.lens_model(camera_id, dataset, mode)
        return Response(payload, media_type='application/octet-stream',
                        headers={'X-Lens-Model': json.dumps(meta, separators=(',', ':'))})

    @app.get('/api/point-cloud')
    def point_cloud():
        return Response(work.cloud_buffer(), media_type='application/octet-stream',
                        headers={'X-Calibration-SHA256': work.sha})

    def unpack(body):
        c = work.camera(body.camera_id)
        k, d, _, _ = work.geometry(body.camera_id, body.dataset, body.mode)
        pose = move_camera(body.camera_to_world, body.translation, body.rotation)
        pairs = [p.model_dump() for p in body.pairs]
        info = work.video_info(body.dataset, body.camera_id)
        for p in pairs:
            if p['image'] is not None and not (0 <= p['image'][0] < info['width'] and 0 <= p['image'][1] < info['height']):
                raise ValueError('Image point is outside this frame.')
        return c, k, d, pose, pairs

    @app.post('/api/project')
    def preview(body: PoseRequest):
        c, k, d, pose, pairs = unpack(body)
        return dict(camera_to_world=pose.tolist(), camera_matrix=k.tolist(),
                    current=evaluate(pose, pairs, k, d), original=evaluate(c['camera_to_world'], pairs, k, d))

    @app.post('/api/refine')
    def fit(body: PoseRequest):
        _, k, d, pose, pairs = unpack(body)
        return refine(pose, pairs, k, d)

    @app.post('/api/export')
    def export(body: ExportRequest):
        if body.source_sha256 != work.sha: raise ValueError('Project belongs to a different calibration source.')
        document = export_calibration(work.document, body.edits, work.sha)
        return Response(json.dumps(document, indent=2, allow_nan=False), media_type='application/json',
            headers={'Content-Disposition': 'attachment; filename="calibration_manual.json"'})

    @app.get('/assets/{name}')
    def asset(name: str):
        if name not in ('map.glb', 'map.json'): raise HTTPException(404)
        return FileResponse(work.assets_dir/name)

    app.mount('/static', StaticFiles(directory=WEB), name='static')
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument('--data-dir', type=Path, default=REPO/'apps/edge_client/data')
    parser.add_argument('--assets-dir', type=Path, default=REPO/'apps/frontend_api/assets')
    parser.add_argument('--calibration-width', type=int, default=1920)
    parser.add_argument('--calibration-height', type=int, default=1080)
    parser.add_argument('--port', type=int, default=8092)
    args = parser.parse_args()
    if min(args.calibration_width, args.calibration_height) <= 0: parser.error('Invalid calibration image size.')
    if not (WEB/'dist/editor.js').is_file(): parser.error('Run build_manual_editor.sh first.')
    import uvicorn
    uvicorn.run(create_app(args.calibration, args.data_dir, args.assets_dir,
                          (args.calibration_width, args.calibration_height)), host='127.0.0.1', port=args.port)


if __name__ == '__main__': main()
