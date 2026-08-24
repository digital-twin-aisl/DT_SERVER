from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
import omni.ext
from omni.kit.menu.utils import MenuItemDescription, add_menu_items, remove_menu_items
from omni.kit.viewport.utility import get_active_viewport
import omni.replicator.core as rep
import omni.ui as ui
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, Vt

from .camera_config import (
    CAMERAS,
    CameraCalibration,
    opencv_camera_to_usd_matrix_rows,
    usd_camera_intrinsics,
)
from .video_encoder import FFmpegVideoEncoder, select_video_codec


@dataclass
class _RecordingStream:
    name: str
    camera_path: str
    render_product: Any
    annotator: Any
    encoder: FFmpegVideoEncoder


class MultiCameraRecorderExtension(omni.ext.IExt):
    WINDOW_TITLE = "Meta Sejong Multi-Camera Recorder"
    CAPTURE_BATCH_SIZE = 3

    def on_startup(self, ext_id: str) -> None:
        self._state = "idle"
        self._shutting_down = False
        self._capture_task: asyncio.Task | None = None
        self._streams: list[_RecordingStream] = []
        self._recording_stage = None
        self._session_layer = None
        self._camera_root_path: str | None = None
        self._session_directory: Path | None = None
        self._session_started_utc: str | None = None
        self._session_started_monotonic: float | None = None
        self._session_stopped_monotonic: float | None = None
        self._session_settings: dict[str, Any] = {}
        self._capture_error: str | None = None

        self._output_directory_model = ui.SimpleStringModel(
            str(self._default_output_directory())
        )
        self._width_model = ui.SimpleIntModel(640)
        self._height_model = ui.SimpleIntModel(360)
        self._fps_model = ui.SimpleIntModel(5)
        self._status_label = None
        self._start_button = None
        self._stop_button = None

        self._window = ui.Window(self.WINDOW_TITLE, width=590, height=330)
        self._window.set_visibility_changed_fn(self._on_visibility_changed)
        self._build_ui()
        self._menu_items = [
            MenuItemDescription(
                name=self.WINDOW_TITLE,
                onclick_fn=self._show_window,
            )
        ]
        add_menu_items(self._menu_items, "Window")

    @staticmethod
    def _default_output_directory() -> Path:
        # extension.py -> package -> meta_sejong -> extension root -> exts
        # -> isaac_sim_client
        return Path(__file__).resolve().parents[4] / "recordings"

    def _build_ui(self) -> None:
        with self._window.frame:
            with ui.VStack(spacing=9, height=0):
                ui.Label(
                    "현재 Viewport + 보정 카메라 8대 (총 9개 MP4)",
                    height=26,
                )
                with ui.HStack(height=28):
                    ui.Label("저장 폴더", width=120)
                    ui.StringField(model=self._output_directory_model)
                with ui.HStack(height=28, spacing=8):
                    ui.Label("해상도", width=120)
                    ui.IntField(model=self._width_model, width=105)
                    ui.Label("x", width=15)
                    ui.IntField(model=self._height_model, width=105)
                    ui.Spacer(width=15)
                    ui.Label("FPS", width=38)
                    ui.IntField(model=self._fps_model, width=75)
                ui.Label(
                    "성능 기본값: 640x360 / 5 FPS, 캡처 시에만 렌더링",
                    height=22,
                )
                with ui.HStack(height=42, spacing=10):
                    self._start_button = ui.Button(
                        "녹화 시작",
                        clicked_fn=self._on_start_recording,
                    )
                    self._stop_button = ui.Button(
                        "녹화 종료 및 저장",
                        clicked_fn=self._on_stop_recording,
                    )
                self._status_label = ui.Label(
                    "준비됨",
                    word_wrap=True,
                    height=82,
                )
        self._update_button_state()

    def _show_window(self) -> None:
        if self._window is not None:
            self._window.visible = True
            self._window.focus()

    def _on_visibility_changed(self, visible: bool) -> None:
        # Closing the panel must not interrupt an active recording.
        if visible:
            self._update_button_state()

    def _update_button_state(self) -> None:
        if self._start_button is not None:
            self._start_button.enabled = self._state == "idle"
        if self._stop_button is not None:
            self._stop_button.enabled = self._state == "recording"

    def _set_status(self, text: str) -> None:
        if self._status_label is not None:
            self._status_label.text = text
        print(f"[Multi-Camera Recorder] {text}")

    def _on_start_recording(self) -> None:
        if self._state != "idle":
            return
        self._state = "starting"
        self._update_button_state()
        try:
            width, height, fps = self._validated_video_settings()
            ffmpeg_path = shutil.which("ffmpeg")
            if ffmpeg_path is None:
                raise RuntimeError(
                    "ffmpeg 실행 파일을 찾을 수 없습니다. "
                    "시스템에 ffmpeg를 설치하세요."
                )
            self._set_status("사용 가능한 하드웨어 H.264 인코더 확인 중...")
            video_codec = select_video_codec(ffmpeg_path, stream_count=9)

            context = omni.usd.get_context()
            stage = context.get_stage()
            if stage is None:
                raise RuntimeError("먼저 녹화할 USD Stage를 여세요.")
            viewport = get_active_viewport()
            if viewport is None:
                raise RuntimeError("활성 Viewport를 찾을 수 없습니다.")
            viewport_camera_path = str(viewport.camera_path)
            if not viewport_camera_path:
                raise RuntimeError("활성 Viewport 카메라를 찾을 수 없습니다.")

            base_directory = Path(
                self._output_directory_model.as_string
            ).expanduser().resolve()
            base_directory.mkdir(parents=True, exist_ok=True)
            session_directory = self._new_session_directory(base_directory)
            session_directory.mkdir(parents=False, exist_ok=False)

            self._recording_stage = stage
            self._session_layer = stage.GetSessionLayer()
            self._session_directory = session_directory
            self._session_started_utc = datetime.now(timezone.utc).isoformat()
            self._session_started_monotonic = time.monotonic()
            self._session_stopped_monotonic = None
            self._capture_error = None
            meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage) or 1.0)
            self._camera_root_path = self._create_calibrated_cameras(
                stage,
                meters_per_unit,
            )
            camera_paths = [
                f"{self._camera_root_path}/{camera.name}" for camera in CAMERAS
            ]
            stream_count = 1 + len(CAMERAS)
            batch_count = (
                stream_count + self.CAPTURE_BATCH_SIZE - 1
            ) // self.CAPTURE_BATCH_SIZE
            self._session_settings = {
                "width": width,
                "height": height,
                "fps": fps,
                "ffmpeg": ffmpeg_path,
                "video_codec": video_codec,
                "render_products_on_demand": True,
                "capture_batch_size": self.CAPTURE_BATCH_SIZE,
                "capture_batch_count": batch_count,
                "maximum_inter_camera_skew_seconds": (
                    (batch_count - 1) / (fps * batch_count)
                ),
                "viewport_camera_path": viewport_camera_path,
                "meters_per_unit": meters_per_unit,
                "stage": stage.GetRootLayer().identifier,
            }

            stream_specs = [("viewport", viewport_camera_path)] + [
                (camera.name, camera_path)
                for camera, camera_path in zip(CAMERAS, camera_paths)
            ]
            self._streams = self._create_streams(
                stream_specs,
                session_directory,
                ffmpeg_path,
                width,
                height,
                fps,
                video_codec,
            )
            self._state = "recording"
            self._update_button_state()
            self._set_status(
                f"녹화 중: 9개 화면, {width}x{height} @ {fps} FPS "
                f"({video_codec}, 3개씩 순환 렌더링)\n"
                f"임시 저장 위치: {session_directory}"
            )
            self._capture_task = asyncio.ensure_future(self._record_loop())
        except Exception as exc:
            self._capture_error = str(exc)
            self._release_rendering(self._streams)
            self._close_encoders(self._streams)
            self._remove_session_cameras()
            self._streams = []
            self._state = "idle"
            self._update_button_state()
            self._set_status(f"녹화 시작 실패: {exc}")

    def _validated_video_settings(self) -> tuple[int, int, int]:
        width = self._width_model.as_int
        height = self._height_model.as_int
        fps = self._fps_model.as_int
        if not 64 <= width <= 4096 or not 64 <= height <= 4096:
            raise ValueError("해상도는 각 축 64~4096 범위여야 합니다.")
        if width % 2 or height % 2:
            raise ValueError("H.264 저장을 위해 가로와 세로는 짝수여야 합니다.")
        if not 1 <= fps <= 60:
            raise ValueError("FPS는 1~60 범위여야 합니다.")
        return width, height, fps

    @staticmethod
    def _new_session_directory(base_directory: Path) -> Path:
        stem = datetime.now().strftime("recording_%Y%m%d_%H%M%S")
        candidate = base_directory / stem
        suffix = 2
        while candidate.exists():
            candidate = base_directory / f"{stem}_{suffix}"
            suffix += 1
        return candidate

    def _create_calibrated_cameras(self, stage, meters_per_unit: float) -> str:
        root_path = "/MetaSejongRecorderCameras"
        suffix = 2
        while stage.GetPrimAtPath(root_path).IsValid():
            root_path = f"/MetaSejongRecorderCameras_{suffix}"
            suffix += 1

        session_layer = stage.GetSessionLayer()
        with Usd.EditContext(stage, Usd.EditTarget(session_layer)):
            root = UsdGeom.Xform.Define(stage, root_path).GetPrim()
            root.SetCustomDataByKey("metaSejong:temporaryRecorderCameras", True)
            for camera in CAMERAS:
                self._define_camera(
                    stage,
                    f"{root_path}/{camera.name}",
                    camera,
                    meters_per_unit,
                )
        return root_path

    @staticmethod
    def _define_camera(
        stage,
        prim_path: str,
        calibration: CameraCalibration,
        meters_per_unit: float,
    ) -> None:
        camera = UsdGeom.Camera.Define(stage, prim_path)
        camera.CreateProjectionAttr().Set(UsdGeom.Tokens.perspective)
        intrinsics = usd_camera_intrinsics(calibration)
        camera.CreateFocalLengthAttr().Set(intrinsics["focal_length"])
        camera.CreateHorizontalApertureAttr().Set(
            intrinsics["horizontal_aperture"]
        )
        camera.CreateVerticalApertureAttr().Set(intrinsics["vertical_aperture"])
        camera.CreateHorizontalApertureOffsetAttr().Set(
            intrinsics["horizontal_aperture_offset"]
        )
        camera.CreateVerticalApertureOffsetAttr().Set(
            intrinsics["vertical_aperture_offset"]
        )
        camera.CreateClippingRangeAttr().Set(
            Gf.Vec2f(0.05 / meters_per_unit, 10000.0 / meters_per_unit)
        )
        camera.GetPrim().SetCustomDataByKey(
            "metaSejong:sourceImageSize",
            Vt.IntArray(calibration.image_size),
        )

        matrix_rows = opencv_camera_to_usd_matrix_rows(
            calibration,
            meters_per_unit,
        )
        matrix = Gf.Matrix4d(*(value for row in matrix_rows for value in row))
        xformable = UsdGeom.Xformable(camera.GetPrim())
        xformable.ClearXformOpOrder()
        xformable.AddTransformOp(
            precision=UsdGeom.XformOp.PrecisionDouble
        ).Set(matrix)

    @staticmethod
    def _create_streams(
        stream_specs: list[tuple[str, str]],
        output_directory: Path,
        ffmpeg_path: str,
        width: int,
        height: int,
        fps: int,
        video_codec: str,
    ) -> list[_RecordingStream]:
        streams: list[_RecordingStream] = []
        try:
            for name, camera_path in stream_specs:
                render_product = None
                annotator = None
                encoder = None
                try:
                    render_product = rep.create.render_product(
                        camera_path,
                        resolution=(width, height),
                        name=f"MetaSejongRecorder_{name}",
                    )
                    render_product.hydra_texture.set_updates_enabled(False)
                    # The frame is converted to immutable bytes immediately,
                    # so Replicator does not need to make an extra array copy.
                    annotator = rep.AnnotatorRegistry.get_annotator(
                        "rgb",
                        do_array_copy=False,
                    )
                    annotator.attach(render_product)
                    encoder = FFmpegVideoEncoder(
                        ffmpeg_path=ffmpeg_path,
                        output_path=output_directory / f"{name}.mp4",
                        width=width,
                        height=height,
                        fps=fps,
                        codec=video_codec,
                    )
                except Exception:
                    if encoder is not None:
                        encoder.close()
                    if annotator is not None and render_product is not None:
                        annotator.detach(render_product)
                    if render_product is not None:
                        render_product.destroy()
                    raise
                streams.append(
                    _RecordingStream(
                        name=name,
                        camera_path=camera_path,
                        render_product=render_product,
                        annotator=annotator,
                        encoder=encoder,
                    )
                )
        except Exception:
            MultiCameraRecorderExtension._release_rendering(streams)
            MultiCameraRecorderExtension._close_encoders(streams)
            raise
        return streams

    async def _record_loop(self) -> None:
        fps = int(self._session_settings["fps"])
        batch_count = (
            len(self._streams) + self.CAPTURE_BATCH_SIZE - 1
        ) // self.CAPTURE_BATCH_SIZE
        tick_period = 1.0 / (fps * batch_count)
        batch_index = 0
        next_frame_time = time.monotonic()
        last_status_time = 0.0
        capture_error = None
        try:
            while self._state == "recording":
                if omni.usd.get_context().get_stage() is not self._recording_stage:
                    raise RuntimeError(
                        "녹화 중 Stage가 변경되어 녹화를 종료합니다."
                    )

                start = batch_index * self.CAPTURE_BATCH_SIZE
                batch = self._streams[start : start + self.CAPTURE_BATCH_SIZE]
                self._set_render_products_enabled(batch, True)
                try:
                    await rep.orchestrator.step_async(
                        delta_time=0.0,
                        pause_timeline=False,
                    )
                    for stream in batch:
                        try:
                            rgba_bytes = self._as_rgba_bytes(
                                stream.annotator.get_data(),
                                stream.encoder.width,
                                stream.encoder.height,
                            )
                            stream.encoder.enqueue(rgba_bytes)
                        except ValueError:
                            # A newly attached RenderProduct can return an empty
                            # buffer for its first update. Retry next frame.
                            continue
                finally:
                    self._set_render_products_enabled(batch, False)
                batch_index = (batch_index + 1) % batch_count

                now = time.monotonic()
                if now - last_status_time >= 1.0:
                    elapsed = now - (self._session_started_monotonic or now)
                    frames = min(
                        (stream.encoder.submitted_frames for stream in self._streams),
                        default=0,
                    )
                    self._set_status(
                        f"녹화 중: {elapsed:.1f}초 / 최소 {frames} 프레임\n"
                        f"저장 위치: {self._session_directory}"
                    )
                    last_status_time = now

                next_frame_time += tick_period
                if next_frame_time < now - tick_period:
                    next_frame_time = now
                delay = next_frame_time - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            capture_error = str(exc)
        finally:
            if not self._shutting_down:
                self._state = "stopping"
                self._update_button_state()
                if capture_error:
                    self._capture_error = capture_error
                    self._set_status(f"녹화 오류, 저장 마무리 중: {capture_error}")
                else:
                    self._set_status("MP4 저장 마무리 중...")
                await self._finalize_recording()

    @staticmethod
    def _as_rgba_bytes(data: Any, width: int, height: int) -> bytes:
        if hasattr(data, "numpy"):
            data = data.numpy()
        image = np.asarray(data)
        if image.shape[:2] != (height, width) or image.ndim != 3:
            raise ValueError(f"unexpected RGB buffer shape: {image.shape}")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.shape[2] == 3:
            rgba = np.empty((height, width, 4), dtype=np.uint8)
            rgba[:, :, :3] = image
            rgba[:, :, 3] = 255
            image = rgba
        elif image.shape[2] != 4:
            raise ValueError(f"unexpected RGB channel count: {image.shape[2]}")
        return np.ascontiguousarray(image).tobytes()

    def _on_stop_recording(self) -> None:
        if self._state != "recording":
            return
        self._state = "stopping"
        self._update_button_state()
        self._set_status("녹화를 중지하고 9개 MP4를 저장하는 중...")

    async def _finalize_recording(self) -> None:
        self._session_stopped_monotonic = time.monotonic()
        streams = list(self._streams)
        self._release_rendering(streams)
        self._remove_session_cameras()
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, self._close_encoders, streams)
        finished_utc = datetime.now(timezone.utc).isoformat()
        self._write_manifest(streams, results, finished_utc)

        failures = [result for result in results if result.get("error")]
        session_directory = self._session_directory
        self._streams = []
        self._recording_stage = None
        self._capture_task = None
        self._state = "idle"
        self._update_button_state()
        if failures:
            names = ", ".join(Path(item["file"]).stem for item in failures)
            self._set_status(
                f"저장 완료(일부 오류: {names})\n"
                f"상세 내용: {session_directory / 'recording.json'}"
            )
        else:
            self._set_status(f"9개 MP4 저장 완료\n{session_directory}")

    @staticmethod
    def _set_render_products_enabled(
        streams: list[_RecordingStream],
        enabled: bool,
    ) -> None:
        for stream in streams:
            stream.render_product.hydra_texture.set_updates_enabled(enabled)

    @staticmethod
    def _release_rendering(streams: list[_RecordingStream]) -> None:
        for stream in streams:
            try:
                stream.render_product.hydra_texture.set_updates_enabled(False)
            except Exception as exc:
                print(f"[Multi-Camera Recorder] Render disable failed: {exc}")
            try:
                stream.annotator.detach(stream.render_product)
            except Exception as exc:
                print(f"[Multi-Camera Recorder] Annotator detach failed: {exc}")
            try:
                stream.render_product.destroy()
            except Exception as exc:
                print(f"[Multi-Camera Recorder] RenderProduct cleanup failed: {exc}")

    @staticmethod
    def _close_encoders(streams: list[_RecordingStream]) -> list[dict[str, Any]]:
        results = []
        for stream in streams:
            result = stream.encoder.close()
            result["name"] = stream.name
            result["camera_path"] = stream.camera_path
            results.append(result)
        return results

    def _remove_session_cameras(self) -> None:
        if self._session_layer is not None and self._camera_root_path:
            try:
                if self._session_layer.GetPrimAtPath(self._camera_root_path) is not None:
                    edit = Sdf.BatchNamespaceEdit()
                    edit.Add(Sdf.NamespaceEdit.Remove(self._camera_root_path))
                    if not self._session_layer.Apply(edit):
                        raise RuntimeError(
                            f"Could not remove {self._camera_root_path}"
                        )
            except Exception as exc:
                print(f"[Multi-Camera Recorder] Camera cleanup failed: {exc}")
        self._camera_root_path = None
        self._session_layer = None

    def _write_manifest(
        self,
        streams: list[_RecordingStream],
        results: list[dict[str, Any]],
        finished_utc: str,
    ) -> None:
        if self._session_directory is None:
            return
        duration = None
        if self._session_started_monotonic is not None:
            stopped = self._session_stopped_monotonic or time.monotonic()
            duration = max(0.0, stopped - self._session_started_monotonic)
        result_by_name = {item["name"]: item for item in results}
        stream_metadata = []
        for stream in streams:
            stream_metadata.append(
                result_by_name.get(stream.name, {"name": stream.name})
            )

        manifest = {
            "schema_version": 1,
            "started_at": self._session_started_utc,
            "finished_at": finished_utc,
            "wall_duration_seconds": duration,
            "capture_error": self._capture_error,
            "settings": self._session_settings,
            "coordinate_convention": {
                "source": "OpenCV: +X right, +Y down, +Z forward",
                "usd_camera": "+X right, +Y up, -Z forward",
            },
            "streams": stream_metadata,
            "calibrated_cameras": [
                {
                    "name": camera.name,
                    "image_size": list(camera.image_size),
                    "camera_matrix": [list(row) for row in camera.camera_matrix],
                    "camera_to_world": [
                        list(row) for row in camera.camera_to_world
                    ],
                }
                for camera in CAMERAS
            ],
        }
        manifest_path = self._session_directory / "recording.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def on_shutdown(self) -> None:
        self._shutting_down = True
        if hasattr(self, "_menu_items"):
            remove_menu_items(self._menu_items, "Window")
            self._menu_items = []
        if self._capture_task is not None and not self._capture_task.done():
            self._capture_task.cancel()

        streams = list(self._streams)
        if streams and self._session_stopped_monotonic is None:
            self._session_stopped_monotonic = time.monotonic()
        self._release_rendering(streams)
        self._remove_session_cameras()
        results = self._close_encoders(streams)
        if streams:
            self._capture_error = self._capture_error or "Extension was disabled"
            self._write_manifest(
                streams,
                results,
                datetime.now(timezone.utc).isoformat(),
            )
        self._streams = []
        self._window = None
