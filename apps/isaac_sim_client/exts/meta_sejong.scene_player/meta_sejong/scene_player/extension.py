# SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Isaac/Kit UI. File indexing happens off the UI thread; USD edits do not."""

import asyncio
from pathlib import Path
import time
import threading

import omni.ext
import omni.kit.app
from omni.kit.menu.utils import MenuItemDescription, add_menu_items, remove_menu_items
import omni.ui as ui
import omni.usd

from .playback import Playback, SceneRecording
from .renderer import SceneRenderer


class ScenePlayerExtension(omni.ext.IExt):
    TITLE = "Meta Sejong Scene Player"

    def on_startup(self, ext_id):
        self._closed = False
        self._loading = None
        self._load_cancelled = None
        self._people_count = 0
        self._player = None
        self._renderer = None
        self._rendered_index = None
        self._updating_ui = False
        self._last_tick = time.perf_counter()
        self._path = ui.SimpleStringModel("")
        self._frame = ui.SimpleIntModel(0)
        self._speed = ui.SimpleFloatModel(1.0)
        self._loop = ui.SimpleBoolModel(False)
        self._window = ui.Window(self.TITLE, width=660, height=330)
        with self._window.frame:
            with ui.VStack(spacing=8, height=0):
                ui.Label("SceneOutput JSONL playback | Open a matching USD stage first.", height=24)
                with ui.HStack(height=28, spacing=6):
                    ui.StringField(model=self._path)
                    self._load_button = ui.Button("Open File", width=90, clicked_fn=self._load)
                with ui.HStack(height=32, spacing=5):
                    ui.Button("Play", clicked_fn=self._play)
                    ui.Button("Pause", clicked_fn=self._pause)
                    ui.Button("First", clicked_fn=lambda: self._seek(0))
                    ui.Button("Previous", clicked_fn=lambda: self._step(-1))
                    ui.Button("Next", clicked_fn=lambda: self._step(1))
                    ui.Button("Unload", clicked_fn=self._unload)
                with ui.HStack(height=26, spacing=6):
                    ui.Label("Frame (0-based)", width=112)
                    self._slider = ui.IntSlider(model=self._frame, min=0, max=1)
                    ui.IntField(model=self._frame, width=80)
                with ui.HStack(height=26, spacing=8):
                    ui.Label("Speed (0.1-8)", width=110)
                    ui.FloatField(model=self._speed, width=75)
                    ui.CheckBox(model=self._loop, width=20)
                    ui.Label("Loop", width=40)
                    self._time_label = ui.Label("No file loaded")
                self._status = ui.Label("Enter the absolute path to a JSONL file, then click Open File.",
                                        word_wrap=True, height=70)
        self._frame_sub = self._frame.subscribe_value_changed_fn(self._frame_changed)
        self._menu = [MenuItemDescription(name=self.TITLE, onclick_fn=self._show)]
        add_menu_items(self._menu, "Window")
        self._update_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
            self._update, name="Meta Sejong offline scene playback",
        )

    def _show(self):
        self._window.visible = True
        self._window.focus()

    def _message(self, value):
        if not self._closed:
            self._status.text = value
            print(f"[Scene Player] {value}")

    def _load(self):
        if self._loading is not None:
            return
        self._pause()
        if not self._path.as_string.strip():
            self._message("Enter the absolute path to a JSONL file.")
            return
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            self._message("First open a USD stage with the same coordinate system as the recording.")
            return
        self._load_button.enabled = False
        self._message("Validating file and indexing frames...")
        path = Path(self._path.as_string).expanduser()
        self._load_cancelled = threading.Event()
        self._loading = asyncio.ensure_future(self._load_async(path, stage))

    async def _load_async(self, path, stage):
        try:
            recording = await asyncio.get_running_loop().run_in_executor(
                None, SceneRecording.load, path, self._load_cancelled,
            )
            if self._closed:
                return
            if omni.usd.get_context().get_stage() != stage:
                raise ValueError("The stage has changed. Open the recording again.")
            self._clear_scene()
            self._renderer = SceneRenderer(stage)
            self._player = Playback(recording)
            self._slider.max = max(1, len(recording.offsets) - 1)
            self._render()
            self._sync_ui()
            self._message(f"{len(recording.offsets)} frames | {recording.duration:.3f}s\n"
                          f"{recording.path}\n{recording.warning}")
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            self._message(f"Failed to open file: {exc}")
        finally:
            if not self._closed:
                self._load_button.enabled = True
            self._loading = None

    def _play(self):
        if self._player is not None:
            self._player.play()
            self._last_tick = time.perf_counter()

    def _pause(self):
        if self._player is not None:
            self._player.playing = False

    def _seek(self, frame):
        if self._player is None:
            return
        self._pause()
        try:
            self._player.seek_frame(frame)
            self._render()
            self._sync_ui()
        except Exception as exc:
            self._message(f"Seek failed: {exc}")

    def _step(self, delta):
        if self._player is not None:
            self._seek(self._player.index + delta)

    def _frame_changed(self, model):
        if not self._updating_ui:
            self._seek(model.as_int)

    def _render(self):
        if self._player.index != self._rendered_index:
            scene = self._player.recording.frame(self._player.index)
            self._renderer.apply(scene)
            self._people_count = len(scene["people"])
            self._rendered_index = self._player.index

    def _sync_ui(self):
        self._updating_ui = True
        try:
            self._frame.set_value(self._player.index)
            state = "Playing" if self._player.playing else "Paused"
            self._time_label.text = (f"{state} {self._player.elapsed:.3f} / "
                                     f"{self._player.recording.duration:.3f}s | People: {self._people_count}")
        finally:
            self._updating_ui = False

    def _update(self, event):
        now = time.perf_counter()
        dt, self._last_tick = now - self._last_tick, now
        if self._closed or self._player is None:
            return
        try:
            if omni.usd.get_context().get_stage() != self._renderer.stage:
                self._unload()
                self._message("Playback cleared because the stage changed. Open the recording again.")
                return
            if self._player.playing:
                self._player.speed = self._speed.as_float
                self._player.loop = self._loop.as_bool
                self._player.advance(dt)
                self._render()
                self._sync_ui()
        except Exception as exc:
            self._pause()
            self._message(f"Playback stopped: {exc}")

    def _clear_scene(self):
        self._player = None
        self._rendered_index = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _unload(self):
        if self._loading is not None:
            self._load_cancelled.set()
            self._loading.cancel()
        self._clear_scene()
        if not self._closed:
            self._time_label.text = "No file loaded"
            self._message("Recording unloaded. The original USD has not been modified.")

    def on_shutdown(self):
        self._closed = True
        self._update_sub = None
        self._frame_sub = None
        self._unload()
        remove_menu_items(self._menu, "Window")
        if self._window is not None:
            self._window.destroy()
            self._window = None
