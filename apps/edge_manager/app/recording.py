# SPDX-FileCopyrightText: 2025-2026 DT_SERVER contributors
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Bounded scene recording: storage failure must not stop live processing."""

import os
from pathlib import Path
import queue
import shutil
import threading


class SceneRecorder:
    def __init__(self, path, *, max_bytes=None, capacity=256):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes or int(
            os.getenv("DT_RECORDING_MAX_BYTES", str(1024**3))
        )
        self.queue = queue.Queue(maxsize=capacity)
        self.closed = threading.Event()
        self.error = None
        self.frames = self.bytes = self.dropped = 0
        self.thread = threading.Thread(
            target=self._write, name="scene-recorder", daemon=True
        )
        self.thread.start()

    def put(self, payload):
        if self.closed.is_set() or self.error:
            return
        try:
            self.queue.put_nowait(bytes(payload))
        except queue.Full:
            self.dropped += 1

    def _write(self):
        try:
            with self.path.open("xb") as stream:
                while not self.closed.is_set() or not self.queue.empty():
                    try:
                        payload = self.queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if self.bytes + len(payload) + 1 > self.max_bytes:
                        raise OSError(
                            "recording size limit reached; live processing continues"
                        )
                    if (
                        self.frames % 100 == 0
                        and shutil.disk_usage(self.path.parent).free < 256 * 1024**2
                    ):
                        raise OSError(
                            "free disk space is low; recording stopped, live processing continues"
                        )
                    stream.write(payload.rstrip(b"\n") + b"\n")
                    stream.flush()
                    self.bytes += len(payload) + 1
                    self.frames += 1
        except OSError as exc:
            self.error = str(exc)

    def status(self):
        return {
            "active": not self.closed.is_set() and self.error is None,
            "frames": self.frames,
            "bytes": self.bytes,
            "dropped": self.dropped,
            "error": self.error,
        }

    def close(self):
        self.closed.set()
        self.thread.join(timeout=5)
