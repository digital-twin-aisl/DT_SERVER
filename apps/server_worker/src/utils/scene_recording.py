# SPDX-FileCopyrightText: 2025-2026 DT_SERVER contributors
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Lossless, opt-in recording of final SceneOutput JSON (before transport)."""

from pathlib import Path


class SceneRecordingError(RuntimeError):
    """A requested recording failed; do not silently continue an incomplete run."""


class SceneOutputRecorder:
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Never append a second run or overwrite a previous experiment.
        self._stream = self.path.open("xb")
        self.frames = 0
        self.bytes = 0

    def write_payload(self, payload: bytes):
        """Write the exact serialized scene sent to consumers, including empty scenes."""
        try:
            line = payload.rstrip(b"\r\n") + b"\n"
            self._stream.write(line)
            self._stream.flush()
        except (OSError, ValueError) as exc:
            raise SceneRecordingError(f"Scene recording failed: {self.path}: {exc}") from exc
        self.frames += 1
        self.bytes += len(line)

    def close(self):
        self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
