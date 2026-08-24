"""Non-blocking raw RGBA to MP4 encoding through FFmpeg."""

from __future__ import annotations

import os
from pathlib import Path
import queue
import subprocess
import tempfile
import threading
import time
from typing import Any


def build_ffmpeg_command(
    ffmpeg_path: str,
    output_path: str | Path,
    width: int,
    height: int,
    fps: int,
) -> list[str]:
    """Build the bounded-CPU H.264 command used for each recorder stream."""
    if width <= 0 or height <= 0 or fps <= 0:
        raise ValueError("width, height, and fps must be positive")
    return [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgba",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-tune",
        "zerolatency",
        "-crf",
        "23",
        "-threads",
        "2",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]


class FFmpegVideoEncoder:
    """Feed an FFmpeg subprocess from a small, non-blocking worker queue."""

    def __init__(
        self,
        ffmpeg_path: str,
        output_path: str | Path,
        width: int,
        height: int,
        fps: int,
    ) -> None:
        self.output_path = Path(output_path)
        self.partial_path = self.output_path.with_name(
            f"{self.output_path.stem}.part{self.output_path.suffix}"
        )
        self.width = width
        self.height = height
        self.fps = fps
        self.submitted_frames = 0
        self.dropped_frames = 0
        self.error: str | None = None
        self._queue: queue.Queue[bytes | None] = queue.Queue(maxsize=2)
        self._close_lock = threading.Lock()
        self._closed = False
        self._close_result: dict[str, Any] | None = None
        self._stderr = tempfile.TemporaryFile(mode="w+b")
        command = build_ffmpeg_command(
            ffmpeg_path,
            self.partial_path,
            width,
            height,
            fps,
        )
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self._stderr,
        )
        self._thread = threading.Thread(
            target=self._write_frames,
            name=f"ffmpeg-{self.output_path.stem}",
            daemon=True,
        )
        self._thread.start()

    def enqueue(self, rgba_bytes: bytes) -> bool:
        """Queue one frame without ever blocking the Isaac Sim UI thread."""
        if len(rgba_bytes) != self.width * self.height * 4:
            raise ValueError(
                f"invalid RGBA byte count for {self.output_path.name}: "
                f"{len(rgba_bytes)}"
            )
        if self._closed or self.error is not None or self._process.poll() is not None:
            self.dropped_frames += 1
            return False
        try:
            self._queue.put_nowait(rgba_bytes)
        except queue.Full:
            self.dropped_frames += 1
            return False
        self.submitted_frames += 1
        return True

    def _write_frames(self) -> None:
        stdin = self._process.stdin
        if stdin is None:
            self.error = "FFmpeg stdin was not created"
            return
        try:
            while True:
                frame = self._queue.get()
                if frame is None:
                    break
                stdin.write(frame)
        except (BrokenPipeError, OSError) as exc:
            self.error = f"FFmpeg input failed: {exc}"
        finally:
            try:
                stdin.close()
            except OSError:
                pass

    def close(self, timeout: float = 120.0) -> dict[str, Any]:
        """Drain queued frames, finalize MP4, and return serializable statistics."""
        with self._close_lock:
            if self._close_result is not None:
                return dict(self._close_result)
            self._closed = True

            deadline = time.monotonic() + timeout
            while self._thread.is_alive() and time.monotonic() < deadline:
                try:
                    self._queue.put(None, timeout=0.2)
                    break
                except queue.Full:
                    continue
            self._thread.join(timeout=max(0.0, deadline - time.monotonic()))
            if self._thread.is_alive():
                self.error = "FFmpeg frame writer did not stop before timeout"
                self._process.terminate()
                self._thread.join(timeout=5.0)

            try:
                return_code = self._process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                return_code = self._process.wait(timeout=5.0)
                self.error = self.error or "FFmpeg did not exit before timeout"

            ffmpeg_error = self._read_stderr()
            if return_code != 0:
                self.error = (
                    self.error
                    or ffmpeg_error
                    or f"FFmpeg exited with {return_code}"
                )
            elif self.submitted_frames == 0:
                self.error = self.error or "No rendered frames were received"
            elif not self.partial_path.is_file():
                self.error = self.error or "FFmpeg produced no output file"
            else:
                os.replace(self.partial_path, self.output_path)

            try:
                self._stderr.close()
            except OSError:
                pass

            self._close_result = {
                "file": str(self.output_path),
                "frames": self.submitted_frames,
                "dropped_frames": self.dropped_frames,
                "encoded_duration_seconds": self.submitted_frames / self.fps,
                "return_code": return_code,
                "error": self.error,
            }
            return dict(self._close_result)

    def _read_stderr(self) -> str:
        try:
            self._stderr.flush()
            self._stderr.seek(0)
            value = self._stderr.read().decode("utf-8", errors="replace").strip()
            return value[-4000:]
        except OSError:
            return ""
