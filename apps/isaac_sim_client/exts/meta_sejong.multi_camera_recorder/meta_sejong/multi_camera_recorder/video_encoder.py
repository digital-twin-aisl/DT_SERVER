"""Non-blocking raw RGBA to MP4 encoding through FFmpeg."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import os
from pathlib import Path
import queue
import subprocess
import tempfile
import threading
import time
from typing import Any


HARDWARE_ENCODERS = ("h264_nvenc", "h264_v4l2m2m", "h264_omx")


def _codec_arguments(codec: str) -> list[str]:
    if codec == "libx264":
        return [
            "-c:v",
            codec,
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-crf",
            "23",
            # Nine two-thread encoders oversubscribe an 8-core Orin. One
            # thread per stream is slower in isolation but keeps Kit usable.
            "-threads",
            "1",
        ]
    if codec == "h264_nvenc":
        return ["-c:v", codec, "-preset", "fast", "-b:v", "4M"]
    if codec in {"h264_v4l2m2m", "h264_omx"}:
        return ["-c:v", codec, "-b:v", "4M"]
    raise ValueError(f"unsupported H.264 encoder: {codec}")


def build_ffmpeg_command(
    ffmpeg_path: str,
    output_path: str | Path,
    width: int,
    height: int,
    fps: int,
    codec: str = "libx264",
) -> list[str]:
    """Build an H.264 command for one recorder stream."""
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
        *_codec_arguments(codec),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]


def _available_encoders(ffmpeg_path: str) -> set[str]:
    try:
        result = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-encoders"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return set()
    output = f"{result.stdout}\n{result.stderr}"
    return {
        codec
        for codec in (*HARDWARE_ENCODERS, "libx264")
        if codec in output
    }


def _probe_encoder_instance(
    ffmpeg_path: str,
    codec: str,
    output_path: Path,
    barrier: threading.Barrier,
) -> bool:
    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-re",
        "-f",
        "lavfi",
        "-i",
        "color=black:s=128x72:r=2",
        "-t",
        "1",
        "-an",
        *_codec_arguments(codec),
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    try:
        barrier.wait(timeout=5.0)
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10.0,
        )
    except (OSError, subprocess.TimeoutExpired, threading.BrokenBarrierError):
        return False
    return result.returncode == 0 and output_path.is_file()


def _supports_concurrent_streams(
    ffmpeg_path: str,
    codec: str,
    stream_count: int,
) -> bool:
    """Verify that all hardware sessions can really be opened concurrently."""
    with tempfile.TemporaryDirectory(prefix="recorder-codec-probe-") as directory:
        root = Path(directory)
        barrier = threading.Barrier(stream_count)
        with ThreadPoolExecutor(max_workers=stream_count) as executor:
            futures = [
                executor.submit(
                    _probe_encoder_instance,
                    ffmpeg_path,
                    codec,
                    root / f"probe_{index}.mp4",
                    barrier,
                )
                for index in range(stream_count)
            ]
            return all(future.result() for future in futures)


@lru_cache(maxsize=8)
def select_video_codec(ffmpeg_path: str, stream_count: int = 9) -> str:
    """Pick hardware H.264 only when the requested session count works."""
    if stream_count <= 0:
        raise ValueError("stream_count must be positive")
    available = _available_encoders(ffmpeg_path)
    for codec in HARDWARE_ENCODERS:
        if codec in available and _supports_concurrent_streams(
            ffmpeg_path,
            codec,
            stream_count,
        ):
            return codec
    if "libx264" not in available:
        raise RuntimeError("FFmpeg does not provide a usable H.264 encoder")
    return "libx264"


class FFmpegVideoEncoder:
    """Feed an FFmpeg subprocess from a small, non-blocking worker queue."""

    def __init__(
        self,
        ffmpeg_path: str,
        output_path: str | Path,
        width: int,
        height: int,
        fps: int,
        codec: str = "libx264",
    ) -> None:
        self.output_path = Path(output_path)
        self.partial_path = self.output_path.with_name(
            f"{self.output_path.stem}.part{self.output_path.suffix}"
        )
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = codec
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
            codec,
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
                "codec": self.codec,
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
