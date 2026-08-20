import glob
import os.path as osp
from pathlib import Path
import re
import threading
import time
from multiprocessing import Process, shared_memory

import cv2
import numpy as np


CAMERA_VIDEO_PATTERN = re.compile(r"^camera_([1-9][0-9]*)\.(?:mkv|mp4)$", re.I)
LEGACY_VIDEO_PATTERN = re.compile(r".*_([1-9][0-9]*)\.mp4$", re.I)


def discover_dataset_videos(path):
    """Return ``(camera_id, path)`` pairs in physical camera-number order."""
    root = Path(path)
    modern = []
    for video_path in root.iterdir() if root.is_dir() else ():
        match = CAMERA_VIDEO_PATTERN.fullmatch(video_path.name)
        if match is not None:
            modern.append((int(match.group(1)), str(video_path)))
    if modern:
        return sorted(modern)

    legacy = []
    for value in glob.glob(osp.join(path, "hdVideos", "*.mp4")):
        match = LEGACY_VIDEO_PATTERN.fullmatch(Path(value).name)
        if match is not None:
            legacy.append((int(match.group(1)), value))
    return sorted(legacy)


class SharedFrameProcess(Process):
    """Base process that exposes one shared-memory frame per video source."""

    def __init__(self, paths, stop_event, conn, camera_ids=None):
        super().__init__(daemon=False)
        self.paths = list(paths)
        self.camera_ids = list(camera_ids or range(1, len(self.paths) + 1))
        if len(self.camera_ids) != len(self.paths):
            raise ValueError("camera_ids and paths must have the same length")
        self.stop_event = stop_event
        self.conn = conn
        self.shms = [None] * len(self.paths)
        self.shapes = [None] * len(self.paths)
        self.dtypes = [None] * len(self.paths)

    def _open_sources(self):
        captures = [cv2.VideoCapture(path) for path in self.paths]
        for index, capture in enumerate(captures):
            ok, frame = capture.read()
            if not ok:
                self.conn.send(("ERR", index))
                continue

            shm = shared_memory.SharedMemory(create=True, size=frame.nbytes)
            self.shms[index] = shm
            self.shapes[index] = frame.shape
            self.dtypes[index] = frame.dtype
            self.conn.send(
                ("OK", index, shm.name, frame.shape, frame.dtype.str)
            )
            self._write_frame(index, frame)
        return captures

    def _write_frame(self, index, frame):
        shape = self.shapes[index]
        dtype = self.dtypes[index]
        if frame.shape != shape:
            frame = cv2.resize(frame, (shape[1], shape[0]))
        if frame.dtype != dtype:
            frame = frame.astype(dtype)
        np.ndarray(shape, dtype=dtype, buffer=self.shms[index].buf)[:] = frame

    def _release_sources(self, captures):
        for capture in captures:
            capture.release()
        for shm in self.shms:
            if shm is None:
                continue
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass


class ExampleDataset(SharedFrameProcess):
    def __init__(self, path, stop_event, conn, input_flag):
        sources = discover_dataset_videos(path)
        if not sources:
            raise ValueError(f"dataset contains no supported videos: {path}")
        camera_ids, paths = zip(*sources)
        super().__init__(paths, stop_event, conn, camera_ids)
        self.input_flag = input_flag
        capture = cv2.VideoCapture(self.paths[0])
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
        finally:
            capture.release()
        self.fps = fps if np.isfinite(fps) and fps > 0 else 30.0

    def run(self):
        captures = self._open_sources()
        try:
            while not self.stop_event.is_set():
                if not self.input_flag.value:
                    time.sleep(0.01)
                    continue

                frames = []
                for capture in captures:
                    ok, frame = capture.read()
                    if not ok:
                        frames = []
                        break
                    frames.append(frame)
                if not frames:
                    self.stop_event.set()
                    self.input_flag.value = False
                    break

                for index, frame in enumerate(frames):
                    if self.shms[index] is None:
                        continue
                    self._write_frame(index, frame)
                self.input_flag.value = False
        finally:
            self._release_sources(captures)


class IPCamera(SharedFrameProcess):
    def __init__(self, cameras, stop_event, conn, input_flag=None):
        del input_flag  # Kept in the signature for legacy callers.
        super().__init__(
            (camera["url"] for camera in cameras),
            stop_event,
            conn,
            (int(camera["id"]) for camera in cameras),
        )

    def run(self):
        captures = self._open_sources()
        try:
            for index, capture in enumerate(captures):
                if self.shms[index] is not None:
                    threading.Thread(
                        target=self._read_camera,
                        args=(index, capture),
                        daemon=True,
                    ).start()
            self.stop_event.wait()
        finally:
            self.stop_event.set()
            self._release_sources(captures)

    def _read_camera(self, index, capture):
        while not self.stop_event.is_set():
            ok, frame = capture.read()
            if ok:
                self._write_frame(index, frame)
            else:
                time.sleep(0.005)
