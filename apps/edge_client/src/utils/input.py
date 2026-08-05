import glob
import os.path as osp
import threading
import time
from multiprocessing import Process, shared_memory

import cv2
import numpy as np


class SharedFrameProcess(Process):
    """Base process that exposes one shared-memory frame per video source."""

    def __init__(self, paths, stop_event, conn):
        super().__init__(daemon=False)
        self.paths = list(paths)
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
        paths = sorted(glob.glob(osp.join(path, "hdVideos", "*.mp4")))
        super().__init__(paths, stop_event, conn)
        self.input_flag = input_flag

    def run(self):
        captures = self._open_sources()
        try:
            while not self.stop_event.is_set():
                if not self.input_flag.value:
                    time.sleep(0.01)
                    continue

                read_frame = False
                for index, capture in enumerate(captures):
                    if self.shms[index] is None:
                        continue
                    ok, frame = capture.read()
                    if ok:
                        self._write_frame(index, frame)
                        read_frame = True

                if not read_frame:
                    break
                time.sleep(0.05)
        finally:
            self._release_sources(captures)


class IPCamera(SharedFrameProcess):
    def __init__(self, cameras, stop_event, conn, input_flag=None):
        del input_flag  # Kept in the signature for legacy callers.
        super().__init__((camera["url"] for camera in cameras), stop_event, conn)

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
