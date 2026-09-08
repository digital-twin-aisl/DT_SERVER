import glob
import multiprocessing as mp
import os
import os.path as osp
from pathlib import Path
import re
import threading
import time
from dataclasses import dataclass
from contextlib import ExitStack
from multiprocessing import Process, shared_memory
from urllib.parse import urlsplit

import cv2
import numpy as np
import yaml


CAMERA_VIDEO_PATTERN = re.compile(r"^camera_([1-9][0-9]*)\.(?:mkv|mp4)$", re.I)
LEGACY_VIDEO_PATTERN = re.compile(r".*_([1-9][0-9]*)\.mp4$", re.I)
EDGE_ASSIGNMENT_PATTERN = re.compile(r"^\s*([^:#]+?)\s*:\s*(.*?)\s*$")
RTSP_SCHEMES = {"rtsp", "rtsps"}


class FrameUnavailableError(RuntimeError):
    """A live camera has not produced a sufficiently fresh frame."""


@dataclass(frozen=True)
class FrameBundle:
    frames: list
    timestamps: list[float]
    sequences: list[int]
    timestamp_kind: str = "receive_unix"


def snapshot_live_bundle(buffers, process, *, max_age, max_skew, previous_sequences=None):
    """Select a near-time set from per-camera rings using receiver timestamps.

    OpenCV does not expose a common camera exposure clock here. These timestamps
    are explicitly labelled receive_unix, never advertised as capture time.
    """
    depth = process.buffer_depth
    with ExitStack() as locks:
        for lock in process.frame_locks:
            locks.enter_context(lock)
        now = time.monotonic()
        candidates = []
        for index in range(len(buffers)):
            slots = [slot for slot in range(depth)
                     if process.frame_timestamps[index * depth + slot] > 0
                     and now - process.frame_timestamps[index * depth + slot] <= max_age
                     and (previous_sequences is None or
                          process.frame_sequences[index * depth + slot] > previous_sequences[index])]
            if not slots:
                raise FrameUnavailableError(f"camera {process.camera_ids[index]} has no new fresh frame")
            candidates.append(slots)
        target = min(max(process.frame_timestamps[index * depth + slot] for slot in slots)
                     for index, slots in enumerate(candidates))
        selected = [min(slots, key=lambda slot: abs(process.frame_timestamps[index * depth + slot] - target))
                    for index, slots in enumerate(candidates)]
        selected_mono = [process.frame_timestamps[index * depth + slot]
                         for index, slot in enumerate(selected)]
        if max(selected_mono) - min(selected_mono) > max_skew:
            raise FrameUnavailableError(f"RTSP frame arrival skew exceeds {max_skew:.3f}s")
        frames, timestamps, sequences = [], [], []
        for index, ((shm, shape, dtype), slot) in enumerate(zip(buffers, selected)):
            array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            frames.append((array[slot] if depth > 1 else array).copy())
            timestamps.append(float(process.frame_wall_timestamps[index * depth + slot]))
            sequences.append(int(process.frame_sequences[index * depth + slot]))
    return FrameBundle(frames, timestamps, sequences)


def load_camera_sources(path):
    """Load the edge-private camera YAML without exposing its RTSP URLs."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
    cameras = document.get("CAMERAS")
    if not isinstance(cameras, list):
        raise ValueError(f"camera config CAMERAS must be a list: {config_path}")
    return cameras


def select_live_cameras(cameras, camera_ids=None, num_views=None):
    """Validate and order RTSP sources by the deployment/identity camera IDs."""
    sources = {}
    for index, camera in enumerate(cameras):
        if not isinstance(camera, dict):
            raise ValueError(f"camera entry {index} must be an object")
        try:
            camera_id = int(camera["id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"camera entry {index} has an invalid ID") from exc
        if camera_id < 1 or camera_id in sources:
            raise ValueError(f"camera IDs must be unique positive integers: {camera_id}")
        url = str(camera.get("url", "")).strip()
        parsed = urlsplit(url)
        if parsed.scheme.lower() not in RTSP_SCHEMES or not parsed.hostname:
            raise ValueError(f"camera {camera_id} must have a valid RTSP URL")
        sources[camera_id] = {**camera, "id": camera_id, "url": url}

    selected_ids = (
        [int(value) for value in camera_ids]
        if camera_ids is not None
        else sorted(sources)
    )
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("selected live camera IDs must be unique")
    missing = [camera_id for camera_id in selected_ids if camera_id not in sources]
    if missing:
        raise ValueError(
            "camera config is missing required RTSP sources: "
            + ", ".join(str(value) for value in missing)
        )
    if num_views is not None and len(selected_ids) != int(num_views):
        raise ValueError(
            f"live input selects {len(selected_ids)} cameras, "
            f"but the model expects {int(num_views)}"
        )
    return [sources[camera_id] for camera_id in selected_ids]


def snapshot_shared_frames(
    buffers,
    frame_locks,
    frame_timestamps=None,
    *,
    max_age=None,
    max_skew=None,
):
    """Copy one coherent frame per source while its writer is excluded."""
    frames = []
    timestamps = []
    for index, ((shm, shape, dtype), lock) in enumerate(
        zip(buffers, frame_locks, strict=True)
    ):
        with lock:
            frame = np.ndarray(shape, dtype=dtype, buffer=shm.buf).copy()
            timestamp = (
                float(frame_timestamps[index])
                if frame_timestamps is not None
                else None
            )
        frames.append(frame)
        if timestamp is not None:
            timestamps.append(timestamp)

    if timestamps:
        now = time.monotonic()
        unavailable = [
            index
            for index, timestamp in enumerate(timestamps)
            if timestamp <= 0.0
            or (max_age is not None and now - timestamp > max_age)
        ]
        if unavailable:
            raise FrameUnavailableError(
                "waiting for fresh RTSP frames from source indexes: "
                + ", ".join(str(value) for value in unavailable)
            )
        if max_skew is not None and max(timestamps) - min(timestamps) > max_skew:
            raise FrameUnavailableError(
                f"RTSP frame arrival skew exceeds {max_skew:.3f}s"
            )
    return frames


def find_dataset_calibration(path):
    """Return the single modern USD calibration file, if the dataset has one."""
    result_paths = sorted(Path(path).glob("calibration_result*.json"))
    if len(result_paths) > 1:
        raise ValueError(
            "dataset must contain exactly one calibration_result JSON"
        )
    return result_paths[0].resolve() if result_paths else None


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


def load_dataset_edge_assignments(path):
    """Load optional ``edge_info.txt`` camera assignments from a dataset."""
    info_path = Path(path) / "edge_info.txt"
    if not info_path.is_file():
        return {}

    assignments = {}
    for line_number, raw_line in enumerate(
        info_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        match = EDGE_ASSIGNMENT_PATTERN.fullmatch(line)
        if match is None:
            raise ValueError(
                f"invalid edge assignment at {info_path}:{line_number}: {raw_line!r}"
            )
        edge_id = match.group(1).strip()
        try:
            camera_ids = [
                int(value.strip())
                for value in match.group(2).split(",")
                if value.strip()
            ]
        except ValueError as exc:
            raise ValueError(
                f"invalid camera ID at {info_path}:{line_number}: {raw_line!r}"
            ) from exc
        if not camera_ids or any(camera_id < 1 for camera_id in camera_ids):
            raise ValueError(
                f"camera IDs must be positive at {info_path}:{line_number}"
            )
        if len(camera_ids) != len(set(camera_ids)):
            raise ValueError(
                f"duplicate camera ID at {info_path}:{line_number}"
            )
        if edge_id in assignments:
            raise ValueError(
                f"duplicate edge ID at {info_path}:{line_number}: {edge_id}"
            )
        assignments[edge_id] = camera_ids
    return assignments


def select_dataset_camera_ids(path, edge_id, num_views):
    """Select the edge-local camera order for a dataset recording."""
    available = [camera_id for camera_id, _ in discover_dataset_videos(path)]
    if not available:
        raise ValueError(f"dataset contains no supported videos: {path}")

    assignments = load_dataset_edge_assignments(path)
    if edge_id in assignments:
        selected = assignments[edge_id]
    elif len(available) == num_views:
        selected = available
    elif assignments:
        known = ", ".join(sorted(assignments))
        raise ValueError(
            f"dataset has {len(available)} videos and requires an edge assignment; "
            f"edge_id {edge_id!r} is not in edge_info.txt (available: {known})"
        )
    else:
        raise ValueError(
            f"dataset has {len(available)} videos, but the model expects {num_views}; "
            "add edge_info.txt or use an edge-specific dataset folder"
        )

    missing = [camera_id for camera_id in selected if camera_id not in available]
    if missing:
        raise ValueError(
            "dataset is missing assigned videos: "
            + ", ".join(f"camera_{camera_id}" for camera_id in missing)
        )
    if len(selected) != num_views:
        raise ValueError(
            f"edge {edge_id!r} selects {len(selected)} cameras, "
            f"but the model expects {num_views}"
        )
    return selected


class SharedFrameProcess(Process):
    """Base process that exposes one shared-memory frame per video source."""

    def __init__(self, paths, stop_event, conn, camera_ids=None, buffer_depth=1):
        super().__init__(daemon=False)
        self.buffer_depth = int(buffer_depth)
        if not 1 <= self.buffer_depth <= 32:
            raise ValueError("buffer_depth must be between 1 and 32")
        self.paths = list(paths)
        self.camera_ids = list(camera_ids or range(1, len(self.paths) + 1))
        if len(self.camera_ids) != len(self.paths):
            raise ValueError("camera_ids and paths must have the same length")
        self.stop_event = stop_event
        self.conn = conn
        self.shms = [None] * len(self.paths)
        self.shapes = [None] * len(self.paths)
        self.dtypes = [None] * len(self.paths)
        self.frame_locks = [mp.Lock() for _ in self.paths]
        self.frame_timestamps = mp.Array(
            "d",
            len(self.paths) * self.buffer_depth,
            lock=False,
        )
        self.frame_wall_timestamps = mp.Array("d", len(self.paths) * self.buffer_depth, lock=False)
        self.frame_sequences = mp.Array("Q", len(self.paths) * self.buffer_depth, lock=False)
        self.frame_counts = [0] * len(self.paths)

    def _create_capture(self, path):
        return cv2.VideoCapture(path)

    def _validate_initial_frame(self, index, frame):
        del index, frame

    def _open_sources(self):
        captures = []
        for index, path in enumerate(self.paths):
            try:
                capture = self._create_capture(path)
            except Exception as exc:
                self.conn.send(
                    ("ERR", index, self.camera_ids[index], str(exc))
                )
                captures.append(None)
                continue
            captures.append(capture)
            try:
                ok, frame = (
                    capture.read() if capture.isOpened() else (False, None)
                )
            except Exception as exc:
                capture.release()
                self.conn.send(
                    ("ERR", index, self.camera_ids[index], str(exc))
                )
                continue
            if not ok:
                self.conn.send(
                    (
                        "ERR",
                        index,
                        self.camera_ids[index],
                        "could not read the first frame",
                    )
                )
                continue
            try:
                self._validate_initial_frame(index, frame)
            except Exception as exc:
                capture.release()
                self.conn.send(
                    ("ERR", index, self.camera_ids[index], str(exc))
                )
                continue

            shm = shared_memory.SharedMemory(create=True, size=frame.nbytes * self.buffer_depth)
            self.shms[index] = shm
            self.shapes[index] = frame.shape
            self.dtypes[index] = frame.dtype
            self._write_frame(index, frame)
            shared_shape = (self.buffer_depth, *frame.shape) if self.buffer_depth > 1 else frame.shape
            self.conn.send(("OK", index, shm.name, shared_shape, frame.dtype.str))
        return captures

    def _write_frame(self, index, frame):
        shape = self.shapes[index]
        dtype = self.dtypes[index]
        if frame.shape != shape:
            frame = cv2.resize(frame, (shape[1], shape[0]))
        if frame.dtype != dtype:
            frame = frame.astype(dtype)
        with self.frame_locks[index]:
            slot = self.frame_counts[index] % self.buffer_depth
            shared_shape = (self.buffer_depth, *shape) if self.buffer_depth > 1 else shape
            array = np.ndarray(shared_shape, dtype=dtype, buffer=self.shms[index].buf)
            (array[slot] if self.buffer_depth > 1 else array)[:] = frame
            self.frame_counts[index] += 1
            location = index * self.buffer_depth + slot
            self.frame_timestamps[location] = time.monotonic()
            self.frame_wall_timestamps[location] = time.time()
            self.frame_sequences[location] = self.frame_counts[index]

    def _invalidate_frame(self, index):
        with self.frame_locks[index]:
            for slot in range(self.buffer_depth):
                self.frame_timestamps[index * self.buffer_depth + slot] = 0.0

    def _release_sources(self, captures):
        for capture in captures:
            if capture is not None:
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
    def __init__(self, path, stop_event, conn, input_flag, camera_ids=None):
        sources = discover_dataset_videos(path)
        if not sources:
            raise ValueError(f"dataset contains no supported videos: {path}")
        sources_by_id = dict(sources)
        selected_ids = (
            list(camera_ids) if camera_ids is not None else list(sources_by_id)
        )
        missing = [
            camera_id
            for camera_id in selected_ids
            if camera_id not in sources_by_id
        ]
        if missing:
            raise ValueError(
                "dataset is missing selected videos: "
                + ", ".join(f"camera_{camera_id}" for camera_id in missing)
            )
        paths = [sources_by_id[camera_id] for camera_id in selected_ids]
        super().__init__(paths, stop_event, conn, selected_ids)
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
    def __init__(
        self,
        cameras,
        stop_event,
        conn,
        input_flag=None,
        *,
        transport="tcp",
        open_timeout_ms=8000,
        read_timeout_ms=3000,
        reconnect_delay=0.5,
        expected_size=None,
        buffer_depth=4,
    ):
        del input_flag  # Kept in the signature for legacy callers.
        if transport not in {"tcp", "udp"}:
            raise ValueError("RTSP transport must be 'tcp' or 'udp'")
        if min(open_timeout_ms, read_timeout_ms) < 1:
            raise ValueError("RTSP timeouts must be positive")
        if reconnect_delay <= 0:
            raise ValueError("RTSP reconnect delay must be positive")
        self.transport = transport
        self.open_timeout_ms = int(open_timeout_ms)
        self.read_timeout_ms = int(read_timeout_ms)
        self.reconnect_delay = float(reconnect_delay)
        self.expected_size = (
            None
            if expected_size is None
            else tuple(int(value) for value in expected_size)
        )
        super().__init__(
            (camera["url"] for camera in cameras),
            stop_event,
            conn,
            (int(camera["id"]) for camera in cameras),
            buffer_depth=buffer_depth,
        )

    def _create_capture(self, path):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            f"rtsp_transport;{self.transport}"
        )
        parameters = []
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            parameters.extend(
                [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.open_timeout_ms]
            )
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            parameters.extend(
                [cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.read_timeout_ms]
            )
        capture = cv2.VideoCapture()
        capture.open(path, cv2.CAP_FFMPEG, parameters)
        if capture.isOpened() and hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def _validate_frame_size(self, index, frame):
        if self.expected_size is None:
            return
        expected_width, expected_height = self.expected_size
        height, width = frame.shape[:2]
        if (width, height) != (expected_width, expected_height):
            raise RuntimeError(
                f"camera {self.camera_ids[index]} returned {width}x{height}; "
                f"calibration expects {expected_width}x{expected_height}"
            )

    def _validate_initial_frame(self, index, frame):
        self._validate_frame_size(index, frame)

    def run(self):
        captures = self._open_sources()
        threads = []
        try:
            for index, capture in enumerate(captures):
                if self.shms[index] is not None:
                    thread = threading.Thread(
                        target=self._read_camera,
                        args=(index, capture),
                        daemon=True,
                    )
                    thread.start()
                    threads.append(thread)
            self.stop_event.wait()
        finally:
            self.stop_event.set()
            for thread in threads:
                thread.join(timeout=self.read_timeout_ms / 1000.0 + 1.0)
            self._release_sources(captures)

    def _read_camera(self, index, capture):
        current = capture
        disconnected = False
        try:
            while not self.stop_event.is_set():
                ok, frame = current.read()
                if ok:
                    self._validate_frame_size(index, frame)
                    self._write_frame(index, frame)
                    if disconnected:
                        print(
                            f"RTSP camera {self.camera_ids[index]} reconnected",
                            flush=True,
                        )
                        disconnected = False
                    continue

                self._invalidate_frame(index)
                if not disconnected:
                    print(
                        f"RTSP camera {self.camera_ids[index]} disconnected; "
                        "reconnecting",
                        flush=True,
                    )
                    disconnected = True
                current.release()
                if self.stop_event.wait(self.reconnect_delay):
                    break
                current = self._create_capture(self.paths[index])
                if not current.isOpened():
                    current.release()
                    continue
                ok, frame = current.read()
                if ok:
                    self._validate_frame_size(index, frame)
                    self._write_frame(index, frame)
        except Exception as exc:
            self._invalidate_frame(index)
            print(
                f"RTSP camera {self.camera_ids[index]} reader stopped: {exc}",
                flush=True,
            )
        finally:
            current.release()
