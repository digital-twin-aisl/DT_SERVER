"""CPU-only input admission and scheduling, independently testable from Zenoh."""

from collections import deque
from dataclasses import dataclass
import math
import hashlib
import logging
import time

from dt_common.inference_codec import decode_frame

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CompressedMessage:
    timestamp: float
    payload: bytes
    received_at: float = 0.0


@dataclass(frozen=True, slots=True)
class SynchronizedBatch:
    timestamp: float
    timestamp_spread: float
    frames: dict


class InputScheduler:
    def __init__(
        self,
        edge_ids,
        *,
        mode="independent",
        clock="live",
        buffer_size=10,
        sync_tolerance=0.03,
        max_age=0.75,
        future_tolerance=0.1,
    ):
        if mode not in {"independent", "strict"} or clock not in {"live", "dataset"}:
            raise ValueError("invalid scheduling mode or input clock")
        if buffer_size < 1 or not all(
            math.isfinite(v) and v >= 0
            for v in (sync_tolerance, max_age, future_tolerance)
        ):
            raise ValueError("invalid synchronization settings")
        if clock == "live" and max_age <= 0:
            raise ValueError("live input requires a positive max_age")
        self.edge_ids = tuple(edge_ids)
        self.mode, self.clock = mode, clock
        self.sync_tolerance, self.max_age = sync_tolerance, max_age
        self.future_tolerance = future_tolerance
        self.buffers = {key: deque(maxlen=buffer_size) for key in edge_ids}
        self.discarded = {key: 0 for key in edge_ids}
        self.rejected = {key: 0 for key in edge_ids}
        self.last_seen = {}
        self.sequences = {}
        self.last_batch_time = -math.inf
        self.decode_failures = 0
        self.last_rejection = {}
        self._last_rejection_log = {}

    def _reject(self, edge, reason, count=1):
        self.rejected[edge] += count
        self.last_rejection[edge] = reason
        now = time.monotonic()
        if now - self._last_rejection_log.get(edge, -math.inf) >= 2:
            logger.warning(
                "Rejected input from %s: %s (total=%d)",
                edge,
                reason,
                self.rejected[edge],
            )
            self._last_rejection_log[edge] = now

    def _fresh(self, message, now, monotonic):
        if self.clock == "dataset":
            return 0 <= message.timestamp < 946684800
        return (
            message.timestamp >= 946684800
            and -self.future_tolerance <= now - message.timestamp <= self.max_age
            and (
                not message.received_at
                or monotonic - message.received_at <= self.max_age
            )
        )

    def add(self, edge_id, message, *, now=None, monotonic=None):
        now = time.time() if now is None else now
        monotonic = time.monotonic() if monotonic is None else monotonic
        if not self._fresh(message, now, monotonic):
            self._reject(edge_id, "timestamp outside configured clock/freshness bounds")
            return
        buffer = self.buffers[edge_id]
        if len(buffer) == buffer.maxlen:
            self.discarded[edge_id] += 1
        buffer.append(message)

    def select(self, *, now=None, monotonic=None):
        now = time.time() if now is None else now
        monotonic = time.monotonic() if monotonic is None else monotonic
        for edge, buffer in self.buffers.items():
            fresh = [
                message for message in buffer if self._fresh(message, now, monotonic)
            ]
            if len(buffer) != len(fresh):
                self._reject(
                    edge, "input expired while queued", len(buffer) - len(fresh)
                )
            buffer.clear()
            buffer.extend(sorted(fresh, key=lambda message: message.timestamp))
        selected = {}
        if self.mode == "strict":
            if not all(self.buffers.values()):
                return None
            # Consume the earliest matching pair. Replay never silently skips a
            # valid earlier pair merely because one edge has already run ahead.
            while all(self.buffers.values()):
                stamps = {
                    edge: buffer[0].timestamp for edge, buffer in self.buffers.items()
                }
                if max(stamps.values()) - min(stamps.values()) <= self.sync_tolerance:
                    selected = {
                        edge: buffer.popleft() for edge, buffer in self.buffers.items()
                    }
                    break
                oldest = min(stamps, key=stamps.get)
                self.buffers[oldest].popleft()
                self.discarded[oldest] += 1
        else:
            for edge, buffer in self.buffers.items():
                if buffer:
                    selected[edge] = buffer[-1]
                    self.discarded[edge] += len(buffer) - 1
                    buffer.clear()
        if not selected:
            return None
        frames = {}
        for edge, message in selected.items():
            try:
                frame = decode_frame(message.payload)
                if frame["time"] != message.timestamp:
                    raise ValueError("scheduled/header timestamp mismatch")
                if frame.get("edge_id") != edge:
                    raise ValueError("edge ID does not match topic")
                kind = frame.get("timestamp_kind")
                if kind not in (
                    {"capture_unix", "receive_unix"}
                    if self.clock == "live"
                    else {"dataset_relative"}
                ):
                    raise ValueError("input timestamp domain mismatch")
                frame_stamps = frame.get("frame_timestamps", [])
                if (
                    len(frame_stamps) != len(frame.get("camera_ids", []))
                    or not frame_stamps
                ):
                    raise ValueError("missing per-camera timestamps")
                if not all(
                    isinstance(value, (int, float)) and math.isfinite(value)
                    for value in frame_stamps
                ):
                    raise ValueError("invalid camera timestamps")
                if min(frame_stamps) != frame["time"]:
                    raise ValueError("frame time must equal oldest camera observation")
                if max(frame_stamps) - min(frame_stamps) > self.sync_tolerance + 1e-9:
                    raise ValueError("edge camera timestamp spread exceeds tolerance")
                session, sequence = frame.get("session_id"), frame.get("sequence")
                if (
                    not isinstance(session, str)
                    or not 1 <= len(session) <= 128
                    or type(sequence) is not int
                    or sequence < 0
                ):
                    raise ValueError("missing session/sequence identity")
                previous = self.sequences.get(edge)
                if previous and (
                    message.timestamp <= previous[2]
                    or (session == previous[0] and sequence <= previous[1])
                ):
                    raise ValueError("duplicate or out-of-order inference frame")
                self.sequences[edge] = session, sequence, message.timestamp
                frame["_wire_sha256"] = hashlib.sha256(message.payload).hexdigest()
                frames[edge] = frame
            except Exception as exc:
                self.decode_failures += 1
                self._reject(edge, str(exc))
        if not frames or (self.mode == "strict" and len(frames) != len(self.edge_ids)):
            return None
        stamps = [frame["time"] for frame in frames.values()]
        timestamp = max(stamps)
        if timestamp < self.last_batch_time and self.mode == "strict":
            for edge in frames:
                self.discarded[edge] += 1
            return None
        # Slower independent edges may carry older observations than the last
        # fast edge. Keep decision time monotonic without dropping their data;
        # individual frame timestamps retain the true observation times.
        timestamp = max(timestamp, self.last_batch_time)
        self.last_batch_time = timestamp
        return SynchronizedBatch(timestamp, max(stamps) - min(stamps), frames)
