import logging
import math
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

import torch
import zenoh


from dt_common.inference_codec import inspect_message
from dt_common.zenoh_transport import make_zenoh_config
from .synchronization import CompressedMessage, SynchronizedBatch, InputScheduler


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PreparedInputs:
    heatmaps: list
    roots: torch.Tensor
    reid_items: list
    timestamps: list
    spread: float
    edge_ids: tuple
    frames: dict
    timestamp: float

    def __iter__(self):
        return iter((self.heatmaps, self.roots, self.reid_items, self.timestamps, self.spread))


class ZenohSubscriber:
    def __init__(
        self,
        topics: dict[str, str],
        endpoint: str | None = None,
        config_path: str | None = None,
        queue_size: int = 10,
    ) -> None:
        if not topics:
            raise ValueError("At least one topic is required")
        if queue_size < 1:
            raise ValueError("queue_size must be at least 1")
        self.topics = topics
        self.endpoint = endpoint
        self.config_path = config_path
        self.queues = {
            edge_id: queue.Queue[CompressedMessage](queue_size)
            for edge_id in topics
        }
        self.dropped = {edge_id: 0 for edge_id in topics}
        self.received_messages = {edge_id: 0 for edge_id in topics}
        self.received_bytes = {edge_id: 0 for edge_id in topics}
        self.invalid_messages = {edge_id: 0 for edge_id in topics}
        self.closed = False
        self.last_received = {}
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self.worker = threading.Thread(
            target=self._run, name="zenoh-subscriber", daemon=True
        )
        logger.info(
            "Initializing Zenoh subscriber: endpoint=%s, topics=%s, queue_size=%d",
            endpoint,
            topics,
            queue_size,
        )
        self.worker.start()
        if not self._ready.wait(timeout=10):
            self.close()
            raise TimeoutError("Zenoh subscriber initialization timed out")
        if self._error is not None:
            self.close()
            raise RuntimeError("Zenoh subscriber initialization failed") from self._error

    def _run(self) -> None:
        def on_sample(edge_id: str, sample: Any) -> None:
            try:
                payload = sample.payload.to_bytes()
                timestamp, _ = inspect_message(payload)
                self.received_messages[edge_id] += 1
                self.received_bytes[edge_id] += len(payload)
                self.last_received[edge_id] = time.monotonic()
                message = CompressedMessage(timestamp, payload, self.last_received[edge_id])
                target_queue = self.queues[edge_id]
                try:
                    target_queue.put_nowait(message)
                    return
                except queue.Full:
                    target_queue.get_nowait()
                    self.dropped[edge_id] += 1
                    target_queue.put_nowait(message)
            except (queue.Empty, queue.Full):
                self.dropped[edge_id] += 1
            except Exception:
                self.invalid_messages[edge_id] += 1
                logger.exception("Invalid Zenoh message from %s", edge_id)

        try:
            config = make_zenoh_config(self.endpoint, self.config_path)
            logger.info("Opening Zenoh session")
            with zenoh.open(config) as session:
                subscribers = [
                    session.declare_subscriber(
                        topic,
                        lambda sample, edge_id=edge_id: on_sample(edge_id, sample),
                    )
                    for edge_id, topic in self.topics.items()
                ]
                logger.info("Zenoh subscribers declared: %s", self.topics)
                self._ready.set()
                logger.info("Zenoh subscriber is ready; waiting for messages")
                self._stop.wait()
                _ = subscribers
        except Exception as exc:
            self._error = exc
            logger.exception("Zenoh subscriber worker failed")
            self._ready.set()

    def get(self, edge_id: str, timeout: float = 0.1) -> CompressedMessage | None:
        if self._error is not None:
            raise RuntimeError("Zenoh subscriber worker failed") from self._error
        try:
            return self.queues[edge_id].get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        logger.info("Closing Zenoh subscriber")
        self._stop.set()
        if self.worker.is_alive() and threading.current_thread() is not self.worker:
            self.worker.join(timeout=5)


class ZenohDataLoader:
    """Zenoh 메시지를 동기화하고 서버 추론기의 입력 데이터로 준비한다."""

    def __init__(
        self,
        edge_ids: list[str],
        device: torch.device,
        topics: dict[str, str] | None = None,
        expected_camera_ids: dict[str, list[int]] | None = None,
        expected_spatial_context: dict[str, str] | None = None,
        expected_workspace_ids: dict[str, str] | None = None,
        endpoint: str | None = None,
        config_path: str | None = None,
        topic_template: str = "dt/edges/{edge_id}/inference",
        buffer_size: int = 10,
        sync_tolerance: float = 0.03,
        input_mode: str = "independent",
        input_clock: str = "live",
        max_input_age: float = 0.75,
        future_tolerance: float = 0.1,
        expected_calibration_digests: dict | None = None,
        expected_heatmap_shape: tuple | None = None,
        expected_root_count: int | None = None,
        subscriber=None,
    ) -> None:
        if not edge_ids or len(edge_ids) != len(set(edge_ids)):
            raise ValueError("edge_ids must be non-empty and unique")
        if buffer_size < 1 or sync_tolerance < 0:
            raise ValueError("Invalid synchronization settings")
        if topics is None:
            topics = {
                edge_id: topic_template.format(edge_id=edge_id)
                for edge_id in edge_ids
            }
        elif set(topics) != set(edge_ids):
            raise ValueError("topics must contain exactly one entry for every edge")
        self.edge_ids = tuple(edge_ids)
        if (
            expected_camera_ids is not None
            and set(expected_camera_ids) != set(edge_ids)
        ):
            raise ValueError(
                "expected_camera_ids must contain exactly one entry for every edge"
            )
        self.expected_camera_ids = expected_camera_ids
        self.expected_spatial_context = expected_spatial_context
        if (
            expected_workspace_ids is not None
            and set(expected_workspace_ids) != set(edge_ids)
        ):
            raise ValueError(
                "expected_workspace_ids must contain every configured edge"
            )
        self.expected_workspace_ids = expected_workspace_ids
        self.device = device
        self.sync_tolerance = sync_tolerance
        self.scheduler = InputScheduler(edge_ids, mode=input_mode, clock=input_clock,
                                        buffer_size=buffer_size, sync_tolerance=sync_tolerance,
                                        max_age=max_input_age, future_tolerance=future_tolerance)
        self.buffers = self.scheduler.buffers
        self.synchronization_discarded = self.scheduler.discarded
        self.expected_calibration_digests = expected_calibration_digests
        self.expected_heatmap_shape = expected_heatmap_shape
        self.expected_root_count = expected_root_count
        self.prepare_failures = 0
        self.subscriber = subscriber or ZenohSubscriber(topics, endpoint, config_path, buffer_size)
        self.last_valid_received = {}
        self._batch_count = 0
        logger.info(
            "Zenoh data loader is initialized: edges=%s, sync_tolerance=%.3fs",
            self.edge_ids,
            self.sync_tolerance,
        )

    @property
    def decode_failures(self):
        return self.scheduler.decode_failures

    def edge_status(self):
        now = time.monotonic()
        return {edge: ("online" if now - self.last_valid_received.get(edge, -math.inf)
                       <= self.scheduler.max_age else "offline") for edge in self.edge_ids}

    def get(self, timeout: float = 1.0) -> PreparedInputs | None:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("receive timeout must be finite and positive")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for edge in self.edge_ids:
                while (message := self.subscriber.get(edge, 0)) is not None:
                    self.scheduler.add(edge, message)
            batch = self.scheduler.select()
            if batch is not None:
                if self.scheduler.mode == "independent":
                    valid = {}
                    items = []
                    for edge, frame in batch.frames.items():
                        try:
                            items.append(self._prepare_inputs(SynchronizedBatch(frame["time"], 0, {edge: frame})))
                            valid[edge] = frame
                        except Exception:
                            self.prepare_failures += 1
                            logger.exception("Invalid input from %s", edge)
                    if not valid:
                        continue
                    if any(len(item[0]) != len(items[0][0]) for item in items):
                        raise ValueError("configured edges must use the same number of views")
                    prepared = ([torch.cat([item[0][view] for item in items], dim=0) for view in range(len(items[0][0]))],
                                torch.cat([item[1] for item in items], dim=0),
                                [person for item in items for person in item[2]],
                                [timestamp for item in items for timestamp in item[3]])
                    stamps = prepared[3]
                    batch = SynchronizedBatch(max(batch.timestamp, max(stamps)), max(stamps) - min(stamps), valid)
                else:
                    try:
                        prepared = self._prepare_inputs(batch)
                    except Exception:
                        self.prepare_failures += 1
                        logger.exception("Failed to prepare input batch")
                        continue
                self._batch_count += 1
                self.last_valid_received.update({edge: time.monotonic() for edge in batch.frames})
                return PreparedInputs(*prepared, batch.timestamp_spread,
                                      tuple(batch.frames), batch.frames, batch.timestamp)
            time.sleep(min(0.002, max(0, deadline - time.monotonic())))
        return None

    def _prepare_inputs(
        self,
        batch: SynchronizedBatch,
    ) -> tuple[list[torch.Tensor], torch.Tensor, list[dict[str, Any]], list[float]]:
        """동기화된 Edge 데이터를 서버 추론기가 소비하는 tensor로 변환한다."""
        edge_heatmaps = []
        edge_roots = []
        reid_items = []
        timestamps = []
        view_count = None

        for edge_id, frame in batch.frames.items():
            heatmap_values = frame.get("allheatmaps")
            root_values = frame.get("roots")
            if not heatmap_values or root_values is None:
                raise ValueError(f"Missing heatmaps or roots from Edge {edge_id}")

            expected_digest = (getattr(self, "expected_calibration_digests", None) or {}).get(edge_id)
            if expected_digest and frame.get("calibration_digest") != expected_digest:
                raise ValueError(f"Effective calibration mismatch from Edge {edge_id}")
            if len(heatmap_values) != len(frame.get("camera_ids") or []):
                raise ValueError(f"Camera/heatmap count mismatch from {edge_id}")
            views = []
            for value in heatmap_values:
                heatmap = torch.as_tensor(value)
                if heatmap.ndim == 4 and heatmap.shape[0] == 1:
                    heatmap = heatmap.squeeze(0)
                if heatmap.dtype != torch.uint8:
                    raise ValueError(f"Heatmaps from {edge_id} must use uint8")
                if heatmap.ndim != 3:
                    raise ValueError(f"Invalid heatmap shape from Edge {edge_id}")
                shape = getattr(self, "expected_heatmap_shape", None)
                if shape and tuple(heatmap.shape) != tuple(shape):
                    raise ValueError(f"Unexpected model heatmap shape from {edge_id}")
                views.append(heatmap)

            if self.expected_camera_ids is not None:
                received_camera_ids = [
                    int(value) for value in frame.get("camera_ids") or []
                ]
                expected = self.expected_camera_ids[edge_id]
                if received_camera_ids != expected:
                    raise ValueError(
                        f"Camera order mismatch from Edge {edge_id}: "
                        f"expected {expected}, got {received_camera_ids}"
                    )
            expected_spatial_context = getattr(
                self,
                "expected_spatial_context",
                None,
            )
            if expected_spatial_context is not None:
                received_context = frame.get("spatial_context") or {}
                mismatches = {
                    key: (expected, received_context.get(key))
                    for key, expected in expected_spatial_context.items()
                    if received_context.get(key) != expected
                }
                if mismatches:
                    raise ValueError(
                        f"Spatial context mismatch from Edge {edge_id}: {mismatches}"
                    )
            expected_workspace_ids = getattr(self, "expected_workspace_ids", None)
            if expected_workspace_ids is not None:
                received_context = frame.get("spatial_context") or {}
                received_edge_id = received_context.get("edge_id")
                received_workspace_id = received_context.get("workspace_id")
                if (
                    received_edge_id != edge_id
                    or received_workspace_id != expected_workspace_ids[edge_id]
                ):
                    raise ValueError(
                        f"Workspace mismatch from Edge {edge_id}: expected "
                        f"{expected_workspace_ids[edge_id]}, got "
                        f"{received_workspace_id} for {received_edge_id}"
                    )

            view_count = view_count or len(views)
            if len(views) != view_count:
                raise ValueError("Every Edge must provide the same number of camera views")
            edge_heatmaps.append(torch.stack(views))

            roots = torch.as_tensor(root_values, dtype=torch.float32)
            if roots.ndim == 3 and roots.shape[0] == 1:
                roots = roots.squeeze(0)
            if roots.ndim != 2 or roots.shape[-1] != 5:
                raise ValueError(
                    f"Invalid root shape from Edge {edge_id}: "
                    f"expected [num_roots, 5], got {tuple(roots.shape)}"
                )
            if not torch.isfinite(roots).all():
                raise ValueError(f"Non-finite roots from {edge_id}")
            count = getattr(self, "expected_root_count", None)
            if count is not None and roots.shape[0] != count:
                raise ValueError(f"Unexpected root candidate count from {edge_id}")
            edge_roots.append(roots)

            for item in frame.get("reid") or []:
                if isinstance(item, dict) and item.get("feature") is not None:
                    item = dict(item)
                    item["edge_id"] = edge_id
                    reid_items.append(item)
            timestamps.append(float(frame.get("time", batch.timestamp)))

        heatmaps = torch.stack(edge_heatmaps).to(
            device=self.device,
            dtype=torch.float32,
            non_blocking=True,
        )
        heatmaps = heatmaps.div_(255).permute(1, 0, 2, 3, 4).contiguous()
        roots = torch.stack(edge_roots).to(device=self.device, non_blocking=True)
        return list(heatmaps.unbind(0)), roots, reid_items, timestamps

    def close(self) -> None:
        self.subscriber.close()
        for buffer in self.buffers.values():
            buffer.clear()
