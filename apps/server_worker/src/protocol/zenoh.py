import json
import logging
import math
import pickle
import queue
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import torch
import zenoh
import zstandard as zstd


logger = logging.getLogger(__name__)
PAYLOAD_HEADER = struct.Struct("!4sd")
PAYLOAD_MAGIC = b"ZNH1"

@dataclass(frozen=True, slots=True)
class CompressedMessage:
    timestamp: float
    payload: bytes

@dataclass(frozen=True, slots=True)
class SynchronizedBatch:
    timestamp: float
    timestamp_spread: float
    frames: dict[str, Any]

PreparedInputs = tuple[
    list[torch.Tensor],
    torch.Tensor,
    list[dict[str, Any]],
    list[float],
    float,
]


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
        self.closed = False
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
                magic, timestamp = PAYLOAD_HEADER.unpack_from(payload)
                if magic != PAYLOAD_MAGIC or not math.isfinite(timestamp):
                    raise ValueError("Invalid payload header")
                message = CompressedMessage(timestamp, payload[PAYLOAD_HEADER.size:])
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
                logger.exception("Invalid Zenoh message from %s", edge_id)

        try:
            if self.config_path:
                config = zenoh.Config.from_file(self.config_path)
            elif self.endpoint:
                endpoint = self.endpoint
                if "/" not in endpoint:
                    endpoint = f"tcp/{endpoint}"
                config_data = {"mode": "client", "connect": {"endpoints": [endpoint]}}
                config = zenoh.Config.from_json5(json.dumps(config_data))
            else:
                config = zenoh.Config()
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
        endpoint: str | None = None,
        config_path: str | None = None,
        topic_template: str = "edge/{edge_id}",
        buffer_size: int = 10,
        sync_tolerance: float = 0.03,
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
        self.device = device
        self.sync_tolerance = sync_tolerance
        self.buffers = {edge_id: deque(maxlen=buffer_size) for edge_id in edge_ids}
        self.subscriber = ZenohSubscriber(topics, endpoint, config_path, buffer_size)
        self._decompressor = zstd.ZstdDecompressor()
        self._batch_count = 0
        logger.info(
            "Zenoh data loader is initialized: edges=%s, sync_tolerance=%.3fs",
            self.edge_ids,
            self.sync_tolerance,
        )

    def get(self, timeout: float = 1.0) -> PreparedInputs | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for edge_id, buffer in self.buffers.items():
                if not buffer:
                    remaining = max(0.0, deadline - time.monotonic())
                    message = self.subscriber.get(edge_id, remaining)
                    if message is None:
                        return None
                    buffer.append(message)
                while (message := self.subscriber.get(edge_id, 0)) is not None:
                    buffer.append(message)
            watermark = min(buffer[-1].timestamp for buffer in self.buffers.values())
            selected = {}
            for edge_id, buffer in self.buffers.items():
                index = min(
                    range(len(buffer)),
                    key=lambda i: abs(buffer[i].timestamp - watermark),
                )
                selected[edge_id] = index, buffer[index]
            timestamps = [message.timestamp for _, message in selected.values()]
            spread = max(timestamps) - min(timestamps)
            if spread <= self.sync_tolerance:
                for edge_id, (index, _) in selected.items():
                    for _ in range(index + 1):
                        self.buffers[edge_id].popleft()
                try:
                    frames = {
                        edge_id: pickle.loads(
                            self._decompressor.decompress(message.payload)
                        )
                        for edge_id, (_, message) in selected.items()
                    }
                except Exception:
                    logger.exception("Failed to decode synchronized batch")
                    continue
                batch = SynchronizedBatch(
                    sum(timestamps) / len(timestamps),
                    spread,
                    frames,
                )
                try:
                    heatmaps, roots, reid_items, frame_timestamps = (
                        self._prepare_inputs(batch)
                    )
                except Exception:
                    logger.exception("Failed to prepare synchronized batch")
                    continue
                self._batch_count += 1
                if self._batch_count == 1:
                    logger.info(
                        "First Zenoh batch prepared: views=%d, reid_items=%d",
                        len(heatmaps),
                        len(reid_items),
                    )
                return heatmaps, roots, reid_items, frame_timestamps, spread
            oldest_edge = min(selected, key=lambda edge_id: selected[edge_id][1].timestamp)
            oldest_index = selected[oldest_edge][0]
            for _ in range(oldest_index + 1):
                self.buffers[oldest_edge].popleft()
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

        for edge_id in self.edge_ids:
            frame = batch.frames[edge_id]
            heatmap_values = frame.get("allheatmaps")
            root_values = frame.get("roots")
            if not heatmap_values or root_values is None:
                raise ValueError(f"Missing heatmaps or roots from Edge {edge_id}")

            views = []
            for value in heatmap_values:
                heatmap = torch.as_tensor(value)
                if heatmap.ndim == 4 and heatmap.shape[0] == 1:
                    heatmap = heatmap.squeeze(0)
                if heatmap.ndim != 3:
                    raise ValueError(f"Invalid heatmap shape from Edge {edge_id}")
                views.append(heatmap)

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
