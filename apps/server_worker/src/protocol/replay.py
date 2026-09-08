"""Lossless, step-driven replay of per-edge ZNH2 recordings (no Zenoh required)."""

from pathlib import Path
import queue

from dt_common.inference_codec import MAX_MESSAGE_BYTES, inspect_message
from .synchronization import CompressedMessage
from .zenoh import ZenohDataLoader


class ReplaySubscriber:
    def __init__(self, root, edge_ids):
        paths = {
            edge: sorted((Path(root) / edge).glob("*.dtframe")) for edge in edge_ids
        }
        if any(not (Path(root) / edge).is_dir() for edge in edge_ids):
            raise ValueError(
                "replay requires one recording directory for each configured edge"
            )
        if any(not items for items in paths.values()):
            raise ValueError("each replay edge requires at least one recorded frame")
        self.paths = {edge: iter(items) for edge, items in paths.items()}
        self.queues = {edge: queue.Queue(1) for edge in edge_ids}
        self.dropped = dict.fromkeys(edge_ids, 0)
        self.received_messages = dict.fromkeys(edge_ids, 0)
        self.received_bytes = dict.fromkeys(edge_ids, 0)
        self.invalid_messages = dict.fromkeys(edge_ids, 0)
        self.exhausted = set()

    def fill(self, buffers):
        for edge, paths in self.paths.items():
            if buffers[edge] or not self.queues[edge].empty() or edge in self.exhausted:
                continue
            path = next(paths, None)
            if path is None:
                self.exhausted.add(edge)
                continue
            if path.stat().st_size > MAX_MESSAGE_BYTES:
                raise ValueError(f"recording packet too large: {path}")
            payload = path.read_bytes()
            timestamp, _ = inspect_message(payload)
            self.queues[edge].put_nowait(CompressedMessage(timestamp, payload))
            self.received_messages[edge] += 1
            self.received_bytes[edge] += len(payload)

    def get(self, edge, timeout=0):
        try:
            return self.queues[edge].get_nowait()
        except queue.Empty:
            return None

    def close(self):
        pass


class ReplayDataLoader(ZenohDataLoader):
    def __init__(self, root, **kwargs):
        source = ReplaySubscriber(root, kwargs["edge_ids"])
        super().__init__(subscriber=source, **kwargs)
        self.finished = False

    def get(self, timeout=1.0):
        while True:
            self.subscriber.fill(self.buffers)
            drained = [
                edge
                for edge in self.subscriber.exhausted
                if not self.buffers[edge] and self.subscriber.queues[edge].empty()
            ]
            if (self.scheduler.mode == "strict" and drained) or len(drained) == len(
                self.edge_ids
            ):
                if len(drained) != len(self.edge_ids):
                    raise ValueError(
                        "recordings end at different frames; refusing a truncated comparison"
                    )
                self.finished = True
                return None
            prepared = super().get(timeout=min(timeout, 0.005))
            if (
                any(self.scheduler.discarded.values())
                or any(self.scheduler.rejected.values())
                or self.prepare_failures
                or self.decode_failures
            ):
                raise ValueError(
                    "replay input rejected or synchronization lost; refusing an incomplete comparison"
                )
            if prepared is not None:
                return prepared
