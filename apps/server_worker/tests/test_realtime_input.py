import queue
import time

import numpy as np
import torch
import pytest

from dt_common.inference_codec import encode_frame
from apps.server_worker.src.protocol.synchronization import (
    CompressedMessage,
    InputScheduler,
)
from apps.server_worker.src.protocol.zenoh import ZenohDataLoader
from apps.server_worker.src.protocol.replay import ReplayDataLoader


def frame(edge, stamp, sequence=0, *, clock="live", session="run", **extra):
    return {
        "time": stamp,
        "edge_id": edge,
        "sequence": sequence,
        "session_id": session,
        "timestamp_kind": "receive_unix" if clock == "live" else "dataset_relative",
        "frame_timestamps": [stamp] * 4,
        "camera_ids": [1, 2, 3, 4],
        "allheatmaps": [
            np.full((1, 15, 128, 240), 255, dtype=np.uint8) for _ in range(4)
        ],
        "roots": np.zeros((1, 10, 5), dtype=np.float32),
        "reid": [],
        **extra,
    }


def message(edge, stamp, sequence=0, **extra):
    return CompressedMessage(stamp, encode_frame(frame(edge, stamp, sequence, **extra)))


class MemorySubscriber:
    def __init__(self, edges):
        self.queues = {edge: queue.Queue() for edge in edges}

    def get(self, edge, timeout=0):
        try:
            return self.queues[edge].get_nowait()
        except queue.Empty:
            return None

    def close(self):
        pass


def test_independent_edge_keeps_working_and_second_edge_recovers():
    scheduler = InputScheduler(["a", "b"])
    now = time.time()
    scheduler.add("a", message("a", now), now=now)
    assert set(scheduler.select(now=now).frames) == {"a"}
    scheduler.add("b", message("b", now + 0.05), now=now + 0.05)
    assert set(scheduler.select(now=now + 0.05).frames) == {"b"}


def test_slow_edge_is_processed_even_when_fast_edge_has_newer_timestamp():
    source = MemorySubscriber(["a", "b"])
    loader = ZenohDataLoader(["a", "b"], torch.device("cpu"), subscriber=source)
    now = time.time()
    source.queues["a"].put(message("a", now))
    first = loader.get(0.1)
    source.queues["b"].put(message("b", now - 0.2))
    second = loader.get(0.1)
    assert second.edge_ids == ("b",)
    assert second.timestamps == [now - 0.2]
    assert second.timestamp == first.timestamp
    assert loader.edge_status() == {"a": "online", "b": "online"}


def test_stale_future_and_wrong_clock_inputs_are_rejected():
    scheduler = InputScheduler(["a"])
    now = time.time()
    for stamp in (now - 60, now + 10, 1.0):
        scheduler.add("a", message("a", stamp), now=now)
    assert scheduler.select(now=now) is None
    assert scheduler.rejected["a"] == 3


def test_queue_age_rechecked_and_duplicate_sequence_rejected():
    scheduler = InputScheduler(["a"])
    now = time.time()
    scheduler.add("a", message("a", now), now=now)
    assert scheduler.select(now=now + 1) is None
    scheduler.add("a", message("a", now + 2, 5), now=now + 2)
    assert scheduler.select(now=now + 2)
    scheduler.add("a", message("a", now + 2.1, 5), now=now + 2.1)
    assert scheduler.select(now=now + 2.1) is None
    scheduler.add("a", message("a", now + 2.2, 0, session="restart"), now=now + 2.2)
    assert scheduler.select(now=now + 2.2)


def test_strict_dataset_preserves_earliest_matching_pair():
    scheduler = InputScheduler(["a", "b"], clock="dataset", mode="strict")
    for edge, stamp, seq in [
        ("a", 0.0, 0),
        ("a", 1.0, 1),
        ("b", 0.01, 0),
        ("b", 0.9, 1),
    ]:
        scheduler.add(edge, message(edge, stamp, seq, clock="dataset"))
    batch = scheduler.select()
    assert batch.timestamp == 0.01
    assert batch.frames["a"]["sequence"] == 0


def test_camera_skew_not_hidden_by_edge_timestamp():
    scheduler = InputScheduler(["a"])
    now = time.time()
    scheduler.add(
        "a", message("a", now, frame_timestamps=[now, now, now, now + 0.5]), now=now
    )
    assert scheduler.select(now=now) is None


def test_bad_calibration_on_one_edge_does_not_discard_valid_edge():
    source = MemorySubscriber(["a", "b"])
    loader = ZenohDataLoader(
        ["a", "b"],
        torch.device("cpu"),
        subscriber=source,
        expected_calibration_digests={"a": "correct", "b": "correct"},
    )
    now = time.time()
    source.queues["a"].put(message("a", now, calibration_digest="correct"))
    source.queues["b"].put(message("b", now, calibration_digest="wrong"))
    batch = loader.get(0.1)
    assert batch.edge_ids == ("a",)
    assert len(batch.heatmaps) == 4 and batch.heatmaps[0].shape == (1, 15, 128, 240)
    assert batch.heatmaps[0].min() == 1
    assert loader.prepare_failures == 1
    assert loader.edge_status() == {"a": "online", "b": "offline"}


def test_lossless_step_replay_and_eof(tmp_path):
    for edge in ("a", "b"):
        (tmp_path / edge).mkdir()
        for index in range(3):
            (tmp_path / edge / f"{index:09d}.dtframe").write_bytes(
                message(edge, index * 0.1, index, clock="dataset").payload
            )
    loader = ReplayDataLoader(
        tmp_path,
        edge_ids=["a", "b"],
        device=torch.device("cpu"),
        input_mode="strict",
        input_clock="dataset",
    )
    batches = [loader.get() for _ in range(3)]
    assert [batch.timestamp for batch in batches] == [0.0, 0.1, 0.2]
    assert all(batch.edge_ids == ("a", "b") for batch in batches)
    assert loader.get() is None and loader.finished


@pytest.mark.parametrize("failure", ["truncated", "desynchronized", "corrupt", "empty"])
def test_replay_refuses_incomplete_comparisons(tmp_path, failure):
    for edge in ("a", "b"):
        (tmp_path / edge).mkdir()
        count = 1 if edge == "b" and failure == "truncated" else 2
        if edge == "b" and failure == "empty":
            count = 0
        for index in range(count):
            stamp = index * 0.1 + (
                0.05 if edge == "b" and failure == "desynchronized" else 0
            )
            extra = (
                {"timestamp_kind": "wrong"}
                if edge == "b" and failure == "corrupt"
                else {}
            )
            payload = message(edge, stamp, index, clock="dataset", **extra).payload
            (tmp_path / edge / f"{index:09d}.dtframe").write_bytes(payload)
    with pytest.raises(ValueError):
        loader = ReplayDataLoader(
            tmp_path,
            edge_ids=["a", "b"],
            device=torch.device("cpu"),
            input_mode="strict",
            input_clock="dataset",
        )
        for _ in range(3):
            loader.get()
