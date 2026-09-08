from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest

from apps.edge_client.src.utils.input import FrameUnavailableError, snapshot_live_bundle


def inputs():
    now = time.monotonic()
    # The fastest camera's newest frame is 100 ms ahead. Its previous frame
    # aligns with the slower camera, which a one-slot buffer cannot recover.
    process = SimpleNamespace(
        buffer_depth=2,
        frame_locks=[threading.Lock(), threading.Lock()],
        camera_ids=[2, 4],
        frame_timestamps=[now - 0.1, now, now - 0.2, now - 0.101],
        frame_wall_timestamps=[100.1, 100.2, 100.0, 100.099],
        frame_sequences=[2, 3, 1, 2],
    )
    arrays = [
        np.array([[[[2]]], [[[3]]]], dtype=np.uint8),
        np.array([[[[1]]], [[[2]]]], dtype=np.uint8),
    ]
    buffers = [
        (SimpleNamespace(buf=bytearray(array.tobytes())), array.shape, array.dtype)
        for array in arrays
    ]
    return process, buffers


def test_ring_selects_matching_older_frame_and_preserves_metadata():
    process, buffers = inputs()
    bundle = snapshot_live_bundle(buffers, process, max_age=0.75, max_skew=0.03)
    assert [int(frame.item()) for frame in bundle.frames] == [2, 2]
    assert bundle.sequences == [2, 2]
    assert bundle.timestamps == [100.1, 100.099]
    assert bundle.timestamp_kind == "receive_unix"


def test_duplicate_camera_frame_is_not_inferred_twice():
    process, buffers = inputs()
    with pytest.raises(FrameUnavailableError, match="no new fresh frame"):
        snapshot_live_bundle(
            buffers, process, max_age=0.75, max_skew=0.03, previous_sequences=[2, 2]
        )


def test_no_frame_is_returned_when_camera_disconnected():
    process, buffers = inputs()
    process.frame_timestamps[2:] = [0, 0]
    with pytest.raises(FrameUnavailableError):
        snapshot_live_bundle(buffers, process, max_age=0.75, max_skew=0.03)
