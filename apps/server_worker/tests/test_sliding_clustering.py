from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from apps.server_worker.src.reid.sliding_clustering import (  # noqa: E402
    ClusteringSliding,
)


FEATURE_A = [1.0, 0.0, 0.0]
FEATURE_B = [0.0, 1.0, 0.0]
FEATURE_GHOST = [0.0, 0.0, 1.0]


def observation(frame, feature, *, camera=1, person_idx=0):
    return {
        "edge_id": "0_edge",
        "cam": camera,
        "frame": frame,
        "person_idx": person_idx,
        "bbox": [0, 0, 10, 10],
        "feature": feature,
    }


def global_id_for(results, feature):
    return next(
        item["global_id"]
        for item in results
        if item["feature"] == feature
    )


def test_confirmed_id_survives_temporary_detection_gap():
    tracker = ClusteringSliding(edges=["0_edge"], window_size=10)

    results = []
    for frame in range(12):
        results = tracker.process_realtime(
            [
                observation(frame, FEATURE_A, person_idx=0),
                observation(frame, FEATURE_B, person_idx=1),
            ]
        )

    original_id = global_id_for(results, FEATURE_A)
    assert {item["frame"] for item in results} == {11}

    for frame in range(12, 18):
        tracker.process_realtime(
            [observation(frame, FEATURE_B, person_idx=1)]
        )

    returned_results = []
    for frame in range(18, 23):
        returned_results = tracker.process_realtime(
            [
                observation(frame, FEATURE_A, person_idx=0),
                observation(frame, FEATURE_B, person_idx=1),
            ]
        )

    assert global_id_for(returned_results, FEATURE_A) == original_id


def test_one_batch_ghost_does_not_consume_global_id():
    tracker = ClusteringSliding(edges=["0_edge"], window_size=5)

    for frame in range(7):
        tracker.process_realtime([observation(frame, FEATURE_B)])
    assert tracker.next_global_id == 1

    ghost_batch = [observation(7, FEATURE_B)]
    ghost_batch.extend(
        observation(
            7,
            FEATURE_GHOST,
            camera=camera,
            person_idx=camera,
        )
        for camera in range(1, 6)
    )
    results = tracker.process_realtime(ghost_batch)
    assert all(item["feature"] != FEATURE_GHOST for item in results)

    for frame in range(8, 12):
        tracker.process_realtime([observation(frame, FEATURE_B)])
    assert tracker.next_global_id == 1
