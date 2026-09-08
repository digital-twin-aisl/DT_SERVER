import copy

import numpy as np
import pytest

from apps.server_worker.poc_evaluation import Trajectories, compare_decisions
from apps.server_worker.priority_engine import PriorityEngine, PriorityConfig


def records(policy, selected):
    return [
        {
            "timestamp": 0.0,
            "policy": policy,
            "observations": [],
            "input_sha256": {"a": "same-packet"},
            "selected_ids": selected,
            "pose_runtime_ms": 2.0,
            "configuration": {
                "lambda_aoi": 0,
                "priority_interval": 0.5,
                "hazards_metres": [[0, 0]],
            },
        }
    ]


def truth():
    return Trajectories(
        [
            {"global_id": key, "timestamp_s": t, "x_mm": x, "y_mm": 0}
            for key, x in [(1, 0), (2, 10000)]
            for t in np.arange(0, 3, 0.1)
        ]
    )


def test_oracle_missed_hazard_is_lower_when_dangerous_object_selected():
    engine = PriorityEngine(PriorityConfig(hazards=((0, 0),), lambda_aoi=0))
    summary, rows = compare_decisions(
        records("rank", [1]),
        records("zone", [2]),
        truth(),
        engine,
        interval=0.5,
        warmup_batches=0,
    )
    assert rows[0]["rank_missed_hazard"] < rows[0]["zone_missed_hazard"]
    assert rows[0]["rank_selected_count"] == rows[0]["zone_selected_count"] == 1
    assert summary["future_horizon_s"] == 2.0
    assert summary["rank"]["pose_runtime"]["mean_ms"] == 2


def test_oracle_uses_actual_future_path_instead_of_linear_prediction():
    trajectories = Trajectories(
        [
            {"global_id": 1, "timestamp_s": t, "x_mm": x, "y_mm": 0}
            for t, x in [(0, 4000), (0.5, 4000), (1, 0)]
        ],
        max_gap=0.5,
    )
    engine = PriorityEngine(
        PriorityConfig(hazards=((0, 0),), prediction_steps=2, lambda_aoi=0)
    )
    actual, missing = trajectories.scores(0, engine, 0.5)
    predicted = engine.assign_priority(
        {"persons": [{"track_id": 1, "position": {"x": 4, "y": 0}}]}, timestamp=0
    )["persons"][0]["hazard_score"]
    assert not missing
    assert actual[1] != pytest.approx(predicted)


def test_incomplete_future_is_excluded_instead_of_extrapolated():
    engine = PriorityEngine(PriorityConfig(hazards=((0, 0),), lambda_aoi=0))
    rank, zone = records("rank", [1]), records("zone", [2])
    rank[0]["timestamp"] = zone[0]["timestamp"] = 2.5
    summary, rows = compare_decisions(rank, zone, truth(), engine, interval=0.5)
    assert summary["evaluated_batches"] == 0
    assert not rows[0]["evaluated"]


def test_comparison_rejects_different_replayed_observations():
    rank, zone = records("rank", [1]), copy.deepcopy(records("zone", [2]))
    zone[0]["observations"] = [{"track_id": 999}]
    with pytest.raises(ValueError, match="identical observations"):
        compare_decisions(rank, zone, truth(), PriorityEngine(), interval=0.5)


def test_comparison_rejects_different_heatmaps_despite_equal_roots():
    rank, zone = records("rank", [1]), records("zone", [2])
    zone[0]["input_sha256"] = {"a": "different-packet"}
    with pytest.raises(ValueError, match="identical recorded packets"):
        compare_decisions(rank, zone, truth(), PriorityEngine(), interval=0.5)


def test_gap_at_current_time_does_not_silently_remove_oracle_object():
    trajectories = Trajectories(
        [
            {"global_id": 1, "timestamp_s": t, "x_mm": 0, "y_mm": 0}
            for t in [0, 0.1, 1.0, 1.1]
        ]
    )
    scores, incomplete = trajectories.scores(0.5, PriorityEngine(), 0.5)
    assert scores == {} and incomplete == [1]
