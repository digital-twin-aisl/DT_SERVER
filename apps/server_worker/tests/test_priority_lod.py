from pathlib import Path
import sys

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from apps.server_worker import inference  # noqa: E402
from apps.server_worker.priority_engine import (  # noqa: E402
    PriorityConfig,
    PriorityEngine,
)


def test_build_priority_input_maps_ids_to_roots_in_metres():
    roots = torch.tensor(
        [
            [[0, 0, 0, -1, 0.1], [1200, -800, 300, 0, 0.9]],
            [[-500, 2000, 100, 0, 0.8], [0, 0, 0, -1, 0.1]],
        ],
        dtype=torch.float32,
    )

    result = inference.build_priority_input(
        ids=[[None, 5], [7, None]],
        roots=roots,
    )

    assert [person["track_id"] for person in result["persons"]] == [5, 7]
    assert result["persons"][0]["position"] == pytest.approx(
        {"x": 1.2, "y": -0.8, "z": 0.3}
    )
    assert result["persons"][1]["position"] == pytest.approx(
        {"x": -0.5, "y": 2.0, "z": 0.1}
    )


def test_assign_priority_lods_selects_nearest_person_for_lod2():
    engine = PriorityEngine(
        PriorityConfig(
            hazards=((0.0, 0.0),),
            prediction_steps=0,
            sigma_d=1.0,
            lambda_aoi=0.0,
        )
    )
    roots = torch.tensor(
        [[[0, 0, 0, 0, 0.9], [4000, 0, 0, 0, 0.8]]],
        dtype=torch.float32,
    )

    assignments = inference.assign_priority_lods(
        ids=[[10, 20]],
        roots=roots,
        timestamps=[100.0],
        engine=engine,
        lod2_count=1,
    )

    assert assignments == {10: 2, 20: 1}


def test_assign_priority_lods_keeps_aoi_state_between_frames():
    engine = PriorityEngine(
        PriorityConfig(
            hazards=((0.0, 0.0),),
            prediction_steps=0,
            sigma_d=1.0,
            sigma_aoi=1.0,
            lambda_aoi=10.0,
        )
    )
    roots = torch.tensor(
        [[[0, 0, 0, 0, 0.9], [4000, 0, 0, 0, 0.8]]],
        dtype=torch.float32,
    )

    first = inference.assign_priority_lods(
        ids=[[10, 20]],
        roots=roots,
        timestamps=[100.0],
        engine=engine,
        lod2_count=1,
    )
    second = inference.assign_priority_lods(
        ids=[[10, 20]],
        roots=roots,
        timestamps=[101.0],
        engine=engine,
        lod2_count=1,
    )

    assert first == {10: 2, 20: 1}
    assert second == {10: 1, 20: 2}


def test_assign_priority_lods_rejects_negative_lod2_count():
    with pytest.raises(ValueError, match="lod2_count"):
        inference.assign_priority_lods(
            ids=[[1]],
            roots=torch.zeros((1, 1, 5), dtype=torch.float32),
            timestamps=[100.0],
            engine=PriorityEngine(),
            lod2_count=-1,
        )
