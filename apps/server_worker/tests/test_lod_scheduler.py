from apps.server_worker.lod_scheduler import LodScheduler
from apps.server_worker.priority_engine import PriorityEngine, PriorityConfig


def person(key, x, edge="a"):
    return {"track_id": key, "edge_id": edge, "position": {"x": x, "y": 0, "z": 0}}


def scheduler(**kwargs):
    return LodScheduler(
        PriorityEngine(
            PriorityConfig(hazards=((0, 0),), lambda_aoi=0, prediction_steps=0)
        ),
        **kwargs,
    )


def test_rank_and_lod_held_for_half_second_and_newcomer_defaults_low():
    policy = scheduler(interval=0.5, lod2_count=1)
    assert policy.assign([person(1, 0), person(2, 10)], 0) == {1: 2, 2: 1}
    assert policy.assign([person(1, 10), person(2, 0), person(3, 0)], 0.1) == {
        1: 2,
        2: 1,
        3: 1,
    }
    assert not policy.updated
    assert policy.assign([person(1, 10), person(2, 0)], 0.5) == {1: 1, 2: 2}
    assert policy.updated


def test_partial_updates_share_one_lod_budget():
    policy = scheduler(interval=0.5, lod2_count=1)
    policy.assign([person(1, 5, "a")], 0)
    assert policy.assign([person(2, 0, "b")], 0.1) == {2: 1}
    assert policy.assign([person(1, 5, "a")], 0.5) == {1: 1}
    assert policy.assignments == {1: 1, 2: 2}


def test_zone_baseline_uses_configured_edge_regions():
    policy = scheduler(policy="zone", edge_lods={"a": 2, "b": 1})
    assert policy.assign([person(1, 100, "a"), person(2, 0, "b")], 0) == {1: 2, 2: 1}


def test_time_rewind_resets_old_held_decision():
    policy = scheduler(interval=0.5)
    policy.assign([person(1, 0)], 10)
    assert policy.assign([person(2, 0)], 0) == {2: 2}
    assert set(policy.observations) == {2}


def test_confirmed_identity_keeps_held_lod_without_duplicate_budget():
    policy = scheduler(interval=0.5)
    policy.assign([person(1000000000, 0), person(2, 2)], 0)
    policy.merge_identities({1000000000: 1})
    assert policy.assign([person(1, 0)], 0.1) == {1: 2}
    assert set(policy.observations) == {1, 2}
    assert policy.assignments == {1: 2, 2: 1}
    assert len(policy.decision["persons"]) == 2
    assert not policy.updated


def test_all_policy_includes_newcomers_between_decision_ticks():
    policy = scheduler(policy="all")
    policy.assign([person(1, 0)], 0)
    assert policy.assign([person(2, 10)], 0.1) == {2: 2}


def test_velocity_uses_observation_time_when_retained_at_later_decision():
    policy = scheduler(interval=0.5)
    policy.assign([{**person(1, 0), "timestamp": 0}], 0)
    policy.assign([{**person(1, 0.4), "timestamp": 0.4}], 0.5)
    policy.assign([], 1.0)
    policy.assign([{**person(1, 1.4), "timestamp": 1.4}], 1.5)
    # Retaining the 0.4 s sample at the 1.0 s decision must not overwrite
    # its observation clock and inflate the subsequent velocity.
    assert policy.engine._tracks[1].velocity_x == 1.0
