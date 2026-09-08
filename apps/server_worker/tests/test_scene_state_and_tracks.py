import torch

from apps.server_worker.inference import SceneOutput, RootOutput, PersonEntity
from apps.server_worker.root_tracks import RootTracks, TEMPORARY_ID_START
from apps.server_worker.scene_state import SceneState


def scene(edge, stamp, key):
    return SceneOutput(
        stamp,
        0,
        [PersonEntity(key, 2, RootOutput(edge, 0, [0, 0, 0], 0.9, stamp), None)],
    )


def test_partial_scene_retains_fresh_edge_but_expires_offline_people():
    state = SceneState(["a", "b"], ttl=0.75)
    state.update(scene("a", 1, 1), ["a"], 1, {1: 2}, 10)
    result = state.update(scene("b", 1.1, 2), ["b"], 1.1, {1: 1, 2: 2}, 10)
    assert {person.global_id: person.lod for person in result.people} == {1: 1, 2: 2}
    result = state.update(SceneOutput(2, 0, []), [], 2, {1: 1, 2: 2}, 10)
    assert result.people == []


def test_empty_observation_clears_people_from_that_edge_immediately():
    state = SceneState(["a"])
    state.update(scene("a", 1, 1), ["a"], 1, {1: 2}, 10)
    assert state.update(SceneOutput(1.1, 0, []), ["a"], 1.1, {}, 10).people == []


def test_confirmed_id_replaces_tentative_id_on_retained_edge():
    state = SceneState(["a", "b"])
    state.update(
        scene("a", 1, TEMPORARY_ID_START), ["a"], 1, {TEMPORARY_ID_START: 2}, 10
    )
    state.merge_identities({TEMPORARY_ID_START: 7})
    result = state.update(scene("b", 1.1, 7), ["b"], 1.1, {7: 2}, 10)
    assert len(result.people) == 1 and result.people[0].global_id == 7


def test_reid_failure_uses_stable_spatial_id_then_links_confirmed_id():
    tracker = RootTracks()
    roots = torch.tensor([[[1000, 2000, 0, 0, 0.9]]])
    first = tracker.assign([[None]], roots, ["a"], [1])
    identity = first[0][0]
    assert identity >= TEMPORARY_ID_START
    assert tracker.assign([[None]], roots, ["a"], [1.1]) == first
    assert tracker.assign([[7]], roots, ["a"], [1.2]) == [[7]]
    assert tracker.aliases == {identity: 7}
    assert tracker.assign([[None]], roots, ["a"], [1.3]) == [[7]]


def test_cross_edge_duplicate_root_does_not_create_second_temporary_person():
    tracker = RootTracks()
    roots = torch.tensor([[[1000, 2000, 0, 0, 0.9]], [[1050, 2000, 0, 0, 0.8]]])
    result = tracker.assign([[None], [None]], roots, ["a", "b"], [1, 1])
    assert sum(identity is not None for edge in result for identity in edge) == 1


def test_late_edge_does_not_expire_or_rewind_newer_spatial_track():
    tracker = RootTracks()
    roots = torch.tensor([[[1000, 2000, 0, 0, 0.9]]])
    identity = tracker.assign([[None]], roots, ["a"], [1])[0][0]
    assert tracker.assign([[None]], roots, ["b"], [0.8]) == [[identity]]
    assert tracker.tracks[identity][1] == 1
