# SPDX-FileCopyrightText: 2025-2026 DT_SERVER contributors
# SPDX-License-Identifier: LGPL-2.1-or-later
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT),str(ROOT/'packages/dt_common/src')]
import numpy as np
import pytest
from dt_common.contracts.scene import PersonEntity,PoseOutput,RootOutput
from apps.server_worker.domain.appearance_roots import AppearanceHypothesis,AppearanceRootConfig,AppearanceRootGate


def hypothesis(index=0,position=0.,feature=(1.,0.,0.),views=('1','2'),error=5.,edge='edge_1',instance=None):
    pose = np.zeros((15,3)); pose[:,0] = position
    person = PersonEntity(-index-1,2,RootOutput(edge,index,[position,0.,0.],.8,0.),
                          PoseOutput('voxelpose_15j_xyz',pose.tolist()))
    return AppearanceHypothesis(person,{cid:dict(detection=index if instance is None else instance,error_px=error) for cid in views},
                                {cid:np.asarray(feature) for cid in views})


def confirmed(gate,**kwargs):
    for time in [0.,.1,.2,.3,.4,.5]:
        output,report = gate.process([hypothesis(**kwargs)],time)
    assert len(output) == 1
    return output[0].global_id


def test_birth_requires_temporal_multiview_evidence():
    g = AppearanceRootGate()
    for t in [0.,.01,.02,.03,.04,.05]: assert not g.process([hypothesis()],t)[0]
    assert len(g.process([hypothesis()],.2)[0]) == 0
    assert len(g.process([hypothesis()],.35)[0]) == 1


def test_single_view_never_creates_id():
    g = AppearanceRootGate()
    for t in np.arange(0,2,.1): assert g.process([hypothesis(views=('1',))],float(t))[0] == []
    assert g.next_id == 1


def test_no_forced_population_or_stale_outputs():
    g = AppearanceRootGate(AppearanceRootConfig(max_identities=5))
    identity = confirmed(g)
    assert identity == 1
    people,report = g.process([],1.)
    assert people == [] and report['confirmed_identities'] == 1


def test_dormant_identity_persists_and_reactivates():
    g = AppearanceRootGate(); identity = confirmed(g)
    g.process([],10.)
    people,report = g.process([hypothesis(position=1200.)],20.)
    assert [p.global_id for p in people] == [identity]
    assert report['accepted'][0]['reactivated'] and not report['births']


def test_dormant_single_view_does_not_force_reconnect_or_birth():
    g = AppearanceRootGate(); confirmed(g)
    people,report = g.process([hypothesis(views=('1',))],10.)
    assert people == [] and not report['births']


def test_one_root_per_id_and_exclusive_instance_across_rigs():
    g = AppearanceRootGate(); identity = confirmed(g)
    a = hypothesis(position=20.,error=5.)
    b = hypothesis(index=1,position=80.,error=15.,edge='edge_2',instance=0)
    people,report = g.process([b,a],.6)
    assert len(people) == 1 and people[0].global_id == identity
    assert people[0].root == a.person.root and people[0].pose == a.person.pose
    assert report['rejected'][0]['reason'] == '2d_instance_already_owned'


def test_similar_but_distinct_people_can_coexist():
    g = AppearanceRootGate()
    for t in np.arange(0,.7,.1):
        people,report = g.process([hypothesis(),hypothesis(index=1,position=1500.)],float(t))
    assert len(people) == 2 and len({p.global_id for p in people}) == 2


def test_distant_impossible_jump_not_assigned_old_id():
    g = AppearanceRootGate(); identity = confirmed(g)
    people,_ = g.process([hypothesis(position=10000.)],.6)
    assert not people
    assert g.next_id == identity+1


def test_tentative_discontinuity_resets_confirmation():
    g = AppearanceRootGate()
    for t in [0.,.1,.2,.3,.4]: assert not g.process([hypothesis()],t)[0]
    people,report = g.process([hypothesis()],1.)
    assert people == [] and report['confirmed_identities'] == 0


def test_closed_world_budget_does_not_create_extra_confirmed_ids():
    g = AppearanceRootGate(AppearanceRootConfig(max_identities=1)); confirmed(g)
    for t in np.arange(.6,1.6,.1):
        people,report = g.process([hypothesis(),hypothesis(index=1,position=2000.,feature=(0.,1.,0.))],float(t))
        assert len(people) == 1 and report['confirmed_identities'] == 1
    assert any(r['reason'] == 'closed_world_identity_budget' for r in report['rejected'])


def test_ambiguity_does_not_allocate_new_id():
    g = AppearanceRootGate()
    for t in np.arange(0,.7,.1):
        g.process([hypothesis(position=-200.,feature=(1.,0.,0.)),
                   hypothesis(index=1,position=200.,feature=(.5,.866,0.))],float(t))
    people,report = g.process([hypothesis(position=0.,feature=(.866,.5,0.))],.8)
    assert people == [] and not report['births'] and report['confirmed_identities'] == 2
    assert report['rejected'][0]['reason'] == 'ambiguous_existing_id'


@pytest.mark.parametrize('value',[0.,float('nan'),float('inf')])
def test_bad_timestamps_fail(value):
    g = AppearanceRootGate(); g.process([],0.)
    with pytest.raises(ValueError):g.process([],value)


@pytest.mark.parametrize('field,value',[('max_identities',0),('max_identities',1.5),('match_similarity',2.),('confirmation_hits',0)])
def test_bad_config(field,value):
    with pytest.raises(ValueError): AppearanceRootConfig(**{field:value})


def test_invalid_embedding_fails():
    g = AppearanceRootGate()
    with pytest.raises(ValueError): g.process([hypothesis(feature=(0.,0.,0.))],0.)


def test_old_memory_does_not_beat_recent_continuity():
    from apps.server_worker.domain.appearance_roots import _Identity
    from collections import deque
    g = AppearanceRootGate()
    feature = np.array([1.,0.,0.],dtype=np.float32)
    for token,seen,position in [(0,0.,1100.),(1,10.,1000.)]:
        g.identities[token] = _Identity(token,seen,np.array([position,0.,0.]),global_id=token+1,
            gallery={cid:deque([(seen,feature.copy())]) for cid in ['1','2']})
    g.next_id = 3; g.next_token = 2
    output,report = g.process([hypothesis(position=1000.)],10.1)
    assert [p.global_id for p in output] == [2]
    assert not report['accepted'][0]['reactivated']


def test_evidence_is_not_mutated():
    g = AppearanceRootGate(); h = hypothesis(feature=(5.,0.,0.))
    original = h.features
    g.process([h],0.)
    assert h.features is original
    np.testing.assert_array_equal(h.features['1'],[5.,0.,0.])


def test_unconfirmable_tentative_cannot_steal_after_budget_full():
    from copy import deepcopy
    g = AppearanceRootGate(AppearanceRootConfig(max_identities=1)); confirmed(g)
    tentative = deepcopy(g.identities[0]); tentative.token = 1; tentative.global_id = None
    g.identities[1] = tentative
    output,report = g.process([hypothesis()],.6)
    assert [p.global_id for p in output] == [1]
    assert report['tentative_identities'] == 0 and set(g.identities) == {0}
