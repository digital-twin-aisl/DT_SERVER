# SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Causal, opt-in identity bank: no anonymous root fallback and no pose synthesis.

IDs require persistent multi-view RGB evidence. A confirmed ID survives missing
observations, but never produces a stale/extrapolated root. Reset per recording.
The optional closed-world identity budget is a user-supplied prior, not evidence.
"""
from collections import deque
from dataclasses import asdict, dataclass, field, replace
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from .pose_duplicates import candidate_key, valid_pose


@dataclass(frozen=True)
class AppearanceRootConfig:
    min_birth_views: int = 2
    confirmation_hits: int = 6
    confirmation_seconds: float = .3
    tentative_gap_seconds: float = .25
    active_seconds: float = 1.
    match_similarity: float = .70
    dormant_similarity: float = .80
    ambiguity_margin: float = .04
    birth_exclusion_similarity: float = .65
    position_slack_mm: float = 500.
    max_speed_mm_s: float = 2500.
    gallery_size: int = 12
    gallery_interval_seconds: float = .25
    max_identities: int | None = None

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key == 'max_identities' and value is None: continue
            if not math.isfinite(value) or value <= 0: raise ValueError(key)
        for key in ['min_birth_views','confirmation_hits','gallery_size','max_identities']:
            value = getattr(self,key)
            if value is not None and type(value) is not int: raise ValueError(key)
        for key in ['match_similarity','dormant_similarity','birth_exclusion_similarity','ambiguity_margin']:
            if getattr(self,key) > 1: raise ValueError(key)


@dataclass
class AppearanceHypothesis:
    person: object
    views: dict  # camera -> {detection: frame-local instance index, error_px: ...}
    features: dict  # camera -> normalized RGB appearance vector

    @property
    def position(self): return valid_pose(self.person)[2]

    @property
    def tokens(self): return {(str(cid),int(v['detection'])) for cid,v in self.views.items()}

    @property
    def error(self): return float(np.median([v['error_px'] for v in self.views.values()]))


@dataclass
class _Identity:
    token: int
    last: float
    position: np.ndarray
    global_id: int | None = None
    gallery: dict = field(default_factory=dict)
    strong_times: list = field(default_factory=list)


def _unit(value):
    value = np.asarray(value,dtype=np.float32)
    if value.ndim != 1 or not np.isfinite(value).all() or np.linalg.norm(value) < 1e-6:
        raise ValueError('Invalid appearance vector')
    return value/np.linalg.norm(value)


class AppearanceRootGate:
    def __init__(self, config=None):
        self.config = config or AppearanceRootConfig()
        self.identities = {}; self.next_token = 0; self.next_id = 1
        self.latest = -math.inf

    def similarity(self, identity, hypothesis):
        # Prefer same-camera appearance; cross-view fallback is required at rig handoff.
        # Median of the best three samples avoids one accidental gallery outlier.
        values = []
        for cid, feature in hypothesis.features.items():
            references = identity.gallery.get(cid)
            if references is None:
                references = [v for history in identity.gallery.values() for v in history]
            if not references: continue
            scores = sorted(float(np.dot(feature,item[1])) for item in references)
            values.append(float(np.median(scores[-3:])))
        return float(np.median(values)) if values else -1.

    def _update(self, identity, hypothesis, now):
        identity.last = now; identity.position = hypothesis.position.copy()
        for cid, feature in hypothesis.features.items():
            history = identity.gallery.setdefault(cid,deque(maxlen=self.config.gallery_size))
            if not history or now-history[-1][0] >= self.config.gallery_interval_seconds:
                history.append((now,feature.copy()))
        if len(hypothesis.views) >= self.config.min_birth_views:
            if identity.strong_times and now-identity.strong_times[-1] > self.config.tentative_gap_seconds:
                identity.strong_times.clear()
            identity.strong_times.append(now)
            identity.strong_times = identity.strong_times[-120:]

    def _can_confirm(self, identity):
        times = identity.strong_times
        return (len(times) >= self.config.confirmation_hits and
                times[-1]-times[0] >= self.config.confirmation_seconds)

    def process(self, hypotheses, timestamp):
        if not math.isfinite(timestamp) or timestamp <= self.latest:
            raise ValueError('Strictly increasing timestamps required; reset per clip')
        self.latest = timestamp; c = self.config
        budget_full = c.max_identities is not None and self.next_id > c.max_identities
        self.identities = {k:v for k,v in self.identities.items()
                           if v.global_id is not None or (not budget_full and timestamp-v.last <= c.tentative_gap_seconds)}
        hs = []
        for h in hypotheses:
            if valid_pose(h.person) is None: raise ValueError('Expected finite 15-joint pose')
            if set(h.features) != set(h.views): raise ValueError('Appearance/view mismatch')
            if not h.views: continue
            # Do not mutate caller-owned evidence when comparing configurations.
            h = replace(h,features={str(cid):_unit(f) for cid,f in h.features.items()},
                        views={str(cid):v for cid,v in h.views.items()})
            if any(not math.isfinite(v['error_px']) or v['error_px'] < 0 for v in h.views.values()):
                raise ValueError('Invalid reprojection error')
            hs.append(h)
        keys = [candidate_key(h.person) for h in hs]
        if len(keys) != len(set(keys)): raise ValueError('Duplicate root candidate')
        tracks = sorted(self.identities.values(),key=lambda t:(t.global_id is None,t.token))
        costs = np.full((len(tracks),len(hs)),1e6)
        similarities = np.full_like(costs,-1.)
        admissible = np.zeros_like(costs,dtype=bool)
        for i,t in enumerate(tracks):
            dt = timestamp-t.last; dormant = dt > c.active_seconds
            for j,h in enumerate(hs):
                distance = float(np.linalg.norm(t.position-h.position))
                similarity = self.similarity(t,h); similarities[i,j] = similarity
                admissible[i,j] = distance <= c.position_slack_mm+c.max_speed_mm_s*dt
                threshold = c.dormant_similarity if dormant else c.match_similarity
                if not admissible[i,j] or similarity < threshold: continue
                if dormant and len(h.views) < c.min_birth_views: continue
                # Do not let a one-view feature pull a person to a distant root.
                if len(h.views) == 1 and distance > c.position_slack_mm: continue
                # Do not divide the ranking distance by track age: a long-lost
                # identity would otherwise beat a nearby, just-seen identity.
                # Keep the age-dependent motion gate, but prefer active tracks.
                costs[i,j] = (1-similarity + .15*min(distance/1000.,2)
                              + .10*h.error/30 - .025*min(len(h.views),4)
                              + (.20 if dormant else 0)
                              + (.15 if t.global_id is None else 0))
        ambiguous = set()
        for j in range(len(hs)):
            eligible = [i for i,t in enumerate(tracks) if t.global_id is not None and costs[i,j]<1e6]
            ranked = sorted((costs[i,j],i) for i in eligible)
            if len(ranked)>1 and ranked[1][0]-ranked[0][0] < c.ambiguity_margin:
                costs[:,j] = 1e6; ambiguous.add(j)
        assignments = []
        if costs.size:
            ii,jj = linear_sum_assignment(costs)
            assignments = sorted([(costs[i,j],i,j) for i,j in zip(ii,jj) if costs[i,j]<1e6])
        used_tokens = set(); consumed = set(); selected = []; rejected = []; accepted = []; births = []
        matched = {}
        for cost,i,j in assignments:
            h,t = hs[j],tracks[i]
            # One camera detection may never support two different roots in a frame.
            if h.tokens & used_tokens:
                rejected.append(dict(candidate=keys[j],reason='2d_instance_already_owned')); consumed.add(j); continue
            was_dormant = timestamp-t.last > c.active_seconds
            self._update(t,h,timestamp); used_tokens.update(h.tokens); consumed.add(j); matched[t.token] = h
            selected.append((t,h,similarities[i,j],was_dormant))
        for j in sorted(set(range(len(hs)))-consumed,key=lambda j:(-len(hs[j].views),hs[j].error,keys[j])):
            h = hs[j]; reason = None
            if j in ambiguous: reason = 'ambiguous_existing_id'
            elif h.tokens & used_tokens: reason = '2d_instance_already_owned'
            elif len(h.views) < c.min_birth_views: reason = 'insufficient_birth_views'
            else:
                # Ambiguous reappearance is NOT a reason to allocate a new ID.
                # A distinct simultaneous box in a common camera is a cannot-link.
                for i,t in enumerate(tracks):
                    if not admissible[i,j] or similarities[i,j] < c.birth_exclusion_similarity: continue
                    other = matched.get(t.token)
                    distinct = other is not None and any(
                        cid in other.views and view['detection'] != other.views[cid]['detection']
                        for cid,view in h.views.items())
                    if not distinct: reason = 'possible_existing_identity'; break
            if reason:
                rejected.append(dict(candidate=keys[j],reason=reason)); continue
            if c.max_identities is not None and self.next_id > c.max_identities:
                rejected.append(dict(candidate=keys[j],reason='closed_world_identity_budget')); continue
            t = _Identity(self.next_token,timestamp,h.position.copy()); self.next_token += 1
            self.identities[t.token] = t; self._update(t,h,timestamp)
            used_tokens.update(h.tokens); selected.append((t,h,None,False))
        output = []
        for t,h,similarity,reactivated in selected:
            if t.global_id is None and self._can_confirm(t):
                if c.max_identities is None or self.next_id <= c.max_identities:
                    t.global_id = self.next_id; self.next_id += 1
                    births.append(dict(global_id=t.global_id,candidate=candidate_key(h.person),
                        hits=len(t.strong_times),span_seconds=t.strong_times[-1]-t.strong_times[0]))
            if t.global_id is None: continue
            output.append(replace(h.person,global_id=t.global_id))
            accepted.append(dict(global_id=t.global_id,candidate=candidate_key(h.person),
                similarity=None if similarity is None else float(similarity),reactivated=reactivated,
                views=h.views,median_reprojection_px=h.error))
        if c.max_identities is not None and self.next_id > c.max_identities:
            # An unconfirmable tentative must not steal observations from the
            # closed-world bank on later frames.
            self.identities = {k:t for k,t in self.identities.items() if t.global_id is not None}
        report = dict(input_candidates=len(hypotheses),supported_candidates=len(hs),
            output_people=len(output),confirmed_identities=self.next_id-1,
            tentative_identities=sum(t.global_id is None for t in self.identities.values()),
            accepted=accepted,rejected=rejected,births=births,
            identity_policy='One current root per confirmed RGB identity, exclusive 2D observations, no anonymous fallback or stale pose')
        assert len(output) == len({p.global_id for p in output}) <= self.next_id-1
        return output,report
