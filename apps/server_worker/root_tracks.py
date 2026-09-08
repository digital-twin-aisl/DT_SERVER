"""Short-lived spatial fallback when appearance observations are unavailable."""

import math

import numpy as np


TEMPORARY_ID_START = 1_000_000_000


class RootTracks:
    def __init__(self, ttl=0.75, match_distance_mm=750, duplicate_distance_mm=250):
        self.ttl = ttl
        self.match_distance = match_distance_mm
        self.duplicate_distance = duplicate_distance_mm
        self.tracks = {}
        self.next_id = TEMPORARY_ID_START
        self.aliases = {}
        self.latest_timestamp = -math.inf

    def assign(self, ids, roots, edge_ids, timestamps):
        now = max(self.latest_timestamp, max(timestamps))
        self.latest_timestamp = now
        self.tracks = {
            key: state
            for key, state in self.tracks.items()
            if 0 <= now - state[1] <= self.ttl
        }
        self.aliases = {}
        result = [list(values) for values in ids]
        values = roots.detach().cpu().numpy()
        candidates = []
        used = set()
        selected = []
        for edge_index, edge in enumerate(edge_ids):
            for index, root in enumerate(values[edge_index]):
                if root[3] < 0 or not np.isfinite(root).all():
                    continue
                identity = result[edge_index][index]
                if identity is None:
                    candidates.append((float(root[4]), edge_index, index, root[:3]))
                    continue
                used.add(identity)
                selected.append((edge, identity, root[:3]))
                nearby_temporary = [
                    (np.linalg.norm(root[:3] - state[0]), key)
                    for key, state in self.tracks.items()
                    if key >= TEMPORARY_ID_START
                ]
                if nearby_temporary:
                    distance, old = min(nearby_temporary)
                    if distance <= self.match_distance:
                        self.aliases[old] = identity
                        self.tracks.pop(old)
                if (
                    identity not in self.tracks
                    or timestamps[edge_index] >= self.tracks[identity][1]
                ):
                    self.tracks[identity] = (root[:3].copy(), timestamps[edge_index])
        for _, edge_index, index, position in sorted(
            candidates, key=lambda item: (-item[0], item[1], item[2])
        ):
            edge = edge_ids[edge_index]
            if any(
                other_edge != edge
                and np.linalg.norm(position - other) <= self.duplicate_distance
                for other_edge, _, other in selected
            ):
                continue
            choices = [
                (np.linalg.norm(position - state[0]), key)
                for key, state in self.tracks.items()
                if key not in used
            ]
            distance, identity = min(choices, default=(math.inf, None))
            if distance > self.match_distance:
                identity = self.next_id
                self.next_id += 1
            result[edge_index][index] = identity
            used.add(identity)
            selected.append((edge, identity, position))
            if (
                identity not in self.tracks
                or timestamps[edge_index] >= self.tracks[identity][1]
            ):
                self.tracks[identity] = (position.copy(), timestamps[edge_index])
        return result
