"""Bounded last-observation state for partial edge updates, without extrapolation."""

from dataclasses import replace


class SceneState:
    def __init__(self, edge_ids, ttl=0.75):
        self.edge_ids = tuple(edge_ids)
        self.ttl = ttl
        self.by_edge = {}

    def merge_identities(self, aliases):
        for edge, people in self.by_edge.items():
            self.by_edge[edge] = [
                replace(
                    person, global_id=aliases.get(person.global_id, person.global_id)
                )
                for person in people
            ]

    def update(self, scene, active_edges, now, assignments, max_people):
        for edge in active_edges:
            self.by_edge[edge] = [
                person for person in scene.people if person.root.edge_id == edge
            ]
        people = {}
        for edge, observations in self.by_edge.items():
            retained = [
                person
                for person in observations
                if now - person.root.timestamp <= self.ttl
            ]
            self.by_edge[edge] = retained
            for person in retained:
                lod = assignments.get(person.global_id, 1)
                person = replace(
                    person, lod=lod, pose=person.pose if lod == 2 else None
                )
                previous = people.get(person.global_id)
                if previous is None or person.root.timestamp > previous.root.timestamp:
                    people[person.global_id] = person
        ordered = sorted(
            people.values(),
            key=lambda person: (-person.lod, -person.root.confidence, person.global_id),
        )
        return replace(scene, people=ordered[:max_people], timestamp=now)
