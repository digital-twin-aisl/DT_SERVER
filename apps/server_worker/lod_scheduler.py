"""PoC decisions use observation time and stay fixed over a configured horizon."""

from copy import deepcopy
import math

from apps.server_worker.priority_engine import PriorityEngine


def inside_polygon(x, y, polygon):
    inside = False
    for first, second in zip(polygon, polygon[1:] + polygon[:1]):
        ax, ay = first
        bx, by = second
        cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)
        if (
            abs(cross) < 1e-9
            and min(ax, bx) <= x <= max(ax, bx)
            and min(ay, by) <= y <= max(ay, by)
        ):
            return True
        if (ay > y) != (by > y) and x < (bx - ax) * (y - ay) / (by - ay) + ax:
            inside = not inside
    return inside


def validate_zones(zones):
    for zone in zones:
        polygon = zone.get("polygon_xy_m", [])
        if len(polygon) < 3 or zone.get("lod") not in (0, 1, 2):
            raise ValueError(
                "each zone requires polygon_xy_m (3+ points) and lod 0/1/2"
            )
        if any(
            len(point) != 2 or not all(math.isfinite(float(v)) for v in point)
            for point in polygon
        ):
            raise ValueError("zone coordinates must be finite metre-based XY points")


class LodScheduler:
    """Rank only on decision ticks; newcomers remain LOD 1 until the next tick.

    Latest observations are retained briefly so independent edge arrivals do not
    allocate a separate LOD-2 budget per edge. Missing tracks expire by seconds.
    Zone uses configured polygons, not a newly invented distance threshold.
    """

    def __init__(
        self,
        engine: PriorityEngine,
        *,
        interval=0.5,
        lod2_count=1,
        policy="rank",
        zones=(),
        edge_lods=None,
        observation_ttl=0.75,
    ):
        if not math.isfinite(interval) or interval < 0:
            raise ValueError("priority interval must be finite and nonnegative")
        if not math.isfinite(observation_ttl) or observation_ttl <= 0 or lod2_count < 0:
            raise ValueError("invalid LOD budget or observation TTL")
        if policy not in {"rank", "zone", "all"}:
            raise ValueError("invalid LOD policy")
        self.engine = engine
        self.interval = interval
        self.lod2_count = lod2_count
        self.policy = policy
        self.zones = list(zones)
        self.edge_lods = dict(edge_lods or {})
        if any(
            type(value) is not int or value not in (0, 1, 2)
            for value in self.edge_lods.values()
        ):
            raise ValueError("edge zone LOD must be 0/1/2")
        validate_zones(self.zones)
        if policy == "zone" and not self.zones and not self.edge_lods:
            raise ValueError("zone policy requires configured polygons or edge_lods")
        self.observation_ttl = observation_ttl
        self.last_timestamp = None
        self.last_decision = None
        self.observations = {}
        self.assignments = {}
        self.decision = None
        self.updated = False

    def merge_identities(self, aliases):
        self.engine.merge_identities(aliases)
        for old, new in aliases.items():
            previous = self.observations.pop(old, None)
            current = self.observations.get(new)
            if previous is not None and (current is None or previous[0] > current[0]):
                previous[1]["track_id"] = new
                self.observations[new] = previous
            previous_lod = self.assignments.pop(old, 1)
            if previous_lod == 2 or new in self.assignments:
                self.assignments[new] = max(previous_lod, self.assignments.get(new, 0))
        if self.decision:
            merged = {}
            for person in self.decision["persons"]:
                key = aliases.get(person["track_id"], person["track_id"])
                person = {
                    **person,
                    "track_id": key,
                    "lod": self.assignments.get(key, 1),
                }
                if key not in merged or person["rank"] < merged[key]["rank"]:
                    merged[key] = person
            self.decision["persons"] = list(merged.values())

    def assign(self, persons, timestamp):
        timestamp = float(timestamp)
        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if self.last_timestamp is not None and timestamp < self.last_timestamp - 1e-6:
            self.engine.reset()
            self.observations.clear()
            self.assignments.clear()
            self.last_decision = None
        self.last_timestamp = timestamp
        current_ids = []
        for person in persons:
            track_id = int(person["track_id"])
            if track_id in current_ids:
                raise ValueError("duplicate track ID")
            current_ids.append(track_id)
            observed_at = float(person.get("timestamp", timestamp))
            if not math.isfinite(observed_at) or observed_at > timestamp + 1e-6:
                raise ValueError("invalid observation timestamp")
            previous = self.observations.get(track_id)
            if previous is None or observed_at >= previous[0]:
                self.observations[track_id] = (observed_at, deepcopy(person))
        self.observations = {
            key: value
            for key, value in self.observations.items()
            if timestamp - value[0] <= self.observation_ttl
        }
        self.updated = (
            self.last_decision is None
            or timestamp - self.last_decision + 1e-9 >= self.interval
        )
        if self.updated:
            available = [person for _, person in self.observations.values()]
            ranked = self.engine.assign_priority(
                {"persons": available},
                timestamp=timestamp,
                selected_count=self.lod2_count,
            )
            assignments = {}
            for person in ranked["persons"]:
                track_id = int(person["track_id"])
                if self.policy == "all":
                    lod = 2
                elif self.policy == "rank":
                    lod = 2 if person["rank"] <= self.lod2_count else 1
                else:
                    position = person["position"]
                    matches = [
                        zone["lod"]
                        for zone in self.zones
                        if inside_polygon(
                            position["x"], position["y"], zone["polygon_xy_m"]
                        )
                    ]
                    lod = max(
                        matches, default=self.edge_lods.get(person.get("edge_id"), 1)
                    )
                assignments[track_id] = lod
                person["lod"] = lod
            self.assignments = assignments
            self.last_decision = timestamp
            self.decision = {
                "timestamp": timestamp,
                "policy": self.policy,
                "interval_s": self.interval,
                "persons": ranked["persons"],
            }
        if self.policy == "all":
            self.assignments.update(dict.fromkeys(current_ids, 2))
        return {track_id: self.assignments.get(track_id, 1) for track_id in current_ids}
