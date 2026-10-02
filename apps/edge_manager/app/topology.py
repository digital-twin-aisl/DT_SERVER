# SPDX-FileCopyrightText: 2025-2026 DT_SERVER contributors
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Region membership references deployment manifests; camera order has one owner."""

from copy import deepcopy
from pathlib import Path
import threading

from dt_common.contracts.edge import validate_edge_id
from apps.edge_manager.adapters.json_store import read_json, atomic_json

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CATALOG = PROJECT_ROOT / "apps/deployments/regions.json"






class RegionCatalog:
    def __init__(self, path=DEFAULT_CATALOG):
        self.path = Path(path).resolve()
        self.lock = threading.RLock()
        self.document = read_json(self.path)
        if self.document.get("version") != 1 or not isinstance(
            self.document.get("regions"), dict
        ):
            raise ValueError("region catalog requires version=1 and regions object")
        self.expanded = self._validate(self.document)

    def resolve(self, value):
        return (self.path.parent / value).resolve()

    def _expand(self, region_id, raw):
        validate_edge_id(region_id)
        if not str(raw.get("name", "")).strip():
            raise ValueError("region name is required")
        deployment = self.resolve(raw["deployment"])
        manifest = read_json(deployment)
        profile = self.resolve(raw["server_profile"])
        settings = read_json(profile)
        if settings.get("schema_version") != 1:
            raise ValueError("server profile requires schema_version=1")
        if (
            profile.parent / settings.get("path_base", ".") / settings["arguments"]["deployment"]
        ).resolve() != deployment:
            raise ValueError("server profile and region must use the same deployment")
        edges = []
        for entry in manifest["edges"]:
            if not entry.get("enabled", True):
                continue
            edge_id = validate_edge_id(entry["id"])
            ids = entry["camera_ids"]
            if (
                not ids
                or any(type(x) is not int or x <= 0 for x in ids)
                or len(set(ids)) != len(ids)
            ):
                raise ValueError(
                    f"{edge_id}: camera IDs must be unique positive integers"
                )
            edges.append(
                {
                    "edge_id": edge_id,
                    "cameras": [
                        {"camera_id": str(x), "id": f"{edge_id}/camera/{x}"}
                        for x in ids
                    ],
                }
            )
        if not edges or len({e["edge_id"] for e in edges}) != len(edges):
            raise ValueError("region must have unique enabled edges")
        topic = raw.get("scene_topic", f"dt/regions/{region_id}/scene")
        if not topic or any(c in topic for c in "*?#"):
            raise ValueError("scene_topic must be a concrete Zenoh key")
        return {
            **deepcopy(raw),
            "region_id": region_id,
            "deployment": str(deployment),
            "server_profile": str(profile),
            "scene_topic": topic,
            "topic_root": manifest.get("topic_root", "dt/edges"),
            "edges": edges,
        }

    def _validate(self, document):
        owners, topics = set(), set()
        expanded = {}
        for region_id, raw in document["regions"].items():
            region = self._expand(region_id, raw)
            if region["scene_topic"] in topics:
                raise ValueError("regions must use distinct scene topics")
            topics.add(region["scene_topic"])
            for edge in region["edges"]:
                if edge["edge_id"] in owners:
                    raise ValueError(
                        f"edge belongs to more than one region: {edge['edge_id']}"
                    )
                owners.add(edge["edge_id"])
            expanded[region_id] = region
        return expanded

    def all(self):
        with self.lock:
            return deepcopy(list(self.expanded.values()))

    def get(self, region_id):
        with self.lock:
            return deepcopy(self.expanded[region_id])

    def put(self, region_id, settings):
        allowed = {"name", "deployment", "server_profile", "scene_topic"}
        if set(settings) - allowed:
            raise ValueError("unknown region settings")
        with self.lock:
            candidate = deepcopy(self.document)
            candidate["regions"][region_id] = settings
            expanded = self._validate(candidate)
            atomic_json(self.path, candidate)
            self.document = candidate
            self.expanded = expanded
        return self.get(region_id)

    def remove(self, region_id):
        with self.lock:
            candidate = deepcopy(self.document)
            del candidate["regions"][region_id]
            atomic_json(self.path, candidate)
            self.document = candidate
            self.expanded.pop(region_id, None)
