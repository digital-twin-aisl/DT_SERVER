# SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Version 1 scene output, shared by producers and consumers. No model/transport SDKs."""

from dataclasses import dataclass
from typing import Any, Literal

LodLevel = Literal[0, 1, 2]
OUTPUT_SCHEMA_VERSION = 1
OUTPUT_COORDINATE_SYSTEM = {"frame": "USD world", "up_axis": "Z", "unit": "millimetre"}


@dataclass(frozen=True, slots=True)
class RootOutput:
    """USD world 좌표계의 millimetre 단위 사람 root."""

    edge_id: str
    candidate_index: int
    position: list[float]
    confidence: float
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "candidate_index": self.candidate_index,
            "position": self.position,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }



@dataclass(frozen=True, slots=True)
class PoseOutput:
    """root와 동일한 USD world millimetre 단위 3D joint 좌표."""

    joint_format: str
    joints: list[list[float]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint_format": self.joint_format,
            "joints": self.joints,
        }



@dataclass(frozen=True, slots=True)
class PersonEntity:
    """서버 파이프라인이 출력하는 사람 단위 entity."""

    global_id: int
    lod: LodLevel
    root: RootOutput
    pose: PoseOutput | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_id": self.global_id,
            "lod": self.lod,
            "root": self.root.to_dict(),
            "pose": self.pose.to_dict() if self.pose is not None else None,
        }



@dataclass(frozen=True, slots=True)
class SceneOutput:
    """동기화된 한 시점의 전체 사람 entity 출력."""

    timestamp: float
    sync_spread_seconds: float
    people: list[PersonEntity]
    runtime: dict | None = None

    def to_dict(self) -> dict[str, Any]:
        output = {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "coordinate_system": OUTPUT_COORDINATE_SYSTEM,
            "timestamp": self.timestamp,
            "sync_spread_seconds": self.sync_spread_seconds,
            "people": [person.to_dict() for person in self.people],
        }
        if self.runtime is not None:
            output["runtime"] = self.runtime
        return output

