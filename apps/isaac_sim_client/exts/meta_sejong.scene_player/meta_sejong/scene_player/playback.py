# SPDX-FileCopyrightText: 2025-2026 DT_SERVER contributors
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Standard-library-only JSONL indexing and timestamp-based playback."""

from bisect import bisect_right
from dataclasses import dataclass
import json
import math
from pathlib import Path

MAX_LINE_BYTES = 16 * 1024 * 1024
MAX_FRAMES = 2_000_000


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _point(value):
    return isinstance(value, list) and len(value) == 3 and all(map(_finite, value))


def validate_scene(scene):
    if not isinstance(scene, dict) or scene.get("schema_version") != 1:
        raise ValueError("Expected schema_version=1 SceneOutput")
    convention = scene.get("coordinate_system")
    if convention != {"frame": "USD world", "up_axis": "Z", "unit": "millimetre"}:
        raise ValueError("Expected USD world / Z-up / millimetre coordinates")
    if not _finite(scene.get("timestamp")):
        raise ValueError("Scene timestamp must be finite")
    people = scene.get("people")
    if not isinstance(people, list) or len(people) > 1000:
        raise ValueError("Scene people must be a list of at most 1000 people")
    seen = set()
    for person in people:
        if not isinstance(person, dict):
            raise ValueError("Invalid person")
        identity = person.get("global_id")
        if type(identity) is not int or identity < 0 or identity in seen:
            raise ValueError("Person IDs must be unique non-negative integers")
        seen.add(identity)
        root = person.get("root")
        if not isinstance(root, dict) or not _point(root.get("position")):
            raise ValueError("Invalid root position")
        pose = person.get("pose")
        if pose is not None:
            if not isinstance(pose, dict) or pose.get("joint_format") != "voxelpose_15j_xyz":
                raise ValueError("Unsupported pose format")
            joints = pose.get("joints")
            if not isinstance(joints, list) or len(joints) != 15 or not all(map(_point, joints)):
                raise ValueError("Expected 15 finite XYZ joints")
    return scene


@dataclass
class SceneRecording:
    path: Path
    offsets: list[int]
    timestamps: list[float]
    warning: str = ""

    @classmethod
    def load(cls, path, cancelled=None):
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"Not a regular recording file: {path}")
        offsets, timestamps = [], []
        warning = ""
        with path.open("rb") as stream:
            line_number = 0
            while True:
                if cancelled is not None and cancelled.is_set():
                    raise ValueError("Loading cancelled")
                offset = stream.tell()
                line = stream.readline(MAX_LINE_BYTES + 1)
                if not line:
                    break
                line_number += 1
                if len(line) > MAX_LINE_BYTES:
                    raise ValueError(f"Line {line_number}: scene exceeds 16 MiB")
                if not line.strip():
                    continue
                try:
                    scene = json.loads(line)
                except (ValueError, UnicodeError) as exc:
                    if not line.endswith(b"\n") and not stream.read(1):
                        warning = f"Ignored incomplete final line {line_number} (interrupted recording)"
                        break
                    raise ValueError(f"Line {line_number}: invalid JSON") from exc
                try:
                    validate_scene(scene)
                    stamp = float(scene["timestamp"])
                    if timestamps and stamp < timestamps[-1]:
                        raise ValueError("Timestamps go backwards; use one recording per run")
                except ValueError as exc:
                    raise ValueError(f"Line {line_number}: {exc}") from exc
                offsets.append(offset)
                timestamps.append(stamp)
                if len(offsets) > MAX_FRAMES:
                    raise ValueError("Recording exceeds 2 million scenes; split the run")
        if not offsets:
            raise ValueError("Recording has no complete scenes")
        return cls(path, offsets, timestamps, warning)

    @property
    def duration(self):
        return self.timestamps[-1] - self.timestamps[0]

    def frame(self, index):
        if not 0 <= index < len(self.offsets):
            raise IndexError(index)
        with self.path.open("rb") as stream:
            stream.seek(self.offsets[index])
            scene = validate_scene(json.loads(stream.readline(MAX_LINE_BYTES + 1)))
        if scene["timestamp"] != self.timestamps[index]:
            raise ValueError("Recording changed after loading; reload the file")
        return scene


class Playback:
    def __init__(self, recording):
        self.recording = recording
        self.index = 0
        self.elapsed = 0.0
        self.playing = False
        self.speed = 1.0
        self.loop = False

    def seek_frame(self, index):
        self.index = max(0, min(int(index), len(self.recording.offsets) - 1))
        self.elapsed = self.recording.timestamps[self.index] - self.recording.timestamps[0]

    def play(self):
        if self.index == len(self.recording.offsets) - 1:
            self.seek_frame(0)
        self.playing = True

    def advance(self, dt):
        if not _finite(dt) or dt < 0 or not _finite(self.speed) or not 0.1 <= self.speed <= 8:
            raise ValueError("Expected finite dt >= 0 and playback speed 0.1-8")
        if not self.playing:
            return
        duration = self.recording.duration
        self.elapsed += dt * self.speed
        if self.elapsed >= duration:
            if self.loop and duration > 0:
                self.elapsed %= duration
            else:
                self.seek_frame(len(self.recording.offsets) - 1)
                self.playing = False
                return
        self.index = max(0, bisect_right(
            self.recording.timestamps, self.recording.timestamps[0] + self.elapsed,
        ) - 1)
