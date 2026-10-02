# SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Generate explicitly synthetic 15-joint demo scenes from authoritative CSV paths.

No inference, services, or GPU required. Positions are USD world millimetres;
CSV time is mapped to recording time by an explicit offset (default zero).
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / "packages/dt_common/src"))
from dt_common.contracts.scene import PersonEntity, PoseOutput, RootOutput, SceneOutput
from dt_common.spatial.ground import GroundSurface

STRIDE_MM = 1000.0  # Travel per full left/right cycle.
LEG_MM = 440.0
ANKLE_MM = 60.0
SELECTION_COLUMNS = ("all", "zone_selected", "rank_selected")


@dataclass
class Track:
    identity: int
    times: np.ndarray
    xy: np.ndarray
    flags: list[dict]

    def __post_init__(self):
        self.distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(self.xy, axis=0), axis=1))]

    def at(self, timestamp):
        return np.array([np.interp(timestamp, self.times, self.xy[:, axis]) for axis in (0, 1)])

    def along(self, distance):
        return np.array([np.interp(distance, self.distance, self.xy[:, axis]) for axis in (0, 1)])

    def direction(self, distance):
        delta = self.along(distance + 180.0) - self.along(distance - 180.0)
        length = np.linalg.norm(delta)
        return delta / length if length > 1e-6 else np.array([1.0, 0.0])


def read_tracks(path, max_gap=1.01):
    grouped = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"time_stamp", "global_id", "x_mm", "y_mm"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV requires columns: {sorted(required)}")
        for line, row in enumerate(reader, 2):
            try:
                identity = int(row["global_id"])
                values = [float(row[key]) for key in ("time_stamp", "x_mm", "y_mm")]
                if identity < 0 or identity > 2**53 - 1 or not all(map(math.isfinite, values)):
                    raise ValueError("invalid ID or non-finite coordinate/time")
                flags = {key: int(row[key]) for key in ("zone_selected", "rank_selected") if row.get(key)}
            except (ValueError, TypeError) as exc:
                raise ValueError(f"CSV line {line}: {exc}") from exc
            grouped.setdefault(identity, []).append((values, flags))
    tracks = []
    for identity, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row[0][0])
        segment = []
        for row in rows:
            if segment:
                gap = row[0][0] - segment[-1][0][0]
                if gap <= 0:
                    raise ValueError(f"Duplicate timestamp for ID {identity}: {row[0][0]}")
                if gap > max_gap:
                    tracks.append(_track(identity, segment))
                    segment = []
            segment.append(row)
        tracks.append(_track(identity, segment))
    if not tracks:
        raise ValueError("CSV contains no people")
    return tracks


def _track(identity, rows):
    values = np.array([row[0] for row in rows])
    return Track(identity, values[:, 0], values[:, 1:], [row[1] for row in rows])


def ground_height(ground, xy):
    height = ground.height_mm(*xy)
    if not math.isfinite(height):
        raise ValueError(f"No ground beneath XY {xy.tolist()}; check coordinate alignment/ground coverage")
    return height


def knee_between(hip, ankle, forward):
    """Two equal rigid links, knee bent forward in the hip/ankle plane."""
    delta = ankle - hip
    length = np.linalg.norm(delta)
    if not 1e-6 < length < 2 * LEG_MM:
        raise ValueError(f"Unreachable leg target: {length:.1f} mm")
    axis = delta / length
    bend = forward - np.dot(forward, axis) * axis
    bend /= np.linalg.norm(bend)
    return (hip + ankle) / 2 + bend * math.sqrt(LEG_MM**2 - (length / 2)**2)


def synthetic_person(track, csv_time, scene_time, ground, selection_column="all"):
    xy = track.at(csv_time)
    distance = float(np.interp(csv_time, track.times, track.distance))
    forward_xy = track.direction(distance)
    forward = np.r_[forward_xy, 0.0]
    left = np.array([-forward[1], forward[0], 0.0])
    before, after = max(track.times[0], csv_time - .18), min(track.times[-1], csv_time + .18)
    speed = np.linalg.norm(track.at(after) - track.at(before)) / max(after - before, 1e-6)
    moving = min(1.0, speed / 250.0)
    cycle = distance / STRIDE_MM
    feet = []
    for side, phase_offset in ((1, 0.0), (-1, .5)):
        phase = cycle + phase_offset
        step, fraction = math.floor(phase), phase % 1.0
        contact = (step + .25 - phase_offset) * STRIDE_MM
        start_xy = track.along(contact)
        start_dir = track.direction(contact)
        start_xy += side * 105.0 * np.array([-start_dir[1], start_dir[0]])
        start_z = ground_height(ground, start_xy) + ANKLE_MM
        foot = np.r_[start_xy, start_z]
        if fraction > .5:
            end_distance = contact + STRIDE_MM
            end_xy = track.along(end_distance)
            end_dir = track.direction(end_distance)
            end_xy += side * 105.0 * np.array([-end_dir[1], end_dir[0]])
            end_z = ground_height(ground, end_xy) + ANKLE_MM
            swing = (fraction - .5) * 2.0
            blend = swing * swing * (3.0 - 2.0 * swing)
            foot = foot * (1.0 - blend) + np.r_[end_xy, end_z] * blend
            # Clear intermediate sloped/stepped ground as well as the endpoints.
            foot[2] = max(foot[2], ground_height(ground, foot[:2]) + ANKLE_MM)
            foot[2] += 110.0 * math.sin(math.pi * swing)**2
        if moving < 1.0:
            standing_xy = xy + side * 105.0 * left[:2]
            standing = np.r_[standing_xy, ground_height(ground, standing_xy) + ANKLE_MM]
            foot = moving * foot + (1.0 - moving) * standing
            foot[2] = max(foot[2], ground_height(ground, foot[:2]) + ANKLE_MM)
        feet.append(foot)

    # Lower pelvis on slopes/stairs as needed; never stretch the leg bones.
    root_z = ground_height(ground, xy) + 865.0 + 12.0 * moving * math.cos(4 * math.pi * cycle)
    for side, foot in zip((1, -1), feet):
        hip_xy = xy + side * 105.0 * left[:2]
        horizontal_sq = float(np.sum((hip_xy - foot[:2])**2))
        if horizontal_sq >= (2 * LEG_MM * .995)**2:
            raise ValueError("Path turn/step exceeds leg reach")
        root_z = min(root_z, foot[2] + math.sqrt((2 * LEG_MM * .995)**2 - horizontal_sq))
    root = np.r_[xy, root_z]
    joints = np.zeros((15, 3))
    joints[2] = root
    joints[0] = root + [0.0, 0.0, 510.0]  # Neck, head, pelvis: VoxelPose order.
    joints[1] = joints[0] + [0.0, 0.0, 205.0]
    for side, shoulder, hip, foot in ((1, 3, 6, feet[0]), (-1, 9, 12, feet[1])):
        joints[hip] = root + side * 105.0 * left
        joints[hip + 2] = foot
        joints[hip + 1] = knee_between(joints[hip], foot, forward)
        joints[shoulder] = joints[0] + side * 195.0 * left + [0.0, 0.0, -45.0]
        angle = -side * .40 * moving * math.cos(2 * math.pi * cycle)
        joints[shoulder + 1] = joints[shoulder] + 285.0 * (
            math.sin(angle) * forward + np.array([0.0, 0.0, -math.cos(angle)]))
        elbow_angle = angle + .28
        joints[shoulder + 2] = joints[shoulder + 1] + 260.0 * (
            math.sin(elbow_angle) * forward + np.array([0.0, 0.0, -math.cos(elbow_angle)]))
    person = PersonEntity(track.identity, 2,
        RootOutput("scenario", 0, root.tolist(), 1.0, scene_time),
        PoseOutput("voxelpose_15j_xyz", joints.tolist())).to_dict()
    index = max(0, bisect_right(track.times, csv_time) - 1)
    person["scenario"] = {"position_source": "csv", "pose_source": "procedural_gait",
                          "synthetic_pose": True, **track.flags[index]}
    if selection_column not in SELECTION_COLUMNS:
        raise ValueError(f"Unknown selection column: {selection_column}")
    selected = 1 if selection_column == "all" else track.flags[index].get(selection_column)
    if selected not in (0, 1):
        raise ValueError(f"ID {track.identity}: {selection_column} must be 0 or 1")
    person["scenario"].update(selection_column=selection_column, selected=selected,
                              display="skeleton" if selected else "capsule")
    if not selected:
        # Existing renderers select their root marker when no pose is supplied.
        # CSV selection 0/1 maps to internal LOD 1/2, not LOD 0/1.
        person["lod"] = 1
        person["pose"] = None
    return person


def reference_bounds(path):
    first = last = None
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            timestamp = float(json.loads(line)["timestamp"])
            if not math.isfinite(timestamp) or (last is not None and timestamp < last):
                raise ValueError("Reference recording has invalid/backward timestamps")
            first = timestamp if first is None else first
            last = timestamp
    if first is None:
        raise ValueError("Reference recording is empty")
    return first, last


def digest(path):
    # hashlib.file_digest is Python 3.11+; the server still supports 3.10.
    sha256 = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def generate(args):
    for key in ("fps", "max_gap", "time_offset"):
        if not math.isfinite(getattr(args, key)):
            raise ValueError(f"{key} must be finite")
    if args.fps <= 0 or args.max_gap <= 0:
        raise ValueError("fps and max-gap must be positive")
    tracks = read_tracks(args.csv, args.max_gap)
    selection_column = getattr(args, "selection_column", "all")
    if selection_column not in SELECTION_COLUMNS:
        raise ValueError(f"Unknown selection column: {selection_column}")
    if selection_column != "all":
        for track in tracks:
            for timestamp, flags in zip(track.times, track.flags):
                if flags.get(selection_column) not in (0, 1):
                    raise ValueError(f"ID {track.identity} at {timestamp}: {selection_column} must be 0 or 1")
    ground = GroundSurface.from_cache(args.ground_cache)
    csv_start = min(track.times[0] for track in tracks)
    csv_end = max(track.times[-1] for track in tracks)
    start, end = (reference_bounds(args.reference_recording) if args.reference_recording else
                  (csv_start + args.time_offset, csv_end + args.time_offset))
    if csv_end + args.time_offset < start or csv_start + args.time_offset > end:
        raise ValueError("CSV and recording do not overlap; check --time-offset")
    count = math.ceil((end - start) * args.fps) + 1
    if count > 2_000_000:
        raise ValueError("Output exceeds 2 million frames")
    provenance = {"mode": "scenario_synthetic", "synthetic_pose": True,
                  "position_source": str(args.csv.resolve()), "csv_sha256": digest(args.csv),
                  "ground_sha256": digest(args.ground_cache),
                  "time_offset_seconds": args.time_offset, "fps": args.fps,
                  "time_mapping": "scene_time = csv_time + time_offset_seconds",
                  "alignment_verified": False, "max_interpolation_gap_seconds": args.max_gap,
                  "selection_column": selection_column,
                  "selection_sampling": "previous_csv_row",
                  "pose_source": "procedural_gait", "reference_usage": "timeline_bounds_only"}
    if args.reference_recording:
        provenance["reference_recording"] = str(args.reference_recording.resolve())
        provenance["reference_sha256"] = digest(args.reference_recording)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    people_count = 0
    with args.output.open("x", encoding="utf-8") as stream:
        try:
            for index in range(count):
                timestamp = min(end, start + index / args.fps)
                csv_time = timestamp - args.time_offset
                scene = SceneOutput(timestamp, 0.0, [], runtime={
                    "timestamp_kind": "dataset_relative", "scenario": provenance}).to_dict()
                scene["people"] = [synthetic_person(track, csv_time, timestamp, ground, selection_column)
                    for track in tracks if track.times[0] <= csv_time <= track.times[-1]]
                people_count += len(scene["people"])
                stream.write(json.dumps(scene, separators=(",", ":"), allow_nan=False) + "\n")
        except BaseException:
            # Only remove the incomplete file this invocation created exclusively.
            args.output.unlink()
            raise
    return {"output": str(args.output.resolve()), "frames": count,
            "person_frames": people_count, "ids": sorted({track.identity for track in tracks}),
            "time_range": [start, end], "synthetic_pose": True, "selection_column": selection_column}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New JSONL; never overwrites an existing file")
    parser.add_argument("--ground-cache", type=Path,
                        default=PROJECT / "apps/deployments/cache/scene_0812_2_ground.npz")
    parser.add_argument("--reference-recording", type=Path, help="Use only its start/end times, not its IDs/poses")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--selection-column", choices=SELECTION_COLUMNS, default="all",
                        help="CSV display rule: 0=root capsule, 1=skeleton; all always shows skeletons")
    parser.add_argument("--time-offset", type=float, default=0.0,
                        help="Seconds added to CSV time; 0 preserves original timestamps")
    parser.add_argument("--max-gap", type=float, default=1.01,
                        help="Split tracks across larger CSV gaps; do not extrapolate")
    args = parser.parse_args()
    try:
        result = generate(args)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
