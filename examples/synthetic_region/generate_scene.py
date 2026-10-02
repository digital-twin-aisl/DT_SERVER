# SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Generate a synthetic SceneOutput JSONL recording; no cameras or people needed.

The file plays in the frontend_api viewer ("기록 파일 재생") and the Isaac Sim
scene player. It contains only computed walkers, never real observations, and
every scene is marked ``runtime.playback=true`` so it cannot be mistaken for a
live input.

    python examples/synthetic_region/generate_scene.py --output /tmp/demo.jsonl
"""

import argparse
import json
import math
from pathlib import Path
import random
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "packages/dt_common/src"))

from dt_common.contracts.scene import (  # noqa: E402
    PersonEntity, PoseOutput, RootOutput, SceneOutput,
)

# Default walking area: the center-b1-corridor region, USD world millimetres.
DEFAULT_AREA_MM = (78000.0, -1000.0, 82000.0, 3000.0)  # inside edge_1 workspace
DEFAULT_GROUND_Z_MM = 13900.0
ROOT_HEIGHT_MM = 900.0

# VoxelPose/Panoptic 15-joint order, offsets from the root (mid-hip) in a
# body frame: x forward, y left, z up (mm).
JOINT_OFFSETS = [
    (0, 0, 500),      # 0 neck
    (60, 0, 650),     # 1 nose
    (0, 0, 0),        # 2 mid-hip (root)
    (0, 180, 480),    # 3 left shoulder
    (0, 220, 200),    # 4 left elbow
    (0, 230, -50),    # 5 left wrist
    (0, 100, 0),      # 6 left hip
    (0, 110, -450),   # 7 left knee
    (0, 110, -850),   # 8 left ankle
    (0, -180, 480),   # 9 right shoulder
    (0, -220, 200),   # 10 right elbow
    (0, -230, -50),   # 11 right wrist
    (0, -100, 0),     # 12 right hip
    (0, -110, -450),  # 13 right knee
    (0, -110, -850),  # 14 right ankle
]
LEFT_SWING = {4: 0.4, 5: 1.0, 7: -0.5, 8: -1.0}
RIGHT_SWING = {10: 0.4, 11: 1.0, 13: -0.5, 14: -1.0}


class Walker:
    def __init__(self, rng, area):
        self.area = area
        self.x = rng.uniform(area[0], area[2])
        self.y = rng.uniform(area[1], area[3])
        self.heading = rng.uniform(-math.pi, math.pi)
        self.speed = rng.uniform(900.0, 1400.0)  # mm/s
        self.phase = rng.uniform(0, 2 * math.pi)

    def step(self, dt, rng):
        self.heading += rng.gauss(0.0, 0.15) * dt
        nx = self.x + math.cos(self.heading) * self.speed * dt
        ny = self.y + math.sin(self.heading) * self.speed * dt
        if not self.area[0] <= nx <= self.area[2]:
            self.heading = math.pi - self.heading
        if not self.area[1] <= ny <= self.area[3]:
            self.heading = -self.heading
        self.x = min(max(nx, self.area[0]), self.area[2])
        self.y = min(max(ny, self.area[1]), self.area[3])
        self.phase += 2 * math.pi * 1.8 * dt  # ~1.8 strides per second

    def joints(self, root):
        c, s = math.cos(self.heading), math.sin(self.heading)
        swing = math.sin(self.phase) * 250.0
        result = []
        for index, (fx, fy, fz) in enumerate(JOINT_OFFSETS):
            fx += LEFT_SWING.get(index, 0.0) * swing - RIGHT_SWING.get(index, 0.0) * swing
            result.append([
                round(root[0] + fx * c - fy * s, 1),
                round(root[1] + fx * s + fy * c, 1),
                round(root[2] + fz, 1),
            ])
        return result


def generate(people, seconds, fps, seed, area, ground_z, pose_every):
    rng = random.Random(seed)
    walkers = [Walker(rng, area) for _ in range(people)]
    dt = 1.0 / fps
    for frame in range(int(seconds * fps)):
        timestamp = round(frame * dt, 6)
        entities = []
        for global_id, walker in enumerate(walkers):
            walker.step(dt, rng)
            root = [round(walker.x, 1), round(walker.y, 1), ground_z + ROOT_HEIGHT_MM]
            # Like the live server, only some people get a full pose (LOD 2).
            with_pose = (global_id + frame // pose_every) % 2 == 0
            entities.append(PersonEntity(
                global_id=global_id,
                lod=2 if with_pose else 1,
                root=RootOutput("synthetic", global_id, root, 1.0, timestamp),
                pose=PoseOutput("voxelpose_15j_xyz", walker.joints(root)) if with_pose else None,
            ))
        yield SceneOutput(
            timestamp=timestamp,
            sync_spread_seconds=0.0,
            people=entities,
            runtime={"playback": True, "source": "synthetic", "seed": seed},
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--people", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--area-mm", type=float, nargs=4, default=DEFAULT_AREA_MM,
                        metavar=("XMIN", "YMIN", "XMAX", "YMAX"))
    parser.add_argument("--ground-z-mm", type=float, default=DEFAULT_GROUND_Z_MM)
    parser.add_argument("--pose-every", type=int, default=30,
                        help="frames between LOD-2 pose hand-overs")
    parser.add_argument("--force", action="store_true", help="overwrite --output")
    args = parser.parse_args(argv)
    if args.people < 1 or args.seconds <= 0 or args.fps <= 0 or args.pose_every < 1:
        parser.error("people, seconds, fps and pose-every must be positive")
    if args.output.exists() and not args.force:
        parser.error(f"{args.output} exists; pass --force to overwrite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as stream:
        for scene in generate(args.people, args.seconds, args.fps, args.seed,
                              args.area_mm, args.ground_z_mm, args.pose_every):
            stream.write(json.dumps(scene.to_dict(), separators=(",", ":")) + "\n")
            count += 1
    print(f"Wrote {count} scenes to {args.output}")


if __name__ == "__main__":
    main()
