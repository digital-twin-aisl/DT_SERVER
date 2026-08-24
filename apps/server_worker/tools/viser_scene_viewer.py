"""Temporary playback viewer for data/viser_scenes.jsonl.

Run:
    pip install viser
    python tools/viser_scene_viewer.py
"""

import json
from pathlib import Path
import time

import numpy as np


# ===== TEMP VISER DEBUG: this whole file can be deleted later =====
SERVER_WORKER_DIR = Path(__file__).resolve().parents[1]
SCENE_PATH = SERVER_WORKER_DIR / "data" / "viser_scenes.jsonl"
VISER_HOST = "0.0.0.0"
VISER_PORT = 8080
PLAYBACK_FPS = 10.0
LOOP_PLAYBACK = True

USD_WORLD_OFFSET_METERS = np.zeros(3, dtype=np.float32)

LIMBS_15 = np.array(
    [
        [0, 1],
        [0, 2],
        [0, 3],
        [3, 4],
        [4, 5],
        [0, 9],
        [9, 10],
        [10, 11],
        [2, 6],
        [2, 12],
        [6, 7],
        [7, 8],
        [12, 13],
        [13, 14],
    ],
    dtype=np.int64,
)

ID_COLORS = (
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 212),
)


def edge_offset(_edge_id: str) -> np.ndarray:
    """Both distributed edges already publish one shared USD world frame."""
    return USD_WORLD_OFFSET_METERS


def draw_scene(server, scene: dict, old_handles: list) -> list:
    for handle in reversed(old_handles):
        handle.remove()

    handles = []
    for person in scene.get("people", []):
        global_id = int(person["global_id"])
        lod = int(person["lod"])
        root = person["root"]
        edge_id = str(root["edge_id"])
        offset = edge_offset(edge_id)
        color = ID_COLORS[global_id % len(ID_COLORS)]
        node = f"/scene/{edge_id}/person_{global_id}"

        # VoxelPose coordinates are millimeters; Viser scene coordinates are meters.
        root_position = (
            np.asarray(root["position"], dtype=np.float32) / 1000.0 + offset
        )
        handles.append(
            server.scene.add_point_cloud(
                f"{node}/root",
                points=root_position[None, :],
                colors=color,
                point_size=0.08,
                point_shape="circle",
            )
        )
        handles.append(
            server.scene.add_label(
                f"{node}/label",
                text=(
                    f"ID {global_id} | LoD {lod} | {edge_id} | "
                    f"conf {float(root['confidence']):.2f}"
                ),
                position=root_position + np.array([0.0, 0.0, 0.15]),
            )
        )

        pose = person.get("pose")
        if pose is None:
            continue
        joints = np.asarray(pose["joints"], dtype=np.float32) / 1000.0 + offset
        if joints.ndim != 2 or joints.shape[1] != 3:
            continue

        handles.append(
            server.scene.add_point_cloud(
                f"{node}/joints",
                points=joints,
                colors=color,
                point_size=0.04,
                point_shape="circle",
            )
        )
        valid_limbs = LIMBS_15[np.all(LIMBS_15 < len(joints), axis=1)]
        if len(valid_limbs):
            handles.append(
                server.scene.add_line_segments(
                    f"{node}/limbs",
                    points=joints[valid_limbs],
                    colors=color,
                    line_width=4.0,
                )
            )

    print(
        f"updated timestamp={scene.get('timestamp')} "
        f"people={len(scene.get('people', []))}"
    )
    return handles


def load_scenes() -> list[dict]:
    if not SCENE_PATH.exists():
        return []

    scenes = []
    for line_number, line in enumerate(
        SCENE_PATH.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            scenes.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"ignoring incomplete JSONL record at line {line_number}")
    return scenes


def main() -> None:
    try:
        import viser
    except ImportError as exc:
        raise SystemExit("viser is not installed: pip install viser") from exc

    server = viser.ViserServer(host=VISER_HOST, port=VISER_PORT)
    server.scene.set_up_direction("+z")
    print(f"playing {SCENE_PATH} at {PLAYBACK_FPS:.1f} FPS")

    scenes = []
    frame_index = 0
    handles = []
    while True:
        if frame_index >= len(scenes):
            scenes = load_scenes()
            if frame_index >= len(scenes):
                if LOOP_PLAYBACK and scenes:
                    frame_index = 0
                else:
                    time.sleep(1.0 / PLAYBACK_FPS)
                    continue

        handles = draw_scene(server, scenes[frame_index], handles)
        frame_index += 1
        time.sleep(1.0 / PLAYBACK_FPS)


if __name__ == "__main__":
    main()
# ===== END TEMP VISER DEBUG =====
