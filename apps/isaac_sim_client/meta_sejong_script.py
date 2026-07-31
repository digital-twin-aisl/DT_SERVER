"""Receive live people scenes and visualize their roots in Isaac Sim.

The default transport is Zenoh. WebSocket support remains available for the
legacy sim_backend path by setting ISAAC_SCENE_TRANSPORT=websocket.
"""

import asyncio
import atexit
import json
import os
import queue
import threading
import traceback

import omni.kit.app
import omni.usd
from pxr import Gf, UsdGeom


DEFAULT_USD_PATH = "/home/dojan/All/2025_SejongUniv_All.usd"
DEFAULT_ZENOH_ENDPOINT = "127.0.0.1:7447"
DEFAULT_ZENOH_TOPIC = "meta-sejong/scene/v1"

USD_PATH = os.environ.get("ISAAC_USD_PATH", DEFAULT_USD_PATH)
SCENE_TRANSPORT = os.environ.get("ISAAC_SCENE_TRANSPORT", "zenoh").lower()
ZENOH_ENDPOINT = os.environ.get("ISAAC_ZENOH_ENDPOINT", DEFAULT_ZENOH_ENDPOINT)
ZENOH_TOPIC = os.environ.get("ISAAC_ZENOH_TOPIC", DEFAULT_ZENOH_TOPIC)
SIM_BACKEND_WS_URL = os.environ.get(
    "SIM_BACKEND_WS_URL",
    "ws://127.0.0.1:8004/ws/sim",
)
USD_PEOPLE_ROOT = "/World/MetaSejong_People"

if SCENE_TRANSPORT not in {"zenoh", "websocket", "both"}:
    raise ValueError(
        "ISAAC_SCENE_TRANSPORT must be one of: zenoh, websocket, both"
    )


omni.usd.get_context().open_stage(USD_PATH)
print(f"[Meta Sejong] Loading stage from {USD_PATH} ...")


def _id_color(global_id: int) -> Gf.Vec3f:
    palette = (
        (0.90, 0.10, 0.29),
        (0.24, 0.71, 0.29),
        (1.00, 0.88, 0.10),
        (0.00, 0.51, 0.78),
        (0.96, 0.51, 0.19),
        (0.57, 0.12, 0.71),
    )
    return Gf.Vec3f(*palette[global_id % len(palette)])


class MetaSejongDigitalTwin:
    """Update USD people prims from complete scene snapshots."""

    def __init__(self):
        self.stage = omni.usd.get_context().get_stage()
        if self.stage is None:
            raise RuntimeError("USD stage is not ready")
        self.units_per_meter = UsdGeom.GetStageMetersPerUnit(self.stage)
        self._ensure_root_group()

    def _ensure_root_group(self):
        root_prim = self.stage.GetPrimAtPath(USD_PEOPLE_ROOT)
        if not root_prim.IsValid():
            UsdGeom.Xform.Define(self.stage, USD_PEOPLE_ROOT)

    def update_person_root(self, global_id: int, position_meters) -> None:
        """Create or move the root capsule for one global ID."""
        person_path = f"{USD_PEOPLE_ROOT}/Person_{global_id}"
        person_prim = self.stage.GetPrimAtPath(person_path)

        if not person_prim.IsValid():
            capsule = UsdGeom.Capsule.Define(self.stage, person_path)
            capsule.GetHeightAttr().Set(1.7 / self.units_per_meter)
            capsule.GetRadiusAttr().Set(0.3 / self.units_per_meter)
            capsule.GetDisplayColorAttr().Set([_id_color(global_id)])

            xformable = UsdGeom.Xformable(capsule)
            xformable.ClearXformOpOrder()
            xformable.AddTranslateOp()
            person_prim = capsule.GetPrim()

        UsdGeom.Imageable(person_prim).MakeVisible()
        xformable = UsdGeom.Xformable(person_prim)
        translate_op = next(
            (
                op
                for op in xformable.GetOrderedXformOps()
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
            ),
            None,
        )
        if translate_op is None:
            translate_op = xformable.AddTranslateOp()

        scale = 1.0 / self.units_per_meter
        x, y, z = (float(value) * scale for value in position_meters)
        translate_op.Set(Gf.Vec3d(x, y, z))

    def apply_scene(self, scene: dict) -> None:
        """Apply one schema_version=1 SceneOutput snapshot."""
        active_paths = set()
        for person in scene.get("people", []):
            global_id = int(person["global_id"])
            root_position_mm = person.get("root", {}).get("position")
            if not isinstance(root_position_mm, list) or len(root_position_mm) != 3:
                continue

            # VoxelPose SceneOutput coordinates are millimeters.
            root_position_m = [value / 1000.0 for value in root_position_mm]
            self.update_person_root(global_id, root_position_m)
            active_paths.add(f"{USD_PEOPLE_ROOT}/Person_{global_id}")

        root_prim = self.stage.GetPrimAtPath(USD_PEOPLE_ROOT)
        for child in root_prim.GetChildren():
            if str(child.GetPath()) not in active_paths:
                UsdGeom.Imageable(child).MakeInvisible()

    def apply_legacy_object(self, track_id: int, position: dict) -> None:
        """Apply the legacy WebSocket position contract, whose unit is meters."""
        self.update_person_root(
            int(track_id),
            (
                position.get("x", 0.0),
                position.get("y", 0.0),
                position.get("z", 0.0),
            ),
        )


class LatestScene:
    """A single-slot handoff from a network thread to Isaac's update loop."""

    def __init__(self):
        self._queue = queue.Queue(maxsize=1)
        self.dropped = 0

    def put(self, scene: dict) -> None:
        if self._queue.full():
            try:
                self._queue.get_nowait()
                self.dropped += 1
            except queue.Empty:
                pass
        try:
            self._queue.put_nowait(scene)
        except queue.Full:
            self.dropped += 1

    def take(self):
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None


class ZenohSceneReceiver:
    def __init__(self, endpoint: str, topic: str, scenes: LatestScene):
        self.endpoint = endpoint
        self.topic = topic
        self.scenes = scenes
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error = None
        self._worker = threading.Thread(
            target=self._run,
            name="isaac-zenoh-scene-subscriber",
            daemon=True,
        )
        self._worker.start()

    def _config(self):
        import zenoh

        endpoint = self.endpoint
        if "/" not in endpoint:
            endpoint = f"tcp/{endpoint}"
        return zenoh.Config.from_json5(
            json.dumps(
                {
                    "mode": "client",
                    "connect": {"endpoints": [endpoint]},
                }
            )
        )

    def _on_sample(self, sample) -> None:
        try:
            scene = json.loads(sample.payload.to_bytes().decode("utf-8"))
            if scene.get("schema_version") != 1:
                print(
                    "[Meta Sejong] Ignoring unsupported scene schema:",
                    scene.get("schema_version"),
                )
                return
            if not isinstance(scene.get("people"), list):
                raise ValueError("scene.people must be a list")
            self.scenes.put(scene)
        except Exception:
            print("[Meta Sejong] Invalid Zenoh scene payload")
            traceback.print_exc()

    def _run(self) -> None:
        try:
            import zenoh

            with zenoh.open(self._config()) as session:
                subscriber = session.declare_subscriber(
                    self.topic,
                    self._on_sample,
                )
                self._ready.set()
                print(
                    "[Meta Sejong] Zenoh subscriber is ready: "
                    f"endpoint={self.endpoint} topic={self.topic}"
                )
                self._stop.wait()
                _ = subscriber
        except Exception as exc:
            self._error = exc
            self._ready.set()
            print(f"[Meta Sejong] Zenoh subscriber failed: {exc}")
            traceback.print_exc()

    def close(self) -> None:
        self._stop.set()
        if self._worker.is_alive() and threading.current_thread() is not self._worker:
            self._worker.join(timeout=5)


async def zenoh_update_task(scenes: LatestScene):
    digital_twin = None
    last_timestamp = None
    while True:
        await omni.kit.app.get_app().next_update_async()
        scene = scenes.take()
        if scene is None:
            continue

        try:
            if digital_twin is None:
                digital_twin = MetaSejongDigitalTwin()
            digital_twin.apply_scene(scene)
            timestamp = scene.get("timestamp")
            if last_timestamp is None:
                print(
                    "[Meta Sejong] First Zenoh scene received: "
                    f"timestamp={timestamp} people={len(scene['people'])}"
                )
            last_timestamp = timestamp
        except Exception:
            print("[Meta Sejong] Failed to apply Zenoh scene")
            traceback.print_exc()


async def websocket_listener_task():
    import websockets

    digital_twin = MetaSejongDigitalTwin()
    print(f"[Meta Sejong] Connecting to WebSocket {SIM_BACKEND_WS_URL} ...")
    try:
        async with websockets.connect(SIM_BACKEND_WS_URL) as websocket:
            print("[Meta Sejong] WebSocket connected")
            while True:
                data = json.loads(await websocket.recv())
                if data.get("action") != "UpdateTransforms":
                    continue
                for obj in data.get("objects", []):
                    track_id = obj.get("track_id")
                    position = obj.get("position")
                    if track_id is not None and position is not None:
                        digital_twin.apply_legacy_object(track_id, position)
    except websockets.exceptions.ConnectionClosed:
        print("[Meta Sejong] WebSocket connection closed")
    except Exception:
        print("[Meta Sejong] WebSocket listener failed")
        traceback.print_exc()


_zenoh_receiver = None
if SCENE_TRANSPORT in {"zenoh", "both"}:
    _latest_scenes = LatestScene()
    _zenoh_receiver = ZenohSceneReceiver(
        ZENOH_ENDPOINT,
        ZENOH_TOPIC,
        _latest_scenes,
    )
    atexit.register(_zenoh_receiver.close)
    asyncio.ensure_future(zenoh_update_task(_latest_scenes))

if SCENE_TRANSPORT in {"websocket", "both"}:
    asyncio.ensure_future(websocket_listener_task())

